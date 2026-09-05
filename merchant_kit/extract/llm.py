"""
Tier 2 extraction: LLM-based, from raw page content. This runs ONLY when
extract/structured.py finds zero products -- this is the one place in
the toolkit where AI is doing necessary work rather than just parsing,
and the report must say so plainly rather than badge the whole pipeline
"AI-powered."

Exercised against a real client-hydrated Next.js SPA product page: zero
JSON-LD/microdata/RDFa in static (non-JS-executed) HTML, but real
server-rendered text content -- exactly the case this module exists for.
"""

from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup

from merchant_kit.schemas import RawProduct

MAX_CONTENT_CHARS = 24_000  # ~6k tokens, a safe chunking target

_SYSTEM_PROMPT = (
    "You extract structured product data from raw e-commerce page content. "
    "Only report a product if the page is clearly a single product's detail "
    "page. If the price is shown as a range, or you are not confident, set "
    "found_product to false rather than guessing at a value. `confidence` is "
    "your own 0.0-1.0 estimate of how certain you are the extracted fields "
    "are correct -- it is stored as untrusted provenance metadata and does "
    "not by itself authorize anything."
)

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "found_product": {"type": "BOOLEAN"},
        "title": {"type": "STRING"},
        "description": {"type": "STRING"},
        "category": {"type": "STRING"},
        "price_rupees": {"type": "NUMBER"},
        "currency": {"type": "STRING"},
        "in_stock": {"type": "BOOLEAN"},
        "sku": {"type": "STRING"},
        "brand": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["found_product", "confidence"],
}


def _strip_boilerplate(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "svg", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:MAX_CONTENT_CHARS]


def _price_to_paise(price_rupees) -> int | None:
    if price_rupees is None:
        return None
    try:
        return round(float(price_rupees) * 100)
    except (TypeError, ValueError):
        return None


_NULL_LIKE = {"null", "none", "n/a", "na", "unknown", ""}


def _clean_str(value) -> str | None:
    """Gemini has been observed (real 2026-08-24 run against a BigBasket
    snapshot) to return the literal STRING "null" for an absent optional
    field instead of omitting it or using JSON null, despite the schema
    saying STRING. `data.get(k) or None` doesn't catch this -- a non-empty
    string is truthy -- so a bogus SKU value ("null") silently made it
    into a RawProduct. This is the fix: treat a small set of null-like
    string tokens as genuinely absent, case-insensitively."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if cleaned.lower() in _NULL_LIKE:
        return None
    return cleaned


def extract_llm(
    html: str,
    source_url: str,
    *,
    provider,
    model: str,
    call_site: str = "merchant_kit.extract_llm",
    record_key=None,
) -> list[RawProduct]:
    content = _strip_boilerplate(html)
    if not content:
        return []

    kwargs: dict = dict(
        model=model,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "text": f"Page content:\n\n{content}"}],
        response_schema=_RESPONSE_SCHEMA,
        max_output_tokens=500,
        call_site=call_site,
    )
    if record_key is not None:
        kwargs["record_key"] = record_key

    completion = provider.complete(**kwargs)

    try:
        data = json.loads(completion.text)
    except json.JSONDecodeError:
        return []

    if not data.get("found_product"):
        return []

    brand = _clean_str(data.get("brand"))
    currency = _clean_str(data.get("currency")) or "INR"

    return [
        RawProduct(
            sku=_clean_str(data.get("sku")),
            title=_clean_str(data.get("title")),
            description=_clean_str(data.get("description")),
            category=_clean_str(data.get("category")),
            price_paise=_price_to_paise(data.get("price_rupees")),
            currency=currency,
            in_stock=data.get("in_stock"),
            image_url=None,  # not requested -- an LLM transcribing an image URL from
            # stripped text content is unreliable, and a wrong URL is worse than none
            attributes={"brand": brand} if brand else {},
            source_url=source_url,
            extraction_method="llm_inferred",
            confidence=data.get("confidence"),
        )
    ]
