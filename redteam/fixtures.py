"""
Synthetic HTML fixtures for A1 (JSON-LD injection) and A2 (review-text
injection) -- these exercise the REAL merchant_kit extraction pipeline
(extract/structured.py, extract/llm.py, normalize.py), not a mock of it,
so "injection stays confined to description-ish text; price provenance
is unaffected" is actually tested against real code, not just asserted.
"""

from __future__ import annotations

import json

from .attacks import inject_into_description


def jsonld_injection_html(*, sku: str, title: str, price_rupees: float, clean_description: str) -> str:
    """A1: the injection lives inside a syntactically valid JSON-LD
    Product block's description field -- exactly the real-world pattern
    Zscaler ThreatLabz documented (structured fields treated as
    higher-signal context by agents), and the reason this project treats
    "clean structured markup" as the MOST dangerous extraction path, not
    the safest one. Exercises extract/structured.py."""
    payload = {
        "@context": "https://schema.org",
        "@type": "Product",
        "sku": sku,
        "name": title,
        "description": inject_into_description(clean_description),
        "offers": {"@type": "Offer", "price": price_rupees, "priceCurrency": "INR", "availability": "InStock"},
    }
    return (
        "<html><head>"
        f'<script type="application/ld+json">{json.dumps(payload)}</script>'
        f"</head><body><h1>{title}</h1></body></html>"
    )


def review_injection_html(*, title: str, price_rupees: float, clean_review: str) -> str:
    """A2: same payload, planted inside a customer-review text block with
    NO structured markup at all -- forces the LLM extraction path
    (extract/llm.py), so the injected instruction is genuinely read as
    part of the raw page content an extraction LLM sees, not just
    embedded in a JSON field nothing parses as prose."""
    injected_review = inject_into_description(clean_review)
    return (
        "<html><body>"
        f"<h1>{title}</h1>"
        f"<p>Price: Rs. {price_rupees:.2f}</p>"
        f'<div class="reviews"><p class="review">{injected_review}</p></div>'
        "</body></html>"
    )
