"""
Tier 1 extraction: schema.org markup (JSON-LD / microdata / RDFa) via
extruct. No LLM involved -- this is pure parsing, and the plan is
explicit that this must not be oversold as "AI-powered." It only runs
LLM extraction (extract/llm.py) as a fallback when this yields nothing.

Verified against real 2026-08-24 snapshots before writing this: Chumbak's
JSON-LD nests Offer under `offers` and is rich (Brand, AggregateRating,
Review); Fabindia's is minimal (sku/name/image/offers only, no brand or
rating); BigBasket's static HTML has zero JSON-LD/microdata/RDFa at all
(a Next.js SPA that hydrates client-side) -- which is the real case this
Tier-1/Tier-2 split exists for, not a hypothetical.
"""

from __future__ import annotations

import extruct

from merchant_kit.schemas import RawProduct

_PRODUCT_TYPES = {"Product", "ProductGroup"}
_IN_STOCK_MARKERS = {"instock", "limitedavailability", "onlineonly", "presale", "preorder"}
_OUT_OF_STOCK_MARKERS = {"outofstock", "discontinued", "soldout"}


def _as_type_set(raw_type) -> set[str]:
    if raw_type is None:
        return set()
    if isinstance(raw_type, str):
        return {raw_type.rsplit("/", 1)[-1]}  # strips a possible "http://schema.org/" prefix
    if isinstance(raw_type, list):
        out: set[str] = set()
        for t in raw_type:
            out |= _as_type_set(t)
        return out
    return set()


def _first_offer(item: dict) -> dict | None:
    offers = item.get("offers")
    if offers is None:
        return None
    if isinstance(offers, list):
        return offers[0] if offers else None
    if isinstance(offers, dict):
        return offers
    return None


def _price_to_paise(price) -> int | None:
    """schema.org `price` is in the currency's major unit (rupees for
    INR). Converting to int paise is the one place a float-rounding bug
    could violate money-is-int-paise-everywhere -- hence the explicit
    round(), not int() truncation, and the isolation of this logic to
    one tested function."""
    if price is None:
        return None
    try:
        return round(float(price) * 100)
    except (TypeError, ValueError):
        return None


def _availability_to_in_stock(availability) -> bool | None:
    if not isinstance(availability, str):
        return None
    token = availability.rsplit("/", 1)[-1].strip().lower()
    if token in _IN_STOCK_MARKERS:
        return True
    if token in _OUT_OF_STOCK_MARKERS:
        return False
    return None


def _image_to_url(image) -> str | None:
    if isinstance(image, str):
        return image
    if isinstance(image, list) and image:
        return _image_to_url(image[0])
    if isinstance(image, dict):
        return image.get("url") or image.get("@id")
    return None


def _brand_to_attr(brand) -> str | None:
    if isinstance(brand, str):
        return brand
    if isinstance(brand, dict):
        return brand.get("name")
    return None


def extract_structured(html: str, source_url: str) -> list[RawProduct]:
    try:
        data = extruct.extract(
            html, base_url=source_url, syntaxes=["json-ld", "microdata", "rdfa"],
            uniform=True, errors="ignore",
        )
    except Exception:
        # A pathological page (e.g. severely malformed markup extruct's
        # parsers can't recover from) should degrade to "no structured
        # data found" and fall through to the LLM path, not crash the
        # whole toolkit run.
        return []

    products: list[RawProduct] = []
    for syntax, method in (("json-ld", "schema_org"), ("microdata", "microdata"), ("rdfa", "rdfa")):
        for item in data.get(syntax, []):
            if not isinstance(item, dict):
                continue
            if not (_as_type_set(item.get("@type")) & _PRODUCT_TYPES):
                continue
            try:
                offer = _first_offer(item) or {}
                attributes: dict[str, str] = {}
                brand = _brand_to_attr(item.get("brand"))
                if brand:
                    attributes["brand"] = brand

                products.append(
                    RawProduct(
                        sku=item.get("sku") or item.get("mpn") or item.get("productID"),
                        title=item.get("name"),
                        description=item.get("description"),
                        category=item.get("category"),
                        price_paise=_price_to_paise(offer.get("price")),
                        currency=offer.get("priceCurrency", "INR"),
                        in_stock=_availability_to_in_stock(offer.get("availability")),
                        image_url=_image_to_url(item.get("image")),
                        attributes=attributes,
                        source_url=source_url,
                        extraction_method=method,
                    )
                )
            except Exception:
                continue  # one malformed item must not take down the rest of the extraction

    return products
