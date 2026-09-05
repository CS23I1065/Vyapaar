"""
Unifies RawProduct output (from either extract/structured.py or
extract/llm.py -- never both for the same page: LLM extraction only
runs when structured extraction finds nothing) into provenance-tagged
CatalogItems.

A RawProduct missing title or price is rejected rather than silently
promoted with a made-up value -- and the rejection is recorded, not
dropped, because a stray near-empty "Product"-typed JSON-LD block turned
out to be a REAL thing found on Chumbak's own page during research (a
second Product block just named "Chumbak" with no price or sku, likely a
theme/brand-identity block mistagged as Product). The readiness report
needs to show that kind of noise, not hide it.

A missing SKU is handled differently: many real product pages (verified
against a live BigBasket page during research) simply don't expose an
internal SKU in visible text, so requiring one outright would make the
LLM fallback path frequently useless even for a perfectly good page. A
stable synthetic SKU is derived from the source URL instead -- the URL
itself uniquely identifies the product on that merchant's site -- and
the derivation is recorded (not hidden) so it's visibly distinguishable
from a real merchant-issued SKU in the readiness report.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from mandates.schemas import Provenance

from .schemas import CatalogItem, RawProduct

REQUIRED_FIELDS = ("title", "price_paise")


def _derive_sku(source_url: str) -> str:
    digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:12]
    return f"url-derived-{digest}"


def normalize(
    raw_products: list[RawProduct], *, now: datetime
) -> tuple[list[CatalogItem], list[dict], list[dict]]:
    """Returns (catalog_items, rejected, derived_skus) -- all three are
    surfaced in the readiness report (see report.py)."""
    items: list[CatalogItem] = []
    rejected: list[dict] = []
    derived_skus: list[dict] = []

    for raw in raw_products:
        missing = [f for f in REQUIRED_FIELDS if getattr(raw, f) is None]
        if missing:
            rejected.append(
                {
                    "source_url": raw.source_url,
                    "extraction_method": raw.extraction_method,
                    "title": raw.title,
                    "missing_fields": missing,
                }
            )
            continue

        sku = raw.sku
        if sku is None:
            sku = _derive_sku(raw.source_url)
            derived_skus.append({"source_url": raw.source_url, "derived_sku": sku, "title": raw.title})

        prov = Provenance(
            source_url=raw.source_url,
            extraction_method=raw.extraction_method,
            # normalize.py only PRODUCES data to be signed later
            # (merchant_kit/sign.py). Nothing is cryptographically
            # verified at this stage -- and the buyer side re-verifies
            # independently against the merchant's published signature at
            # negotiation time regardless of what this flag said here.
            signature_verified=False,
            extracted_at=now,
            confidence=raw.confidence,
        )
        field_provenance = {
            field: prov for field in ("title", "description", "category", "price_paise", "in_stock")
        }

        items.append(
            CatalogItem(
                sku=sku,
                title=raw.title,
                description=raw.description or "",
                category=raw.category or "uncategorized",
                price_paise=raw.price_paise,
                currency=raw.currency,
                # Unknown stock status defaults to NOT purchasable (fail
                # closed), not to "assume it's available" -- consistent
                # with the rest of this project's fail-closed posture.
                in_stock=raw.in_stock if raw.in_stock is not None else False,
                image_url=raw.image_url,
                attributes=raw.attributes,
                field_provenance=field_provenance,
            )
        )

    return items, rejected, derived_skus
