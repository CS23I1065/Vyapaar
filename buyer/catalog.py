"""
The buyer's independently-verified view of a merchant's published
catalog. Without this, a merchant could publish an item at Rs200 in its
own signed .well-known/agent-catalog.json and then sign a cart charging
Rs480 for that exact SKU, and every gate would still pass -- each one
only checks the cart against the BUYER's limits, never against the
MERCHANT's own published claims. This module is what makes that second
check possible.

Two deliberate choices here:

1. **The wire document is parsed, not imported.** This module does not
   import merchant_kit.schemas.Catalog. A published catalog is untrusted
   data arriving from another party, and the buyer should read exactly
   the fields it needs out of it rather than adopting the producer's own
   model (which would silently accept anything that model accepts, and
   couple the buyer's trust boundary to the toolkit's release cycle).

2. **Verification happens HERE, not in the policy engine.** Signature
   checking needs a KeyDirectory and, in a real deployment, an HTTP
   fetch. `evaluate()` is pure by contract -- no I/O, no network -- so it
   receives an already-verified VerifiedCatalog the same way it already
   receives already-fetched Envelopes. Everything below the
   `load_verified_catalog` boundary is trusted typed data; everything
   above it is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from mandates.keys import KeyDirectory
from mandates.schemas import Envelope
from mandates.verify import verify_envelope


class CatalogVerificationError(Exception):
    """Raised instead of returning a partial/unverified catalog. A caller
    that cannot verify a merchant's catalog must not proceed with a
    weaker one -- the G8.x gates treat a missing catalog as a REJECT, so
    swallowing this error would only move the failure somewhere less
    obvious."""


@dataclass(frozen=True)
class CatalogEntry:
    sku: str
    title: str
    category: str
    price_paise: int
    in_stock: bool


@dataclass(frozen=True)
class VerifiedCatalog:
    """A merchant's published catalog, signature-verified against that
    merchant's own key, reduced to the fields the policy engine binds
    against. `entries` is read-only: the engine is pure and must not be
    able to mutate a shared catalog between gates."""

    merchant_id: str
    merchant_kid: str
    generated_at: str
    entries: Mapping[str, CatalogEntry]

    def get(self, sku: str) -> CatalogEntry | None:
        return self.entries.get(sku)

    def __len__(self) -> int:
        return len(self.entries)


def load_verified_catalog(
    envelope: Envelope,
    key_directory: KeyDirectory,
    *,
    expected_merchant_id: str | None = None,
    required_kid: str | None = None,
) -> VerifiedCatalog:
    """Verify a signed agent-catalog.json envelope and index it by SKU.

    `required_kid` should be the merchant_kid the buyer expects (i.e. the
    one on the CartMandate it is about to evaluate). Without it, a
    catalog signed by ANY key already in the directory would verify --
    which would let one merchant's signed catalog vouch for another
    merchant's prices."""
    result = verify_envelope(envelope, key_directory, required_kid=required_kid)
    if not result.valid:
        raise CatalogVerificationError(f"merchant catalog signature invalid: {result.reason}")

    payload = envelope.payload
    merchant_id = payload.get("merchant_id")
    if not isinstance(merchant_id, str) or not merchant_id:
        raise CatalogVerificationError("merchant catalog has no merchant_id")
    if expected_merchant_id is not None and merchant_id != expected_merchant_id:
        raise CatalogVerificationError(
            f"catalog is for merchant {merchant_id!r}, expected {expected_merchant_id!r}"
        )

    entries: dict[str, CatalogEntry] = {}
    for raw in payload.get("items", []) or []:
        sku = raw.get("sku")
        price = raw.get("price_paise")
        if not isinstance(sku, str) or not sku:
            raise CatalogVerificationError("catalog item has no usable sku")
        if not isinstance(price, int) or isinstance(price, bool) or price < 0:
            # int paise only -- a float price at a money boundary is a
            # bug, and a float that arrived over the wire is worse.
            raise CatalogVerificationError(
                f"catalog item {sku!r} has a non-integer or negative price_paise: {price!r}"
            )
        if sku in entries:
            raise CatalogVerificationError(
                f"catalog publishes SKU {sku!r} more than once -- ambiguous published price"
            )
        entries[sku] = CatalogEntry(
            sku=sku,
            title=str(raw.get("title", "")),
            category=str(raw.get("category", "uncategorized")),
            price_paise=price,
            in_stock=bool(raw.get("in_stock", False)),
        )

    return VerifiedCatalog(
        merchant_id=merchant_id,
        merchant_kid=required_kid or (result.verified_kids[0] if result.verified_kids else ""),
        generated_at=str(payload.get("generated_at", "")),
        entries=MappingProxyType(entries),
    )
