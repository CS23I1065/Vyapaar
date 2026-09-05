"""
Deterministic upsell candidate generation -- the revenue side of a
merchant's agent: grow revenue while staying sellable to AI buyers.
An LLM may phrase the pitch text shown to a
human (driver.py, optional); it never decides what to offer, at what
price, or whether to offer at all.

Structural asymmetry, not just a convention: these functions don't take
an IntentMandate parameter at all -- there is no `intent.budget` to read
here even by mistake, because `intent` was never passed in. A real
salesperson pitches from their own catalog and margin rules, not the
customer's wallet; the buyer's policy engine, which DOES see the budget,
is the only thing that decides whether an offer is affordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from merchant_kit.schemas import CatalogItem

UpsellType = Literal["ACCESSORY", "QUANTITY", "PREMIUM_SUB"]
# ACCESSORY exists as a function but is NOT in the default priority order:
# a salesperson pitching a mug to a magnet buyer is not helpful upselling,
# it's noise. The default path is: premium version of what you want first,
# then more of the same. ACCESSORY is available for callers that explicitly
# want cross-category complementary pitches.


@dataclass(frozen=True)
class UpsellCandidate:
    kind: UpsellType
    item: CatalogItem
    quantity: int = 1


def find_accessory(primary: CatalogItem, catalog: list[CatalogItem]) -> UpsellCandidate | None:
    """A complementary item, cheaper than the primary. Deliberately
    requires a DIFFERENT category, not the same one -- a real accessory
    (a pickle alongside rice, a case alongside a phone) is typically a
    different kind of thing entirely; "same category, cheaper" is what
    find_premium_substitute's inverse already models (same category,
    pricier), and conflating the two produced a genuinely wrong result
    in testing: it picked a same-category item as an "accessory" when it
    was really just a cheaper alternative to the primary, not a
    complementary add-on."""
    candidates = [c for c in catalog if c.category != primary.category and c.price_paise < primary.price_paise]
    if not candidates:
        return None
    return UpsellCandidate(kind="ACCESSORY", item=min(candidates, key=lambda c: c.price_paise))


def find_quantity_bump(primary: CatalogItem, *, bump_to: int = 2) -> UpsellCandidate:
    """A bundle/volume step-up on the SAME sku that was requested, as an
    ADDITIONAL line (is_upsell=True) rather than mutating the primary
    line's own quantity -- keeps "what was asked for" vs "what was
    pitched" auditable as two distinct lines."""
    return UpsellCandidate(kind="QUANTITY", item=primary, quantity=bump_to)


def find_premium_substitute(primary: CatalogItem, catalog: list[CatalogItem]) -> UpsellCandidate | None:
    """A higher-tier item in the same category, strictly pricier --
    picks the cheapest of the pricier options, not the most expensive."""
    candidates = [c for c in catalog if c.category == primary.category and c.sku != primary.sku and c.price_paise > primary.price_paise]
    if not candidates:
        return None
    return UpsellCandidate(kind="PREMIUM_SUB", item=min(candidates, key=lambda c: c.price_paise))


# Deterministic priority order for benign mode: try each in turn, use the
# first that finds a candidate. A business rule, not an LLM decision.
# Priority: "for a bit more, get the better version" (premium substitute).
# find_quantity_bump is excluded from default priority to prevent artificial
# quantity inflation (x2) on requested items.
UPSELL_PRIORITY: tuple = (find_premium_substitute,)


def select_one_upsell(primary: CatalogItem, catalog: list[CatalogItem]) -> UpsellCandidate | None:
    for finder in UPSELL_PRIORITY:
        candidate = finder(primary, catalog)
        if candidate is not None:
            return candidate
    return None

