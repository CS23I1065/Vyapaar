"""
The concrete implementation of buyer.negotiate.MerchantDriver -- the
benign half of merchant_agent/ (adversarial mode lives in redteam/).
Matches a buyer's HARD constraints against its own catalog, proposes a
signed CartMandate for the matched primary item, and in "benign" upsell
mode adds ONE additional pitched line from its own catalog/margin rules
-- capped at one offer per session.

One MerchantAgentDriver instance per negotiation session: `_offer_made`
is mutable session state, and reusing one instance across different
buyers' negotiations would incorrectly suppress a second buyer's upsell.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from buyer.negotiate import MerchantResponse
from mandates.hashing import mandate_hash
from mandates.keys import KeyPair
from mandates.schemas import CartLine, CartMandate, IntentMandate, Provenance
from mandates.sign import sign_mandate
from merchant_kit.schemas import CatalogItem

from .upsell import UpsellCandidate, select_one_upsell

UpsellMode = Literal["off", "benign"]  # "adversarial" lives in redteam/


def _merchant_asserted_provenance(source_url: str, now: datetime) -> Provenance:
    """Re-stamped, not copied from the catalog's own build-time
    provenance (which is always signature_verified=False -- see
    merchant_kit/normalize.py's docstring: it only PRODUCES data to be
    signed later). By the time the merchant proposes a CartMandate it's
    about to sign with its own key, it is directly asserting these
    values -- that IS what extraction_method="merchant_signed" +
    signature_verified=True means. Without this re-stamp, every CartLine
    built from a CatalogItem would fail G1.1 (PROVENANCE_INSUFFICIENT)
    unconditionally, since normalize.py's own output never carries
    verified=True."""
    return Provenance(source_url=source_url, extraction_method="merchant_signed", signature_verified=True, extracted_at=now)


def _catalog_item_to_line(
    item: CatalogItem, *, now: datetime, quantity: int = 1, is_substitute: bool = False, is_upsell: bool = False
) -> CartLine:
    source_url = next(iter(item.field_provenance.values())).source_url if item.field_provenance else "unknown"
    prov = _merchant_asserted_provenance(source_url, now)
    # NOTE: CartLine's field is `unit_price_paise`, not `price_paise` --
    # CatalogItem (merchant_kit/schemas.py) and CartLine (mandates/schemas.py)
    # name the same concept differently. Using the wrong key here silently
    # produced a field_provenance dict with NO entry under the key G1.1
    # actually reads, which made every generated cart fail
    # PROVENANCE_INSUFFICIENT unconditionally -- caught by the first
    # integration test that ran a cart through the real policy engine.
    field_provenance = {f: prov for f in ("title", "description", "category", "unit_price_paise", "in_stock")}
    return CartLine(
        sku=item.sku, title=item.title, category=item.category, unit_price_paise=item.price_paise,
        quantity=quantity, attributes=item.attributes, is_substitute=is_substitute, is_upsell=is_upsell,
        field_provenance=field_provenance,
    )


def _matching_items(intent: IntentMandate, catalog: list[CatalogItem]) -> list[CatalogItem]:
    """Reads intent.hard ONLY (category, required/excluded categories,
    attributes) -- normal search behaviour, exactly what a storefront
    search page shows. Never reads intent.budget or intent.soft; that
    separation is what upsell.py's stronger guarantee (never even
    receiving `intent` at all) builds on.

    Returned cheapest-first. That ordering is the MERCHANT's own default
    presentation, not a claim about what the buyer wants -- the buyer
    ranks for itself in buyer/rank.py, where the budget and the soft
    preferences actually live."""
    candidates = list(catalog)
    if intent.hard.category is not None:
        candidates = [c for c in candidates if c.category == intent.hard.category]
    for key, value in intent.hard.required_attributes.items():
        candidates = [c for c in candidates if c.attributes.get(key) == value]
    candidates = [c for c in candidates if c.category not in intent.hard.excluded_categories]
    return sorted(candidates, key=lambda c: (c.price_paise, c.sku))


def _match_primary(intent: IntentMandate, catalog: list[CatalogItem]) -> CatalogItem | None:
    matches = _matching_items(intent, catalog)
    return matches[0] if matches else None


def _default_pitch_text(candidate: UpsellCandidate) -> str:
    price = candidate.item.price_paise / 100
    if candidate.kind == "ACCESSORY":
        return f"Since you're getting that, would you also like {candidate.item.title} for ₹{price:.2f}?"
    if candidate.kind == "PREMIUM_SUB":
        return f"For a bit more, our {candidate.item.title} is a popular upgrade at ₹{price:.2f}."
    if candidate.kind == "QUANTITY":
        return f"Want to grab {candidate.quantity}x {candidate.item.title} instead of just one?"
    return "We have a suggestion for you."


@dataclass
class MerchantAgentDriver:
    catalog: list[CatalogItem]
    merchant_id: str
    merchant_keypair: KeyPair
    now: datetime
    upsell_mode: UpsellMode = "off"
    max_alternatives: int = 0
    # How many additional matching items to offer as separately-signed
    # alternative carts, for the buyer to rank. 0 (the default) preserves
    # the original single-cart behaviour exactly. Bounded on purpose --
    # an unbounded feed of alternatives is the same discipline problem
    # `_offer_made` already caps for upsells.
    validity: timedelta = timedelta(minutes=10)
    pitch_phraser: Callable[[UpsellCandidate], str] = _default_pitch_text
    _offer_made: bool = field(default=False, init=False, repr=False)

    def _sign_single_line_cart(self, item: CatalogItem, intent: IntentMandate, turn: int, tag: str):
        line = _catalog_item_to_line(item, now=self.now, quantity=1)
        cart = CartMandate(
            mandate_id=f"cart-{tag}-{turn}-{secrets.token_hex(4)}",
            issued_at=self.now,
            expires_at=self.now + self.validity,
            merchant_id=self.merchant_id,
            merchant_kid=self.merchant_keypair.kid,
            intent_hash=mandate_hash(intent),
            lines=[line],
            subtotal_paise=line.line_total_paise,
            tax_paise=0,
            shipping_paise=0,
            total_paise=line.line_total_paise,
        )
        return sign_mandate(cart, self.merchant_keypair)

    def __call__(self, turn: int, intent: IntentMandate) -> MerchantResponse:
        matches = _matching_items(intent, self.catalog)
        if not matches:
            return MerchantResponse(kind="decline", decline_reason="no matching product in catalog")
        primary = matches[0]

        lines = [_catalog_item_to_line(primary, now=self.now, quantity=1)]
        pitch_text = None
        alternatives_list = []

        if self.upsell_mode == "benign" and not self._offer_made:
            candidate = select_one_upsell(primary, self.catalog)
            if candidate is not None:
                # Add the clean, non-upsold primary cart as the first alternative
                alternatives_list.append(self._sign_single_line_cart(primary, intent, turn, "primary-clean"))
                
                if candidate.kind in ("QUANTITY", "PREMIUM_SUB"):
                    # Mutate/replace primary line to avoid G8.3 duplicate SKUs or double-charging
                    lines[0] = _catalog_item_to_line(candidate.item, now=self.now, quantity=candidate.quantity, is_upsell=True)
                else:
                    lines.append(_catalog_item_to_line(candidate.item, now=self.now, quantity=candidate.quantity, is_upsell=True))
                pitch_text = self.pitch_phraser(candidate)
                self._offer_made = True

        subtotal = sum(line.line_total_paise for line in lines)
        cart = CartMandate(
            mandate_id=f"cart-{turn}-{secrets.token_hex(4)}",
            issued_at=self.now,
            expires_at=self.now + self.validity,
            merchant_id=self.merchant_id,
            merchant_kid=self.merchant_keypair.kid,
            intent_hash=mandate_hash(intent),
            lines=lines,
            subtotal_paise=subtotal,
            tax_paise=0,
            shipping_paise=0,
            total_paise=subtotal,
        )
        envelope = sign_mandate(cart, self.merchant_keypair)

        # Each alternative is a fully signed, independently evaluable
        # CartMandate bound to this same intent hash -- not a hint, not a
        # price quote. The buyer runs the SAME evaluate() over it.
        alternatives_list.extend(
            self._sign_single_line_cart(item, intent, turn, "alt")
            for item in matches[1 : 1 + self.max_alternatives]
        )
        alternatives = tuple(alternatives_list)
        return MerchantResponse(
            kind="cart", cart_envelope=envelope, pitch_text=pitch_text,
            alternative_envelopes=alternatives,
        )
