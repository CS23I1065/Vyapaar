"""
Shop around: negotiate against N merchants, evaluate every cart each one
offers, and pick the best APPROVED one by buyer/rank.py's scoring.

The problem this replaces
--------------------------
buyer/negotiate.py's negotiate() talks to exactly one MerchantDriver. So
the flow a user actually got was: one shop, its cheapest matching item,
take it or leave it. If that shop declined, the purchase simply failed --
even when another onboarded merchant had the item in stock. "Go buy me
this" implies shopping around; a human comparing several shops is
exactly the tedium an agent should absorb, and this was the part nothing
did.

What this module adds
----------------------
Two things fall out for free once negotiation happens against several
merchants instead of one:

  1. A merchant declining becomes a non-event instead of a dead end --
     shop_around() keeps going to the next one.
  2. The audit trail gains something genuinely worth showing a user:
     "three shops checked, here's what each offered, here's why this one
     won" (ShoppingResult.rows), which is a far better answer to "why did
     you buy this?" than a single gate code.

Purity boundary, kept the same as everywhere else in buyer/: this module
does I/O-shaped orchestration (it calls merchant drivers, which is the
same "network boundary" negotiate.py already crosses), but the DECISION
for each candidate cart is made by the real, unmodified, pure evaluate()
-- shop_around() never approves anything itself. It only chooses which
of several independently-APPROVED decisions to act on, via the ranker,
which is scoring, not authorization.

Bounded by design: MAX_MERCHANTS caps how many shops get
tried in one run, and each one still gets negotiate.py's own MAX_TURNS
cap per merchant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from buyer.catalog import VerifiedCatalog
from buyer.negotiate import MerchantDriver, negotiate
from buyer.policy_engine import Decision, VerificationContext, evaluate
from buyer.rank import Candidate, rank
from mandates.keys import KeyDirectory
from mandates.schemas import CartMandate, Envelope, IntentMandate, MerchantPolicy

MAX_MERCHANTS = 5


@dataclass(frozen=True)
class MerchantEndpoint:
    """Everything shop_around() needs to negotiate with and evaluate
    against one merchant. `catalog` must already be independently
    verified (buyer.catalog.load_verified_catalog) -- this module does
    not fetch or verify anything itself, same purity boundary as
    evaluate()'s own merchant_catalog input."""

    merchant_id: str
    driver: MerchantDriver
    merchant_policy: MerchantPolicy
    catalog: VerifiedCatalog
    key_directory: KeyDirectory  # must have the merchant's key registered


@dataclass(frozen=True)
class MerchantOutcome:
    """One row of the "here's what each shop offered" audit trail."""

    merchant_id: str
    negotiation_outcome: str  # "cart_received" | "declined" | "timeout"
    reason: str
    decision: Decision | None
    cart: CartMandate | None
    turns_used: int


@dataclass(frozen=True)
class ShoppingResult:
    winner: MerchantOutcome | None  # None iff nobody produced an APPROVED cart
    winning_cart_envelope: Envelope | None
    rows: tuple[MerchantOutcome, ...]  # every merchant tried, in order

    @property
    def approved(self) -> bool:
        return self.winner is not None

    def narrative(self) -> str:
        """A short readable answer to "why did you buy this?" -- the
        thing audit/'s hash-chained log is built FOR a machine to verify,
        not for a shopper to read. This is the human-facing summary."""
        lines = [f"Checked {len(self.rows)} shop(s):"]
        for row in self.rows:
            if row.decision is not None and row.decision.outcome == "APPROVED":
                mark = "WON" if self.winner is not None and row.merchant_id == self.winner.merchant_id else "approved but not chosen"
                total = f"Rs{row.cart.total_paise / 100:.2f}" if row.cart else "?"
                lines.append(f"  - {row.merchant_id}: {total} ({mark})")
            elif row.decision is not None:
                lines.append(f"  - {row.merchant_id}: {row.decision.outcome} ({row.decision.blocking_code})")
            else:
                lines.append(f"  - {row.merchant_id}: {row.negotiation_outcome} ({row.reason})")
        return "\n".join(lines)


def _cart_to_candidate(merchant_id: str, cart: CartMandate) -> Candidate:
    primary = next((line for line in cart.lines if not line.is_upsell), cart.lines[0])
    return Candidate(
        merchant_id=merchant_id,
        sku=primary.sku,
        title=primary.title,
        category=primary.category,
        unit_price_paise=primary.unit_price_paise,
        in_stock=True,  # a cart that reached us at all is, definitionally, available
        attributes=primary.attributes,
        total_paise=cart.total_paise,
    )


def shop_around(
    intent: IntentMandate,
    intent_envelope: Envelope,
    endpoints: Mapping[str, MerchantEndpoint],
    *,
    now: datetime,
    max_merchants: int = MAX_MERCHANTS,
    require_agent_identity: bool = False,
    webbotauth_verified: bool = False,
    low_confidence_interpretation: bool = False,
) -> ShoppingResult:
    """Negotiates against each endpoint in turn (bounded by
    max_merchants), evaluates every cart offered, and picks the best
    APPROVED one by buyer.rank's scoring against intent.soft.

    Only merchants in intent.allowed_merchants (if set) are tried at
    all -- this module does not widen what the signed intent already
    authorizes; it only chooses among what it permits."""
    allowed = set(intent.allowed_merchants) if intent.allowed_merchants is not None else None
    tried = [
        merchant_id for merchant_id in endpoints
        if allowed is None or merchant_id in allowed
    ][:max_merchants]

    rows: list[MerchantOutcome] = []
    approved_candidates: list[tuple[Candidate, MerchantOutcome, Envelope]] = []

    for merchant_id in tried:
        endpoint = endpoints[merchant_id]
        negotiation = negotiate(intent, merchant_driver=endpoint.driver)

        if negotiation.outcome != "cart_received":
            rows.append(
                MerchantOutcome(
                    merchant_id=merchant_id, negotiation_outcome=negotiation.outcome,
                    reason=negotiation.reason, decision=None, cart=None,
                    turns_used=negotiation.turns_used,
                )
            )
            continue

        cart_envelope = negotiation.cart_envelope
        assert cart_envelope is not None
        cart = CartMandate.model_validate(cart_envelope.payload)
        ctx = VerificationContext(
            key_directory=endpoint.key_directory,
            intent_envelope=intent_envelope,
            cart_envelope=cart_envelope,
            require_agent_identity=require_agent_identity,
            webbotauth_verified=webbotauth_verified,
            low_confidence_interpretation=low_confidence_interpretation,
            merchant_catalog=endpoint.catalog,
        )
        decision = evaluate(intent, cart, endpoint.merchant_policy, ctx, now)
        outcome = MerchantOutcome(
            merchant_id=merchant_id, negotiation_outcome=negotiation.outcome,
            reason=negotiation.reason, decision=decision, cart=cart,
            turns_used=negotiation.turns_used,
        )
        rows.append(outcome)

        if decision.outcome == "APPROVED":
            approved_candidates.append((_cart_to_candidate(merchant_id, cart), outcome, cart_envelope))

    if not approved_candidates:
        return ShoppingResult(winner=None, winning_cart_envelope=None, rows=tuple(rows))

    ranked = rank([c for c, _, _ in approved_candidates], intent.soft)
    best_sku_merchant = (ranked[0].candidate.merchant_id, ranked[0].candidate.sku)
    winner_outcome, winner_envelope = next(
        (outcome, envelope)
        for candidate, outcome, envelope in approved_candidates
        if (candidate.merchant_id, candidate.sku) == best_sku_merchant
    )
    return ShoppingResult(winner=winner_outcome, winning_cart_envelope=winner_envelope, rows=tuple(rows))
