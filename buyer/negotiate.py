"""
The bounded merchant<->buyer negotiation loop. Hard cap MAX_TURNS=6, then
forced resolution -- prevents drift/looping.

The merchant side (merchant_agent/) is not imported here at all --
negotiate.py only depends on a MerchantDriver callable contract, avoiding
any circular dependency and keeping this module fully testable with stub
drivers independent of merchant_agent/.

Structural anti-injection property, not just a convention: a merchant's
only way to influence what happens next is by returning a "cart" (a
signed CartMandate, which is DATA the policy engine evaluates) or a
"decline". There is no "pitch"/free-text response kind that feeds back
into this loop's control flow or into any LLM prompt here -- narrative
text a merchant wants to show a human rides along purely for audit/
display (MerchantResponse.pitch_text) and is never read by this module's
own logic. Merchant free text is never appended to the planner's
instruction context.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from mandates.schemas import Envelope, IntentMandate

MAX_TURNS = 6


@dataclass(frozen=True)
class MerchantResponse:
    kind: Literal["cart", "decline"]
    cart_envelope: Envelope | None = None  # present iff kind == "cart"
    decline_reason: str | None = None  # present iff kind == "decline"
    pitch_text: str | None = None  # audit/display only -- see module docstring; never read below
    alternative_envelopes: tuple[Envelope, ...] = ()
    # Other signed carts the merchant is willing to honour for this same
    # intent. The merchant proposes these from its OWN catalog and margin
    # rules, still blind to the buyer's budget and soft preferences
    # (merchant_agent/upsell.py does not accept an IntentMandate at all,
    # and that stays true) -- the buyer ranks them itself in buyer/rank.py.
    #
    # This exists because selection used to be min(price) on the merchant
    # side, which made the buyer's signed brand and target-price
    # preferences unreadable dead weight: with exactly one cart on offer
    # there was nothing to rank. Every alternative is a fully signed
    # CartMandate and is put through the SAME evaluate() as the primary --
    # an alternative is a candidate, never a shortcut.


MerchantDriver = Callable[[int, IntentMandate], MerchantResponse]


@dataclass(frozen=True)
class NegotiationResult:
    outcome: Literal["cart_received", "declined", "timeout"]
    cart_envelope: Envelope | None
    turns_used: int
    reason: str
    alternative_envelopes: tuple[Envelope, ...] = ()


def negotiate(intent: IntentMandate, *, merchant_driver: MerchantDriver, max_turns: int = MAX_TURNS) -> NegotiationResult:
    for turn in range(1, max_turns + 1):
        response = merchant_driver(turn, intent)
        if response.kind == "cart":
            if response.cart_envelope is None:
                raise ValueError('MerchantResponse.kind == "cart" but cart_envelope is None')
            return NegotiationResult(
                "cart_received", response.cart_envelope, turn, "merchant proposed a cart",
                alternative_envelopes=response.alternative_envelopes,
            )
        if response.kind == "decline":
            return NegotiationResult("declined", None, turn, response.decline_reason or "merchant declined")
        raise ValueError(f"unknown MerchantResponse.kind {response.kind!r}")

    return NegotiationResult("timeout", None, max_turns, "NEGOTIATION_TIMEOUT: turn cap reached with no resolution")
