"""Hash-chain linkage: CartMandate.intent_hash must match the IntentMandate
it responds to; PaymentMandate.{intent_hash,cart_hash} must match both
mandates it authorizes. This is what makes the mandate chain tamper-
evident end to end, not just each mandate individually signed."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .hashing import mandate_hash
from .schemas import CartMandate, IntentMandate, PaymentMandate


@dataclass(frozen=True)
class ChainResult:
    intact: bool
    violations: tuple[str, ...] = ()


def verify_chain(
    intent: IntentMandate,
    cart: CartMandate,
    payment: PaymentMandate | None = None,
) -> ChainResult:
    violations: list[str] = []
    intent_h = mandate_hash(intent)

    if cart.intent_hash != intent_h:
        violations.append(
            f"cart.intent_hash ({cart.intent_hash}) != hash(intent) ({intent_h})"
        )

    if payment is not None:
        cart_h = mandate_hash(cart)
        if payment.cart_hash != cart_h:
            violations.append(
                f"payment.cart_hash ({payment.cart_hash}) != hash(cart) ({cart_h})"
            )
        if payment.intent_hash != intent_h:
            violations.append(
                f"payment.intent_hash ({payment.intent_hash}) != hash(intent) ({intent_h})"
            )

    return ChainResult(intact=not violations, violations=tuple(violations))


def is_expired(mandate: IntentMandate | CartMandate, now: datetime) -> bool:
    return now > mandate.expires_at
