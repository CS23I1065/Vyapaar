"""
Runs ONLY on Decision.outcome == "APPROVED". Builds a PaymentMandate,
binds its hash chain into the Razorpay order's `notes` (the load-bearing
mandate-binding claim), and returns enough to drive Tier 1 capture-time
reconciliation.

ExecutionRefused is a hard invariant, not a soft warning: there must be
no code path in this codebase that reaches Razorpay without a passing
evaluate() immediately before it. See buyer/approval.py's docstring for
the other half of this -- an approved escalation still runs through a
fresh Decision, and only THAT Decision is what this function trusts.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from mandates.hashing import mandate_hash
from mandates.schemas import CartMandate, IntentMandate, PaymentMandate
from razorpay_client.client import RazorpayClient

from .policy_engine import Decision


class ExecutionRefused(RuntimeError):
    pass


def execute(
    decision: Decision, intent: IntentMandate, cart: CartMandate, *, client: RazorpayClient, now: datetime
) -> tuple[PaymentMandate, dict]:
    if decision.outcome != "APPROVED":
        raise ExecutionRefused(f"refusing to execute a {decision.outcome} decision")

    intent_hash = mandate_hash(intent)
    cart_hash = mandate_hash(cart)
    payment_mandate_id = str(uuid.uuid4())

    order = client.create_order(
        amount_paise=cart.total_paise,
        receipt=payment_mandate_id,
        notes={"intent_hash": intent_hash, "cart_hash": cart_hash, "payment_mandate_id": payment_mandate_id},
    )

    payment_mandate = PaymentMandate(
        version="1.0",
        mandate_id=payment_mandate_id,
        created_at=now,
        intent_hash=intent_hash,
        cart_hash=cart_hash,
        amount_paise=cart.total_paise,
        rail="razorpay_test",
        razorpay_order_id=order["id"],
    )
    return payment_mandate, order
