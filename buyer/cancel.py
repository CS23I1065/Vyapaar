"""
Cancellation/refund: the recourse for once execute.py has created an
order and there is otherwise no path back.

razorpay_client/client.py has working capture_payment and create_refund
methods; this module is what actually calls them from the buyer side.
For a system whose entire premise is "an agent spends your money without
you watching," "I changed my mind" and "it bought the wrong thing" are
not edge cases -- they are the first two questions any real user asks.

The design mirrors execute.py deliberately: cancel_payment() is a hard
gate, not a bare API call. A CancellationMandate must actually reference
(by hash) the PaymentMandate it reverses, mirroring the same chain-
binding discipline mandates/chain.py already enforces between Intent ->
Cart -> Payment. A cancellation that does not chain-bind to a real prior
payment is refused before Razorpay is ever called -- fail closed, same
as execute.py's ExecutionRefused.

Two triggers, both real:
  1. build_cancellation_mandate(..., reason="user_requested", ...) -- an
     explicit ask ("cancel the last order").
  2. build_cancellation_mandate(..., reason="reconciliation_mismatch",
     ...) -- wired from razorpay_client/recon.py when a captured amount
     does not match what was actually signed. Detection without remedy
     is only half a control; this is the remedy half.

Every cancellation is itself signed by the human (principal_kid) -- an
agent proposes a cancellation the same way it proposes anything else; it
never authorizes one on its own.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from mandates.hashing import mandate_hash
from mandates.keys import KeyPair
from mandates.schemas import CancellationMandate, PaymentMandate
from mandates.sign import sign_mandate
from razorpay_client.client import RazorpayClient


class CancellationRefused(RuntimeError):
    """Raised instead of ever calling create_refund with a mandate that
    does not genuinely reference the payment it claims to reverse."""


def build_cancellation_mandate(
    payment: PaymentMandate,
    *,
    principal_kid: str,
    reason: Literal["user_requested", "reconciliation_mismatch"],
    now: datetime,
    amount_paise: int | None = None,
) -> CancellationMandate:
    return CancellationMandate(
        mandate_id=str(uuid.uuid4()),
        created_at=now,
        payment_hash=mandate_hash(payment),
        principal_kid=principal_kid,
        reason=reason,
        amount_paise=amount_paise,
    )


def cancel_payment(
    cancellation: CancellationMandate,
    payment: PaymentMandate,
    *,
    client: RazorpayClient,
) -> dict:
    """Refuses to call Razorpay at all unless cancellation.payment_hash
    genuinely matches hash(payment) -- the same "no code path skips
    verification" discipline execute.py holds for ExecutionRefused.
    `payment` must be the SAME PaymentMandate execute.py actually built
    and used, not a reconstruction from user-supplied fields; passing a
    forged one is exactly what this check exists to catch."""
    actual_hash = mandate_hash(payment)
    if cancellation.payment_hash != actual_hash:
        raise CancellationRefused(
            f"cancellation.payment_hash ({cancellation.payment_hash}) does not match "
            f"hash(payment) ({actual_hash}) -- refusing to refund a payment this "
            f"cancellation does not actually reference"
        )
    if payment.razorpay_order_id is None:
        raise CancellationRefused("payment has no razorpay_order_id -- nothing was ever charged")

    order = client.fetch_order(payment.razorpay_order_id)
    payments = client.fetch_order_payments(payment.razorpay_order_id)
    captured = [p for p in payments.get("items", []) if p.get("status") == "captured"]
    if not captured:
        raise CancellationRefused(
            f"order {payment.razorpay_order_id} has no captured payment to refund "
            f"(order status: {order.get('status')})"
        )
    payment_id = captured[0]["id"]

    return client.create_refund(
        payment_id,
        amount_paise=cancellation.amount_paise,
        notes={
            "cancellation_mandate_id": cancellation.mandate_id,
            "payment_hash": cancellation.payment_hash,
            "reason": cancellation.reason,
        },
    )


def sign_and_cancel(
    payment: PaymentMandate,
    *,
    principal_kid: str,
    reason: Literal["user_requested", "reconciliation_mismatch"],
    now: datetime,
    human_keypair: KeyPair,
    client: RazorpayClient,
    amount_paise: int | None = None,
):
    """Convenience: build, sign, chain-verify, and execute a cancellation
    in one call -- the shape a CLI or the demo actually wants. Returns
    (cancellation_envelope, refund_response)."""
    cancellation = build_cancellation_mandate(
        payment, principal_kid=principal_kid, reason=reason, now=now, amount_paise=amount_paise
    )
    envelope = sign_mandate(cancellation, human_keypair)
    refund = cancel_payment(cancellation, payment, client=client)
    return envelope, refund
