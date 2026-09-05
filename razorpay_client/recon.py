"""
Tier 1, capture-time reconciliation -- closes the loop from signed
authorization to money actually moved. Compares the amount Razorpay
actually captured against `cart.total_paise`, the number the merchant's
own CartMandate signature promised. A non-zero delta produces a
DisputeArtifact bundling both signed mandates plus the Razorpay record --
enough to show, byte for byte, "the merchant charged X; the merchant's
own signature said Y."

This is a real control, not a formality: `buyer/execute.py` binds
`intent_hash`/`cart_hash`/`payment_mandate_id` into the Razorpay order's
`notes` at creation time, so Tier 1 is checking Razorpay's own capture
record against a hash that was written there before any money moved --
there is no window for the comparison itself to be fed stale or
attacker-controlled data.

Independence caveat, stated plainly rather than overclaimed: Tier 1
reconciles against what the BUYER AGENT itself fetched from Razorpay, so
in principle a channel that could fool the agent's own HTTP client could
fool this too. Tier 2 (webhooks.py) is the independent channel -- the
event arrives from Razorpay out of band, not through a request this
process initiated.

A mismatch here is wired to the same remedy buyer/cancel.py provides for
an explicit user request: `reason="reconciliation_mismatch"`. Detection
without a way to act on it is only half a control.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mandates.schemas import CartMandate, Envelope, PaymentMandate
from razorpay_client.client import RazorpayClient


@dataclass(frozen=True)
class ReconDelta:
    matched: bool
    signed_amount_paise: int
    charged_amount_paise: int | None  # None if nothing was ever captured
    delta_paise: int
    order_id: str
    captured_payment_id: str | None
    detail: str

    @property
    def is_overcharge(self) -> bool:
        return self.charged_amount_paise is not None and self.delta_paise > 0

    @property
    def is_undercharge(self) -> bool:
        return self.charged_amount_paise is not None and self.delta_paise < 0


@dataclass(frozen=True)
class DisputeArtifact:
    """Everything needed to show a human, or a Razorpay support ticket,
    exactly what was promised versus what happened. Deliberately holds
    the raw Envelope (payload + signature) rather than the parsed
    CartMandate -- a dispute artifact is stronger evidence when it
    carries the actual signed bytes, not a re-serialization of them."""

    cart_envelope: Envelope
    payment_mandate: PaymentMandate
    razorpay_order: dict
    razorpay_payments: dict
    delta: ReconDelta
    created_at: datetime


def reconcile_capture(
    cart: CartMandate,
    cart_envelope: Envelope,
    payment_mandate: PaymentMandate,
    *,
    client: RazorpayClient,
    now: datetime,
) -> DisputeArtifact:
    """Fetches the order and its payments fresh from Razorpay and
    compares the captured total against cart.total_paise. Always returns
    a DisputeArtifact -- matched=True is not a special case, it is a
    dispute artifact whose delta happens to be zero, so the same audit
    shape covers both the clean path and the mismatch path."""
    if payment_mandate.razorpay_order_id is None:
        raise ValueError("payment_mandate has no razorpay_order_id -- nothing was ever charged")

    order = client.fetch_order(payment_mandate.razorpay_order_id)
    payments = client.fetch_order_payments(payment_mandate.razorpay_order_id)

    captured = [p for p in payments.get("items", []) if p.get("status") == "captured"]
    charged_amount = sum(p["amount"] for p in captured) if captured else None
    captured_payment_id = captured[0]["id"] if len(captured) == 1 else None

    signed_amount = cart.total_paise
    delta = (charged_amount - signed_amount) if charged_amount is not None else 0
    matched = charged_amount is not None and delta == 0 and len(captured) <= 1

    if charged_amount is None:
        detail = f"order {order.get('id')} has no captured payment yet"
    elif len(captured) > 1:
        detail = f"order {order.get('id')} has {len(captured)} captured payments -- ambiguous, treating as unmatched"
    elif delta == 0:
        detail = f"charged amount matches the signed cart total ({signed_amount} paise)"
    else:
        direction = "OVERCHARGED" if delta > 0 else "undercharged"
        detail = (
            f"{direction} by {abs(delta)} paise: Razorpay captured {charged_amount}, "
            f"the merchant's own signed cart said {signed_amount}"
        )

    result = ReconDelta(
        matched=matched, signed_amount_paise=signed_amount, charged_amount_paise=charged_amount,
        delta_paise=delta, order_id=payment_mandate.razorpay_order_id,
        captured_payment_id=captured_payment_id, detail=detail,
    )
    return DisputeArtifact(
        cart_envelope=cart_envelope, payment_mandate=payment_mandate,
        razorpay_order=order, razorpay_payments=payments, delta=result, created_at=now,
    )


def verify_notes_binding(razorpay_order: dict, intent_hash: str, cart_hash: str, payment_mandate_id: str) -> list[str]:
    """Checks the mandate-binding claim execute.py made at order-creation
    time (`notes`) against what Razorpay actually has on file for this
    order. Returns a list of violation strings, empty if the binding
    holds. This is what makes Tier 1 more than an amount check: the
    hashes prove the order was created FOR this exact intent and cart,
    not merely for the same rupee amount by coincidence."""
    notes = razorpay_order.get("notes", {})
    violations = []
    if notes.get("intent_hash") != intent_hash:
        violations.append(f"order notes intent_hash {notes.get('intent_hash')!r} != {intent_hash!r}")
    if notes.get("cart_hash") != cart_hash:
        violations.append(f"order notes cart_hash {notes.get('cart_hash')!r} != {cart_hash!r}")
    if notes.get("payment_mandate_id") != payment_mandate_id:
        violations.append(
            f"order notes payment_mandate_id {notes.get('payment_mandate_id')!r} != {payment_mandate_id!r}"
        )
    return violations
