"""
Tests for buyer/cancel.py -- the cancel/refund path. razorpay_client
already had a working create_refund; this is what actually calls it
from the buyer side, so "I changed my mind" has a real code path.

Uses httpx.MockTransport to fake the three Razorpay calls involved
(fetch_order, fetch_order_payments, create_refund) -- no cassette needed
since these are new endpoints in a new module, and a mock transport
keeps the request/response shape explicit right here in the test.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from buyer.cancel import (
    CancellationRefused,
    build_cancellation_mandate,
    cancel_payment,
    sign_and_cancel,
)
from mandates.hashing import mandate_hash
from mandates.keys import generate_keypair
from mandates.schemas import PaymentMandate
from razorpay_client.client import RazorpayClient

UTC = timezone.utc
HUMAN_KP = generate_keypair()


def _now() -> datetime:
    return datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC)


def _payment(order_id="order_abc123") -> PaymentMandate:
    return PaymentMandate(
        mandate_id="payment-1", created_at=_now(), intent_hash="h_intent", cart_hash="h_cart",
        amount_paise=30_000, rail="razorpay_test", razorpay_order_id=order_id,
    )


def _mock_client(handler) -> RazorpayClient:
    transport = httpx.MockTransport(handler)
    return RazorpayClient(key_id="rzp_test_x", key_secret="secret", transport=transport)


def test_cancellation_mandate_chain_binds_to_the_real_payment():
    payment = _payment()
    cancellation = build_cancellation_mandate(
        payment, principal_kid=HUMAN_KP.kid, reason="user_requested", now=_now()
    )
    assert cancellation.payment_hash == mandate_hash(payment)


def test_forged_cancellation_is_refused_before_any_network_call():
    """The hard invariant, mirroring execute.py's ExecutionRefused: a
    cancellation whose payment_hash does not genuinely match the payment
    it claims to reverse must never reach Razorpay at all."""
    payment = _payment()
    forged = build_cancellation_mandate(
        payment, principal_kid=HUMAN_KP.kid, reason="user_requested", now=_now()
    ).model_copy(update={"payment_hash": "not-the-real-hash"})

    def unreachable(request):
        raise AssertionError("must never call Razorpay for a forged cancellation")

    client = _mock_client(unreachable)
    with pytest.raises(CancellationRefused, match="does not match hash"):
        cancel_payment(forged, payment, client=client)


def test_payment_with_no_order_id_is_refused():
    payment = _payment(order_id=None)
    cancellation = build_cancellation_mandate(
        payment, principal_kid=HUMAN_KP.kid, reason="user_requested", now=_now()
    )

    def unreachable(request):
        raise AssertionError("must never call Razorpay when nothing was ever charged")

    with pytest.raises(CancellationRefused, match="no razorpay_order_id"):
        cancel_payment(cancellation, payment, client=_mock_client(unreachable))


def test_order_with_no_captured_payment_is_refused():
    payment = _payment()
    cancellation = build_cancellation_mandate(
        payment, principal_kid=HUMAN_KP.kid, reason="user_requested", now=_now()
    )

    def handler(request):
        if request.url.path.endswith("/payments"):
            return httpx.Response(200, json={"items": [{"id": "pay_x", "status": "created"}]})
        return httpx.Response(200, json={"id": "order_abc123", "status": "attempted"})

    with pytest.raises(CancellationRefused, match="no captured payment"):
        cancel_payment(cancellation, payment, client=_mock_client(handler))


def test_successful_cancellation_refunds_the_captured_payment():
    payment = _payment()
    cancellation = build_cancellation_mandate(
        payment, principal_kid=HUMAN_KP.kid, reason="user_requested", now=_now()
    )
    refund_calls = []

    def handler(request):
        if request.url.path.endswith("/payments"):
            return httpx.Response(200, json={"items": [{"id": "pay_captured_1", "status": "captured"}]})
        if request.url.path.endswith("/refund"):
            refund_calls.append(request)
            return httpx.Response(200, json={"id": "rfnd_1", "payment_id": "pay_captured_1", "status": "processed"})
        return httpx.Response(200, json={"id": "order_abc123", "status": "paid"})

    result = cancel_payment(cancellation, payment, client=_mock_client(handler))

    assert result["id"] == "rfnd_1"
    assert len(refund_calls) == 1
    assert refund_calls[0].url.path == "/v1/payments/pay_captured_1/refund"


def test_reconciliation_mismatch_is_a_valid_trigger():
    """The second cancellation trigger: recon flagging a mismatch
    between what was signed and what was charged must be actionable,
    not just loggable. Detection without remedy is only half a
    control."""
    payment = _payment()
    cancellation = build_cancellation_mandate(
        payment, principal_kid=HUMAN_KP.kid, reason="reconciliation_mismatch", now=_now()
    )
    assert cancellation.reason == "reconciliation_mismatch"

    def handler(request):
        if request.url.path.endswith("/payments"):
            return httpx.Response(200, json={"items": [{"id": "pay_1", "status": "captured"}]})
        if request.url.path.endswith("/refund"):
            return httpx.Response(200, json={"id": "rfnd_2", "status": "processed"})
        return httpx.Response(200, json={"id": "order_abc123", "status": "paid"})

    result = cancel_payment(cancellation, payment, client=_mock_client(handler))
    assert result["status"] == "processed"


def test_sign_and_cancel_produces_a_verifiable_signature():
    from mandates.keys import KeyDirectory
    from mandates.verify import verify_envelope

    payment = _payment()

    def handler(request):
        if request.url.path.endswith("/payments"):
            return httpx.Response(200, json={"items": [{"id": "pay_1", "status": "captured"}]})
        if request.url.path.endswith("/refund"):
            return httpx.Response(200, json={"id": "rfnd_3", "status": "processed"})
        return httpx.Response(200, json={"id": "order_abc123", "status": "paid"})

    envelope, refund = sign_and_cancel(
        payment, principal_kid=HUMAN_KP.kid, reason="user_requested", now=_now(),
        human_keypair=HUMAN_KP, client=_mock_client(handler),
    )

    assert refund["status"] == "processed"
    directory = KeyDirectory()
    directory.register_keypair(HUMAN_KP)
    verified = verify_envelope(envelope, directory, required_kid=HUMAN_KP.kid)
    assert verified.valid, verified.reason
