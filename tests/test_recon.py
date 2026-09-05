"""
Tests for razorpay_client/recon.py and razorpay_client/webhooks.py --
closing the loop from signed authorization to money actually moved.
Covers capture-time REST reconciliation and HMAC-verified webhook
reconciliation, both against realistic Razorpay response shapes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from mandates.hashing import mandate_hash
from mandates.keys import generate_keypair
from mandates.schemas import (
    Budget,
    CartLine,
    CartMandate,
    HardConstraints,
    IntentMandate,
    PaymentMandate,
    Provenance,
    SoftPreferences,
)
from mandates.sign import sign_mandate
from razorpay_client.client import RazorpayClient
from razorpay_client.recon import reconcile_capture, verify_notes_binding
from razorpay_client.webhooks import (
    WebhookDeduper,
    WebhookPayloadError,
    WebhookReplay,
    WebhookVerificationError,
    parse_payment_captured_event,
    process_payment_captured_webhook,
    verify_signature,
)

UTC = timezone.utc
HUMAN_KP = generate_keypair()
MERCHANT_KP = generate_keypair()


def _now() -> datetime:
    return datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC)


def _intent() -> IntentMandate:
    return IntentMandate(
        mandate_id="intent-1", issued_at=_now(), expires_at=_now() + timedelta(hours=1),
        principal_kid=HUMAN_KP.kid, agent_kid=HUMAN_KP.kid, request_text="buy rice",
        hard=HardConstraints(category="grocery"), soft=SoftPreferences(substitution_tolerance_pct=10.0),
        budget=Budget(total_paise=50_000, per_item_paise=50_000, max_quantity=2, escalate_above_paise=50_000),
        allowed_merchants=["merchant-1"],
    )


def _cart(intent: IntentMandate, total_paise: int = 30_000) -> CartMandate:
    prov = Provenance(source_url="https://x", extraction_method="merchant_signed", signature_verified=True, extracted_at=_now())
    line = CartLine(sku="SKU-1", title="Rice", category="grocery", unit_price_paise=total_paise, quantity=1,
                    field_provenance={"unit_price_paise": prov})
    return CartMandate(
        mandate_id="cart-1", issued_at=_now(), expires_at=_now() + timedelta(hours=1),
        merchant_id="merchant-1", merchant_kid=MERCHANT_KP.kid, intent_hash=mandate_hash(intent),
        lines=[line], subtotal_paise=total_paise, tax_paise=0, shipping_paise=0, total_paise=total_paise,
    )


def _payment(intent: IntentMandate, cart: CartMandate, order_id="order_abc") -> PaymentMandate:
    return PaymentMandate(
        mandate_id="payment-1", created_at=_now(), intent_hash=mandate_hash(intent), cart_hash=mandate_hash(cart),
        amount_paise=cart.total_paise, rail="razorpay_test", razorpay_order_id=order_id,
    )


def _mock_client(handler) -> RazorpayClient:
    return RazorpayClient(key_id="rzp_test_x", key_secret="secret", transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------
# Tier 1 -- capture-time REST recon
# ---------------------------------------------------------------------


def test_matched_capture_produces_a_clean_dispute_artifact():
    intent = _intent()
    cart = _cart(intent)
    payment = _payment(intent, cart)
    cart_envelope = sign_mandate(cart, MERCHANT_KP)
    notes = {"intent_hash": payment.intent_hash, "cart_hash": payment.cart_hash, "payment_mandate_id": payment.mandate_id}

    def handler(request):
        if request.url.path.endswith("/payments"):
            return httpx.Response(200, json={"items": [{"id": "pay_1", "amount": 30_000, "status": "captured"}]})
        return httpx.Response(200, json={"id": "order_abc", "status": "paid", "notes": notes})

    artifact = reconcile_capture(cart, cart_envelope, payment, client=_mock_client(handler), now=_now())

    assert artifact.delta.matched is True
    assert artifact.delta.delta_paise == 0
    assert artifact.delta.charged_amount_paise == 30_000


def test_overcharge_produces_a_non_zero_dispute_delta():
    """The plan's never-cut item: a merchant charging more than its own
    signed cart said must be independently detectable at capture time,
    not just at policy-evaluation time."""
    intent = _intent()
    cart = _cart(intent, total_paise=30_000)
    payment = _payment(intent, cart)
    cart_envelope = sign_mandate(cart, MERCHANT_KP)

    def handler(request):
        if request.url.path.endswith("/payments"):
            return httpx.Response(200, json={"items": [{"id": "pay_1", "amount": 48_000, "status": "captured"}]})
        return httpx.Response(200, json={"id": "order_abc", "status": "paid", "notes": {}})

    artifact = reconcile_capture(cart, cart_envelope, payment, client=_mock_client(handler), now=_now())

    assert artifact.delta.matched is False
    assert artifact.delta.is_overcharge
    assert artifact.delta.delta_paise == 18_000
    assert "OVERCHARGED" in artifact.delta.detail


def test_undercharge_is_also_flagged_not_just_overcharge():
    intent = _intent()
    cart = _cart(intent, total_paise=30_000)
    payment = _payment(intent, cart)
    cart_envelope = sign_mandate(cart, MERCHANT_KP)

    def handler(request):
        if request.url.path.endswith("/payments"):
            return httpx.Response(200, json={"items": [{"id": "pay_1", "amount": 20_000, "status": "captured"}]})
        return httpx.Response(200, json={"id": "order_abc", "status": "paid", "notes": {}})

    artifact = reconcile_capture(cart, cart_envelope, payment, client=_mock_client(handler), now=_now())
    assert artifact.delta.is_undercharge
    assert artifact.delta.delta_paise == -10_000


def test_no_capture_yet_is_reported_not_treated_as_matched():
    intent = _intent()
    cart = _cart(intent)
    payment = _payment(intent, cart)
    cart_envelope = sign_mandate(cart, MERCHANT_KP)

    def handler(request):
        if request.url.path.endswith("/payments"):
            return httpx.Response(200, json={"items": []})
        return httpx.Response(200, json={"id": "order_abc", "status": "created", "notes": {}})

    artifact = reconcile_capture(cart, cart_envelope, payment, client=_mock_client(handler), now=_now())
    assert artifact.delta.matched is False
    assert artifact.delta.charged_amount_paise is None


def test_notes_binding_catches_a_hash_that_does_not_match():
    """execute.py binds intent_hash/cart_hash/payment_mandate_id into the
    order's notes at creation time. If Razorpay's own record of those
    notes disagrees with what the mandates actually hash to, that is a
    real integrity problem regardless of the amount matching."""
    intent = _intent()
    cart = _cart(intent)
    payment = _payment(intent, cart)
    order = {"notes": {"intent_hash": "tampered", "cart_hash": mandate_hash(cart), "payment_mandate_id": payment.mandate_id}}

    violations = verify_notes_binding(order, mandate_hash(intent), mandate_hash(cart), payment.mandate_id)
    assert any("intent_hash" in v for v in violations)


def test_notes_binding_passes_when_everything_matches():
    intent = _intent()
    cart = _cart(intent)
    payment = _payment(intent, cart)
    order = {"notes": {"intent_hash": mandate_hash(intent), "cart_hash": mandate_hash(cart), "payment_mandate_id": payment.mandate_id}}
    assert verify_notes_binding(order, mandate_hash(intent), mandate_hash(cart), payment.mandate_id) == []


# ---------------------------------------------------------------------
# Tier 2 -- HMAC-verified webhook recon (the INDEPENDENT channel)
# ---------------------------------------------------------------------


def _signed_body(payload: dict, secret: str) -> tuple[bytes, str]:
    raw = json.dumps(payload).encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return raw, sig


def _captured_payload(*, amount_paise, order_id, notes, status="captured"):
    return {
        "entity": "event", "event": "payment.captured", "contains": ["payment"],
        "payload": {"payment": {"entity": {
            "id": "pay_webhook_1", "amount": amount_paise, "currency": "INR",
            "status": status, "order_id": order_id, "notes": notes,
        }}},
        "created_at": 1234567890,
    }


def test_verify_signature_over_raw_bytes_matches_hmac_sha256():
    """The signature is over the RAW body -- re-serializing (even a
    key-order-preserving json.dumps) can change the bytes actually
    signed, so this must take bytes, never a dict."""
    secret = "whsec_test_123"
    raw = b'{"event":"payment.captured"}'
    sig = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    assert verify_signature(raw, sig, secret) is True
    assert verify_signature(raw, sig, "wrong-secret") is False
    assert verify_signature(raw + b" ", sig, secret) is False  # any byte difference fails


def test_full_webhook_path_matches_when_everything_agrees():
    intent = _intent()
    cart = _cart(intent, total_paise=30_000)
    payment = _payment(intent, cart)
    notes = {"intent_hash": mandate_hash(intent), "cart_hash": mandate_hash(cart), "payment_mandate_id": payment.mandate_id}
    secret = "whsec_test_123"
    raw, sig = _signed_body(_captured_payload(amount_paise=30_000, order_id="order_abc", notes=notes), secret)

    result = process_payment_captured_webhook(
        raw, signature=sig, event_id="evt_1", secret=secret, deduper=WebhookDeduper(),
        intent=intent, cart=cart, payment_mandate=payment,
    )
    assert result.matched is True
    assert result.violations == ()


def test_bad_signature_is_refused_before_any_reconciliation():
    """An unverified webhook must not be acted on at all -- not
    partially processed, not logged as probable. It is indistinguishable
    from one forged by an attacker with no relationship to Razorpay."""
    intent = _intent()
    cart = _cart(intent)
    payment = _payment(intent, cart)
    notes = {"intent_hash": mandate_hash(intent), "cart_hash": mandate_hash(cart), "payment_mandate_id": payment.mandate_id}
    raw, _ = _signed_body(_captured_payload(amount_paise=30_000, order_id="order_abc", notes=notes), "whsec_real")

    with pytest.raises(WebhookVerificationError):
        process_payment_captured_webhook(
            raw, signature="deadbeef" * 8, event_id="evt_2", secret="whsec_real",
            deduper=WebhookDeduper(), intent=intent, cart=cart, payment_mandate=payment,
        )


def test_replayed_event_id_is_refused_the_second_time():
    """Webhooks can be delivered more than once. Dedup is by event ID,
    not payload content."""
    intent = _intent()
    cart = _cart(intent)
    payment = _payment(intent, cart)
    notes = {"intent_hash": mandate_hash(intent), "cart_hash": mandate_hash(cart), "payment_mandate_id": payment.mandate_id}
    secret = "whsec_test_123"
    raw, sig = _signed_body(_captured_payload(amount_paise=30_000, order_id="order_abc", notes=notes), secret)
    deduper = WebhookDeduper()

    first = process_payment_captured_webhook(
        raw, signature=sig, event_id="evt_dup", secret=secret, deduper=deduper,
        intent=intent, cart=cart, payment_mandate=payment,
    )
    assert first.matched is True
    with pytest.raises(WebhookReplay):
        process_payment_captured_webhook(
            raw, signature=sig, event_id="evt_dup", secret=secret, deduper=deduper,
            intent=intent, cart=cart, payment_mandate=payment,
        )


def test_webhook_amount_mismatch_is_caught_by_the_independent_channel():
    """The scenario Tier 2 exists for: Razorpay's own out-of-band
    notification disagrees with what was signed, independent of
    whatever the buyer agent's own REST fetch (Tier 1) would have shown."""
    intent = _intent()
    cart = _cart(intent, total_paise=30_000)
    payment = _payment(intent, cart)
    notes = {"intent_hash": mandate_hash(intent), "cart_hash": mandate_hash(cart), "payment_mandate_id": payment.mandate_id}
    secret = "whsec_test_123"
    raw, sig = _signed_body(_captured_payload(amount_paise=48_000, order_id="order_abc", notes=notes), secret)

    result = process_payment_captured_webhook(
        raw, signature=sig, event_id="evt_3", secret=secret, deduper=WebhookDeduper(),
        intent=intent, cart=cart, payment_mandate=payment,
    )
    assert result.matched is False
    assert any("amount" in v for v in result.violations)


def test_webhook_note_hash_mismatch_is_caught():
    intent = _intent()
    cart = _cart(intent)
    payment = _payment(intent, cart)
    notes = {"intent_hash": "tampered", "cart_hash": mandate_hash(cart), "payment_mandate_id": payment.mandate_id}
    secret = "whsec_test_123"
    raw, sig = _signed_body(_captured_payload(amount_paise=30_000, order_id="order_abc", notes=notes), secret)

    result = process_payment_captured_webhook(
        raw, signature=sig, event_id="evt_4", secret=secret, deduper=WebhookDeduper(),
        intent=intent, cart=cart, payment_mandate=payment,
    )
    assert result.matched is False
    assert any("intent_hash" in v for v in result.violations)


def test_wrong_order_id_is_caught():
    intent = _intent()
    cart = _cart(intent)
    payment = _payment(intent, cart, order_id="order_real")
    notes = {"intent_hash": mandate_hash(intent), "cart_hash": mandate_hash(cart), "payment_mandate_id": payment.mandate_id}
    secret = "whsec_test_123"
    raw, sig = _signed_body(_captured_payload(amount_paise=30_000, order_id="order_different", notes=notes), secret)

    result = process_payment_captured_webhook(
        raw, signature=sig, event_id="evt_5", secret=secret, deduper=WebhookDeduper(),
        intent=intent, cart=cart, payment_mandate=payment,
    )
    assert result.matched is False
    assert any("order_id" in v for v in result.violations)


def test_test_and_live_secrets_do_not_cross_verify():
    """A test-mode secret must never successfully verify a live-mode
    event or vice versa."""
    raw, sig = _signed_body({"event": "payment.captured"}, "whsec_TEST_secret")
    assert verify_signature(raw, sig, "whsec_TEST_secret") is True
    assert verify_signature(raw, sig, "whsec_LIVE_secret") is False


def test_malformed_payload_after_valid_signature_raises_payload_error():
    """The signature can verify while the payload still isn't a
    payment.captured event shape -- a different event type, say. That is
    a shape failure, not a trust failure, and gets a different exception."""
    secret = "whsec_test_123"
    raw, sig = _signed_body({"event": "refund.processed", "payload": {}}, secret)
    with pytest.raises(WebhookPayloadError):
        parse_payment_captured_event(raw, "evt_6")
    assert verify_signature(raw, sig, secret) is True  # the signature itself was fine
