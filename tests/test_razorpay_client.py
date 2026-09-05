"""
Tests for razorpay_client/. The cassette tests are pure/offline. The
live tests hit the real Razorpay test-mode API (skipped automatically if
no key is configured) -- create_order/fetch_order/fetch_order_payments
and fetch_settlement_recon are all confirmed working live. S2S direct
payment creation is NOT (404 on an account without that enablement, not
a code bug), so there is no live "create a real payment then capture it"
test here.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import httpx
import pytest

from razorpay_client.cassettes import CassetteMiss, ReplayTransport
from razorpay_client.client import RazorpayClient, RazorpayError, client_from_env

requires_live_razorpay = pytest.mark.skipif(
    not os.environ.get("RAZORPAY_KEY_ID"), reason="no RAZORPAY_KEY_ID in environment"
)


# ---------------------------------------------------------------------
# Cassette replay -- offline, deterministic
# ---------------------------------------------------------------------


def test_replay_transport_returns_recorded_response(tmp_path):
    """The cassette key must be computed from EXACTLY the bytes httpx's
    own `json=` parameter would produce -- httpx uses compact separators
    internally, which differ from plain `json.dumps()`'s default
    (spaced) separators. A hand-built key using the wrong separators
    silently produces a different hash and a spurious CassetteMiss."""
    cassette = tmp_path / "orders.json"
    order_body = {"id": "order_TEST123", "amount": 39500, "currency": "INR", "status": "created"}
    body = {"amount": 39500, "currency": "INR", "receipt": "r1", "notes": {}}

    from razorpay_client.cassettes import _key

    with httpx.Client(base_url="https://api.razorpay.com/v1") as probe:
        prepared = probe.build_request("POST", "/orders", json=body)

    cassette.write_text(json.dumps({_key(prepared): {"status_code": 200, "body": json.dumps(order_body)}}))

    client = RazorpayClient(key_id="rzp_test_x", key_secret="secret", transport=ReplayTransport(cassette))
    result = client.create_order(amount_paise=39500, receipt="r1", notes={})
    assert result == order_body


def test_replay_transport_raises_cassette_miss_on_unrecorded_request(tmp_path):
    cassette = tmp_path / "empty.json"
    cassette.write_text("{}")
    client = RazorpayClient(key_id="rzp_test_x", key_secret="secret", transport=ReplayTransport(cassette))
    with pytest.raises(CassetteMiss):
        client.create_order(amount_paise=100, receipt="r2", notes={})


def test_replay_transport_missing_cassette_file_raises():
    with pytest.raises(CassetteMiss):
        ReplayTransport(cassette_path=__import__("pathlib").Path("/nonexistent/cassette.json"))


# ---------------------------------------------------------------------
# Live, against real Razorpay test-mode API
# ---------------------------------------------------------------------


@requires_live_razorpay
def test_live_create_order_round_trips_mandate_hashes_in_notes():
    client = client_from_env()
    receipt = f"test-{datetime.now(timezone.utc).timestamp()}"
    notes = {
        "intent_hash": "a" * 64,
        "cart_hash": "b" * 64,
        "payment_mandate_id": "pm-test-1",
    }
    order = client.create_order(amount_paise=39_500, receipt=receipt, notes=notes, currency="INR")
    assert order["status"] == "created"
    assert order["amount"] == 39_500

    fetched = client.fetch_order(order["id"])
    # This is the load-bearing assertion for the whole mandate-binding
    # claim: the hashes must survive the round trip through Razorpay's
    # own ledger intact, byte for byte.
    assert fetched["notes"]["intent_hash"] == notes["intent_hash"]
    assert fetched["notes"]["cart_hash"] == notes["cart_hash"]
    assert fetched["notes"]["payment_mandate_id"] == notes["payment_mandate_id"]

    payments = client.fetch_order_payments(order["id"])
    assert payments["entity"] == "collection"  # no payment exists yet -- order creation alone doesn't authorize one


@requires_live_razorpay
def test_live_capture_on_nonexistent_payment_raises_razorpay_error():
    client = client_from_env()
    with pytest.raises(RazorpayError) as exc_info:
        client.capture_payment("pay_doesnotexist000000", amount_paise=100)
    assert exc_info.value.status_code >= 400


@requires_live_razorpay
def test_live_fetch_settlement_recon_does_not_error():
    client = client_from_env()
    now = datetime.now(timezone.utc)
    result = client.fetch_settlement_recon(year=now.year, month=now.month, day=now.day)
    assert result["entity"] == "collection"  # Razorpay test mode generates no settlements
