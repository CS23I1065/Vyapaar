"""
Tests for demo/ -- runs every beat offline and checks it completes
without error and hits the story points that matter. Not a re-test of
the underlying modules (those have their own exhaustive suites); this is
a smoke test that the DEMO SCRIPT ITSELF wires them together correctly,
end to end, with zero network and zero LLM cost.
"""

from __future__ import annotations

from pathlib import Path

from demo.fixtures import demo_catalog_shirts, fake_razorpay_client, now
from demo.run_demo import (
    beat_1_onboarding,
    beat_2_revenue,
    beat_3_comparison_shopping,
    beat_4_blocked_purchase_explained,
    beat_5_injection_blocked,
)


def test_fixtures_are_internally_consistent():
    """The demo catalog must actually let the upsell engine find both an
    accessory (different category, cheaper) and a premium substitute
    (same category, pricier) -- otherwise Beat 2 would silently fall
    through to the weakest pitch (a quantity bump) without anyone
    noticing."""
    from merchant_agent.upsell import find_accessory, find_premium_substitute

    t = now()
    items = demo_catalog_shirts(t)
    primary = next(i for i in items if i.sku == "SHIRT-PLAIN")
    assert find_accessory(primary, items) is not None
    assert find_premium_substitute(primary, items) is not None


def test_fake_razorpay_supports_the_full_order_payment_refund_cycle():
    client = fake_razorpay_client()
    order = client.create_order(amount_paise=50_000, receipt="test-receipt", notes={"k": "v"})
    assert order["status"] == "created"

    payments = client.fetch_order_payments(order["id"])
    assert payments["items"][0]["status"] == "captured"
    assert payments["items"][0]["amount"] == 50_000

    refund = client.create_refund(payments["items"][0]["id"])
    assert refund["status"] == "processed"


def test_beat_1_onboarding_runs_without_error(capsys):
    beat_1_onboarding()
    out = capsys.readouterr().out
    assert "signature_valid=True" in out


def test_beat_2_revenue_approves_and_returns_a_populated_audit_log(capsys):
    log = beat_2_revenue(live_razorpay=False, live_llm=False)
    out = capsys.readouterr().out
    assert "Decision: APPROVED" in out
    assert "Order created" in out
    assert log.path.exists()


def test_beat_3_comparison_shopping_picks_the_preferred_brand(capsys):
    beat_3_comparison_shopping()
    out = capsys.readouterr().out
    assert "Checked 3 shop(s)" in out
    assert "shop-b" in out and "WON" in out


def test_beat_4_explains_a_rejection_in_plain_language(capsys):
    beat_4_blocked_purchase_explained()
    out = capsys.readouterr().out
    assert "PER_ITEM_CAP_EXCEEDED" in out  # the gate code, kept
    assert "I didn't buy this" in out  # the plain-language translation


def test_beat_5_shows_byte_identical_decisions_and_an_intact_audit_chain(capsys):
    import tempfile

    from audit.log import AuditLog

    log = AuditLog(path=Path(tempfile.mkdtemp(prefix="hope-demo-test-")) / "log.jsonl")
    beat_5_injection_blocked(log)
    out = capsys.readouterr().out
    assert "Byte-identical decisions: True" in out
    assert "Hash chain intact: True" in out


def test_full_demo_runs_offline_end_to_end():
    """The whole thing, via subprocess, exactly as a user would invoke
    it -- `python -m demo.run_demo` with no flags, so no --live-* branch
    is exercised and nothing here touches the network."""
    import subprocess
    import sys

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "demo.run_demo"], cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Five beats complete" in result.stdout
    assert "Byte-identical decisions: True" in result.stdout

