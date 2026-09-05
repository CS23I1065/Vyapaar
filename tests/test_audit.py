"""
audit/ tests -- the hash-chained log is only worth building if tampering
actually gets caught, so this suite exercises tamper/delete/reorder
directly against a real log file on disk, not just the happy path.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from audit.log import AuditLog
from audit.verify import verify_log
from audit.view import render_timeline
from buyer.policy_engine import Check, Decision
from rich.console import Console

UTC = timezone.utc


def _now() -> datetime:
    return datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)


def test_append_produces_an_intact_chain(tmp_path):
    log = AuditLog(path=tmp_path / "audit.jsonl")
    log.append("note", {"msg": "first"}, now=_now())
    log.append("note", {"msg": "second"}, now=_now() + timedelta(seconds=1))
    log.append("note", {"msg": "third"}, now=_now() + timedelta(seconds=2))

    result = verify_log(log.path)
    assert result.intact
    assert result.record_count == 3


def test_empty_or_nonexistent_log_is_trivially_intact(tmp_path):
    result = verify_log(tmp_path / "does_not_exist.jsonl")
    assert result.intact
    assert result.record_count == 0


def test_resuming_an_existing_log_continues_the_same_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    first_log = AuditLog(path=path)
    first_log.append("note", {"msg": "before restart"}, now=_now())

    # Simulate a fresh process re-opening the same log file.
    second_log = AuditLog(path=path)
    second_log.append("note", {"msg": "after restart"}, now=_now() + timedelta(seconds=1))

    result = verify_log(path)
    assert result.intact
    assert result.record_count == 2


def test_tampering_a_record_payload_breaks_verification(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path)
    log.append("decision", {"outcome": "APPROVED"}, now=_now())
    log.append("decision", {"outcome": "APPROVED"}, now=_now() + timedelta(seconds=1))

    lines = path.read_text().splitlines()
    tampered = json.loads(lines[0])
    tampered["payload"]["outcome"] = "REJECTED"  # attacker rewrites history
    lines[0] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n")

    result = verify_log(path)
    assert not result.intact
    assert result.broken_at_line == 1
    assert "record_hash mismatch" in result.reason


def test_deleting_a_middle_record_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path)
    log.append("note", {"msg": "1"}, now=_now())
    log.append("note", {"msg": "2"}, now=_now() + timedelta(seconds=1))
    log.append("note", {"msg": "3"}, now=_now() + timedelta(seconds=2))

    lines = path.read_text().splitlines()
    del lines[1]  # remove the middle record -- record 3's prev_hash now points to a hash that no longer precedes it
    path.write_text("\n".join(lines) + "\n")

    result = verify_log(path)
    assert not result.intact
    assert "prev_hash mismatch" in result.reason


def test_reordering_records_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path)
    log.append("note", {"msg": "1"}, now=_now())
    log.append("note", {"msg": "2"}, now=_now() + timedelta(seconds=1))

    lines = path.read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    path.write_text("\n".join(lines) + "\n")

    result = verify_log(path)
    assert not result.intact


def test_log_decision_round_trips_a_real_decision_and_stays_intact(tmp_path):
    decision = Decision(
        outcome="REJECTED",
        checks=(
            Check(gate="G4.3", code="BUDGET_EXCEEDED", passed=False, detail="cart total 50000 exceeds budget 30000", evidence={"total_paise": 50000, "limit_paise": 30000}),
            Check(gate="G0.1", code="MANDATE_EXPIRED", passed=True, detail="not expired", evidence={}),
        ),
        blocking_code="BUDGET_EXCEEDED",
        evaluated_at=_now(),
    )
    log = AuditLog(path=tmp_path / "audit.jsonl")
    record = log.log_decision(decision, now=_now())

    assert record["payload"]["outcome"] == "REJECTED"
    assert len(record["payload"]["checks"]) == 2
    assert verify_log(log.path).intact


def test_log_convenience_methods_all_produce_intact_chain(tmp_path):
    log = AuditLog(path=tmp_path / "audit.jsonl")
    log.log_llm_call(call_site="x", model="gemini-3.1-flash-lite", prompt_hash="abc123", input_tokens=10, output_tokens=5, cost_usd=0.0001, now=_now())
    log.log_tool_call(tool="create_order", request={"amount": 1000}, response={"id": "order_1"}, now=_now())
    log.log_signature_verification(subject="intent", valid=True, reason=None, kid="kid-1", now=_now())
    log.log_mandate(mandate_kind="IntentMandate", mandate_id="intent-1", mandate_hash_value="a" * 64, now=_now())
    log.log_spend_guard(action="continue", projected_usd=1.2, budget_usd=5.0, reason="within budget", now=_now())

    result = verify_log(log.path)
    assert result.intact
    assert result.record_count == 5


def test_render_timeline_runs_without_error(tmp_path):
    log = AuditLog(path=tmp_path / "audit.jsonl")
    decision = Decision(
        outcome="APPROVED",
        checks=(Check(gate="G4.3", code="BUDGET_EXCEEDED", passed=True, detail="within budget", evidence={}),),
        blocking_code=None,
        evaluated_at=_now(),
    )
    log.log_decision(decision, now=_now())
    log.log_mandate(mandate_kind="CartMandate", mandate_id="cart-1", mandate_hash_value="b" * 64, now=_now())
    log.log_signature_verification(subject="cart", valid=True, reason=None, kid="kid-2", now=_now())
    log.log_llm_call(call_site="interpret", model="gemini-3.1-flash-lite", prompt_hash="x", input_tokens=1, output_tokens=1, cost_usd=0.0, now=_now())
    log.log_tool_call(tool="fetch_order", request={}, response={}, now=_now())
    log.log_spend_guard(action="continue", projected_usd=0.1, budget_usd=5.0, reason="ok", now=_now())

    console = Console(record=True, width=120)
    render_timeline(log.path, console=console)  # must not raise
    output = console.export_text()
    assert "APPROVED" in output
    assert "BUDGET_EXCEEDED" in output


def test_render_timeline_on_missing_log_does_not_raise(tmp_path):
    console = Console(record=True, width=120)
    render_timeline(tmp_path / "nope.jsonl", console=console)
    assert "No audit log found" in console.export_text()
