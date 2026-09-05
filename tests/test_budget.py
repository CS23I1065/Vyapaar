from __future__ import annotations

import pytest

from eval.budget import BudgetExceeded, BudgetGuard


def test_check_passes_under_ceiling():
    g = BudgetGuard(ceiling_usd=1.00)
    g.check(0.50)  # must not raise
    g.record(0.50)
    assert g.spent_usd == 0.50
    assert not g.aborted


def test_check_aborts_before_breaching_ceiling():
    """The whole point: check() must raise BEFORE the call happens, not
    after record() would have pushed spend over the cap."""
    g = BudgetGuard(ceiling_usd=1.00)
    g.record(0.90)
    with pytest.raises(BudgetExceeded):
        g.check(0.20)  # 0.90 + 0.20 > 1.00
    assert g.aborted
    assert g.abort_reason is not None
    # spend was NOT incremented by the aborted call -- check() never calls record()
    assert g.spent_usd == 0.90


def test_exact_boundary_is_not_an_abort():
    g = BudgetGuard(ceiling_usd=1.00)
    g.record(0.80)
    g.check(0.20)  # exactly at ceiling -- must be allowed, not rejected
    assert not g.aborted


def test_one_cent_over_boundary_aborts():
    g = BudgetGuard(ceiling_usd=1.00)
    g.record(0.80)
    with pytest.raises(BudgetExceeded):
        g.check(0.2001)


def test_from_env_reads_eval_budget_usd(monkeypatch):
    monkeypatch.setenv("EVAL_BUDGET_USD", "2.50")
    g = BudgetGuard.from_env()
    assert g.ceiling_usd == 2.50


def test_from_env_default_is_five(monkeypatch):
    monkeypatch.delenv("EVAL_BUDGET_USD", raising=False)
    g = BudgetGuard.from_env()
    assert g.ceiling_usd == 5.00


def test_flush_log_writes_and_clears_events(tmp_path):
    g = BudgetGuard(ceiling_usd=1.00)
    g.record(0.10)
    log_path = tmp_path / "budget.jsonl"
    g.flush_log(log_path)
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    g.flush_log(log_path)  # second flush with no new events appends nothing
    lines_after = log_path.read_text().strip().splitlines()
    assert len(lines_after) == 1


def test_provider_spend_guard_protocol_shape():
    """GeminiProvider expects an object matching llm.provider.SpendGuard
    structurally (check/record). Confirm BudgetGuard satisfies that shape
    without importing eval/ into llm/ (see provider.py's Protocol note)."""
    from llm.provider import SpendGuard

    g = BudgetGuard(ceiling_usd=1.00)
    assert isinstance(g, SpendGuard)
