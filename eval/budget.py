"""
Hard spend ceiling for the eval matrix.

This exists because of a deliberate design choice: the eval harness is
itself a bounded, gated, audited action with a stopping rule — the same
property the whole submission argues merchants and buyers need. A run
that could silently blow past its budget would undercut that claim, so
the guard is enforced in three independent places:

  1. A $5 cap set in Google Cloud billing (out of band, not this code).
  2. This module — aborts a run BEFORE the call that would breach the
     cap, not after (check() runs pre-flight in provider.py, record()
     runs post-flight).
  3. A running cost meter, printed by whatever calls summary().

check() and record() together implement llm.provider.SpendGuard
structurally (see that module — matched by shape, not inheritance).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


class BudgetExceeded(RuntimeError):
    """Raised by check() when a projected call would breach the ceiling.
    The caller (llm.provider.GeminiProvider) does not catch this — it is
    meant to propagate and abort the whole run."""


@dataclass
class BudgetGuard:
    ceiling_usd: float
    spent_usd: float = 0.0
    call_count: int = 0
    checks: int = 0
    aborted: bool = False
    abort_reason: str | None = None
    _events: list[dict] = field(default_factory=list, repr=False)

    @classmethod
    def from_env(cls, default: float = 5.00) -> "BudgetGuard":
        raw = os.environ.get("EVAL_BUDGET_USD")
        ceiling = float(raw) if raw else default
        return cls(ceiling_usd=ceiling)

    def check(self, projected_cost_usd: float) -> None:
        self.checks += 1
        if self.spent_usd + projected_cost_usd > self.ceiling_usd:
            self.aborted = True
            self.abort_reason = (
                f"projected spend ${self.spent_usd + projected_cost_usd:.4f} "
                f"would exceed ceiling ${self.ceiling_usd:.4f} "
                f"(already spent ${self.spent_usd:.4f} over {self.call_count} calls)"
            )
            self._events.append(
                {
                    "event": "BUDGET_ABORT",
                    "ts": time.time(),
                    "reason": self.abort_reason,
                    "spent_usd": self.spent_usd,
                    "projected_add_usd": projected_cost_usd,
                    "ceiling_usd": self.ceiling_usd,
                }
            )
            raise BudgetExceeded(self.abort_reason)

    def record(self, actual_cost_usd: float) -> None:
        self.spent_usd += actual_cost_usd
        self.call_count += 1
        self._events.append(
            {
                "event": "SPEND",
                "ts": time.time(),
                "amount_usd": actual_cost_usd,
                "running_total_usd": self.spent_usd,
            }
        )

    def summary(self) -> dict:
        return {
            "ceiling_usd": self.ceiling_usd,
            "spent_usd": round(self.spent_usd, 4),
            "remaining_usd": round(self.ceiling_usd - self.spent_usd, 4),
            "call_count": self.call_count,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
        }

    def print_summary(self) -> None:
        s = self.summary()
        status = "ABORTED" if s["aborted"] else "OK"
        print(
            f"[budget] {status}  spent=${s['spent_usd']:.4f} / "
            f"${s['ceiling_usd']:.2f} cap  ({s['call_count']} calls, "
            f"${s['remaining_usd']:.4f} remaining)"
        )
        if s["aborted"]:
            print(f"[budget] abort reason: {s['abort_reason']}")

    def flush_log(self, path: Path) -> None:
        """Append every check/record/abort event as JSONL -- a
        budget-specific record independent of audit/log.py's own
        spend-guard logging."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for event in self._events:
                f.write(json.dumps(event) + "\n")
        self._events.clear()
