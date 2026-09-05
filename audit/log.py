"""
Append-only, hash-chained audit log. Every record includes a hash of its
own content plus the previous record's hash (prev_hash) -- tampering
with, deleting, or reordering any past record breaks the chain from that
point forward, detectable by audit/verify.py without needing to trust
anything about how the log file itself was stored.

Records every proposed action and policy check: LLM calls, tool calls,
policy Decisions (with every Check, not just the final outcome),
signature verifications, mandates, and spend-guard decisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

GENESIS_HASH = "0" * 64


def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


@dataclass
class AuditLog:
    path: Path
    _prev_hash: str = field(default=GENESIS_HASH, init=False)

    def __post_init__(self) -> None:
        if self.path.exists():
            # Resume the SAME chain rather than resetting to genesis --
            # a fresh process appending to an existing log must continue
            # from the real last hash, not silently start a new chain
            # that verify_log would (correctly) refuse to link up.
            last = None
            with self.path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last = json.loads(line)
            if last is not None:
                self._prev_hash = last["record_hash"]

    def append(self, kind: str, payload: dict, *, now: datetime) -> dict:
        body = {"kind": kind, "at": now.isoformat(), "prev_hash": self._prev_hash, "payload": payload}
        record_hash = hashlib.sha256(_canonical(body)).hexdigest()
        record = {**body, "record_hash": record_hash}

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        self._prev_hash = record_hash
        return record

    # -- convenience wrappers for each record kind the plan calls for ----

    def log_llm_call(
        self, *, call_site: str, model: str, prompt_hash: str, input_tokens: int, output_tokens: int,
        cost_usd: float, now: datetime,
    ) -> dict:
        return self.append(
            "llm_call",
            {"call_site": call_site, "model": model, "prompt_hash": prompt_hash, "input_tokens": input_tokens,
             "output_tokens": output_tokens, "cost_usd": cost_usd},
            now=now,
        )

    def log_tool_call(self, *, tool: str, request: dict, response: dict, now: datetime) -> dict:
        return self.append("tool_call", {"tool": tool, "request": request, "response": response}, now=now)

    def log_decision(self, decision, *, now: datetime) -> dict:
        """`decision` is a buyer.policy_engine.Decision -- not type-hinted
        directly to avoid audit/ importing buyer/ at module load time
        (audit is a cross-cutting utility; buyer/ never needs to import
        audit/, and this keeps that one-directional)."""
        return self.append(
            "decision",
            {
                "outcome": decision.outcome,
                "blocking_code": decision.blocking_code,
                "checks": [
                    {"gate": c.gate, "code": c.code, "passed": c.passed, "detail": c.detail, "evidence": c.evidence}
                    for c in decision.checks
                ],
            },
            now=now,
        )

    def log_signature_verification(
        self, *, subject: str, valid: bool, reason: str | None, kid: str | None, now: datetime
    ) -> dict:
        return self.append("signature_verification", {"subject": subject, "valid": valid, "reason": reason, "kid": kid}, now=now)

    def log_mandate(self, *, mandate_kind: str, mandate_id: str, mandate_hash_value: str, now: datetime) -> dict:
        return self.append("mandate", {"mandate_kind": mandate_kind, "mandate_id": mandate_id, "hash": mandate_hash_value}, now=now)

    def log_spend_guard(self, *, action: str, projected_usd: float, budget_usd: float, reason: str, now: datetime) -> dict:
        return self.append("spend_guard", {"action": action, "projected_usd": projected_usd, "budget_usd": budget_usd, "reason": reason}, now=now)
