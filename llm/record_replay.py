"""
Record/replay wrappers around a Provider (see provider.py).

Why this exists: the ablation matrix compares configs that differ ONLY
in the deterministic policy-engine layer.
The LLM-driven stages (intent interpretation, negotiation) are identical
for a given (scenario, attack, repeat) regardless of which config is being
scored. So each transcript is generated ONCE via RecordingProvider, then
every config replays it via ReplayProvider — the policy engine is pure, so
replays cost zero LLM calls and the defense is isolated as the only
variable instead of being confounded with LLM stochasticity.

A replay is a cache miss is an error, not a silent fallback to a live
call — if it fell back silently, a config that was supposed to cost $0
could quietly re-spend, and the ablation's "identical transcripts across
configs" guarantee would be violated without anyone noticing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

from .provider import Completion, Usage


@runtime_checkable
class Provider(Protocol):
    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        response_schema: dict | None = None,
        thinking_budget: int = 0,
        call_site: str = "unspecified",
    ) -> Completion: ...


class RecordKey(NamedTuple):
    scenario_id: str
    attack_id: str
    repeat_k: int
    call_site: str

    def hash(self) -> str:
        raw = f"{self.scenario_id}|{self.attack_id}|{self.repeat_k}|{self.call_site}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class ReplayMiss(RuntimeError):
    """Raised when a ReplayProvider is asked for a key with no recorded
    transcript. This is a hard error on purpose — see module docstring."""


class RecordingProvider:
    """Wraps a real Provider. If record_key is given to complete(), the
    full request+response is persisted to transcripts_dir before returning.
    If record_key is omitted, behaves as a transparent passthrough (for
    non-eval callers, e.g. the demo, that don't need replay)."""

    def __init__(self, inner: Provider, transcripts_dir: Path) -> None:
        self._inner = inner
        self._dir = transcripts_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def complete(self, *, record_key: RecordKey | None = None, **kwargs) -> Completion:
        result = self._inner.complete(**kwargs)
        if record_key is not None:
            path = self._dir / f"{record_key.hash()}.json"
            payload = {
                "key": record_key._asdict(),
                "request": {k: v for k, v in kwargs.items() if k != "response_schema"},
                "response": {
                    "text": result.text,
                    "model": result.model,
                    "usage": asdict(result.usage),
                },
            }
            path.write_text(json.dumps(payload, indent=2, default=str))
        return result


class ReplayProvider:
    """Never touches the network. Reads a previously recorded transcript by
    (scenario_id, attack_id, repeat_k, call_site) and reconstructs a
    Completion with cost_usd=0.0 — replays are free by construction, which
    is the entire point of the replay-based ablation design."""

    def __init__(self, transcripts_dir: Path) -> None:
        self._dir = transcripts_dir

    def complete(self, *, record_key: RecordKey, **_kwargs) -> Completion:
        path = self._dir / f"{record_key.hash()}.json"
        if not path.exists():
            raise ReplayMiss(
                f"No recorded transcript for {record_key!r} at {path}. "
                f"Run the recording pass first (eval.run --record) — a "
                f"replay must never silently fall back to a live call."
            )
        payload = json.loads(path.read_text())
        resp = payload["response"]
        usage_dict = dict(resp["usage"])
        usage_dict["cost_usd"] = 0.0  # replays are free, on purpose
        return Completion(
            text=resp["text"],
            usage=Usage(**usage_dict),
            model=resp["model"],
            raw=None,
        )
