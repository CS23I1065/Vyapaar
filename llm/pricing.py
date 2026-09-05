"""
Gemini pricing and the model ban list.

Everything here was verified live against a real API key, not copied
from a training cutoff -- for a fast-moving API that's exactly the kind
of thing that goes stale silently. Re-verify before trusting either
table if it's been a while.
"""

from __future__ import annotations

from typing import NamedTuple


class ModelPricing(NamedTuple):
    input_per_million_usd: float
    output_per_million_usd: float


# Paid-tier per-1M-token rates. Source: https://ai.google.dev/gemini-api/docs/pricing
# checked 2026-08-24. Thinking tokens bill as OUTPUT tokens, which is why the
# provider forces thinking_budget=0 by default (see provider.py) — a "Say OK"
# smoke test burned 276 thinking tokens against 8 real output tokens before
# that was disabled, a ~30x inflation on a model billed at $9/M output.
PRICING: dict[str, ModelPricing] = {
    "gemini-3.5-flash": ModelPricing(1.50, 9.00),
    "gemini-3.6-flash": ModelPricing(0.75, 3.75),  # introductory rate, rises 2027-01-01
    "gemini-2.5-flash": ModelPricing(0.30, 2.50),
    "gemini-3.1-flash-lite": ModelPricing(0.25, 1.50),
}

# Verified live 2026-08-24: both of these accept the connection and then never
# respond (HTTP 000 after 60s+), even with thinking disabled. Not a quota
# error (those return 429 immediately) and not an auth error (ListModels on
# the same key returns instantly). GeminiProvider refuses to call these
# regardless of what LLM_MODEL_EVAL / LLM_MODEL_DEMO say, so a stale .env
# can't silently reintroduce the hang. Re-verify with a bounded `curl
# --max-time 60` before removing either from this set.
BANNED_MODELS: frozenset[str] = frozenset({"gemini-3.7-flash", "gemini-flash-latest"})


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    output_tokens should already include any thinking tokens — Gemini bills
    them as output, and callers (provider.py) are expected to have summed
    them before calling this.
    """
    if model not in PRICING:
        raise ValueError(
            f"No pricing entry for {model!r}. Add it to llm/pricing.py.PRICING "
            f"before using this model — silently assuming a price is how a "
            f"budget cap gets blown."
        )
    p = PRICING[model]
    return (input_tokens / 1_000_000) * p.input_per_million_usd + (
        output_tokens / 1_000_000
    ) * p.output_per_million_usd
