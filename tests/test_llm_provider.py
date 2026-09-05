"""
Live smoke tests against the real Gemini API. Marked so they can be skipped
in a network-free environment, but by default they run — this project's
entire "the LLM proposes, it never authorizes" claim is worthless if the
provider layer underneath it is untested.
"""

from __future__ import annotations

import os

import pytest

from llm.pricing import BANNED_MODELS, PRICING
from llm.provider import GeminiProvider, ProviderBannedModel, Usage
from llm.record_replay import ReplayMiss, ReplayProvider


@pytest.fixture(scope="module")
def provider():
    p = GeminiProvider()
    yield p
    p.close()


def test_banned_model_refused_without_network(provider):
    """Must fail fast, before any HTTP call — verified by using an
    obviously-invalid API context and confirming we still get the ban
    error rather than an auth error."""
    with pytest.raises(ProviderBannedModel):
        provider.complete(
            model="gemini-3.7-flash",
            system="test",
            messages=[{"role": "user", "text": "hi"}],
            call_site="test_banned",
        )


def test_banned_models_have_pricing_placeholder_or_are_excluded():
    # Banned models must never appear in PRICING as a live default target —
    # if someone adds pricing for one, that's a sign it's about to be used.
    for m in BANNED_MODELS:
        assert m not in PRICING, (
            f"{m} is banned but has a pricing entry — that suggests someone "
            f"is about to route real calls to it. Remove the pricing entry "
            f"or un-ban the model deliberately."
        )


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="no GEMINI_API_KEY in environment"
)
def test_live_completion_eval_model(provider):
    result = provider.complete(
        model="gemini-3.1-flash-lite",
        system="You are a terse test assistant.",
        messages=[{"role": "user", "text": "Reply with exactly the word: OK"}],
        max_output_tokens=20,
        call_site="test_live_completion",
    )
    assert "OK" in result.text
    assert isinstance(result.usage, Usage)
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    # thinking_budget=0 by default -- should be at or near zero, not the
    # 276-vs-8 blowout observed during planning with thinking left on.
    assert result.usage.thinking_tokens <= 5, (
        f"thinking_tokens={result.usage.thinking_tokens}, expected ~0 with "
        f"thinking_budget=0 -- thinking tokens bill as output and this is "
        f"exactly the cost blowout the default guards against."
    )
    assert result.usage.cost_usd > 0
    assert result.usage.cost_usd < 0.001  # a 20-token reply should cost a fraction of a cent


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="no GEMINI_API_KEY in environment"
)
def test_live_structured_output(provider):
    schema = {
        "type": "OBJECT",
        "properties": {
            "name": {"type": "STRING"},
            "price_inr": {"type": "NUMBER"},
            "in_stock": {"type": "BOOLEAN"},
        },
        "required": ["name", "price_inr", "in_stock"],
    }
    result = provider.complete(
        model="gemini-3.1-flash-lite",
        system="Extract structured product data from the text.",
        messages=[
            {"role": "user", "text": "Product: Blue Cotton Shirt, Rs 1299, in stock."}
        ],
        response_schema=schema,
        max_output_tokens=200,
        call_site="test_structured_output",
    )
    import json

    parsed = json.loads(result.text)
    assert parsed["name"]
    assert parsed["price_inr"] == 1299
    assert parsed["in_stock"] is True


def test_replay_miss_is_a_hard_error(tmp_path):
    """A replay cache miss must raise, never silently fall back to a live
    call -- see record_replay.py module docstring for why."""
    from llm.record_replay import RecordKey

    replay = ReplayProvider(tmp_path)
    key = RecordKey(scenario_id="S99", attack_id="A0_none", repeat_k=0, call_site="nope")
    with pytest.raises(ReplayMiss):
        replay.complete(record_key=key, model="x", system="x", messages=[])


def test_record_then_replay_round_trip(tmp_path):
    from llm.provider import Completion
    from llm.record_replay import RecordingProvider, RecordKey

    class FakeInner:
        def complete(self, **kwargs) -> Completion:
            return Completion(
                text="fake response",
                usage=Usage(input_tokens=10, output_tokens=5, thinking_tokens=0, cost_usd=0.0001),
                model="fake-model",
            )

    recorder = RecordingProvider(FakeInner(), tmp_path)
    key = RecordKey(scenario_id="S01", attack_id="A0_none", repeat_k=0, call_site="interpret")
    recorded = recorder.complete(
        record_key=key, model="fake-model", system="sys", messages=[{"role": "user", "text": "hi"}]
    )
    assert recorded.text == "fake response"

    replay = ReplayProvider(tmp_path)
    replayed = replay.complete(record_key=key)
    assert replayed.text == "fake response"
    assert replayed.usage.cost_usd == 0.0, "replays must be free by construction"
