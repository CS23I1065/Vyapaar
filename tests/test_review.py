"""
Tests for buyer/review.py -- the orchestrator that makes "exactly one
blocking human interaction per purchase" an actual code path.

Nothing here touches the network: interpret() is driven by a stub
provider returning canned JSON, exactly like tests/test_buyer_loop.py's
budget tests. What's under test is the ORCHESTRATION -- resolving
ambiguity before signing, applying the human's real answers (tripwire,
substitution policy, near-miss notification) to the signed intent, and
never calling the Reviewer a second time once a purchase is signed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from buyer.review import (
    MAX_CLARIFICATION_ROUNDS,
    ReviewAnswers,
    ScriptedReviewer,
    conduct_review,
)
from mandates.keys import KeyDirectory, generate_keypair
from mandates.verify import verify_envelope

UTC = timezone.utc
HUMAN_KP = generate_keypair()


def _now() -> datetime:
    return datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC)


class _StubCompletion:
    def __init__(self, text):
        self.text = text


class _ScriptedProvider:
    """Returns one canned JSON payload per call, in order -- lets a
    multi-round clarification exchange be driven deterministically."""

    def __init__(self, payloads: list[dict]):
        self._payloads = list(payloads)
        self.calls = 0

    def complete(self, **kwargs):
        payload = self._payloads[min(self.calls, len(self._payloads) - 1)]
        self.calls += 1
        return _StubCompletion(json.dumps(payload))


def _base_kwargs(provider):
    return dict(
        provider=provider, model="stub", principal_kid=HUMAN_KP.kid, agent_kid=HUMAN_KP.kid,
        allowed_merchants=["merchant-1"], now=_now(), human_keypair=HUMAN_KP,
    )


def test_stated_budget_and_confirmed_review_signs_in_one_pass():
    """The golden path: no ambiguity, human confirms. Exactly one call
    into the reviewer (review()), zero clarification round-trips."""
    provider = _ScriptedProvider(
        [{"needs_clarification": False, "budget_total_rupees": 500, "category": "apparel"}]
    )
    reviewer = ScriptedReviewer(answers=ReviewAnswers(confirmed=True))

    result = conduct_review(
        "buy me a white t-shirt under 500 rupees", reviewer=reviewer, **_base_kwargs(provider)
    )

    assert result.envelope is not None
    assert result.intent is not None
    assert result.intent.budget.total_paise == 50_000
    assert result.intent.budget.escalate_above_paise == 50_000  # no tripwire by default
    assert provider.calls == 1  # exactly one interpret() call -- no clarification needed

    directory = KeyDirectory()
    directory.register_keypair(HUMAN_KP)
    verified = verify_envelope(result.envelope, directory, required_kid=HUMAN_KP.kid)
    assert verified.valid, verified.reason


def test_unstated_budget_asks_exactly_once_then_signs():
    """The #9 fix end to end: an unstated budget is resolved by ASKING at
    the single touchpoint, not by inventing a number or crashing."""
    provider = _ScriptedProvider(
        [
            {"needs_clarification": True, "ambiguous_field": "budget_total_rupees",
             "clarification_question": "What's your limit for this?"},
            {"needs_clarification": False, "budget_total_rupees": 500, "category": "apparel"},
        ]
    )
    reviewer = ScriptedReviewer(
        answers=ReviewAnswers(confirmed=True), clarification_answers=("under 500 rupees",)
    )

    result = conduct_review("buy me a plain white t-shirt", reviewer=reviewer, **_base_kwargs(provider))

    assert result.envelope is not None
    assert result.intent.budget.total_paise == 50_000
    assert provider.calls == 2  # one clarifying round, then a signable draft


def test_giving_up_after_max_rounds_does_not_raise():
    """If the interpreter still can't produce a signable draft, the
    result is a declined ReviewResult, never an exception -- a refusal to
    sign is not a bug."""
    provider = _ScriptedProvider(
        [{"needs_clarification": True, "ambiguous_field": "budget_total_rupees",
          "clarification_question": "What's your limit?"}]
    )
    reviewer = ScriptedReviewer(
        answers=ReviewAnswers(confirmed=True),
        clarification_answers=("still not sure",) * MAX_CLARIFICATION_ROUNDS,
    )

    result = conduct_review("buy me something", reviewer=reviewer, **_base_kwargs(provider))

    assert result.envelope is None
    assert result.declined_reason is not None
    assert "budget_total_rupees" in result.declined_reason


def test_human_declining_at_review_produces_no_signature():
    provider = _ScriptedProvider(
        [{"needs_clarification": False, "budget_total_rupees": 500, "category": "apparel"}]
    )
    reviewer = ScriptedReviewer(answers=ReviewAnswers(confirmed=False))

    result = conduct_review("buy a t-shirt under 500", reviewer=reviewer, **_base_kwargs(provider))

    assert result.envelope is None
    assert result.intent is not None  # the draft existed, it just wasn't signed
    assert result.declined_reason == "human declined to sign"


def test_review_answers_set_the_tripwire_the_human_actually_chose():
    """The tripwire is a deliberate capability, set at review time -- not
    derived from the total budget the way ESCALATE_FRACTION used to."""
    provider = _ScriptedProvider(
        [{"needs_clarification": False, "budget_total_rupees": 5000, "category": "electronics"}]
    )
    reviewer = ScriptedReviewer(
        answers=ReviewAnswers(confirmed=True, escalate_above_paise=200_000)
    )

    result = conduct_review("buy a speaker under 5000 rupees", reviewer=reviewer, **_base_kwargs(provider))

    assert result.intent.budget.total_paise == 500_000
    assert result.intent.budget.escalate_above_paise == 200_000


def test_review_answers_set_substitution_preapproval_and_tolerance():
    """Answered once, up front -- G7.2 should not need to re-ask a
    question the human already answered here."""
    provider = _ScriptedProvider(
        [{"needs_clarification": False, "budget_total_rupees": 1000, "category": "grocery"}]
    )
    reviewer = ScriptedReviewer(
        answers=ReviewAnswers(
            confirmed=True, substitutions_preapproved=True, substitution_tolerance_pct=15.0
        )
    )

    result = conduct_review("buy rice under 1000 rupees", reviewer=reviewer, **_base_kwargs(provider))

    assert result.intent.soft.substitutions_preapproved is True
    assert result.intent.soft.substitution_tolerance_pct == 15.0


def test_review_answers_set_near_miss_notification_preference():
    provider = _ScriptedProvider(
        [{"needs_clarification": False, "budget_total_rupees": 1000, "category": "grocery"}]
    )
    reviewer = ScriptedReviewer(answers=ReviewAnswers(confirmed=True, notify_on_near_miss=False))

    result = conduct_review("buy rice under 1000 rupees", reviewer=reviewer, **_base_kwargs(provider))

    assert result.intent.soft.notify_on_near_miss is False


def test_scripted_reviewer_running_out_of_answers_raises_a_clear_error():
    """A test-harness footgun, not a production path: if a scenario is
    scripted with too few clarification answers, fail loudly rather than
    hang or silently reuse an unrelated answer."""
    reviewer = ScriptedReviewer(answers=ReviewAnswers(confirmed=True), clarification_answers=())
    with pytest.raises(ValueError, match="ran out of clarification_answers"):
        reviewer.ask_clarification("What's your limit?")
