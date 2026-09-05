"""
The pre-signature review screen -- the single human touchpoint the
design is built around.

Without it, a completely in-policy purchase would touch the human
TWICE (sign, then escalate), with a full merchant re-negotiation in
between, because every question the system might ask would be deferred
to escalation time instead of asked while the human is already sitting
there. This module makes "ask once, up front" an actual code path:

  - buyer/interpret.py returns needs_clarification / ambiguous_field /
    clarification_question and nothing ever consumed them -- the only
    callers were two tests.
  - The escalation tripwire was DERIVED (0.8 x budget) rather than asked
    about, so a cart in the top 20% of a budget the human themselves set
    always escalated for no added safety.
  - substitutions_preapproved could only be set by the POST-HOC amendment
    flow, so a substitute inside a tolerance the human already set still
    stopped the loop the first time.

conducto_review() is the one orchestrator: resolve ambiguity (bounded),
then collect review answers, then sign. Three Reviewer implementations:

  AIReviewer (the DEFAULT): reads preferences the LLM already extracted
  from the request text (tripwire, substitution policy, tolerance) and
  applies safe defaults for anything not stated. Zero human questions
  before the cart. The only human interaction is the final "Buy this?"
  shown with the real cart in hand -- which is the RIGHT checkpoint.

  ScriptedReviewer: used by the eval harness so it never blocks on
  stdin. Answers everything from a fixed script.

  The rule, stated hard: exactly one blocking human interaction per
  purchase. If you find yourself wanting to call back into a Reviewer a
  second time for the SAME purchase, the feature belongs in the review
  screen, not after it.

What this module does not do: it never decides whether a purchase is
approved. It builds and signs an IntentMandate; buyer/policy_engine.py's
evaluate() is the only thing that ever produces a Decision. The LLM
proposes (via interpret()), the human authorizes (via this module's
signature), and the engine enforces -- unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from buyer.interpret import InterpretResult, enrich_request_text, interpret
from mandates.keys import KeyPair
from mandates.schemas import Envelope, IntentMandate
from mandates.sign import sign_mandate

MAX_CLARIFICATION_ROUNDS = 2
# A hard cap, not just a convention: if the interpreter still cannot
# produce a signable draft after this many answers, something is wrong
# with the request or the model, and looping forever on clarification
# would itself become the second touchpoint this design forbids.


@dataclass(frozen=True)
class ReviewAnswers:
    """What a human actually decided at the one touchpoint. Every field
    here is either a genuine choice (the tripwire, substitution policy,
    near-miss notification) or an explicit confirm/reject -- nothing is
    inferred silently and re-presented as though it were the user's own
    request."""

    confirmed: bool
    escalate_above_paise: int | None = None  # None = no tripwire (== total_paise)
    substitutions_preapproved: bool = False
    substitution_tolerance_pct: float | None = None  # None = keep interpret()'s default
    notify_on_near_miss: bool = True


@dataclass(frozen=True)
class ReviewResult:
    envelope: Envelope | None  # None iff the human declined or gave up on clarification
    intent: IntentMandate | None
    declined_reason: str | None = None


class Reviewer(Protocol):
    def ask_clarification(self, question: str) -> str: ...

    def review(self, draft: IntentMandate) -> ReviewAnswers: ...


@dataclass(frozen=True)
class AIReviewer:
    """The default Reviewer: reads preferences the LLM already extracted
    from the request text (tripwire, substitution policy, tolerance) and
    applies safe defaults for anything not stated. Zero human questions
    asked before the cart exists. The only remaining human interaction is
    the final \"Buy this?\" shown with the real signed cart in hand.

    `raw_response` is InterpretResult.raw_response -- the dict the LLM
    returned on the last interpret() call. Fields read:
      escalate_above_rupees  -> tripwire (None = no tripwire, safe default)
      substitutions_ok       -> substitutions_preapproved (False if absent)
      substitution_tolerance_pct -> tolerance override (None = keep default)
      notify_on_near_miss    -> notification preference (True if absent)

    ask_clarification() still fires for the one thing interpret() cannot
    proceed without: an unstated budget. That is a genuine blocker, not a
    preference question -- there is no honest IntentMandate without it."""

    raw_response: dict

    def ask_clarification(self, question: str) -> str:
        return input(f"{question} ")

    def review(self, draft: IntentMandate) -> ReviewAnswers:
        raw = self.raw_response
        escalate_rupees = raw.get("escalate_above_rupees")
        if escalate_rupees is not None:
            try:
                escalate_paise = min(
                    round(float(escalate_rupees) * 100),
                    draft.budget.total_paise,
                )
            except (TypeError, ValueError):
                escalate_paise = None
        else:
            escalate_paise = None  # no tripwire -- safe default

        tolerance = raw.get("substitution_tolerance_pct")
        if tolerance is not None:
            try:
                tolerance = float(tolerance)
            except (TypeError, ValueError):
                tolerance = None

        return ReviewAnswers(
            confirmed=True,  # AI reviewed the intent -- no human question needed
            escalate_above_paise=escalate_paise,
            substitutions_preapproved=bool(raw.get("substitutions_ok", False)),
            substitution_tolerance_pct=tolerance,
            notify_on_near_miss=bool(raw.get("notify_on_near_miss", True)),
        )


@dataclass(frozen=True)
class ScriptedReviewer:
    """Used by the eval harness so it never blocks on stdin. `answers` is
    consulted once; `clarification_answers` are consumed in order if the
    interpreter needs more than one round."""

    answers: ReviewAnswers
    clarification_answers: tuple[str, ...] = ()
    _clarification_index: int = 0

    def ask_clarification(self, question: str) -> str:
        idx = self._clarification_index
        if idx >= len(self.clarification_answers):
            raise ValueError(
                f"ScriptedReviewer ran out of clarification_answers (asked at round {idx + 1}): {question!r}"
            )
        object.__setattr__(self, "_clarification_index", idx + 1)
        return self.clarification_answers[idx]

    def review(self, draft: IntentMandate) -> ReviewAnswers:
        return self.answers


def _apply_review_answers(intent: IntentMandate, answers: ReviewAnswers) -> IntentMandate:
    new_budget = intent.budget
    if answers.escalate_above_paise is not None:
        new_budget = new_budget.model_copy(update={"escalate_above_paise": answers.escalate_above_paise})

    new_soft = intent.soft.model_copy(
        update={
            "substitutions_preapproved": answers.substitutions_preapproved,
            "notify_on_near_miss": answers.notify_on_near_miss,
            **(
                {"substitution_tolerance_pct": answers.substitution_tolerance_pct}
                if answers.substitution_tolerance_pct is not None
                else {}
            ),
        }
    )
    return intent.model_copy(update={"budget": new_budget, "soft": new_soft})


def conduct_review(
    request_text: str,
    *,
    provider,
    model: str,
    principal_kid: str,
    agent_kid: str,
    allowed_merchants: list[str] | None,
    now: datetime,
    reviewer: Reviewer,
    human_keypair: KeyPair,
    validity: timedelta | None = None,
    max_clarification_rounds: int = MAX_CLARIFICATION_ROUNDS,
) -> ReviewResult:
    """The one orchestrator: interpret -> resolve ambiguity (bounded) ->
    present the full review screen -> sign. Every question the human
    might have been asked at escalation time is asked HERE instead, so
    G7.x's job after this point is to catch a genuine surprise, not a
    routine one.

    Returns envelope=None if the human declines, or if the interpreter
    still cannot produce a signable draft after max_clarification_rounds
    -- both are "no purchase," never a raised exception, because a
    refusal to sign is not a bug."""
    kwargs = dict(
        provider=provider, model=model, principal_kid=principal_kid, agent_kid=agent_kid,
        allowed_merchants=allowed_merchants, now=now,
    )
    if validity is not None:
        kwargs["validity"] = validity

    text = request_text
    result: InterpretResult = interpret(text, **kwargs)

    rounds = 0
    while not result.is_signable:
        if rounds >= max_clarification_rounds:
            return ReviewResult(
                envelope=None, intent=None,
                declined_reason=(
                    f"could not resolve '{result.ambiguous_field}' after "
                    f"{max_clarification_rounds} clarification round(s)"
                ),
            )
        question = result.clarification_question or "Could you clarify that?"
        answer = reviewer.ask_clarification(question)
        text = enrich_request_text(text, question, answer)
        result = interpret(text, **kwargs)
        rounds += 1

    draft = result.draft_intent
    assert draft is not None  # guaranteed by is_signable

    # A second round of ambiguity (needs_clarification=True but a budget
    # WAS stated) gets the same treatment: ask, enrich, re-run -- G7.3 is
    # the backstop for whatever survives this, not the first line of
    # defense.
    while result.needs_clarification and rounds < max_clarification_rounds:
        question = result.clarification_question or "Could you clarify that?"
        answer = reviewer.ask_clarification(question)
        text = enrich_request_text(text, question, answer)
        result = interpret(text, **kwargs)
        if not result.is_signable:
            break
        draft = result.draft_intent
        rounds += 1

    if draft is None:
        return ReviewResult(
            envelope=None, intent=None,
            declined_reason=f"could not resolve '{result.ambiguous_field}' after clarification",
        )

    answers = reviewer.review(draft)
    if not answers.confirmed:
        return ReviewResult(envelope=None, intent=draft, declined_reason="human declined to sign")

    final_intent = _apply_review_answers(draft, answers)
    envelope = sign_mandate(final_intent, human_keypair)
    return ReviewResult(envelope=envelope, intent=final_intent, declined_reason=None)
