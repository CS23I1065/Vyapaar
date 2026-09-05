"""
The real AI evaluation -- interpret() against a real model, not a
ground-truth-constructed stand-in.

Everything in eval/harness.py, eval/ablation.py, and eval/metrics.py is
deliberately zero-LLM-cost: scenario intents are built directly from
authored ground truth, which proves the DETERMINISTIC layer (the policy
engine, the merchant matching, the ranker) is correct, but says nothing
about whether the LLM can actually turn a real sentence into a usable,
safe intent. For an AI hackathon submission, that second question is the
one that actually matters -- this module answers it, with real API
calls and real (small) cost.

Three things this module measures, that nothing else in this repo does:

1. **Interpretation fidelity** -- does a live interpret() call correctly
   extract category / budget / required_attributes from each of the 30
   corpus scenarios' own request_text, repeated k times to catch
   sampling variance?
2. **Ambiguity handling** -- the 30-scenario corpus is a bad test of
   this, because every one of those 30 states a clean budget by
   construction. eval/ambiguity_corpus.py's 8 deliberately messy
   requests test the thing interpret() is actually FOR: recognizing
   when it doesn't know something and asking, without also asking on
   requests that were perfectly answerable.
3. **End-to-end with the real output** -- takes what a live model
   ACTUALLY produced (not the synthetic ground-truth intent), passes it
   through apply_simulated_review() (an approximation of the one human
   review step the real pipeline requires before anything is signable --
   see that function's docstring), and runs the result through the exact
   same attack/config matrix eval/harness.py already has, via
   run_instance's `intent_override` hook. This is the number that
   answers "does the safety story hold when a real, occasionally-
   imperfect LLM is doing the interpreting" -- not just "when I hand the
   engine a hand-built perfect intent."

Cost, threaded through eval.budget.BudgetGuard like every other live LLM
path in this repo: ~250-300 calls at gemini-3.1-flash-lite pricing, well
under EVAL_BUDGET_USD's $5 default. See the docstring on
run_full_live_evaluation() for the exact count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

from buyer.interpret import interpret
from eval.ambiguity_corpus import AMBIGUOUS_REQUESTS, AmbiguousRequest
from eval.budget import BudgetGuard
from eval.config import CONFIGS
from eval.corpus import Scenario, load_corpus
from eval.harness import InstanceResult, run_instance
from mandates.keys import generate_keypair
from mandates.schemas import IntentMandate
from redteam.attacks import AttackId

EVAL_MODEL_DEFAULT = "gemini-3.1-flash-lite"

# Gemini's free tier caps gemini-3.1-flash-lite at 15 requests/minute,
# a rate-limit constraint independent of and in addition to
# EVAL_BUDGET_USD's cost cap -- the provider's own retry/backoff
# (llm/provider.py, 4 attempts, ~1.5-12s exponential) is for transient
# blips, not sustained throughput above a hard per-minute ceiling. 4.5s
# between calls keeps sustained throughput under 15/min with margin.
DEFAULT_PACE_SECONDS = 4.5


def _normalize(s: str | None) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _category_matches(draft_category: str | None, ground_truth_category: str | None) -> bool:
    """Substring-tolerant both ways, same normalization rule as
    buyer/rank.py's brand matching -- an LLM's own category noun choice
    ("shirt" vs "shirts" vs "apparel") is exactly the kind of surface
    variation an exact string comparison would wrongly fail on."""
    if ground_truth_category is None:
        return True
    a, b = _normalize(draft_category), _normalize(ground_truth_category)
    if not a or not b:
        return False
    return a in b or b in a


def _budget_matches(draft_paise: int | None, ground_truth_rupees: float | None) -> bool:
    if ground_truth_rupees is None:
        return draft_paise is None
    if draft_paise is None:
        return False
    return abs(draft_paise - round(ground_truth_rupees * 100)) <= 100  # within Re.1


def _attributes_match(draft_attrs: dict[str, str], ground_truth_attrs: list[dict]) -> bool:
    """Recall on ground truth's stated attributes -- the draft may
    reasonably surface MORE detail than ground truth asked for; missing
    a stated one is the failure this checks for."""
    for item in ground_truth_attrs:
        key, value = item["key"], item["value"]
        if _normalize(draft_attrs.get(key)) != _normalize(value):
            return False
    return True


@dataclass(frozen=True)
class FidelityResult:
    scenario_id: str
    k: int
    is_signable: bool
    category_correct: bool
    budget_correct: bool
    attributes_correct: bool
    raw_response: dict
    draft_intent: IntentMandate | None = None
    # Kept so the E2E matrix can reuse this exact live output without a
    # second, non-reproducible interpret() call -- see
    # run_full_live_evaluation()'s use of this field.

    @property
    def fully_correct(self) -> bool:
        return self.is_signable and self.category_correct and self.budget_correct and self.attributes_correct


def run_interpretation_fidelity(
    scenarios: list[Scenario], *, provider, model: str, k: int, now: datetime,
    pace_seconds: float = DEFAULT_PACE_SECONDS, use_catalog_categories: bool = True,
    use_catalog_attribute_keys: bool = True,
) -> list[FidelityResult]:
    """One live interpret() call per (scenario, repeat) -- `k` x
    `len(scenarios)` calls total. Every call is independent (fresh
    keypairs, no shared state), so results are directly comparable
    across repeats. Paced at `pace_seconds` between calls -- see
    DEFAULT_PACE_SECONDS for why this isn't just cost-guarded.

    `use_catalog_categories` (default True) feeds each scenario's own
    catalog category vocabulary into interpret()'s `available_categories`,
    constraining the model's `category` output to an enum over the real
    taxonomy instead of free text. Set False to reproduce the original
    free-text behaviour for an explicit before/after comparison.

    `use_catalog_attribute_keys` (default True) does the same one level
    down -- feeds each scenario's own catalog `attributes` dict KEYS
    (never values, see interpret()'s docstring), grouped BY CATEGORY,
    into `available_attribute_keys`. Per-category (not a flat union
    across the whole catalog) so a request in one category can't be
    offered another category's attribute keys as valid. Set False to
    reproduce the original free-text behaviour."""
    results: list[FidelityResult] = []
    first_call = True
    for scenario in scenarios:
        categories = sorted({item.category for item in scenario.catalog}) if use_catalog_categories else None
        attribute_keys: dict[str, list[str]] | None = None
        if use_catalog_attribute_keys:
            attribute_keys = {}
            for item in scenario.catalog:
                attribute_keys.setdefault(item.category, set()).update(item.attributes.keys())
            attribute_keys = {cat: sorted(keys) for cat, keys in attribute_keys.items()}
        for rep in range(k):
            if not first_call and pace_seconds > 0:
                time.sleep(pace_seconds)
            first_call = False
            human_kp = generate_keypair()
            outcome = interpret(
                scenario.request_text, provider=provider, model=model,
                principal_kid=human_kp.kid, agent_kid=human_kp.kid,
                allowed_merchants=[scenario.merchant_id], now=now,
                available_categories=categories,
                available_attribute_keys=attribute_keys,
            )
            gt = scenario.ground_truth
            if outcome.is_signable:
                draft = outcome.draft_intent
                results.append(
                    FidelityResult(
                        scenario_id=scenario.id, k=rep, is_signable=True,
                        category_correct=_category_matches(draft.hard.category, gt.get("category")),
                        budget_correct=_budget_matches(draft.budget.total_paise, gt.get("budget_total_rupees")),
                        attributes_correct=_attributes_match(draft.hard.required_attributes, gt.get("required_attributes", [])),
                        raw_response=outcome.raw_response, draft_intent=draft,
                    )
                )
            else:
                results.append(
                    FidelityResult(
                        scenario_id=scenario.id, k=rep, is_signable=False,
                        category_correct=False, budget_correct=False, attributes_correct=False,
                        raw_response=outcome.raw_response,
                    )
                )
    return results


def compute_fidelity_metrics(results: list[FidelityResult], scenarios: list[Scenario]) -> dict:
    n = len(results)
    by_scenario: dict[str, list[FidelityResult]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario_id, []).append(r)

    pass_k_scenarios = sum(1 for rs in by_scenario.values() if all(r.fully_correct for r in rs))
    return {
        "n_calls": n,
        "n_scenarios": len(by_scenario),
        "signable_rate": sum(1 for r in results if r.is_signable) / n if n else None,
        "category_accuracy": sum(1 for r in results if r.category_correct) / n if n else None,
        "budget_accuracy": sum(1 for r in results if r.budget_correct) / n if n else None,
        "attributes_accuracy": sum(1 for r in results if r.attributes_correct) / n if n else None,
        "fully_correct_rate": sum(1 for r in results if r.fully_correct) / n if n else None,
        "pass_at_k": pass_k_scenarios / len(by_scenario) if by_scenario else None,
    }


@dataclass(frozen=True)
class AmbiguityResult:
    request_id: str
    k: int
    needs_clarification: bool
    matches_expected: bool | None  # None for unscored (should_need_clarification-agnostic) entries


def run_ambiguity_stress_test(
    requests: tuple[AmbiguousRequest, ...], *, provider, model: str, k: int, now: datetime,
    pace_seconds: float = DEFAULT_PACE_SECONDS,
) -> list[AmbiguityResult]:
    results: list[AmbiguityResult] = []
    first_call = True
    for req in requests:
        for rep in range(k):
            if not first_call and pace_seconds > 0:
                time.sleep(pace_seconds)
            first_call = False
            human_kp = generate_keypair()
            outcome = interpret(
                req.request_text, provider=provider, model=model,
                principal_kid=human_kp.kid, agent_kid=human_kp.kid,
                allowed_merchants=["stress-test-merchant"], now=now,
            )
            actually_needs_it = outcome.needs_clarification or not outcome.is_signable
            matches = actually_needs_it == req.should_need_clarification
            results.append(
                AmbiguityResult(request_id=req.id, k=rep, needs_clarification=actually_needs_it, matches_expected=matches)
            )
    return results


def compute_ambiguity_metrics(results: list[AmbiguityResult]) -> dict:
    scored = [r for r in results if r.matches_expected is not None]
    n = len(scored)
    return {
        "n_calls": len(results),
        "n_scored": n,
        "accuracy": sum(1 for r in scored if r.matches_expected) / n if n else None,
    }


CART_TAMPERING_ATTACKS: tuple[AttackId, ...] = ("A4_price_swap", "A5_quantity_inflation", "A6_catalog_overcharge")


def apply_simulated_review(
    draft_intent: IntentMandate, scenario: Scenario, raw_response: dict,
) -> IntentMandate | None:
    """Approximates buyer/review.py::conduct_review() for the live E2E
    replay, which otherwise signs interpret()'s draft directly and never
    routes it through the one module that acts on `needs_clarification`
    or sets an escalation tripwire (FIXES.md #30).

    A draft still flagged needs_clarification is treated as NOT
    SIGNABLE, same as a draft interpret() never built at all -- a real
    human would stop and ask here, and answering that live would cost
    another API call this replay doesn't make. The scenario is excluded
    from the E2E matrix, not silently scored as a pass or a fail.

    escalate_above_paise is a review-screen-only capability interpret()
    structurally cannot produce (see its own docstring on why there is
    no ESCALATE_FRACTION derivation) -- filled in here from the
    scenario's own authored ground truth, exactly as
    eval/harness.py::build_ground_truth_intent() already does for the
    deterministic matrix. Both paths make the same "as if a human had
    already reviewed and set this" assumption; this applies it
    consistently to the live path too, instead of only the synthetic
    one."""
    if raw_response.get("needs_clarification"):
        return None

    escalate_above_rupees = scenario.ground_truth.get("escalate_above_rupees")
    if escalate_above_rupees is None:
        return draft_intent

    escalate_above_paise = round(escalate_above_rupees * 100)
    amended_budget = draft_intent.budget.model_copy(update={"escalate_above_paise": escalate_above_paise})
    return draft_intent.model_copy(update={"budget": amended_budget})


def run_live_e2e_matrix(
    scenarios: list[Scenario], live_intents: dict[str, IntentMandate], *, now: datetime,
) -> list[InstanceResult]:
    """Runs A0_none plus the cart-tampering attacks across every config,
    for every scenario that has a live intent to run against --
    `live_intents` maps scenario_id -> an actual signable IntentMandate a
    live interpret() call produced (the caller picks which repeat,
    typically the first signable one, and applies apply_simulated_review()
    to it; see run_full_live_evaluation()).

    A scenario missing from `live_intents` is skipped, not silently
    treated as a pass or a fail -- it means no repeat of the live model's
    output ever survived interpretation AND simulated review, which is
    itself the finding to report, not paper over with a fallback."""
    results: list[InstanceResult] = []
    for scenario in scenarios:
        intent = live_intents.get(scenario.id)
        if intent is None:
            continue
        for config in CONFIGS:
            results.append(
                run_instance(scenario, attack_id="A0_none", config=config, upsell_mode="off", k=0, now=now, intent_override=intent)
            )
            for attack_id in CART_TAMPERING_ATTACKS:
                results.append(
                    run_instance(scenario, attack_id=attack_id, config=config, upsell_mode="off", k=0, now=now, intent_override=intent)
                )
    return results


@dataclass(frozen=True)
class LiveEvaluationResult:
    fidelity_results: list[FidelityResult]
    fidelity_metrics: dict
    ambiguity_results: list[AmbiguityResult]
    ambiguity_metrics: dict
    e2e_results: list[InstanceResult]
    e2e_scenarios_covered: int
    e2e_scenarios_total: int
    budget_summary: dict
    e2e_metrics_summary: list[dict] | None = None
    # Precomputed {config_id, n_instances, benign_utility,
    # utility_under_attack, false_approval_rate} rows -- lets the report
    # be regenerated from a SAVED run (e2e_results isn't JSON-serializable
    # as-is without re-deriving IntentMandate objects) without spending
    # money again. compute_all_metrics(e2e_results) is still the source
    # of truth when e2e_results is present; this is the fallback.


def run_full_live_evaluation(
    *, provider, model: str = EVAL_MODEL_DEFAULT, k: int = 8, now: datetime, budget: BudgetGuard,
    pace_seconds: float = DEFAULT_PACE_SECONDS, use_catalog_categories: bool = True,
    use_catalog_attribute_keys: bool = True,
) -> LiveEvaluationResult:
    """The whole live-AI evaluation in one call.

    Cost: len(scenarios) x k live interpret() calls for fidelity, plus
    len(AMBIGUOUS_REQUESTS) x k for the ambiguity stress test. At the
    default k=8 and the current 30-scenario / 8-request corpus, that is
    (30 + 8) x 8 = 304 live calls. At gemini-3.1-flash-lite pricing
    (~$0.0003/call for this prompt's size), that is roughly $0.05-$0.15
    total -- well under EVAL_BUDGET_USD's $5 default. k=8 matches
    tau-bench's own cited pass^k methodology rather than being an
    arbitrary choice. `budget` aborts the run before exceeding its
    ceiling regardless of this estimate being off.

    The end-to-end matrix costs ZERO additional calls -- it replays the
    fidelity pass's own live outputs (the first signable draft per
    scenario, passed through apply_simulated_review()) through the
    existing free, deterministic attack/config matrix via run_instance's
    intent_override hook."""
    scenarios = load_corpus()

    fidelity_results = run_interpretation_fidelity(
        scenarios, provider=provider, model=model, k=k, now=now, pace_seconds=pace_seconds,
        use_catalog_categories=use_catalog_categories,
        use_catalog_attribute_keys=use_catalog_attribute_keys,
    )
    fidelity_metrics = compute_fidelity_metrics(fidelity_results, scenarios)

    if pace_seconds > 0:
        time.sleep(pace_seconds)  # keep the gap at the fidelity/ambiguity boundary too
    ambiguity_results = run_ambiguity_stress_test(AMBIGUOUS_REQUESTS, provider=provider, model=model, k=k, now=now, pace_seconds=pace_seconds)
    ambiguity_metrics = compute_ambiguity_metrics(ambiguity_results)

    # First signable draft per scenario, in k order -- a real caller
    # (a human at the review screen) would only ever see ONE
    # interpretation per request anyway; this is that same choice,
    # applied consistently rather than picking whichever happens to
    # look best. That one draft still has to survive
    # apply_simulated_review() to count as usable -- a scenario whose
    # first signable draft needs clarification is excluded outright, not
    # retried against a later repeat looking for one that doesn't (that
    # would be cherry-picking a lucky sample, not simulating review).
    scenarios_by_id = {s.id: s for s in scenarios}
    picked_scenario_ids: set[str] = set()
    live_intents: dict[str, IntentMandate] = {}
    for r in sorted(fidelity_results, key=lambda r: r.k):
        if r.scenario_id in picked_scenario_ids or not r.is_signable:
            continue
        picked_scenario_ids.add(r.scenario_id)
        reviewed = apply_simulated_review(r.draft_intent, scenarios_by_id[r.scenario_id], r.raw_response)
        if reviewed is not None:
            live_intents[r.scenario_id] = reviewed

    e2e_results = run_live_e2e_matrix(scenarios, live_intents, now=now)

    return LiveEvaluationResult(
        fidelity_results=fidelity_results, fidelity_metrics=fidelity_metrics,
        ambiguity_results=ambiguity_results, ambiguity_metrics=ambiguity_metrics,
        e2e_results=e2e_results, e2e_scenarios_covered=len(live_intents), e2e_scenarios_total=len(scenarios),
        budget_summary=budget.summary(),
    )
