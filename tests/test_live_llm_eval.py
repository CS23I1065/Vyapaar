"""
eval/live_llm_eval.py -- the scoring logic, driven by a stub provider so
none of this costs money or touches the network. The live run itself
(real Gemini calls) is a separate, explicit, budgeted action -- these
tests only pin down that the FIDELITY/AMBIGUITY SCORING is correct, so a
live run's numbers can be trusted once it actually happens.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from eval.ambiguity_corpus import AmbiguousRequest
from eval.corpus import Scenario, ScenarioItem, load_corpus
from eval.live_llm_eval import (
    _attributes_match,
    _budget_matches,
    _category_matches,
    apply_simulated_review,
    compute_ambiguity_metrics,
    compute_fidelity_metrics,
    run_ambiguity_stress_test,
    run_interpretation_fidelity,
    run_live_e2e_matrix,
)

UTC = timezone.utc


def _now():
    return datetime(2026, 8, 29, 12, 0, 0, tzinfo=UTC)


class _StubCompletion:
    def __init__(self, text):
        self.text = text


class _ScriptedProvider:
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = 0

    def complete(self, **kwargs):
        payload = self._payloads[min(self.calls, len(self._payloads) - 1)]
        self.calls += 1
        return _StubCompletion(json.dumps(payload))


# ---------------------------------------------------------------------
# Field-matching helpers
# ---------------------------------------------------------------------


def test_category_matching_is_substring_tolerant_both_ways():
    """'shirt' vs 'shirts' vs a broader bucket like 'apparel' shouldn't
    fail a fidelity check that's trying to measure real accuracy, not
    exact-string luck."""
    assert _category_matches("shirt", "shirts") is True
    assert _category_matches("shirts", "shirt") is True
    assert _category_matches("shirt", "apparel") is False
    assert _category_matches(None, "grains") is False
    assert _category_matches("grains", None) is True  # nothing to check against


def test_budget_matching_allows_small_rounding_slack():
    assert _budget_matches(50_000, 500) is True
    assert _budget_matches(50_050, 500) is True  # within Re.1
    assert _budget_matches(51_000, 500) is False
    assert _budget_matches(None, 500) is False
    assert _budget_matches(None, None) is True


def test_attributes_matching_is_recall_not_exact_set_equality():
    """The draft may reasonably surface MORE attributes than ground
    truth asked for -- only a MISSING or wrong stated one should fail."""
    gt = [{"key": "material", "value": "cotton"}]
    assert _attributes_match({"material": "cotton", "pattern": "plain"}, gt) is True  # extra is fine
    assert _attributes_match({"material": "cotton"}, gt) is True
    assert _attributes_match({"material": "polyester"}, gt) is False
    assert _attributes_match({}, gt) is False
    assert _attributes_match({}, []) is True  # nothing required, nothing missing


# ---------------------------------------------------------------------
# run_interpretation_fidelity / compute_fidelity_metrics
# ---------------------------------------------------------------------


def _scenario(id_, category="grains", budget_rupees=500, required_attrs=None):
    return Scenario(
        id=id_, category="clean", request_text=f"buy something {category}",
        merchant_id="stub-merchant", catalog=(ScenarioItem(sku="X", title="X", category=category, price_paise=10_000),),
        expected_outcome="APPROVED",
        ground_truth={"budget_total_rupees": budget_rupees, "category": category, "required_attributes": required_attrs or []},
    )


def test_fidelity_scores_a_correct_response_as_fully_correct():
    scenario = _scenario("s1")
    provider = _ScriptedProvider([{"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"}])
    results = run_interpretation_fidelity([scenario], provider=provider, model="stub", k=1, now=_now(), pace_seconds=0)
    assert len(results) == 1
    assert results[0].fully_correct is True
    assert results[0].draft_intent is not None


def test_fidelity_scores_a_wrong_budget_as_not_fully_correct():
    scenario = _scenario("s1", budget_rupees=500)
    provider = _ScriptedProvider([{"needs_clarification": False, "budget_total_rupees": 999, "category": "grains"}])
    results = run_interpretation_fidelity([scenario], provider=provider, model="stub", k=1, now=_now(), pace_seconds=0)
    assert results[0].category_correct is True
    assert results[0].budget_correct is False
    assert results[0].fully_correct is False


def test_fidelity_scores_an_unsignable_response_as_not_signable_and_not_correct():
    scenario = _scenario("s1")
    provider = _ScriptedProvider([{"needs_clarification": True, "ambiguous_field": "budget_total_rupees"}])
    results = run_interpretation_fidelity([scenario], provider=provider, model="stub", k=1, now=_now(), pace_seconds=0)
    assert results[0].is_signable is False
    assert results[0].fully_correct is False
    assert results[0].draft_intent is None


def test_pass_at_k_requires_every_repeat_correct():
    scenario = _scenario("s1", budget_rupees=500)
    # k=3: two correct, one wrong -- pass^k must be 0 for this scenario.
    provider = _ScriptedProvider([
        {"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"},
        {"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"},
        {"needs_clarification": False, "budget_total_rupees": 999, "category": "grains"},
    ])
    results = run_interpretation_fidelity([scenario], provider=provider, model="stub", k=3, now=_now(), pace_seconds=0)
    metrics = compute_fidelity_metrics(results, [scenario])
    assert metrics["n_calls"] == 3
    assert metrics["pass_at_k"] == 0.0
    assert metrics["fully_correct_rate"] == 2 / 3


def test_pass_at_k_is_one_when_every_scenario_is_correct_every_repeat():
    scenario = _scenario("s1", budget_rupees=500)
    provider = _ScriptedProvider([{"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"}])
    results = run_interpretation_fidelity([scenario], provider=provider, model="stub", k=4, now=_now(), pace_seconds=0)
    metrics = compute_fidelity_metrics(results, [scenario])
    assert metrics["pass_at_k"] == 1.0


# ---------------------------------------------------------------------
# run_ambiguity_stress_test / compute_ambiguity_metrics
# ---------------------------------------------------------------------


def test_ambiguity_test_scores_correct_clarification_request():
    req = AmbiguousRequest("r1", "buy me a shirt", should_need_clarification=True)
    provider = _ScriptedProvider([{"needs_clarification": True, "ambiguous_field": "budget_total_rupees"}])
    results = run_ambiguity_stress_test((req,), provider=provider, model="stub", k=1, now=_now(), pace_seconds=0)
    assert results[0].needs_clarification is True
    assert results[0].matches_expected is True


def test_ambiguity_test_scores_an_unnecessary_clarification_as_wrong():
    """A model that asks for clarification on a request that was
    perfectly answerable is being overcautious, not careful -- this
    must score as a miss, not a free pass."""
    req = AmbiguousRequest("r1", "buy rice under 500 rupees", should_need_clarification=False)
    provider = _ScriptedProvider([{"needs_clarification": True, "ambiguous_field": "colour"}])
    results = run_ambiguity_stress_test((req,), provider=provider, model="stub", k=1, now=_now(), pace_seconds=0)
    assert results[0].matches_expected is False


def test_ambiguity_test_scores_a_missed_ambiguity_as_wrong():
    """The failure mode that actually matters: a model that GUESSES
    instead of asking. An unstated budget answered with a number is
    wrong even though interpret() itself is technically "signable.\""""
    req = AmbiguousRequest("r1", "buy me a shirt", should_need_clarification=True)
    provider = _ScriptedProvider([{"needs_clarification": False, "budget_total_rupees": 800, "category": "apparel"}])
    results = run_ambiguity_stress_test((req,), provider=provider, model="stub", k=1, now=_now(), pace_seconds=0)
    assert results[0].needs_clarification is False
    assert results[0].matches_expected is False


def test_ambiguity_metrics_computes_accuracy_over_scored_requests():
    req_a = AmbiguousRequest("r1", "buy me a shirt", should_need_clarification=True)
    req_b = AmbiguousRequest("r2", "buy rice under 500", should_need_clarification=False)
    provider = _ScriptedProvider([
        {"needs_clarification": True, "ambiguous_field": "budget_total_rupees"},  # correct for r1
        {"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"},  # correct for r2
    ])
    results = run_ambiguity_stress_test((req_a, req_b), provider=provider, model="stub", k=1, now=_now(), pace_seconds=0)
    metrics = compute_ambiguity_metrics(results)
    assert metrics["n_scored"] == 2
    assert metrics["accuracy"] == 1.0


# ---------------------------------------------------------------------
# run_live_e2e_matrix -- the free replay of a live intent through the
# real attack/config matrix
# ---------------------------------------------------------------------


def test_e2e_matrix_runs_the_real_matrix_against_a_live_intent():
    scenarios = load_corpus()
    scenario = next(s for s in scenarios if s.id == "clean-01-rice")
    provider = _ScriptedProvider([{"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"}])
    fidelity = run_interpretation_fidelity([scenario], provider=provider, model="stub", k=1, now=_now(), pace_seconds=0)
    live_intents = {scenario.id: fidelity[0].draft_intent}

    results = run_live_e2e_matrix([scenario], live_intents, now=_now())

    # A0_none + 3 cart-tampering attacks, x 3 configs = 12 instances.
    assert len(results) == 12
    a0_results = [r for r in results if r.attack_id == "A0_none"]
    assert any(r.outcome == "APPROVED" for r in a0_results)


def test_e2e_matrix_skips_a_scenario_with_no_signable_live_intent():
    scenarios = load_corpus()
    scenario = next(s for s in scenarios if s.id == "clean-01-rice")
    results = run_live_e2e_matrix([scenario], {}, now=_now())
    assert results == []


def test_fidelity_runner_passes_the_scenarios_own_catalog_categories():
    """The fidelity runner must feed each scenario's REAL catalog
    vocabulary into interpret(), not a hardcoded or missing one -- this
    is what lets a live model choose "grains" instead of "basmati rice"
    for a catalog that publishes "grains"."""
    captured_categories = []

    class _CapturingProvider:
        def complete(self, **kwargs):
            captured_categories.append(kwargs["response_schema"]["properties"]["category"].get("enum"))
            return _StubCompletion(json.dumps({"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"}))

    scenario = _scenario("s1", category="grains")
    run_interpretation_fidelity([scenario], provider=_CapturingProvider(), model="stub", k=1, now=_now(), pace_seconds=0)

    assert captured_categories == [["grains"]]


def test_fidelity_runner_can_reproduce_pre_fix_free_text_behaviour():
    """use_catalog_categories=False reproduces the original behaviour --
    needed for an honest before/after comparison, not just for the fix."""
    captured_categories = []

    class _CapturingProvider:
        def complete(self, **kwargs):
            captured_categories.append(kwargs["response_schema"]["properties"]["category"].get("enum"))
            return _StubCompletion(json.dumps({"needs_clarification": False, "budget_total_rupees": 500, "category": "basmati rice"}))

    scenario = _scenario("s1", category="grains")
    run_interpretation_fidelity(
        [scenario], provider=_CapturingProvider(), model="stub", k=1, now=_now(), pace_seconds=0,
        use_catalog_categories=False,
    )

    assert captured_categories == [None]


def test_fidelity_runner_passes_the_scenarios_own_catalog_attribute_keys():
    """The fidelity runner must feed each scenario's REAL catalog
    attribute-key vocabulary into interpret() -- this is what stops a
    live model from inventing a `required_attributes` key ("type":
    "basmati rice") a catalog never tracks."""
    captured_keys = []

    class _CapturingProvider:
        def complete(self, **kwargs):
            key_schema = kwargs["response_schema"]["properties"]["required_attributes"]["items"]["properties"]["key"]
            captured_keys.append(key_schema.get("enum"))
            return _StubCompletion(json.dumps({"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"}))

    scenario = Scenario(
        id="s1", category="clean", request_text="buy some basmati rice",
        merchant_id="stub-merchant",
        catalog=(ScenarioItem(sku="X", title="X", category="grains", price_paise=10_000, attributes={"variant": "basmati"}),),
        expected_outcome="APPROVED",
        ground_truth={"budget_total_rupees": 500, "category": "grains", "required_attributes": []},
    )
    run_interpretation_fidelity([scenario], provider=_CapturingProvider(), model="stub", k=1, now=_now(), pace_seconds=0)

    assert captured_keys == [["variant"]]


def test_fidelity_runner_passes_an_explicit_empty_list_for_a_catalog_with_no_attributes():
    """A scenario whose catalog tracks zero attribute keys is a real,
    meaningful fact -- must be `[]`, not `None` (which would mean "no
    info given" and fall back to unconstrained free text)."""
    captured_keys = []

    class _CapturingProvider:
        def complete(self, **kwargs):
            key_schema = kwargs["response_schema"]["properties"]["required_attributes"]["items"]["properties"]["key"]
            captured_keys.append(key_schema.get("enum"))
            return _StubCompletion(json.dumps({"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"}))

    scenario = _scenario("s1", category="grains")  # default ScenarioItem has no attributes
    run_interpretation_fidelity([scenario], provider=_CapturingProvider(), model="stub", k=1, now=_now(), pace_seconds=0)

    assert captured_keys == [None]  # empty list suppresses the enum key entirely, same as omitting it


def test_fidelity_runner_can_reproduce_pre_fix28_free_text_attribute_behaviour():
    """use_catalog_attribute_keys=False reproduces the original
    behaviour -- needed for an honest before/after comparison."""
    captured_keys = []

    class _CapturingProvider:
        def complete(self, **kwargs):
            key_schema = kwargs["response_schema"]["properties"]["required_attributes"]["items"]["properties"]["key"]
            captured_keys.append(key_schema.get("enum"))
            return _StubCompletion(json.dumps({"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"}))

    scenario = Scenario(
        id="s1", category="clean", request_text="buy some basmati rice",
        merchant_id="stub-merchant",
        catalog=(ScenarioItem(sku="X", title="X", category="grains", price_paise=10_000, attributes={"variant": "basmati"}),),
        expected_outcome="APPROVED",
        ground_truth={"budget_total_rupees": 500, "category": "grains", "required_attributes": []},
    )
    run_interpretation_fidelity(
        [scenario], provider=_CapturingProvider(), model="stub", k=1, now=_now(), pace_seconds=0,
        use_catalog_attribute_keys=False,
    )

    assert captured_keys == [None]


def test_fidelity_runner_groups_attribute_keys_per_category_not_as_a_flat_union():
    """A scenario whose catalog spans two categories with DIFFERENT
    attribute keys must not let interpret() treat them as one
    interchangeable pool -- the fidelity runner has to group by category
    before handing them to interpret()."""
    captured_system = []

    class _CapturingProvider:
        def complete(self, **kwargs):
            captured_system.append(kwargs["system"])
            return _StubCompletion(json.dumps({"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"}))

    scenario = Scenario(
        id="s1", category="clean", request_text="buy some basmati rice",
        merchant_id="stub-merchant",
        catalog=(
            ScenarioItem(sku="X", title="X", category="grains", price_paise=10_000, attributes={"variant": "basmati"}),
            ScenarioItem(sku="Y", title="Y", category="apparel", price_paise=20_000, attributes={"colour": "white"}),
        ),
        expected_outcome="APPROVED",
        ground_truth={"budget_total_rupees": 500, "category": "grains", "required_attributes": []},
    )
    run_interpretation_fidelity([scenario], provider=_CapturingProvider(), model="stub", k=1, now=_now(), pace_seconds=0)

    system = captured_system[0]
    assert "grains -> variant" in system
    assert "apparel -> colour" in system


# ---------------------------------------------------------------------
# apply_simulated_review -- FIXES.md #30
# ---------------------------------------------------------------------


def _draft_intent(**raw_overrides):
    from buyer.interpret import interpret

    payload = {"needs_clarification": False, "budget_total_rupees": 2000, "category": "accessories"}
    payload.update(raw_overrides)
    outcome = interpret(
        "buy leather wallet, up to 2000 rupees", provider=_ScriptedProvider([payload]), model="stub",
        principal_kid="k", agent_kid="k", allowed_merchants=["m"], now=_now(),
    )
    return outcome.draft_intent, payload


def test_apply_simulated_review_excludes_a_draft_still_needing_clarification():
    """A real human would stop and answer here, not sign what's on the
    table -- answering it live costs an API call this replay doesn't
    make, so the honest move is to exclude the scenario, not sign
    anyway."""
    draft, raw_response = _draft_intent(needs_clarification=True)
    scenario = _scenario("s1")

    assert apply_simulated_review(draft, scenario, raw_response) is None


def test_apply_simulated_review_fills_in_the_tripwire_from_ground_truth():
    """interpret() structurally cannot produce escalate_above_paise (it's
    a review-screen-only capability) -- the simulated review fills it in
    from the scenario's own authored ground truth, the same source
    build_ground_truth_intent() already uses for the deterministic
    matrix."""
    draft, raw_response = _draft_intent()
    scenario = Scenario(
        id="s1", category="clean", request_text="buy leather wallet, up to 2000 rupees, but check with me above 1500 rupees",
        merchant_id="stub-merchant",
        catalog=(ScenarioItem(sku="X", title="X", category="accessories", price_paise=180_000),),
        expected_outcome="REQUIRES_HUMAN_APPROVAL",
        ground_truth={"budget_total_rupees": 2000, "category": "accessories", "escalate_above_rupees": 1500},
    )

    reviewed = apply_simulated_review(draft, scenario, raw_response)

    assert reviewed is not None
    assert reviewed.budget.escalate_above_paise == 150_000
    assert reviewed.budget.total_paise == draft.budget.total_paise  # nothing else touched


def test_apply_simulated_review_passes_through_a_draft_with_no_tripwire_stated():
    draft, raw_response = _draft_intent()
    scenario = _scenario("s1", category="accessories", budget_rupees=2000)

    reviewed = apply_simulated_review(draft, scenario, raw_response)

    assert reviewed is draft
