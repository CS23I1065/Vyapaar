"""
Runs the full safety + revenue matrix and writes raw results as JSONL.

  Safety arm:  30 scenarios x 6 attack conditions (A1-A6) x k repeats,
               PLUS the A0_none baseline (30 x k), across every config
               in eval.config.CONFIGS.
  Revenue arm: 30 scenarios x A0_none x k x upsell_mode in {off, benign},
               reported under C4 specifically (all guardrails on).

Zero LLM cost (see eval/harness.py's module docstring: scenario intents
are constructed directly from authored ground truth, not via a live
interpret() call), so `k` repeats are for catching any accidental
nondeterminism in the pipeline itself, not LLM sampling variance -- this
harness has none by construction (buyer/rank.py ties break
deterministically by sku), so k=1 is sufficient for the free matrix. A
higher k only matters once run_live_interpretation_check() is wired in
for the live-LLM companion check.

eval.budget.BudgetGuard is threaded through even though this arm is
free, to keep the invariant "every eval entry point respects the same
spend ceiling" true regardless of which parts of a future run might add
a live-LLM cost.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from eval.budget import BudgetGuard
from eval.config import CONFIGS
from eval.corpus import Scenario, load_corpus
from eval.harness import ATTACK_DEPENDENT_MECHANISMS, InstanceResult, run_instance

DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "instances.jsonl"

# A1/A2 act on merchant_kit's extraction stage against fixture HTML, not
# on a per-purchase negotiation -- see eval/harness.py's run_instance
# docstring. They are covered by merchant_kit's and redteam's own test
# suites, not re-run here per scenario.

# CART_TAMPERING_ATTACKS modify the cart's actual data (price, quantity)
# -- meaningful against every scenario, clean included, since the point
# is whether a corrupted/inflated cart derived from an otherwise-fine
# purchase gets caught.
CART_TAMPERING_ATTACKS: tuple = ("A4_price_swap", "A5_quantity_inflation", "A6_catalog_overcharge")

# NARRATIVE_ATTACKS change only merchant-facing free text (pitch_text),
# never any typed field the engine reads. redteam/goal_predicates.py's
# own docstring is explicit that goal_injection_produced_an_order is only
# a meaningful signal "in a scenario constructed so a non-hijacked agent
# would REJECT or ESCALATE" -- running it against a scenario that was
# always going to be APPROVED regardless of the injection produces a
# false-positive-looking ASR number for an attack that had zero actual
# effect. Restricted to non-APPROVED ground truth so the reported ASR
# reflects real effect, not scenarios that would have been approved
# either way.
NARRATIVE_ATTACKS: tuple = ("A3_branded_whisper",)

SAFETY_ATTACK_IDS = CART_TAMPERING_ATTACKS + NARRATIVE_ATTACKS


def _instance_to_dict(r: InstanceResult) -> dict:
    return asdict(r)


def run_safety_arm(
    scenarios: list[Scenario], *, k: int, now: datetime, budget: BudgetGuard,
) -> list[InstanceResult]:
    results: list[InstanceResult] = []
    for scenario in scenarios:
        for config in CONFIGS:
            for rep in range(k):
                budget.check(0.0)  # zero-cost by construction; see module docstring
                results.append(run_instance(scenario, attack_id="A0_none", config=config, upsell_mode="off", k=rep, now=now))
            for attack_id in SAFETY_ATTACK_IDS:
                if attack_id in NARRATIVE_ATTACKS and (
                    scenario.expected_outcome == "APPROVED" or scenario.mechanism in ATTACK_DEPENDENT_MECHANISMS
                ):
                    # See NARRATIVE_ATTACKS docstring above. The mechanism
                    # exclusion is the same reasoning one level deeper: a
                    # scenario whose REJECTED label only holds under its
                    # OWN designated attack (e.g. bad-06 under A6) is,
                    # under any OTHER attack (including a no-op narrative
                    # one), just an honest purchase that should legitimately
                    # be approved -- not a case where the injection "won."
                    continue
                for rep in range(k):
                    budget.check(0.0)
                    results.append(run_instance(scenario, attack_id=attack_id, config=config, upsell_mode="off", k=rep, now=now))
    return results


def run_revenue_arm(scenarios: list[Scenario], *, k: int, now: datetime, budget: BudgetGuard) -> list[InstanceResult]:
    c4 = next(c for c in CONFIGS if c.id == "C4")
    results: list[InstanceResult] = []
    for scenario in scenarios:
        for upsell_mode in ("off", "benign"):
            for rep in range(k):
                budget.check(0.0)
                results.append(run_instance(scenario, attack_id="A0_none", config=c4, upsell_mode=upsell_mode, k=rep, now=now))
    return results


def run_full_matrix(
    *, corpus_path: Path | None = None, k: int = 3, now: datetime | None = None,
    budget: BudgetGuard | None = None, results_path: Path = DEFAULT_RESULTS_PATH,
) -> list[InstanceResult]:
    scenarios = load_corpus(corpus_path) if corpus_path else load_corpus()
    now = now or datetime.now(timezone.utc)
    budget = budget or BudgetGuard.from_env()

    safety = run_safety_arm(scenarios, k=k, now=now, budget=budget)
    revenue = run_revenue_arm(scenarios, k=k, now=now, budget=budget)
    all_results = safety + revenue

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        for r in all_results:
            f.write(json.dumps(_instance_to_dict(r)) + "\n")

    return all_results


def load_results(path: Path = DEFAULT_RESULTS_PATH) -> list[InstanceResult]:
    results = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(InstanceResult(**json.loads(line)))
    return results
