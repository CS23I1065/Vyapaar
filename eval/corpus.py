"""
Scenario corpus for the eval harness -- ground truth attached BY
CONSTRUCTION, not inferred after the fact. Every scenario specifies its
own `expected_outcome` and the `ground_truth` structured fields a correct
interpretation of `request_text` should produce; the harness's job is to
report where the system's actual behaviour disagreed with that label,
which is what a false-approval or false-escalation rate actually means.

Format: TOML, not YAML. There is no dependency manifest in this repo and
no pyyaml installed; `tomllib` is stdlib (3.11+) and is already this
project's choice for exactly this kind of structured config (see
merchant_kit/policy_config.py). Adding a new third-party dependency for a
30-entry corpus file was not worth it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent / "corpus" / "scenarios.toml"

ScenarioCategory = Literal["clean", "policy_violating", "ambiguous"]
ExpectedOutcome = Literal["APPROVED", "REJECTED", "REQUIRES_HUMAN_APPROVAL"]


@dataclass(frozen=True)
class ScenarioItem:
    sku: str
    title: str
    category: str
    price_paise: int
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    id: str
    category: ScenarioCategory
    request_text: str
    merchant_id: str
    catalog: tuple[ScenarioItem, ...]
    expected_outcome: ExpectedOutcome
    ground_truth: dict
    # The structured fields a CORRECT interpretation of request_text
    # should produce -- category / required_attributes / budget_total_rupees
    # / etc, in interpret()'s own vocabulary. Comparing a live interpret()
    # call's actual output against this is what makes "interpretation
    # accuracy" a measurable thing rather than an assertion.
    notes: str = ""
    mechanism: str | None = None
    # Only set on scenarios eval/harness.py cannot produce from the real
    # MerchantAgentDriver's ordinary catalog matching alone -- a
    # deliberately over-cap quantity, a duplicate SKU, a substitute item,
    # a merchant-authorization mismatch. Most scenarios need none of
    # this: a real driver against `catalog` is enough.


def _item_from_dict(d: dict) -> ScenarioItem:
    return ScenarioItem(
        sku=d["sku"], title=d["title"], category=d["category"],
        price_paise=int(d["price_paise"]), attributes=d.get("attributes", {}) or {},
    )


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> list[Scenario]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    for raw in data.get("scenario", []):
        if raw["id"] in seen_ids:
            raise ValueError(f"duplicate scenario id {raw['id']!r} in {path}")
        seen_ids.add(raw["id"])
        scenarios.append(
            Scenario(
                id=raw["id"], category=raw["category"], request_text=raw["request_text"],
                merchant_id=raw["merchant_id"],
                catalog=tuple(_item_from_dict(i) for i in raw["catalog"]),
                expected_outcome=raw["expected_outcome"], ground_truth=raw["ground_truth"],
                notes=raw.get("notes", ""), mechanism=raw.get("mechanism"),
            )
        )
    if not scenarios:
        raise ValueError(f"{path} contains no scenarios")
    return scenarios


def corpus_counts(scenarios: list[Scenario]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in scenarios:
        counts[s.category] = counts.get(s.category, 0) + 1
    return counts
