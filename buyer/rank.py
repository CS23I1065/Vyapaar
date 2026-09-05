"""
Buyer-side preference scoring. This is the scorer the interpret prompt
has been promising the whole time and that did not exist.

What was wrong
--------------
`preferred_brands` had ZERO readers anywhere in the repo -- the schema
definition and the line in interpret.py that populated it, and nothing
else. `target_price_paise` was read in exactly one place, G5.1, and only
to bound how far a *substitute* may drift; it never influenced which
product got chosen. Meanwhile the actual selection was `min(price)` over
whatever matched the hard constraints.

So "I'd like a Fabindia shirt, around Rs1500, up to Rs2000" got you the
cheapest shirt in the catalog: brand ignored, target ignored -- and the
system recorded both preferences in a cryptographically signed document
while disregarding them. Worse, the interpret prompt told the model that
a soft preference "only affects scoring/tie-breaking," so the LLM
dutifully sorted the user's wishes into a bucket labelled "this will
influence the choice" that was wired to nothing.

Why it lives on the BUYER side
------------------------------
The merchant must keep choosing from its own catalog blind to the
buyer's wallet -- merchant_agent/upsell.py does not even accept an
IntentMandate parameter, and that asymmetry is load-bearing: a merchant
that learned the budget would tune every "alternative" to land just
under it. So the merchant proposes candidates from its own catalog and
margin rules, and ranking happens here, where the budget already lives.

Purity
------
Every function here is pure: no I/O, no clock, no LLM. Scoring never
decides whether money moves -- it only orders candidates that
buyer/policy_engine.py has already independently approved. A high score
cannot rescue a cart the engine rejected, and a low one cannot block a
cart the engine approved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from mandates.schemas import SoftPreferences

# Weights are explicit constants rather than magic numbers inline so the
# ranking can be explained to a user in the same terms it was computed
# in ("matched your preferred brand", "closest to your target price").
BRAND_MATCH_WEIGHT = 50.0
TARGET_PRICE_WEIGHT = 30.0
IN_STOCK_WEIGHT = 20.0
# Cheapness is the TIE-BREAK, not the objective. It is scored last and
# small on purpose: making price dominant is how a scorer quietly
# degenerates back into min(price), which is the behaviour this module
# exists to replace.
CHEAPNESS_WEIGHT = 5.0


@dataclass(frozen=True)
class Candidate:
    """A rankable offer. Deliberately not a CatalogItem or a CartLine:
    the same scorer ranks items within one merchant's catalog AND carts
    across several merchants, and neither of those types can represent
    both."""

    merchant_id: str
    sku: str
    title: str
    category: str
    unit_price_paise: int
    in_stock: bool = True
    attributes: Mapping[str, str] = field(default_factory=dict)
    total_paise: int | None = None  # cart total when ranking whole carts

    @property
    def brand(self) -> str | None:
        return self.attributes.get("brand")

    @property
    def effective_price_paise(self) -> int:
        return self.total_paise if self.total_paise is not None else self.unit_price_paise


@dataclass(frozen=True)
class ScoreComponent:
    name: str
    points: float
    detail: str


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    score: float
    components: tuple[ScoreComponent, ...]

    def rationale(self) -> str:
        """One line per component, in the user's own terms. This is what
        makes "why did you buy this?" answerable with something better
        than a gate code."""
        return "; ".join(c.detail for c in self.components if c.points != 0.0) or "no preferences to match"


def _normalize_brand(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _brand_matches(candidate_brand: str | None, preferred: list[str]) -> str | None:
    """Returns the matched preference, or None. Normalized and
    substring-tolerant in both directions because real catalog brand
    strings are messy: Chumbak's own JSON-LD publishes the brand as
    'chumbakdesign', which an exact match against a user typing 'Chumbak'
    would miss entirely."""
    if not candidate_brand or not preferred:
        return None
    actual = _normalize_brand(candidate_brand)
    if not actual:
        return None
    for want in preferred:
        wanted = _normalize_brand(want)
        if wanted and (wanted in actual or actual in wanted):
            return want
    return None


def score_candidate(candidate: Candidate, soft: SoftPreferences) -> ScoredCandidate:
    """Pure. Higher is better. Components are additive and each is
    independently explainable."""
    components: list[ScoreComponent] = []

    matched = _brand_matches(candidate.brand, list(soft.preferred_brands))
    if soft.preferred_brands:
        if matched:
            components.append(
                ScoreComponent("brand", BRAND_MATCH_WEIGHT, f"matches your preferred brand {matched!r}")
            )
        else:
            components.append(
                ScoreComponent("brand", 0.0, f"not one of your preferred brands ({candidate.brand or 'no brand listed'})")
            )

    target = soft.target_price_paise
    if target is not None and target > 0:
        price = candidate.effective_price_paise
        # Distance normalized by the target itself, so "Rs100 off a
        # Rs200 target" is penalised far more than "Rs100 off Rs5000".
        # Clamped at 1.0 so an absurdly-priced candidate scores zero here
        # rather than dragging the total negative and swamping the other
        # components.
        distance = min(abs(price - target) / target, 1.0)
        points = TARGET_PRICE_WEIGHT * (1.0 - distance)
        rupees = price / 100
        target_rupees = target / 100
        components.append(
            ScoreComponent(
                "target_price", points,
                f"Rs{rupees:.2f} against your target of Rs{target_rupees:.2f}",
            )
        )

    if candidate.in_stock:
        components.append(ScoreComponent("stock", IN_STOCK_WEIGHT, "in stock"))
    else:
        components.append(ScoreComponent("stock", 0.0, "out of stock"))

    # Cheapness, scaled within a fixed reference so it stays a tie-break.
    # Never allowed to exceed CHEAPNESS_WEIGHT.
    price = max(candidate.effective_price_paise, 1)
    components.append(
        ScoreComponent("price", CHEAPNESS_WEIGHT / (1.0 + price / 100_000), f"Rs{price / 100:.2f}")
    )

    return ScoredCandidate(
        candidate=candidate,
        score=sum(c.points for c in components),
        components=tuple(components),
    )


def rank(candidates: list[Candidate], soft: SoftPreferences) -> list[ScoredCandidate]:
    """Best first. Ties broken deterministically by price then sku, so
    the same inputs always produce the same order -- the eval harness
    replays this and a nondeterministic ranking would make ablation
    comparisons meaningless."""
    scored = [score_candidate(c, soft) for c in candidates]
    return sorted(
        scored,
        key=lambda s: (-s.score, s.candidate.effective_price_paise, s.candidate.sku),
    )


def best(candidates: list[Candidate], soft: SoftPreferences) -> ScoredCandidate | None:
    ranked = rank(candidates, soft)
    return ranked[0] if ranked else None
