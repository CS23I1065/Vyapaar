"""
Genuinely ambiguous, real-world-style requests -- the stress test the
30-scenario corpus can't provide, because every one of those 30 states a
clean budget and a clear category by construction. That's fine for
testing the deterministic downstream pipeline, but it never exercises
the one behaviour interpret() is actually supposed to be good at:
recognizing when it does NOT know something and asking, instead of
guessing.

Each entry states what a CORRECT interpret() call should do -- not a
full ground-truth intent (there often isn't one; that's the point), but
whether it should need clarification, and if so, roughly which field.
`loose_field_match` is a substring/keyword check against
`ambiguous_field` or `clarification_question`, not an exact-match
requirement -- there are several reasonable ways for a model to phrase
"I don't know the budget."

A model that is NOT confused (it correctly extracts a single-quantity
budget) can still get overridden by an overly-rigid field check in
interpret() -- and the reverse failure mode, a model that IS uncertain
but guesses anyway, needs its own coverage too. This corpus exercises
both directions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AmbiguousRequest:
    id: str
    request_text: str
    should_need_clarification: bool
    expected_field_keyword: str | None = None
    # A keyword expected to appear in ambiguous_field or
    # clarification_question, case-insensitive -- e.g. "budget", "colour".
    # None when should_need_clarification is False (nothing to check).
    notes: str = ""


AMBIGUOUS_REQUESTS: tuple[AmbiguousRequest, ...] = (
    AmbiguousRequest(
        "amb-01-no-budget-at-all", "Buy me a plain white t-shirt", True, "budget",
        "The canonical case: no number anywhere in the request.",
    ),
    AmbiguousRequest(
        "amb-02-vague-everything", "Get me something nice for my desk", True, None,
        "No category, no budget, no attributes -- maximally underspecified. Any clarification is acceptable; "
        "the point is that it asks AT ALL rather than picking an item.",
    ),
    AmbiguousRequest(
        "amb-03-conflicting-numbers", "Buy a jacket, around 1500 but honestly up to 4000 if it's good", False, None,
        "NOT expected to need clarification -- 'up to 4000' is a real, if generous, stated ceiling; "
        "target_price_rupees=1500 and budget_total_rupees=4000 both being extractable is the correct read, "
        "not an unresolvable conflict. A model that asks here is being overcautious, not careful.",
    ),
    AmbiguousRequest(
        "amb-04-relative-price-only", "Buy the cheapest phone case you can find", True, "budget",
        "'Cheapest' names no ceiling at all -- there is no number to extract, and 'cheapest' is not one.",
    ),
    AmbiguousRequest(
        "amb-05-budget-stated-differently", "I've got 500 bucks for a t-shirt, nothing fancy", False, None,
        "Should NOT need clarification -- 500 is a clear budget even though it's phrased colloquially "
        "('bucks', not 'rupees'). Tests whether interpret() is fragile to informal phrasing.",
    ),
    AmbiguousRequest(
        "amb-06-typo-riddled", "buy me a palin cottn shrt under 800 rupes", False, None,
        "Should NOT need clarification despite three typos -- the budget and category are still clear to "
        "a human reader, and should be to the model too.",
    ),
    AmbiguousRequest(
        "amb-07-multi-item-request", "Get rice, a pickle, and a notebook, keep the whole thing under 1000", False, None,
        "A genuinely multi-item request the schema has no field for (hard.category is singular). Not scored "
        "as needing clarification -- interpret() reasonably picks the dominant/first item; this is documented "
        "as a known schema limitation, not a pass/fail case, and is included so the failure mode (if any) is "
        "at least observed rather than assumed away.",
    ),
    AmbiguousRequest(
        "amb-08-no-request-content", "buy it", True, None,
        "No product named at all. Should need clarification on more or less everything.",
    ),
)
