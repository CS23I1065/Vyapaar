"""
Turns an ambiguous natural-language purchase request into an explicit,
unsigned IntentMandate draft. The LLM proposes hard/soft/budget field
VALUES; it decides nothing, and everything downstream of a human's own
signature is deterministic.

Structurally: the output of interpret() is a plain IntentMandate object,
which has no signature at all until a human calls mandates.sign_mandate()
on it themselves. mandates/schemas.py's G0.2 gate (signature verification)
means an unsigned draft cannot pass evaluate() under any circumstance --
so there is no path from "the LLM said this" to money moving that skips
human sign-off.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from mandates.schemas import Budget, HardConstraints, IntentMandate, SoftPreferences

DEFAULT_VALIDITY = timedelta(hours=1)
DEFAULT_SUBSTITUTION_TOLERANCE_PCT = 10.0
DEFAULT_MAX_QUANTITY = 1

# The canonical question for the one thing this module must never guess.
BUDGET_CLARIFICATION_QUESTION = "What's the most you want to spend on this?"

# No derived ESCALATE_FRACTION here. Deriving one (e.g. 0.8 x total_paise)
# turns one stated ceiling into two -- a hard REJECT at the real budget
# and an ESCALATE just under it -- but good shopping USES the budget: the
# best in-budget item usually lands in the top 20%, exactly the band that
# would always trip that second ceiling. It buys no safety the real
# ceiling didn't already provide, and costs an expensive interrupt
# (resolve_escalation() has to re-sign and re-negotiate a fresh cart).
#
# escalate_above_paise defaults to total_paise instead, meaning "no
# tripwire" (see mandates/schemas.py::Budget). The tripwire survives as a
# capability the human sets deliberately at the review screen -- never
# derived from a number the user gave for a different purpose.

_SYSTEM_PROMPT = (
    "You turn a shopper's natural-language purchase request into explicit "
    "structured constraints for an automated buying agent. Separate HARD "
    "requirements (must-have: category, specific attributes like colour) "
    "from SOFT preferences (brand preference, an approximate target price) "
    "-- a hard requirement, if violated, means the purchase is REJECTED "
    "outright; a soft preference only affects scoring/tie-breaking, never "
    "a hard block.\n\n"
    "Set needs_clarification=true ONLY when a preference is genuinely "
    "load-bearing (the request is meaningfully ambiguous about something "
    "that would change what gets bought) AND cannot be reasonably "
    "inferred -- not for minor details a sensible default resolves fine.\n\n"
    "NEVER invent a budget. If the shopper did not state how much they "
    "are willing to spend, omit budget_total_rupees entirely (or return "
    "null) -- do not estimate one from typical prices, and do not infer "
    "one from a target price. An unstated budget is the single clearest "
    "case for asking rather than guessing.\n\n"
    "You never decide whether a purchase is approved. You only propose "
    "structured constraints for a human to review and sign, and a "
    "separate deterministic policy engine to enforce. Prices you output "
    "are in rupees (not paise) -- conversion happens in code.\n\n"
    "If the shopper mentions a spending tripwire ('alert me if it costs "
    "more than 400', 'check with me above Rs300'), extract "
    "escalate_above_rupees as that threshold in rupees. If they say "
    "alternatives or substitutes are fine ('close alternatives ok', "
    "'similar is fine'), set substitutions_ok=true. If they mention a "
    "tolerance percentage ('within 10%', 'up to 15% more'), extract "
    "substitution_tolerance_pct. Omit any of these if not mentioned -- "
    "the system uses safe defaults for anything not stated."
)

_CATALOG_CATEGORY_NOTE = (
    "\n\nThis merchant's catalog uses EXACTLY these category names: {categories}. "
    "You must set `category` to the single closest match from this exact list -- "
    "never invent your own word for it, even if a more specific or more natural "
    "term exists. If the shopper's request doesn't clearly fit any of these, pick "
    "the closest one rather than leaving it unset."
)

_CATALOG_ATTRIBUTE_NOTE_WITH_KEYS = (
    "\n\nThis merchant's catalog tracks these product attribute keys, PER "
    "CATEGORY: {breakdown}. When you record a `required_attributes` entry, "
    "its `key` MUST be one of the keys listed for the category you chose "
    "above -- never a key that belongs to a different category's list, and "
    "never one that isn't listed anywhere. A descriptive word from the "
    "shopper's request that doesn't correspond to a listed key for that "
    "category (colour, material, weight, or any other detail that category's "
    "catalog entries don't carry as a field) is not an enforceable hard "
    "constraint here -- leave it out of `required_attributes` rather than "
    "inventing a key for it."
)

_CATALOG_ATTRIBUTE_NOTE_NO_KEYS = (
    "\n\nThis merchant's catalog does not track any structured product "
    "attributes beyond category and price -- it has no attribute keys at "
    "all. Do NOT invent a `required_attributes` entry for descriptive words "
    "in the shopper's request (colour, material, weight, style, and so on); "
    "leave `required_attributes` empty. Those details are still present in "
    "the request text itself for a human reviewer to judge -- they are just "
    "not fields this catalog can enforce."
)

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "category": {"type": "STRING"},
        "required_attributes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"key": {"type": "STRING"}, "value": {"type": "STRING"}},
                "required": ["key", "value"],
            },
        },
        "excluded_categories": {"type": "ARRAY", "items": {"type": "STRING"}},
        "preferred_brands": {"type": "ARRAY", "items": {"type": "STRING"}},
        "target_price_rupees": {"type": "NUMBER"},
        "budget_total_rupees": {"type": "NUMBER"},
        "budget_per_item_rupees": {"type": "NUMBER"},
        "max_quantity": {"type": "INTEGER"},
        "needs_clarification": {"type": "BOOLEAN"},
        "ambiguous_field": {"type": "STRING"},
        "clarification_question": {"type": "STRING"},
        # Preference fields read by AIReviewer -- all optional, safe defaults used if absent.
        "escalate_above_rupees": {"type": "NUMBER"},
        # Extract if shopper states a tripwire: "alert me above Rs400", "check with me above 300".
        # Absent -> no tripwire (escalate_above_paise == total_paise, safe default).
        "substitutions_ok": {"type": "BOOLEAN"},
        # Extract if shopper says alternatives/substitutes are fine.
        # Absent -> False (G7.2 will ask at escalation time if a substitute actually appears).
        "substitution_tolerance_pct": {"type": "NUMBER"},
        # Extract if shopper states a tolerance: "within 10%", "up to 15% more".
        # Absent -> None (keep DEFAULT_SUBSTITUTION_TOLERANCE_PCT).
    },
    # budget_total_rupees is deliberately NOT required. It used to be, and
    # interpret() raised a bare ValueError when it came back missing --
    # so an ordinary request like "buy me a plain white t-shirt" had
    # exactly two possible outcomes, both bad: the model invented a
    # spending limit which a human then cryptographically SIGNED, or the
    # run died on an unhandled exception. The first is worse. A budget is
    # the most consequential field in the system -- it is the ceiling on
    # how much of someone's money an agent may spend -- and every gate
    # downstream would faithfully enforce a number nobody chose. All the
    # machinery guarding the limit does not matter if the limit itself
    # was hallucinated.
    "required": ["needs_clarification"],
}


@dataclass(frozen=True)
class InterpretResult:
    draft_intent: IntentMandate | None
    # UNSIGNED -- a human must sign before use. `None` when the request
    # did not state a budget: there is no honest IntentMandate to draft
    # without one, and drafting a placeholder is how an invented number
    # gets anchored in front of someone who is about to sign it.
    needs_clarification: bool
    ambiguous_field: str | None
    clarification_question: str | None
    raw_response: dict
    budget_stated: bool = True
    # False means the shopper never said a number. The review screen must
    # show this: a prefilled budget the user rubber-stamps is barely
    # better than an invented one, so an inferred figure has to be
    # visibly labelled as inferred, not stated.

    @property
    def is_signable(self) -> bool:
        return self.draft_intent is not None


def _rupees_to_paise(value) -> int | None:
    if value is None:
        return None
    try:
        return round(float(value) * 100)
    except (TypeError, ValueError):
        return None


def interpret(
    request_text: str,
    *,
    provider,
    model: str,
    principal_kid: str,
    agent_kid: str,
    allowed_merchants: list[str] | None,
    now: datetime,
    validity: timedelta = DEFAULT_VALIDITY,
    call_site: str = "buyer.interpret",
    available_categories: list[str] | None = None,
    available_attribute_keys: dict[str, list[str]] | None = None,
) -> InterpretResult:
    """`available_categories`, when given, is a merchant catalog's own
    published category taxonomy -- constrains the model's `category`
    output to an enum over exactly those values, instead of letting it
    freely name the product in its own words. This closes the gap where
    a model describes a product as "basmati rice" while the catalog's
    own field says "grains": merchant_agent/driver.py matches categories
    by exact equality, so an ungrounded guess finds zero matches even
    when the model understood the request perfectly. Omitted (the
    default) preserves the original free-text behaviour -- no caller
    that doesn't pass this is affected.

    `available_attribute_keys` is the same idea one level down:
    `{category: [attribute keys that category's catalog items carry]}`.
    Keyed by category rather than a flat list, because a single
    schema-level enum can't be conditioned on another field's value --
    a flat union across categories would let the model attach an
    apparel-only key to a groceries item. Per-category detail goes into
    the prompt instead: the model is told which keys are valid for
    whichever category it chose, and warned off borrowing a different
    category's key. Without this, the model treats almost any
    descriptive word in a request ("plain", "basmati", "wired") as a
    structured hard constraint, and merchant_agent/driver.py's exact
    `c.attributes.get(key) == value` match then declines the purchase
    against a catalog that never carried that key at all. A category
    whose catalog entries carry zero attribute keys should still appear
    in the dict, mapped to `[]` -- that's a real fact about the
    category, distinct from "no info given" (`None`, the default,
    which preserves the original free-text behaviour)."""
    schema = _RESPONSE_SCHEMA
    system = _SYSTEM_PROMPT
    if available_categories or available_attribute_keys is not None:
        schema = json.loads(json.dumps(_RESPONSE_SCHEMA))  # cheap deep copy, schema is JSON-safe
    if available_categories:
        schema["properties"]["category"] = {"type": "STRING", "enum": list(available_categories)}
        system = system + _CATALOG_CATEGORY_NOTE.format(categories=", ".join(available_categories))
    if available_attribute_keys is not None:
        all_keys = sorted({key for keys in available_attribute_keys.values() for key in keys})
        key_schema = schema["properties"]["required_attributes"]["items"]["properties"]["key"]
        if all_keys:
            key_schema["enum"] = all_keys
            breakdown = ", ".join(
                f"{category} -> {', '.join(keys) if keys else '(no attribute keys tracked)'}"
                for category, keys in sorted(available_attribute_keys.items())
            )
            system = system + _CATALOG_ATTRIBUTE_NOTE_WITH_KEYS.format(breakdown=breakdown)
        else:
            system = system + _CATALOG_ATTRIBUTE_NOTE_NO_KEYS

    completion = provider.complete(
        model=model,
        system=system,
        messages=[{"role": "user", "text": request_text}],
        response_schema=schema,
        max_output_tokens=800,
        call_site=call_site,
    )
    data = json.loads(completion.text)

    hard = HardConstraints(
        category=data.get("category") or None,
        required_attributes={item["key"]: item["value"] for item in data.get("required_attributes", []) if item.get("key")},
        excluded_categories=[c for c in data.get("excluded_categories", []) if c],
    )
    soft = SoftPreferences(
        preferred_brands=[b for b in data.get("preferred_brands", []) if b],
        target_price_paise=_rupees_to_paise(data.get("target_price_rupees")),
        substitution_tolerance_pct=DEFAULT_SUBSTITUTION_TOLERANCE_PCT,
    )

    budget_total_paise = _rupees_to_paise(data.get("budget_total_rupees"))
    if budget_total_paise is None and int(data.get("max_quantity") or DEFAULT_MAX_QUANTITY) <= 1:
        # A single-unit purchase's per-item figure IS the total: a stated
        # amount can land in budget_per_item_rupees instead of
        # budget_total_rupees when the model treats them as synonyms at
        # quantity 1. Only safe for quantity<=1 -- at any higher quantity
        # a per-item figure is NOT the total, and treating it as one
        # would read a bigger budget than was ever stated.
        budget_total_paise = _rupees_to_paise(data.get("budget_per_item_rupees"))
    if budget_total_paise is None or budget_total_paise <= 0:
        # Do not guess, ASK. This is the canonical needs_clarification
        # case, and asking costs one exchange at the pre-signature review
        # -- the one place the design permits an interrupt anyway, so it
        # is effectively free.
        return InterpretResult(
            draft_intent=None,
            needs_clarification=True,
            ambiguous_field="budget_total_rupees",
            clarification_question=(
                data.get("clarification_question") or BUDGET_CLARIFICATION_QUESTION
            ),
            raw_response=data,
            budget_stated=False,
        )

    per_item_paise = _rupees_to_paise(data.get("budget_per_item_rupees")) or budget_total_paise
    per_item_paise = min(per_item_paise, budget_total_paise)

    budget = Budget(
        total_paise=budget_total_paise,
        per_item_paise=per_item_paise,
        max_quantity=int(data.get("max_quantity") or DEFAULT_MAX_QUANTITY),
        # No tripwire by default -- see the ESCALATE_FRACTION note above.
        escalate_above_paise=budget_total_paise,
    )

    draft_intent = IntentMandate(
        version="1.0",
        mandate_id=str(uuid.uuid4()),
        issued_at=now,
        expires_at=now + validity,
        principal_kid=principal_kid,
        agent_kid=agent_kid,
        request_text=request_text,
        hard=hard,
        soft=soft,
        budget=budget,
        allowed_merchants=allowed_merchants,
    )

    return InterpretResult(
        draft_intent=draft_intent,
        needs_clarification=bool(data.get("needs_clarification", False)),
        ambiguous_field=data.get("ambiguous_field") or None,
        clarification_question=data.get("clarification_question") or None,
        raw_response=data,
        budget_stated=True,
    )


def enrich_request_text(original: str, question: str, answer: str) -> str:
    """Folds a clarifying answer back into the request text, so the
    re-run of interpret() sees ONE coherent request rather than a patched
    result object.

    Re-running the interpreter (instead of splicing the answer into the
    draft) matters: the answer can legitimately change more than the
    field that was asked about -- "under Rs2000, and it must be cotton"
    carries a hard constraint too. Splicing would silently drop it. It
    also keeps the invariant that every IntentMandate is a whole
    interpretation of a whole request, which is what request_text
    promises to an auditor reading the mandate later."""
    return f"{original.strip()}\n\n[clarification] {question.strip()} {answer.strip()}"
