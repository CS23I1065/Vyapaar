"""
Gate code -> plain sentence, at the user boundary ONLY.

Every failure mode in this system is precise and well-coded, and every
one of them used to surface as a stack trace or a term of art: an
unhandled ValueError for a missing budget, a bare SystemExit(1) for a
robots block, `ExecutionRefused`, or a Decision carrying codes like
PROVENANCE_INSUFFICIENT and MANDATE_CHAIN_BROKEN. Those are excellent
ENGINEERING vocabulary and they are unchanged in the audit trail. But
MANDATE_CHAIN_BROKEN is not an explanation a shopper can act on. Someone
asked to buy a shirt and got a term from a protocol they have never
heard of.

This is the last mile of the whole project. Every gate is built on the
idea that refusing is a FEATURE -- but a refusal the user cannot
understand reads as the thing being broken, and "it just failed" is what
they will remember rather than the careful reasoning behind it.

Three rules this module holds to:

1. **Translation is additive, never substitutive.** Nothing here is
   written into the audit log, and no gate code is renamed. audit/ keeps
   the codes; the user gets the sentence. If the two ever disagree the
   code is the truth.
2. **Say what happened in the USER's units, and what would change it.**
   "The cheapest match was Rs620, which is over your Rs500 limit" beats
   "BUDGET_EXCEEDED" because it names the number, the limit, and the gap
   between them.
3. **Never invent a reason.** An unknown code falls back to the gate's
   own detail string rather than a reassuring guess. A translation layer
   that smooths over a code it does not recognise is worse than no
   translation layer, because it sounds authoritative while saying
   nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from buyer.policy_engine import Check, Decision


def rupees(paise: int | None) -> str:
    if paise is None:
        return "an unknown amount"
    return f"Rs{paise / 100:,.2f}".replace(".00", "")


@dataclass(frozen=True)
class Explanation:
    code: str
    headline: str          # what happened, one sentence, user's units
    what_would_help: str   # what would change the outcome
    gate: str              # kept so a curious user can be shown the code too

    def as_text(self) -> str:
        return f"{self.headline} {self.what_would_help}".strip()


def _ev(check: Check, key: str):
    return check.evidence.get(key)


def explain_check(check: Check) -> Explanation:
    """Translate one failed Check. Pure -- takes the Check's own recorded
    evidence and never recomputes anything, so the sentence and the audit
    row can never drift apart."""
    code = check.code

    if code == "BUDGET_EXCEEDED":
        return Explanation(
            code,
            f"The cheapest match came to {rupees(_ev(check, 'total_paise'))}, "
            f"which is over your {rupees(_ev(check, 'limit_paise'))} limit.",
            "Raising the limit, or waiting for the price to drop, would let this through.",
            check.gate,
        )

    if code == "PER_ITEM_CAP_EXCEEDED":
        cap = _ev(check, "per_item_cap_paise")
        return Explanation(
            code,
            f"At least one item costs more than your {rupees(cap)} per-item cap.",
            "You set a per-item cap as well as a total; raising it would allow a pricier single item.",
            check.gate,
        )

    if code == "QUANTITY_CAP_EXCEEDED":
        return Explanation(
            code,
            f"The shop tried to sell more units than the {_ev(check, 'max_quantity')} you approved.",
            "Approving a larger quantity would allow it.",
            check.gate,
        )

    if code == "CATALOG_PRICE_MISMATCH":
        lines = _ev(check, "offending_lines") or [{}]
        worst = max(lines, key=lambda o: o.get("overcharge_paise", 0))
        return Explanation(
            code,
            f"The shop charged {rupees(worst.get('charged_paise'))} for something its own "
            f"published price list says costs {rupees(worst.get('published_paise'))}.",
            "That's the shop contradicting its own price list, not a limit you set.",
            check.gate,
        )

    if code == "CATALOG_ITEM_UNKNOWN":
        return Explanation(
            code,
            "The shop tried to sell something that isn't in the price list it publishes.",
            "With nothing published to check the price against, I stopped.",
            check.gate,
        )

    if code == "CART_SHAPE_INVALID":
        return Explanation(
            code,
            "The basket the shop sent back was malformed -- repeated items, or far more lines than a real order.",
            "I stopped rather than guess what was meant.",
            check.gate,
        )

    if code == "PROVENANCE_INSUFFICIENT":
        return Explanation(
            code,
            "This shop's price couldn't be verified against a signed source, so I didn't buy.",
            "I only spend against prices a shop has cryptographically signed for, not ones read off a page.",
            check.gate,
        )

    if code == "MANDATE_CHAIN_BROKEN":
        return Explanation(
            code,
            "The shop's offer didn't match what you approved, so I stopped.",
            "This usually means the terms changed between your approval and the offer; re-running will re-ask.",
            check.gate,
        )

    if code == "SIGNATURE_INVALID":
        return Explanation(
            code,
            "I couldn't confirm the offer really came from this shop.",
            "The signature didn't check out, so I treated the offer as untrustworthy and stopped.",
            check.gate,
        )

    if code == "MANDATE_EXPIRED":
        return Explanation(
            code,
            "Your approval had expired by the time the shop replied.",
            "Approvals are deliberately short-lived; approving again will start a fresh one.",
            check.gate,
        )

    if code == "MERCHANT_NOT_AUTHORIZED":
        return Explanation(
            code,
            f"The offer came from {_ev(check, 'merchant_id') or 'a shop'}, which isn't one you authorized.",
            "Adding that shop to the approved list would allow it.",
            check.gate,
        )

    if code == "AGENT_IDENTITY_UNVERIFIED":
        return Explanation(
            code,
            "I couldn't prove my own identity to the shop on this run.",
            "This configuration requires verified agent identity before spending.",
            check.gate,
        )

    if code == "CART_ARITHMETIC_MISMATCH":
        return Explanation(
            code,
            f"The shop's total ({rupees(_ev(check, 'declared_total_paise'))}) doesn't match "
            f"what its own line items add up to ({rupees(_ev(check, 'computed_total_paise'))}).",
            "I don't pay a total I can't reproduce.",
            check.gate,
        )

    if code == "HARD_CONSTRAINT_VIOLATED":
        return Explanation(
            code,
            "What the shop offered isn't what you asked for.",
            "It missed a requirement you marked as essential rather than a preference.",
            check.gate,
        )

    if code == "CATEGORY_NOT_ALLOWED":
        return Explanation(
            code,
            "The basket included something from a category you excluded.",
            "Removing that exclusion, or a different shop, would be needed.",
            check.gate,
        )

    if code == "SUBSTITUTION_OUT_OF_TOLERANCE":
        return Explanation(
            code,
            f"The shop offered a substitute priced further from your target than the "
            f"{_ev(check, 'tolerance_pct')}% you allowed.",
            "Widening the substitution tolerance would let a substitute like this through.",
            check.gate,
        )

    if code == "MERCHANT_POLICY_CONFLICT":
        return Explanation(
            code,
            "The shop's own rules don't allow this order.",
            "This is the shop's limit, not yours -- a smaller order may go through.",
            check.gate,
        )

    if code == "ESCALATION_THRESHOLD":
        return Explanation(
            code,
            f"This came to {rupees(_ev(check, 'total_paise'))}, above the "
            f"{rupees(_ev(check, 'effective_threshold_paise'))} you asked to be checked with about.",
            "You asked to review anything over that amount before it's bought.",
            check.gate,
        )

    if code == "AMBIGUOUS_SUBSTITUTION":
        return Explanation(
            code,
            "The shop offered a substitute rather than the exact item.",
            "You didn't pre-approve substitutions, so I'm checking with you first.",
            check.gate,
        )

    if code == "LOW_CONFIDENCE_INTERPRETATION":
        return Explanation(
            code,
            "I wasn't confident I understood part of your request.",
            "Rather than guess with your money, I'm asking.",
            check.gate,
        )

    if code == "ENGINE_ERROR":
        return Explanation(
            code,
            "Something went wrong while checking this purchase, so I stopped.",
            "I refuse by default when a check can't complete -- nothing was bought.",
            check.gate,
        )

    # Deliberately does NOT invent a reason.
    return Explanation(code, f"I stopped because of a check I can't phrase plainly ({code}).", check.detail, check.gate)


def explain_decision(decision: Decision) -> list[Explanation]:
    """All failed checks, most-blocking first. The blocking code leads,
    because that is the one that actually decided the outcome."""
    failures = list(decision.failed_checks)
    failures.sort(key=lambda c: (c.code != decision.blocking_code,))
    return [explain_check(c) for c in failures]


def summarize_decision(decision: Decision) -> str:
    """A short paragraph a shopper can read. APPROVED gets a sentence
    too -- silence on success is how a user learns to distrust a system
    that only speaks when it fails."""
    if decision.outcome == "APPROVED":
        return "Approved: this purchase is inside everything you authorized."

    explanations = explain_decision(decision)
    if not explanations:
        return f"{decision.outcome} (no failing check recorded)."

    lead = explanations[0]
    if decision.outcome == "REQUIRES_HUMAN_APPROVAL":
        opener = "I need your go-ahead before buying this."
    else:
        opener = "I didn't buy this."

    body = [f"{opener} {lead.as_text()}"]
    for extra in explanations[1:]:
        body.append(f"Also: {extra.headline}")
    return "\n".join(body)
