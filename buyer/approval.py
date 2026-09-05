"""
The human approval gate for REQUIRES_HUMAN_APPROVAL decisions.

A human approval must re-sign an amended IntentMandate and re-run
evaluate() from scratch -- never let approval short-circuit the engine.
This module structurally enforces that, rather than just following it
by convention:

- An Approver only ever returns a plain approve/decline boolean -- it
  has no way to inject an outcome, set a Decision, or otherwise touch
  anything evaluate() reads.
- resolve_escalation() is the ONLY thing that builds a new signed intent
  and re-runs evaluate(). There is no code path from "human said yes" to
  execute.py that skips this re-evaluation -- see execute.py's own
  ExecutionRefused guard for the other half of that invariant.
- Re-evaluation is genuine: if the amendment doesn't actually resolve
  every escalated gate (or a completely unrelated gate now fails for
  some other reason), the final Decision can still come back REJECTED or
  even REQUIRES_HUMAN_APPROVAL again. Approval is not a rubber stamp.

Any amendment to the intent changes its canonical hash, which breaks
G0.3 (MANDATE_CHAIN_BROKEN) against the ORIGINAL cart -- a CartMandate
cryptographically promises it was built against those EXACT terms.
That's the chain-binding gate doing exactly its job, not a bug to route
around. The correct fix is that an amendment genuinely requires a
freshly-signed cart from the merchant, bound to the amended intent's
hash, before re-evaluation can mean anything -- so resolve_escalation()
re-negotiates via the same bounded buyer/negotiate.py loop rather than
re-evaluating the stale cart against a new intent.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal, Protocol

from rich.console import Console
from rich.prompt import Confirm

from buyer.explain import summarize_decision
from buyer.negotiate import MerchantDriver, negotiate
from buyer.policy_engine import GATE_SEVERITY, Decision, VerificationContext, evaluate
from mandates.keys import KeyPair
from mandates.schemas import CartMandate, Envelope, IntentMandate, MerchantPolicy
from mandates.sign import sign_mandate


class Approver(Protocol):
    def decide(self, *, decision: Decision, intent: IntentMandate, cart: CartMandate) -> bool: ...


@dataclass(frozen=True)
class CliApprover:
    """A human decides here, so this screen speaks the human's units --
    rupees, and buyer/explain.py's plain-language summary -- not the
    audit log's own paise-denominated engineering vocabulary
    (Check.detail). Keeping those separate is the same discipline
    buyer/explain.py's own module docstring states: the raw codes and
    details stay in the audit trail: what a person approving a real
    purchase reads here must not require decoding them first."""

    console: Console | None = None

    def decide(self, *, decision: Decision, intent: IntentMandate, cart: CartMandate) -> bool:
        console = self.console or Console()
        console.print("[bold]Escalation requires human approval[/bold]")
        console.print(summarize_decision(decision))
        console.print(f"[bold]Cart total:[/bold] Rs{cart.total_paise / 100:.2f}")
        console.print(f"[bold]Signed budget cap:[/bold] Rs{intent.budget.total_paise / 100:.2f}")
        return Confirm.ask("Approve this purchase?", console=console, default=False)


@dataclass(frozen=True)
class ScriptedApprover:
    """Used by the eval harness so it never blocks on stdin -- reads a
    fixed approve/decline decision from the corpus scenario instead."""

    on_escalation: Literal["approve", "decline"]

    def decide(self, *, decision: Decision, intent: IntentMandate, cart: CartMandate) -> bool:
        return self.on_escalation == "approve"


@dataclass(frozen=True)
class EscalationResolution:
    approved: bool
    amended_intent: IntentMandate | None
    amended_envelope: Envelope | None
    final_decision: Decision | None
    audit_note: str


def build_amended_intent(
    original: IntentMandate, decision: Decision, cart: CartMandate, *, now: datetime
) -> IntentMandate:
    """Resolves exactly the ESCALATE-severity reasons present in
    `decision` -- nothing else about the original intent's limits is
    loosened. The human is approving THIS SPECIFIC purchase, not handing
    out a blank check for future ones."""
    escalate_codes = {check.code for check in decision.failed_checks if GATE_SEVERITY[check.code] == "ESCALATE"}

    new_budget = original.budget
    new_soft = original.soft

    if "ESCALATION_THRESHOLD" in escalate_codes:
        # Widen just enough to cover THIS cart and nothing more: the
        # threshold moves to exactly this cart's total, so the approved
        # amount stops tripping G7.1 while anything above it still does.
        #
        # This used to be `cart.total_paise + 1`, a workaround for
        # Budget's old validator requiring escalate_above_paise STRICTLY
        # below total_paise. That constraint is gone (equality now means
        # "no tripwire", see mandates/schemas.py::Budget), so the +1 --
        # which quietly granted one paise more authority than the human
        # approved -- is no longer needed.
        new_total = max(original.budget.total_paise, cart.total_paise)
        new_budget = original.budget.model_copy(
            update={"total_paise": new_total, "escalate_above_paise": cart.total_paise}
        )

    if "AMBIGUOUS_SUBSTITUTION" in escalate_codes:
        new_soft = new_soft.model_copy(update={"substitutions_preapproved": True})

    # LOW_CONFIDENCE_INTERPRETATION (G7.3) is resolved via
    # VerificationContext, not the signed intent -- see
    # resolve_escalation() below, which passes low_confidence_interpretation=False
    # on the re-evaluation once a human has reviewed it.

    validity = original.expires_at - original.issued_at
    return IntentMandate(
        version="1.0",
        mandate_id=f"{original.mandate_id}-amended-{secrets.token_hex(4)}",
        issued_at=now,
        expires_at=now + validity,
        principal_kid=original.principal_kid,
        agent_kid=original.agent_kid,
        request_text=original.request_text,
        hard=original.hard,
        soft=new_soft,
        budget=new_budget,
        allowed_merchants=original.allowed_merchants,
    )


def resolve_escalation(
    decision: Decision,
    intent: IntentMandate,
    cart: CartMandate,
    merchant_policy: MerchantPolicy,
    ctx: VerificationContext,
    *,
    approver: Approver,
    human_keypair: KeyPair,
    now: datetime,
    merchant_driver: MerchantDriver,
) -> EscalationResolution:
    """`merchant_driver` is required, not optional: the amended intent
    always has a different hash than the original (any real field change
    does), so the original cart's chain binding to it is necessarily
    broken -- see this module's docstring. Re-negotiating a fresh,
    correctly-bound cart via the same bounded loop buyer/negotiate.py
    already uses is the only way re-evaluation means anything, not an
    extra step bolted on."""
    if decision.outcome != "REQUIRES_HUMAN_APPROVAL":
        raise ValueError(f"resolve_escalation called on a {decision.outcome} decision, not REQUIRES_HUMAN_APPROVAL")

    if not approver.decide(decision=decision, intent=intent, cart=cart):
        return EscalationResolution(False, None, None, None, "human declined")

    amended_intent = build_amended_intent(intent, decision, cart, now=now)
    amended_envelope = sign_mandate(amended_intent, human_keypair)

    negotiation = negotiate(amended_intent, merchant_driver=merchant_driver)
    if negotiation.outcome != "cart_received":
        return EscalationResolution(
            True, amended_intent, amended_envelope, None,
            f"human approved, but re-negotiation against the amended terms did not produce a cart: {negotiation.reason}",
        )

    new_cart = CartMandate.model_validate(negotiation.cart_envelope.payload)
    resolved_ctx = replace(
        ctx, intent_envelope=amended_envelope, cart_envelope=negotiation.cart_envelope, low_confidence_interpretation=False
    )
    final_decision = evaluate(amended_intent, new_cart, merchant_policy, resolved_ctx, now)

    return EscalationResolution(
        True, amended_intent, amended_envelope, final_decision,
        f"human approved; re-negotiated and re-evaluated from scratch -> {final_decision.outcome}",
    )
