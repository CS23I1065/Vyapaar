"""
Wire-format schemas for the signed authorization chain: Intent -> Cart ->
Payment, AP2-shaped but implemented as Ed25519 detached signatures over
RFC 8785 canonical JSON -- not full W3C VC/DID.

Money is `int` paise everywhere. No floats, no Decimal, at any money
boundary in this file -- a float rupee amount here is a bug.

Field-level (not item-level) provenance on CartLine is deliberate: a
product can have a schema.org-sourced title but an LLM-inferred price.
Item-level tagging would let an inferred price ride on a trusted title's
coattails past the policy engine's provenance gate (G1.1).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenModel(BaseModel):
    """Base for every typed mandate/value object below. Immutable on
    purpose: the policy engine (buyer/policy_engine.py) verifies a
    signature against an Envelope's raw payload dict, then evaluates gates
    against separately-typed IntentMandate/CartMandate objects. If those
    objects were mutable, a bug (or a deliberately crafted attack) could
    mutate them in place *after* signature verification but *before* the
    budget/provenance/constraint gates run -- passing a valid signature
    check while every other gate sees different data than what was
    actually signed. Freezing closes that gap at the type level instead of
    relying on every future caller to remember not to mutate in place.

    Envelope itself is deliberately NOT frozen -- its `payload` dict is the
    stand-in for "bytes on the wire," and tests simulate tampering by
    mutating it directly, which is the correct model of a real attacker
    modifying a JSON payload in transit.

    extra="forbid" is the second half of the same argument, and it is a
    change of DEFAULT, not just an added check. Pydantic's default is
    extra="ignore", which means an unrecognised field in a signed payload
    was silently DROPPED at validation: the signature covered bytes that
    the typed object -- and therefore every policy gate -- never saw. The
    engine would then evaluate a strictly smaller document than the one
    that was actually signed, and report APPROVED on it. Forbidding
    extras makes an unrecognised field a loud rejection instead of a
    shrug, which is the same "approve only what you can affirmatively
    account for" posture as the G8.x band in buyer/policy_engine.py."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def _require_tz_aware(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError(
            "datetime must be timezone-aware (use UTC) -- naive datetimes "
            "compared against each other silently do the wrong thing, and "
            "this project's expiry gate (G0.1) depends on that comparison "
            "being correct."
        )
    return v


class Provenance(FrozenModel):
    source_url: str
    extraction_method: Literal[
        "schema_org", "microdata", "rdfa", "llm_inferred", "merchant_signed"
    ]
    signature_verified: bool
    extracted_at: datetime
    confidence: float | None = None  # populated only for llm_inferred

    _v_extracted_at = field_validator("extracted_at")(_require_tz_aware)

    @model_validator(mode="after")
    def _confidence_only_for_llm_inferred(self) -> "Provenance":
        if self.extraction_method != "llm_inferred" and self.confidence is not None:
            raise ValueError(
                f"confidence should only be set for extraction_method="
                f"'llm_inferred', got {self.extraction_method!r} with "
                f"confidence={self.confidence}"
            )
        return self


class Budget(FrozenModel):
    total_paise: int = Field(ge=0)
    per_item_paise: int = Field(ge=0)
    max_quantity: int = Field(ge=1)
    escalate_above_paise: int = Field(ge=0)
    currency: Literal["INR"] = "INR"

    @model_validator(mode="after")
    def _escalate_above_cannot_exceed_total(self) -> "Budget":
        # This validator used to require escalate_above_paise < total_paise
        # strictly, on the theory that a threshold which can never fire is
        # a silent misconfiguration. That was wrong, and it was enforcing a
        # product decision I no longer think is right: it made
        # "spend up to Rs500 without asking me" INEXPRESSIBLE. Every
        # IntentMandate was born with a mandatory interrupt built into it,
        # so a completely in-policy purchase touched the human twice --
        # sign, then escalate -- with a full merchant re-negotiation in
        # between (buyer/approval.py::resolve_escalation). An agent that
        # interrupts on every in-budget purchase is strictly worse than
        # the user buying the thing themselves.
        #
        # escalate_above_paise == total_paise is now the DEFAULT and means
        # "no tripwire": G4.3 already rejects anything over total, so G7.1
        # simply never fires. The tripwire survives as a deliberate
        # CAPABILITY -- "spend up to Rs5000 but check with me above
        # Rs2000" is a genuinely useful thing to be able to say -- but the
        # human sets it at review time. It is never derived from a number
        # they gave for a different purpose.
        #
        # Above total_paise is still a real config error: it would be a
        # threshold sitting in unreachable territory beyond a hard cap.
        if self.escalate_above_paise > self.total_paise:
            raise ValueError(
                f"escalate_above_paise ({self.escalate_above_paise}) must be "
                f"<= total_paise ({self.total_paise}) -- a threshold above "
                f"the hard budget cap is unreachable, since G4.3 rejects "
                f"anything over total_paise before G7.1 could fire"
            )
        return self


class HardConstraints(FrozenModel):
    category: str | None = None
    required_attributes: dict[str, str] = Field(default_factory=dict)
    excluded_categories: list[str] = Field(default_factory=list)


class SoftPreferences(FrozenModel):
    preferred_brands: list[str] = Field(default_factory=list)
    target_price_paise: int | None = Field(default=None, ge=0)
    substitution_tolerance_pct: float = Field(ge=0.0, le=100.0)
    # Set in one of TWO places, both of them a real human choice:
    #
    #   1. Up front, at the pre-signature review (buyer/review.py) --
    #      "if the exact item isn't available, substitute within +/-X%?"
    #      Answering yes here is what keeps G7.2 from re-asking a question
    #      the human already answered. Originally this could ONLY be set
    #      post-hoc, which meant a substitute sitting comfortably inside a
    #      tolerance the human themselves set still stopped the loop.
    #   2. Post-hoc by buyer/approval.py's amendment flow, when a human
    #      has reviewed and approved a G7.2 (AMBIGUOUS_SUBSTITUTION)
    #      escalation -- G7.2 is a binary "any substitute present" check
    #      with no numeric threshold to widen, so re-evaluation after
    #      approval needs an explicit signed flag to resolve cleanly
    #      through the SAME gate, not a bypass.
    #
    # Either way it is a signed field on the intent, never an inference.
    substitutions_preapproved: bool = False
    # Whether the buyer wants to be SHOWN close-but-non-compliant
    # alternatives (never asked about them -- see buyer/suggest.py). This
    # is authorized once, at the same single touchpoint, so that even
    # "may I show you alternatives?" is not a second interrupt.
    notify_on_near_miss: bool = True


class IntentMandate(FrozenModel):
    version: Literal["1.0"] = "1.0"
    mandate_id: str
    issued_at: datetime
    expires_at: datetime
    principal_kid: str  # human key thumbprint
    agent_kid: str  # buyer agent key thumbprint
    request_text: str  # original NL request, kept for audit
    hard: HardConstraints
    soft: SoftPreferences
    budget: Budget
    allowed_merchants: list[str] | None = None  # None = any signed merchant

    _v_issued_at = field_validator("issued_at")(_require_tz_aware)
    _v_expires_at = field_validator("expires_at")(_require_tz_aware)

    @model_validator(mode="after")
    def _expiry_after_issuance(self) -> "IntentMandate":
        if self.expires_at <= self.issued_at:
            raise ValueError(
                f"expires_at ({self.expires_at}) must be after issued_at "
                f"({self.issued_at})"
            )
        return self


class CartLine(FrozenModel):
    sku: str
    title: str
    category: str
    unit_price_paise: int = Field(ge=0)
    quantity: int = Field(ge=1)
    attributes: dict[str, str] = Field(default_factory=dict)
    is_substitute: bool = False
    is_upsell: bool = False
    field_provenance: dict[str, Provenance]  # per-FIELD, not per-item

    @property
    def line_total_paise(self) -> int:
        return self.unit_price_paise * self.quantity


class CartMandate(FrozenModel):
    version: Literal["1.0"] = "1.0"
    mandate_id: str
    issued_at: datetime
    expires_at: datetime
    merchant_id: str
    merchant_kid: str
    intent_hash: str  # sha256 hex of the canonical IntentMandate payload
    lines: list[CartLine]
    subtotal_paise: int = Field(ge=0)
    tax_paise: int = Field(ge=0)
    shipping_paise: int = Field(ge=0)
    total_paise: int = Field(ge=0)

    _v_issued_at = field_validator("issued_at")(_require_tz_aware)
    _v_expires_at = field_validator("expires_at")(_require_tz_aware)

    @model_validator(mode="after")
    def _expiry_after_issuance(self) -> "CartMandate":
        if self.expires_at <= self.issued_at:
            raise ValueError(
                f"expires_at ({self.expires_at}) must be after issued_at "
                f"({self.issued_at})"
            )
        return self

    @property
    def computed_subtotal_paise(self) -> int:
        return sum(line.line_total_paise for line in self.lines)

    @property
    def arithmetic_consistent(self) -> bool:
        """True iff subtotal/tax/shipping/total actually add up. This does
        NOT enforce the invariant at construction time -- a cart with
        deliberately wrong arithmetic is exactly what attack variant A4
        (price swap) needs to be constructible for the red-team corpus.
        The policy engine's G2.1 gate is what rejects it at evaluation
        time, not this schema. Keeping validation and detection in
        separate places is deliberate: the schema must be able to
        represent an invalid cart so the engine has something real to
        catch."""
        return (
            self.computed_subtotal_paise == self.subtotal_paise
            and self.subtotal_paise + self.tax_paise + self.shipping_paise
            == self.total_paise
        )


class PaymentMandate(FrozenModel):
    version: Literal["1.0"] = "1.0"
    mandate_id: str
    created_at: datetime
    intent_hash: str
    cart_hash: str
    amount_paise: int = Field(ge=0)
    rail: Literal["razorpay_test"] = "razorpay_test"
    razorpay_order_id: str | None = None  # filled post-creation

    _v_created_at = field_validator("created_at")(_require_tz_aware)


class CancellationMandate(FrozenModel):
    """A signed request to reverse a specific payment -- the recourse for
    once execute.py has created an order and there is no path back other
    than calling the working capture_payment/create_refund methods on
    razorpay_client.

    For a system whose entire premise is that an agent spends a user's
    money without them watching, "I changed my mind" and "it bought the
    wrong thing" are not edge cases -- they are the first two questions
    any real user asks. This mandate makes a reversal as auditable and
    attributable as the purchase was: it references the ORIGINAL
    PaymentMandate by hash, exactly the way CartMandate.intent_hash and
    PaymentMandate.{intent_hash,cart_hash} reference what came before
    them, so a cancellation is chain-bound to a real prior payment rather
    than a bare, untraceable API call."""

    version: Literal["1.0"] = "1.0"
    mandate_id: str
    created_at: datetime
    payment_hash: str  # sha256 hex of the canonical PaymentMandate payload
    principal_kid: str  # who authorized the cancellation
    reason: Literal["user_requested", "reconciliation_mismatch"]
    amount_paise: int | None = Field(default=None, ge=0)
    # None = refund the full original amount. Set for a partial refund.

    _v_created_at = field_validator("created_at")(_require_tz_aware)


class MerchantPolicy(FrozenModel):
    """The merchant's own transaction rules -- published in
    .well-known/agent-policy.json and enforced by the buyer's policy
    engine gate G6.1 alongside the buyer's own IntentMandate.budget. Two
    independent policies, both must be satisfied."""

    merchant_id: str
    max_order_value_paise: int = Field(ge=0)
    substitution_tolerance_pct: float = Field(ge=0.0, le=100.0)
    max_quantity_per_sku: int = Field(ge=1)
    restricted_categories: list[str] = Field(default_factory=list)
    escalation_threshold_paise: int = Field(ge=0)


class Signature(BaseModel):
    kid: str
    alg: Literal["EdDSA"] = "EdDSA"
    sig: str  # base64url, unpadded


class Envelope(BaseModel):
    """Generic signed wrapper: {"payload": {...}, "signatures": [...]}.
    `payload` is the JSON-mode dump of an IntentMandate/CartMandate/
    PaymentMandate -- see mandates/sign.py and mandates/verify.py."""

    payload: dict
    signatures: list[Signature] = Field(default_factory=list)
