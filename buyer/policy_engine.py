"""
The deterministic authority: the LLM proposes, it never authorizes.
Nothing in this module calls an LLM, does I/O, or reads the wall clock --
`evaluate()` is a pure function of its arguments, which is what makes it:

  - exhaustively unit-testable (this file's test suite covers every gate
    code, plus boundary values at exactly each cap),
  - free to replay in the eval harness's ablation -- replaying evaluate()
    against a recorded LLM transcript costs zero LLM calls because this
    function never touches the network,
  - and the actual anti-injection defense: an attacker who gets a
    merchant's LLM to say anything at all still has to get that text
    turned into a typed, provenance-tagged CartMandate field before it
    can influence a Decision. Free text never reaches this module.

Gate table (gate id, code, severity):

  G0.1  MANDATE_EXPIRED              REJECT
  G0.2  SIGNATURE_INVALID            REJECT
  G0.3  MANDATE_CHAIN_BROKEN         REJECT
  G0.4  MERCHANT_NOT_AUTHORIZED      REJECT
  G0.5  AGENT_IDENTITY_UNVERIFIED    REJECT  (only active when ctx.require_agent_identity)
  G1.1  PROVENANCE_INSUFFICIENT      REJECT  -- the injection kill-switch
  G2.1  CART_ARITHMETIC_MISMATCH     REJECT
  G3.1  HARD_CONSTRAINT_VIOLATED     REJECT
  G3.2  CATEGORY_NOT_ALLOWED         REJECT
  G4.1  PER_ITEM_CAP_EXCEEDED        REJECT
  G4.2  QUANTITY_CAP_EXCEEDED        REJECT
  G4.3  BUDGET_EXCEEDED              REJECT
  G5.1  SUBSTITUTION_OUT_OF_TOLERANCE REJECT
  G6.1  MERCHANT_POLICY_CONFLICT     REJECT
  G7.1  ESCALATION_THRESHOLD         ESCALATE
  G7.2  AMBIGUOUS_SUBSTITUTION       ESCALATE
  G7.3  LOW_CONFIDENCE_INTERPRETATION ESCALATE

Resolution: every gate always runs and is recorded (full audit visibility
even when the outcome was decided by a different gate). Outcome resolves
by severity across all recorded checks: any REJECT-failure -> REJECTED;
else any ESCALATE-failure -> REQUIRES_HUMAN_APPROVAL; else APPROVED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from buyer.catalog import VerifiedCatalog
from mandates.chain import is_expired, verify_chain
from mandates.keys import KeyDirectory
from mandates.schemas import CartMandate, Envelope, IntentMandate, MerchantPolicy
from mandates.verify import verify_envelope

GateSeverity = Literal["REJECT", "ESCALATE"]

GATE_SEVERITY: dict[str, GateSeverity] = {
    "MANDATE_EXPIRED": "REJECT",
    "SIGNATURE_INVALID": "REJECT",
    "MANDATE_CHAIN_BROKEN": "REJECT",
    "MERCHANT_NOT_AUTHORIZED": "REJECT",
    "AGENT_IDENTITY_UNVERIFIED": "REJECT",
    "PROVENANCE_INSUFFICIENT": "REJECT",
    "CART_ARITHMETIC_MISMATCH": "REJECT",
    "HARD_CONSTRAINT_VIOLATED": "REJECT",
    "CATEGORY_NOT_ALLOWED": "REJECT",
    "PER_ITEM_CAP_EXCEEDED": "REJECT",
    "QUANTITY_CAP_EXCEEDED": "REJECT",
    "BUDGET_EXCEEDED": "REJECT",
    "SUBSTITUTION_OUT_OF_TOLERANCE": "REJECT",
    "MERCHANT_POLICY_CONFLICT": "REJECT",
    "ESCALATION_THRESHOLD": "ESCALATE",
    "AMBIGUOUS_SUBSTITUTION": "ESCALATE",
    "LOW_CONFIDENCE_INTERPRETATION": "ESCALATE",
    "CATALOG_PRICE_MISMATCH": "REJECT",
    "CATALOG_ITEM_UNKNOWN": "REJECT",
    "CART_SHAPE_INVALID": "REJECT",
    "ENGINE_ERROR": "REJECT",
}

# Cart shape bounds (G8.3). Deliberately generous -- these are not
# business limits (the budget gates are), they are sanity bounds that
# make a pathological cart a loud stop instead of a slow one.
MAX_CART_LINES = 20
MAX_UPSELL_LINES = 1


@dataclass(frozen=True)
class VerificationContext:
    """Runtime verification context. NOT a signed wire artifact -- unlike
    IntentMandate/CartMandate, this carries process-local facts needed to
    evaluate a decision (which envelopes to check signatures against,
    whether this run's ablation config requires verified agent identity,
    whether the interpretation stage flagged unresolved ambiguity)."""

    key_directory: KeyDirectory
    intent_envelope: Envelope
    cart_envelope: Envelope
    require_agent_identity: bool = False  # ablation config C4 toggle (G0.5)
    webbotauth_verified: bool = False
    low_confidence_interpretation: bool = False  # feeds G7.3
    merchant_catalog: VerifiedCatalog | None = None
    # The merchant's own published, signature-verified catalog, fetched
    # and verified OUTSIDE this module (buyer/catalog.py) so evaluate()
    # stays pure. It lives on the context rather than as a positional
    # argument for the same reason the envelopes do: it is a
    # already-verified input to verification, not a signed wire artifact
    # this engine itself validates. `None` is not a permissive default --
    # G8.1/G8.2 REJECT on it (see the module docstring).


@dataclass(frozen=True)
class Check:
    gate: str
    code: str
    passed: bool
    detail: str
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    outcome: Literal["APPROVED", "REQUIRES_HUMAN_APPROVAL", "REJECTED"]
    checks: tuple[Check, ...]
    blocking_code: str | None
    evaluated_at: datetime

    @property
    def failed_checks(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def check_for(self, code: str) -> Check | None:
        for c in self.checks:
            if c.code == code:
                return c
        return None


def _ok(gate: str, code: str, detail: str, evidence: dict | None = None) -> Check:
    return Check(gate=gate, code=code, passed=True, detail=detail, evidence=evidence or {})


def _fail(gate: str, code: str, detail: str, evidence: dict | None = None) -> Check:
    return Check(gate=gate, code=code, passed=False, detail=detail, evidence=evidence or {})


# ---------------------------------------------------------------------
# G0.x -- structural / trust-boundary gates
# ---------------------------------------------------------------------


def _g0_1_mandate_expired(intent: IntentMandate, cart: CartMandate, now: datetime) -> Check:
    intent_expired = is_expired(intent, now)
    cart_expired = is_expired(cart, now)
    evidence = {
        "now": now.isoformat(),
        "intent_expires_at": intent.expires_at.isoformat(),
        "cart_expires_at": cart.expires_at.isoformat(),
    }
    if intent_expired or cart_expired:
        which = "intent" if intent_expired else "cart"
        return _fail("G0.1", "MANDATE_EXPIRED", f"{which} mandate expired as of {now.isoformat()}", evidence)
    return _ok("G0.1", "MANDATE_EXPIRED", "neither mandate is expired", evidence)


def _g0_2_signature_invalid(intent: IntentMandate, cart: CartMandate, ctx: VerificationContext) -> Check:
    intent_result = verify_envelope(
        ctx.intent_envelope, ctx.key_directory, required_kid=intent.principal_kid
    )
    if not intent_result.valid:
        return _fail(
            "G0.2", "SIGNATURE_INVALID", f"IntentMandate signature invalid: {intent_result.reason}",
            {"principal_kid": intent.principal_kid},
        )
    cart_result = verify_envelope(
        ctx.cart_envelope, ctx.key_directory, required_kid=cart.merchant_kid
    )
    if not cart_result.valid:
        return _fail(
            "G0.2", "SIGNATURE_INVALID", f"CartMandate signature invalid: {cart_result.reason}",
            {"merchant_kid": cart.merchant_kid},
        )
    return _ok(
        "G0.2", "SIGNATURE_INVALID", "both IntentMandate and CartMandate signatures verified",
        {"principal_kid": intent.principal_kid, "merchant_kid": cart.merchant_kid},
    )


def _g0_3_mandate_chain_broken(intent: IntentMandate, cart: CartMandate) -> Check:
    result = verify_chain(intent, cart)
    if not result.intact:
        return _fail("G0.3", "MANDATE_CHAIN_BROKEN", "; ".join(result.violations), {"violations": list(result.violations)})
    return _ok("G0.3", "MANDATE_CHAIN_BROKEN", "cart.intent_hash matches hash(intent)")


def _g0_4_merchant_not_authorized(intent: IntentMandate, cart: CartMandate) -> Check:
    if intent.allowed_merchants is None:
        return _ok("G0.4", "MERCHANT_NOT_AUTHORIZED", "intent places no merchant restriction")
    if cart.merchant_id not in intent.allowed_merchants:
        return _fail(
            "G0.4", "MERCHANT_NOT_AUTHORIZED",
            f"merchant {cart.merchant_id!r} not in allowed_merchants",
            {"merchant_id": cart.merchant_id, "allowed_merchants": intent.allowed_merchants},
        )
    return _ok(
        "G0.4", "MERCHANT_NOT_AUTHORIZED", f"merchant {cart.merchant_id!r} is authorized",
        {"merchant_id": cart.merchant_id},
    )


def _g0_5_agent_identity_unverified(ctx: VerificationContext) -> Check:
    if not ctx.require_agent_identity:
        return _ok("G0.5", "AGENT_IDENTITY_UNVERIFIED", "not required for this config (C1-C3)")
    if not ctx.webbotauth_verified:
        return _fail("G0.5", "AGENT_IDENTITY_UNVERIFIED", "RFC 9421 agent identity signature not verified")
    return _ok("G0.5", "AGENT_IDENTITY_UNVERIFIED", "agent identity verified via Web Bot Auth")


# ---------------------------------------------------------------------
# G1.x -- the injection kill-switch
# ---------------------------------------------------------------------

_UNTRUSTED_EXTRACTION_METHODS = {"llm_inferred"}


def _g1_1_provenance_insufficient(cart: CartMandate) -> Check:
    """An LLM-guessed or unsigned price cannot be authorized against,
    regardless of what any merchant text claims. This is what makes the
    Branded-Whisper-style attack fail structurally: the merchant agent can
    say anything it wants, but nothing it *says* ever becomes a trusted
    unit_price_paise -- only a schema.org/merchant_signed field with
    signature_verified=True does."""
    offending: list[dict] = []
    for line in cart.lines:
        prov = line.field_provenance.get("unit_price_paise")
        if prov is None:
            offending.append({"sku": line.sku, "reason": "no provenance recorded for unit_price_paise"})
            continue
        if prov.extraction_method in _UNTRUSTED_EXTRACTION_METHODS or not prov.signature_verified:
            offending.append(
                {"sku": line.sku, "extraction_method": prov.extraction_method, "signature_verified": prov.signature_verified}
            )
    if offending:
        return _fail(
            "G1.1", "PROVENANCE_INSUFFICIENT",
            f"{len(offending)} line(s) have unverified/inferred pricing", {"offending_lines": offending},
        )
    return _ok("G1.1", "PROVENANCE_INSUFFICIENT", "all line prices are signed/verified, non-inferred")


# ---------------------------------------------------------------------
# G2.x -- arithmetic
# ---------------------------------------------------------------------


def _g2_1_cart_arithmetic_mismatch(cart: CartMandate) -> Check:
    computed_subtotal = cart.computed_subtotal_paise
    computed_total = cart.subtotal_paise + cart.tax_paise + cart.shipping_paise
    evidence = {
        "computed_subtotal_paise": computed_subtotal,
        "declared_subtotal_paise": cart.subtotal_paise,
        "computed_total_paise": computed_total,
        "declared_total_paise": cart.total_paise,
    }
    if not cart.arithmetic_consistent:
        return _fail("G2.1", "CART_ARITHMETIC_MISMATCH", "declared totals do not match line-item arithmetic", evidence)
    return _ok("G2.1", "CART_ARITHMETIC_MISMATCH", "cart arithmetic is internally consistent", evidence)


# ---------------------------------------------------------------------
# G3.x -- hard constraints
# ---------------------------------------------------------------------


def _g3_1_hard_constraint_violated(intent: IntentMandate, cart: CartMandate) -> Check:
    violations: list[dict] = []
    qualifying_lines = [line for line in cart.lines if not line.is_upsell]
    for line in qualifying_lines:
        if intent.hard.category is not None and line.category != intent.hard.category:
            violations.append(
                {"sku": line.sku, "reason": "category mismatch", "expected": intent.hard.category, "actual": line.category}
            )
        for key, expected_value in intent.hard.required_attributes.items():
            actual_value = line.attributes.get(key)
            if actual_value != expected_value:
                violations.append(
                    {"sku": line.sku, "reason": f"required attribute {key!r} mismatch", "expected": expected_value, "actual": actual_value}
                )
    if violations:
        return _fail("G3.1", "HARD_CONSTRAINT_VIOLATED", f"{len(violations)} hard-constraint violation(s)", {"violations": violations})
    return _ok(
        "G3.1", "HARD_CONSTRAINT_VIOLATED", "all non-upsell lines satisfy hard constraints",
        {"checked_lines": [line.sku for line in qualifying_lines]},
    )


def _g3_2_category_not_allowed(intent: IntentMandate, cart: CartMandate) -> Check:
    offending = [line.sku for line in cart.lines if line.category in intent.hard.excluded_categories]
    if offending:
        return _fail(
            "G3.2", "CATEGORY_NOT_ALLOWED", f"{len(offending)} line(s) in an excluded category",
            {"offending_skus": offending, "excluded_categories": intent.hard.excluded_categories},
        )
    return _ok("G3.2", "CATEGORY_NOT_ALLOWED", "no line falls in an excluded category")


# ---------------------------------------------------------------------
# G4.x -- per-line and total budget caps (apply uniformly, INCLUDING
# upsell lines -- this is the gate that catches an over-cap upsell pitch,
# e.g. the demo's "signature moment")
# ---------------------------------------------------------------------


def _g4_1_per_item_cap_exceeded(intent: IntentMandate, cart: CartMandate) -> Check:
    offending = [
        {"sku": line.sku, "unit_price_paise": line.unit_price_paise}
        for line in cart.lines
        if line.unit_price_paise > intent.budget.per_item_paise
    ]
    if offending:
        return _fail(
            "G4.1", "PER_ITEM_CAP_EXCEEDED", f"{len(offending)} line(s) exceed per-item cap",
            {"offending_lines": offending, "per_item_cap_paise": intent.budget.per_item_paise},
        )
    return _ok("G4.1", "PER_ITEM_CAP_EXCEEDED", "all lines within per-item cap", {"per_item_cap_paise": intent.budget.per_item_paise})


def _g4_2_quantity_cap_exceeded(intent: IntentMandate, cart: CartMandate) -> Check:
    offending = [
        {"sku": line.sku, "quantity": line.quantity} for line in cart.lines if line.quantity > intent.budget.max_quantity
    ]
    if offending:
        return _fail(
            "G4.2", "QUANTITY_CAP_EXCEEDED", f"{len(offending)} line(s) exceed max quantity",
            {"offending_lines": offending, "max_quantity": intent.budget.max_quantity},
        )
    return _ok("G4.2", "QUANTITY_CAP_EXCEEDED", "all lines within max quantity", {"max_quantity": intent.budget.max_quantity})


def _g4_3_budget_exceeded(intent: IntentMandate, cart: CartMandate) -> Check:
    evidence = {"total_paise": cart.total_paise, "limit_paise": intent.budget.total_paise}
    if cart.total_paise > intent.budget.total_paise:
        return _fail("G4.3", "BUDGET_EXCEEDED", f"cart total {cart.total_paise} exceeds budget {intent.budget.total_paise}", evidence)
    return _ok("G4.3", "BUDGET_EXCEEDED", f"cart total {cart.total_paise} within budget {intent.budget.total_paise}", evidence)


# ---------------------------------------------------------------------
# G5.x -- substitution tolerance
# ---------------------------------------------------------------------


def _g5_1_substitution_out_of_tolerance(
    intent: IntentMandate, merchant_policy: MerchantPolicy, cart: CartMandate
) -> Check:
    tolerance_pct = min(intent.soft.substitution_tolerance_pct, merchant_policy.substitution_tolerance_pct)
    target = intent.soft.target_price_paise
    if target is None:
        return _ok(
            "G5.1", "SUBSTITUTION_OUT_OF_TOLERANCE",
            "no target_price_paise set on intent -- nothing to compare a substitute against",
            {"tolerance_pct": tolerance_pct},
        )
    offending: list[dict] = []
    for line in cart.lines:
        if not line.is_substitute:
            continue
        delta_pct = abs(line.unit_price_paise - target) / target * 100
        if delta_pct > tolerance_pct:
            offending.append({"sku": line.sku, "unit_price_paise": line.unit_price_paise, "delta_pct": round(delta_pct, 2)})
    if offending:
        return _fail(
            "G5.1", "SUBSTITUTION_OUT_OF_TOLERANCE", f"{len(offending)} substitute(s) exceed {tolerance_pct}% tolerance",
            {"offending_lines": offending, "tolerance_pct": tolerance_pct, "target_price_paise": target},
        )
    return _ok(
        "G5.1", "SUBSTITUTION_OUT_OF_TOLERANCE", "all substitutes within tolerance",
        {"tolerance_pct": tolerance_pct, "target_price_paise": target},
    )


# ---------------------------------------------------------------------
# G6.x -- merchant's own policy
# ---------------------------------------------------------------------


def _g6_1_merchant_policy_conflict(merchant_policy: MerchantPolicy, cart: CartMandate) -> Check:
    violations: list[str] = []
    if cart.total_paise > merchant_policy.max_order_value_paise:
        violations.append(
            f"total_paise {cart.total_paise} exceeds merchant max_order_value_paise {merchant_policy.max_order_value_paise}"
        )
    for line in cart.lines:
        if line.quantity > merchant_policy.max_quantity_per_sku:
            violations.append(f"{line.sku} quantity {line.quantity} exceeds merchant max_quantity_per_sku {merchant_policy.max_quantity_per_sku}")
        if line.category in merchant_policy.restricted_categories:
            violations.append(f"{line.sku} category {line.category!r} is merchant-restricted")
    if violations:
        return _fail("G6.1", "MERCHANT_POLICY_CONFLICT", "; ".join(violations), {"violations": violations})
    return _ok("G6.1", "MERCHANT_POLICY_CONFLICT", "cart satisfies merchant's own policy")


# ---------------------------------------------------------------------
# G7.x -- escalation (soft) gates
# ---------------------------------------------------------------------


def _g7_1_escalation_threshold(
    intent: IntentMandate, merchant_policy: MerchantPolicy, cart: CartMandate
) -> Check:
    """Honours BOTH thresholds, tighter wins -- mirroring what G5.1
    already does for substitution_tolerance_pct.

    MerchantPolicy.escalation_threshold_paise used to be a dead field: it
    was in the schema, the toolkit set it, emit.py SIGNED it into
    .well-known/agent-policy.json, and nothing anywhere read it. Every
    merchant onboarded published a cryptographically signed escalation
    threshold with zero effect on any decision -- which is worse than an
    unused variable. It is a signed public claim the enforcer ignores,
    exactly the thing this project criticises other protocols for."""
    threshold = min(intent.budget.escalate_above_paise, merchant_policy.escalation_threshold_paise)
    evidence = {
        "total_paise": cart.total_paise,
        "effective_threshold_paise": threshold,
        "buyer_escalate_above_paise": intent.budget.escalate_above_paise,
        "merchant_escalation_threshold_paise": merchant_policy.escalation_threshold_paise,
    }
    if cart.total_paise > threshold:
        source = "buyer" if threshold == intent.budget.escalate_above_paise else "merchant"
        return _fail(
            "G7.1", "ESCALATION_THRESHOLD",
            f"cart total {cart.total_paise} exceeds the {source}'s escalation threshold {threshold}",
            evidence,
        )
    return _ok("G7.1", "ESCALATION_THRESHOLD", "cart total below escalation threshold", evidence)


def _g7_2_ambiguous_substitution(intent: IntentMandate, cart: CartMandate) -> Check:
    substitute_skus = [line.sku for line in cart.lines if line.is_substitute]
    if not substitute_skus:
        return _ok("G7.2", "AMBIGUOUS_SUBSTITUTION", "no substitute lines present")
    if intent.soft.substitutions_preapproved:
        # Set only via buyer/approval.py's re-signed amendment after a
        # human has actually reviewed this exact escalation -- see
        # SoftPreferences.substitutions_preapproved's docstring. This is
        # the SAME gate resolving cleanly on re-evaluation, not a bypass.
        return _ok(
            "G7.2", "AMBIGUOUS_SUBSTITUTION",
            f"{len(substitute_skus)} substitute line(s) present but pre-approved by signed intent",
            {"substitute_skus": substitute_skus},
        )
    return _fail("G7.2", "AMBIGUOUS_SUBSTITUTION", f"{len(substitute_skus)} substitute line(s) present, needs human confirmation", {"substitute_skus": substitute_skus})


def _g7_3_low_confidence_interpretation(ctx: VerificationContext) -> Check:
    if ctx.low_confidence_interpretation:
        return _fail("G7.3", "LOW_CONFIDENCE_INTERPRETATION", "interpretation stage flagged unresolved, load-bearing ambiguity")
    return _ok("G7.3", "LOW_CONFIDENCE_INTERPRETATION", "interpretation was unambiguous or ambiguity was resolved")


# ---------------------------------------------------------------------
# G8.x -- the anomaly band: bind the cart to what the MERCHANT itself
# published, and to a sane cart shape. All REJECT. These fail CLOSED on a
# missing catalog: "I could not account for this" is a stop, not a pass.
# See the module docstring for why this is a reason code and not a fourth
# Decision outcome.
# ---------------------------------------------------------------------


def _catalog_or_failure(ctx: VerificationContext, cart: CartMandate, gate: str, code: str) -> Check | None:
    """Shared precondition for G8.1/G8.2. Returns a failing Check when
    there is no usable catalog to bind against, else None."""
    catalog = ctx.merchant_catalog
    if catalog is None:
        return _fail(
            gate, code,
            "no verified merchant catalog available -- cart prices cannot be "
            "bound to anything the merchant actually published",
            {"merchant_id": cart.merchant_id},
        )
    if catalog.merchant_id != cart.merchant_id:
        return _fail(
            gate, code,
            f"catalog belongs to merchant {catalog.merchant_id!r} but the cart "
            f"is from {cart.merchant_id!r} -- one merchant's signed catalog "
            f"cannot vouch for another's prices",
            {"catalog_merchant_id": catalog.merchant_id, "cart_merchant_id": cart.merchant_id},
        )
    return None


def _g8_1_catalog_price_mismatch(cart: CartMandate, ctx: VerificationContext) -> Check:
    """A signed cart may charge LESS than the published price (a discount
    is the merchant's own business) but never MORE. Charging above your
    own published price is the definition of a dishonestly-priced cart,
    and it is invisible to every budget gate as long as it stays under
    the cap."""
    precondition = _catalog_or_failure(ctx, cart, "G8.1", "CATALOG_PRICE_MISMATCH")
    if precondition is not None:
        return precondition
    catalog = ctx.merchant_catalog
    assert catalog is not None  # narrowed by _catalog_or_failure

    offending: list[dict] = []
    for line in cart.lines:
        entry = catalog.get(line.sku)
        if entry is None:
            continue  # G8.2's business, not this gate's
        if line.unit_price_paise > entry.price_paise:
            offending.append(
                {
                    "sku": line.sku,
                    "charged_paise": line.unit_price_paise,
                    "published_paise": entry.price_paise,
                    "overcharge_paise": line.unit_price_paise - entry.price_paise,
                }
            )
    if offending:
        worst = max(offending, key=lambda o: o["overcharge_paise"])
        return _fail(
            "G8.1", "CATALOG_PRICE_MISMATCH",
            f"{len(offending)} line(s) charge above the merchant's own published price "
            f"(worst: {worst['sku']} at {worst['charged_paise']} vs published "
            f"{worst['published_paise']})",
            {"offending_lines": offending, "catalog_generated_at": catalog.generated_at},
        )
    return _ok(
        "G8.1", "CATALOG_PRICE_MISMATCH",
        "every line is priced at or below the merchant's published catalog price",
        {"catalog_size": len(catalog), "catalog_generated_at": catalog.generated_at},
    )


def _g8_2_catalog_item_unknown(cart: CartMandate, ctx: VerificationContext) -> Check:
    """Every line must correspond to a published catalog item -- same SKU
    AND same category. An unpublished SKU has no price to check against,
    so letting it through would reopen G8.1 by simply inventing a SKU;
    and a line claiming a different category than the catalog publishes
    for that SKU is how an excluded-category item would sneak past G3.2
    while still being priced legitimately.

    The category check also narrows G3.1's deliberate is_upsell exemption:
    a pitched accessory is legitimately not the thing the buyer asked
    for, so it is exempt from the buyer's HARD constraints -- but it must
    still be a real, correctly-described item from this merchant's own
    published catalog."""
    precondition = _catalog_or_failure(ctx, cart, "G8.2", "CATALOG_ITEM_UNKNOWN")
    if precondition is not None:
        return precondition
    catalog = ctx.merchant_catalog
    assert catalog is not None

    offending: list[dict] = []
    for line in cart.lines:
        entry = catalog.get(line.sku)
        if entry is None:
            offending.append({"sku": line.sku, "reason": "sku not present in published catalog"})
        elif entry.category != line.category:
            offending.append(
                {
                    "sku": line.sku,
                    "reason": "category differs from published catalog",
                    "cart_category": line.category,
                    "published_category": entry.category,
                }
            )
    if offending:
        return _fail(
            "G8.2", "CATALOG_ITEM_UNKNOWN",
            f"{len(offending)} line(s) do not match a published catalog item",
            {"offending_lines": offending, "catalog_size": len(catalog)},
        )
    return _ok(
        "G8.2", "CATALOG_ITEM_UNKNOWN",
        "every line matches a published catalog item by sku and category",
        {"catalog_size": len(catalog)},
    )


def _g8_3_cart_shape_invalid(cart: CartMandate) -> Check:
    """Structural sanity the other gates all quietly assumed. Each of
    these was constructible before this gate existed: a fifty-line cart
    of the same duplicated SKU, every line flagged is_upsell, validated
    and evaluated without complaint."""
    violations: list[str] = []

    if not cart.lines:
        violations.append("cart has no lines")
    if len(cart.lines) > MAX_CART_LINES:
        violations.append(f"cart has {len(cart.lines)} lines, over the {MAX_CART_LINES}-line cap")

    seen: set[str] = set()
    duplicates = sorted({line.sku for line in cart.lines if line.sku in seen or seen.add(line.sku)})
    if duplicates:
        violations.append(
            f"duplicate SKU(s) {duplicates} -- a quantity belongs in one line's "
            f"quantity field, where G4.2 can see it, not spread across lines"
        )

    upsell_count = sum(1 for line in cart.lines if line.is_upsell)
    if upsell_count > MAX_UPSELL_LINES:
        violations.append(
            f"{upsell_count} upsell lines, over the {MAX_UPSELL_LINES}-per-cart cap "
            f"(merchant_agent/driver.py already caps itself at one offer per session)"
        )

    if violations:
        return _fail(
            "G8.3", "CART_SHAPE_INVALID", "; ".join(violations),
            {"violations": violations, "line_count": len(cart.lines), "upsell_count": upsell_count},
        )
    return _ok(
        "G8.3", "CART_SHAPE_INVALID",
        f"cart shape is sane ({len(cart.lines)} line(s), {upsell_count} upsell)",
        {"line_count": len(cart.lines), "upsell_count": upsell_count},
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

_GATE_ORDER = [
    "G0.1", "G0.2", "G0.3", "G0.4", "G0.5",
    "G1.1",
    "G2.1",
    "G3.1", "G3.2",
    "G4.1", "G4.2", "G4.3",
    "G5.1",
    "G6.1",
    "G7.1", "G7.2", "G7.3",
    "G8.1", "G8.2", "G8.3",
]


def evaluate(
    intent: IntentMandate,
    cart: CartMandate,
    merchant_policy: MerchantPolicy,
    ctx: VerificationContext,
    now: datetime,
) -> Decision:
    """Pure. No I/O, no network, no LLM, no wall-clock read (now is
    injected). Any uncaught exception is treated as a REJECTED decision
    with code ENGINE_ERROR -- fail closed, never fail open."""
    try:
        checks = (
            _g0_1_mandate_expired(intent, cart, now),
            _g0_2_signature_invalid(intent, cart, ctx),
            _g0_3_mandate_chain_broken(intent, cart),
            _g0_4_merchant_not_authorized(intent, cart),
            _g0_5_agent_identity_unverified(ctx),
            _g1_1_provenance_insufficient(cart),
            _g2_1_cart_arithmetic_mismatch(cart),
            _g3_1_hard_constraint_violated(intent, cart),
            _g3_2_category_not_allowed(intent, cart),
            _g4_1_per_item_cap_exceeded(intent, cart),
            _g4_2_quantity_cap_exceeded(intent, cart),
            _g4_3_budget_exceeded(intent, cart),
            _g5_1_substitution_out_of_tolerance(intent, merchant_policy, cart),
            _g6_1_merchant_policy_conflict(merchant_policy, cart),
            _g7_1_escalation_threshold(intent, merchant_policy, cart),
            _g7_2_ambiguous_substitution(intent, cart),
            _g7_3_low_confidence_interpretation(ctx),
            _g8_1_catalog_price_mismatch(cart, ctx),
            _g8_2_catalog_item_unknown(cart, ctx),
            _g8_3_cart_shape_invalid(cart),
        )
    except Exception as e:  # fail closed, never fail open
        err = Check(gate="ENGINE", code="ENGINE_ERROR", passed=False, detail=f"{type(e).__name__}: {e}")
        return Decision(outcome="REJECTED", checks=(err,), blocking_code="ENGINE_ERROR", evaluated_at=now)

    failed = [c for c in checks if not c.passed]
    reject = [c for c in failed if GATE_SEVERITY[c.code] == "REJECT"]
    escalate = [c for c in failed if GATE_SEVERITY[c.code] == "ESCALATE"]

    if reject:
        outcome: Literal["APPROVED", "REQUIRES_HUMAN_APPROVAL", "REJECTED"] = "REJECTED"
        blocking = reject[0].code
    elif escalate:
        outcome = "REQUIRES_HUMAN_APPROVAL"
        blocking = escalate[0].code
    else:
        outcome = "APPROVED"
        blocking = None

    return Decision(outcome=outcome, checks=checks, blocking_code=blocking, evaluated_at=now)
