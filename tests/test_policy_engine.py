"""
Exhaustive tests for buyer/policy_engine.py -- every gate code, boundary
values at exactly each cap (one unit under, exactly at, one unit over),
and the outcome-resolution semantics (REJECT beats ESCALATE beats
APPROVED). This is the module the whole submission's safety claim rests
on, so "exhaustive" is not an exaggeration here -- see module docstring
in policy_engine.py for why purity is what makes this possible.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from buyer.catalog import CatalogEntry, VerifiedCatalog
from buyer.policy_engine import Decision, VerificationContext, evaluate
from mandates import (
    Budget,
    CartLine,
    CartMandate,
    HardConstraints,
    IntentMandate,
    KeyDirectory,
    MerchantPolicy,
    Provenance,
    SoftPreferences,
    generate_keypair,
    mandate_hash,
    sign_mandate,
)

UTC = timezone.utc
HUMAN_KP = generate_keypair()
MERCHANT_KP = generate_keypair()
OTHER_KP = generate_keypair()  # a keypair NOT the expected signer, for forgery tests


def _now() -> datetime:
    return datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)


def _scenario(
    *,
    now: datetime | None = None,
    budget_total_paise: int = 50_000,
    budget_per_item_paise: int = 50_000,
    budget_max_quantity: int = 2,
    budget_escalate_above_paise: int = 45_000,
    target_price_paise: int | None = None,
    buyer_substitution_tolerance_pct: float = 10.0,
    substitutions_preapproved: bool = False,
    hard_category: str | None = "grocery",
    required_attributes: dict | None = None,
    excluded_categories: list | None = None,
    allowed_merchants: tuple | None = ("merchant-1",),
    line_category: str = "grocery",
    line_quantity: int = 1,
    line_unit_price_paise: int = 30_000,
    is_substitute: bool = False,
    is_upsell: bool = False,
    line_attributes: dict | None = None,
    price_extraction_method: str = "schema_org",
    price_signature_verified: bool = True,
    price_confidence: float | None = None,
    missing_price_provenance: bool = False,
    tax_paise: int = 0,
    shipping_paise: int = 0,
    declared_subtotal_paise: int | None = None,
    declared_total_paise: int | None = None,
    intent_hash_override: str | None = None,
    merchant_max_order_value_paise: int = 1_000_000,
    merchant_substitution_tolerance_pct: float = 10.0,
    merchant_max_quantity_per_sku: int = 5,
    merchant_restricted_categories: list | None = None,
    merchant_escalation_threshold_paise: int = 999_999,
    require_agent_identity: bool = False,
    webbotauth_verified: bool = False,
    low_confidence_interpretation: bool = False,
    sign_intent_with=None,
    sign_cart_with=None,
    intent_expired: bool = False,
    cart_expired: bool = False,
    catalog_price_paise: int | None = None,
    catalog_category: str | None = None,
    catalog_skus: tuple[str, ...] | None = None,
    catalog_merchant_id: str = "merchant-1",
    omit_catalog: bool = False,
    extra_lines: list | None = None,
):
    """Builds a fully self-consistent (golden, APPROVED-by-default)
    scenario, with every knob overridable to isolate exactly one gate
    failure at a time. Defaults: cart total 30,000 paise, comfortably
    under every cap (per-item 50k, escalate-above 45k, budget-total 50k),
    single non-substitute non-upsell grocery line, verified schema.org
    pricing, matching merchant policy -- i.e. APPROVED with all 20 checks
    passing, unless a specific override says otherwise.

    The merchant catalog defaults to publishing exactly the cart's own
    line at exactly the cart's own price, so the G8.x anomaly band passes
    by default and every pre-existing gate test stays isolated to the one
    gate it was written for. `catalog_price_paise` / `catalog_category` /
    `catalog_skus` / `omit_catalog` are what the G8.x tests themselves
    drive."""
    now = now or _now()
    required_attributes = required_attributes or {}
    excluded_categories = excluded_categories or []
    merchant_restricted_categories = merchant_restricted_categories or []
    line_attributes = line_attributes or {}

    budget = Budget(
        total_paise=budget_total_paise,
        per_item_paise=budget_per_item_paise,
        max_quantity=budget_max_quantity,
        escalate_above_paise=budget_escalate_above_paise,
    )
    hard = HardConstraints(category=hard_category, required_attributes=required_attributes, excluded_categories=excluded_categories)
    soft = SoftPreferences(
        target_price_paise=target_price_paise, substitution_tolerance_pct=buyer_substitution_tolerance_pct,
        substitutions_preapproved=substitutions_preapproved,
    )

    intent_issued = now - timedelta(minutes=1)
    intent_expires = (now - timedelta(seconds=1)) if intent_expired else now + timedelta(minutes=10)
    intent = IntentMandate(
        mandate_id="intent-1",
        issued_at=intent_issued,
        expires_at=intent_expires,
        principal_kid=HUMAN_KP.kid,
        agent_kid="agent-kid-1",
        request_text="test request",
        hard=hard,
        soft=soft,
        budget=budget,
        allowed_merchants=list(allowed_merchants) if allowed_merchants is not None else None,
    )

    field_provenance = {}
    if not missing_price_provenance:
        field_provenance["unit_price_paise"] = Provenance(
            source_url="https://merchant.example/p",
            extraction_method=price_extraction_method,
            signature_verified=price_signature_verified,
            extracted_at=now,
            confidence=price_confidence,
        )
    field_provenance["title"] = Provenance(
        source_url="https://merchant.example/p", extraction_method="schema_org", signature_verified=True, extracted_at=now
    )

    line = CartLine(
        sku="SKU-1",
        title="Test Item",
        category=line_category,
        unit_price_paise=line_unit_price_paise,
        quantity=line_quantity,
        attributes=line_attributes,
        is_substitute=is_substitute,
        is_upsell=is_upsell,
        field_provenance=field_provenance,
    )

    computed_subtotal = line.line_total_paise
    subtotal = declared_subtotal_paise if declared_subtotal_paise is not None else computed_subtotal
    total = declared_total_paise if declared_total_paise is not None else subtotal + tax_paise + shipping_paise

    intent_hash = intent_hash_override if intent_hash_override is not None else mandate_hash(intent)

    cart_issued = now - timedelta(minutes=1)
    cart_expires = (now - timedelta(seconds=1)) if cart_expired else now + timedelta(minutes=10)
    cart = CartMandate(
        mandate_id="cart-1",
        issued_at=cart_issued,
        expires_at=cart_expires,
        merchant_id="merchant-1",
        merchant_kid=MERCHANT_KP.kid,
        intent_hash=intent_hash,
        lines=[line, *(extra_lines or [])],
        subtotal_paise=subtotal,
        tax_paise=tax_paise,
        shipping_paise=shipping_paise,
        total_paise=total,
    )

    merchant_policy = MerchantPolicy(
        merchant_id="merchant-1",
        max_order_value_paise=merchant_max_order_value_paise,
        substitution_tolerance_pct=merchant_substitution_tolerance_pct,
        max_quantity_per_sku=merchant_max_quantity_per_sku,
        restricted_categories=merchant_restricted_categories,
        escalation_threshold_paise=merchant_escalation_threshold_paise,
    )

    directory = KeyDirectory()
    directory.register_keypair(HUMAN_KP)
    directory.register_keypair(MERCHANT_KP)
    directory.register_keypair(OTHER_KP)

    intent_envelope = sign_mandate(intent, sign_intent_with or HUMAN_KP)
    cart_envelope = sign_mandate(cart, sign_cart_with or MERCHANT_KP)

    # By default the merchant publishes exactly what it is charging --
    # an honest merchant, so the G8.x band is a no-op and every other
    # gate's test stays isolated.
    catalog = None
    if not omit_catalog:
        published_skus = catalog_skus if catalog_skus is not None else tuple(l.sku for l in cart.lines)
        catalog = VerifiedCatalog(
            merchant_id=catalog_merchant_id,
            merchant_kid=MERCHANT_KP.kid,
            generated_at=now.isoformat(),
            entries={
                sku: CatalogEntry(
                    sku=sku,
                    title="Test Item",
                    category=catalog_category if catalog_category is not None else line_category,
                    price_paise=(
                        catalog_price_paise if catalog_price_paise is not None else line_unit_price_paise
                    ),
                    in_stock=True,
                )
                for sku in published_skus
            },
        )

    ctx = VerificationContext(
        key_directory=directory,
        intent_envelope=intent_envelope,
        cart_envelope=cart_envelope,
        require_agent_identity=require_agent_identity,
        webbotauth_verified=webbotauth_verified,
        low_confidence_interpretation=low_confidence_interpretation,
        merchant_catalog=catalog,
    )

    return intent, cart, merchant_policy, ctx, now


def _decide(**overrides) -> Decision:
    intent, cart, merchant_policy, ctx, now = _scenario(**overrides)
    return evaluate(intent, cart, merchant_policy, ctx, now)


def _only_failed(decision: Decision) -> set[str]:
    return {c.code for c in decision.failed_checks}


# ---------------------------------------------------------------------
# Golden path
# ---------------------------------------------------------------------


def test_golden_scenario_is_approved_with_all_20_checks_passing():
    decision = _decide()
    assert decision.outcome == "APPROVED", decision.failed_checks
    assert decision.blocking_code is None
    assert len(decision.checks) == 20
    assert all(c.passed for c in decision.checks)


def test_golden_scenario_does_not_escalate_on_an_in_policy_cart():
    """The product invariant behind the one-touchpoint rule: an entirely
    in-policy purchase must reach APPROVED without pulling the human back
    in. A default 80%-of-budget tripwire used to make a cart at 90% of a
    budget the human themselves set escalate anyway -- deterministic
    nagging dressed up as a safety feature. G7.x must be silent here."""
    decision = _decide(
        budget_total_paise=50_000,
        budget_escalate_above_paise=50_000,  # no tripwire -- now expressible
        line_unit_price_paise=49_000,        # 98% of budget, still fine
    )
    assert decision.outcome == "APPROVED", decision.failed_checks
    escalate_codes = {"ESCALATION_THRESHOLD", "AMBIGUOUS_SUBSTITUTION", "LOW_CONFIDENCE_INTERPRETATION"}
    assert not (_only_failed(decision) & escalate_codes)


def test_all_20_gate_codes_always_present_regardless_of_outcome():
    """Every gate must run and be recorded even when the outcome is
    already decided by an earlier gate -- full audit visibility."""
    decision = _decide(line_unit_price_paise=999_999)  # trip several gates at once
    codes = {c.code for c in decision.checks}
    expected = {
        "MANDATE_EXPIRED", "SIGNATURE_INVALID", "MANDATE_CHAIN_BROKEN", "MERCHANT_NOT_AUTHORIZED",
        "AGENT_IDENTITY_UNVERIFIED", "PROVENANCE_INSUFFICIENT", "CART_ARITHMETIC_MISMATCH",
        "HARD_CONSTRAINT_VIOLATED", "CATEGORY_NOT_ALLOWED", "PER_ITEM_CAP_EXCEEDED",
        "QUANTITY_CAP_EXCEEDED", "BUDGET_EXCEEDED", "SUBSTITUTION_OUT_OF_TOLERANCE",
        "MERCHANT_POLICY_CONFLICT", "ESCALATION_THRESHOLD", "AMBIGUOUS_SUBSTITUTION",
        "LOW_CONFIDENCE_INTERPRETATION",
        "CATALOG_PRICE_MISMATCH", "CATALOG_ITEM_UNKNOWN", "CART_SHAPE_INVALID",
    }
    assert codes == expected


# ---------------------------------------------------------------------
# G0.1 MANDATE_EXPIRED
# ---------------------------------------------------------------------


def test_g0_1_intent_expired_rejects():
    decision = _decide(intent_expired=True)
    assert decision.outcome == "REJECTED"
    assert "MANDATE_EXPIRED" in _only_failed(decision)


def test_g0_1_cart_expired_rejects():
    decision = _decide(cart_expired=True)
    assert decision.outcome == "REJECTED"
    assert "MANDATE_EXPIRED" in _only_failed(decision)


# ---------------------------------------------------------------------
# G0.2 SIGNATURE_INVALID
# ---------------------------------------------------------------------


def test_g0_2_intent_signed_by_wrong_key_rejects():
    decision = _decide(sign_intent_with=OTHER_KP)
    assert decision.outcome == "REJECTED"
    assert "SIGNATURE_INVALID" in _only_failed(decision)


def test_g0_2_cart_signed_by_wrong_key_rejects():
    decision = _decide(sign_cart_with=OTHER_KP)
    assert decision.outcome == "REJECTED"
    assert "SIGNATURE_INVALID" in _only_failed(decision)


# ---------------------------------------------------------------------
# G0.3 MANDATE_CHAIN_BROKEN
# ---------------------------------------------------------------------


def test_g0_3_broken_intent_hash_rejects():
    decision = _decide(intent_hash_override="0" * 64)
    assert decision.outcome == "REJECTED"
    assert "MANDATE_CHAIN_BROKEN" in _only_failed(decision)


# ---------------------------------------------------------------------
# G0.4 MERCHANT_NOT_AUTHORIZED
# ---------------------------------------------------------------------


def test_g0_4_merchant_not_in_allowed_list_rejects():
    decision = _decide(allowed_merchants=("some-other-merchant",))
    assert decision.outcome == "REJECTED"
    assert "MERCHANT_NOT_AUTHORIZED" in _only_failed(decision)


def test_g0_4_none_allowed_merchants_means_any_merchant_ok():
    decision = _decide(allowed_merchants=None)
    assert "MERCHANT_NOT_AUTHORIZED" not in _only_failed(decision)


# ---------------------------------------------------------------------
# G0.5 AGENT_IDENTITY_UNVERIFIED (ablation config C4 toggle)
# ---------------------------------------------------------------------


def test_g0_5_not_required_passes_even_if_unverified():
    decision = _decide(require_agent_identity=False, webbotauth_verified=False)
    assert "AGENT_IDENTITY_UNVERIFIED" not in _only_failed(decision)


def test_g0_5_required_and_unverified_rejects():
    decision = _decide(require_agent_identity=True, webbotauth_verified=False)
    assert decision.outcome == "REJECTED"
    assert "AGENT_IDENTITY_UNVERIFIED" in _only_failed(decision)


def test_g0_5_required_and_verified_passes():
    decision = _decide(require_agent_identity=True, webbotauth_verified=True)
    assert "AGENT_IDENTITY_UNVERIFIED" not in _only_failed(decision)


# ---------------------------------------------------------------------
# G1.1 PROVENANCE_INSUFFICIENT -- the injection kill-switch
# ---------------------------------------------------------------------


def test_g1_1_llm_inferred_price_rejects():
    decision = _decide(price_extraction_method="llm_inferred", price_confidence=0.8)
    assert decision.outcome == "REJECTED"
    assert "PROVENANCE_INSUFFICIENT" in _only_failed(decision)


def test_g1_1_unverified_signature_rejects_even_if_schema_org():
    decision = _decide(price_extraction_method="schema_org", price_signature_verified=False)
    assert decision.outcome == "REJECTED"
    assert "PROVENANCE_INSUFFICIENT" in _only_failed(decision)


def test_g1_1_missing_provenance_rejects():
    decision = _decide(missing_price_provenance=True)
    assert decision.outcome == "REJECTED"
    assert "PROVENANCE_INSUFFICIENT" in _only_failed(decision)


def test_g1_1_verified_schema_org_passes():
    decision = _decide(price_extraction_method="schema_org", price_signature_verified=True)
    assert "PROVENANCE_INSUFFICIENT" not in _only_failed(decision)


def test_g1_1_merchant_signed_passes():
    decision = _decide(price_extraction_method="merchant_signed", price_signature_verified=True)
    assert "PROVENANCE_INSUFFICIENT" not in _only_failed(decision)


# ---------------------------------------------------------------------
# G2.1 CART_ARITHMETIC_MISMATCH
# ---------------------------------------------------------------------


def test_g2_1_wrong_subtotal_rejects():
    decision = _decide(declared_subtotal_paise=99_999)
    assert decision.outcome == "REJECTED"
    assert "CART_ARITHMETIC_MISMATCH" in _only_failed(decision)


def test_g2_1_wrong_total_rejects():
    decision = _decide(declared_total_paise=1)
    assert decision.outcome == "REJECTED"
    assert "CART_ARITHMETIC_MISMATCH" in _only_failed(decision)


def test_g2_1_correct_arithmetic_with_tax_and_shipping_passes():
    decision = _decide(
        line_unit_price_paise=1000, tax_paise=180, shipping_paise=50,
        declared_subtotal_paise=1000, declared_total_paise=1230,
    )
    assert "CART_ARITHMETIC_MISMATCH" not in _only_failed(decision)


# ---------------------------------------------------------------------
# G3.1 HARD_CONSTRAINT_VIOLATED
# ---------------------------------------------------------------------


def test_g3_1_category_mismatch_on_primary_line_rejects():
    decision = _decide(hard_category="grocery", line_category="electronics")
    assert decision.outcome == "REJECTED"
    assert "HARD_CONSTRAINT_VIOLATED" in _only_failed(decision)


def test_g3_1_category_mismatch_on_upsell_line_is_exempt():
    """Upsell lines don't need to satisfy the original item's hard
    constraints -- a complementary accessory can be a different category."""
    decision = _decide(hard_category="grocery", line_category="electronics", is_upsell=True)
    assert "HARD_CONSTRAINT_VIOLATED" not in _only_failed(decision)


def test_g3_1_required_attribute_mismatch_rejects():
    decision = _decide(required_attributes={"colour": "blue"}, line_attributes={"colour": "red"})
    assert decision.outcome == "REJECTED"
    assert "HARD_CONSTRAINT_VIOLATED" in _only_failed(decision)


def test_g3_1_required_attribute_match_passes():
    decision = _decide(required_attributes={"colour": "blue"}, line_attributes={"colour": "blue"})
    assert "HARD_CONSTRAINT_VIOLATED" not in _only_failed(decision)


# ---------------------------------------------------------------------
# G3.2 CATEGORY_NOT_ALLOWED (uniform, including upsells)
# ---------------------------------------------------------------------


def test_g3_2_excluded_category_rejects():
    decision = _decide(excluded_categories=["grocery"])
    assert decision.outcome == "REJECTED"
    assert "CATEGORY_NOT_ALLOWED" in _only_failed(decision)


def test_g3_2_excluded_category_applies_to_upsell_lines_too():
    decision = _decide(excluded_categories=["grocery"], is_upsell=True)
    assert "CATEGORY_NOT_ALLOWED" in _only_failed(decision)


# ---------------------------------------------------------------------
# G4.1 PER_ITEM_CAP_EXCEEDED -- boundary tested at the cap
# ---------------------------------------------------------------------


def test_g4_1_exactly_at_cap_passes():
    decision = _decide(budget_per_item_paise=30_000, line_unit_price_paise=30_000)
    assert "PER_ITEM_CAP_EXCEEDED" not in _only_failed(decision)


def test_g4_1_one_paise_over_cap_rejects():
    decision = _decide(budget_per_item_paise=30_000, line_unit_price_paise=30_001)
    assert decision.outcome == "REJECTED"
    assert "PER_ITEM_CAP_EXCEEDED" in _only_failed(decision)


def test_g4_1_one_paise_under_cap_passes():
    decision = _decide(budget_per_item_paise=30_000, line_unit_price_paise=29_999)
    assert "PER_ITEM_CAP_EXCEEDED" not in _only_failed(decision)


# ---------------------------------------------------------------------
# G4.2 QUANTITY_CAP_EXCEEDED -- boundary
# ---------------------------------------------------------------------


def test_g4_2_exactly_at_max_quantity_passes():
    decision = _decide(budget_max_quantity=3, line_quantity=3, line_unit_price_paise=1000)
    assert "QUANTITY_CAP_EXCEEDED" not in _only_failed(decision)


def test_g4_2_one_over_max_quantity_rejects():
    decision = _decide(budget_max_quantity=3, line_quantity=4, line_unit_price_paise=1000)
    assert decision.outcome == "REJECTED"
    assert "QUANTITY_CAP_EXCEEDED" in _only_failed(decision)


# ---------------------------------------------------------------------
# G4.3 BUDGET_EXCEEDED -- boundary
# ---------------------------------------------------------------------


def test_g4_3_exactly_at_budget_total_passes():
    decision = _decide(budget_total_paise=30_000, budget_escalate_above_paise=29_999, line_unit_price_paise=30_000)
    assert "BUDGET_EXCEEDED" not in _only_failed(decision)


def test_g4_3_one_paise_over_budget_total_rejects():
    decision = _decide(budget_total_paise=30_000, budget_escalate_above_paise=29_999, line_unit_price_paise=30_001)
    assert decision.outcome == "REJECTED"
    assert "BUDGET_EXCEEDED" in _only_failed(decision)


# ---------------------------------------------------------------------
# G5.1 SUBSTITUTION_OUT_OF_TOLERANCE
# ---------------------------------------------------------------------


def test_g5_1_no_target_price_means_gate_cannot_fire():
    decision = _decide(target_price_paise=None, is_substitute=True, line_unit_price_paise=999_999_00)
    # can't evaluate tolerance without a baseline -- must not be the gate
    # that blocks this (some OTHER gate like budget will, which is fine)
    assert "SUBSTITUTION_OUT_OF_TOLERANCE" not in _only_failed(decision)


def test_g5_1_substitute_within_tolerance_passes():
    # target 1000, tolerance min(buyer 10%, merchant 10%) = 10% -> up to 1100 ok
    decision = _decide(
        target_price_paise=1000, buyer_substitution_tolerance_pct=10.0, merchant_substitution_tolerance_pct=10.0,
        is_substitute=True, line_unit_price_paise=1100,
    )
    assert "SUBSTITUTION_OUT_OF_TOLERANCE" not in _only_failed(decision)


def test_g5_1_substitute_over_tolerance_rejects():
    decision = _decide(
        target_price_paise=1000, buyer_substitution_tolerance_pct=10.0, merchant_substitution_tolerance_pct=10.0,
        is_substitute=True, line_unit_price_paise=1101,
    )
    assert decision.outcome == "REJECTED"
    assert "SUBSTITUTION_OUT_OF_TOLERANCE" in _only_failed(decision)


def test_g5_1_uses_the_stricter_of_buyer_and_merchant_tolerance():
    # buyer allows 20%, merchant only allows 5% -> effective tolerance 5%
    decision = _decide(
        target_price_paise=1000, buyer_substitution_tolerance_pct=20.0, merchant_substitution_tolerance_pct=5.0,
        is_substitute=True, line_unit_price_paise=1060,  # 6% over -> exceeds merchant's 5%
    )
    assert decision.outcome == "REJECTED"
    assert "SUBSTITUTION_OUT_OF_TOLERANCE" in _only_failed(decision)


# ---------------------------------------------------------------------
# G6.1 MERCHANT_POLICY_CONFLICT
# ---------------------------------------------------------------------


def test_g6_1_over_merchant_max_order_value_rejects():
    decision = _decide(merchant_max_order_value_paise=10_000, line_unit_price_paise=10_001)
    assert decision.outcome == "REJECTED"
    assert "MERCHANT_POLICY_CONFLICT" in _only_failed(decision)


def test_g6_1_over_merchant_max_quantity_per_sku_rejects():
    decision = _decide(merchant_max_quantity_per_sku=1, line_quantity=2, budget_max_quantity=5, line_unit_price_paise=1000)
    assert decision.outcome == "REJECTED"
    assert "MERCHANT_POLICY_CONFLICT" in _only_failed(decision)


def test_g6_1_merchant_restricted_category_rejects():
    decision = _decide(merchant_restricted_categories=["grocery"])
    assert decision.outcome == "REJECTED"
    assert "MERCHANT_POLICY_CONFLICT" in _only_failed(decision)


# ---------------------------------------------------------------------
# G7.1 ESCALATION_THRESHOLD -- boundary, ESCALATE not REJECT
# ---------------------------------------------------------------------


def test_g7_1_exactly_at_escalation_threshold_passes():
    decision = _decide(budget_escalate_above_paise=30_000, line_unit_price_paise=30_000)
    assert "ESCALATION_THRESHOLD" not in _only_failed(decision)
    assert decision.outcome == "APPROVED"


def test_g7_1_one_paise_over_escalation_threshold_escalates_not_rejects():
    decision = _decide(budget_total_paise=50_000, budget_escalate_above_paise=30_000, line_unit_price_paise=30_001)
    assert decision.outcome == "REQUIRES_HUMAN_APPROVAL"
    assert "ESCALATION_THRESHOLD" in _only_failed(decision)


# ---------------------------------------------------------------------
# G7.2 AMBIGUOUS_SUBSTITUTION
# ---------------------------------------------------------------------


def test_g7_2_substitute_line_escalates():
    decision = _decide(is_substitute=True)
    assert decision.outcome == "REQUIRES_HUMAN_APPROVAL"
    assert "AMBIGUOUS_SUBSTITUTION" in _only_failed(decision)


def test_g7_2_no_substitute_passes():
    decision = _decide(is_substitute=False)
    assert "AMBIGUOUS_SUBSTITUTION" not in _only_failed(decision)


def test_g7_2_preapproved_substitute_passes_cleanly_not_bypassed():
    """The re-approval path (buyer/approval.py) works by re-signing an
    intent with substitutions_preapproved=True and re-running evaluate()
    from scratch -- this confirms the SAME gate resolves cleanly given
    that flag, rather than something short-circuiting around it."""
    decision = _decide(is_substitute=True, substitutions_preapproved=True)
    assert "AMBIGUOUS_SUBSTITUTION" not in _only_failed(decision)
    assert decision.outcome == "APPROVED"


# ---------------------------------------------------------------------
# G7.3 LOW_CONFIDENCE_INTERPRETATION
# ---------------------------------------------------------------------


def test_g7_3_low_confidence_flag_escalates():
    decision = _decide(low_confidence_interpretation=True)
    assert decision.outcome == "REQUIRES_HUMAN_APPROVAL"
    assert "LOW_CONFIDENCE_INTERPRETATION" in _only_failed(decision)


# ---------------------------------------------------------------------
# G8.x -- the anomaly band. See policy_engine.py's module docstring: the
# resolution rule used to end in `else APPROVED`, so approve was the
# DEFAULT whenever nothing enumerated fired. These gates bind a cart to
# what the merchant itself published, and fail closed on absence.
# ---------------------------------------------------------------------


def test_g8_1_the_overcharge_hole_that_all_seventeen_gates_missed():
    """The overcharge hole G8.1 exists to close: a merchant publishes an
    item at Rs200, signs a cart charging Rs480 for that same SKU, and
    stays comfortably inside a Rs500 budget. Every G0-G7 gate passes --
    valid signature, intact chain, merchant_signed provenance,
    consistent arithmetic, under every cap. Without G8.1 this would be
    APPROVED."""
    decision = _decide(
        budget_total_paise=50_000,
        budget_per_item_paise=50_000,
        budget_escalate_above_paise=50_000,
        line_unit_price_paise=48_000,   # what the merchant charges
        catalog_price_paise=20_000,     # what the merchant PUBLISHES
    )
    assert decision.outcome == "REJECTED"
    assert "CATALOG_PRICE_MISMATCH" in _only_failed(decision)

    # ...and the budget gates genuinely did NOT object -- which is the
    # whole point. Without G8.1 nothing here fires at all.
    assert "BUDGET_EXCEEDED" not in _only_failed(decision)
    assert "PER_ITEM_CAP_EXCEEDED" not in _only_failed(decision)
    assert "PROVENANCE_INSUFFICIENT" not in _only_failed(decision)


def test_g8_1_charging_exactly_the_published_price_passes():
    decision = _decide(line_unit_price_paise=30_000, catalog_price_paise=30_000)
    assert decision.outcome == "APPROVED", decision.failed_checks


def test_g8_1_one_paise_over_published_price_rejects():
    decision = _decide(line_unit_price_paise=30_001, catalog_price_paise=30_000)
    assert decision.outcome == "REJECTED"
    assert "CATALOG_PRICE_MISMATCH" in _only_failed(decision)


def test_g8_1_charging_less_than_published_is_fine():
    """A discount is the merchant's own business. G8.1 is one-directional
    on purpose: charging below your published price is not dishonesty."""
    decision = _decide(line_unit_price_paise=10_000, catalog_price_paise=30_000)
    assert decision.outcome == "APPROVED", decision.failed_checks


def test_g8_1_reports_the_overcharge_amount_as_evidence():
    decision = _decide(line_unit_price_paise=48_000, catalog_price_paise=20_000)
    check = decision.check_for("CATALOG_PRICE_MISMATCH")
    assert check is not None and not check.passed
    assert check.evidence["offending_lines"][0]["overcharge_paise"] == 28_000


def test_g8_1_missing_catalog_fails_closed():
    """The absence of a catalog is not permission. If the buyer could not
    verify what the merchant publishes, it cannot account for the price,
    and unaccounted-for is a stop."""
    decision = _decide(omit_catalog=True)
    assert decision.outcome == "REJECTED"
    assert "CATALOG_PRICE_MISMATCH" in _only_failed(decision)
    assert "CATALOG_ITEM_UNKNOWN" in _only_failed(decision)


def test_g8_1_another_merchants_catalog_cannot_vouch_for_this_cart():
    decision = _decide(catalog_merchant_id="some-other-merchant")
    assert decision.outcome == "REJECTED"
    assert "CATALOG_PRICE_MISMATCH" in _only_failed(decision)


def test_g8_2_sku_absent_from_published_catalog_rejects():
    """Closes the obvious way around G8.1: invent a SKU nobody published
    and there is no price to compare against."""
    decision = _decide(catalog_skus=("SOME-OTHER-SKU",))
    assert decision.outcome == "REJECTED"
    assert "CATALOG_ITEM_UNKNOWN" in _only_failed(decision)


def test_g8_2_category_laundering_rejects():
    """A line claiming a different category than the catalog publishes
    for that SKU is how an excluded-category item would slip past G3.2
    while still being priced legitimately."""
    decision = _decide(line_category="grocery", catalog_category="alcohol")
    assert decision.outcome == "REJECTED"
    assert "CATALOG_ITEM_UNKNOWN" in _only_failed(decision)


def test_g8_2_narrows_the_upsell_exemption_from_hard_constraints():
    """G3.1 deliberately exempts is_upsell lines from the buyer's hard
    constraints -- a pitched accessory legitimately isn't the thing that
    was asked for. G8.2 still requires it to be a real, correctly
    described item from this merchant's own published catalog, so the
    exemption is not a blank cheque."""
    decision = _decide(is_upsell=True, catalog_skus=("NOT-THE-UPSELL-SKU",))
    assert "HARD_CONSTRAINT_VIOLATED" not in _only_failed(decision)  # exemption intact
    assert "CATALOG_ITEM_UNKNOWN" in _only_failed(decision)          # but still bound
    assert decision.outcome == "REJECTED"


def test_g8_3_duplicate_skus_reject():
    """Fifty lines of the same SKU used to validate and evaluate without
    complaint -- spreading quantity across lines hides it from G4.2,
    which only ever looked at one line's quantity field."""
    dupe = CartLine(
        sku="SKU-1", title="Test Item", category="grocery", unit_price_paise=30_000, quantity=1,
        field_provenance={
            "unit_price_paise": Provenance(
                source_url="https://merchant.example/p", extraction_method="schema_org",
                signature_verified=True, extracted_at=_now(),
            )
        },
    )
    decision = _decide(extra_lines=[dupe], declared_subtotal_paise=60_000, declared_total_paise=60_000)
    assert decision.outcome == "REJECTED"
    assert "CART_SHAPE_INVALID" in _only_failed(decision)


def test_g8_3_too_many_lines_rejects():
    from buyer.policy_engine import MAX_CART_LINES

    extras = [
        CartLine(
            sku=f"SKU-EXTRA-{i}", title="Test Item", category="grocery",
            unit_price_paise=100, quantity=1,
            field_provenance={
                "unit_price_paise": Provenance(
                    source_url="https://merchant.example/p", extraction_method="schema_org",
                    signature_verified=True, extracted_at=_now(),
                )
            },
        )
        for i in range(MAX_CART_LINES)
    ]
    total = 30_000 + 100 * len(extras)
    decision = _decide(
        extra_lines=extras,
        declared_subtotal_paise=total,
        declared_total_paise=total,
        catalog_skus=("SKU-1", *(f"SKU-EXTRA-{i}" for i in range(MAX_CART_LINES))),
        catalog_price_paise=30_000,
    )
    assert decision.outcome == "REJECTED"
    assert "CART_SHAPE_INVALID" in _only_failed(decision)


def test_g8_3_more_than_one_upsell_line_rejects():
    second_upsell = CartLine(
        sku="SKU-2", title="Test Item", category="grocery", unit_price_paise=1_000, quantity=1,
        is_upsell=True,
        field_provenance={
            "unit_price_paise": Provenance(
                source_url="https://merchant.example/p", extraction_method="schema_org",
                signature_verified=True, extracted_at=_now(),
            )
        },
    )
    decision = _decide(
        is_upsell=True, extra_lines=[second_upsell],
        declared_subtotal_paise=31_000, declared_total_paise=31_000,
        catalog_skus=("SKU-1", "SKU-2"), catalog_price_paise=30_000,
    )
    assert decision.outcome == "REJECTED"
    assert "CART_SHAPE_INVALID" in _only_failed(decision)


def test_g8_3_a_single_upsell_line_is_fine():
    decision = _decide(is_upsell=True)
    assert "CART_SHAPE_INVALID" not in _only_failed(decision)


# ---------------------------------------------------------------------
# Outcome resolution: REJECT beats ESCALATE beats APPROVED
# ---------------------------------------------------------------------


def test_reject_wins_over_escalate_when_both_present():
    """A cart that is both over budget (REJECT) and contains a substitute
    (ESCALATE) must resolve to REJECTED, with both checks still recorded
    as failed in the audit trail."""
    decision = _decide(
        budget_total_paise=1000, budget_escalate_above_paise=999, line_unit_price_paise=1001, is_substitute=True,
    )
    assert decision.outcome == "REJECTED"
    failed = _only_failed(decision)
    assert "BUDGET_EXCEEDED" in failed
    assert "AMBIGUOUS_SUBSTITUTION" in failed  # still recorded, even though it didn't decide the outcome


def test_multiple_escalations_still_resolves_to_single_requires_approval():
    decision = _decide(is_substitute=True, low_confidence_interpretation=True)
    assert decision.outcome == "REQUIRES_HUMAN_APPROVAL"
    failed = _only_failed(decision)
    assert "AMBIGUOUS_SUBSTITUTION" in failed
    assert "LOW_CONFIDENCE_INTERPRETATION" in failed


# ---------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------


def test_purity_1000_identical_calls_produce_identical_decisions():
    intent, cart, merchant_policy, ctx, now = _scenario()
    first = evaluate(intent, cart, merchant_policy, ctx, now)
    for _ in range(1000):
        again = evaluate(intent, cart, merchant_policy, ctx, now)
        assert again == first


def test_fail_closed_on_internal_exception(monkeypatch):
    """Mechanically verifies the try/except in evaluate() actually catches
    and converts to REJECTED/ENGINE_ERROR -- fail closed, never fail open.
    There's no way to trigger a genuine exception through valid pydantic
    inputs, so this monkeypatches one gate function to simulate a bug."""
    import buyer.policy_engine as pe

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated bug in a gate function")

    monkeypatch.setattr(pe, "_g4_3_budget_exceeded", _boom)
    intent, cart, merchant_policy, ctx, now = _scenario()
    decision = pe.evaluate(intent, cart, merchant_policy, ctx, now)
    assert decision.outcome == "REJECTED"
    assert decision.blocking_code == "ENGINE_ERROR"
    assert len(decision.checks) == 1
    assert "simulated bug" in decision.checks[0].detail
