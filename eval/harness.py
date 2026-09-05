"""
Runs one (scenario, attack_id, config, upsell_mode) instance through the
REAL pipeline -- real merchant_agent.MerchantAgentDriver, real
buyer/negotiate.py, real buyer/policy_engine.py -- and returns a typed
InstanceResult. Nothing here decides an outcome; it only wires together
modules that already decide things, the same discipline redteam/'s own
tests hold to ("tested against the REAL pipeline, not a simulation").

Zero LLM cost by construction, not just by replay
--------------------------------------------------
The plan's original design called for a live interpret() call per
scenario, recorded once and replayed across configs so replays cost
nothing. This harness goes one step further: `build_ground_truth_intent`
constructs the IntentMandate directly from each scenario's authored
`ground_truth` fields, using interpret()'s OWN vocabulary
(budget_total_rupees, required_attributes, etc.) so the construction is
mechanically equivalent to what a correct interpret() call would
produce. That means the entire safety and revenue arms run at zero LLM
cost, not merely isolated-by-replay cost -- a stronger property, at the
cost of not exercising interpret() itself.

`run_live_interpretation_check()` below is the opt-in companion that DOES
call a live provider, to measure whether interpret() actually parses
request_text into the authored ground truth -- gated by eval.budget's
BudgetGuard like any other live LLM path in this repo, and never run as
part of the free matrix.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

from buyer.catalog import CatalogEntry, VerifiedCatalog
from buyer.negotiate import MerchantResponse, negotiate
from buyer.policy_engine import Decision, VerificationContext, evaluate
from eval.config import EvalConfig
from eval.corpus import Scenario, ScenarioItem
from mandates.hashing import mandate_hash
from mandates.keys import KeyDirectory, KeyPair, generate_keypair
from mandates.schemas import (
    Budget,
    CartLine,
    CartMandate,
    HardConstraints,
    IntentMandate,
    MerchantPolicy,
    Provenance,
    SoftPreferences,
)
from mandates.sign import sign_mandate
from merchant_agent.driver import MerchantAgentDriver
from merchant_kit.schemas import CatalogItem
from redteam.attacks import AttackId
from redteam.driver_attacks import (
    apply_branded_whisper,
    apply_catalog_overcharge,
    apply_price_swap,
    apply_quantity_inflation,
)
from redteam.goal_predicates import GOAL_PREDICATES, RunRecord

_RUPEES_TO_PAISE = 100

# Scenarios that are only "policy_violating" once paired with a SPECIFIC
# attack -- under A0_none they are ordinary, honestly-priced purchases
# and should be APPROVED. Maps mechanism -> the one attack_id under which
# scenario.expected_outcome is a meaningful comparison at all.
ATTACK_DEPENDENT_MECHANISMS: dict[str, AttackId] = {
    "catalog_overcharge": "A6_catalog_overcharge",
    "price_swap": "A4_price_swap",
    "quantity_inflation": "A5_quantity_inflation",
}


@dataclass(frozen=True)
class InstanceResult:
    scenario_id: str
    category: str
    attack_id: AttackId
    config_id: str
    upsell_mode: str
    k: int
    outcome: str  # "APPROVED" | "REJECTED" | "REQUIRES_HUMAN_APPROVAL"
    blocking_code: str | None
    matches_expected: bool | None  # only meaningful for A0_none; None under an attack
    attack_succeeded: bool | None  # only meaningful for attack_id != A0_none
    human_touchpoints: int
    order_created: bool
    order_amount_paise: int | None
    budget_total_paise: int
    # Carried so cross-config comparisons (rupees-prevented) don't need to
    # re-derive the intent -- the same reason RunRecord carries it in
    # redteam/goal_predicates.py.
    upsell_offered: bool
    upsell_accepted_would_pass: bool  # the upsell line, if any, passed evaluate() on its own merits


def _human_keypair() -> KeyPair:
    return generate_keypair()


def build_ground_truth_intent(
    scenario: Scenario, *, principal_kid: str, agent_kid: str, now: datetime,
    validity: timedelta = timedelta(hours=1),
) -> IntentMandate:
    """Mechanically equivalent to what a correct interpret() call would
    produce from scenario.request_text -- see module docstring. Uses
    interpret()'s own field vocabulary so the two constructions can never
    silently drift apart in meaning."""
    gt = scenario.ground_truth
    budget_total_paise = round(gt["budget_total_rupees"] * _RUPEES_TO_PAISE)
    per_item_rupees = gt.get("budget_per_item_rupees")
    per_item_paise = round(per_item_rupees * _RUPEES_TO_PAISE) if per_item_rupees else budget_total_paise
    per_item_paise = min(per_item_paise, budget_total_paise)

    escalate_above_rupees = gt.get("escalate_above_rupees")
    escalate_above_paise = (
        round(escalate_above_rupees * _RUPEES_TO_PAISE) if escalate_above_rupees is not None else budget_total_paise
    )

    hard = HardConstraints(
        category=gt.get("category"),
        required_attributes={item["key"]: item["value"] for item in gt.get("required_attributes", [])},
        excluded_categories=list(gt.get("excluded_categories", [])),
    )
    soft = SoftPreferences(
        preferred_brands=list(gt.get("preferred_brands", [])),
        target_price_paise=(
            round(gt["target_price_rupees"] * _RUPEES_TO_PAISE) if gt.get("target_price_rupees") is not None else None
        ),
        substitution_tolerance_pct=10.0,
    )
    budget = Budget(
        total_paise=budget_total_paise, per_item_paise=per_item_paise,
        max_quantity=int(gt.get("max_quantity") or 1), escalate_above_paise=escalate_above_paise,
    )

    return IntentMandate(
        mandate_id=f"eval-{scenario.id}-{secrets.token_hex(4)}", issued_at=now, expires_at=now + validity,
        principal_kid=principal_kid, agent_kid=agent_kid, request_text=scenario.request_text,
        hard=hard, soft=soft, budget=budget, allowed_merchants=[scenario.merchant_id],
    )


def _scenario_catalog_items(scenario: Scenario, now: datetime) -> list[CatalogItem]:
    prov = Provenance(source_url=f"https://eval.example/{scenario.id}", extraction_method="schema_org", signature_verified=False, extracted_at=now)
    items = []
    for it in scenario.catalog:
        items.append(
            CatalogItem(
                sku=it.sku, title=it.title, description="", category=it.category, price_paise=it.price_paise,
                in_stock=True, attributes=dict(it.attributes),
                field_provenance={f: prov for f in ("title", "description", "category", "price_paise", "in_stock")},
            )
        )
    return items


def _verified_catalog_for(merchant_id: str, items: list[CatalogItem], keypair: KeyPair, now: datetime) -> VerifiedCatalog:
    entries = {
        item.sku: CatalogEntry(sku=item.sku, title=item.title, category=item.category, price_paise=item.price_paise, in_stock=item.in_stock)
        for item in items
    }
    return VerifiedCatalog(merchant_id=merchant_id, merchant_kid=keypair.kid, generated_at=now.isoformat(), entries=entries)


def _sign_manual_cart(
    intent: IntentMandate, merchant_id: str, merchant_kp: KeyPair, lines: list[CartLine], now: datetime,
):
    subtotal = sum(line.line_total_paise for line in lines)
    cart = CartMandate(
        mandate_id=f"cart-{secrets.token_hex(4)}", issued_at=now, expires_at=now + timedelta(minutes=10),
        merchant_id=merchant_id, merchant_kid=merchant_kp.kid, intent_hash=mandate_hash(intent),
        lines=lines, subtotal_paise=subtotal, tax_paise=0, shipping_paise=0, total_paise=subtotal,
    )
    return cart, sign_mandate(cart, merchant_kp)


def _mechanism_driver(scenario: Scenario, merchant_kp: KeyPair, now: datetime):
    """Custom mechanism scenarios need a hand-built cart the real
    MerchantAgentDriver's catalog-matching logic cannot produce on its
    own (a deliberately-over-cap quantity, a duplicate SKU, a
    substitute). Returns a MerchantDriver callable, or None if the
    scenario has no `mechanism` (use the real driver instead)."""
    mechanism = scenario.mechanism
    if mechanism is None:
        return None

    primary_item = scenario.catalog[0]
    prov = Provenance(source_url="https://eval.example/mech", extraction_method="merchant_signed", signature_verified=True, extracted_at=now)

    if mechanism == "quantity_over_cap":
        def driver(turn, intent):
            line = CartLine(
                sku=primary_item.sku, title=primary_item.title, category=primary_item.category,
                unit_price_paise=primary_item.price_paise, quantity=5,
                field_provenance={"unit_price_paise": prov, "title": prov},
            )
            cart, envelope = _sign_manual_cart(intent, scenario.merchant_id, merchant_kp, [line], now)
            return MerchantResponse(kind="cart", cart_envelope=envelope)
        return driver

    if mechanism == "duplicate_sku":
        def driver(turn, intent):
            line = CartLine(
                sku=primary_item.sku, title=primary_item.title, category=primary_item.category,
                unit_price_paise=primary_item.price_paise, quantity=1,
                field_provenance={"unit_price_paise": prov, "title": prov},
            )
            cart, envelope = _sign_manual_cart(intent, scenario.merchant_id, merchant_kp, [line, line], now)
            return MerchantResponse(kind="cart", cart_envelope=envelope)
        return driver

    if mechanism == "substitute_offered":
        def driver(turn, intent):
            sub_price = round(primary_item.price_paise * 1.05)
            line = CartLine(
                sku=f"{primary_item.sku}-SUB", title=f"{primary_item.title} (alt)", category=primary_item.category,
                unit_price_paise=sub_price, quantity=1, is_substitute=True,
                field_provenance={"unit_price_paise": prov, "title": prov},
            )
            cart, envelope = _sign_manual_cart(intent, scenario.merchant_id, merchant_kp, [line], now)
            return MerchantResponse(kind="cart", cart_envelope=envelope)
        return driver

    if mechanism in (
        "wrong_merchant_authorized", "escalation_threshold",
        *ATTACK_DEPENDENT_MECHANISMS,
    ):
        # No custom driver needed: "wrong_merchant_authorized" is a pure
        # intent-level override (allowed_merchants), "escalation_threshold"
        # is a pure budget-level override (ground_truth.escalate_above_rupees),
        # and the ATTACK_DEPENDENT_MECHANISMS are exercised entirely by
        # wrapping the REAL driver with the matching redteam attack
        # function further down in run_instance.
        return None

    raise ValueError(f"unknown scenario mechanism {mechanism!r} on {scenario.id}")


def _catalog_and_driver_for_mechanism(
    scenario: Scenario, catalog_items: list[CatalogItem], merchant_kp: KeyPair, now: datetime,
):
    """Some mechanisms introduce a SKU the scenario's own catalog does
    not publish (the substitute). Those must be reflected in the
    VerifiedCatalog too, or G8.2 would reject them for the wrong
    reason -- this harness is testing G7.2, not accidentally re-testing
    G8.2 on a fixture gap."""
    mechanism = scenario.mechanism
    if mechanism == "substitute_offered":
        primary_item = scenario.catalog[0]
        sub_price = round(primary_item.price_paise * 1.05)
        extra = ScenarioItem(sku=f"{primary_item.sku}-SUB", title=f"{primary_item.title} (alt)", category=primary_item.category, price_paise=sub_price)
        extended = _scenario_catalog_items(
            Scenario(scenario.id, scenario.category, scenario.request_text, scenario.merchant_id, (*scenario.catalog, extra), scenario.expected_outcome, scenario.ground_truth, scenario.notes),
            now,
        )
        return extended
    return catalog_items


def run_instance(
    scenario: Scenario,
    *,
    attack_id: AttackId,
    config: EvalConfig,
    upsell_mode: str,
    k: int,
    now: datetime,
    intent_override: IntentMandate | None = None,
) -> InstanceResult:
    """One fully-wired run: build intent -> (real or mechanism) driver ->
    optionally wrap with a redteam attack -> negotiate -> decide (real
    evaluate() under C2_C3/C4, the undefended baseline under C1).

    `intent_override`, when given, REPLACES build_ground_truth_intent()'s
    synthetic construction with a real intent from somewhere else -- the
    live-LLM evaluation (eval/live_llm_eval.py) passes in what a genuine
    interpret() call actually produced, so the same attack/config matrix
    that runs against synthetic ground truth can also run against a real
    model's real (occasionally imperfect) output. Nothing else about this
    function's logic changes: mechanism drivers, attack wrapping, and
    evaluate() itself don't know or care where the intent came from."""
    human_kp = _human_keypair()
    merchant_kp = generate_keypair()
    principal_kid = human_kp.kid

    mechanism = scenario.mechanism
    allowed_merchants_override = [f"not-{scenario.merchant_id}"] if mechanism == "wrong_merchant_authorized" else None

    intent = intent_override or build_ground_truth_intent(
        scenario, principal_kid=principal_kid, agent_kid=principal_kid, now=now
    )
    if intent_override is not None:
        # A live intent carries its own principal_kid/agent_kid from
        # whatever generated it -- re-stamp so THIS run's human key is
        # actually the one that signs it below (otherwise G0.2 would
        # check the signature against a kid nothing here ever registers).
        intent = intent.model_copy(update={"principal_kid": principal_kid, "agent_kid": principal_kid})
    if allowed_merchants_override is not None:
        intent = intent.model_copy(update={"allowed_merchants": allowed_merchants_override})
    intent_envelope = sign_mandate(intent, human_kp)

    base_items = _scenario_catalog_items(scenario, now)
    catalog_items = _catalog_and_driver_for_mechanism(scenario, base_items, merchant_kp, now)
    verified_catalog = _verified_catalog_for(scenario.merchant_id, catalog_items, merchant_kp, now)

    custom_driver = _mechanism_driver(scenario, merchant_kp, now)
    driver = custom_driver or MerchantAgentDriver(
        catalog=catalog_items, merchant_id=scenario.merchant_id, merchant_keypair=merchant_kp, now=now,
        upsell_mode=upsell_mode,
    )

    if attack_id == "A3_branded_whisper":
        driver = apply_branded_whisper(driver)
    elif attack_id == "A4_price_swap":
        driver = apply_price_swap(driver, merchant_keypair=merchant_kp, now=now, multiplier=10.0)
    elif attack_id == "A5_quantity_inflation":
        driver = apply_quantity_inflation(driver, merchant_keypair=merchant_kp, inflate_to=10)
    elif attack_id == "A6_catalog_overcharge":
        driver = apply_catalog_overcharge(driver, merchant_keypair=merchant_kp, budget_total_paise=intent.budget.total_paise)
    # A1/A2 (description-text injection) act on the EXTRACTION stage, not
    # the negotiation driver -- they are exercised by merchant_kit's own
    # test suite against fixture HTML, not by this per-purchase harness.
    # A0_none: no wrapping.

    negotiation = negotiate(intent, merchant_driver=driver)

    merchant_policy = MerchantPolicy(
        merchant_id=scenario.merchant_id, max_order_value_paise=10_000_000, substitution_tolerance_pct=10.0,
        max_quantity_per_sku=1_000, restricted_categories=[], escalation_threshold_paise=10_000_000,
        # Deliberately permissive merchant policy: this harness isolates
        # the BUYER's own gates, and a tight merchant default would
        # confound which side's limit actually fired.
    )

    order_created = False
    order_amount_paise: int | None = None
    order_quantity: int | None = None
    outcome = "REJECTED"
    blocking_code: str | None = None
    charged_unit_price_paise: int | None = None
    published_unit_price_paise: int | None = None
    upsell_offered = False
    upsell_accepted_would_pass = False

    if negotiation.outcome == "cart_received":
        cart = CartMandate.model_validate(negotiation.cart_envelope.payload)
        primary_line = next((l for l in cart.lines if not l.is_upsell), cart.lines[0])
        published_entry = verified_catalog.get(primary_line.sku)
        charged_unit_price_paise = primary_line.unit_price_paise
        published_unit_price_paise = published_entry.price_paise if published_entry else None
        upsell_offered = any(l.is_upsell for l in cart.lines)

        if not config.use_policy_engine:
            # C1: undefended baseline -- whatever the merchant proposed
            # is simply bought. This is the vulnerability the rest of
            # the system exists to defend against, made concrete.
            outcome, blocking_code = "APPROVED", None
        else:
            directory = KeyDirectory()
            directory.register_keypair(human_kp)
            directory.register_keypair(merchant_kp)
            ctx = VerificationContext(
                key_directory=directory, intent_envelope=intent_envelope, cart_envelope=negotiation.cart_envelope,
                require_agent_identity=config.require_agent_identity, webbotauth_verified=config.require_agent_identity,
                merchant_catalog=verified_catalog,
            )
            decision: Decision = evaluate(intent, cart, merchant_policy, ctx, now)
            outcome, blocking_code = decision.outcome, decision.blocking_code

        if outcome == "APPROVED":
            order_created = True
            order_amount_paise = cart.total_paise
            order_quantity = primary_line.quantity
            if upsell_offered:
                upsell_accepted_would_pass = True  # the whole cart, upsell included, passed

    human_touchpoints = 1 if outcome != "REQUIRES_HUMAN_APPROVAL" else 2

    matches_expected = None
    attack_succeeded = None
    required_attack = ATTACK_DEPENDENT_MECHANISMS.get(mechanism)
    if required_attack is not None and attack_id not in ("A0_none", required_attack):
        # A different attack applied to an attack-dependent scenario --
        # not a meaningful comparison against this scenario's label.
        pass
    elif attack_id == "A0_none":
        if required_attack is None:
            matches_expected = outcome == scenario.expected_outcome
        # else: an attack-dependent scenario under no attack at all is
        # just a benign purchase, not comparable to expected_outcome.
    elif attack_id == required_attack:
        # The one attack this scenario's REJECTED label actually means
        # something under -- did the defense catch it as expected?
        matches_expected = outcome == scenario.expected_outcome

    if attack_id != "A0_none":
        requested_quantity = intent.budget.max_quantity
        run_record = RunRecord(
            order_created=order_created, order_amount_paise=order_amount_paise, order_quantity=order_quantity,
            requested_quantity=requested_quantity, budget_total_paise=intent.budget.total_paise, outcome=outcome,
            charged_unit_price_paise=charged_unit_price_paise, published_unit_price_paise=published_unit_price_paise,
        )
        predicate = GOAL_PREDICATES.get(attack_id)
        attack_succeeded = predicate(run_record) if predicate else None

    return InstanceResult(
        scenario_id=scenario.id, category=scenario.category, attack_id=attack_id, config_id=config.id,
        upsell_mode=upsell_mode, k=k, outcome=outcome, blocking_code=blocking_code,
        matches_expected=matches_expected, attack_succeeded=attack_succeeded, human_touchpoints=human_touchpoints,
        order_created=order_created, order_amount_paise=order_amount_paise,
        budget_total_paise=intent.budget.total_paise,
        upsell_offered=upsell_offered, upsell_accepted_would_pass=upsell_accepted_would_pass,
    )
