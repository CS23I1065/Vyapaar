"""
merchant_agent/ tests -- the revenue half of the track. Two things this
suite exists specifically to prove:

1. The buyer-budget asymmetry is structural, not just observed behavior:
   upsell.py's candidate-selection functions don't even ACCEPT an
   IntentMandate parameter (checked via inspect.signature, not a fragile
   text grep), so there is literally nothing in scope for them to read
   `intent.budget` from.
2. A full integration through negotiate() + evaluate(): an in-policy
   upsell gets APPROVED for real, and an over-budget one gets REJECTED
   for real -- the same policy engine, unmodified, deciding both.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

from buyer.negotiate import negotiate
from buyer.catalog import VerifiedCatalog, load_verified_catalog
from buyer.policy_engine import VerificationContext, evaluate
from mandates import (
    Budget,
    HardConstraints,
    IntentMandate,
    KeyDirectory,
    MerchantPolicy,
    Provenance,
    SoftPreferences,
    generate_keypair,
    sign_mandate,
)
from mandates.schemas import CartMandate, Envelope
from merchant_agent.driver import MerchantAgentDriver, _match_primary
from merchant_agent.upsell import find_accessory, find_premium_substitute, find_quantity_bump, select_one_upsell
from merchant_kit.schemas import Catalog, CatalogItem

UTC = timezone.utc
HUMAN_KP = generate_keypair()
MERCHANT_KP = generate_keypair()


def _now() -> datetime:
    return datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)


def _prov(now: datetime) -> Provenance:
    return Provenance(source_url="https://merchant.example/p", extraction_method="schema_org", signature_verified=False, extracted_at=now)


def _catalog_item(sku: str, title: str, category: str, price_paise: int, **attrs) -> CatalogItem:
    now = _now()
    prov = _prov(now)
    return CatalogItem(
        sku=sku, title=title, description="", category=category, price_paise=price_paise, in_stock=True,
        attributes=attrs, field_provenance={f: prov for f in ("title", "description", "category", "price_paise", "in_stock")},
    )


CATALOG = [
    _catalog_item("RICE-1", "Basmati Rice 2kg", "grains", 30_000),
    _catalog_item("PICKLE-1", "Mango Pickle 200g", "condiments", 12_000),  # different category, cheaper -> accessory
    _catalog_item("RICE-PREMIUM", "Aged Basmati Rice 2kg", "grains", 45_000),  # same category, pricier -> premium sub
    _catalog_item("SHIRT-1", "Blue Cotton Shirt", "apparel", 80_000),
]


def _intent(**overrides) -> IntentMandate:
    defaults = dict(
        mandate_id="intent-1", issued_at=_now(), expires_at=_now() + timedelta(hours=1),
        principal_kid=HUMAN_KP.kid, agent_kid="agent-1", request_text="buy rice",
        hard=HardConstraints(category="grains"), soft=SoftPreferences(substitution_tolerance_pct=10.0),
        budget=Budget(total_paise=50_000, per_item_paise=50_000, max_quantity=2, escalate_above_paise=45_000),
        allowed_merchants=["merchant-1"],
    )
    defaults.update(overrides)
    return IntentMandate(**defaults)


def _merchant_policy() -> MerchantPolicy:
    return MerchantPolicy(
        merchant_id="merchant-1", max_order_value_paise=1_000_000, substitution_tolerance_pct=10.0,
        max_quantity_per_sku=5, restricted_categories=[], escalation_threshold_paise=999_999,
    )


def _verified_catalog(items=None) -> VerifiedCatalog:
    """Signs the same in-process catalog the merchant driver sells from,
    then loads it back through the REAL verification path -- so these
    tests exercise load_verified_catalog(), not a shortcut around it. The
    G8.x band binds a cart to what the merchant published, and the whole
    point is that "published" means signature-verified."""
    doc = Catalog(merchant_id="merchant-1", generated_at=_now().isoformat(), items=items or CATALOG)
    envelope = sign_mandate(doc, MERCHANT_KP)
    directory = KeyDirectory()
    directory.register_keypair(MERCHANT_KP)
    return load_verified_catalog(
        envelope, directory, expected_merchant_id="merchant-1", required_kid=MERCHANT_KP.kid
    )

def _ctx(intent_envelope: Envelope, cart_envelope: Envelope, *, catalog=None) -> VerificationContext:
    directory = KeyDirectory()
    directory.register_keypair(HUMAN_KP)
    directory.register_keypair(MERCHANT_KP)
    return VerificationContext(
        key_directory=directory, intent_envelope=intent_envelope, cart_envelope=cart_envelope,
        merchant_catalog=catalog if catalog is not None else _verified_catalog(),
    )


# ---------------------------------------------------------------------
# Structural asymmetry: enforced by signature, not convention
# ---------------------------------------------------------------------


def test_upsell_functions_never_accept_an_intent_parameter():
    """The merchant genuinely cannot read the buyer's budget when
    selecting an upsell -- not "chooses not to," structurally cannot,
    because IntentMandate is never in scope. Checked via inspect, which
    catches this even if someone later renames things in a way a text
    grep for ".budget" would miss."""
    for fn in (find_accessory, find_premium_substitute, find_quantity_bump, select_one_upsell):
        params = inspect.signature(fn).parameters
        for name, param in params.items():
            annotation = str(param.annotation)
            assert "IntentMandate" not in annotation, f"{fn.__name__} parameter {name!r} accepts an IntentMandate"
        assert "intent" not in params, f"{fn.__name__} has an 'intent' parameter"


# ---------------------------------------------------------------------
# upsell.py candidate selection
# ---------------------------------------------------------------------


def test_find_accessory_picks_cheapest_cheaper_item():
    primary = CATALOG[0]  # rice, 30_000
    candidate = find_accessory(primary, CATALOG)
    assert candidate is not None
    assert candidate.kind == "ACCESSORY"
    assert candidate.item.sku == "PICKLE-1"


def test_find_premium_substitute_picks_cheapest_pricier_item():
    primary = CATALOG[0]
    candidate = find_premium_substitute(primary, CATALOG)
    assert candidate is not None
    assert candidate.kind == "PREMIUM_SUB"
    assert candidate.item.sku == "RICE-PREMIUM"


def test_find_quantity_bump_uses_same_sku():
    primary = CATALOG[0]
    candidate = find_quantity_bump(primary)
    assert candidate.kind == "QUANTITY"
    assert candidate.item.sku == primary.sku
    assert candidate.quantity == 2


def test_select_one_upsell_prefers_premium_when_available():
    candidate = select_one_upsell(CATALOG[0], CATALOG)
    assert candidate.kind == "PREMIUM_SUB"
    assert candidate.item.sku == "RICE-PREMIUM"


def test_select_one_upsell_falls_back_when_no_premium():
    lonely_item = _catalog_item("SOLO-1", "Solo Item", "unique-category", 5_000)
    candidate = select_one_upsell(lonely_item, [lonely_item])
    assert candidate.kind == "QUANTITY"  # only option left, since no other item shares its category


# ---------------------------------------------------------------------
# driver.py -- primary matching
# ---------------------------------------------------------------------


def test_match_primary_finds_cheapest_matching_item():
    intent = _intent(hard=HardConstraints(category="grains"))
    match = _match_primary(intent, CATALOG)
    assert match.sku == "RICE-1"  # cheapest of the two "grains" items (RICE-1 vs RICE-PREMIUM)


def test_match_primary_none_when_no_category_match():
    intent = _intent(hard=HardConstraints(category="electronics"))
    assert _match_primary(intent, CATALOG) is None


def test_driver_declines_when_no_match():
    driver = MerchantAgentDriver(catalog=CATALOG, merchant_id="merchant-1", merchant_keypair=MERCHANT_KP, now=_now())
    response = driver(1, _intent(hard=HardConstraints(category="electronics")))
    assert response.kind == "decline"


# ---------------------------------------------------------------------
# driver.py -- upsell mode behavior
# ---------------------------------------------------------------------


def test_driver_off_mode_never_upsells():
    driver = MerchantAgentDriver(catalog=CATALOG, merchant_id="merchant-1", merchant_keypair=MERCHANT_KP, now=_now(), upsell_mode="off")
    response = driver(1, _intent())
    assert response.kind == "cart"
    cart = CartMandate.model_validate(response.cart_envelope.payload)
    assert len(cart.lines) == 1
    assert response.pitch_text is None


def test_driver_benign_mode_adds_one_upsell_line():
    driver = MerchantAgentDriver(catalog=CATALOG, merchant_id="merchant-1", merchant_keypair=MERCHANT_KP, now=_now(), upsell_mode="benign")
    response = driver(1, _intent())
    assert response.kind == "cart"
    cart = CartMandate.model_validate(response.cart_envelope.payload)
    assert cart.lines[0].is_upsell is True
    # The clean alternative should be in the alternative envelopes
    assert len(response.alternative_envelopes) > 0
    alt_cart = CartMandate.model_validate(response.alternative_envelopes[0].payload)
    assert alt_cart.lines[0].sku == "RICE-1"
    assert not alt_cart.lines[0].is_upsell
    assert response.pitch_text is not None


def test_driver_benign_mode_offers_at_most_once_per_session():
    driver = MerchantAgentDriver(catalog=CATALOG, merchant_id="merchant-1", merchant_keypair=MERCHANT_KP, now=_now(), upsell_mode="benign")
    first = driver(1, _intent())
    second = driver(2, _intent())  # same driver instance -- simulates a second turn in one negotiation
    cart1 = CartMandate.model_validate(first.cart_envelope.payload)
    cart2 = CartMandate.model_validate(second.cart_envelope.payload)
    assert any(line.is_upsell for line in cart1.lines)
    assert not any(line.is_upsell for line in cart2.lines)


def test_driver_cart_signature_verifies():
    from mandates import verify_envelope

    driver = MerchantAgentDriver(catalog=CATALOG, merchant_id="merchant-1", merchant_keypair=MERCHANT_KP, now=_now())
    response = driver(1, _intent())
    directory = KeyDirectory()
    directory.register_keypair(MERCHANT_KP)
    result = verify_envelope(response.cart_envelope, directory, required_kid=MERCHANT_KP.kid)
    assert result.valid, result.reason


# ---------------------------------------------------------------------
# Full integration: negotiate() + evaluate() -- the actual revenue claim
# ---------------------------------------------------------------------


def test_integration_in_policy_upsell_gets_approved_for_real():
    """The demo's revenue beat: an upsell that fits within the buyer's
    own signed budget gets APPROVED by the SAME unmodified policy engine
    that would reject an over-budget one below."""
    intent = _intent(budget=Budget(total_paise=80_000, per_item_paise=50_000, max_quantity=2, escalate_above_paise=75_000))
    intent_envelope = sign_mandate(intent, HUMAN_KP)

    driver = MerchantAgentDriver(catalog=CATALOG, merchant_id="merchant-1", merchant_keypair=MERCHANT_KP, now=_now(), upsell_mode="benign")
    negotiation = negotiate(intent, merchant_driver=driver)
    assert negotiation.outcome == "cart_received"

    cart = CartMandate.model_validate(negotiation.cart_envelope.payload)
    assert cart.total_paise == 45_000  # premium sub upsell replaces the 30k item
    assert len(cart.lines) == 1
    assert cart.lines[0].is_upsell is True

    ctx = _ctx(intent_envelope, negotiation.cart_envelope)
    decision = evaluate(intent, cart, _merchant_policy(), ctx, _now())
    assert decision.outcome == "APPROVED", decision.failed_checks


def test_integration_over_budget_upsell_gets_rejected_for_real():
    """Same merchant, same upsell logic, but a tight budget that the
    upsell pushes over -- must REJECT, not silently drop the upsell or
    approve anyway. This is the other half of the "guardrails don't cost
    revenue, but don't get overridden either" claim."""
    intent = _intent(budget=Budget(total_paise=35_000, per_item_paise=35_000, max_quantity=2, escalate_above_paise=30_000))
    intent_envelope = sign_mandate(intent, HUMAN_KP)

    driver = MerchantAgentDriver(catalog=CATALOG, merchant_id="merchant-1", merchant_keypair=MERCHANT_KP, now=_now(), upsell_mode="benign")
    negotiation = negotiate(intent, merchant_driver=driver)
    assert negotiation.outcome == "cart_received"

    cart = CartMandate.model_validate(negotiation.cart_envelope.payload)
    assert cart.total_paise == 45_000  # 45_000 -- over the 35_000 budget

    ctx = _ctx(intent_envelope, negotiation.cart_envelope)
    decision = evaluate(intent, cart, _merchant_policy(), ctx, _now())
    assert decision.outcome == "REJECTED"
    assert "BUDGET_EXCEEDED" in {c.code for c in decision.failed_checks}
