"""
Tests for buyer/shop.py -- multi-merchant comparison. Real
MerchantAgentDriver instances, real negotiate() and evaluate() -- the
same "full integration, not a simulation" discipline test_merchant_agent
and test_redteam already hold to.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from buyer.catalog import load_verified_catalog
from buyer.shop import MerchantEndpoint, shop_around
from mandates.keys import KeyDirectory, generate_keypair
from mandates.schemas import (
    Budget,
    HardConstraints,
    IntentMandate,
    MerchantPolicy,
    Provenance,
    SoftPreferences,
)
from mandates.sign import sign_mandate
from merchant_agent.driver import MerchantAgentDriver
from merchant_kit.schemas import Catalog, CatalogItem

UTC = timezone.utc
HUMAN_KP = generate_keypair()
SHOP_A_KP = generate_keypair()
SHOP_B_KP = generate_keypair()
SHOP_C_KP = generate_keypair()


def _now() -> datetime:
    return datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC)


def _prov(now):
    return Provenance(source_url="https://x", extraction_method="schema_org", signature_verified=False, extracted_at=now)


def _item(sku, title, category, price_paise, **attrs):
    now = _now()
    return CatalogItem(
        sku=sku, title=title, description="", category=category, price_paise=price_paise, in_stock=True,
        attributes=attrs,
        field_provenance={f: _prov(now) for f in ("title", "description", "category", "price_paise", "in_stock")},
    )


def _endpoint(merchant_id: str, keypair, items: list[CatalogItem]) -> MerchantEndpoint:
    driver = MerchantAgentDriver(catalog=items, merchant_id=merchant_id, merchant_keypair=keypair, now=_now())
    doc = Catalog(merchant_id=merchant_id, generated_at=_now().isoformat(), items=items)
    envelope = sign_mandate(doc, keypair)
    directory = KeyDirectory()
    directory.register_keypair(keypair)
    directory.register_keypair(HUMAN_KP)  # G0.2 verifies BOTH envelopes against this directory
    catalog = load_verified_catalog(envelope, directory, expected_merchant_id=merchant_id, required_kid=keypair.kid)
    return MerchantEndpoint(
        merchant_id=merchant_id, driver=driver,
        merchant_policy=MerchantPolicy(
            merchant_id=merchant_id, max_order_value_paise=1_000_000, substitution_tolerance_pct=10.0,
            max_quantity_per_sku=5, restricted_categories=[], escalation_threshold_paise=1_000_000,
        ),
        catalog=catalog, key_directory=directory,
    )


def _intent(allowed_merchants=None, target_price_paise=None, preferred_brands=None) -> IntentMandate:
    return IntentMandate(
        mandate_id="intent-1", issued_at=_now(), expires_at=_now() + timedelta(hours=1),
        principal_kid=HUMAN_KP.kid, agent_kid=HUMAN_KP.kid, request_text="buy rice",
        hard=HardConstraints(category="grains"),
        soft=SoftPreferences(
            substitution_tolerance_pct=10.0, target_price_paise=target_price_paise,
            preferred_brands=preferred_brands or [],
        ),
        budget=Budget(total_paise=50_000, per_item_paise=50_000, max_quantity=2, escalate_above_paise=50_000),
        allowed_merchants=allowed_merchants,
    )


def test_a_declining_merchant_is_a_non_event_not_a_dead_end():
    """The core fix: single-merchant negotiate() used to make the WHOLE
    purchase fail if that one shop declined, even when another onboarded
    merchant had the item."""
    intent = _intent(allowed_merchants=["shop-a", "shop-b"])
    intent_envelope = sign_mandate(intent, HUMAN_KP)
    endpoints = {
        "shop-a": _endpoint("shop-a", SHOP_A_KP, [_item("X-1", "Something Else", "electronics", 10_000)]),  # no match
        "shop-b": _endpoint("shop-b", SHOP_B_KP, [_item("RICE-1", "Basmati Rice", "grains", 30_000)]),
    }

    result = shop_around(intent, intent_envelope, endpoints, now=_now())

    assert result.approved
    assert result.winner.merchant_id == "shop-b"
    declined_row = next(r for r in result.rows if r.merchant_id == "shop-a")
    assert declined_row.negotiation_outcome == "declined"


def test_picks_the_cheaper_of_two_matching_offers_with_no_preferences():
    intent = _intent(allowed_merchants=["shop-a", "shop-b"])
    intent_envelope = sign_mandate(intent, HUMAN_KP)
    endpoints = {
        "shop-a": _endpoint("shop-a", SHOP_A_KP, [_item("RICE-A", "Basmati Rice", "grains", 35_000)]),
        "shop-b": _endpoint("shop-b", SHOP_B_KP, [_item("RICE-B", "Basmati Rice", "grains", 28_000)]),
    }

    result = shop_around(intent, intent_envelope, endpoints, now=_now())

    assert result.winner.merchant_id == "shop-b"
    assert result.winner.cart.total_paise == 28_000


def test_preferred_brand_can_beat_a_cheaper_offer():
    """The point of wiring buyer/rank.py in here: a signed brand
    preference must actually be able to change which SHOP wins, not just
    which item within one shop's catalog."""
    intent = _intent(allowed_merchants=["shop-a", "shop-b"], preferred_brands=["IndiaGate"])
    intent_envelope = sign_mandate(intent, HUMAN_KP)
    endpoints = {
        "shop-a": _endpoint("shop-a", SHOP_A_KP, [_item("RICE-A", "Basmati Rice", "grains", 25_000, brand="Generic")]),
        "shop-b": _endpoint("shop-b", SHOP_B_KP, [_item("RICE-B", "Basmati Rice", "grains", 32_000, brand="IndiaGate")]),
    }

    result = shop_around(intent, intent_envelope, endpoints, now=_now())

    assert result.winner.merchant_id == "shop-b"  # pricier, but matches the preferred brand


def test_a_merchant_not_in_allowed_merchants_is_never_tried():
    """shop_around must not widen what the signed intent already
    authorizes -- it only chooses among what it permits."""
    intent = _intent(allowed_merchants=["shop-a"])
    intent_envelope = sign_mandate(intent, HUMAN_KP)
    endpoints = {
        "shop-a": _endpoint("shop-a", SHOP_A_KP, [_item("RICE-A", "Basmati Rice", "grains", 30_000)]),
        "shop-b": _endpoint("shop-b", SHOP_B_KP, [_item("RICE-B", "Basmati Rice", "grains", 10_000)]),  # cheaper but not authorized
    }

    result = shop_around(intent, intent_envelope, endpoints, now=_now())

    assert result.winner.merchant_id == "shop-a"
    assert {r.merchant_id for r in result.rows} == {"shop-a"}


def test_no_merchant_has_a_match_returns_unapproved_not_an_exception():
    intent = _intent(allowed_merchants=["shop-a", "shop-b"])
    intent_envelope = sign_mandate(intent, HUMAN_KP)
    endpoints = {
        "shop-a": _endpoint("shop-a", SHOP_A_KP, [_item("X", "Nope", "electronics", 10_000)]),
        "shop-b": _endpoint("shop-b", SHOP_B_KP, [_item("Y", "Also Nope", "apparel", 10_000)]),
    }

    result = shop_around(intent, intent_envelope, endpoints, now=_now())

    assert not result.approved
    assert result.winner is None
    assert result.winning_cart_envelope is None
    assert len(result.rows) == 2


def test_a_dishonest_merchant_never_wins_even_if_cheapest():
    """The comparison must run every candidate through the SAME
    evaluate() -- a merchant offering the "best" price by lying about its
    own published catalog (G8.1) must not out-rank an honest one."""
    from redteam.driver_attacks import apply_catalog_overcharge

    intent = _intent(allowed_merchants=["shop-a", "shop-b"])
    intent_envelope = sign_mandate(intent, HUMAN_KP)
    honest_items = [_item("RICE-B", "Basmati Rice", "grains", 32_000)]
    dishonest_items = [_item("RICE-A", "Basmati Rice", "grains", 20_000)]  # published cheap...

    dishonest_endpoint = _endpoint("shop-a", SHOP_A_KP, dishonest_items)
    cheating_driver = apply_catalog_overcharge(
        dishonest_endpoint.driver, merchant_keypair=SHOP_A_KP, budget_total_paise=intent.budget.total_paise,
    )
    endpoints = {
        "shop-a": MerchantEndpoint(
            merchant_id="shop-a", driver=cheating_driver, merchant_policy=dishonest_endpoint.merchant_policy,
            catalog=dishonest_endpoint.catalog, key_directory=dishonest_endpoint.key_directory,
        ),  # ...but signs a cart charging near the full budget for it
        "shop-b": _endpoint("shop-b", SHOP_B_KP, honest_items),
    }

    result = shop_around(intent, intent_envelope, endpoints, now=_now())

    assert result.winner.merchant_id == "shop-b"
    cheat_row = next(r for r in result.rows if r.merchant_id == "shop-a")
    assert cheat_row.decision.outcome == "REJECTED"
    assert "CATALOG_PRICE_MISMATCH" in {c.code for c in cheat_row.decision.failed_checks}


def test_narrative_reads_as_an_answer_to_why_did_you_buy_this():
    intent = _intent(allowed_merchants=["shop-a", "shop-b"])
    intent_envelope = sign_mandate(intent, HUMAN_KP)
    endpoints = {
        "shop-a": _endpoint("shop-a", SHOP_A_KP, [_item("X", "Nope", "electronics", 10_000)]),
        "shop-b": _endpoint("shop-b", SHOP_B_KP, [_item("RICE-B", "Basmati Rice", "grains", 28_000)]),
    }

    result = shop_around(intent, intent_envelope, endpoints, now=_now())
    text = result.narrative()

    assert "Checked 2 shop(s)" in text
    assert "shop-b" in text and "WON" in text
    assert "shop-a" in text
