"""
Tests for buyer/negotiate.py, buyer/approval.py, buyer/execute.py, and
buyer/interpret.py -- the modules that wire mandates + policy_engine +
razorpay_client + identity into the actual buyer agent loop.
"""

from __future__ import annotations

import json

import os
from datetime import datetime, timedelta, timezone

import pytest

from buyer.approval import ScriptedApprover, build_amended_intent, resolve_escalation
from buyer.catalog import CatalogEntry, VerifiedCatalog
from buyer.execute import ExecutionRefused, execute
from buyer.negotiate import MerchantResponse, negotiate
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
from razorpay_client.cassettes import ReplayTransport
from razorpay_client.client import RazorpayClient, client_from_env

UTC = timezone.utc
HUMAN_KP = generate_keypair()
MERCHANT_KP = generate_keypair()


def _now() -> datetime:
    return datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)


def _clean_provenance(now: datetime) -> Provenance:
    return Provenance(source_url="https://merchant.example/p", extraction_method="schema_org", signature_verified=True, extracted_at=now)


def _intent(**overrides) -> IntentMandate:
    defaults = dict(
        mandate_id="intent-1", issued_at=_now(), expires_at=_now() + timedelta(hours=1),
        principal_kid=HUMAN_KP.kid, agent_kid="agent-1", request_text="buy rice",
        hard=HardConstraints(category="grocery"),
        soft=SoftPreferences(substitution_tolerance_pct=10.0),
        budget=Budget(total_paise=50_000, per_item_paise=50_000, max_quantity=2, escalate_above_paise=45_000),
        allowed_merchants=["merchant-1"],
    )
    defaults.update(overrides)
    return IntentMandate(**defaults)


def _cart(intent: IntentMandate, *, total_paise: int = 30_000, is_substitute: bool = False) -> CartMandate:
    line = CartLine(
        sku="SKU-1", title="Rice", category="grocery", unit_price_paise=total_paise, quantity=1,
        is_substitute=is_substitute,
        field_provenance={"unit_price_paise": _clean_provenance(_now()), "title": _clean_provenance(_now())},
    )
    return CartMandate(
        mandate_id="cart-1", issued_at=_now(), expires_at=_now() + timedelta(hours=1),
        merchant_id="merchant-1", merchant_kid=MERCHANT_KP.kid, intent_hash=mandate_hash(intent),
        lines=[line], subtotal_paise=total_paise, tax_paise=0, shipping_paise=0, total_paise=total_paise,
    )


def _merchant_policy() -> MerchantPolicy:
    return MerchantPolicy(
        merchant_id="merchant-1", max_order_value_paise=1_000_000, substitution_tolerance_pct=10.0,
        max_quantity_per_sku=5, restricted_categories=[], escalation_threshold_paise=999_999,
    )


def _catalog(*, price_paise: int = 100_000, category: str = "grocery") -> VerifiedCatalog:
    """The merchant publishes SKU-1 generously, so the G8.x anomaly band
    is satisfied and these tests stay isolated to the loop behaviour they
    were written for. G8.1 is one-directional -- charging BELOW the
    published price is legitimate -- so a high published price is the
    neutral default here, not a loophole."""
    return VerifiedCatalog(
        merchant_id="merchant-1",
        merchant_kid=MERCHANT_KP.kid,
        generated_at=_now().isoformat(),
        entries={
            "SKU-1": CatalogEntry(
                sku="SKU-1", title="Rice", category=category,
                price_paise=price_paise, in_stock=True,
            )
        },
    )


def _ctx() -> VerificationContext:
    directory = KeyDirectory()
    directory.register_keypair(HUMAN_KP)
    directory.register_keypair(MERCHANT_KP)
    intent = _intent()
    cart = _cart(intent)
    return VerificationContext(
        key_directory=directory, intent_envelope=sign_mandate(intent, HUMAN_KP),
        cart_envelope=sign_mandate(cart, MERCHANT_KP), merchant_catalog=_catalog(),
    )


# ---------------------------------------------------------------------
# negotiate.py
# ---------------------------------------------------------------------


def test_negotiate_cart_on_first_turn():
    intent = _intent()
    cart = _cart(intent)
    envelope = sign_mandate(cart, MERCHANT_KP)

    def driver(turn, intent):
        return MerchantResponse(kind="cart", cart_envelope=envelope)

    result = negotiate(intent, merchant_driver=driver)
    assert result.outcome == "cart_received"
    assert result.turns_used == 1
    assert result.cart_envelope is envelope


def test_negotiate_decline():
    def driver(turn, intent):
        return MerchantResponse(kind="decline", decline_reason="out of stock")

    result = negotiate(_intent(), merchant_driver=driver)
    assert result.outcome == "declined"
    assert result.reason == "out of stock"
    assert result.turns_used == 1


def test_negotiate_timeout_after_max_turns():
    """`decline` is a terminal outcome by design (see module docstring:
    the only two things a merchant can return are "cart" or "decline"),
    so the clean, deterministic way to exercise the timeout path is
    max_turns=0 -- the driver is never even called, and the loop must
    still resolve to a well-formed timeout result rather than erroring."""
    calls = []

    def driver(turn, intent):
        calls.append(turn)
        return MerchantResponse(kind="decline", decline_reason="x")

    result = negotiate(_intent(), merchant_driver=driver, max_turns=0)
    assert result.outcome == "timeout"
    assert result.turns_used == 0
    assert "NEGOTIATION_TIMEOUT" in result.reason
    assert calls == []  # driver never called when max_turns=0


def test_negotiate_calls_driver_at_most_max_turns_times():
    calls = []

    def driver(turn, intent):
        calls.append(turn)
        return MerchantResponse(kind="decline", decline_reason="x")  # decline resolves on turn 1

    negotiate(_intent(), merchant_driver=driver)
    assert calls == [1]  # decline is terminal -- confirms the loop doesn't over-call


def test_negotiate_unknown_response_kind_raises():
    def driver(turn, intent):
        return MerchantResponse(kind="bogus")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unknown MerchantResponse"):
        negotiate(_intent(), merchant_driver=driver)


def test_negotiate_cart_response_without_envelope_raises():
    def driver(turn, intent):
        return MerchantResponse(kind="cart", cart_envelope=None)

    with pytest.raises(ValueError, match="cart_envelope is None"):
        negotiate(_intent(), merchant_driver=driver)


# ---------------------------------------------------------------------
# approval.py
# ---------------------------------------------------------------------


def test_build_amended_intent_widens_escalation_threshold_only():
    intent = _intent(budget=Budget(total_paise=50_000, per_item_paise=50_000, max_quantity=2, escalate_above_paise=20_000))
    cart = _cart(intent, total_paise=30_000)  # over escalate_above_paise, under total_paise -> pure ESCALATE
    ctx = _ctx()
    decision = evaluate(intent, cart, _merchant_policy(), ctx, _now())
    assert decision.outcome == "REQUIRES_HUMAN_APPROVAL"

    amended = build_amended_intent(intent, decision, cart, now=_now())
    assert amended.budget.escalate_above_paise >= cart.total_paise
    assert amended.budget.total_paise > amended.budget.escalate_above_paise
    assert amended.hard == intent.hard  # untouched
    assert amended.soft.substitutions_preapproved is False  # untouched -- only the relevant field changed


def test_build_amended_intent_sets_preapproved_for_substitution():
    intent = _intent()
    cart = _cart(intent, total_paise=30_000, is_substitute=True)
    ctx = _ctx()
    decision = evaluate(intent, cart, _merchant_policy(), ctx, _now())
    assert decision.outcome == "REQUIRES_HUMAN_APPROVAL"

    amended = build_amended_intent(intent, decision, cart, now=_now())
    assert amended.soft.substitutions_preapproved is True
    assert amended.budget == intent.budget  # untouched -- only the relevant field changed


def _matching_merchant_driver(total_paise: int, *, is_substitute: bool = False) -> callable:
    """A stub standing in for merchant_agent/: builds a cart correctly
    bound (via intent_hash) to WHATEVER intent it's handed -- including
    an amended one -- exactly like a real merchant re-confirming against
    updated terms would."""

    def driver(turn, intent):
        line = CartLine(
            sku="SKU-1", title="Rice", category="grocery", unit_price_paise=total_paise, quantity=1,
            is_substitute=is_substitute,
            field_provenance={"unit_price_paise": _clean_provenance(_now()), "title": _clean_provenance(_now())},
        )
        cart = CartMandate(
            mandate_id="cart-renegotiated", issued_at=_now(), expires_at=_now() + timedelta(hours=1),
            merchant_id="merchant-1", merchant_kid=MERCHANT_KP.kid, intent_hash=mandate_hash(intent),
            lines=[line], subtotal_paise=total_paise, tax_paise=0, shipping_paise=0, total_paise=total_paise,
        )
        return MerchantResponse(kind="cart", cart_envelope=sign_mandate(cart, MERCHANT_KP))

    return driver


def test_resolve_escalation_declined_does_not_reevaluate():
    intent = _intent(budget=Budget(total_paise=50_000, per_item_paise=50_000, max_quantity=2, escalate_above_paise=20_000))
    cart = _cart(intent, total_paise=30_000)
    ctx = _ctx()
    decision = evaluate(intent, cart, _merchant_policy(), ctx, _now())

    def unreachable_driver(turn, intent):
        raise AssertionError("must never re-negotiate when the human declined")

    result = resolve_escalation(
        decision, intent, cart, _merchant_policy(), ctx,
        approver=ScriptedApprover(on_escalation="decline"), human_keypair=HUMAN_KP, now=_now(),
        merchant_driver=unreachable_driver,
    )
    assert result.approved is False
    assert result.final_decision is None
    assert result.amended_intent is None


def test_resolve_escalation_approved_reevaluates_from_scratch_and_passes():
    intent = _intent(budget=Budget(total_paise=50_000, per_item_paise=50_000, max_quantity=2, escalate_above_paise=20_000))
    cart = _cart(intent, total_paise=30_000)
    ctx = _ctx()
    decision = evaluate(intent, cart, _merchant_policy(), ctx, _now())
    assert decision.outcome == "REQUIRES_HUMAN_APPROVAL"

    result = resolve_escalation(
        decision, intent, cart, _merchant_policy(), ctx,
        approver=ScriptedApprover(on_escalation="approve"), human_keypair=HUMAN_KP, now=_now(),
        merchant_driver=_matching_merchant_driver(30_000),
    )
    assert result.approved is True
    assert result.final_decision is not None
    assert result.final_decision.outcome == "APPROVED"
    # the amended envelope must actually verify -- it's a real signature,
    # not a rubber stamp
    assert result.amended_envelope is not None


def test_resolve_escalation_renegotiation_declined_does_not_claim_approved():
    intent = _intent(budget=Budget(total_paise=50_000, per_item_paise=50_000, max_quantity=2, escalate_above_paise=20_000))
    cart = _cart(intent, total_paise=30_000)
    ctx = _ctx()
    decision = evaluate(intent, cart, _merchant_policy(), ctx, _now())

    def declining_driver(turn, intent):
        return MerchantResponse(kind="decline", decline_reason="price changed")

    result = resolve_escalation(
        decision, intent, cart, _merchant_policy(), ctx,
        approver=ScriptedApprover(on_escalation="approve"), human_keypair=HUMAN_KP, now=_now(),
        merchant_driver=declining_driver,
    )
    assert result.approved is True  # the human DID approve
    assert result.final_decision is None  # but there is no final APPROVED outcome to act on
    assert "did not produce a cart" in result.audit_note


def test_resolve_escalation_does_not_bypass_an_unrelated_reject_gate():
    """The core non-negotiable property: approving an escalation must NOT
    make an unrelated REJECT-severity problem disappear. Here the cart
    itself has bad arithmetic (a completely different gate, G2.1) -- no
    amount of human approval for the escalation should paper over that."""
    intent = _intent(budget=Budget(total_paise=50_000, per_item_paise=50_000, max_quantity=2, escalate_above_paise=20_000))
    line = CartLine(
        sku="SKU-1", title="Rice", category="grocery", unit_price_paise=30_000, quantity=1, is_substitute=True,
        field_provenance={"unit_price_paise": _clean_provenance(_now()), "title": _clean_provenance(_now())},
    )
    bad_cart = CartMandate(
        mandate_id="cart-bad", issued_at=_now(), expires_at=_now() + timedelta(hours=1),
        merchant_id="merchant-1", merchant_kid=MERCHANT_KP.kid, intent_hash=mandate_hash(intent),
        lines=[line], subtotal_paise=30_000, tax_paise=0, shipping_paise=0, total_paise=999,  # arithmetic broken
    )
    ctx = _ctx()
    decision = evaluate(intent, bad_cart, _merchant_policy(), ctx, _now())
    assert decision.outcome == "REJECTED"  # not escalated at all -- REJECT wins

    def unreachable_driver(turn, intent):
        raise AssertionError("must never be called -- resolve_escalation should raise before any negotiation")

    with pytest.raises(ValueError, match="not REQUIRES_HUMAN_APPROVAL"):
        resolve_escalation(
            decision, intent, bad_cart, _merchant_policy(), ctx,
            approver=ScriptedApprover(on_escalation="approve"), human_keypair=HUMAN_KP, now=_now(),
            merchant_driver=unreachable_driver,
        )


# ---------------------------------------------------------------------
# execute.py
# ---------------------------------------------------------------------


def test_execute_refuses_non_approved_decision():
    intent = _intent()
    cart = _cart(intent)
    fake_decision = Decision(outcome="REJECTED", checks=(), blocking_code="BUDGET_EXCEEDED", evaluated_at=_now())
    with pytest.raises(ExecutionRefused):
        execute(fake_decision, intent, cart, client=RazorpayClient(key_id="x", key_secret="y"), now=_now())


def test_execute_against_empty_cassette_raises_cassette_miss(tmp_path):
    """execute() generates a fresh uuid receipt on every call, so a
    pre-recorded cassette can never contain a matching entry -- this
    confirms that (a real replay test for execute() lives in the live
    test below, since the request is receipt-random by design) and, as a
    side effect, that execute() reaches the network call at all rather
    than silently short-circuiting somewhere before it."""
    from razorpay_client.cassettes import CassetteMiss

    intent = _intent()
    cart = _cart(intent)
    cassette = tmp_path / "empty.json"
    cassette.write_text("{}")
    client = RazorpayClient(key_id="rzp_test_x", key_secret="secret", transport=ReplayTransport(cassette))

    fake_decision = Decision(outcome="APPROVED", checks=(), blocking_code=None, evaluated_at=_now())
    with pytest.raises(CassetteMiss):
        execute(fake_decision, intent, cart, client=client, now=_now())


requires_live_razorpay = pytest.mark.skipif(not os.environ.get("RAZORPAY_KEY_ID"), reason="no RAZORPAY_KEY_ID in environment")


@requires_live_razorpay
def test_execute_live_binds_mandate_hashes_into_real_order():
    intent = _intent()
    cart = _cart(intent)
    client = client_from_env()
    fake_decision = Decision(outcome="APPROVED", checks=(), blocking_code=None, evaluated_at=_now())

    payment_mandate, order = execute(fake_decision, intent, cart, client=client, now=datetime.now(UTC))
    assert order["status"] == "created"
    assert payment_mandate.razorpay_order_id == order["id"]

    fetched = client.fetch_order(order["id"])
    assert fetched["notes"]["intent_hash"] == mandate_hash(intent)
    assert fetched["notes"]["cart_hash"] == mandate_hash(cart)
    assert fetched["notes"]["payment_mandate_id"] == payment_mandate.mandate_id


# ---------------------------------------------------------------------
# interpret.py -- live, skipped if no key
# ---------------------------------------------------------------------

requires_gemini = pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="no GEMINI_API_KEY in environment")


@requires_gemini
def test_interpret_live_clean_request_produces_valid_draft():
    from llm.provider import GeminiProvider

    from buyer.interpret import interpret

    provider = GeminiProvider()
    try:
        result = interpret(
            "Order 2kg basmati rice, keep it under 500 rupees",
            provider=provider, model=os.environ.get("LLM_MODEL_EVAL", "gemini-3.1-flash-lite"),
            principal_kid="kid-human", agent_kid="kid-agent", allowed_merchants=["merchant-1"], now=_now(),
        )
    finally:
        provider.close()

    assert result.budget_stated is True
    assert result.draft_intent.budget.total_paise > 0
    # No derived tripwire: a stated budget produces exactly ONE ceiling.
    # It used to produce two (a hard cap and an 80% escalate), so a cart
    # in the top 20% of a budget the human themselves set stopped the
    # loop for no added safety.
    assert result.draft_intent.budget.escalate_above_paise == result.draft_intent.budget.total_paise
    assert result.draft_intent.request_text.startswith("Order 2kg")


@requires_gemini
def test_interpret_live_signed_draft_passes_signature_gate():
    """The structural guarantee: interpret()'s output has NO signature at
    all until a human signs it. Confirms the unsigned draft is a plain
    IntentMandate that only becomes usable via mandates.sign_mandate."""
    from llm.provider import GeminiProvider

    from buyer.interpret import interpret

    provider = GeminiProvider()
    try:
        result = interpret(
            "Buy a plain white t-shirt, budget 800 rupees",
            provider=provider, model=os.environ.get("LLM_MODEL_EVAL", "gemini-3.1-flash-lite"),
            principal_kid=HUMAN_KP.kid, agent_kid="kid-agent", allowed_merchants=["merchant-1"], now=_now(),
        )
    finally:
        provider.close()

    envelope = sign_mandate(result.draft_intent, HUMAN_KP)
    from mandates import KeyDirectory as _KD
    from mandates import verify_envelope

    directory = _KD()
    directory.register_keypair(HUMAN_KP)
    verify_result = verify_envelope(envelope, directory, required_kid=HUMAN_KP.kid)
    assert verify_result.valid, verify_result.reason


# ---------------------------------------------------------------------
# interpret.py -- the unstated-budget path, offline (stubbed provider)
# ---------------------------------------------------------------------


class _StubCompletion:
    def __init__(self, text):
        self.text = text


class _StubProvider:
    """Returns one canned JSON body. Lets the budget-handling branches be
    tested exhaustively without spending a live call on each."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        return _StubCompletion(json.dumps(self.payload))


def _interpret_with(payload, request_text="buy me a plain white t-shirt"):
    from buyer.interpret import interpret

    return interpret(
        request_text, provider=_StubProvider(payload), model="stub",
        principal_kid="kid-human", agent_kid="kid-agent",
        allowed_merchants=["merchant-1"], now=_now(),
    )


def test_unstated_budget_asks_instead_of_inventing_one():
    """The worst thing the system could do: hallucinate a spending
    ceiling and have a human cryptographically sign it. The letter of
    "the LLM proposes, it never authorizes" survived that -- a human
    still signed -- but they were signing a number the model made up and
    presented as if it came from their own request."""
    result = _interpret_with({"needs_clarification": False, "category": "apparel"})

    assert result.budget_stated is False
    assert result.needs_clarification is True
    assert result.ambiguous_field == "budget_total_rupees"
    assert result.clarification_question
    # ...and crucially there is nothing signable to rubber-stamp.
    assert result.draft_intent is None
    assert result.is_signable is False


def test_unstated_budget_does_not_raise():
    """The other half of the old behaviour: when the model declined to
    invent a number, the whole run died on an unhandled ValueError."""
    result = _interpret_with({"needs_clarification": False, "budget_total_rupees": None})
    assert result.draft_intent is None


def test_zero_or_negative_budget_is_treated_as_unstated():
    for value in (0, -100):
        result = _interpret_with({"needs_clarification": False, "budget_total_rupees": value})
        assert result.draft_intent is None, value
        assert result.budget_stated is False


def test_stated_budget_produces_a_signable_draft_with_no_tripwire():
    result = _interpret_with(
        {"needs_clarification": False, "budget_total_rupees": 500, "category": "apparel"},
        request_text="buy me a white t-shirt under 500 rupees",
    )
    assert result.is_signable
    budget = result.draft_intent.budget
    assert budget.total_paise == 50_000
    assert budget.escalate_above_paise == budget.total_paise  # one ceiling, not two


def test_enrich_request_text_reruns_on_a_whole_request():
    """The clarifying answer folds back into the request text and the
    interpreter runs again on the whole thing -- an answer can carry more
    than the field that was asked about ("under Rs2000, and it must be
    cotton"), and splicing it into the draft would silently drop that."""
    from buyer.interpret import enrich_request_text

    enriched = enrich_request_text(
        "buy me a plain white t-shirt", "What's the most you want to spend?", "under 2000, cotton only"
    )
    assert "plain white t-shirt" in enriched
    assert "under 2000, cotton only" in enriched

    result = _interpret_with(
        {"needs_clarification": False, "budget_total_rupees": 2000,
         "category": "apparel", "required_attributes": [{"key": "material", "value": "cotton"}]},
        request_text=enriched,
    )
    assert result.is_signable
    assert result.draft_intent.budget.total_paise == 200_000
    assert result.draft_intent.hard.required_attributes == {"material": "cotton"}
    # request_text is the whole request, which is what an auditor reading
    # the signed mandate later needs it to be.
    assert "plain white t-shirt" in result.draft_intent.request_text


def test_missing_total_falls_back_to_per_item_at_single_quantity():
    """Asked for "a plain cotton shirt, up to 2000 rupees," a live model
    can put 2000 in budget_per_item_rupees and leave budget_total_rupees
    unset (needs_clarification=False on its own; it isn't confused, it
    just named the field differently for an amount that means the same
    thing at quantity 1). A per-item figure at quantity 1 IS the total,
    so this is a safe fallback, not a guess:
    it never reads a bigger budget than was ever stated."""
    result = _interpret_with(
        {"needs_clarification": False, "budget_per_item_rupees": 2000, "category": "apparel"},
    )
    assert result.is_signable
    assert result.budget_stated is True
    assert result.draft_intent.budget.total_paise == 200_000
    assert result.draft_intent.budget.per_item_paise == 200_000


def test_missing_total_does_not_fall_back_at_quantity_above_one():
    """The fallback is only safe at quantity<=1 -- at any higher
    quantity a per-item figure is NOT the total, and treating it as one
    would silently authorize spending more than was ever stated."""
    result = _interpret_with(
        {"needs_clarification": False, "budget_per_item_rupees": 2000, "max_quantity": 3, "category": "apparel"},
    )
    assert result.draft_intent is None
    assert result.budget_stated is False
    assert result.ambiguous_field == "budget_total_rupees"


def test_available_categories_constrains_the_schema_as_an_enum():
    """When a merchant's real category taxonomy is given, interpret()
    must actually constrain the model's output to it, not just mention
    it in prose the model can ignore."""
    captured = {}

    class _CapturingProvider:
        def complete(self, **kwargs):
            captured.update(kwargs)
            return _StubCompletion(json.dumps(
                {"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"}
            ))

    from buyer.interpret import interpret

    interpret(
        "buy me some basmati rice, under 500 rupees", provider=_CapturingProvider(), model="stub",
        principal_kid="k", agent_kid="k", allowed_merchants=["m"], now=_now(),
        available_categories=["grains", "condiments", "snacks"],
    )

    schema = captured["response_schema"]
    assert schema["properties"]["category"]["enum"] == ["grains", "condiments", "snacks"]
    assert "grains, condiments, snacks" in captured["system"]


def test_omitting_available_categories_preserves_free_text_category():
    """No caller that doesn't opt in is affected -- the original
    free-text behaviour must be exactly unchanged."""
    captured = {}

    class _CapturingProvider:
        def complete(self, **kwargs):
            captured.update(kwargs)
            return _StubCompletion(json.dumps(
                {"needs_clarification": False, "budget_total_rupees": 500, "category": "anything the model wants"}
            ))

    from buyer.interpret import interpret

    interpret(
        "buy me a plain white t-shirt under 500 rupees", provider=_CapturingProvider(), model="stub",
        principal_kid="k", agent_kid="k", allowed_merchants=["m"], now=_now(),
    )

    schema = captured["response_schema"]
    assert "enum" not in schema["properties"]["category"]
    assert schema["properties"]["category"] == {"type": "STRING"}


def test_available_categories_flows_through_to_the_signed_draft():
    from buyer.interpret import interpret

    result = interpret(
        "buy me some basmati rice, under 500 rupees",
        provider=_StubProvider({"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"}),
        model="stub", principal_kid="k", agent_kid="k", allowed_merchants=["m"], now=_now(),
        available_categories=["grains", "condiments"],
    )
    assert result.is_signable
    assert result.draft_intent.hard.category == "grains"


def test_available_attribute_keys_constrains_the_schema_as_an_enum():
    """Giving interpret() a catalog's real attribute-key vocabulary must
    actually constrain the `key` field of each required_attributes
    entry, not just mention it in prose."""
    captured = {}

    class _CapturingProvider:
        def complete(self, **kwargs):
            captured.update(kwargs)
            return _StubCompletion(json.dumps(
                {"needs_clarification": False, "budget_total_rupees": 500, "category": "grains",
                 "required_attributes": [{"key": "variant", "value": "basmati"}]}
            ))

    from buyer.interpret import interpret

    interpret(
        "buy me some basmati rice, under 500 rupees", provider=_CapturingProvider(), model="stub",
        principal_kid="k", agent_kid="k", allowed_merchants=["m"], now=_now(),
        available_attribute_keys={"grains": ["variant", "weight_g"]},
    )

    schema = captured["response_schema"]
    assert schema["properties"]["required_attributes"]["items"]["properties"]["key"]["enum"] == ["variant", "weight_g"]
    assert "grains -> variant, weight_g" in captured["system"]


def test_available_attribute_keys_is_scoped_per_category_in_the_prompt():
    """A flat union across categories would let the model attach one
    category's attribute key to a different category's item
    (a schema-level enum can't be conditioned on the `category` field's
    value) -- the per-category breakdown in the prompt is what actually
    closes that gap, so every category's keys must be named separately,
    not merged into one list."""
    captured = {}

    class _CapturingProvider:
        def complete(self, **kwargs):
            captured.update(kwargs)
            return _StubCompletion(json.dumps(
                {"needs_clarification": False, "budget_total_rupees": 500, "category": "grains"}
            ))

    from buyer.interpret import interpret

    interpret(
        "buy me some basmati rice, under 500 rupees", provider=_CapturingProvider(), model="stub",
        principal_kid="k", agent_kid="k", allowed_merchants=["m"], now=_now(),
        available_attribute_keys={"grains": ["variant"], "apparel": ["colour", "size"]},
    )

    schema = captured["response_schema"]
    # The schema-level enum is necessarily the union (JSON Schema can't
    # condition one field's enum on another field's value) -- the
    # per-category scoping lives in the prompt text instead.
    assert schema["properties"]["required_attributes"]["items"]["properties"]["key"]["enum"] == ["colour", "size", "variant"]
    assert "grains -> variant" in captured["system"]
    assert "apparel -> colour, size" in captured["system"]
    assert "never a key that belongs to a different category" in captured["system"]


def test_omitting_available_attribute_keys_preserves_free_text_key():
    """No caller that doesn't opt in is affected -- the original
    free-text behaviour must be exactly unchanged."""
    captured = {}

    class _CapturingProvider:
        def complete(self, **kwargs):
            captured.update(kwargs)
            return _StubCompletion(json.dumps(
                {"needs_clarification": False, "budget_total_rupees": 500, "category": "apparel",
                 "required_attributes": [{"key": "colour", "value": "white"}]}
            ))

    from buyer.interpret import interpret

    interpret(
        "buy me a plain white t-shirt under 500 rupees", provider=_CapturingProvider(), model="stub",
        principal_kid="k", agent_kid="k", allowed_merchants=["m"], now=_now(),
    )

    schema = captured["response_schema"]
    assert "enum" not in schema["properties"]["required_attributes"]["items"]["properties"]["key"]


def test_empty_available_attribute_keys_suppresses_enum_but_adds_a_note():
    """`[]` (a catalog that genuinely tracks zero attribute keys) is a
    distinct, meaningful signal from `None` (no info given) -- an empty
    JSON Schema enum is invalid, so this must fall back to a strong
    prompt instruction instead, not a broken schema."""
    captured = {}

    class _CapturingProvider:
        def complete(self, **kwargs):
            captured.update(kwargs)
            return _StubCompletion(json.dumps(
                {"needs_clarification": False, "budget_total_rupees": 500, "category": "apparel"}
            ))

    from buyer.interpret import interpret

    interpret(
        "buy me a plain white t-shirt under 500 rupees", provider=_CapturingProvider(), model="stub",
        principal_kid="k", agent_kid="k", allowed_merchants=["m"], now=_now(),
        available_attribute_keys={"apparel": []},
    )

    schema = captured["response_schema"]
    assert "enum" not in schema["properties"]["required_attributes"]["items"]["properties"]["key"]
    assert "does not track any structured product" in captured["system"]


def test_available_attribute_keys_flows_through_to_the_signed_draft():
    from buyer.interpret import interpret

    result = interpret(
        "buy me some basmati rice, under 500 rupees",
        provider=_StubProvider({"needs_clarification": False, "budget_total_rupees": 500, "category": "grains",
                                 "required_attributes": [{"key": "variant", "value": "basmati"}]}),
        model="stub", principal_kid="k", agent_kid="k", allowed_merchants=["m"], now=_now(),
        available_attribute_keys={"grains": ["variant"]},
    )
    assert result.is_signable
    assert result.draft_intent.hard.required_attributes == {"variant": "basmati"}
