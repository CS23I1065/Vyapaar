"""
Pure, no-I/O unit tests for mandates/. This is the module the plan singles
out for "unit tests first" -- it's the foundation the policy engine (step
4) and everything downstream trusts, so tamper-detection and chain-
breakage need to be exhaustively covered here, not discovered later via a
flaky end-to-end demo run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from mandates.canonical import canonical_bytes
from mandates.chain import is_expired, verify_chain
from mandates.hashing import mandate_hash
from mandates.keys import KeyDirectory, generate_keypair
from mandates.schemas import (
    Budget,
    CartLine,
    CartMandate,
    HardConstraints,
    IntentMandate,
    PaymentMandate,
    Provenance,
    SoftPreferences,
)
from mandates.sign import sign_mandate
from mandates.verify import verify_envelope

UTC = timezone.utc


def _now() -> datetime:
    return datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)


def _budget(**overrides) -> Budget:
    defaults = dict(total_paise=50_000, per_item_paise=50_000, max_quantity=2, escalate_above_paise=40_000)
    defaults.update(overrides)
    return Budget(**defaults)


def _intent(**overrides) -> IntentMandate:
    defaults = dict(
        mandate_id="intent-1",
        issued_at=_now(),
        expires_at=_now() + timedelta(minutes=10),
        principal_kid="kid-human",
        agent_kid="kid-buyer-agent",
        request_text="2kg basmati rice under 500 rupees",
        hard=HardConstraints(category="grocery", excluded_categories=["alcohol"]),
        soft=SoftPreferences(substitution_tolerance_pct=10.0),
        budget=_budget(),
        allowed_merchants=["merchant-1"],
    )
    defaults.update(overrides)
    return IntentMandate(**defaults)


def _clean_provenance() -> Provenance:
    return Provenance(
        source_url="https://merchant.example/product/rice",
        extraction_method="schema_org",
        signature_verified=True,
        extracted_at=_now(),
    )


def _cart(intent: IntentMandate, *, total_paise: int = 45_000, **overrides) -> CartMandate:
    line = CartLine(
        sku="RICE-2KG",
        title="Basmati Rice 2kg",
        category="grocery",
        unit_price_paise=total_paise,
        quantity=1,
        field_provenance={"unit_price_paise": _clean_provenance(), "title": _clean_provenance()},
    )
    defaults = dict(
        mandate_id="cart-1",
        issued_at=_now(),
        expires_at=_now() + timedelta(minutes=10),
        merchant_id="merchant-1",
        merchant_kid="kid-merchant",
        intent_hash=mandate_hash(intent),
        lines=[line],
        subtotal_paise=total_paise,
        tax_paise=0,
        shipping_paise=0,
        total_paise=total_paise,
    )
    defaults.update(overrides)
    return CartMandate(**defaults)


# ---------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------


def test_naive_datetime_rejected():
    with pytest.raises(ValidationError, match="timezone-aware"):
        _intent(issued_at=datetime(2026, 8, 24, 12, 0, 0))  # no tzinfo


def test_expiry_before_issuance_rejected():
    with pytest.raises(ValidationError, match="must be after"):
        _intent(expires_at=_now() - timedelta(minutes=1))


def test_escalate_above_the_hard_cap_rejected_as_config_error():
    """Still a real misconfiguration: a tripwire sitting beyond the hard
    budget cap is unreachable, because G4.3 rejects anything over
    total_paise before G7.1 could ever fire."""
    with pytest.raises(ValidationError, match="unreachable"):
        _budget(total_paise=1000, escalate_above_paise=1001)


def test_escalate_equal_to_total_means_no_tripwire_and_is_allowed():
    """This used to raise, which made "spend up to Rs500 without asking
    me" literally inexpressible -- every IntentMandate was born with a
    mandatory interrupt, so an in-policy purchase touched the human twice
    (sign, then escalate) with a full merchant re-negotiation in between.
    Equality is now the default and simply means the G7.1 tripwire is
    off; G4.3 still enforces the hard cap."""
    b = _budget(total_paise=1000, escalate_above_paise=1000)
    assert b.escalate_above_paise == b.total_paise


def test_escalate_strictly_below_total_is_still_supported():
    """The tripwire survives as a deliberate capability -- "spend up to
    Rs1000 but check with me above Rs999" -- set by the human at review
    time, never derived by the agent."""
    b = _budget(total_paise=1000, escalate_above_paise=999)
    assert b.escalate_above_paise < b.total_paise


def test_confidence_forbidden_outside_llm_inferred():
    with pytest.raises(ValidationError, match="should only be set"):
        Provenance(
            source_url="https://x",
            extraction_method="schema_org",
            signature_verified=True,
            extracted_at=_now(),
            confidence=0.9,
        )


def test_confidence_allowed_for_llm_inferred():
    p = Provenance(
        source_url="https://x",
        extraction_method="llm_inferred",
        signature_verified=False,
        extracted_at=_now(),
        confidence=0.7,
    )
    assert p.confidence == 0.7


def test_negative_paise_rejected():
    with pytest.raises(ValidationError):
        Budget(total_paise=-1, per_item_paise=100, max_quantity=1, escalate_above_paise=0)


# ---------------------------------------------------------------------
# Canonicalization determinism
# ---------------------------------------------------------------------


def test_canonical_bytes_independent_of_key_order():
    a = canonical_bytes({"b": 1, "a": [1, 2, 3], "c": {"z": 1, "y": 2}})
    b = canonical_bytes({"a": [1, 2, 3], "c": {"y": 2, "z": 1}, "b": 1})
    assert a == b


def test_mandate_hash_deterministic_across_calls():
    intent = _intent()
    assert mandate_hash(intent) == mandate_hash(intent)


def test_mandate_hash_changes_on_any_field_change():
    intent = _intent()
    mutated = _intent(mandate_id="intent-2")
    assert mandate_hash(intent) != mandate_hash(mutated)


# ---------------------------------------------------------------------
# Sign / verify round trip and tamper detection
# ---------------------------------------------------------------------


def test_sign_then_verify_round_trip():
    kp = generate_keypair()
    directory = KeyDirectory()
    directory.register_keypair(kp)

    intent = _intent()
    envelope = sign_mandate(intent, kp)
    result = verify_envelope(envelope, directory)

    assert result.valid, result.reason
    assert kp.kid in result.verified_kids


def test_verify_fails_for_unknown_kid():
    kp = generate_keypair()
    empty_directory = KeyDirectory()  # kp never registered

    envelope = sign_mandate(_intent(), kp)
    result = verify_envelope(envelope, empty_directory)

    assert not result.valid
    assert "unknown kid" in result.reason


def test_tamper_after_signing_is_detected():
    """The core tamper test the plan calls out explicitly: mutate
    cart.total_paise post-signature -> verification must fail."""
    kp = generate_keypair()
    directory = KeyDirectory()
    directory.register_keypair(kp)

    intent = _intent()
    cart = _cart(intent)
    envelope = sign_mandate(cart, kp)

    # Simulate an attacker (or a bug) mutating the payload after signing --
    # e.g. attack variant A4_price_swap rewriting the total post-hoc.
    tampered = envelope.model_copy(deep=True)
    tampered.payload["total_paise"] = tampered.payload["total_paise"] * 10

    result = verify_envelope(tampered, directory)
    assert not result.valid
    assert "invalid" in result.reason


def test_tamper_on_nested_field_is_detected():
    kp = generate_keypair()
    directory = KeyDirectory()
    directory.register_keypair(kp)

    intent = _intent()
    cart = _cart(intent)
    envelope = sign_mandate(cart, kp)

    tampered = envelope.model_copy(deep=True)
    tampered.payload["lines"][0]["unit_price_paise"] = 1

    result = verify_envelope(tampered, directory)
    assert not result.valid


def test_envelope_with_no_signatures_is_invalid():
    from mandates.schemas import Envelope

    envelope = Envelope(payload={"a": 1}, signatures=[])
    result = verify_envelope(envelope, KeyDirectory())
    assert not result.valid
    assert "no signatures" in result.reason


def test_required_kid_enforced():
    signer = generate_keypair()
    other = generate_keypair()
    directory = KeyDirectory()
    directory.register_keypair(signer)
    directory.register_keypair(other)

    envelope = sign_mandate(_intent(), signer)

    ok = verify_envelope(envelope, directory, required_kid=signer.kid)
    assert ok.valid

    bad = verify_envelope(envelope, directory, required_kid=other.kid)
    assert not bad.valid
    assert "did not sign" in bad.reason


def test_wrong_key_cannot_forge_signature():
    """A different, valid keypair's signature must not verify against a
    payload it didn't actually sign, even though both are individually
    well-formed Ed25519 signatures."""
    real_signer = generate_keypair()
    attacker = generate_keypair()
    directory = KeyDirectory()
    directory.register_keypair(real_signer)  # attacker's key is NOT registered

    envelope = sign_mandate(_intent(), real_signer)
    # Attacker swaps in their own signature over the same payload bytes,
    # claiming to be real_signer's kid.
    from mandates.canonical import canonical_bytes
    from mandates.keys import b64url_encode
    from mandates.schemas import Signature

    forged_sig = attacker.private_key.sign(canonical_bytes(envelope.payload))
    forged = envelope.model_copy(
        update={"signatures": [Signature(kid=real_signer.kid, sig=b64url_encode(forged_sig))]}
    )
    result = verify_envelope(forged, directory)
    assert not result.valid  # real_signer's public key can't verify attacker's signature


# ---------------------------------------------------------------------
# Chain linkage
# ---------------------------------------------------------------------


def test_chain_intact_for_correctly_linked_mandates():
    intent = _intent()
    cart = _cart(intent)
    payment = PaymentMandate(
        mandate_id="pay-1",
        created_at=_now(),
        intent_hash=mandate_hash(intent),
        cart_hash=mandate_hash(cart),
        amount_paise=cart.total_paise,
    )
    result = verify_chain(intent, cart, payment)
    assert result.intact, result.violations


def test_chain_broken_when_cart_intent_hash_wrong():
    intent = _intent()
    cart = _cart(intent, intent_hash="0" * 64)
    result = verify_chain(intent, cart)
    assert not result.intact
    assert any("cart.intent_hash" in v for v in result.violations)


def test_chain_broken_when_payment_cart_hash_wrong():
    intent = _intent()
    cart = _cart(intent)
    payment = PaymentMandate(
        mandate_id="pay-1",
        created_at=_now(),
        intent_hash=mandate_hash(intent),
        cart_hash="0" * 64,  # wrong
        amount_paise=cart.total_paise,
    )
    result = verify_chain(intent, cart, payment)
    assert not result.intact
    assert any("payment.cart_hash" in v for v in result.violations)


def test_chain_verification_without_payment_only_checks_intent_cart():
    intent = _intent()
    cart = _cart(intent)
    result = verify_chain(intent, cart, payment=None)
    assert result.intact


# ---------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------


def test_is_expired_false_at_exact_boundary():
    intent = _intent(issued_at=_now() - timedelta(minutes=1), expires_at=_now())
    assert not is_expired(intent, now=_now())  # now > expires_at is False when equal


def test_is_expired_true_one_microsecond_after():
    intent = _intent(issued_at=_now() - timedelta(minutes=1), expires_at=_now())
    assert is_expired(intent, now=_now() + timedelta(microseconds=1))


def test_is_expired_false_one_microsecond_before():
    intent = _intent(expires_at=_now() + timedelta(microseconds=1))
    assert not is_expired(intent, now=_now())


# ---------------------------------------------------------------------
# KeyDirectory / JWKS
# ---------------------------------------------------------------------


def test_jwks_round_trip():
    kp = generate_keypair()
    jwks = {"keys": [kp.to_jwk()]}
    directory = KeyDirectory.from_jwks(jwks)
    resolved = directory.resolve(kp.kid)
    assert resolved is not None
    assert resolved.public_bytes_raw() if hasattr(resolved, "public_bytes_raw") else True


def test_jwks_entry_with_mismatched_kid_rejected():
    kp = generate_keypair()
    jwk = kp.to_jwk()
    jwk["kid"] = "not-the-real-thumbprint"
    with pytest.raises(ValueError, match="doesn't match its key"):
        KeyDirectory.from_jwks({"keys": [jwk]})


def test_jwks_ignores_non_ed25519_entries():
    directory = KeyDirectory.from_jwks({"keys": [{"kty": "RSA", "n": "...", "e": "AQAB"}]})
    assert directory.resolve("anything") is None


# ---------------------------------------------------------------------
# Immutability (closes the "mutated after verification" gap)
# ---------------------------------------------------------------------


def test_cart_mandate_is_frozen_against_in_place_mutation():
    """If this test ever fails, the type-level protection against
    'signature verified against original bytes, but gates evaluated
    against a since-mutated object' has been silently removed -- see
    FrozenModel's docstring in mandates/schemas.py for why that matters."""
    cart = _cart(_intent())
    with pytest.raises((ValidationError, TypeError)):
        cart.total_paise = 1


def test_intent_mandate_is_frozen_against_in_place_mutation():
    intent = _intent()
    with pytest.raises((ValidationError, TypeError)):
        intent.budget = _budget(total_paise=1)


def test_envelope_payload_dict_remains_mutable_for_tamper_simulation():
    """Envelope is deliberately NOT frozen -- its payload dict stands in
    for wire bytes, and tests need to mutate it to simulate tampering."""
    kp = generate_keypair()
    envelope = sign_mandate(_intent(), kp)
    envelope.payload["request_text"] = "mutated"  # must not raise
    assert envelope.payload["request_text"] == "mutated"
