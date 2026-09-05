"""
Tests for identity/. Two things this suite exists specifically to prove:

1. Bidirectional cross-verification against an INDEPENDENT RFC 9421
   implementation (http_message_signatures' own Signer/Verifier, used
   directly here, bypassing our wrapper) -- a self-round-trip only proves
   our signer and our verifier agree with each other, which says nothing
   about whether either would pass against a real, independent verifier.
2. The two checks the library does NOT do on its own: independent
   Content-Digest recomputation, and a real created/expires window (the
   library's own max_age parameter does not reject a signature even at
   max_age=0).
"""

from __future__ import annotations

import datetime

import httpx
import pytest
from http_message_signatures import HTTPMessageSigner, HTTPMessageVerifier, algorithms
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from identity.content_digest import compute_content_digest
from identity.middleware import IdentityMiddleware
from identity.signer import COVERED_COMPONENTS, sign_request
from identity.verifier import NonceCache, SignedMessage, verify_request
from mandates.keys import KeyDirectory, generate_keypair

UTC = datetime.timezone.utc


def _now() -> datetime.datetime:
    return datetime.datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _freeze_library_wall_clock(monkeypatch):
    """The underlying http_message_signatures library hardcodes
    `datetime.datetime.now()` inside its own expiry validation with no
    injection point at all: passing `max_age` does NOT bypass it, since
    expires-in-the-past is checked unconditionally against the real wall
    clock, separately from max_age. Without this fixture every test
    using a fixed `_now()` in the past relative to real wall-clock time
    would fail with "expires parameter is set to a time in the past"
    regardless of what this
    project's own clock-injection logic says.

    This freezes the library's internal clock to match `_now()`, using
    the same naive-local epoch conversion `_parse_integer_timestamp`
    itself uses internally, so the comparison stays self-consistent
    regardless of the test machine's timezone."""
    import http_message_signatures.signatures as sig_module

    frozen_instant = datetime.datetime.fromtimestamp(_now().timestamp())

    class _FrozenDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen_instant if tz is None else frozen_instant.astimezone(tz)

    monkeypatch.setattr(sig_module.datetime, "datetime", _FrozenDateTime)


def _directory_with(*keypairs) -> KeyDirectory:
    d = KeyDirectory()
    for kp in keypairs:
        d.register_keypair(kp)
    return d


# ---------------------------------------------------------------------
# Golden path
# ---------------------------------------------------------------------


def test_sign_then_verify_round_trip():
    kp = generate_keypair()
    directory = _directory_with(kp)
    request = sign_request(
        "POST", "https://merchant.local/checkout", body=b'{"a":1}',
        keypair=kp, signature_agent_url="https://buyer.local/directory", now=_now(),
    )
    message = SignedMessage.from_httpx(request)
    check = verify_request(message, directory=directory, nonce_cache=NonceCache(), now=_now())
    assert check.passed, check.detail
    assert check.code == "OK"
    assert check.keyid == kp.kid


# ---------------------------------------------------------------------
# Bidirectional cross-verification against the independent library,
# bypassing our own wrapper on one side each time
# ---------------------------------------------------------------------


def test_our_signer_output_verifies_under_the_raw_independent_library():
    """Our signer.py -> the library's own HTTPMessageVerifier directly."""
    kp = generate_keypair()
    request = sign_request(
        "POST", "https://merchant.local/checkout", body=b'{"a":1}',
        keypair=kp, signature_agent_url="https://buyer.local/directory", now=_now(),
    )

    class _Resolver:
        def resolve_public_key(self, key_id):
            return kp.public_key

    raw_verifier = HTTPMessageVerifier(signature_algorithm=algorithms.ED25519, key_resolver=_Resolver())
    results = raw_verifier.verify(request)  # raises on failure
    assert results[0].parameters["keyid"] == kp.kid


def test_the_raw_independent_library_output_verifies_under_our_verifier():
    """The library's own HTTPMessageSigner -> our verify_request()."""
    kp = generate_keypair()
    directory = _directory_with(kp)
    body = b'{"a":1}'
    content_digest = compute_content_digest(body)
    request = httpx.Request(
        "POST", "https://merchant.local/checkout", content=body,
        headers={"content-digest": content_digest, "signature-agent": '"https://buyer.local/directory"'},
    )

    class _Resolver:
        def resolve_private_key(self, key_id):
            return kp.private_key

    raw_signer = HTTPMessageSigner(signature_algorithm=algorithms.ED25519, key_resolver=_Resolver())
    now = _now()
    raw_signer.sign(
        request, key_id=kp.kid, created=now, expires=now + datetime.timedelta(seconds=300),
        nonce="independent-nonce-1", covered_component_ids=COVERED_COMPONENTS,
    )

    message = SignedMessage.from_httpx(request)
    check = verify_request(message, directory=directory, nonce_cache=NonceCache(), now=now)
    assert check.passed, check.detail
    assert check.keyid == kp.kid


# ---------------------------------------------------------------------
# Failure modes -- distinct codes
# ---------------------------------------------------------------------


def test_missing_signature_headers():
    message = SignedMessage(method="POST", url="https://x/y", headers={}, content=b"{}")
    check = verify_request(message, directory=_directory_with(), nonce_cache=NonceCache(), now=_now())
    assert not check.passed and check.code == "SIG_MISSING"


def test_unknown_keyid():
    kp = generate_keypair()
    other = generate_keypair()
    request = sign_request(
        "POST", "https://merchant.local/checkout", body=b"{}",
        keypair=kp, signature_agent_url="https://buyer.local/directory", now=_now(),
    )
    # directory only knows `other`, not `kp`
    check = verify_request(
        SignedMessage.from_httpx(request), directory=_directory_with(other), nonce_cache=NonceCache(), now=_now()
    )
    assert not check.passed and check.code == "SIG_KEY_UNKNOWN"


def test_tampered_signature_bytes_rejected():
    kp = generate_keypair()
    directory = _directory_with(kp)
    request = sign_request(
        "POST", "https://merchant.local/checkout", body=b"{}",
        keypair=kp, signature_agent_url="https://buyer.local/directory", now=_now(),
    )
    original = request.headers["signature"]
    request.headers["signature"] = original[:-6] + "AAAAA:"
    check = verify_request(SignedMessage.from_httpx(request), directory=directory, nonce_cache=NonceCache(), now=_now())
    assert not check.passed and check.code in ("SIG_BAD",)


def test_signature_stale_before_created_window():
    kp = generate_keypair()
    directory = _directory_with(kp)
    now = _now()
    request = sign_request(
        "POST", "https://merchant.local/checkout", body=b"{}",
        keypair=kp, signature_agent_url="https://buyer.local/directory", now=now, ttl_seconds=300,
    )
    too_early = now - datetime.timedelta(seconds=301)
    check = verify_request(SignedMessage.from_httpx(request), directory=directory, nonce_cache=NonceCache(), now=too_early)
    assert not check.passed and check.code == "SIG_STALE"


def test_signature_stale_after_expiry_window():
    kp = generate_keypair()
    directory = _directory_with(kp)
    now = _now()
    request = sign_request(
        "POST", "https://merchant.local/checkout", body=b"{}",
        keypair=kp, signature_agent_url="https://buyer.local/directory", now=now, ttl_seconds=300,
    )
    too_late = now + datetime.timedelta(seconds=300 + 301)
    check = verify_request(SignedMessage.from_httpx(request), directory=directory, nonce_cache=NonceCache(), now=too_late)
    assert not check.passed and check.code == "SIG_STALE"


def test_within_skew_tolerance_still_passes():
    kp = generate_keypair()
    directory = _directory_with(kp)
    now = _now()
    request = sign_request(
        "POST", "https://merchant.local/checkout", body=b"{}",
        keypair=kp, signature_agent_url="https://buyer.local/directory", now=now, ttl_seconds=300,
    )
    slightly_late = now + datetime.timedelta(seconds=300 + 100)  # within the +/-300s skew
    check = verify_request(SignedMessage.from_httpx(request), directory=directory, nonce_cache=NonceCache(), now=slightly_late)
    assert check.passed, check.detail


def test_nonce_replay_rejected_on_second_use():
    kp = generate_keypair()
    directory = _directory_with(kp)
    now = _now()
    request = sign_request(
        "POST", "https://merchant.local/checkout", body=b"{}",
        keypair=kp, signature_agent_url="https://buyer.local/directory", now=now,
    )
    cache = NonceCache()
    first = verify_request(SignedMessage.from_httpx(request), directory=directory, nonce_cache=cache, now=now)
    assert first.passed
    second = verify_request(SignedMessage.from_httpx(request), directory=directory, nonce_cache=cache, now=now)
    assert not second.passed and second.code == "SIG_REPLAY"


def test_nonce_cache_prunes_expired_entries():
    cache = NonceCache()
    assert cache.check_and_record("kid-1", "n1", expires_epoch=100.0, now_epoch=50.0)
    # advance past expiry -- a NEW nonce should trigger pruning of the old entry
    assert cache.check_and_record("kid-1", "n2", expires_epoch=500.0, now_epoch=200.0)
    assert ("kid-1", "n1") not in cache._seen


def test_content_digest_mismatch_after_body_substitution_is_caught():
    """The library alone does NOT catch a body substitution behind a
    stale-but-validly-signed Content-Digest header -- the independent
    recomputation must."""
    kp = generate_keypair()
    directory = _directory_with(kp)
    now = _now()
    request = sign_request(
        "POST", "https://merchant.local/checkout", body=b'{"amount_paise": 1000}',
        keypair=kp, signature_agent_url="https://buyer.local/directory", now=now,
    )
    tampered = SignedMessage(
        method=request.method, url=str(request.url), headers=request.headers,
        content=b'{"amount_paise": 999999999}',  # swapped, but content-digest+signature headers left stale
    )
    check = verify_request(tampered, directory=directory, nonce_cache=NonceCache(), now=now)
    assert not check.passed, check.detail
    assert check.code == "CONTENT_DIGEST_MISMATCH"


# ---------------------------------------------------------------------
# Middleware: browse-only downgrade
# ---------------------------------------------------------------------


def _build_app(directory: KeyDirectory, clock):
    async def catalog(request):
        return JSONResponse({"ok": True})

    async def checkout(request):
        return JSONResponse({"ok": True, "verified_keyid": request.state.verified_keyid})

    app = Starlette(routes=[Route("/.well-known/agent-catalog.json", catalog), Route("/checkout", checkout, methods=["POST"])])
    app.add_middleware(
        IdentityMiddleware, directory=directory, requires_identity=lambda path: path == "/checkout", clock=clock
    )
    return app


def test_middleware_allows_unsigned_catalog_read():
    app = _build_app(_directory_with(), clock=_now)
    client = TestClient(app)
    resp = client.get("/.well-known/agent-catalog.json")
    assert resp.status_code == 200


def test_middleware_blocks_unsigned_checkout():
    app = _build_app(_directory_with(), clock=_now)
    client = TestClient(app)
    resp = client.post("/checkout", json={"x": 1})
    assert resp.status_code == 403
    assert resp.json()["error"] == "agent_identity_required"


def test_middleware_allows_signed_checkout():
    kp = generate_keypair()
    directory = _directory_with(kp)
    app = _build_app(directory, clock=_now)
    client = TestClient(app)

    body = b'{"x": 1}'
    request = sign_request(
        "POST", "http://testserver/checkout", body=body,
        keypair=kp, signature_agent_url="https://buyer.local/directory", now=_now(),
    )
    resp = client.post(
        "/checkout", content=body,
        headers={
            "content-digest": request.headers["content-digest"],
            "signature-agent": request.headers["signature-agent"],
            "signature-input": request.headers["signature-input"],
            "signature": request.headers["signature"],
            "content-type": "application/json",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["verified_keyid"] == kp.kid
