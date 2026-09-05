"""
RFC 9421 verification, wrapping http_message_signatures.HTTPMessageVerifier
with two checks it does not do on its own:

1. Independent Content-Digest recomputation. The library only checks that
   the DECLARED digest header matches what was covered in the signature
   base at signing time; it does not recompute the digest from the
   actual current body. Tampering the body while keeping a stale-but-
   validly-signed Content-Digest header still passes the library's own
   verify() -- without this module's check, that's a body-substitution
   hole.
2. Independent created/expires window enforcement. The library's own
   `max_age` parameter does not reject a signature even at
   max_age=timedelta(seconds=0), so it isn't used here -- the +/-300s
   freshness window is enforced independently instead.

Failure modes get distinct codes: SIG_MISSING, SIG_KEY_UNKNOWN, SIG_BAD,
SIG_STALE, SIG_REPLAY, CONTENT_DIGEST_MISMATCH.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import httpx
from http_message_signatures import HTTPMessageVerifier, algorithms
from http_message_signatures.exceptions import HTTPMessageSignaturesException

from mandates.keys import KeyDirectory

from .content_digest import content_digest_matches
from .resolver import MandatesKeyResolver

DEFAULT_SKEW_SECONDS = 300


@dataclass(frozen=True)
class SignedMessage:
    """Transport-agnostic view of a signed HTTP message -- lets
    verify_request() work identically whether the caller has an
    httpx.Request (tests, or the buyer re-verifying its own request) or a
    Starlette Request (the merchant server's middleware, after awaiting
    the body). The RFC 9421 component resolver only needs .method, .url,
    .headers, duck-typed against either.

    `headers` is deliberately `httpx.Headers`, not a plain dict -- the
    underlying library looks headers up by exact Title-Case name
    (`"Signature-Input"`, `"Signature"`) internally, which silently
    fails to find a lowercase-keyed plain dict even though the header is
    genuinely present. httpx.Headers is case-insensitive on lookup,
    which is what HTTP actually requires."""

    method: str
    url: str
    headers: httpx.Headers
    content: bytes

    @classmethod
    def from_httpx(cls, request) -> "SignedMessage":
        return cls(method=request.method, url=str(request.url), headers=request.headers, content=request.content or b"")

    @classmethod
    async def from_starlette(cls, request) -> "SignedMessage":
        body = await request.body()
        return cls(method=request.method, url=str(request.url), headers=httpx.Headers(dict(request.headers)), content=body)


@dataclass(frozen=True)
class IdentityCheck:
    passed: bool
    code: str
    detail: str
    keyid: str | None = None


@dataclass
class NonceCache:
    """In-memory (kid, nonce) replay cache. Entries are pruned lazily
    (on the next check) once past their signature's own expiry -- no
    unbounded growth from a long-running merchant server."""

    _seen: dict[tuple[str, str], float] = field(default_factory=dict)

    def check_and_record(self, keyid: str, nonce: str, expires_epoch: float, now_epoch: float) -> bool:
        expired = [k for k, exp in self._seen.items() if exp < now_epoch]
        for k in expired:
            del self._seen[k]
        key = (keyid, nonce)
        if key in self._seen:
            return False
        self._seen[key] = expires_epoch
        return True


def verify_request(
    message: SignedMessage,
    *,
    directory: KeyDirectory,
    nonce_cache: NonceCache,
    now: datetime,
    skew_seconds: int = DEFAULT_SKEW_SECONDS,
) -> IdentityCheck:
    if "signature-input" not in message.headers or "signature" not in message.headers:
        return IdentityCheck(False, "SIG_MISSING", "Signature-Input/Signature header absent")

    verifier = HTTPMessageVerifier(
        signature_algorithm=algorithms.ED25519, key_resolver=MandatesKeyResolver(directory=directory)
    )

    try:
        # max_age is intentionally huge: the library's own expiry check
        # runs against the real wall clock, not the injected `now`, so a
        # short max_age here would just be a second, untestable,
        # wall-clock-coupled expiry check fighting the one below. The
        # created/expires window is enforced independently using the
        # injected `now`, the only clock this function trusts (same
        # discipline as buyer/policy_engine.py).
        results = verifier.verify(message, max_age=timedelta(days=36500))
    except KeyError as e:
        return IdentityCheck(False, "SIG_KEY_UNKNOWN", str(e))
    except HTTPMessageSignaturesException as e:
        return IdentityCheck(False, "SIG_BAD", str(e))

    if not results:
        return IdentityCheck(False, "SIG_BAD", "verifier returned no signature results")

    result = results[0]  # we only ever produce one signature label ("sig1" / "pyhms")
    keyid = result.parameters.get("keyid")
    created = result.parameters.get("created")
    expires = result.parameters.get("expires")
    nonce = result.parameters.get("nonce")

    if created is None or expires is None:
        return IdentityCheck(False, "SIG_BAD", "created/expires missing from signature parameters", keyid)

    now_epoch = now.timestamp()
    if now_epoch < created - skew_seconds or now_epoch > expires + skew_seconds:
        return IdentityCheck(
            False, "SIG_STALE",
            f"now={now_epoch} outside signature window [{created - skew_seconds}, {expires + skew_seconds}]",
            keyid,
        )

    if not nonce:
        return IdentityCheck(False, "SIG_BAD", "nonce missing from signature parameters", keyid)
    if not nonce_cache.check_and_record(keyid, nonce, expires + skew_seconds, now_epoch):
        return IdentityCheck(False, "SIG_REPLAY", f"nonce {nonce!r} already seen for keyid {keyid!r}", keyid)

    declared_digest = message.headers.get("content-digest")
    if declared_digest is None:
        return IdentityCheck(False, "SIG_BAD", "content-digest header missing", keyid)
    if not content_digest_matches(declared_digest, message.content):
        return IdentityCheck(False, "CONTENT_DIGEST_MISMATCH", "declared content-digest does not match actual body", keyid)

    return IdentityCheck(True, "OK", "signature verified", keyid)
