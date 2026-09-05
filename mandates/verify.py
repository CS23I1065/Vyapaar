"""Verifying a signed Envelope against a KeyDirectory.

Deliberately returns a VerifyResult rather than raising or returning a
bare bool -- the policy engine (buyer/policy_engine.py) needs a reason
string to put in its Check.detail, and "verify() raises" would force
every caller into a try/except just to extract that detail."""

from __future__ import annotations

from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature

from .canonical import canonical_bytes
from .keys import KeyDirectory, b64url_decode
from .schemas import Envelope


@dataclass(frozen=True)
class VerifyResult:
    valid: bool
    reason: str | None = None
    verified_kids: tuple[str, ...] = ()


def verify_envelope(
    envelope: Envelope,
    key_directory: KeyDirectory,
    *,
    required_kid: str | None = None,
) -> VerifyResult:
    if not envelope.signatures:
        return VerifyResult(valid=False, reason="no signatures present on envelope")

    payload_bytes = canonical_bytes(envelope.payload)
    verified: list[str] = []
    for sig in envelope.signatures:
        pub = key_directory.resolve(sig.kid)
        if pub is None:
            return VerifyResult(
                valid=False, reason=f"unknown kid {sig.kid!r} -- not in key directory"
            )
        try:
            pub.verify(b64url_decode(sig.sig), payload_bytes)
        except InvalidSignature:
            return VerifyResult(valid=False, reason=f"signature invalid for kid {sig.kid!r}")
        verified.append(sig.kid)

    if required_kid is not None and required_kid not in verified:
        return VerifyResult(
            valid=False,
            reason=f"required signer {required_kid!r} did not sign this envelope",
            verified_kids=tuple(verified),
        )
    return VerifyResult(valid=True, reason=None, verified_kids=tuple(verified))
