"""Signing a mandate model into a transmittable, verifiable Envelope."""

from __future__ import annotations

from pydantic import BaseModel

from .canonical import canonical_bytes
from .hashing import mandate_payload
from .keys import KeyPair, b64url_encode
from .schemas import Envelope, Signature


def sign_mandate(mandate: BaseModel, keypair: KeyPair) -> Envelope:
    payload = mandate_payload(mandate)
    sig_bytes = keypair.private_key.sign(canonical_bytes(payload))
    return Envelope(
        payload=payload,
        signatures=[Signature(kid=keypair.kid, sig=b64url_encode(sig_bytes))],
    )


def add_signature(envelope: Envelope, keypair: KeyPair) -> Envelope:
    """Co-sign an existing envelope. Signs exactly the bytes already
    present in envelope.payload (not a re-derivation from a model), so
    multiple signers provably agree on the same bytes."""
    sig_bytes = keypair.private_key.sign(canonical_bytes(envelope.payload))
    new_sig = Signature(kid=keypair.kid, sig=b64url_encode(sig_bytes))
    return envelope.model_copy(update={"signatures": [*envelope.signatures, new_sig]})
