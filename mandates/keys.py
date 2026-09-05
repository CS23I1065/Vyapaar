"""
Ed25519 key generation, RFC 7638 JWK thumbprints (used as `kid` values
throughout -- principal_kid, agent_kid, merchant_kid), and a KeyDirectory
for resolving a kid back to a public key during verification.

This is the one place Ed25519 key handling lives; mandates/sign.py,
mandates/verify.py, and identity/ (RFC 9421 request signing) all build
on it rather than each rolling their own key plumbing.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

__all__ = [
    "InvalidSignature",
    "KeyDirectory",
    "KeyPair",
    "b64url_decode",
    "b64url_encode",
    "generate_keypair",
    "jwk_thumbprint",
    "load_keypair",
]


def b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def jwk_thumbprint(pub: Ed25519PublicKey) -> str:
    """RFC 7638 JWK thumbprint for an Ed25519 (OKP) key. Used as the `kid`
    for principal_kid / agent_kid / merchant_kid throughout mandates/.

    RFC 7638 requires the thumbprint input to be the *exact* required JWK
    members in lexicographic order with no whitespace -- constructing that
    string by hand (rather than via a generic canonicalizer) makes the
    required member set and ordering explicit and auditable."""
    raw = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
    x = b64url_encode(raw)
    jwk_canonical = f'{{"crv":"Ed25519","kty":"OKP","x":"{x}"}}'
    digest = hashlib.sha256(jwk_canonical.encode("utf-8")).digest()
    return b64url_encode(digest)


@dataclass(frozen=True)
class KeyPair:
    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey
    kid: str  # RFC 7638 thumbprint of public_key

    def to_jwk(self) -> dict:
        raw = self.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        return {"kty": "OKP", "crv": "Ed25519", "x": b64url_encode(raw), "kid": self.kid}

    def save(self, path: Path) -> None:
        """Persists both keys. The private half is written 0600 -- this is
        demo/test tooling, not a production KMS, and that boundary is
        stated in the README rather than pretended away."""
        raw_priv = self.private_key.private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kid": self.kid,
            "private_key_b64url": b64url_encode(raw_priv),
            "public_jwk": self.to_jwk(),
        }
        path.write_text(json.dumps(payload, indent=2))
        path.chmod(0o600)


def generate_keypair() -> KeyPair:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    return KeyPair(private_key=priv, public_key=pub, kid=jwk_thumbprint(pub))


def load_keypair(path: Path) -> KeyPair:
    payload = json.loads(path.read_text())
    raw_priv = b64url_decode(payload["private_key_b64url"])
    priv = Ed25519PrivateKey.from_private_bytes(raw_priv)
    pub = priv.public_key()
    kid = jwk_thumbprint(pub)
    if kid != payload["kid"]:
        raise ValueError(
            f"key file {path} is corrupt: stored kid {payload['kid']!r} does "
            f"not match the thumbprint derived from its own private key "
            f"({kid!r})"
        )
    return KeyPair(private_key=priv, public_key=pub, kid=kid)


class KeyDirectory:
    """Resolves a kid to an Ed25519PublicKey. Used by mandates/verify.py
    (mandate signature verification) and, structurally, by identity/
    (RFC 9421 request signature verification) -- same shape, different
    population source (a JWKS fetched from a merchant's
    .well-known/agent-keys.json vs an in-process key registered by the
    same process)."""

    def __init__(self) -> None:
        self._keys: dict[str, Ed25519PublicKey] = {}

    def register(self, kid: str, pub: Ed25519PublicKey) -> None:
        self._keys[kid] = pub

    def register_keypair(self, kp: KeyPair) -> None:
        self.register(kp.kid, kp.public_key)

    @classmethod
    def from_jwks(cls, jwks: dict) -> "KeyDirectory":
        """jwks: {"keys": [{"kty":"OKP","crv":"Ed25519","x":"...","kid":"..."}]}
        -- the .well-known/agent-keys.json shape (merchant_kit/sign.py)."""
        directory = cls()
        for jwk in jwks.get("keys", []):
            if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519":
                continue  # not our key type; ignore rather than error, so a
                # future non-Ed25519 key in the same JWKS doesn't break us
            raw = b64url_decode(jwk["x"])
            pub = Ed25519PublicKey.from_public_bytes(raw)
            real_kid = jwk_thumbprint(pub)
            claimed_kid = jwk.get("kid")
            if claimed_kid and claimed_kid != real_kid:
                raise ValueError(
                    f"JWKS entry claims kid={claimed_kid!r} but its own key "
                    f"material thumbprints to {real_kid!r} -- refusing to "
                    f"trust a kid that doesn't match its key"
                )
            directory.register(real_kid, pub)
        return directory

    def resolve(self, kid: str) -> Ed25519PublicKey | None:
        return self._keys.get(kid)

    def __contains__(self, kid: str) -> bool:
        return kid in self._keys
