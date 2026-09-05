"""
Bridges mandates.keys' Ed25519 infrastructure into
http_message_signatures' HTTPSignatureKeyResolver interface, so identity/
doesn't maintain a second, parallel key representation. The buyer agent's
Web Bot Auth identity key IS its mandate-signing key -- same KeyPair,
same trust anchor -- not a separate credential to provision and rotate.
"""

from __future__ import annotations

from http_message_signatures import HTTPSignatureKeyResolver

from mandates.keys import KeyDirectory, KeyPair


class MandatesKeyResolver(HTTPSignatureKeyResolver):
    def __init__(self, *, directory: KeyDirectory | None = None, signing_keypair: KeyPair | None = None) -> None:
        self._directory = directory
        self._signing_keypair = signing_keypair

    def resolve_public_key(self, key_id: str):
        if self._directory is None:
            raise KeyError(f"no key directory configured to resolve keyid {key_id!r}")
        pub = self._directory.resolve(key_id)
        if pub is None:
            raise KeyError(f"unknown keyid {key_id!r}")
        return pub

    def resolve_private_key(self, key_id: str):
        if self._signing_keypair is None or self._signing_keypair.kid != key_id:
            raise KeyError(f"no private key available for keyid {key_id!r}")
        return self._signing_keypair.private_key
