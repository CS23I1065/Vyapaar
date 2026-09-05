"""
Signs a Catalog and MerchantPolicy with the merchant's Ed25519 key,
producing the Envelope wire format from mandates/. Reused directly rather
than reimplemented -- mandates.sign_mandate operates on any pydantic
BaseModel, not just Intent/Cart/Payment.

Both .well-known/agent-catalog.json and .well-known/agent-policy.json are
signed, not just the catalog -- an unsigned policy document would let a
compromised or malicious server quietly serve a more "generous" policy to
trick a buyer client that doesn't independently verify it.
"""

from __future__ import annotations

from mandates.keys import KeyPair
from mandates.schemas import Envelope, MerchantPolicy
from mandates.sign import sign_mandate

from .schemas import Catalog


def sign_catalog(catalog: Catalog, merchant_keypair: KeyPair) -> Envelope:
    return sign_mandate(catalog, merchant_keypair)


def sign_policy(policy: MerchantPolicy, merchant_keypair: KeyPair) -> Envelope:
    return sign_mandate(policy, merchant_keypair)


def jwks_for(merchant_keypair: KeyPair) -> dict:
    return {"keys": [merchant_keypair.to_jwk()]}
