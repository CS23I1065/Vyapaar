"""Writes the .well-known/ trio a buyer agent needs to discover and
trust a merchant: agent-catalog.json (signed), agent-policy.json
(signed), agent-keys.json (JWKS, unsigned by construction -- it's the
root of trust the two signatures are verified against)."""

from __future__ import annotations

import json
from pathlib import Path

from mandates.keys import KeyPair
from mandates.schemas import MerchantPolicy

from .schemas import Catalog
from .sign import jwks_for, sign_catalog, sign_policy


def emit(catalog: Catalog, policy: MerchantPolicy, merchant_keypair: KeyPair, *, out_dir: Path) -> dict[str, Path]:
    well_known = out_dir / ".well-known"
    well_known.mkdir(parents=True, exist_ok=True)

    catalog_envelope = sign_catalog(catalog, merchant_keypair)
    policy_envelope = sign_policy(policy, merchant_keypair)
    jwks = jwks_for(merchant_keypair)

    paths = {
        "catalog": well_known / "agent-catalog.json",
        "policy": well_known / "agent-policy.json",
        "keys": well_known / "agent-keys.json",
    }
    paths["catalog"].write_text(catalog_envelope.model_dump_json(indent=2))
    paths["policy"].write_text(policy_envelope.model_dump_json(indent=2))
    paths["keys"].write_text(json.dumps(jwks, indent=2))

    return paths
