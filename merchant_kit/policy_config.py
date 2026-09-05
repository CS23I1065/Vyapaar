"""
Per-merchant policy configuration, so a merchant's signed
.well-known/agent-policy.json is at least *attributable to a deliberate
choice* rather than to a constant in the toolkit's source.

Every merchant onboarded used to get byte-identical policy values from
four hardcoded DEFAULT_* constants in cli.py. That meant G6.1
(MERCHANT_POLICY_CONFLICT) was being exercised against the toolkit's own
placeholders, not against anything Chumbak, Fabindia or BigBasket
actually said -- while the document carrying those numbers was
cryptographically signed, which lends invented values an authority they
have not earned.

This does not turn the numbers into real merchant terms; nothing here
scrapes or negotiates a merchant's actual rules, and the README says so
plainly next to the point-in-time-catalog caveat. What it does is make
the values an explicit, reviewable, per-merchant input instead of an
invisible global one.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from mandates.schemas import MerchantPolicy

# Used only when no --policy file is supplied. Deliberately permissive:
# the buyer's own IntentMandate is the binding limit, and a merchant
# default that quietly tightened things would make the buyer's signed
# budget look like it was being honoured when the merchant's placeholder
# was actually doing the work.
DEFAULT_MAX_ORDER_VALUE_PAISE = 10_000_00
DEFAULT_SUBSTITUTION_TOLERANCE_PCT = 10.0
DEFAULT_MAX_QUANTITY_PER_SKU = 5
# High by default and deliberately so: G7.1 takes min(buyer, merchant),
# so a low merchant threshold would re-introduce exactly the routine
# escalation the buyer side just removed -- the merchant's value would
# become the only one that can fire. A merchant asking for a human check
# on large orders is a real capability; it should be an opt-in number
# someone chose, not a default that nags on every cart.
DEFAULT_ESCALATION_THRESHOLD_PAISE = 10_000_00


def default_policy(merchant_id: str) -> MerchantPolicy:
    return MerchantPolicy(
        merchant_id=merchant_id,
        max_order_value_paise=DEFAULT_MAX_ORDER_VALUE_PAISE,
        substitution_tolerance_pct=DEFAULT_SUBSTITUTION_TOLERANCE_PCT,
        max_quantity_per_sku=DEFAULT_MAX_QUANTITY_PER_SKU,
        restricted_categories=[],
        escalation_threshold_paise=DEFAULT_ESCALATION_THRESHOLD_PAISE,
    )


def load_policy(path: Path, merchant_id: str) -> MerchantPolicy:
    """Reads a TOML policy file. Either a flat table, or a table per
    merchant id keyed under [merchants.<id>], so one file can describe a
    whole onboarding run:

        # flat
        max_order_value_paise = 500000
        restricted_categories = ["alcohol"]

        # or per-merchant
        [merchants.chumbak]
        max_order_value_paise = 500000

    Unknown keys are an error, not a shrug -- a typo'd
    `max_order_value_pasie` that silently fell back to a default would
    produce a signed document that does not say what its author thought
    it said.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    section = data.get("merchants", {}).get(merchant_id, data)
    section = {k: v for k, v in section.items() if k != "merchants"}

    known = {
        "max_order_value_paise",
        "substitution_tolerance_pct",
        "max_quantity_per_sku",
        "restricted_categories",
        "escalation_threshold_paise",
    }
    unknown = set(section) - known
    if unknown:
        raise ValueError(
            f"{path}: unknown policy key(s) {sorted(unknown)} for merchant "
            f"{merchant_id!r}; known keys are {sorted(known)}"
        )

    base = default_policy(merchant_id)
    return MerchantPolicy(
        merchant_id=merchant_id,
        max_order_value_paise=section.get("max_order_value_paise", base.max_order_value_paise),
        substitution_tolerance_pct=section.get("substitution_tolerance_pct", base.substitution_tolerance_pct),
        max_quantity_per_sku=section.get("max_quantity_per_sku", base.max_quantity_per_sku),
        restricted_categories=list(section.get("restricted_categories", base.restricted_categories)),
        escalation_threshold_paise=section.get("escalation_threshold_paise", base.escalation_threshold_paise),
    )


def resolve_policy(policy_path: Path | None, merchant_id: str) -> tuple[MerchantPolicy, str]:
    """Returns (policy, provenance) where provenance is either the config
    file path or the literal string "toolkit defaults" -- surfaced in the
    readiness report so nobody mistakes a default for a merchant term."""
    if policy_path is None:
        return default_policy(merchant_id), "toolkit defaults (not merchant-supplied)"
    return load_policy(policy_path, merchant_id), str(policy_path)
