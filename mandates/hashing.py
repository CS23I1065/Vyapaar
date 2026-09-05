"""Shared payload/hash helpers used by both sign.py (producing the hash
that gets signed) and chain.py (re-deriving the same hash to check
linkage) -- kept in one place so the two can never drift apart on how a
mandate is turned into bytes."""

from __future__ import annotations

from pydantic import BaseModel

from .canonical import canonical_hash


def mandate_payload(mandate: BaseModel) -> dict:
    return mandate.model_dump(mode="json")


def mandate_hash(mandate: BaseModel) -> str:
    return canonical_hash(mandate_payload(mandate))
