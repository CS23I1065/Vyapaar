"""
RFC 8785 JSON Canonicalization Scheme, used to make mandate hashes and
signatures deterministic regardless of key insertion order, whitespace,
or dict-ordering quirks across Python versions/serializers.

Uses the real `rfc8785` library rather than a hand-rolled sorted-keys
json.dumps -- a hand-rolled version looks compliant on simple dicts but
diverges from the spec on number formatting (JCS requires
ECMAScript Number-to-String semantics), which is exactly the kind of
subtle bug that would pass a self-test and fail against any other
RFC 8785 implementation someone else builds against this repo.
"""

from __future__ import annotations

import hashlib

import rfc8785


def canonical_bytes(payload: dict) -> bytes:
    """payload must already be JSON-primitive (str/int/float/bool/None/
    dict/list) -- e.g. the output of pydantic's model_dump(mode="json"),
    which turns datetimes into ISO-8601 strings. This project uses int
    paise everywhere, so JCS's float-formatting edge cases never come
    into play for money fields."""
    return rfc8785.dumps(payload)


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()
