"""
RFC 9530 Content-Digest, computed independently of the RFC 9421 library.

`http_message_signatures` does not recompute this from the actual
request body during verification -- it only checks that the DECLARED
header value matches what was covered in the signature base at signing
time. An attacker who swaps the body while leaving an old, still-
validly-signed Content-Digest header in place would pass RFC 9421
verification cleanly unless something separately recomputes the digest
from the real body and compares. That's this module's entire job.
"""

from __future__ import annotations

import base64
import hashlib


def compute_content_digest(body: bytes) -> str:
    digest = base64.b64encode(hashlib.sha256(body).digest()).decode()
    return f"sha-256=:{digest}:"


def content_digest_matches(declared_header_value: str, body: bytes) -> bool:
    return declared_header_value.strip() == compute_content_digest(body)
