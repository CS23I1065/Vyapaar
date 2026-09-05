"""
RFC 9421 HTTP Message Signatures signer (Web Bot Auth style). Built on
the independent `http_message_signatures` library rather than a
hand-rolled signature-base implementation -- this is a fiddly spec
(component ordering, @authority vs Host, covered-components mismatches),
and reusing a maintained, tested, RFC-9421-conformant library removes
exactly that risk instead of accepting it for the sake of "we built it
ourselves."

The buyer agent signs every outbound request with this. Covered
components match the plan exactly: @method, @authority, @path,
content-digest, signature-agent.
"""

from __future__ import annotations

import datetime
import secrets

import httpx
from http_message_signatures import HTTPMessageSigner, algorithms

from mandates.keys import KeyPair

from .content_digest import compute_content_digest
from .resolver import MandatesKeyResolver

COVERED_COMPONENTS = ("@method", "@authority", "@path", "content-digest", "signature-agent")
DEFAULT_TTL_SECONDS = 300


def sign_request(
    method: str,
    url: str,
    *,
    body: bytes,
    keypair: KeyPair,
    signature_agent_url: str,
    now: datetime.datetime,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> httpx.Request:
    """`now` is injected, not read internally -- consistent with this
    project's clock-injection discipline elsewhere (buyer/policy_engine.py),
    and it's what makes a signed request byte-for-byte reproducible in
    tests instead of depending on wall-clock timing."""
    content_digest = compute_content_digest(body)
    request = httpx.Request(
        method, url, content=body,
        headers={
            "content-digest": content_digest,
            "signature-agent": f'"{signature_agent_url}"',
        },
    )

    signer = HTTPMessageSigner(
        signature_algorithm=algorithms.ED25519,
        key_resolver=MandatesKeyResolver(signing_keypair=keypair),
    )
    nonce = secrets.token_urlsafe(16)
    signer.sign(
        request, key_id=keypair.kid, created=now, expires=now + datetime.timedelta(seconds=ttl_seconds),
        nonce=nonce, covered_component_ids=COVERED_COMPONENTS,
    )
    return request
