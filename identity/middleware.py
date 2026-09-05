"""
Browse-only downgrade: catalog reads succeed for anyone, but any endpoint
the caller marks as requiring identity (checkout/negotiation) returns
403 without a valid RFC 9421 signature. The brief's "no toolkit means no
wired connection to checkout" incentive design, enforced cryptographically
here rather than by convention -- an unsigned agent simply cannot reach a
transacting endpoint at all, regardless of what it sends.

`requires_identity` is caller-supplied rather than hardcoded, since the
concrete transacting paths live in merchant_agent/ -- this middleware
doesn't need to know what they are, only whether a given path needs a
signature.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mandates.keys import KeyDirectory

from .verifier import DEFAULT_SKEW_SECONDS, IdentityCheck, NonceCache, SignedMessage, verify_request


class IdentityMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        directory: KeyDirectory,
        requires_identity: Callable[[str], bool],
        nonce_cache: NonceCache | None = None,
        skew_seconds: int = DEFAULT_SKEW_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(app)
        self._directory = directory
        self._requires_identity = requires_identity
        self._nonce_cache = nonce_cache or NonceCache()
        self._skew_seconds = skew_seconds
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self._requires_identity(request.url.path):
            return await call_next(request)

        message = await SignedMessage.from_starlette(request)
        check: IdentityCheck = verify_request(
            message, directory=self._directory, nonce_cache=self._nonce_cache,
            now=self._clock(), skew_seconds=self._skew_seconds,
        )
        if not check.passed:
            return JSONResponse(
                {"error": "agent_identity_required", "code": check.code, "detail": check.detail},
                status_code=403,
            )

        request.state.verified_keyid = check.keyid
        return await call_next(request)
