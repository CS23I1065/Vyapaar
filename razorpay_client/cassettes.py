"""
Record/replay httpx transport for razorpay_client/, so the eval harness
and CI run offline and deterministically -- the demo runs live. Keyed by
(method, path, sha256(body)), so re-recording is only needed when a
request actually changes.

Matches the real httpx 0.28.1 API: request.url carries `raw_path:
bytes`, and request.content is populated synchronously for a json= body.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx


class CassetteMiss(RuntimeError):
    pass


def _key(request: httpx.Request) -> str:
    raw_path = request.url.raw_path
    path = raw_path.decode("utf-8") if isinstance(raw_path, bytes) else str(raw_path)
    body_hash = hashlib.sha256(request.content or b"").hexdigest()[:16]
    return f"{request.method} {path} {body_hash}"


@dataclass
class RecordingTransport(httpx.BaseTransport):
    """Forwards to a real network transport, then persists
    key -> {status_code, body} to a JSON file on disk."""

    cassette_path: Path
    _inner: httpx.HTTPTransport = field(default_factory=httpx.HTTPTransport)
    _store: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cassette_path.exists():
            self._store = json.loads(self.cassette_path.read_text())

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        body = response.read()
        self._store[_key(request)] = {
            "status_code": response.status_code,
            "body": body.decode("utf-8", errors="replace"),
        }
        self.cassette_path.parent.mkdir(parents=True, exist_ok=True)
        self.cassette_path.write_text(json.dumps(self._store, indent=2, sort_keys=True))
        return httpx.Response(response.status_code, content=body, request=request)

    def close(self) -> None:
        self._inner.close()


@dataclass
class ReplayTransport(httpx.BaseTransport):
    """Zero network calls. Raises CassetteMiss (not a silent fallback to
    the network) when a request wasn't recorded -- a cache miss here must
    be loud, since it means the eval/demo would otherwise diverge from
    what was actually verified."""

    cassette_path: Path
    _store: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cassette_path.exists():
            raise CassetteMiss(f"no cassette found at {self.cassette_path}")
        self._store = json.loads(self.cassette_path.read_text())

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = _key(request)
        entry = self._store.get(key)
        if entry is None:
            raise CassetteMiss(f"no recorded response for {key!r} in {self.cassette_path}")
        return httpx.Response(entry["status_code"], content=entry["body"].encode("utf-8"), request=request)
