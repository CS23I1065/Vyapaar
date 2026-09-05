"""
Read-only HTTP fetch with snapshot caching. Every real fetch is written
to disk under snapshots_dir; --offline replays from there with zero
network calls, so a site changing, rate-limiting, or blocking requests
can't break a demo or make an eval run unreproducible.

No JS execution -- this is a plain HTTP GET, not a browser. That's a
deliberate scope boundary (read-only content extraction only, not live
browser/UI navigation), and it's also why a client-hydrated SPA
correctly yields zero schema.org markup here and falls through to the
LLM extraction path -- that's real behavior being exercised, not a bug.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .robots import USER_AGENT, RobotsCheck, enforce

MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 3
TIMEOUT_S = 15.0
MAX_CRAWL_DELAY_S = 5.0  # cap so a hostile Crawl-delay can't hang the toolkit


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchResult:
    url: str
    status_code: int
    html: str
    fetched_at: datetime
    snapshot_path: Path
    from_snapshot: bool
    robots: RobotsCheck | None = None
    # None in --offline mode only: replaying a local snapshot is not a
    # fetch, and the snapshot was obtained under a live check in the
    # first place. Every LIVE fetch carries the real check that allowed
    # it, so the readiness report can state WHY it was allowed rather
    # than hardcoding "allowed" (see report.py).


def _snapshot_path(snapshots_dir: Path, url: str) -> Path:
    host = urlparse(url).netloc
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return snapshots_dir / host / f"{digest}.html"


def fetch(
    url: str,
    *,
    snapshots_dir: Path,
    offline: bool = False,
    respect_robots: bool = True,
    timeout_s: float = TIMEOUT_S,
) -> FetchResult:
    path = _snapshot_path(snapshots_dir, url)

    if offline:
        if not path.exists():
            raise FetchError(
                f"--offline requested but no snapshot exists for {url} at {path}. "
                f"Run a live fetch first to populate it."
            )
        html = path.read_text(encoding="utf-8", errors="replace")
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return FetchResult(url=url, status_code=200, html=html, fetched_at=mtime, snapshot_path=path, from_snapshot=True)

    robots_result: RobotsCheck | None = None
    if respect_robots:
        robots_result = enforce(url, timeout_s=timeout_s)  # raises RobotsDisallowed if blocked
        if robots_result.crawl_delay:
            time.sleep(min(robots_result.crawl_delay, MAX_CRAWL_DELAY_S))

    with httpx.Client(follow_redirects=True, max_redirects=MAX_REDIRECTS, timeout=timeout_s) as client:
        resp = client.get(url, headers={"User-Agent": USER_AGENT})

    body = resp.content[:MAX_BODY_BYTES]
    html = body.decode(resp.encoding or "utf-8", errors="replace")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")

    return FetchResult(
        url=url, status_code=resp.status_code, html=html,
        fetched_at=datetime.now(timezone.utc), snapshot_path=path, from_snapshot=False,
        robots=robots_result,
    )
