"""
robots.txt enforcement -- checked and enforced BEFORE any product page is
fetched. Disallowed -> raise, log the reason, exit non-zero. This is not
advisory: the toolkit is a read-only crawler on any site that hasn't run
it, and a merchant's Disallow rules are the boundary of what "read-only"
means for us.

We deliberately do NOT try to parse a site's free-text "agent policy"
prose (e.g. a robots.txt comment banner saying "checkouts are for
humans") -- that's not a machine-readable directive, and heuristically
interpreting prose is exactly the kind of brittle, presumptuous parsing
this project argues against elsewhere (see mandates/ provenance gating).
Instead the raw robots.txt text is captured and surfaced in the
readiness report for a human to read, and only the real Disallow/Allow/
Crawl-delay directives are mechanically enforced.

Every outcome is classified, not collapsed
------------------------------------------
Treating only an exact 200 as real content and leaving `raw_text = None`
for everything else would parse to an empty ruleset, which makes
`can_fetch()` return True -- so *every* failure path would end in
"allowed, go fetch," including some that must not. Each response class
is instead handled on its own terms,
and `RobotsCheck.reason` records which one applied so the readiness
report can say WHY something was allowed rather than only that it was:

    2xx            -> parse the rules              reason="parsed"
    3xx            -> follow, up to 5 hops         (resolves to one of the others)
    4xx            -> no rules published, allow    reason="absent_4xx"
    5xx            -> DISALLOW (RFC 9309 asks for  reason="unreachable_5xx"
                      a full disallow on an
                      unreachable robots.txt)
    timeout/DNS/   -> DISALLOW (unspecified by     reason="fetch_failed"
    too many hops     the RFC, but an unreachable
                      server must not silently
                      become an unrestricted one)

The redirect case was a live bug, not a theoretical one: httpx defaults
`follow_redirects=False` (unlike requests, which follows on GET), while
fetch.py already passed `follow_redirects=True` for the *product page*.
So the toolkit would happily follow a redirect to fetch a page while
refusing to follow the same redirect to read the rules governing it --
crawling the destination host under the origin host's non-existent
permissions. Verified live against chumbak.com, which redirects apex ->
https -> www in two hops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib import robotparser
from urllib.parse import urlparse

import httpx

USER_AGENT = "AgentReadyKit/0.1 (+https://github.com/example/hope)"

# RFC 9309 s2.3.1.1: crawlers should follow at least five consecutive
# redirects, and are asked to parse at least 500 KiB of robots.txt.
MAX_ROBOTS_REDIRECTS = 5
MAX_ROBOTS_BYTES = 500 * 1024

RobotsReason = Literal["parsed", "absent_4xx", "unreachable_5xx", "fetch_failed"]


class RobotsDisallowed(Exception):
    def __init__(self, url: str, robots_url: str, reason: str = "parsed") -> None:
        self.url = url
        self.robots_url = robots_url
        self.reason = reason
        super().__init__(
            f"{url} is disallowed by {robots_url} for User-Agent "
            f"{USER_AGENT!r} (reason: {reason})"
        )


@dataclass(frozen=True)
class RobotsCheck:
    allowed: bool
    robots_url: str
    crawl_delay: float | None
    raw_text: str | None  # full robots.txt, kept for human/audit visibility only
    reason: RobotsReason
    status_code: int | None = None

    @property
    def rules_were_read(self) -> bool:
        """True only when a real robots.txt body was actually parsed. An
        `allowed` that is not `rules_were_read` means "no rules published,"
        which is a materially different claim -- the readiness report
        distinguishes the two rather than printing a bare ALLOWED."""
        return self.reason == "parsed"


def _disallowed(robots_url: str, reason: RobotsReason, status_code: int | None) -> RobotsCheck:
    return RobotsCheck(
        allowed=False, robots_url=robots_url, crawl_delay=None,
        raw_text=None, reason=reason, status_code=status_code,
    )


def check(url: str, *, timeout_s: float = 10.0) -> RobotsCheck:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    try:
        with httpx.Client(
            follow_redirects=True,  # the fix: 3xx used to read as "no rules"
            max_redirects=MAX_ROBOTS_REDIRECTS,
            timeout=timeout_s,
        ) as client:
            resp = client.get(robots_url, headers={"User-Agent": USER_AGENT})
    except httpx.HTTPError:
        # Timeout, DNS failure, connection reset, or more than
        # MAX_ROBOTS_REDIRECTS hops. Not RFC-specified, but failing open
        # here would mean network flakiness silently widens what we are
        # allowed to fetch. Fail closed.
        return _disallowed(robots_url, "fetch_failed", None)

    status = resp.status_code

    if 500 <= status < 600:
        # RFC 9309 s2.3.1.4: an unreachable robots.txt means the crawler
        # must assume complete disallow. A shop having a bad five minutes
        # is not blanket consent.
        return _disallowed(robots_url, "unreachable_5xx", status)

    if 400 <= status < 500:
        # RFC 9309 s2.3.1.3: no robots.txt published -> the crawler may
        # access anything. This is the one permissive default that IS
        # correct, and it stays.
        return RobotsCheck(
            allowed=True, robots_url=robots_url, crawl_delay=None,
            raw_text=None, reason="absent_4xx", status_code=status,
        )

    if not (200 <= status < 300):
        # 1xx, or a 3xx that somehow survived redirect following. Neither
        # is a ruleset we can read, so we do not pretend it is one.
        return _disallowed(robots_url, "fetch_failed", status)

    raw_text = resp.content[:MAX_ROBOTS_BYTES].decode(resp.encoding or "utf-8", errors="replace")

    rp = robotparser.RobotFileParser()
    rp.set_url(robots_url)
    rp.parse(raw_text.splitlines())

    return RobotsCheck(
        allowed=rp.can_fetch(USER_AGENT, url),
        robots_url=robots_url,
        crawl_delay=rp.crawl_delay(USER_AGENT),
        raw_text=raw_text,
        reason="parsed",
        status_code=status,
    )


def enforce(url: str, *, timeout_s: float = 10.0) -> RobotsCheck:
    result = check(url, timeout_s=timeout_s)
    if not result.allowed:
        raise RobotsDisallowed(url, result.robots_url, result.reason)
    return result
