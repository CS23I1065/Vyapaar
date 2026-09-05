"""python -m merchant_kit.cli --url URL [--url URL ...] [--urls-file F]
                              [--offline] [--out-dir DIR] [--merchant-id ID]
                              [--key-file PATH] [--policy PATH]

Runs the full toolkit pipeline for every requested product URL: robots
check -> fetch (+snapshot) -> structured extraction -> LLM fallback (only
if structured found nothing) -> normalize, then signs ONE catalog
containing everything found and emits the .well-known/ trio plus a
readiness report.

Three things this CLI used to get wrong, all of which produced signed
documents that misrepresented something:

1. **One item per catalog.** It took a single --url and emitted exactly
   one CatalogItem, so every catalog built from a real site had items: 1.
   That silently disabled two thirds of the upsell engine --
   find_accessory needs an item in a DIFFERENT category and
   find_premium_substitute needs a pricier one in the SAME category, so
   both returned None against real data and every pitch fell through to
   "want two instead of one?". Accepting multiple URLs is what makes the
   accessory and premium-substitute paths real rather than
   fixture-only.

2. **A fresh identity every run.** `keypair or generate_keypair()` with
   main() never passing one meant each invocation minted a new keypair,
   a new kid, and a new agent-keys.json -- so a buyer that verified
   yesterday's catalog could not verify today's, and the merchant had no
   stable cryptographic identity at all, which is most of what
   publishing a key is FOR. KeyPair.save()/load_keypair() already
   existed and were simply never called. --key-file now loads if present
   and generates-and-saves if not.

3. **Identical invented policy for every merchant.** See
   policy_config.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from mandates.keys import KeyPair, generate_keypair, load_keypair

from .emit import emit
from .extract.structured import extract_structured
from .fetch import FetchError, fetch
from .normalize import normalize
from .policy_config import resolve_policy
from .report import ReadinessReport
from .robots import RobotsDisallowed
from .schemas import Catalog

DEFAULT_SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"


def load_or_create_keypair(key_file: Path) -> tuple[KeyPair, bool]:
    """Returns (keypair, created). A merchant's key is its identity: it
    must survive across runs, or every catalog it ever publishes is
    signed by a stranger."""
    if key_file.exists():
        return load_keypair(key_file), False
    keypair = generate_keypair()
    keypair.save(key_file)
    return keypair, True


def extract_one(
    url: str,
    *,
    offline: bool,
    snapshots_dir: Path,
    now: datetime,
) -> tuple[list, list, list, str, str | None]:
    """Runs the read-only pipeline for a single product URL.

    Returns (items, rejected, derived, method_used, robots_reason).
    Raises RobotsDisallowed / FetchError for the caller to report on."""
    fetch_result = fetch(url, snapshots_dir=snapshots_dir, offline=offline)

    raw = extract_structured(fetch_result.html, url)
    method_used = "schema_org" if raw else "none"

    if not raw:
        # Local import: avoids requiring GEMINI_API_KEY / constructing a
        # GeminiProvider for callers that only exercise the structured
        # path (e.g. tests against a clean schema.org snapshot).
        from llm.provider import GeminiProvider

        from .extract.llm import extract_llm

        model = os.environ.get("LLM_MODEL_EVAL", "gemini-3.1-flash-lite")
        provider = GeminiProvider()
        try:
            raw = extract_llm(fetch_result.html, url, provider=provider, model=model)
        finally:
            provider.close()
        method_used = "llm_inferred" if raw else "none"

    items, rejected, derived = normalize(raw, now=now)
    robots_reason = fetch_result.robots.reason if fetch_result.robots else None
    return items, rejected, derived, method_used, robots_reason


def run(
    urls: list[str] | str,
    *,
    offline: bool = False,
    out_dir: Path,
    merchant_id: str,
    keypair: KeyPair | None = None,
    key_file: Path | None = None,
    policy_file: Path | None = None,
    snapshots_dir: Path = DEFAULT_SNAPSHOTS_DIR,
) -> ReadinessReport:
    if isinstance(urls, str):  # single-URL callers (and older tests) still work
        urls = [urls]
    if not urls:
        raise ValueError("run() needs at least one product URL")

    now = datetime.now(timezone.utc)
    all_items: list = []
    all_rejected: list = []
    all_derived: list = []
    methods: list[str] = []
    robots_reasons: list[str] = []
    per_url: list[dict] = []

    for url in urls:
        try:
            items, rejected, derived, method, robots_reason = extract_one(
                url, offline=offline, snapshots_dir=snapshots_dir, now=now
            )
        except RobotsDisallowed as e:
            # Report the block rather than dying with a bare traceback --
            # a refusal nobody can read looks like a broken tool, not a
            # working one. This is also what makes report.py's DISALLOWED
            # branch genuinely reachable instead of dead code.
            print(f"BLOCKED by robots.txt: {e}", file=sys.stderr)
            blocked = ReadinessReport(
                source_url=url, robots_allowed=False, extraction_method_used="none",
                items=(), rejected=(), derived_skus=(), robots_reason=e.reason,
            )
            blocked.print_report()
            raise SystemExit(1) from e
        except FetchError as e:
            print(f"Fetch error for {url}: {e}", file=sys.stderr)
            raise SystemExit(1) from e

        # Two product pages on the same site can legitimately yield the
        # same derived SKU only if they are the same URL; a real
        # collision would make the catalog ambiguous about which item a
        # published price belongs to, so it is an error, not a merge.
        for item in items:
            if any(existing.sku == item.sku for existing in all_items):
                print(
                    f"Duplicate SKU {item.sku!r} from {url} -- already present in this "
                    f"catalog. Refusing to publish an ambiguous price.",
                    file=sys.stderr,
                )
                raise SystemExit(1)

        all_items.extend(items)
        all_rejected.extend(rejected)
        all_derived.extend(derived)
        methods.append(method)
        if robots_reason:
            robots_reasons.append(robots_reason)
        per_url.append({"url": url, "items": len(items), "extraction_method": method})

    method_used = methods[0] if len(set(methods)) == 1 else "mixed"
    report = ReadinessReport(
        source_url=urls[0] if len(urls) == 1 else f"{len(urls)} URLs",
        robots_allowed=True,  # any block would have raised above
        extraction_method_used=method_used,
        items=tuple(all_items),
        rejected=tuple(all_rejected),
        derived_skus=tuple(all_derived),
        robots_reason=robots_reasons[0] if len(set(robots_reasons)) == 1 and robots_reasons else None,
        per_url=tuple(per_url),
    )

    if all_items:
        if keypair is None:
            if key_file is None:
                key_file = out_dir / "merchant-key.json"
            keypair, created = load_or_create_keypair(key_file)
            print(f"{'Generated and saved' if created else 'Loaded'} merchant key: {key_file}")
        catalog = Catalog(merchant_id=merchant_id, generated_at=now.isoformat(), items=all_items)
        policy, policy_source = resolve_policy(policy_file, merchant_id)
        paths = emit(catalog, policy, keypair, out_dir=out_dir)
        print(f"Wrote signed catalog to {paths['catalog']} ({len(all_items)} item(s))")
        print(f"Wrote signed policy to {paths['policy']} (source: {policy_source})")
        print(f"Wrote key directory to {paths['keys']}")
        print(f"Merchant key id (kid): {keypair.kid}")

    return report


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Merchant AI-Readiness Toolkit")
    parser.add_argument("--url", action="append", default=[], help="product page URL (repeatable)")
    parser.add_argument("--urls-file", type=Path, help="file with one product URL per line (# comments ok)")
    parser.add_argument("--offline", action="store_true", help="replay from snapshot, no network")
    parser.add_argument("--out-dir", type=Path, default=Path("./out"))
    parser.add_argument("--merchant-id", default="demo-merchant")
    parser.add_argument(
        "--key-file", type=Path,
        help="merchant signing key; loaded if it exists, generated and saved if not. "
             "Defaults to <out-dir>/merchant-key.json so a merchant's kid is stable across runs.",
    )
    parser.add_argument("--policy", type=Path, help="TOML file of merchant policy values")
    args = parser.parse_args()

    urls = list(args.url)
    if args.urls_file:
        for raw_line in args.urls_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if line:
                urls.append(line)
    if not urls:
        parser.error("at least one --url or a --urls-file is required")

    report = run(
        urls, offline=args.offline, out_dir=args.out_dir, merchant_id=args.merchant_id,
        key_file=args.key_file, policy_file=args.policy,
    )
    print()
    report.print_report()


if __name__ == "__main__":
    main()
