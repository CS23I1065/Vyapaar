"""
Tests for merchant_kit/. Where possible these run against real snapshots
captured from three genuinely different live sites (Chumbak, Fabindia,
BigBasket) rather than synthetic fixtures. Snapshots are local files, so
these tests are deterministic and offline by construction (fetch.py's
own --offline mode is what reads them), not because the sites happened
to be reachable when the suite ran.
"""

from __future__ import annotations

import json

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mandates import Envelope, KeyDirectory, generate_keypair, verify_envelope
from mandates.schemas import MerchantPolicy, Provenance
from merchant_kit.emit import emit
from merchant_kit.extract.structured import extract_structured
from merchant_kit.fetch import FetchError, fetch
from merchant_kit.normalize import normalize
from merchant_kit.robots import check as robots_check
from merchant_kit.schemas import Catalog, CatalogItem, RawProduct

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS_DIR = REPO_ROOT / "merchant_kit" / "snapshots"

CHUMBAK_URL = "https://www.chumbak.com/products/ravana-magnet"
FABINDIA_URL = "https://www.fabindia.com/red-cotton-mid-placket-shirt-10546011"
BIGBASKET_URL = "https://www.bigbasket.com/pd/265885/del-monte-four-seasons-mixed-fruit-drink-100-240-ml-tin/"

_UTC = timezone.utc


def _now() -> datetime:
    return datetime(2026, 8, 24, 12, 0, 0, tzinfo=_UTC)


def _snapshots_present() -> bool:
    return SNAPSHOTS_DIR.exists() and any(SNAPSHOTS_DIR.rglob("*.html"))


requires_snapshots = pytest.mark.skipif(
    not _snapshots_present(), reason="real site snapshots not present -- run merchant_kit.cli live first"
)


# ---------------------------------------------------------------------
# robots.py -- response-class handling, no network
#
# These monkeypatch httpx.Client (not httpx.get): the module follows
# redirects with a capped hop count, which is a Client-level setting.
# Following redirects matters because httpx defaults to
# follow_redirects=False, so a shop serving robots.txt via an apex->www
# redirect would otherwise read as "no rules published, crawl freely."
# ---------------------------------------------------------------------


def _fake_client(monkeypatch, *, status_code=200, body="", raises=None):
    """Installs a fake httpx.Client whose .get() returns one canned
    response (or raises). Records the kwargs the module passed in so the
    redirect-following contract itself can be asserted."""
    import httpx

    recorded: dict = {}

    class _FakeResponse:
        def __init__(self):
            self.status_code = status_code
            self.content = body.encode("utf-8")
            self.encoding = "utf-8"

    class _FakeClient:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, **kwargs):
            if raises is not None:
                raise raises
            return _FakeResponse()

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    return recorded


def test_robots_disallow_blocks_matching_path(monkeypatch):
    _fake_client(monkeypatch, body="User-agent: *\nDisallow: /cart\nDisallow: /checkout\n")
    result = robots_check("https://example.com/cart")
    assert not result.allowed
    assert result.reason == "parsed"


def test_robots_allows_unlisted_path(monkeypatch):
    _fake_client(monkeypatch, body="User-agent: *\nDisallow: /cart\n")
    result = robots_check("https://example.com/products/some-item")
    assert result.allowed
    assert result.reason == "parsed"
    assert result.rules_were_read


def test_robots_404_is_permissive(monkeypatch):
    """RFC 9309 s2.3.1.3 -- a genuinely absent robots.txt means the
    crawler may access anything. The one permissive default that is
    correct, kept deliberately."""
    _fake_client(monkeypatch, status_code=404)
    result = robots_check("https://example.com/anything")
    assert result.allowed
    assert result.reason == "absent_4xx"
    assert result.raw_text is None
    # ...but it must not be mistaken for "we read their rules and they
    # said yes" -- that distinction is what the readiness report prints.
    assert not result.rules_were_read


def test_robots_5xx_fails_closed(monkeypatch):
    """RFC 9309 s2.3.1.4 asks for a full disallow on an unreachable
    robots.txt. This used to be read as "no rules -> allow": a shop
    having a bad five minutes registered as blanket consent."""
    _fake_client(monkeypatch, status_code=503)
    result = robots_check("https://example.com/anything")
    assert not result.allowed
    assert result.reason == "unreachable_5xx"


def test_robots_network_failure_fails_closed(monkeypatch):
    """Not RFC-specified, but an unreachable server must not silently
    become an unrestricted one -- otherwise network flakiness widens what
    we are allowed to fetch."""
    import httpx

    _fake_client(monkeypatch, raises=httpx.ConnectError("no such host"))
    result = robots_check("https://example.com/anything")
    assert not result.allowed
    assert result.reason == "fetch_failed"
    assert result.raw_text is None


def test_robots_timeout_fails_closed(monkeypatch):
    import httpx

    _fake_client(monkeypatch, raises=httpx.ReadTimeout("too slow"))
    result = robots_check("https://example.com/anything")
    assert not result.allowed
    assert result.reason == "fetch_failed"


def test_robots_follows_redirects_with_a_capped_hop_count(monkeypatch):
    """A real site's robots.txt can sit behind apex -> https -> www
    redirects; without follow_redirects the fetch would return 301,
    raw_text would stay None, and an empty ruleset would read as
    allow-all."""
    from merchant_kit.robots import MAX_ROBOTS_REDIRECTS

    recorded = _fake_client(monkeypatch, body="User-agent: *\nDisallow: /cart\n")
    robots_check("https://example.com/cart")
    assert recorded["follow_redirects"] is True
    assert recorded["max_redirects"] == MAX_ROBOTS_REDIRECTS


def test_robots_too_many_redirects_fails_closed(monkeypatch):
    import httpx

    _fake_client(monkeypatch, raises=httpx.TooManyRedirects("redirect loop"))
    result = robots_check("https://example.com/anything")
    assert not result.allowed
    assert result.reason == "fetch_failed"


def test_robots_body_is_capped(monkeypatch):
    """RFC 9309 asks crawlers to parse at least 500 KiB. resp.text was
    previously unbounded, while page fetches were already capped at 2 MB."""
    from merchant_kit.robots import MAX_ROBOTS_BYTES

    huge = "User-agent: *\nDisallow: /cart\n" + ("#" + "x" * 99 + "\n") * 20_000
    assert len(huge.encode()) > MAX_ROBOTS_BYTES
    _fake_client(monkeypatch, body=huge)
    result = robots_check("https://example.com/products/x")
    assert len(result.raw_text.encode("utf-8")) <= MAX_ROBOTS_BYTES
    assert result.allowed  # the Disallow: /cart at the top still parsed


def test_robots_crawl_delay_is_surfaced(monkeypatch):
    _fake_client(monkeypatch, body="User-agent: *\nCrawl-delay: 10\n")
    result = robots_check("https://example.com/x")
    assert result.crawl_delay == 10.0


def test_enforce_raises_with_the_reason_attached(monkeypatch):
    from merchant_kit.robots import RobotsDisallowed, enforce

    _fake_client(monkeypatch, status_code=503)
    with pytest.raises(RobotsDisallowed) as exc:
        enforce("https://example.com/x")
    assert exc.value.reason == "unreachable_5xx"


# ---------------------------------------------------------------------
# fetch.py -- offline replay from real snapshots
# ---------------------------------------------------------------------


@requires_snapshots
def test_fetch_offline_replays_real_snapshot_with_no_network():
    result = fetch(CHUMBAK_URL, snapshots_dir=SNAPSHOTS_DIR, offline=True)
    assert result.from_snapshot
    assert "Ravana" in result.html or "ravana" in result.html.lower()


def test_fetch_offline_missing_snapshot_raises():
    with pytest.raises(FetchError, match="no snapshot exists"):
        fetch("https://nonexistent.example/never-fetched", snapshots_dir=SNAPSHOTS_DIR, offline=True)


# ---------------------------------------------------------------------
# extract/structured.py -- against real, genuinely different markup
# ---------------------------------------------------------------------


@requires_snapshots
def test_structured_extraction_chumbak_rich_jsonld():
    html = fetch(CHUMBAK_URL, snapshots_dir=SNAPSHOTS_DIR, offline=True).html
    products = extract_structured(html, CHUMBAK_URL)
    real_products = [p for p in products if p.price_paise is not None]
    assert len(real_products) == 1
    p = real_products[0]
    assert p.extraction_method == "schema_org"
    assert p.price_paise == 39_500
    assert p.sku  # a real GTIN was present in this page's markup
    assert p.attributes.get("brand")


def test_structured_extraction_chumbak_spurious_block_is_present_and_incomplete():
    """Documents real-world messiness found during research: Chumbak's
    page emits a SECOND @type=Product JSON-LD block (name="Chumbak", no
    price) -- likely a theme/brand-identity block mistagged as Product.
    Structured extraction correctly surfaces it as a RawProduct (it IS
    schema.org-typed Product markup); normalize() is what rejects it for
    missing price -- see test_normalize below."""
    html = fetch(CHUMBAK_URL, snapshots_dir=SNAPSHOTS_DIR, offline=True).html
    products = extract_structured(html, CHUMBAK_URL)
    incomplete = [p for p in products if p.price_paise is None]
    assert len(incomplete) == 1
    assert any(p.price_paise is None for p in products)


@requires_snapshots
def test_structured_extraction_fabindia_minimal_jsonld():
    html = fetch(FABINDIA_URL, snapshots_dir=SNAPSHOTS_DIR, offline=True).html
    products = extract_structured(html, FABINDIA_URL)
    assert len(products) == 1
    p = products[0]
    assert p.extraction_method == "schema_org"
    assert p.price_paise == 169_000
    assert p.sku == "10546011"
    # Fabindia's markup is deliberately the "inconsistent/minimal" case --
    # no brand object present, unlike Chumbak's richer feed.
    assert "brand" not in p.attributes


@requires_snapshots
def test_structured_extraction_bigbasket_yields_nothing():
    """BigBasket is a client-hydrated Next.js SPA with zero JSON-LD in
    static HTML -- this is the real case the LLM fallback path exists
    for, verified against live data rather than assumed."""
    html = fetch(BIGBASKET_URL, snapshots_dir=SNAPSHOTS_DIR, offline=True).html
    products = extract_structured(html, BIGBASKET_URL)
    assert products == []


# ---------------------------------------------------------------------
# normalize.py
# ---------------------------------------------------------------------


def _raw(**overrides) -> RawProduct:
    defaults = dict(
        sku="SKU-1", title="Item", description="desc", category="cat",
        price_paise=1000, currency="INR", in_stock=True, image_url=None,
        attributes={}, source_url="https://merchant.example/p", extraction_method="schema_org",
    )
    defaults.update(overrides)
    return RawProduct(**defaults)


def test_normalize_promotes_complete_product():
    items, rejected, derived = normalize([_raw()], now=_now())
    assert len(items) == 1
    assert not rejected
    assert not derived
    assert items[0].field_provenance["price_paise"].signature_verified is False


def test_normalize_rejects_missing_title():
    items, rejected, derived = normalize([_raw(title=None)], now=_now())
    assert not items
    assert len(rejected) == 1
    assert "title" in rejected[0]["missing_fields"]


def test_normalize_rejects_missing_price():
    items, rejected, derived = normalize([_raw(price_paise=None)], now=_now())
    assert not items
    assert "price_paise" in rejected[0]["missing_fields"]


def test_normalize_derives_sku_when_absent_but_records_it():
    items, rejected, derived = normalize([_raw(sku=None)], now=_now())
    assert len(items) == 1
    assert items[0].sku.startswith("url-derived-")
    assert len(derived) == 1
    assert derived[0]["derived_sku"] == items[0].sku


def test_normalize_unknown_stock_defaults_to_not_purchasable():
    items, _, _ = normalize([_raw(in_stock=None)], now=_now())
    assert items[0].in_stock is False  # fail closed, not "assume available"


@requires_snapshots
def test_normalize_rejects_chumbak_spurious_block_end_to_end():
    html = (SNAPSHOTS_DIR / "www.chumbak.com").glob("*.html")
    html_path = next(html)
    raw = extract_structured(html_path.read_text(), CHUMBAK_URL)
    items, rejected, _ = normalize(raw, now=_now())
    assert len(items) == 1
    assert len(rejected) == 1
    assert rejected[0]["title"] == "Chumbak"


# ---------------------------------------------------------------------
# sign.py / emit.py -- full round trip
# ---------------------------------------------------------------------


def test_emit_then_verify_catalog_round_trip(tmp_path):
    keypair = generate_keypair()
    catalog_item = normalize([_raw()], now=_now())[0][0]
    catalog = Catalog(merchant_id="merchant-1", generated_at=_now().isoformat(), items=[catalog_item])
    policy = MerchantPolicy(
        merchant_id="merchant-1", max_order_value_paise=100_000, substitution_tolerance_pct=10.0,
        max_quantity_per_sku=5, restricted_categories=[], escalation_threshold_paise=50_000,
    )

    paths = emit(catalog, policy, keypair, out_dir=tmp_path)
    assert paths["catalog"].exists()
    assert paths["policy"].exists()
    assert paths["keys"].exists()

    catalog_envelope = Envelope.model_validate_json(paths["catalog"].read_text())
    import json

    jwks = json.loads(paths["keys"].read_text())
    directory = KeyDirectory.from_jwks(jwks)

    result = verify_envelope(catalog_envelope, directory, required_kid=keypair.kid)
    assert result.valid, result.reason


def test_emit_tampered_catalog_fails_verification(tmp_path):
    keypair = generate_keypair()
    catalog_item = normalize([_raw()], now=_now())[0][0]
    catalog = Catalog(merchant_id="merchant-1", generated_at=_now().isoformat(), items=[catalog_item])
    policy = MerchantPolicy(
        merchant_id="merchant-1", max_order_value_paise=100_000, substitution_tolerance_pct=10.0,
        max_quantity_per_sku=5, restricted_categories=[], escalation_threshold_paise=50_000,
    )
    paths = emit(catalog, policy, keypair, out_dir=tmp_path)

    envelope = Envelope.model_validate_json(paths["catalog"].read_text())
    tampered = envelope.model_copy(deep=True)
    tampered.payload["items"][0]["price_paise"] = 1  # simulate a compromised/malicious server

    import json

    directory = KeyDirectory.from_jwks(json.loads(paths["keys"].read_text()))
    result = verify_envelope(tampered, directory)
    assert not result.valid


# ---------------------------------------------------------------------
# cli.py -- stable merchant identity, multi-URL catalogs, policy config
# ---------------------------------------------------------------------


def test_key_file_is_loaded_not_regenerated(tmp_path):
    """A merchant's key IS its identity. The CLI used to call
    `keypair or generate_keypair()` with main() never passing one, so
    every run minted a fresh keypair and a fresh kid -- a buyer that
    verified yesterday's catalog could not verify today's, which is most
    of what publishing a key is for."""
    from merchant_kit.cli import load_or_create_keypair

    key_file = tmp_path / "merchant-key.json"
    first, created_first = load_or_create_keypair(key_file)
    second, created_second = load_or_create_keypair(key_file)

    assert created_first is True and created_second is False
    assert first.kid == second.kid
    assert key_file.exists()


def test_key_file_is_written_private(tmp_path):
    from merchant_kit.cli import load_or_create_keypair

    key_file = tmp_path / "merchant-key.json"
    load_or_create_keypair(key_file)
    assert (key_file.stat().st_mode & 0o777) == 0o600


def test_policy_defaults_are_labelled_as_defaults(tmp_path):
    from merchant_kit.policy_config import resolve_policy

    policy, source = resolve_policy(None, "chumbak")
    assert policy.merchant_id == "chumbak"
    assert "default" in source.lower()
    assert "not merchant-supplied" in source


def test_policy_file_overrides_per_merchant(tmp_path):
    from merchant_kit.policy_config import resolve_policy

    path = tmp_path / "policy.toml"
    path.write_text(
        "[merchants.chumbak]\n"
        "max_order_value_paise = 250000\n"
        'restricted_categories = ["alcohol"]\n'
        "\n[merchants.fabindia]\n"
        "max_order_value_paise = 999999\n"
    )
    chumbak, source = resolve_policy(path, "chumbak")
    fabindia, _ = resolve_policy(path, "fabindia")

    assert chumbak.max_order_value_paise == 250_000
    assert chumbak.restricted_categories == ["alcohol"]
    assert fabindia.max_order_value_paise == 999_999
    assert str(path) in source
    # unspecified keys still fall back to a documented default
    assert chumbak.max_quantity_per_sku == 5


def test_policy_file_rejects_unknown_keys(tmp_path):
    """A typo'd key that silently fell back to a default would produce a
    SIGNED document that does not say what its author thought it said."""
    from merchant_kit.policy_config import resolve_policy

    path = tmp_path / "policy.toml"
    path.write_text("max_order_value_pasie = 250000\n")  # note the typo
    with pytest.raises(ValueError, match="unknown policy key"):
        resolve_policy(path, "chumbak")


def test_escalation_threshold_default_is_high_enough_not_to_nag():
    """G7.1 takes min(buyer, merchant), so a low merchant default would
    become the only threshold that can fire and would re-introduce the
    routine escalation the buyer side just removed."""
    from merchant_kit.policy_config import (
        DEFAULT_ESCALATION_THRESHOLD_PAISE,
        DEFAULT_MAX_ORDER_VALUE_PAISE,
    )

    assert DEFAULT_ESCALATION_THRESHOLD_PAISE >= DEFAULT_MAX_ORDER_VALUE_PAISE


def test_multi_url_run_produces_one_catalog_with_every_item(tmp_path, monkeypatch):
    """The upsell engine needs more than one item to work against real
    data: find_accessory needs a DIFFERENT category and
    find_premium_substitute a pricier item in the SAME one, so a
    one-item catalog made both return None and every pitch fell through
    to a quantity bump."""
    from merchant_kit import cli

    pages = {
        "https://shop.example/rice": ("RICE-1", "Basmati Rice", "grains", 30_000),
        "https://shop.example/pickle": ("PICKLE-1", "Mango Pickle", "condiments", 12_000),
        "https://shop.example/rice-premium": ("RICE-2", "Aged Basmati", "grains", 45_000),
    }

    def fake_extract_one(url, *, offline, snapshots_dir, now):
        sku, title, category, price = pages[url]
        item = CatalogItem(
            sku=sku, title=title, description="", category=category, price_paise=price,
            in_stock=True,
            field_provenance={
                "price_paise": Provenance(
                    source_url=url, extraction_method="schema_org",
                    signature_verified=False, extracted_at=now,
                )
            },
        )
        return [item], [], [], "schema_org", "parsed"

    monkeypatch.setattr(cli, "extract_one", fake_extract_one)
    report = cli.run(list(pages), out_dir=tmp_path, merchant_id="shop")

    assert report.item_count == 3
    assert {i.sku for i in report.items} == {"RICE-1", "PICKLE-1", "RICE-2"}
    assert len(report.per_url) == 3

    envelope = json.loads((tmp_path / ".well-known" / "agent-catalog.json").read_text())
    assert len(envelope["payload"]["items"]) == 3


def test_multi_url_catalog_actually_unblocks_the_upsell_finders(tmp_path, monkeypatch):
    """The point of the change, asserted against the real upsell engine
    rather than against the item count."""
    from merchant_agent.upsell import find_accessory, find_premium_substitute
    from merchant_kit import cli

    pages = {
        "https://shop.example/rice": ("RICE-1", "Basmati Rice", "grains", 30_000),
        "https://shop.example/pickle": ("PICKLE-1", "Mango Pickle", "condiments", 12_000),
        "https://shop.example/rice-premium": ("RICE-2", "Aged Basmati", "grains", 45_000),
    }

    def fake_extract_one(url, *, offline, snapshots_dir, now):
        sku, title, category, price = pages[url]
        return (
            [
                CatalogItem(
                    sku=sku, title=title, description="", category=category,
                    price_paise=price, in_stock=True,
                    field_provenance={
                        "price_paise": Provenance(
                            source_url=url, extraction_method="schema_org",
                            signature_verified=False, extracted_at=now,
                        )
                    },
                )
            ],
            [], [], "schema_org", "parsed",
        )

    monkeypatch.setattr(cli, "extract_one", fake_extract_one)
    items = list(cli.run(list(pages), out_dir=tmp_path, merchant_id="shop").items)
    primary = next(i for i in items if i.sku == "RICE-1")

    assert find_accessory(primary, items) is not None          # was always None
    assert find_premium_substitute(primary, items) is not None  # was always None


def test_duplicate_sku_across_urls_is_refused(tmp_path, monkeypatch):
    """A catalog publishing one SKU twice is ambiguous about which price
    a buyer should bind against -- and G8.1 binds against exactly that."""
    from merchant_kit import cli

    def fake_extract_one(url, *, offline, snapshots_dir, now):
        return (
            [
                CatalogItem(
                    sku="SAME-SKU", title="Thing", description="", category="c",
                    price_paise=100, in_stock=True,
                    field_provenance={
                        "price_paise": Provenance(
                            source_url=url, extraction_method="schema_org",
                            signature_verified=False, extracted_at=now,
                        )
                    },
                )
            ],
            [], [], "schema_org", "parsed",
        )

    monkeypatch.setattr(cli, "extract_one", fake_extract_one)
    with pytest.raises(SystemExit):
        cli.run(["https://a.example/x", "https://b.example/y"], out_dir=tmp_path, merchant_id="shop")


def test_robots_block_prints_a_report_instead_of_a_bare_traceback(tmp_path, monkeypatch, capsys):
    """report.py's DISALLOWED branch used to be unreachable dead code,
    because cli.py hardcoded robots_allowed=True and died on SystemExit
    before building a report at all."""
    from merchant_kit import cli
    from merchant_kit.robots import RobotsDisallowed

    def blocked(url, *, offline, snapshots_dir, now):
        raise RobotsDisallowed(url, "https://shop.example/robots.txt", "unreachable_5xx")

    monkeypatch.setattr(cli, "extract_one", blocked)
    with pytest.raises(SystemExit):
        cli.run(["https://shop.example/p"], out_dir=tmp_path, merchant_id="shop")

    out = capsys.readouterr().out
    assert "DISALLOWED" in out
    assert "5xx" in out
