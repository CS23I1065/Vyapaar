"""
Shared fixtures for demo/run_demo.py -- keys, catalogs, mandate builders.
Split from run_demo.py so the beats read as narration, not plumbing.

Everything here is deterministic and offline by default: no live network,
no live LLM call, no dependency on a recorded Razorpay cassette. Where
the demo needs a Razorpay response, it uses an httpx.MockTransport
double -- the same technique tests/test_cancel.py and tests/test_recon.py
already use, not a new pattern invented for the demo.

The three REAL, live-verified merchant catalogs (out/chumbak,
out/fabindia, out/bigbasket) are used for Beat 1 (onboarding) exactly as
they were produced. Beat 2's multi-item catalog is clearly-labelled DEMO
data, not a re-fetch of a real site: the accessory pitch is a mechanism
demonstration, not a claim about real merchant data, until a real
multi-URL catalog exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from mandates.keys import KeyPair, generate_keypair
from mandates.schemas import Provenance
from merchant_kit.schemas import CatalogItem
from razorpay_client.client import RazorpayClient

UTC = timezone.utc


def now() -> datetime:
    return datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC)


@dataclass(frozen=True)
class DemoKeys:
    human: KeyPair
    chumbak: KeyPair
    fabindia: KeyPair
    bigbasket: KeyPair


def demo_keys() -> DemoKeys:
    return DemoKeys(
        human=generate_keypair(), chumbak=generate_keypair(),
        fabindia=generate_keypair(), bigbasket=generate_keypair(),
    )


def _prov(source_url: str, t: datetime) -> Provenance:
    return Provenance(source_url=source_url, extraction_method="schema_org", signature_verified=False, extracted_at=t)


def _item(sku, title, category, price_paise, source_url, t, **attrs) -> CatalogItem:
    prov = _prov(source_url, t)
    return CatalogItem(
        sku=sku, title=title, description="", category=category, price_paise=price_paise, in_stock=True,
        attributes=attrs, field_provenance={f: prov for f in ("title", "description", "category", "price_paise", "in_stock")},
    )


def demo_catalog_shirts(t: datetime) -> list[CatalogItem]:
    """DEMO catalog data -- not scraped, clearly synthetic. Three items
    across two categories so the upsell engine has something real to
    find: a plain shirt (primary match), a cheaper belt in a DIFFERENT
    category (find_accessory), and a pricier shirt in the SAME category
    (find_premium_substitute). material/pattern attributes are set to
    match what a live interpret() call consistently extracts from "a
    plain cotton shirt" -- without them, required_attributes filtering
    (hard.required_attributes, correctly enforced) matches nothing.

    Category is "shirt", not the broader "apparel": asked to interpret
    "buy me a plain cotton shirt," Gemini consistently extracts
    category="shirt" across repeated calls, the more specific noun in
    the request rather than a parent bucket. A real storefront's own
    taxonomy would just as plausibly use "shirts" as a sub-category, so
    this is the catalog matching realistic categorization, not the demo
    being bent to fit a model quirk."""
    return [
        _item("SHIRT-PLAIN", "Plain Cotton Shirt", "shirt", 90000, "demo://shirts/plain", t,
              brand="Fabindia", material="cotton", pattern="plain"),
        _item("BELT-1", "Leather Belt", "accessories", 45000, "demo://shirts/belt", t, brand="Fabindia"),
        _item("SHIRT-PREMIUM", "Premium Linen Shirt", "shirt", 180000, "demo://shirts/premium", t,
              brand="Fabindia", material="linen", pattern="plain"),
    ]


def demo_catalog_rice_shop_a(t: datetime) -> list[CatalogItem]:
    return [_item("RICE-A", "Basmati Rice 2kg", "grains", 35000, "demo://shop-a/rice", t, brand="Generic")]


def demo_catalog_rice_shop_b(t: datetime) -> list[CatalogItem]:
    return [_item("RICE-B", "Basmati Rice 2kg", "grains", 32000, "demo://shop-b/rice", t, brand="IndiaGate")]


def demo_catalog_rice_shop_c(t: datetime) -> list[CatalogItem]:
    return [_item("RICE-C", "Basmati Rice 2kg", "grains", 28000, "demo://shop-c/rice", t, brand="Generic")]


def fake_razorpay_client() -> RazorpayClient:
    """A self-contained fake Razorpay backend -- not a recorded cassette
    (test-mode S2S payment creation needs per-merchant enablement, so
    none was ever captured), but the same MockTransport technique the
    test suite already relies on. create_order mints a fresh order id
    keyed off the receipt so repeated calls in one demo run don't
    collide; fetch_order_payments always reports the order's own amount
    as captured, which is enough to exercise recon.py's Tier 1 path and
    cancel.py's refund path end to end without a live key."""
    orders: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/orders" and request.method == "POST":
            import json as _json

            body = _json.loads(request.content)
            order_id = f"order_demo_{body['receipt'][:12]}"
            orders[order_id] = {
                "id": order_id, "amount": body["amount"], "status": "created",
                "notes": body.get("notes", {}), "receipt": body["receipt"],
            }
            return httpx.Response(200, json=orders[order_id])
        if path.startswith("/v1/orders/") and path.endswith("/payments"):
            order_id = path.split("/")[3]
            order = orders[order_id]
            return httpx.Response(200, json={"items": [{"id": f"pay_demo_{order_id[-8:]}", "amount": order["amount"], "status": "captured"}]})
        if path.startswith("/v1/orders/"):
            order_id = path.split("/")[3]
            return httpx.Response(200, json=orders[order_id])
        if path.endswith("/refund"):
            payment_id = path.split("/")[3]
            return httpx.Response(200, json={"id": f"rfnd_demo_{payment_id[-8:]}", "payment_id": payment_id, "status": "processed"})
        return httpx.Response(404, json={"error": {"description": f"no demo fixture for {request.method} {path}"}})

    return RazorpayClient(key_id="rzp_test_demo", key_secret="demo_secret", transport=httpx.MockTransport(handler))
