"""
The toolkit's own product schemas. RawProduct is per-extraction-stage
output (one per method that found something); CatalogItem is the
normalized, provenance-tagged result that gets signed and emitted.

CatalogItem.field_provenance uses the SAME mandates.Provenance type as
CartLine.field_provenance (buyer/policy_engine's G1.1 gate reads exactly
this shape) -- so a catalog item's provenance carries straight through to
a CartLine at negotiation time without being re-derived or, worse,
silently dropped and re-invented with weaker guarantees.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mandates.schemas import Provenance


class RawProduct(BaseModel):
    sku: str | None = None
    title: str | None = None
    description: str | None = None
    category: str | None = None
    price_paise: int | None = None
    currency: str = "INR"
    in_stock: bool | None = None
    image_url: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    source_url: str
    extraction_method: str  # "schema_org" | "microdata" | "rdfa" | "llm_inferred"
    confidence: float | None = None  # only meaningful for llm_inferred


class CatalogItem(BaseModel):
    sku: str
    title: str
    description: str
    category: str
    price_paise: int
    currency: str = "INR"
    in_stock: bool
    image_url: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    field_provenance: dict[str, Provenance]


class Catalog(BaseModel):
    """The full signed catalog document -- gets wrapped in a
    mandates.Envelope and written to .well-known/agent-catalog.json.
    Reuses mandates.sign_mandate/verify_envelope directly since those
    operate on any pydantic BaseModel, not just the Intent/Cart/Payment
    mandate types."""

    version: str = "1.0"
    merchant_id: str
    generated_at: str  # ISO-8601; kept as str (not datetime) so this model
    # round-trips through JSON identically regardless of caller timezone
    # handling -- see emit.py, which is the only writer.
    items: list["CatalogItem"]
