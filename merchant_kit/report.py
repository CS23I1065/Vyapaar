"""
The readiness report. Deliberately NOT a 0-100 score -- that's commodity
(Cloudflare's Agent Readiness score, AgentReady, AgentGrade all do this
already). Instead it shows exactly what was parsed vs inferred vs
rejected vs derived, which is the thing the writeup needs to be honest
about: where AI did necessary work (messy-site extraction) vs where it
was just parsing (schema.org sites).
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import CatalogItem


@dataclass(frozen=True)
class ReadinessReport:
    source_url: str
    robots_allowed: bool
    extraction_method_used: str  # "schema_org" | "llm_inferred" | "none"
    items: tuple[CatalogItem, ...]
    rejected: tuple[dict, ...]
    derived_skus: tuple[dict, ...]
    per_url: tuple[dict, ...] = ()
    # One row per product URL in a multi-URL run: which page yielded how
    # many items, by which extraction method. A single "3 items" line
    # cannot show that two came from clean JSON-LD and one from the LLM
    # fallback, which is exactly the distinction the writeup turns on.
    robots_reason: str | None = None
    # "parsed" | "absent_4xx" | "unreachable_5xx" | "fetch_failed" | None
    # (None = --offline replay, where no live check ran). Carried through
    # from the REAL RobotsCheck rather than hardcoded: "allowed" because
    # we read the shop's rules and they permit this path is a different
    # claim from "allowed" because the shop publishes no rules at all,
    # and a readiness report that cannot tell those apart is exactly the
    # kind of thing this project criticises elsewhere.

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def fields_by_extraction_method(self) -> dict[str, int]:
        """Across all items, how many fields came from each extraction
        method."""
        counts: dict[str, int] = {}
        for item in self.items:
            for prov in item.field_provenance.values():
                counts[prov.extraction_method] = counts.get(prov.extraction_method, 0) + 1
        return counts

    @property
    def robots_line(self) -> str:
        verdict = "ALLOWED" if self.robots_allowed else "DISALLOWED"
        explanation = {
            "parsed": "rules published and read; this path is permitted",
            "absent_4xx": "no robots.txt published (4xx) -- permissive by RFC 9309",
            "unreachable_5xx": "robots.txt unreachable (5xx) -- full disallow per RFC 9309",
            "fetch_failed": "robots.txt could not be fetched -- failing closed",
            None: "not checked (--offline snapshot replay, no live fetch)",
        }.get(self.robots_reason, self.robots_reason or "")
        return f"robots.txt: {verdict} ({explanation})"

    def summary_lines(self) -> list[str]:
        lines = [
            f"Source: {self.source_url}",
            self.robots_line,
            f"Extraction method used: {self.extraction_method_used}",
            f"Catalog items produced: {self.item_count}",
        ]
        breakdown = self.fields_by_extraction_method
        if breakdown:
            lines.append("Field provenance breakdown:")
            for method, count in sorted(breakdown.items()):
                lines.append(f"  {method}: {count} field(s)")
        if self.rejected:
            lines.append(f"Rejected ({len(self.rejected)}) -- found but not usable:")
            for r in self.rejected:
                lines.append(f"  - {r.get('title')!r}: missing {r.get('missing_fields')}")
        if len(self.per_url) > 1:
            lines.append("Per-URL breakdown:")
            for row in self.per_url:
                lines.append(f"  {row['items']} item(s) via {row['extraction_method']}: {row['url']}")
        if self.derived_skus:
            lines.append(f"Synthetic SKUs derived from URL (no merchant SKU visible, {len(self.derived_skus)}):")
            for d in self.derived_skus:
                lines.append(f"  - {d['derived_sku']} for {d.get('title')!r}")
        return lines

    def print_report(self) -> None:
        for line in self.summary_lines():
            print(line)
