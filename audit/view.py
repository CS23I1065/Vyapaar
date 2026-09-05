"""
Renders an audit log as a readable rich timeline -- what gets shown on
screen during the demo video. Gates render as a table with green
PASS / red FAIL and evidence numbers inline; mandates as a small tree.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.tree import Tree


def _load_records(path: Path) -> list[dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def render_timeline(path: Path, *, console: Console | None = None) -> None:
    console = console or Console()
    if not path.exists():
        console.print("[yellow]No audit log found.[/yellow]")
        return

    for record in _load_records(path):
        kind = record["kind"]
        at = record["at"]
        payload = record["payload"]

        if kind == "decision":
            table = Table(title=f"[{at}] Decision: {payload['outcome']}")
            table.add_column("Gate")
            table.add_column("Code")
            table.add_column("Result")
            table.add_column("Detail")
            for check in payload["checks"]:
                result = "[green]PASS[/green]" if check["passed"] else "[red]FAIL[/red]"
                table.add_row(check["gate"], check["code"], result, check["detail"])
            console.print(table)

        elif kind == "mandate":
            tree = Tree(f"[{at}] Mandate: {payload['mandate_kind']} ({payload['mandate_id']})")
            tree.add(f"hash: {payload['hash'][:16]}...")
            console.print(tree)

        elif kind == "signature_verification":
            status = "[green]VALID[/green]" if payload["valid"] else "[red]INVALID[/red]"
            console.print(f"[{at}] Signature check ({payload['subject']}): {status} kid={payload.get('kid')}")

        elif kind == "llm_call":
            console.print(
                f"[{at}] LLM call: {payload['call_site']} model={payload['model']} "
                f"tokens_in={payload['input_tokens']} tokens_out={payload['output_tokens']} "
                f"cost=${payload['cost_usd']:.4f}"
            )

        elif kind == "tool_call":
            console.print(f"[{at}] Tool call: {payload['tool']}")

        elif kind == "spend_guard":
            console.print(
                f"[{at}] Spend guard: {payload['action']} "
                f"(${payload['projected_usd']:.2f} / ${payload['budget_usd']:.2f}) -- {payload['reason']}"
            )

        else:
            console.print(f"[{at}] {kind}: {payload}")
