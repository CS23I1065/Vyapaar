"""
Local static server for a merchant's .well-known/ directory (the output
of cli.py's `emit` step). This is what a buyer agent's HTTP client
actually fetches agent-catalog.json / agent-policy.json / agent-keys.json
from during the demo and eval harness.

Scope, stated plainly because an earlier version of this docstring
claimed otherwise: **only the static .well-known/ documents are served.**
There is no HTTP negotiation endpoint here or anywhere else in this repo.
merchant_agent/MerchantAgentDriver is a plain in-process callable
matching the buyer.negotiate.MerchantDriver protocol, so a negotiation is
a sequence of Python function calls inside one process -- nothing crosses
a network boundary.

That is a deliberate and defensible choice for the eval harness (an
in-process driver is exactly reproducible, and the adversarial corpus
needs to wrap the driver, not proxy HTTP), but it has one consequence
worth being precise about in the writeup: RFC 9421 HTTP Message
Signatures are a TRANSPORT concern, so with no HTTP negotiation path
there is no live wire on which agent identity is verified during a
purchase. identity/ is real, tested, and cross-verified against an
independent implementation -- but in this build it guards the fetch of
.well-known/ documents, not a live negotiation. Do not describe this loop
as an over-the-network merchant negotiation.
"""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles


def build_app(out_dir: Path) -> Starlette:
    well_known = out_dir / ".well-known"
    if not well_known.exists():
        raise FileNotFoundError(
            f"{well_known} does not exist -- run `python -m merchant_kit.cli --url ... "
            f"--out-dir {out_dir}` first to generate it."
        )
    return Starlette(routes=[Mount("/.well-known", app=StaticFiles(directory=str(well_known)))])


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Serve a merchant's .well-known/ directory")
    parser.add_argument("--out-dir", type=Path, default=Path("./out"))
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    app = build_app(args.out_dir)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
