"""
Ablation configs for the eval matrix: C1 undefended (LLM decides) -> C2
+policy engine -> C3 +provenance & signed mandates -> C4 +verified agent
identity.

Honest simplification, stated plainly rather than faked: **C2 and C3 are
not independently separable in this codebase.** "A policy engine" and
"provenance/signed-mandate verification" can't be turned on as two
separate steps, because buyer/policy_engine.py doesn't support that --
G0.2 (signature), G0.3 (chain), and G1.1 (provenance) are not optional
gates layered onto a simpler engine, they are structurally fused into
the one deterministic evaluate() this whole project's safety claim rests
on. Building a second, weaker evaluate() just to have something to call
"C2" would mean shipping and maintaining a second policy engine nobody
would actually use, purely to fill an ablation cell. So C2 and C3 are
reported TOGETHER as one config, and the report self-detects which
configs actually ran rather than implying one that wasn't built.

C1 (undefended) and C4 (+ agent identity) both ARE real, distinct code
paths already built for other reasons: C1 models "no policy engine at
all" -- the exact vulnerability this project exists to defend against --
and C4's toggle (`require_agent_identity`) is not new; it is the SAME
VerificationContext field G0.5 has read since buyer/policy_engine.py was
first written (see its own comment: "ablation config C4 toggle").

One more honest caveat, inherited from merchant_kit/serve.py's own
docstring: this build's negotiation is in-process, not over HTTP, so
there is no live wire on which Web Bot Auth verification can genuinely
fail within this harness. C4's `webbotauth_verified` is therefore always
True in this corpus -- it models "agent identity verification is
available and passes," not a spoofing scenario, because there is no
spoofable transport here to model against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ConfigId = Literal["C1", "C2_C3", "C4"]


@dataclass(frozen=True)
class EvalConfig:
    id: ConfigId
    label: str
    use_policy_engine: bool
    require_agent_identity: bool


CONFIGS: tuple[EvalConfig, ...] = (
    EvalConfig("C1", "C1 -- undefended (no policy engine)", use_policy_engine=False, require_agent_identity=False),
    EvalConfig(
        "C2_C3", "C2+C3 -- policy engine incl. provenance & signed mandates (not independently separable, see module docstring)",
        use_policy_engine=True, require_agent_identity=False,
    ),
    EvalConfig("C4", "C4 -- + verified agent identity", use_policy_engine=True, require_agent_identity=True),
)

CONFIGS_BY_ID: dict[ConfigId, EvalConfig] = {c.id: c for c in CONFIGS}
