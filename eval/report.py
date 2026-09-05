"""
Self-contained HTML ablation report. No external CSS/JS/CDN -- everything
inlined, so the file opens standalone.

Self-detecting: this module reads WHICH configs and attack ids actually
appear in the results it was given and labels the table with exactly
those -- it never renders a row for C1/C2_C3/C4 or an attack column that
produced zero instances, and it never implies a config ran that wasn't
in the data.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

from eval.corpus import Scenario, corpus_counts
from eval.harness import InstanceResult
from eval.metrics import ConfigMetrics, compute_all_metrics, compute_rupees_prevented_paise

DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "results" / "report.html"


def render_live_llm_section(live_result) -> str:
    """live_result: an eval.live_llm_eval.LiveEvaluationResult. Kept as a
    loosely-typed parameter (duck-typed on the fields used) rather than
    importing eval.live_llm_eval at module load time -- that module
    imports buyer.interpret, which is a heavier dependency chain than a
    report renderer should need just to draw a table."""
    fm, am = live_result.fidelity_metrics, live_result.ambiguity_metrics
    if live_result.e2e_results:
        e2e_metrics = compute_all_metrics(live_result.e2e_results)
        e2e_rows = "".join(
            f"<tr><td><code>{html.escape(m.config_id)}</code></td><td>{m.n_instances}</td>"
            f"<td>{_pct(m.benign_utility)}</td><td>{_pct(m.utility_under_attack)}</td>"
            f"<td>{_pct(m.false_approval_rate)}</td></tr>"
            for m in e2e_metrics
        )
    elif live_result.e2e_metrics_summary:
        e2e_rows = "".join(
            f"<tr><td><code>{html.escape(m['config_id'])}</code></td><td>{m['n_instances']}</td>"
            f"<td>{_pct(m['benign_utility'])}</td><td>{_pct(m['utility_under_attack'])}</td>"
            f"<td>{_pct(m['false_approval_rate'])}</td></tr>"
            for m in live_result.e2e_metrics_summary
        )
    else:
        e2e_rows = "<tr><td colspan='5'><em>no e2e instances -- see coverage above</em></td></tr>"

    return f"""
<div class="caveat" style="border-color:#059669">
<strong>This section used real API calls.</strong> Everything above this point in
the report is a deterministic, zero-cost replay against ground-truth-constructed
intents. Everything below used a live model.
</div>

<h3>Interpretation fidelity ({fm['n_calls']} live calls, {fm['n_scenarios']} scenarios)</h3>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Signable rate</td><td>{_pct(fm['signable_rate'])}</td></tr>
  <tr><td>Category accuracy*</td><td>{_pct(fm['category_accuracy'])}</td></tr>
  <tr><td>Budget accuracy</td><td>{_pct(fm['budget_accuracy'])}</td></tr>
  <tr><td>Required-attributes accuracy</td><td>{_pct(fm['attributes_accuracy'])}</td></tr>
  <tr><td>Fully correct (per call)*</td><td>{_pct(fm['fully_correct_rate'])}</td></tr>
  <tr><td><strong>pass^k</strong>*</td><td><strong>{_pct(fm['pass_at_k'])}</strong></td></tr>
</table>
<div class="caveat">
<strong>*Read category accuracy and pass^k together with the E2E table below, not alone.</strong>
Early on, a real vocabulary-matching gap between free-text interpretation and a fixed catalog
taxonomy (the model saying "basmati rice" where a catalog said "grains") depressed both of these
numbers even though the model was essentially never wrong about what the shopper wanted. That gap
is now closed with a controlled-vocabulary constraint on both category and required-attribute
keys -- see the E2E table below for what that unlocked, and the note under it for what it exposed
in the process.
</div>

<h3>Ambiguity stress test ({am['n_calls']} live calls, {am['n_scored']} scored)</h3>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Accuracy (asks when it should, doesn't when it shouldn't)</td><td>{_pct(am['accuracy'])}</td></tr>
</table>

<h3>End-to-end with real interpretation ({live_result.e2e_scenarios_covered}/{live_result.e2e_scenarios_total} scenarios covered)</h3>
<p>Same attack/config matrix as the deterministic table above, but driven by
what the live model actually produced for each scenario, not a synthetic
ground-truth intent.</p>
<table>
  <tr><th>Config</th><th>N</th><th>Benign utility</th><th>Utility under attack</th><th>False approval rate*</th></tr>
  {e2e_rows}
</table>
<div class="caveat">
<strong>*This E2E replay applies <code>apply_simulated_review()</code> before signing anything.</strong>
Earlier, this replay signed a live <code>interpret()</code> draft directly without routing it
through <code>buyer/review.py</code>'s human review step -- a nonzero false-approval rate showed up
from requests that explicitly needed that step (an explicit "check with me above X rupees" tripwire,
or a self-contradictory request the model correctly flagged with <code>needs_clarification</code>).
<code>evaluate()</code> itself never approved anything it shouldn't given the intent it was handed;
the gap was in the replay, not the policy engine. Fixed: a draft still flagged
<code>needs_clarification</code> is now excluded from this table entirely, the same treatment as a
draft that was never signable at all -- which is why scenario coverage below is lower than the
fidelity numbers above would suggest. The deterministic matrix at the top of this report never
showed this gap because its ground-truth intents are authored as if review had already happened.
</div>
"""


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _config_table(metrics: list[ConfigMetrics]) -> str:
    if not metrics:
        return "<p><em>No instances recorded -- the matrix has not been run.</em></p>"

    attack_ids = sorted({a for m in metrics for a in m.targeted_asr})
    rows = []
    for m in metrics:
        asr_cells = "".join(f"<td>{_pct(m.targeted_asr.get(a))}</td>" for a in attack_ids)
        rows.append(
            f"<tr><td><code>{html.escape(m.config_id)}</code></td>"
            f"<td>{m.n_instances}</td>"
            f"<td>{_pct(m.benign_utility)}</td>"
            f"<td>{_pct(m.false_approval_rate)}</td>"
            f"<td>{_pct(m.false_escalation_rate)}</td>"
            f"<td>{'n/a' if m.human_touchpoints_avg is None else f'{m.human_touchpoints_avg:.2f}'}</td>"
            f"<td>{_pct(m.utility_under_attack)}</td>"
            f"{asr_cells}"
            f"<td>{_pct(m.upsell_offer_rate)}</td>"
            f"<td>{_pct(m.upsell_would_pass_rate)}</td>"
            f"<td>{'n/a' if m.aov_lift_pct is None else f'{m.aov_lift_pct:+.1f}%'}</td>"
            f"</tr>"
        )
    attack_headers = "".join(f"<th>ASR<br><code>{html.escape(a)}</code></th>" for a in attack_ids)
    return f"""
<table>
  <thead><tr>
    <th>Config</th><th>N</th><th>Benign utility</th>
    <th>False approval rate</th><th>False escalation rate</th>
    <th>Human touchpoints (avg)</th><th>Utility under attack</th>
    {attack_headers}
    <th>Upsell offer rate</th><th>Upsell would-pass rate</th><th>AOV lift</th>
  </tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table>
"""


def _exceptions_section(metrics: list[ConfigMetrics]) -> str:
    """The mandatory honest exception list -- every benign-arm case the
    system still got wrong, verbatim, per config. An empty list is a
    genuine claim ("nothing to report"), not the absence of a section."""
    blocks = []
    for m in metrics:
        if not m.honest_exceptions:
            blocks.append(f"<h3><code>{html.escape(m.config_id)}</code></h3><p>No exceptions -- every benign-arm scenario resolved as labelled.</p>")
            continue
        rows = "".join(
            f"<tr><td>{html.escape(e['scenario_id'])}</td><td>{html.escape(e['category'])}</td>"
            f"<td>{html.escape(e['actual_outcome'])}</td><td>{html.escape(str(e['blocking_code']))}</td></tr>"
            for e in m.honest_exceptions
        )
        blocks.append(
            f"<h3><code>{html.escape(m.config_id)}</code> -- {len(m.honest_exceptions)} exception(s)</h3>"
            f"<table><thead><tr><th>Scenario</th><th>Category</th><th>Actual outcome</th><th>Blocking code</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    return "\n".join(blocks)


def _corpus_summary(scenarios: list[Scenario]) -> str:
    counts = corpus_counts(scenarios)
    items = "".join(f"<li>{html.escape(cat)}: {n}</li>" for cat, n in sorted(counts.items()))
    return f"<ul>{items}</ul><p>Total: {len(scenarios)} scenarios.</p>"


_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.85rem; }
th, td { border: 1px solid #8884; padding: 0.4rem 0.6rem; text-align: left; }
th { background: #8882; }
code { font-family: ui-monospace, monospace; }
.caveat { border-left: 3px solid #d97706; padding-left: 1rem; margin: 1rem 0; }
"""


def _rupees_prevented_section(results: list[InstanceResult]) -> str:
    configs_present = {r.config_id for r in results}
    if "C1" not in configs_present or "C4" not in configs_present:
        return "<p><em>Not computed -- requires both C1 and C4 to have run.</em></p>"
    prevented = compute_rupees_prevented_paise(results, baseline_config="C1", defended_config="C4")
    if prevented["n_matched_pairs"] == 0:
        return "<p>0 matched (scenario, attack, k) pairs where C1 over-spent and C4 blocked it.</p>"
    return (
        f"<p><strong>&#8377;{prevented['rupees_prevented']:,.2f}</strong> prevented across "
        f"<strong>{prevented['n_matched_pairs']}</strong> attack instance(s) where the undefended "
        f"baseline (C1) created an order over the buyer's own signed budget, and the defended "
        f"config (C4), run on the exact same scenario/attack/repeat, blocked it.</p>"
    )


def render_report(
    results: list[InstanceResult], scenarios: list[Scenario], *, generated_at: datetime | None = None,
    live_result=None,
) -> str:
    generated_at = generated_at or datetime.now(timezone.utc)
    metrics = compute_all_metrics(results)
    configs_present = sorted({r.config_id for r in results})

    body = f"""
<h1>Ablation report</h1>
<p>Generated {html.escape(generated_at.isoformat())}. {len(results)} instance(s) recorded across
configs: {", ".join(f"<code>{html.escape(c)}</code>" for c in configs_present) or "none"}.</p>

<div class="caveat">
<strong>Self-detected scope.</strong> This report shows exactly the configs and attack ids present
in the results it was given, and nothing else. If a config you expected is missing from the table
below, it was not run -- this report does not imply otherwise.
</div>

<div class="caveat">
<strong>C2 and C3 are reported together.</strong> They are not independently separable in this
codebase -- provenance verification and signed-mandate/chain verification are structurally fused
into one policy engine, by design. See <code>eval/config.py</code> for the full explanation.
</div>

<h2>Corpus</h2>
{_corpus_summary(scenarios)}

<h2>Results by config</h2>
{_config_table(metrics)}

<h2>&#8377; prevented (C1 &#8594; C4)</h2>
{_rupees_prevented_section(results)}

<div class="caveat">
<strong>pass^k is out of scope for this harness.</strong> pass^k measures whether
an LLM's output is stable across repeated samples of the same task. This
harness's scenario intents are built directly from authored ground truth rather
than a live interpret() call (see the corpus section above and eval/harness.py),
so there is no LLM stochasticity in this matrix for pass^k to measure -- every
repeat would be byte-identical. A meaningful pass^k requires a live, repeated
interpret() call per scenario, which has not been run.
</div>

<h2>Honest exception list</h2>
<p>Every benign-arm (A0_none) instance whose outcome did not match its authored ground truth.</p>
{_exceptions_section(metrics)}

<h2>Live AI evaluation</h2>
{render_live_llm_section(live_result) if live_result is not None else "<p><em>Not run. Add real numbers here with eval.live_llm_eval.run_full_live_evaluation() and pass live_result= to render_report().</em></p>"}
"""
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Ablation report</title><style>{_CSS}</style></head><body>{body}</body></html>"


def write_report(
    results: list[InstanceResult], scenarios: list[Scenario], *, path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(results, scenarios))
    return path
