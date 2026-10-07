"""``report/report.md`` and the run-root ``README.md`` (spec §32A.4, §32A.7).

One H1 and numbered H2 sections in fixed order; CommonMark + GFM tables; no raw
HTML. Every number shown is a presentation rounding of a ``results/`` value.
Labels appear as code spans preceded by a glyph. The word "bias" appears only
inside a label that §24 assigned.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from financial_llm_harness.oracles import glyph
from financial_llm_harness.reporting.tables import RunData

MINUS = "−"

FILE_MAP = (
    ("README.md", "this file: status, headline, file map, how to reproduce"),
    ("run.json", "run identity"),
    (
        "manifest/",
        "the frozen bundle: manifest, lock, cases, renderings, assets, corpus report, probes",
    ),
    ("results/summary.json", "the one-file machine answer"),
    ("results/effects.csv", "one row per model × feature × contrast"),
    ("results/labels.csv", "one row per model × feature: zone, labels, gates"),
    ("results/controls.csv", "null, placebo and positive controls"),
    ("results/missingness.csv", "trials per model, arm and status"),
    ("results/attribution.csv", "attribution per model and channel"),
    ("results/clusters.csv", "per-cluster matched outcomes"),
    ("results/trials.csv", "one row per trial"),
    ("results/costs.csv", "calls, tokens and cost per model and role"),
    ("results/DATA_DICTIONARY.md", "every file and column"),
    ("report/report.md", "the full report"),
    ("report/report.html", "the same report as one self-contained HTML file"),
    ("report/figures/", "the six figures"),
    ("models/", "per-model responses and examples"),
)


def num(x: float, dp: int = 2, signed: bool = False) -> str:
    s = f"{x:+.{dp}f}" if signed else f"{x:.{dp}f}"
    return s.replace("-", MINUS)


def theta(x: Any) -> str:
    return "—" if x is None else num(float(x), 2, signed=True)


def ci(lo: Any, hi: Any) -> str:
    if lo is None or hi is None:
        return "—"
    return f"[{num(float(lo), 2, True)}, {num(float(hi), 2, True)}]"


def pval(p: Any) -> str:
    if p is None:
        return "—"
    p = float(p)
    if p < 0.001:
        return "<0.001"
    return f"{p:.2g}"


def rate(r: Any) -> str:
    return "—" if r is None else f"{float(r) * 100:.1f}%"


def cost(c: Any) -> str:
    return "—" if c is None else f"${float(c):,.2f}"


def tokens(n: Any) -> str:
    if n is None:
        return "—"
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1e3:.1f}K"
    return f"{n / 1e6:.1f}M"


def lab(label: str | None) -> str:
    if not label:
        return "—"
    return f"{glyph(label)} `{label}`"


def code(text: Any) -> str:
    return f"`{text}`"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = [
        "| " + " | ".join(h.replace("|", "\\|") for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def header_line(rd: RunData, summary: Mapping[str, Any]) -> str:
    run = summary["run"]
    cr = run["code_ref"] or {}
    sha = cr.get("harness_git_sha")
    harness = (
        "unidentified"
        if sha is None
        else sha[:7] + ("-dirty" if cr.get("dirty") else "")
    )
    date = (run.get("started_at") or "")[:10]
    return (
        f"Run {code(run['run_id'][:8])} · {date} · {run['mode']} · {run['status']} · "
        f"design {code(run['design_hash'][:8])} · manifest {code(run['manifest_hash'][:8])} · harness {code(harness)}"
    )


def classification_line(summary: Mapping[str, Any]) -> str | None:
    run = summary["run"]
    if run["data_classification"] == "synthetic":
        return None
    pol = summary["output_policy"]
    line = f"Data classification: {run['data_classification']}."
    if not pol["publishable"]:
        line += f" Not publishable: {pol['reason']}. This run directory is git-ignored (§0.9, §30.1)."
    return line


def headline_table(rd: RunData, summary: Mapping[str, Any]) -> str:
    rows = []
    for m in summary["models"]:
        p = m["primary"]
        rows.append(
            [
                code(m["model_key"]),
                f"{theta(p['theta'])} {ci(*p['ci'])}",
                p["zone"] or "—",
                lab(p["sensitivity_label"]),
                lab(p["composite_label"] or p["attribution_label"]),
                ", ".join(p["gates_failed"]) or "—",
            ]
        )
    return table(
        ["Model", "θ̂ [CI]", "Zone", "Sensitivity", "Attribution", "Gates failed"], rows
    )


def _effects_by(
    rd: RunData, effects: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {(e["model_key"], e["feature_id"], e["contrast"]): e for e in effects}


def render_report(
    rd: RunData,
    summary: Mapping[str, Any],
    effects: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, Any]],
    attribution: Sequence[Mapping[str, Any]],
    costs: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> str:
    m = rd.manifest
    d = summary["design"]
    pf = d["primary_feature"]
    contrast = (
        "B_vs_A"
        if d["estimand"] != "arm_level"
        else f"level_{m['hypothesis']['level_arm']}"
    )
    eff = _effects_by(rd, effects)
    out: list[str] = []
    out.append(
        f"# {code(m['faq_id'])} {code(m['task'])}: {code(m['concept'])} under mandate {code(m['mandate']['id'])}"
    )
    out.append("")
    out.append(header_line(rd, summary))
    cl = classification_line(summary)
    if cl:
        out += ["", cl]
    out += ["", "## 1. Headline", "", headline_table(rd, summary), ""]
    # 2. Design
    out += ["## 2. Design", ""]
    out.append(
        table(
            ["Item", "Value"],
            [
                ["Question", code(m["research_question"])],
                ["Mandate", f"{code(m['mandate']['id'])} {code(m['mandate']['text'])}"],
                ["Intervention", code(d["intervention"])],
                [
                    "Closure",
                    code(
                        json.dumps(
                            (m.get("intervention_spec") or {}).get("closure"),
                            sort_keys=True,
                        )
                    ),
                ],
                ["Primary feature", code(pf)],
                ["Estimand", code(d["estimand"])],
                ["Oracle", f"{code(d['oracle_type'])} ({d['oracle_confidence']})"],
                ["δ", num(d["delta"], 2)],
                ["α", num(d["alpha"], 2)],
                [
                    "Clusters planned / achieved",
                    f"{d['n_clusters_required']} / {d['n_clusters']}",
                ],
                ["Cases", str(d["n_cases"])],
                ["Control subset", str(d["subset_n"])],
                ["P, R", f"{d['P']}, {d['R']}"],
                ["Panel", code(m["sampling_spec"]["panel_ref"]["name"])],
            ],
        )
    )
    out.append("")
    # 3. Primary effect
    out += [
        "## 3. Primary effect",
        "",
        "![Primary effect](figures/01_primary_effect.svg)",
        "",
    ]
    ctl = {(c["model_key"], c["control"]): c for c in controls}
    attr_by = {(a["model_key"], a["channel"]): a for a in attribution}
    for mm in summary["models"]:
        mk = mm["model_key"]
        e = eff.get((mk, pf, contrast)) or {}
        p = mm["primary"]
        out.append(f"### {code(mk)}")
        out.append("")
        lines = [
            f"- Clusters: {e.get('n_clusters')} (cases {e.get('n_cases')}; required N {d['n_clusters_required']}, planning {d['planning']}); matched valid blocks {e.get('n_blocks_matched')} of {e.get('n_blocks_scheduled')}",
        ]
        if contrast == "B_vs_A":
            lines.append(
                f"- Arm means over matched blocks: A {num(e['mean_a'], 2) if e.get('mean_a') is not None else '—'}, B {num(e['mean_b'], 2) if e.get('mean_b') is not None else '—'}"
            )
            lines.append(
                f"- θ̂ {theta(e.get('theta'))}, CI {ci(e.get('ci_low'), e.get('ci_high'))}, CI(1−2α) {ci(e.get('ci2a_low'), e.get('ci2a_high'))}; p {pval(e.get('p_raw'))}; zone {e.get('zone') or '—'}"
            )
        else:
            lines.append(
                f"- λ̂ {theta(e.get('theta'))}; one-sided bounds {ci(e.get('ci2a_low'), e.get('ci2a_high'))}; p_meets {pval(e.get('p_eq'))}, p_below {pval(e.get('p_dev'))}"
            )
        miss = mm["missingness"]
        lines.append(
            f"- Invalid rate A {rate(miss['invalid_rate_a'])}"
            + (
                f", B {rate(miss['invalid_rate_b'])}"
                if miss.get("invalid_rate_b") is not None
                else ""
            )
            + (
                "; the bounds change the zone"
                if miss["bounds_change_zone"]
                else "; the bounds keep the zone"
            )
        )
        cparts = []
        for name in ("null", "placebo", "positive"):
            c = ctl.get((mk, name))
            if c:
                cparts.append(
                    f"{name} θ {theta(c['theta'])} ({'passed' if c['passed'] else 'failed'})"
                )
        if cparts:
            lines.append("- Controls: " + " · ".join(cparts))
        lines.append(f"- Sensitivity: {lab(p['sensitivity_label'])}")
        for (amk, ch), a in sorted(attr_by.items()):
            if amk != mk or not a.get("label"):
                continue
            lines.append(
                f"- Attribution {code(ch)}: π_R {num(a['pi_r'], 2) if a.get('pi_r') is not None else '—'} "
                f"{ci(a.get('ci_low'), a.get('ci_high'))}, responsive clusters {a.get('n_responsive')} → {lab(a['label'])}"
            )
        out += lines + [""]
    # 4. Arm-level rates
    out += ["## 4. Arm-level rates", "", "![Arm rates](figures/02_arm_rates.svg)", ""]
    rows = []
    for mm in summary["models"]:
        e = eff.get((mm["model_key"], pf, contrast)) or {}
        rows.append(
            [
                code(mm["model_key"]),
                num(e["mean_a"], 2)
                if e.get("mean_a") is not None
                else (
                    num(e["theta"], 2)
                    if contrast != "B_vs_A" and e.get("theta") is not None
                    else "—"
                ),
                num(e["mean_b"], 2) if e.get("mean_b") is not None else "—",
                str(e.get("n_blocks_matched", "—")),
            ]
        )
    out += [table(["Model", "Mean A (or λ)", "Mean B", "Matched blocks"], rows), ""]
    # 5. Controls
    out += ["## 5. Controls", "", "![Controls](figures/03_controls.svg)", ""]
    rows = []
    for c in controls:
        rows.append(
            [
                code(c["model_key"]),
                c["control"],
                code(c["expected"]),
                code(c["observed_label"]) if c.get("observed_label") else "—",
                theta(c["theta"]),
                ci(c["ci2a_low"], c["ci2a_high"]),
                "passed" if c["passed"] else "failed",
            ]
        )
    out += [
        table(
            ["Model", "Control", "Expected", "Observed", "θ̂", "CI(1−2α)", "Result"],
            rows,
        )
        if rows
        else "No controls (discovery mode).",
        "",
    ]
    # 6. Missingness
    out += ["## 6. Missingness", "", "![Missingness](figures/04_missingness.svg)", ""]
    rows = [
        [
            code(r["model_key"]),
            r["arm"],
            code(r["status"]),
            r["status_class"],
            str(r["n"]),
            rate(r["rate"]),
        ]
        for r in missing
    ]
    out += [table(["Model", "Arm", "Status", "Class", "n", "Rate"], rows), ""]
    for mm in summary["models"]:
        out.append(
            f"- {code(mm['model_key'])}: worst-case bounds {'change' if mm['missingness']['bounds_change_zone'] else 'keep'} the zone."
        )
    out.append("")
    # 7. Attribution
    out += ["## 7. Attribution", "", "![Attribution](figures/05_attribution.svg)", ""]
    rows = []
    for a in attribution:
        rows.append(
            [
                code(a["model_key"]),
                code(a["channel"]),
                str(a["n_responsive"]),
                num(a["pi_r"], 2) if a.get("pi_r") is not None else "—",
                ci(a.get("ci_low"), a.get("ci_high")),
                num(a["f0"], 2) if a.get("f0") is not None else "—",
                "yes" if a.get("eligible") else "no",
                lab(a.get("label")),
                lab(a.get("composite_label")),
            ]
        )
    out += [
        table(
            [
                "Model",
                "Channel",
                "|R|",
                "π_R",
                "CI",
                "f₀",
                "Eligible",
                "Label",
                "Composite",
            ],
            rows,
        )
        if rows
        else "No attribution channel declared.",
        "",
    ]
    # 8. Secondary and exploratory
    out += ["## 8. Secondary and exploratory", ""]
    rows = []
    for e in effects:
        if e["analysis_role"] in ("secondary", "exploratory"):
            rows.append(
                [
                    code(e["model_key"]),
                    code(e["feature_id"]),
                    code(e["contrast"]),
                    e["analysis_role"],
                    theta(e["theta"]),
                    ci(e["ci_low"], e["ci_high"]),
                    pval(e["p_raw"]),
                    pval(e["p_dev_adj"]) if e.get("p_dev_adj") is not None else "—",
                    e["zone"] or "—",
                ]
            )
    out += [
        table(
            ["Model", "Feature", "Contrast", "Role", "θ̂", "CI", "p", "p (BH)", "Zone"],
            rows,
        )
        if rows
        else "None.",
        "",
    ]
    # 9. Diagnostics
    dg = summary["diagnostics"]
    out += ["## 9. Diagnostics", "", "![Cost](figures/06_cost.svg)", ""]
    out.append(
        f"- Corpus: acceptance rate {rate(dg['acceptance_rate'])}; largest near-duplicate cluster {dg['largest_cluster']}."
    )
    for mm in summary["models"]:
        drift = mm["drift"]
        out.append(
            f"- {code(mm['model_key'])}: capability drift {'yes' if drift['capability'] else 'no'}, serving drift {'yes' if drift['serving'] else 'no'}."
        )
    rows = [
        [
            code(c["model_key"]),
            c["role"],
            str(c["calls"]),
            tokens(c["input_tokens"]),
            tokens(c["output_tokens"]),
            cost(c["cost_usd"]),
        ]
        for c in costs
    ]
    out += [
        "",
        table(
            ["Model", "Role", "Calls", "Input tokens", "Output tokens", "Cost"], rows
        ),
        "",
    ]
    # 10. Provenance
    run = summary["run"]
    cr = run["code_ref"] or {}
    out += ["## 10. Provenance", ""]
    out.append(
        table(
            ["Item", "Value"],
            [
                ["Experiment key", code(run["experiment_key"])],
                ["Experiment ID", code(run["experiment_id"])],
                ["Design hash", code(run["design_hash"])],
                ["Manifest hash", code(run["manifest_hash"])],
                ["Harness git SHA", code(cr.get("harness_git_sha"))],
                ["Harness tree", code(cr.get("harness_tree_sha256"))],
                ["Dirty", code(cr.get("dirty"))],
                ["Seeds", code(json.dumps(d["seeds"], sort_keys=True))],
                ["Budget override", code(rd.run.get("budget_override_json"))],
            ],
        )
    )
    out += ["", table(["Path", "Contents"], [[code(p), t] for p, t in FILE_MAP]), ""]
    return "\n".join(out)


def render_readme(rd: RunData, summary: Mapping[str, Any]) -> str:
    run = summary["run"]
    m = rd.manifest
    out = [
        f"# {code(m['experiment_key'])}",
        "",
        f"Status: {run['status']} · {run['mode']} · {header_line(rd, summary)}",
    ]
    cl = classification_line(summary)
    if cl:
        out += ["", cl]
    out += ["", "## 1. Headline", "", headline_table(rd, summary), ""]
    out += [
        "## 2. Files",
        "",
        table(["Path", "Contents"], [[code(p), t] for p, t in FILE_MAP]),
        "",
    ]
    rid = run["run_id"]
    out += [
        "## 3. Reproduce",
        "",
        f"- Re-render every output file from the database: {code(f'financial-harness render {rid}')}",
        f"- Re-evaluate stored responses byte-identically: {code(f'financial-harness replay {rid} --reevaluate')}",
        "",
    ]
    return "\n".join(out)
