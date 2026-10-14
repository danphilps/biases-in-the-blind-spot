"""``programme report`` (spec §32B.4, §32B.5, §25, D-17, D-18, D-20, E-30).

A pure function of ``state.json``, the programme file, each run's database
(extracted from its archive into a temporary directory when ``db/`` is absent)
and the registry. BH runs per (model, claim family) on the p-values stored in
each run's ``analysis`` rows. No effect estimate or confidence interval is
recomputed and no evaluator is called. Re-running gives byte-identical files;
timestamps go only to ``logs/build_info.json``.
"""

from __future__ import annotations

import json
import platform
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import ids, yaml_io
from financial_llm_harness.core.canonical_json import sha256_canonical
from financial_llm_harness.oracles import (
    ATTRIBUTION_ELIGIBLE,
    Oracle,
    bias_flags,
    composite_label,
    label_class,
    programme_label,
    programme_zone,
)
from financial_llm_harness.reporting import markdown as MD
from financial_llm_harness.reporting import tables as T
from financial_llm_harness.reporting.tables import csv_text
from financial_llm_harness.statistics.correction import benjamini_hochberg
from financial_llm_harness.statistics.paired import ci, p_two_sided

ANSWERS_COLUMNS = (
    "row_id",
    "faq_id",
    "variant_id",
    "intervention_id",
    "experiment_key",
    "mode",
    "model_key",
    "run_id",
    "run_status",
    "oracle_type",
    "oracle_direction",
    "oracle_confidence",
    "estimand",
    "theta",
    "ci_low",
    "ci_high",
    "p_dev",
    "p_eq",
    "p_dev_adj",
    "p_eq_adj",
    "run_zone",
    "programme_zone",
    "run_label",
    "programme_label",
    "is_bias",
    "is_normative_failure",
    "attribution_label",
    "attribution_channel",
    "composite_label",
    "gates_failed",
    "run_label_row_id",
)
QUESTION_COLUMNS = (
    "row_id",
    "faq_id",
    "model_key",
    "n_experiments",
    "n_complete",
    "n_good",
    "n_warning",
    "n_serious",
    "n_critical",
    "n_neutral",
    "any_bias",
    "verdict_class",
    "verdict_label",
    "verdict_variant_id",
    "verdict_intervention_id",
    "verdict_answer_row_id",
)
CONSISTENCY_COLUMNS = (
    "row_id",
    "model_key",
    "faq_id",
    "experiment_key_1",
    "experiment_key_2",
    "check",
    "n_cases_matched",
    "rate",
    "ci_low",
    "ci_high",
)
CROSS_VARIANT_COLUMNS = (
    "row_id",
    "model_key",
    "faq_id",
    "intervention_id",
    "feature_id",
    "variant_x",
    "variant_ref",
    "experiment_key_x",
    "experiment_key_ref",
    "estimand",
    "n_clusters",
    "theta_x",
    "theta_ref",
    "gamma",
    "ci_low",
    "ci_high",
    "p_two_sided",
    "alpha",
    "analysis_role",
)
FDR_COLUMNS = (
    "row_id",
    "model_key",
    "experiment_key",
    "fdr_family",
    "test",
    "p_raw",
    "p_adj",
    "rank",
    "bh_threshold",
    "rejected",
    "m",
    "q",
)
RUNS_COLUMNS = (
    "row_id",
    "experiment_key",
    "faq_id",
    "variant_id",
    "intervention_id",
    "mode",
    "manifest_hash",
    "run_id",
    "status",
    "exit_code",
    "calls",
    "cost_usd",
)
COVERAGE_COLUMNS = (
    "row_id",
    "faq_id",
    "wave",
    "priority",
    "n_variants",
    "n_interventions_listed",
    "n_pairs",
    "n_specs",
    "n_complete",
    "n_unimplemented",
)
FAMILIES = ("sensitivity_or_deviation", "equivalence_or_adequacy")
SEVERITY = ("critical", "serious", "warning", "good", "neutral")
TESTS = {
    "INVARIANCE": ("t_two_sided", "tost"),
    "DISCOVERY": ("t_two_sided", "tost"),
    "DIRECTIONAL": ("t_two_sided", "t_under"),
    "MAGNITUDE": ("t_outside_tol", "tost"),
    "LEVEL": ("t_below", "t_meets"),
}


@dataclass
class Experiment:
    """One programme experiment with the run values the report needs (never recomputed)."""

    key: str
    manifest: dict[str, Any]
    state: dict[str, Any]
    run_dir: Path | None
    effects: dict[tuple[str, str, str], dict[str, Any]]
    labels: dict[tuple[str, str], dict[str, Any]]
    attribution: dict[str, dict[str, Any] | None]
    calls: int
    cost_usd: float
    # CR-14: the primary B − A cluster outcomes x_c per model, and a fingerprint of the
    # case set (case_idx, cluster_ordinal, arm A and B states) that pairs runs on it
    primary_clusters: dict[str, dict[int, float]] = field(default_factory=dict)
    case_fingerprint: str | None = None

    @property
    def complete(self) -> bool:
        return self.state["status"] == "COMPLETE" and self.run_dir is not None


def _load_experiment(pdir: Path, e: Mapping[str, Any]) -> Experiment:
    from financial_llm_harness.execution.runner import resolve_run
    from financial_llm_harness.reporting.replay import ensure_db
    from financial_llm_harness.storage.db import db_path

    mdir = pdir / "manifests" / e["experiment_key"]
    manifest = B.load_manifest(mdir) if (mdir / B.MANIFEST_FILE).exists() else {}
    if not e.get("run_id"):
        return Experiment(
            e["experiment_key"], manifest, dict(e), None, {}, {}, {}, 0, 0.0
        )
    run_dir = resolve_run(e["run_id"])
    with ensure_db(run_dir, into_run_dir=False) as data_dir:
        conn = sqlite3.connect(str(db_path(data_dir)))
        try:
            rd = T.load_run_data(run_dir, conn, data_dir)
            effects = {
                (r["model_key"], r["feature_id"], r["contrast"]): r
                for r in T.effects_rows(rd)
            }
            labels = {(r["model_key"], r["feature_id"]): r for r in T.labels_rows(rd)}
            attribution = {mk: T.headline_attribution(rd, mk) for mk in rd.model_keys}
            primary_clusters = _primary_clusters(rd)
            fingerprint = sha256_canonical(
                [
                    [i, c["cluster_ordinal"], c["arms"].get("A"), c["arms"].get("B")]
                    for i, c in sorted(rd.cases.items())
                ]
            )
            calls, cost = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0) FROM response"
            ).fetchone()
        finally:
            conn.close()
    return Experiment(
        e["experiment_key"],
        manifest or rd.manifest,
        dict(e),
        run_dir,
        effects,
        labels,
        attribution,
        int(calls),
        float(cost),
        primary_clusters,
        fingerprint,
    )


def _primary_clusters(rd: Any) -> dict[str, dict[int, float]]:
    """x_c of the primary B_vs_A row per model (empty for LEVEL primaries)."""
    out: dict[str, dict[int, float]] = {}
    for mk in rd.model_keys:
        prim = rd.primary_row(mk)
        if prim is None or prim["contrast"] != "B_vs_A":
            continue
        out[mk] = {
            int(c["cluster_ordinal"]): float(c["x_c"])
            for c in prim["results"].get("clusters") or []
            if c.get("x_c") is not None
        }
    return out


def cross_variant_rows(
    exps: Sequence[Experiment],
    models: Sequence[str],
    variant_order: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Matched cross-variant contrasts γ = θ_x − θ_ref (CR-14; exploratory, never a label).

    Two COMPLETE runs of one item and intervention, with different variants (mandates),
    the same primary feature, a ``mean_paired_difference`` primary and the identical case
    set, are compared cluster by cluster: g_c = d_c^x − d_c^ref over the clusters with an
    outcome in both, then the §23.3 one-sample t-test and CI_{1−α} on g. The reference is
    the item's first variant (``faq.yaml`` order, else sorted). This estimates how the
    mandate changes the dependence directly, rather than comparing two p-values.
    """
    groups: dict[tuple[str, str, str], list[Experiment]] = {}
    for x in exps:
        m = x.manifest
        if not x.complete or m.get("faq_id") in (None, "none"):
            continue
        if m["hypothesis"]["estimand"] != "mean_paired_difference":
            continue
        key = (m["faq_id"], m["intervention_id"], m["hypothesis"]["primary_feature"])
        groups.setdefault(key, []).append(x)
    out: list[dict[str, Any]] = []
    for (faq_id, iid, pf), members in sorted(groups.items()):
        order = list((variant_order or {}).get(faq_id) or [])
        members = sorted(
            members,
            key=lambda e: (
                order.index(e.manifest["variant_id"])
                if e.manifest["variant_id"] in order
                else len(order),
                e.manifest["variant_id"],
            ),
        )
        if len({e.manifest["variant_id"] for e in members}) < 2:
            continue
        ref = members[0]
        for x in members[1:]:
            if x.case_fingerprint is None or x.case_fingerprint != ref.case_fingerprint:
                continue
            alpha = float(x.manifest["hypothesis"]["alpha"])
            for mk in models:
                cx, cr = x.primary_clusters.get(mk), ref.primary_clusters.get(mk)
                if not cx or not cr:
                    continue
                common = sorted(set(cx) & set(cr))
                g = np.array([cx[c] - cr[c] for c in common], dtype=float)
                tx = np.array([cx[c] for c in common], dtype=float)
                tr = np.array([cr[c] for c in common], dtype=float)
                ci_ = ci(g, alpha)
                out.append(
                    {
                        "row_id": ids.row_id("cross_variant", mk, x.key, ref.key),
                        "model_key": mk,
                        "faq_id": faq_id,
                        "intervention_id": iid,
                        "feature_id": pf,
                        "variant_x": x.manifest["variant_id"],
                        "variant_ref": ref.manifest["variant_id"],
                        "experiment_key_x": x.key,
                        "experiment_key_ref": ref.key,
                        "estimand": "difference_of_mean_paired_differences",
                        "n_clusters": len(common),
                        "theta_x": float(tx.mean()) if len(common) else None,
                        "theta_ref": float(tr.mean()) if len(common) else None,
                        "gamma": float(g.mean()) if len(common) else None,
                        "ci_low": None if ci_ is None else ci_[0],
                        "ci_high": None if ci_ is None else ci_[1],
                        "p_two_sided": p_two_sided(g, 0.0),
                        "alpha": alpha,
                        "analysis_role": "exploratory",
                    }
                )
    order_m = {mk: i for i, mk in enumerate(models)}
    out.sort(
        key=lambda r: (
            order_m[r["model_key"]],
            r["faq_id"],
            r["intervention_id"],
            r["experiment_key_x"],
        )
    )
    return out


def _primary_contrast(manifest: Mapping[str, Any]) -> str:
    h = manifest["hypothesis"]
    return f"level_{h['level_arm']}" if h["estimand"] == "arm_level" else "B_vs_A"


def fdr_rows(
    exps: Sequence[Experiment], models: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], Any]]:
    """BH per (model, claim family) over every COMPLETE confirmatory primary (§25, D-17)."""
    out: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str], Any] = {}
    alphas = {float(x.manifest["hypothesis"]["alpha"]) for x in exps if x.complete}
    q = min(alphas) if alphas else 0.05
    for mk in models:
        for fam_i, fam in enumerate(FAMILIES):
            members = []
            tests: dict[str, str] = {}
            for x in exps:
                if not x.complete or x.manifest["mode"] != "confirmatory":
                    continue
                pf = x.manifest["hypothesis"]["primary_feature"]
                eff = x.effects.get((mk, pf, _primary_contrast(x.manifest)))
                if eff is None:
                    continue
                p = eff["p_dev"] if fam_i == 0 else eff["p_eq"]
                if p is None:
                    continue
                members.append((x.key, float(p)))
                otype = Oracle.from_spec(x.manifest["oracle_spec"]).effective_type
                tests[x.key] = TESTS.get(otype, ("", ""))[fam_i]
            for r in benjamini_hochberg(members, q):
                row = {
                    "row_id": ids.row_id("fdr", mk, r.key, fam),
                    "model_key": mk,
                    "experiment_key": r.key,
                    "fdr_family": fam,
                    "test": tests[r.key],
                    "p_raw": r.p_raw,
                    "p_adj": r.p_adj,
                    "rank": r.rank,
                    "bh_threshold": r.bh_threshold,
                    "rejected": r.rejected,
                    "m": r.m,
                    "q": r.q,
                }
                out.append(row)
                by_key[(mk, r.key, fam)] = r
    return out, by_key


def answers_rows(
    exps: Sequence[Experiment],
    models: Sequence[str],
    bh: Mapping[tuple[str, str, str], Any],
) -> list[dict[str, Any]]:
    rows = []
    for x in exps:
        m = x.manifest
        oracle = m.get("oracle_spec") or {}
        for mk in models:
            base = {
                "row_id": ids.row_id("answers", x.key, mk),
                "faq_id": m.get("faq_id"),
                "variant_id": m.get("variant_id"),
                "intervention_id": m.get("intervention_id"),
                "experiment_key": x.key,
                "mode": m.get("mode"),
                "model_key": mk,
                "run_id": x.state.get("run_id"),
                "run_status": x.state.get("status"),
                "oracle_type": oracle.get("type"),
                "oracle_direction": T.oracle_direction(oracle) if oracle else None,
                "oracle_confidence": oracle.get("confidence"),
                "estimand": (m.get("hypothesis") or {}).get("estimand"),
            }
            if not x.complete:
                rows.append(
                    {
                        **base,
                        "run_label": "INDETERMINATE",
                        "programme_label": "INDETERMINATE",
                        "is_bias": False,
                        "is_normative_failure": False,
                        "gates_failed": ["run_not_complete"],
                    }
                )
                continue
            pf = m["hypothesis"]["primary_feature"]
            eff = x.effects.get((mk, pf, _primary_contrast(m))) or {}
            lab = x.labels.get((mk, pf)) or {}
            run_label = lab.get("sensitivity_label")
            run_zone = lab.get("zone")
            orc = Oracle.from_spec(oracle)
            otype = orc.effective_type
            attr = x.attribution.get(mk) or {}
            confirmatory = m["mode"] == "confirmatory"
            if confirmatory:
                dev = bh.get((mk, x.key, FAMILIES[0]))
                eq = bh.get((mk, x.key, FAMILIES[1]))
                dev_rej = bool(dev and dev.rejected)
                eq_rej = bool(eq and eq.rejected)
                p_label = programme_label(
                    otype, run_label or "INDETERMINATE", dev_rej, eq_rej
                )
                p_zone = (
                    programme_zone(otype, run_zone or "", dev_rej, eq_rej)
                    if run_zone
                    else ""
                )
                is_bias, is_nf = bias_flags(m["mode"], orc, p_label)
                eligible = p_label in ATTRIBUTION_ELIGIBLE
                a_label = (
                    attr.get("label_display")
                    if eligible
                    else ("INDETERMINATE" if attr else None)
                )
                comp = (
                    composite_label(attr.get("label") or "", p_label, is_bias)
                    if eligible and attr
                    else None
                )
                p_dev_adj = dev.p_adj if dev else None
                p_eq_adj = eq.p_adj if eq else None
            else:
                p_label, p_zone = run_label, run_zone
                is_bias = lab.get("is_bias", False)
                is_nf = lab.get("is_normative_failure", False)
                a_label = lab.get("attribution_label")
                comp = lab.get("composite_label")
                p_dev_adj = p_eq_adj = None
            rows.append(
                {
                    **base,
                    "theta": eff.get("theta"),
                    "ci_low": eff.get("ci_low"),
                    "ci_high": eff.get("ci_high"),
                    "p_dev": eff.get("p_dev"),
                    "p_eq": eff.get("p_eq"),
                    "p_dev_adj": p_dev_adj,
                    "p_eq_adj": p_eq_adj,
                    "run_zone": run_zone,
                    "programme_zone": p_zone or None,
                    "run_label": run_label,
                    "programme_label": p_label,
                    "is_bias": bool(is_bias),
                    "is_normative_failure": bool(is_nf),
                    "attribution_label": a_label,
                    "attribution_channel": attr.get("channel") if a_label else None,
                    "composite_label": comp,
                    "gates_failed": list(lab.get("gates_failed") or []),
                    "run_label_row_id": lab.get("row_id"),
                }
            )
    order = {mk: i for i, mk in enumerate(models)}
    rows.sort(
        key=lambda r: (
            str(r["faq_id"]),
            str(r["variant_id"]),
            str(r["intervention_id"]),
            r["experiment_key"],
            order[r["model_key"]],
        )
    )
    return rows


def _headline(r: Mapping[str, Any]) -> str | None:
    """A composite replaces its programme label (D-20)."""
    return r.get("composite_label") or r.get("programme_label")


def question_rows(
    answers: Sequence[Mapping[str, Any]], models: Sequence[str]
) -> list[dict[str, Any]]:
    """One verdict per (faq_id, model), without new statistics (D-20)."""
    out = []
    for faq in sorted({str(a["faq_id"]) for a in answers}):
        for mk in models:
            rows = [
                a for a in answers if str(a["faq_id"]) == faq and a["model_key"] == mk
            ]
            complete = [a for a in rows if a["run_status"] == "COMPLETE"]
            counts: dict[str, int] = dict.fromkeys(SEVERITY, 0)
            for a in complete:
                counts[label_class(_headline(a))] += 1
            verdict_class, first = "neutral", None
            if complete:
                for c in SEVERITY:
                    if counts[c]:
                        verdict_class = c
                        first = next(
                            a for a in complete if label_class(_headline(a)) == c
                        )
                        break
            out.append(
                {
                    "row_id": ids.row_id("question_answers", faq, mk),
                    "faq_id": faq,
                    "model_key": mk,
                    "n_experiments": len(rows),
                    "n_complete": len(complete),
                    "n_good": counts["good"],
                    "n_warning": counts["warning"],
                    "n_serious": counts["serious"],
                    "n_critical": counts["critical"],
                    "n_neutral": counts["neutral"],
                    "any_bias": any(a["is_bias"] is True for a in complete),
                    "verdict_class": verdict_class,
                    "verdict_label": _headline(first) if first else "INDETERMINATE",
                    "verdict_variant_id": first["variant_id"] if first else None,
                    "verdict_intervention_id": first["intervention_id"]
                    if first
                    else None,
                    "verdict_answer_row_id": first["row_id"] if first else None,
                }
            )
    return out


def _registry() -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    from financial_llm_harness.taxonomy.registry import (
        generated_dir,
        load_registry,
        taxonomy_dir,
    )

    rep_path = generated_dir() / "expand_report.json"
    faq = None
    if (taxonomy_dir() / "faq.yaml").exists():
        faq = {
            fid: {**item.data, "wave": item.wave}
            for fid, item in load_registry().items.items()
        }
    report = (
        json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else None
    )
    return faq, report


def coverage_rows(
    exps: Sequence[Experiment], programme: Mapping[str, Any]
) -> list[dict[str, Any]]:
    faq, report = _registry()
    covered = {str(x.manifest.get("faq_id")) for x in exps if x.manifest.get("faq_id")}
    gen = (programme.get("select") or {}).get("generated")
    if gen and faq:
        for fid, item in faq.items():
            if (not gen.get("waves") or item.get("wave") in gen["waves"]) and (
                not gen.get("priorities") or item.get("priority") in gen["priorities"]
            ):
                covered.add(fid)
    rows = []
    for fid in sorted(covered):
        item = (faq or {}).get(fid)
        n_var = n_int = n_pairs = None
        if item is not None:
            n_var = len(item.get("variants") or []) or 1
            ints = item.get("interventions") or {}
            n_int = len(ints.get("relevant") or []) + len(ints.get("irrelevant") or [])
            n_pairs = n_var * (n_int + (1 if item.get("correctness_checks") else 0))
        mine = [x for x in exps if str(x.manifest.get("faq_id")) == fid]
        rows.append(
            {
                "row_id": ids.row_id("coverage", fid),
                "faq_id": fid,
                "wave": item.get("wave") if item else None,
                "priority": item.get("priority") if item else None,
                "n_variants": n_var,
                "n_interventions_listed": n_int,
                "n_pairs": n_pairs,
                "n_specs": len(mine),
                "n_complete": sum(1 for x in mine if x.complete),
                "n_unimplemented": None
                if report is None
                else sum(1 for r in report if r.get("faq_id") == fid),
            }
        )
    return rows


def runs_rows(exps: Sequence[Experiment]) -> list[dict[str, Any]]:
    return [
        {
            "row_id": ids.row_id("runs", x.key),
            "experiment_key": x.key,
            "faq_id": x.manifest.get("faq_id"),
            "variant_id": x.manifest.get("variant_id"),
            "intervention_id": x.manifest.get("intervention_id"),
            "mode": x.manifest.get("mode"),
            "manifest_hash": x.state.get("manifest_hash"),
            "run_id": x.state.get("run_id"),
            "status": x.state.get("status"),
            "exit_code": x.state.get("exit_code"),
            "calls": x.calls,
            "cost_usd": x.cost_usd,
        }
        for x in sorted(exps, key=lambda x: x.key)
    ]


def output_policy(exps: Sequence[Experiment]) -> dict[str, Any]:
    """Publishable iff every included run is (§30.1)."""
    policies = []
    for x in exps:
        if x.run_dir is None:
            continue
        s = json.loads(
            (x.run_dir / "results" / "summary.json").read_text(encoding="utf-8")
        )
        policies.append((s["run"]["data_classification"], s["output_policy"]))
    if all(c == "synthetic" for c, _ in policies):
        return {"publishable": True, "reason": "synthetic"}
    bad = [p for _, p in policies if not p["publishable"]]
    if bad:
        return {"publishable": False, "reason": bad[0]["reason"]}
    return {"publishable": True, "reason": "user_output_policy"}


def answer_matrix_svg(
    answers: Sequence[Mapping[str, Any]], models: Sequence[str]
) -> str:
    from xml.sax.saxutils import escape

    from financial_llm_harness.oracles import glyph
    from financial_llm_harness.reporting import svg as S

    keys = []
    for a in answers:
        k = (str(a["faq_id"]), str(a["variant_id"]), str(a["intervention_id"]))
        if k not in keys:
            keys.append(k)
    cell = {
        (
            str(a["faq_id"]),
            str(a["variant_id"]),
            str(a["intervention_id"]),
            a["model_key"],
        ): a
        for a in answers
    }
    left = S.LEFT
    col_w = (S.WIDTH - left - 16) / max(1, len(models))
    c = S.Canvas(len(keys), "Answer matrix")
    c.text(16, 24, "Answer matrix", cls="ink")
    for j, mk in enumerate(models):
        c.text(left + j * col_w + 6, 52, mk, cls="ink2", title=f"model {mk}")
    for i, k in enumerate(keys):
        y = 60 + i * S.ROW_H
        c.text(16, y + 24, " · ".join(k), cls="ink2")
        for j, mk in enumerate(models):
            a = cell.get((*k, mk))
            lab = a.get("programme_label") if a else None
            x = left + j * col_w
            token = S.STATUS_TOKEN[label_class(lab)] if lab else "muted"
            tip = (
                f"{' · '.join(k)} × {mk}: programme label {lab or '—'}; composite {(a or {}).get('composite_label') or '—'}; "
                f"row_id {(a or {}).get('row_id', '—')}"
            )
            c.add(
                f'<rect x="{S.r2(x + 2)}" y="{S.r2(y + 4)}" width="{S.r2(col_w - 4)}" height="{S.r2(S.ROW_H - 8)}" '
                f'style="fill:var(--{token});fill-opacity:0.15"><title>{escape(tip)}</title></rect>'
            )
            c.text(
                x + 8,
                y + 24,
                f"{glyph(lab)} {lab}" if lab else "—",
                cls="ink",
                title=tip,
            )
    return c.svg()


def programme_md(
    programme: Mapping[str, Any],
    state: Mapping[str, Any],
    exps: Sequence[Experiment],
    models: Sequence[str],
    answers: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    fdr: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> str:
    faq, report = _registry()
    out = [f"# Programme {MD.code(programme['programme_key'])}", ""]
    out.append(
        f"Programme run {MD.code(state['programme_run_id'][:8])} · {(state.get('started_at') or '')[:10]} · "
        f"{state['status']} · panel {MD.code(programme['panel'])}"
    )
    if not policy["publishable"]:
        out += [
            "",
            f"Not publishable: {policy['reason']}. This directory is git-ignored (§0.9, §30.1).",
        ]
    out += [
        "",
        "## 1. Answer matrix",
        "",
        "![Answer matrix](figures/01_answer_matrix.svg)",
        "",
    ]
    keys = []
    for a in answers:
        k = (a["faq_id"], a["variant_id"], a["intervention_id"])
        if k not in keys:
            keys.append(k)
    cell = {
        (a["faq_id"], a["variant_id"], a["intervention_id"], a["model_key"]): a
        for a in answers
    }
    rows = [
        [MD.code(k[0]), MD.code(k[1]), MD.code(k[2])]
        + [MD.lab(cell[(*k, mk)]["programme_label"]) for mk in models]
        for k in keys
    ]
    out += [
        MD.table(
            ["Item", "Variant", "Intervention", *[MD.code(m) for m in models]], rows
        ),
        "",
    ]
    out += ["## 2. Answers by question", ""]
    for fid in sorted({str(a["faq_id"]) for a in answers}):
        out.append(f"### {MD.code(fid)}")
        out.append("")
        question = (faq or {}).get(fid, {}).get("question")
        if question is None:
            mine = [
                x for x in exps if str(x.manifest.get("faq_id")) == fid and x.manifest
            ]
            question = mine[0].manifest.get("research_question") if mine else None
        if question:
            out += [f"Question: {MD.code(question)}", ""]
        for q in questions:
            if q["faq_id"] == fid:
                out.append(
                    f"- {MD.code(q['model_key'])}: {MD.lab(q['verdict_label'])} ({q['verdict_class']}"
                    + (
                        f"; variant {MD.code(q['verdict_variant_id'])}, {MD.code(q['verdict_intervention_id'])}"
                        if q["verdict_variant_id"]
                        else ""
                    )
                    + f"; {q['n_complete']} of {q['n_experiments']} experiments complete)"
                )
        out.append("")
        rows = [
            [MD.code(k[1]), MD.code(k[2])]
            + [MD.lab(_headline(cell[(*k, mk)])) for mk in models]
            for k in keys
            if str(k[0]) == fid
        ]
        out += [
            MD.table(["Variant", "Intervention", *[MD.code(m) for m in models]], rows),
            "",
        ]
    out += ["## 3. Multiple testing", ""]
    mt_rows = []
    for mk in models:
        for fam in FAMILIES:
            members = [
                r for r in fdr if r["model_key"] == mk and r["fdr_family"] == fam
            ]
            mt_rows.append(
                [
                    MD.code(mk),
                    MD.code(fam),
                    str(len(members)),
                    str(sum(1 for r in members if r["rejected"])),
                ]
            )
    out += [MD.table(["Model", "Claim family", "m", "Rejected"], mt_rows), ""]
    out.append(
        "BH adjusts p-values only; every θ̂, CI, zone and run label above is the run's own (§25, E-30)."
    )
    out += ["", "## 4. Coverage and gaps", ""]
    cov_rows = [
        [
            MD.code(c["faq_id"]),
            str(c["n_specs"]),
            str(c["n_complete"]),
            "—" if c["n_unimplemented"] is None else str(c["n_unimplemented"]),
        ]
        for c in coverage
    ]
    out += [
        MD.table(["Item", "Specs", "Complete", "Unimplemented pairs"], cov_rows),
        "",
    ]
    covered = {c["faq_id"] for c in coverage}
    gaps = [r for r in report or [] if r.get("faq_id") in covered]
    if gaps:
        out += [
            MD.table(
                ["Item", "Variant", "Intervention", "Wave", "Reason"],
                [
                    [
                        MD.code(g["faq_id"]),
                        MD.code(g["variant_id"]),
                        MD.code(g["intervention_id"]),
                        str(g.get("wave")),
                        MD.code(g["reason"]),
                    ]
                    for g in gaps
                ],
            ),
            "",
        ]
    elif report is None:
        out += ["No `expand_report.json` exists yet; registry gaps are not listed.", ""]
    else:
        out += ["No unimplemented pair among the covered items.", ""]
    out += ["## 5. Runs and cost", ""]
    out += [
        MD.table(
            ["Experiment", "Run", "Status", "Calls", "Cost"],
            [
                [
                    MD.code(r["experiment_key"]),
                    MD.code((r["run_id"] or "—")[:8]),
                    str(r["status"]),
                    str(r["calls"]),
                    MD.cost(r["cost_usd"]),
                ]
                for r in runs
            ],
        ),
        "",
    ]
    out += ["## 6. Provenance", ""]
    out += [
        MD.table(
            ["Item", "Value"],
            [
                ["Programme run ID", MD.code(state["programme_run_id"])],
                ["Programme file sha256", MD.code(state["programme_sha256"])],
                ["Panel file sha256", MD.code(state["panel_sha256"])],
            ]
            + [
                [
                    f"Manifest {MD.code(r['experiment_key'])}",
                    MD.code(r["manifest_hash"]),
                ]
                for r in runs
            ],
        ),
        "",
    ]
    return "\n".join(out)


def tested_models(panel_keys: Sequence[str], exps: Sequence[Experiment]) -> list[str]:
    """Panel models in panel order, less those every manifest skipped for want of an API key."""
    tested: set[str] = set()
    skipped: set[str] = set()
    for x in exps:
        ss = x.manifest.get("sampling_spec") or {}
        tested |= {m["model_key"] for m in ss.get("models") or []}
        skipped |= {m["model_key"] for m in ss.get("models_skipped") or []}
    return [mk for mk in panel_keys if mk in tested or mk not in skipped]


def render_programme(pdir: Path) -> None:
    """Write ``results/``, ``report/`` and ``README.md`` of a programme directory (§32B.4)."""
    from financial_llm_harness.models.panels import load_panel
    from financial_llm_harness.reporting import html as H
    from financial_llm_harness.reporting import svg as S

    state = json.loads((pdir / "state.json").read_text(encoding="utf-8"))
    programme = yaml_io.load(pdir / "programme.yaml")
    assert isinstance(programme, dict)
    exps = [_load_experiment(pdir, e) for e in state["experiments"]]
    models = tested_models(
        [m["model_key"] for m in load_panel(programme["panel"]).models], exps
    )
    fdr, bh = fdr_rows(exps, models)
    answers = answers_rows(exps, models, bh)
    questions = question_rows(answers, models)
    coverage = coverage_rows(exps, programme)
    runs = runs_rows(exps)
    policy = output_policy(exps)
    faq, _ = _registry()
    cross = cross_variant_rows(
        exps,
        models,
        {
            fid: [str(v["variant_id"]) for v in (item.get("variants") or [])]
            for fid, item in (faq or {}).items()
        },
    )
    results = pdir / "results"
    report = pdir / "report"
    (report / "figures").mkdir(parents=True, exist_ok=True)
    results.mkdir(exist_ok=True)
    order = {mk: i for i, mk in enumerate(models)}
    fdr.sort(
        key=lambda r: (
            order[r["model_key"]],
            FAMILIES.index(r["fdr_family"]),
            r["rank"],
        )
    )
    files = {
        "answers.csv": csv_text(ANSWERS_COLUMNS, answers),
        "question_answers.csv": csv_text(QUESTION_COLUMNS, questions),
        "consistency.csv": csv_text(CONSISTENCY_COLUMNS, []),
        "cross_variant.csv": csv_text(CROSS_VARIANT_COLUMNS, cross),
        "fdr.csv": csv_text(FDR_COLUMNS, fdr),
        "runs.csv": csv_text(RUNS_COLUMNS, runs),
        "coverage.csv": csv_text(COVERAGE_COLUMNS, coverage),
    }
    for name, text in files.items():
        B.atomic_write_text(results / name, text)
    summary = {
        "schema_version": "1.0",
        "programme": {
            "programme_key": programme["programme_key"],
            "programme_run_id": state["programme_run_id"],
            "status": state["status"],
            "started_at": state.get("started_at"),
            "panel": programme["panel"],
            "programme_sha256": state["programme_sha256"],
            "panel_sha256": state["panel_sha256"],
        },
        "output_policy": policy,
        "experiments": [
            {
                "experiment_key": r["experiment_key"],
                "run_id": r["run_id"],
                "status": r["status"],
            }
            for r in runs
        ],
        "models": models,
        "questions": [
            {
                "faq_id": q["faq_id"],
                "model_key": q["model_key"],
                "verdict_class": q["verdict_class"],
                "verdict_label": q["verdict_label"],
                "row_id": q["row_id"],
            }
            for q in questions
        ],
        "files": {
            name.removesuffix(".csv"): f"results/{name}" for name in sorted(files)
        },
    }
    B.atomic_write_text(results / "summary.json", B.json_text(summary))
    svg_text = answer_matrix_svg(answers, models)
    B.atomic_write_text(report / "figures" / "01_answer_matrix.svg", svg_text)
    md = programme_md(
        programme, state, exps, models, answers, questions, fdr, coverage, runs, policy
    )
    B.atomic_write_text(report / "programme.md", md + "\n")
    data = H.FigureData(
        "results/answers.csv",
        ANSWERS_COLUMNS,
        [[T.cell(r.get(c)) for c in ANSWERS_COLUMNS] for r in answers],
    )
    B.atomic_write_text(
        report / "programme.html",
        H.render_html(
            md, {"01_answer_matrix.svg": svg_text}, {"01_answer_matrix.svg": data}
        ),
    )
    readme = [
        f"# Programme {MD.code(programme['programme_key'])}",
        "",
        f"Status: {state['status']} · programme run {MD.code(state['programme_run_id'][:8])} · panel {MD.code(programme['panel'])}",
        "",
        "## 1. Answers by question",
        "",
        MD.table(
            ["Item", "Model", "Verdict", "Class"],
            [
                [
                    MD.code(q["faq_id"]),
                    MD.code(q["model_key"]),
                    MD.lab(q["verdict_label"]),
                    q["verdict_class"],
                ]
                for q in questions
            ],
        ),
        "",
        "## 2. Files",
        "",
        MD.table(
            ["Path", "Contents"],
            [
                [
                    MD.code("results/answers.csv"),
                    "one row per experiment × model: run and programme zones and labels side by side",
                ],
                [
                    MD.code("results/question_answers.csv"),
                    "one verdict per question and model (D-20)",
                ],
                [MD.code("results/fdr.csv"), "BH per model and claim family (§25)"],
                [MD.code("results/runs.csv"), "one row per experiment"],
                [MD.code("results/plan.csv"), "the plan against the programme budget"],
                [
                    MD.code("results/coverage.csv"),
                    "covered items and unimplemented pairs",
                ],
                [MD.code("results/consistency.csv"), "cross-format checks (§T2.2)"],
                [
                    MD.code("results/cross_variant.csv"),
                    "matched cross-mandate contrasts γ = θ_x − θ_ref (CR-14, exploratory)",
                ],
                [MD.code("report/programme.md"), "the programme report"],
                [MD.code("manifests/"), "the frozen bundle of every experiment"],
            ],
        ),
        "",
        "## 3. Reproduce",
        "",
        f"- {MD.code('financial-harness programme report programme.yaml --resume ' + state['programme_run_id'])}",
        "",
    ]
    B.atomic_write_text(pdir / "README.md", "\n".join(readme))
    logs = pdir / "logs"
    logs.mkdir(exist_ok=True)
    B.atomic_write_text(
        logs / "build_info.json",
        json.dumps(
            {
                "rendered_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "host": platform.node(),
                "python": platform.python_version(),
            },
            indent=2,
        )
        + "\n",
    )
    del S
