"""``results/summary.json``: the one-file machine answer (spec §32A.3)."""

from __future__ import annotations

import json
from typing import Any

from financial_llm_harness.reporting.tables import (
    RunData,
    controls_rows,
    costs_rows,
    effects_rows,
    headline_attribution,
    labels_rows,
)

SUMMARY_SCHEMA_VERSION = "1.0"


def intervention_text(manifest: dict[str, Any]) -> str:
    spec = manifest.get("intervention_spec")
    if not spec:
        return "none (correctness experiment)"
    a, b = spec.get("baseline"), spec.get("counterfactual")
    if spec.get("form") == "shift":
        d = (
            spec.get("delta")
            if spec.get("delta") is not None
            else spec.get("relative_delta")
        )
        text = (
            f"shift by {d} ({'relative' if spec.get('delta') is None else 'absolute'})"
        )
    else:
        text = f"{a} (A) → {b} (B)"
    if spec.get("via"):
        text += f" via {spec['via']}"
    return f"{spec['node']}: {text}"


def build_summary(rd: RunData) -> dict[str, Any]:
    m = rd.manifest
    run = rd.run
    eff = {
        (r["model_key"], r["feature_id"], r["contrast"]): r for r in effects_rows(rd)
    }
    lab = {(r["model_key"], r["feature_id"]): r for r in labels_rows(rd)}
    ctl = {(r["model_key"], r["control"]): r for r in controls_rows(rd)}
    cost = {(r["model_key"], r["role"]): r for r in costs_rows(rd)}
    pf = rd.primary_feature()["feature_id"]
    h = m["hypothesis"]
    contrast = f"level_{h['level_arm']}" if h["estimand"] == "arm_level" else "B_vs_A"
    report = json.loads(
        (rd.run_dir / "manifest" / "corpus_report.json").read_text(encoding="utf-8")
    )
    power = report.get("power", {})
    drift_rows = rd.conn.execute(
        "SELECT kind, payload_json FROM event WHERE kind IN ('capability_drift', 'serving_drift')"
    ).fetchall()
    drift: dict[str, set[str]] = {}
    for k, p in drift_rows:
        drift.setdefault(json.loads(p).get("model_key"), set()).add(k)
    models = []
    for mm in rd.models:
        mk = mm["model_key"]
        e = eff.get((mk, pf, contrast)) or {}
        lb = lab.get((mk, pf)) or {}
        prim = rd.primary_row(mk) or {"results": {}}
        res = prim["results"]
        ha = headline_attribution(rd, mk)
        rates = (res.get("missingness") or {}).get("invalid_rates") or {}
        controls = {}
        for name in ("null", "placebo", "positive"):
            c = ctl.get((mk, name))
            if c is not None:
                controls[name] = (
                    "PASSED"
                    if c["passed"]
                    else ("FAILED" if c["passed"] is False else "NO_DATA")
                )
        c = cost.get((mk, "target")) or {}
        models.append(
            {
                "model_key": mk,
                "provider": mm["provider"],
                "region": mm.get("region"),
                "serving": mm["capabilities"].get("serving"),
                "primary": {
                    "theta": e.get("theta"),
                    "ci": [e.get("ci_low"), e.get("ci_high")],
                    "ci2a": [e.get("ci2a_low"), e.get("ci2a_high")],
                    "p_dev": e.get("p_dev"),
                    "p_eq": e.get("p_eq"),
                    "zone": e.get("zone"),
                    "sensitivity_label": lb.get("sensitivity_label"),
                    "is_bias": lb.get("is_bias", False),
                    "is_normative_failure": lb.get("is_normative_failure", False),
                    "attribution_label": (ha or {}).get("label_display"),
                    "composite_label": (ha or {}).get("composite_label"),
                    "gates_failed": list(lb.get("gates_failed") or []),
                },
                "controls": controls,
                "missingness": {
                    "invalid_rate_a": rates.get("A")
                    if contrast == "B_vs_A"
                    else rates.get(h.get("level_arm")),
                    "invalid_rate_b": rates.get("B") if contrast == "B_vs_A" else None,
                    "bounds_change_zone": bool(
                        (res.get("missingness") or {}).get("bounds_change", False)
                    ),
                },
                "drift": {
                    "capability": "capability_drift" in drift.get(mk, set()),
                    "serving": "serving_drift" in drift.get(mk, set()),
                },
                "cost": {
                    "calls": c.get("calls", 0),
                    "cost_usd": c.get("cost_usd", 0.0),
                },
                "rows": {"effect": e.get("row_id"), "label": lb.get("row_id")},
            }
        )
    policy = (
        json.loads(run["output_policy_json"])
        if run.get("output_policy_json")
        else {"publishable": True, "reason": "synthetic"}
    )
    out: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run": {
            "run_id": run["run_id"],
            "experiment_key": m["experiment_key"],
            "experiment_id": m["experiment_id"],
            "design_hash": m["design_hash"],
            "manifest_hash": m["manifest_hash"],
            "faq_id": m["faq_id"],
            "variant_id": m["variant_id"],
            "intervention_id": m["intervention_id"],
            "mode": m["mode"],
            "status": run["status"],
            "started_at": run.get("started_at"),
            "completed_at": run.get("completed_at"),
            "data_classification": m["data_classification"],
            "code_ref": json.loads(run["code_ref_json"])
            if run.get("code_ref_json")
            else m["code_ref"],
        },
        "output_policy": policy,
    }
    if h["correction_family"] == "programme" and m["mode"] == "confirmatory":
        out["programme_adjustment"] = "pending"
    out["design"] = {
        "concept": m["concept"],
        "mandate_id": m["mandate"]["id"],
        "intervention": intervention_text(m),
        "primary_feature": pf,
        "estimand": h["estimand"],
        "effect_scale": h["effect_scale"],
        "delta": float(h["delta"]),
        "alpha": float(h["alpha"]),
        "oracle_type": m["oracle_spec"]["type"],
        "oracle_confidence": m["oracle_spec"]["confidence"],
        "n_cases": int(m["scenario_spec"]["n_cases"]),
        "n_clusters": power.get("n_clusters"),
        "n_clusters_required": (power.get("primary") or {}).get("n_required"),
        "subset_n": m["controls"].get("subset_n"),
        "planning": (power.get("primary") or {}).get("source"),
        "planning_variance": (power.get("primary") or {}).get("variance"),
        "P": int(m["paraphrase_spec"]["P"]),
        "R": int(m["sampling_spec"]["R"]),
        "control_alpha": float(m["controls"].get("control_alpha", h["alpha"])),
        "null_margin": m["controls"].get("null_margin"),
        "placebo_margin": m["controls"].get("placebo_margin"),
        "seeds": {k: v for k, v in m["seeds"].items() if k != "trial_seed_rule"},
    }
    out["diagnostics"] = {
        "acceptance_rate": report.get("acceptance_rate"),
        "attempts": report.get("attempts"),
        "rejections": report.get("rejections"),
        "largest_cluster": (report.get("near_duplicates") or {}).get("largest_cluster"),
        "structured_edges": (report.get("near_duplicates") or {}).get(
            "structured_edges"
        ),
        "text_edges": (report.get("near_duplicates") or {}).get("text_edges"),
    }
    out["models"] = models
    out["files"] = {
        "effects": "results/effects.csv",
        "labels": "results/labels.csv",
        "controls": "results/controls.csv",
        "missingness": "results/missingness.csv",
        "attribution": "results/attribution.csv",
        "clusters": "results/clusters.csv",
        "trials": "results/trials.csv",
        "costs": "results/costs.csv",
        "report_md": "report/report.md",
        "report_html": "report/report.html",
    }
    return out
