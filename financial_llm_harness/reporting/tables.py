"""Results files (spec §32A.2, §32A.3): pure functions of ``db/run.sqlite`` and artefacts.

CSV: RFC 4180, UTF-8 without BOM, LF, one snake_case header row, floats via
``repr``, missing = empty field, booleans ``true``/``false``; the first column
is ``row_id``, the UUIDv5 of the row's natural key.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import ids
from financial_llm_harness.core.canonical_json import sha256_canonical
from financial_llm_harness.statistics.paired import status_class

CONTRAST_ORDER = (
    "B_vs_A",
    "A0_vs_A",
    "PL_vs_A",
    "PCB_vs_PCA",
    "invalid_rate_B_vs_A",
    "B_vs_A_bound_lower",
    "B_vs_A_bound_upper",
    "level_A",
    "level_B",
    "level_A_bound_lower",
    "level_A_bound_upper",
    "level_B_bound_lower",
    "level_B_bound_upper",
)
STATUS_ORDER = (
    "SUCCESS",
    "REFUSAL",
    "EMPTY_RESPONSE",
    "TRUNCATED",
    "CONTENT_FILTERED",
    "INVALID_RESPONSE",
    "PARSE_FAILURE",
    "COMPILE_FAILURE",
    "TEST_FAILURE",
    "RUNTIME_FAILURE",
    "EVALUATOR_UNKNOWN",
    "API_ERROR",
    "RATE_LIMIT",
    "TIMEOUT",
    "SANDBOX_FAILURE",
    "EXCLUDED_CONTEXT",
    "EXCLUDED_MEMORISED",
    "BUDGET_EXHAUSTED",
    "PENDING",
    "SUBMITTED",
)
CONTROL_NAMES = {"A0_vs_A": "null", "PL_vs_A": "placebo", "PCB_vs_PCA": "positive"}

EFFECTS_COLUMNS = (
    "row_id",
    "experiment_key",
    "faq_id",
    "variant_id",
    "model_key",
    "feature_id",
    "analysis_role",
    "contrast",
    "estimand",
    "effect_scale",
    "n_clusters",
    "n_cases",
    "n_blocks_scheduled",
    "n_blocks_matched",
    "n_blocks_invalid",
    "n_blocks_missing",
    "n_blocks_excluded",
    "n_trials_a",
    "n_trials_b",
    "mean_a",
    "mean_b",
    "theta",
    "sd_c",
    "ci_low",
    "ci_high",
    "ci2a_low",
    "ci2a_high",
    "boot_ci_low",
    "boot_ci_high",
    "p_raw",
    "p_tost",
    "p_dev",
    "p_eq",
    "p_dev_adj",
    "p_eq_adj",
    "diagnostic_test",
    "p_diagnostic",
    "delta",
    "margin_low",
    "margin_high",
    "zone",
    "correction_family",
)
LABELS_COLUMNS = (
    "row_id",
    "experiment_key",
    "faq_id",
    "variant_id",
    "model_key",
    "feature_id",
    "mode",
    "mandate_id",
    "oracle_type",
    "oracle_direction",
    "oracle_confidence",
    "zone",
    "sensitivity_label",
    "is_bias",
    "is_normative_failure",
    "attribution_label",
    "attribution_channel",
    "composite_label",
    "gates_failed",
    "reasons",
)
CONTROLS_COLUMNS = (
    "row_id",
    "model_key",
    "control",
    "contrast",
    "n_clusters",
    "n_blocks_matched",
    "theta",
    "ci_low",
    "ci_high",
    "ci2a_low",
    "ci2a_high",
    "control_alpha",
    "margin",
    "method",
    "p_raw",
    "p_tost",
    "expected",
    "observed_label",
    "passed",
)
MISSINGNESS_COLUMNS = (
    "row_id",
    "model_key",
    "arm",
    "status",
    "status_class",
    "n",
    "rate",
)
ATTRIBUTION_COLUMNS = (
    "row_id",
    "model_key",
    "channel",
    "reasoning_mode",
    "n_responsive",
    "n_evaluable_clusters",
    "n_dropped",
    "n_evaluable_blocks",
    "pi_r",
    "ci_low",
    "ci_high",
    "n_a0_responsive",
    "pi_0",
    "f0",
    "f0_ci_low",
    "f0_ci_high",
    "f0_upper_times_n",
    "eligible",
    "label",
    "composite_label",
    "baseline_mention_rate",
    "habitually_mentioned",
)
CLUSTERS_COLUMNS = (
    "row_id",
    "model_key",
    "cluster_id",
    "cluster_ordinal",
    "contrast",
    "n_cases",
    "n_blocks_scheduled",
    "n_blocks_matched",
    "n_blocks_invalid",
    "n_blocks_missing",
    "n_blocks_excluded",
    "mean_a",
    "mean_b",
    "x_c",
    "responsive",
)
TRIALS_COLUMNS = (
    "row_id",
    "model_key",
    "case_idx",
    "cluster_id",
    "arm",
    "paraphrase_idx",
    "repetition",
    "seed",
    "seed_parameter",
    "status",
    "finish_reason",
    "primary_value",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "latency_ms",
    "cost_usd",
    "prompt_hash",
    "response_sha256",
    "serving_fingerprint_sha256",
)
COSTS_COLUMNS = (
    "row_id",
    "model_key",
    "role",
    "provider",
    "region",
    "calls",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cost_usd",
    "probe_cost_usd",
    "pricing_table_version",
)


def cell(v: Any) -> str:
    """One CSV field: floats via repr, booleans true/false, missing empty."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list | tuple):
        return ";".join(str(x) for x in v)
    return str(v)


def csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow([cell(r.get(c)) for c in columns])
    return buf.getvalue()


@dataclass
class RunData:
    """Everything the renderers read, loaded once from the DB and the frozen bundle."""

    run_dir: Path
    run: dict[str, Any]
    manifest: dict[str, Any]
    analysis: list[dict[str, Any]]
    conn: sqlite3.Connection
    cases: dict[int, dict[str, Any]]
    data_dir: Path  # holds db/ and artifacts/: the run directory, or a temporary extraction of the archive

    @property
    def models(self) -> list[dict[str, Any]]:
        return list(self.manifest["sampling_spec"]["models"])

    @property
    def model_keys(self) -> list[str]:
        return [m["model_key"] for m in self.models]

    @property
    def mh(self) -> str:
        return str(self.manifest["manifest_hash"])

    def rows(
        self,
        kind: str | None = None,
        model_key: str | None = None,
        feature_id: str | None = None,
        contrast: str | None = None,
    ) -> list[dict[str, Any]]:
        out = []
        for a in self.analysis:
            if kind is not None and a["analysis_kind"] != kind:
                continue
            if model_key is not None and a["model_key"] != model_key:
                continue
            if feature_id is not None and a["feature_id"] != feature_id:
                continue
            if contrast is not None and a["contrast"] != contrast:
                continue
            out.append(a)
        return out

    def primary_feature(self) -> dict[str, Any]:
        return next(
            f for f in self.manifest["feature_spec"] if f.get("primary") is True
        )

    def primary_row(self, model_key: str) -> dict[str, Any] | None:
        pf = self.primary_feature()["feature_id"]
        for a in self.rows("sensitivity", model_key, pf):
            if a["results"].get("analysis_role") == "primary":
                return a
        return None

    def feature_order(self) -> list[str]:
        return [f["feature_id"] for f in self.manifest["feature_spec"]]


def load_run_data(
    run_dir: Path, conn: sqlite3.Connection, data_dir: Path | None = None
) -> RunData:
    conn.row_factory = sqlite3.Row
    run = dict(conn.execute("SELECT * FROM run").fetchone())
    manifest = B.load_manifest(run_dir / "manifest")
    analysis = []
    for r in conn.execute(
        "SELECT model_key, feature_id, contrast, analysis_kind, results_json, label FROM analysis WHERE manifest_hash = ? ORDER BY analysis_id",
        (manifest["manifest_hash"],),
    ).fetchall():
        d = dict(r)
        d["results"] = json.loads(d.pop("results_json"))
        analysis.append(d)
    cases = {
        c["case_idx"]: c for c in B.read_jsonl(run_dir / "manifest" / B.CASES_FILE)
    }
    return RunData(run_dir, run, manifest, analysis, conn, cases, data_dir or run_dir)


def _rid(*key: Any) -> str:
    return ids.row_id(*key)


def _sorted(rd: RunData, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    morder = {k: i for i, k in enumerate(rd.model_keys)}
    forder = {f: i for i, f in enumerate(rd.feature_order())}
    corder = {c: i for i, c in enumerate(CONTRAST_ORDER)}
    return sorted(
        rows,
        key=lambda a: (
            morder.get(a["model_key"], 99),
            forder.get(a["feature_id"], 99),
            corder.get(a["contrast"], 99),
        ),
    )


def effects_rows(rd: RunData) -> list[dict[str, Any]]:
    m = rd.manifest
    out = []
    for a in _sorted(
        rd,
        [
            x
            for x in rd.analysis
            if x["analysis_kind"] in ("sensitivity", "control", "missingness")
        ],
    ):
        r = a["results"]
        level = a["contrast"].startswith("level_")
        ci = r.get("ci") or [None, None]
        ci2a = r.get("ci2a") or [None, None]
        boot = r.get("boot_ci") or [None, None]
        margin = r.get("margin") or [None, None]
        role = r.get("analysis_role")
        row = {
            "row_id": _rid(
                "effects", rd.mh, a["model_key"], a["feature_id"], a["contrast"]
            ),
            "experiment_key": m["experiment_key"],
            "faq_id": m["faq_id"],
            "variant_id": m["variant_id"],
            "model_key": a["model_key"],
            "feature_id": a["feature_id"],
            "analysis_role": role,
            "contrast": a["contrast"],
            "estimand": r.get(
                "estimand", "arm_level" if level else "mean_paired_difference"
            ),
            "effect_scale": r.get("effect_scale", m["hypothesis"]["effect_scale"]),
            "n_clusters": r.get("n_clusters"),
            "n_cases": r.get("n_cases"),
            "n_blocks_scheduled": r.get("n_blocks_scheduled"),
            "n_blocks_matched": r.get("n_blocks_matched"),
            "n_blocks_invalid": r.get("n_blocks_invalid"),
            "n_blocks_missing": r.get("n_blocks_missing"),
            "n_blocks_excluded": r.get("n_blocks_excluded"),
            "n_trials_a": r.get("n_trials_a"),
            "n_trials_b": None if level else r.get("n_trials_b"),
            "mean_a": None if level else r.get("mean_a"),
            "mean_b": None if level else r.get("mean_b"),
            "theta": r.get("theta"),
            "sd_c": r.get("sd_c"),
            "ci_low": ci[0],
            "ci_high": ci[1],
            "ci2a_low": ci2a[0],
            "ci2a_high": ci2a[1],
            "boot_ci_low": boot[0],
            "boot_ci_high": boot[1],
            "p_raw": None if level else r.get("p_raw"),
            "p_tost": None if level else r.get("p_tost"),
            "p_dev": r.get("p_dev"),
            "p_eq": r.get("p_eq"),
            "p_dev_adj": r.get("p_dev_adj") if role == "secondary" else None,
            "p_eq_adj": r.get("p_eq_adj") if role == "secondary" else None,
            "diagnostic_test": r.get("diagnostic_test"),
            "p_diagnostic": r.get("p_diagnostic"),
            "delta": float(m["hypothesis"]["delta"]),
            "margin_low": margin[0],
            "margin_high": margin[1],
            "zone": r.get("zone")
            if role in ("primary", "secondary", "exploratory")
            else None,
            "correction_family": r.get("correction_family")
            if role in ("primary", "secondary", "exploratory")
            else None,
        }
        if level:
            row["n_trials_a"] = r.get("n_trials_a")
        out.append(row)
    return out


def attribution_rows(rd: RunData) -> list[dict[str, Any]]:
    out = []
    for a in _sorted(rd, rd.rows("attribution")):
        r = a["results"]
        ci = r.get("ci") or [None, None]
        f0ci = r.get("f0_ci") or [None, None]
        out.append(
            {
                "row_id": _rid("attribution", rd.mh, a["model_key"], r["channel"]),
                "model_key": a["model_key"],
                "channel": r["channel"],
                "reasoning_mode": r.get("reasoning_mode"),
                "n_responsive": r.get("n_responsive"),
                "n_evaluable_clusters": r.get("n_evaluable_clusters"),
                "n_dropped": r.get("n_dropped"),
                "n_evaluable_blocks": r.get("n_evaluable_blocks"),
                "pi_r": r.get("pi_r"),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "n_a0_responsive": r.get("n_a0_responsive"),
                "pi_0": r.get("pi_0"),
                "f0": r.get("f0"),
                "f0_ci_low": f0ci[0],
                "f0_ci_high": f0ci[1],
                "f0_upper_times_n": r.get("f0_upper_times_n"),
                "eligible": r.get("eligible"),
                "label": r.get("label_display"),
                "composite_label": r.get("composite_label"),
                "baseline_mention_rate": r.get("baseline_mention_rate"),
                "habitually_mentioned": r.get("habitually_mentioned"),
            }
        )
    return out


def headline_attribution(rd: RunData, model_key: str) -> dict[str, Any] | None:
    """The headline channel: the free-text channel when evaluated with a calibrated evaluator, else factors (§24.3)."""
    rows = {
        a["results"]["channel"]: a["results"] for a in rd.rows("attribution", model_key)
    }
    mode = rd.manifest["response_spec"]["reasoning_mode"]
    free = (
        "reasoning"
        if mode == "reasoning_before"
        else ("rationale" if mode == "rationale_after" else None)
    )
    calibrated = {
        e["feature_id"]
        for e in rd.manifest.get("evaluators") or []
        if e.get("calibration_ref")
    }
    if free and free in rows and f"attributes_concept_{free}" in calibrated:
        return rows[free]
    if "factors" in rows:
        return rows["factors"]
    return None


def oracle_direction(oracle: Mapping[str, Any]) -> str:
    if oracle.get("type") == "DIRECTIONAL":
        return "positive" if oracle.get("sign") == 1 else "negative"
    if oracle.get("type") in ("INVARIANCE", "LEVEL"):
        return "none"
    return "two_sided"


def labels_rows(rd: RunData) -> list[dict[str, Any]]:
    m = rd.manifest
    out = []
    seen: set[tuple[str, str]] = set()
    for a in _sorted(rd, rd.rows("sensitivity")):
        key = (a["model_key"], a["feature_id"])
        r = a["results"]
        if key in seen:
            continue
        if r.get("analysis_role") != "primary" and any(
            x["results"].get("analysis_role") == "primary"
            for x in rd.rows("sensitivity", a["model_key"], a["feature_id"])
        ):
            continue
        seen.add(key)
        primary = r.get("analysis_role") == "primary"
        oracle = r.get("oracle") or {}
        attr = headline_attribution(rd, a["model_key"]) if primary else None
        out.append(
            {
                "row_id": _rid("labels", rd.mh, a["model_key"], a["feature_id"]),
                "experiment_key": m["experiment_key"],
                "faq_id": m["faq_id"],
                "variant_id": m["variant_id"],
                "model_key": a["model_key"],
                "feature_id": a["feature_id"],
                "mode": m["mode"],
                "mandate_id": m["oracle_spec"].get("mandate_id") if primary else None,
                "oracle_type": oracle.get("type"),
                "oracle_direction": oracle_direction(oracle),
                "oracle_confidence": oracle.get("confidence"),
                "zone": r.get("zone"),
                "sensitivity_label": r.get("label"),
                "is_bias": bool(r.get("is_bias", False)),
                "is_normative_failure": bool(r.get("is_normative_failure", False)),
                "attribution_label": (attr or {}).get("label_display"),
                "attribution_channel": (attr or {}).get("channel")
                if attr and attr.get("label")
                else None,
                "composite_label": (attr or {}).get("composite_label"),
                "gates_failed": list(r.get("gates_failed") or []),
                "reasons": list(r.get("reasons") or []),
            }
        )
    return out


def controls_rows(rd: RunData) -> list[dict[str, Any]]:
    out = []
    for a in _sorted(rd, rd.rows("control")):
        r = a["results"]
        ci = r.get("ci") or [None, None]
        ci2a = r.get("ci2a") or [None, None]
        out.append(
            {
                "row_id": _rid("controls", rd.mh, a["model_key"], r["control"]),
                "model_key": a["model_key"],
                "control": r["control"],
                "contrast": a["contrast"],
                "n_clusters": r.get("n_clusters"),
                "n_blocks_matched": r.get("n_blocks_matched"),
                "theta": r.get("theta"),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "ci2a_low": ci2a[0],
                "ci2a_high": ci2a[1],
                "control_alpha": r.get("control_alpha"),
                "margin": r.get("margin_value"),
                "method": r.get("method"),
                "p_raw": r.get("p_raw"),
                "p_tost": r.get("p_tost") if r["control"] != "positive" else None,
                "expected": r.get("expected"),
                "observed_label": r.get("observed_label"),
                "passed": r.get("passed"),
            }
        )
    return out


def missingness_rows(rd: RunData) -> list[dict[str, Any]]:
    counts = defaultdict(int)
    totals = defaultdict(int)
    for r in rd.conn.execute(
        "SELECT model_key, arm, status, COUNT(*) FROM trial GROUP BY model_key, arm, status"
    ).fetchall():
        counts[(r[0], r[1], r[2])] += r[3]
        totals[(r[0], r[1])] += r[3]
    unknown = defaultdict(int)
    pf = rd.primary_feature()["feature_id"]
    for r in rd.conn.execute(
        "SELECT t.model_key, t.arm, COUNT(*) FROM observation o JOIN trial t ON t.trial_id = o.trial_id WHERE o.status = 'UNKNOWN' AND o.feature_id = ? GROUP BY t.model_key, t.arm",
        (pf,),
    ).fetchall():
        unknown[(r[0], r[1])] = r[2]
    arm_order = {a: i for i, a in enumerate(rd.manifest["arms"])}
    out = []
    for mk in rd.model_keys:
        for arm in sorted(
            {k[1] for k in totals if k[0] == mk}, key=lambda a: arm_order.get(a, 99)
        ):
            keys = [(mk, arm, s) for s in STATUS_ORDER if counts.get((mk, arm, s))]
            for k in keys:
                n = counts[k]
                out.append(
                    {
                        "row_id": _rid("missingness", rd.mh, mk, arm, k[2]),
                        "model_key": mk,
                        "arm": arm,
                        "status": k[2],
                        "status_class": status_class(k[2]),
                        "n": n,
                        "rate": n / totals[(mk, arm)],
                    }
                )
            if unknown.get((mk, arm)):
                n = unknown[(mk, arm)]
                out.append(
                    {
                        "row_id": _rid(
                            "missingness", rd.mh, mk, arm, "EVALUATOR_UNKNOWN"
                        ),
                        "model_key": mk,
                        "arm": arm,
                        "status": "EVALUATOR_UNKNOWN",
                        "status_class": "missing",
                        "n": n,
                        "rate": n / totals[(mk, arm)],
                    }
                )
    return out


def clusters_rows(rd: RunData) -> list[dict[str, Any]]:
    dh = rd.manifest["design_hash"]
    pf = rd.primary_feature()["feature_id"]
    delta = float(rd.manifest["hypothesis"]["delta"])
    out = []
    for mk in rd.model_keys:
        prim = rd.primary_row(mk)
        theta = (prim or {}).get("results", {}).get("theta")
        eligible = (prim or {}).get("results", {}).get("label") in (
            "SPURIOUS_SENSITIVITY",
            "JUSTIFIED_SENSITIVITY",
            "PERVERSE_SENSITIVITY",
            "OVER_SENSITIVITY",
            "SENSITIVE",
        )
        e = 0 if not theta else (1 if theta > 0 else -1)
        rows = [
            a
            for a in rd.rows(None, mk, pf)
            if a["analysis_kind"] in ("sensitivity", "control")
        ]
        by_contrast = {
            a["contrast"]: a
            for a in rows
            if a["results"].get("analysis_role") in ("primary", "control")
        }
        for contrast in CONTRAST_ORDER:
            a = by_contrast.get(contrast)
            if a is None:
                continue
            for c in a["results"].get("clusters") or []:
                responsive = None
                if (
                    eligible
                    and contrast in ("B_vs_A", "A0_vs_A")
                    and c.get("x_c") is not None
                ):
                    responsive = e * c["x_c"] >= delta
                out.append(
                    {
                        "row_id": _rid(
                            "clusters", rd.mh, mk, c["cluster_ordinal"], contrast
                        ),
                        "model_key": mk,
                        "cluster_id": ids.cluster_id(dh, c["cluster_ordinal"]),
                        "cluster_ordinal": c["cluster_ordinal"],
                        "contrast": contrast,
                        "n_cases": c.get("n_cases"),
                        "n_blocks_scheduled": c.get("n_blocks_scheduled"),
                        "n_blocks_matched": c.get("n_blocks_matched"),
                        "n_blocks_invalid": c.get("n_blocks_invalid"),
                        "n_blocks_missing": c.get("n_blocks_missing"),
                        "n_blocks_excluded": c.get("n_blocks_excluded"),
                        "mean_a": None
                        if contrast.startswith("level_")
                        else c.get("mean_a"),
                        "mean_b": None
                        if contrast.startswith("level_")
                        else c.get("mean_b"),
                        "x_c": c.get("x_c"),
                        "responsive": responsive,
                    }
                )
    morder = {k: i for i, k in enumerate(rd.model_keys)}
    corder = {c: i for i, c in enumerate(CONTRAST_ORDER)}
    out.sort(
        key=lambda r: (
            morder[r["model_key"]],
            r["cluster_ordinal"],
            corder[r["contrast"]],
        )
    )
    return out


def trials_rows(rd: RunData) -> list[dict[str, Any]]:
    dh = rd.manifest["design_hash"]
    pf = rd.primary_feature()["feature_id"]
    caps = {m["model_key"]: m["capabilities"] for m in rd.models}
    obs = {
        r[0]: r[1]
        for r in rd.conn.execute(
            "SELECT trial_id, value_json FROM observation WHERE feature_id = ?", (pf,)
        ).fetchall()
    }
    q = rd.conn.execute(
        "SELECT t.trial_id, t.model_key, t.case_idx, t.arm, t.paraphrase_idx, t.repetition, t.seed, t.status, "
        "r.finish_reason, r.input_tokens, r.output_tokens, r.reasoning_tokens, r.latency_ms, r.cost_usd, rd.prompt_hash, "
        "r.text_artifact, r.serving_fingerprint_json FROM trial t JOIN rendering rd ON rd.rendering_id = t.rendering_id "
        "LEFT JOIN response r ON r.attempt_id = t.used_attempt_id"
    ).fetchall()
    arm_order = {a: i for i, a in enumerate(rd.manifest["arms"])}
    morder = {k: i for i, k in enumerate(rd.model_keys)}
    out = []
    for t in q:
        value = obs.get(t["trial_id"])
        pv = json.loads(value) if value is not None else None
        fp = t["serving_fingerprint_json"]
        out.append(
            {
                "row_id": t["trial_id"],
                "model_key": t["model_key"],
                "case_idx": t["case_idx"],
                "cluster_id": ids.cluster_id(
                    dh, rd.cases[t["case_idx"]]["cluster_ordinal"]
                ),
                "arm": t["arm"],
                "paraphrase_idx": t["paraphrase_idx"],
                "repetition": t["repetition"],
                "seed": t["seed"],
                "seed_parameter": caps[t["model_key"]]["seed_parameter"],
                "status": t["status"],
                "finish_reason": t["finish_reason"],
                "primary_value": pv,
                "input_tokens": t["input_tokens"],
                "output_tokens": t["output_tokens"],
                "reasoning_tokens": t["reasoning_tokens"],
                "latency_ms": t["latency_ms"],
                "cost_usd": t["cost_usd"],
                "prompt_hash": t["prompt_hash"],
                "response_sha256": t["text_artifact"],
                "serving_fingerprint_sha256": sha256_canonical(json.loads(fp))
                if fp
                else None,
            }
        )
    out.sort(
        key=lambda r: (
            morder[r["model_key"]],
            r["case_idx"],
            arm_order.get(r["arm"], 99),
            r["paraphrase_idx"],
            r["repetition"],
        )
    )
    return out


def costs_rows(rd: RunData) -> list[dict[str, Any]]:
    version = rd.manifest["budget"].get("pricing_table_version")
    out = []
    for m in rd.models:
        r = rd.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(r.input_tokens), 0), COALESCE(SUM(r.output_tokens), 0), COALESCE(SUM(r.reasoning_tokens), 0), "
            "COALESCE(SUM(r.cost_usd), 0.0) FROM response r JOIN trial t ON t.trial_id = r.trial_id WHERE t.model_key = ?",
            (m["model_key"],),
        ).fetchone()
        out.append(
            {
                "row_id": _rid("costs", rd.mh, m["model_key"], "target"),
                "model_key": m["model_key"],
                "role": "target",
                "provider": m["provider"],
                "region": m.get("region"),
                "calls": int(r[0]),
                "input_tokens": int(r[1]),
                "output_tokens": int(r[2]),
                "reasoning_tokens": int(r[3]),
                "cost_usd": float(r[4]),
                "probe_cost_usd": (m.get("probe_ref") or {}).get("cost_usd", 0.0)
                if m.get("probe_ref")
                else 0.0,
                "pricing_table_version": version,
            }
        )
    for ev in sorted({e["model_key"] for e in rd.manifest.get("evaluators") or []}):
        r = rd.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), COALESCE(SUM(cost_usd), 0.0) "
            "FROM eval_call WHERE model_key = ? AND (status = 'complete' OR raw_artifact IS NOT NULL)",
            (ev,),
        ).fetchone()
        provider = ev.split("--", 1)[0]
        out.append(
            {
                "row_id": _rid("costs", rd.mh, ev, "evaluator"),
                "model_key": ev,
                "role": "evaluator",
                "provider": provider,
                "region": None,
                "calls": int(r[0]),
                "input_tokens": int(r[1]),
                "output_tokens": int(r[2]),
                "reasoning_tokens": 0,
                "cost_usd": float(r[3]),
                "probe_cost_usd": 0.0,
                "pricing_table_version": version,
            }
        )
    return out
