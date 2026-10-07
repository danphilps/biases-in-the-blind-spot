"""L9 sensitivity and L11 attribution analysis (spec §22–§25, §31A).

Writes one ``analysis`` row per model × feature × contrast × analysis_kind,
keyed by ``analysis_id`` (§29), with every number a report needs in
``results_json``. Controls are classified first, without gates; the §24.2
gates then consume their results.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from financial_llm_harness.core import ids
from financial_llm_harness.core.ids import uint64
from financial_llm_harness.evaluation.rules import numeric_value
from financial_llm_harness.oracles import (
    ATTRIBUTION_ELIGIBLE,
    Oracle,
    attribution_label,
    classify_equivalence_control,
    classify_positive_control,
    classify_primary,
    composite_label,
    meaningful_lower_bound,
    oracle_row_label,
    zone,
)
from financial_llm_harness.statistics import ANALYSIS_VERSION
from financial_llm_harness.statistics.correction import benjamini_hochberg
from financial_llm_harness.statistics.equivalence import (
    OracleStats,
    contrast_stats,
    level_stats,
)
from financial_llm_harness.statistics.paired import (
    ContrastData,
    TrialObs,
    arm_level,
    bootstrap_ci,
    bootstrap_rng,
    index_blocks,
    jsonable,
    mcnemar_exact,
    paired_contrast,
    status_class,
    wilcoxon_pratt,
)
from financial_llm_harness.statistics.power import feature_range

MISSING_SHARE_INCOMPLETE = 0.02
ATTR_MIN_CLUSTERS = 20
BASELINE_SAMPLE = 200
HABITUAL = 0.30


@dataclass
class ModelData:
    model_key: str
    trials: list[dict[str, Any]]
    obs: dict[str, dict[str, Any]]  # trial_id → feature_id → value


def _features(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["feature_id"]: dict(f) for f in manifest["feature_spec"]}


def _primary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return next(dict(f) for f in manifest["feature_spec"] if f.get("primary") is True)


def load_model_data(ctx: Any, model_key: str) -> ModelData:
    trials = [
        dict(r)
        for r in ctx.db.execute(
            "SELECT trial_id, arm, case_idx, paraphrase_idx, repetition, status FROM trial WHERE model_key = ? ORDER BY case_idx, arm, paraphrase_idx, repetition",
            (model_key,),
        ).fetchall()
    ]
    obs: dict[str, dict[str, Any]] = defaultdict(dict)
    for r in ctx.db.execute(
        "SELECT o.trial_id, o.feature_id, o.value_json, o.status FROM observation o JOIN trial t ON t.trial_id = o.trial_id WHERE t.model_key = ?",
        (model_key,),
    ).fetchall():
        value = json.loads(r["value_json"]) if r["value_json"] is not None else None
        obs[r["trial_id"]][r["feature_id"]] = (
            "UNKNOWN" if r["status"] == "UNKNOWN" else value
        )
    ctx.db.commit()
    return ModelData(model_key, trials, dict(obs))


def trial_obs(md: ModelData, feature_id: str) -> list[TrialObs]:
    out = []
    for t in md.trials:
        cls = status_class(t["status"])
        value = None
        if cls == "valid":
            raw = md.obs.get(t["trial_id"], {}).get(feature_id)
            value = numeric_value({}, raw)
        out.append(
            TrialObs(
                t["case_idx"],
                t["arm"],
                t["paraphrase_idx"],
                t["repetition"],
                cls,
                value,
            )
        )
    return out


def case_cluster_map(ctx: Any) -> dict[int, int]:
    return {ci: c["cluster_ordinal"] for ci, c in ctx.cases.items()}


def truth_map(ctx: Any) -> dict[int, float] | None:
    out = {}
    for ci, c in ctx.cases.items():
        t = c.get("truth")
        if t and t.get("d_star") is not None:
            out[ci] = float(t["d_star"])
    return out or None


def _stats_dict(st: OracleStats) -> dict[str, Any]:
    return {
        "n_clusters": st.n,
        "theta": st.theta,
        "sd_c": st.sd_c,
        "ci": list(st.ci) if st.ci else None,
        "ci2a": list(st.ci2a) if st.ci2a else None,
        "p_raw": st.p_raw,
        "p_tost": st.p_tost,
        "margin": list(st.margin),
        "p_dev": st.p_dev,
        "p_eq": st.p_eq,
        "test_dev": st.test_dev,
        "test_eq": st.test_eq,
        "p_under": st.p_under,
        "p_over": st.p_over,
        "p_short": st.p_short,
        "p_meets": st.p_meets,
        "p_below": st.p_below,
    }


def _clusters_list(cd: ContrastData) -> list[dict[str, Any]]:
    return [
        {
            "cluster_ordinal": c.cluster_ordinal,
            "n_cases": c.n_cases,
            "n_blocks_scheduled": c.n_blocks_scheduled,
            "n_blocks_matched": c.n_blocks_matched,
            "n_blocks_invalid": c.n_blocks_invalid,
            "n_blocks_missing": c.n_blocks_missing,
            "n_blocks_excluded": c.n_blocks_excluded,
            "mean_a": c.mean_a,
            "mean_b": c.mean_b,
            "x_c": c.x_c,
        }
        for c in cd.clusters
    ]


def _counts(cd: ContrastData) -> dict[str, Any]:
    out: dict[str, Any] = dict(cd.totals())
    out["n_cases"] = sum(c.n_cases for c in cd.clusters if c.x_c is not None)
    out["n_trials_a"] = cd.n_trials_y
    out["n_trials_b"] = cd.n_trials_x
    out["mean_a"] = cd.mean_a
    out["mean_b"] = cd.mean_b
    return out


def _diagnostics(
    ctx: Any,
    model_key: str,
    feature: Mapping[str, Any],
    contrast: str,
    x: np.ndarray,
    alpha: float,
    single_case_clusters: bool,
) -> dict[str, Any]:
    st = ctx.manifest["statistical_spec"]
    rng = bootstrap_rng(
        int(ctx.manifest["seeds"]["bootstrap"]),
        model_key,
        feature["feature_id"],
        contrast,
    )
    boot = bootstrap_ci(x, alpha, int(st["bootstrap_B"]), rng)
    test, p = None, None
    P = int(ctx.manifest["paraphrase_spec"]["P"])
    R = int(ctx.manifest["sampling_spec"]["R"])
    if not contrast.startswith("level_") and len(x):
        if (
            feature.get("type") == "binary"
            and P == 1
            and R == 1
            and single_case_clusters
        ):
            test, p = "mcnemar_exact", mcnemar_exact(x)
        elif feature.get("type") == "ordinal":
            test, p = "wilcoxon_pratt", wilcoxon_pratt(x)
    return {
        "boot_ci": list(boot) if boot else None,
        "diagnostic_test": test,
        "p_diagnostic": p,
    }


def _write(
    ctx: Any,
    model_key: str,
    feature_id: str,
    contrast: str,
    kind: str,
    results: Mapping[str, Any],
    label: str | None,
) -> None:
    mh = ctx.manifest["manifest_hash"]
    aid = ids.analysis_id(mh, model_key, feature_id, contrast, kind, ANALYSIS_VERSION)
    ctx.db.execute(
        "INSERT OR REPLACE INTO analysis(analysis_id, manifest_hash, model_key, feature_id, contrast, analysis_kind, analysis_version, results_json, label, code_sha, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            aid,
            mh,
            model_key,
            feature_id,
            contrast,
            kind,
            ANALYSIS_VERSION,
            json.dumps(jsonable(dict(results)), sort_keys=True, allow_nan=False),
            label,
            ctx.code_ref.harness_tree_sha256,
            ctx.db.now(),
        ),
    )


def model_incomplete_set(ctx: Any) -> set[str]:
    """Models with > 2% missing trials, capability drift or SERVING_DRIFT (§23.4)."""
    out: set[str] = set()
    for m in ctx.manifest["sampling_spec"]["models"]:
        mk = m["model_key"]
        rows = ctx.db.execute(
            "SELECT status, COUNT(*) FROM trial WHERE model_key = ? GROUP BY status",
            (mk,),
        ).fetchall()
        total = sum(r[1] for r in rows)
        missing = sum(r[1] for r in rows if status_class(r[0]) == "missing")
        if total and missing / total > MISSING_SHARE_INCOMPLETE:
            out.add(mk)
    for r in ctx.db.execute(
        "SELECT kind, payload_json FROM event WHERE kind IN ('capability_drift', 'serving_drift')"
    ).fetchall():
        out.add(json.loads(r[1]).get("model_key"))
    ctx.db.commit()
    return {m for m in out if m}


def _impute(feature: Mapping[str, Any], lower: bool) -> Any:
    lo, hi = feature_range(feature)

    def f(o: TrialObs, side: str) -> float:
        # lower bound: invalid X (B) → min, invalid Y (A) → max; upper: the reverse
        if side == "x":
            return lo if lower else hi
        return hi if lower else lo

    return f


def _level_stats_bounds(
    obs: Sequence[TrialObs],
    case_cluster: Mapping[int, int],
    arm: str,
    alpha: float,
    tau: float,
    bound: float,
) -> tuple[ContrastData, OracleStats]:
    cd = arm_level(obs, case_cluster, arm, impute_value=bound)
    return cd, level_stats(cd.x, alpha, tau)


def _invalid_rates(
    obs: Sequence[TrialObs], arms: Sequence[str]
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for a in arms:
        rel = [o for o in obs if o.arm == a and o.cls in ("valid", "invalid")]
        out[a] = (sum(1 for o in rel if o.cls == "invalid") / len(rel)) if rel else None
    return out


def analyse_sensitivity(ctx: Any) -> None:
    """L9: sensitivity, control, missingness and diagnostic rows for every model."""
    manifest = ctx.manifest
    h = manifest["hypothesis"]
    alpha = float(h["alpha"])
    delta = float(h["delta"])
    feats = _features(manifest)
    primary = _primary(manifest)
    oracle_spec = manifest["oracle_spec"]
    oracle = Oracle.from_spec(oracle_spec)
    controls = manifest["controls"]
    calpha = float(controls.get("control_alpha", alpha))
    case_cluster = case_cluster_map(ctx)
    truth = truth_map(ctx) if h["estimand"] == "paired_difference_vs_truth" else None
    arms = manifest["arms"]
    mode = manifest["mode"]
    incomplete = model_incomplete_set(ctx)
    single_case = all(
        sum(1 for c in ctx.cases.values() if c["cluster_ordinal"] == o) == 1
        for o in set(case_cluster.values())
    )
    semantic_primary = primary.get("extractor") in (
        "semantic_verbalisation",
        "semantic_inclusion",
    )
    uncalibrated = False
    if semantic_primary:
        evs = [
            e
            for e in manifest.get("evaluators") or []
            if e["feature_id"] == primary["feature_id"]
        ]
        uncalibrated = (
            not evs
            or not evs[0].get("calibration_ref")
            or float(evs[0]["calibration_ref"].get("kappa", 0)) < 0.70
        )
    for model in manifest["sampling_spec"]["models"]:
        mk = model["model_key"]
        md = load_model_data(ctx, mk)
        obs = trial_obs(md, primary["feature_id"])
        blocks = index_blocks(obs)
        # --- controls first, without gates (§22) ---------------------------------------
        control_results = []
        if "A0" in arms:
            cd = paired_contrast(blocks, case_cluster, "A0", "A")
            st = contrast_stats(
                cd.x, calpha, "DISCOVERY", delta, margin=float(controls["null_margin"])
            )
            cr = classify_equivalence_control(
                "null", "A0_vs_A", st, float(controls["null_margin"])
            )
            control_results.append(cr)
            _write_control(ctx, mk, primary, "A0_vs_A", cd, st, cr, calpha, single_case)
        if "PL" in arms:
            cd = paired_contrast(blocks, case_cluster, "PL", "A")
            st = contrast_stats(
                cd.x,
                calpha,
                "DISCOVERY",
                delta,
                margin=float(controls["placebo_margin"]),
            )
            cr = classify_equivalence_control(
                "placebo", "PL_vs_A", st, float(controls["placebo_margin"])
            )
            control_results.append(cr)
            _write_control(ctx, mk, primary, "PL_vs_A", cd, st, cr, calpha, single_case)
        if "PCB" in arms:
            pos = controls["positive"]
            po = pos["oracle"]
            cd = paired_contrast(blocks, case_cluster, "PCB", "PCA")
            st = contrast_stats(
                cd.x,
                calpha,
                str(po["type"]),
                delta,
                sign=po.get("sign"),
                theta_star=None,
                tol=po.get("tol"),
            )
            cr = classify_positive_control(po, st, delta)
            control_results.append(cr)
            _write_control(
                ctx, mk, primary, "PCB_vs_PCA", cd, st, cr, calpha, single_case
            )
        # --- primary (§23.2, §23.3) ----------------------------------------------------------
        if h["estimand"] == "arm_level":
            level_arm = str(h["level_arm"])
            contrast = f"level_{level_arm}"
            cd = arm_level(obs, case_cluster, level_arm)
            st = level_stats(cd.x, alpha, float(oracle_spec["tau"]))
            rates = _invalid_rates(obs, [level_arm])
            if mode != "confirmatory":
                # discovery: the DISCOVERY row reads the deviation of λ from τ
                disc = contrast_stats(
                    cd.x - float(oracle_spec["tau"]), alpha, "DISCOVERY", delta
                )
                st = OracleStats(
                    st.n,
                    st.theta,
                    st.sd_c,
                    st.ci,
                    st.ci2a,
                    disc.p_raw,
                    disc.p_tost,
                    st.margin,
                    st.p_dev,
                    st.p_eq,
                    st.test_dev,
                    st.test_eq,
                    p_meets=st.p_meets,
                    p_below=st.p_below,
                )
                disc_st = disc
            else:
                disc_st = None
        else:
            contrast = "B_vs_A"
            cd = paired_contrast(blocks, case_cluster, "B", "A", truth=truth)
            st = contrast_stats(
                cd.x,
                alpha,
                oracle.effective_type,
                delta,
                sign=oracle.sign,
                theta_star=oracle.theta_star,
                tol=oracle.tol,
            )
            rates = _invalid_rates(obs, ["A", "B"])
            disc_st = None
        # --- missingness (§23.4) -------------------------------------------------------------
        md_spec = h["missing_data"]
        max_invalid = float(md_spec["max_invalid_rate"])
        invalid_trigger = any(r is not None and r > max_invalid for r in rates.values())
        bounds_change = False
        base_zone, base_label = oracle_row_label(
            oracle.effective_type,
            st,
            delta,
            oracle.sign,
            oracle.theta_star,
            oracle.tol,
            oracle.tau,
        )
        if h["estimand"] == "arm_level":
            level_arm = str(h["level_arm"])
            for side, value in (("lower", 0.0), ("upper", 1.0)):
                bcd, bst = _level_stats_bounds(
                    obs,
                    case_cluster,
                    level_arm,
                    alpha,
                    float(oracle_spec["tau"]),
                    value,
                )
                _, blabel = oracle_row_label(
                    "LEVEL", bst, delta, tau=float(oracle_spec["tau"])
                )
                if md_spec.get("bounds", True):
                    bounds_change |= blabel != base_label
                    _write_missingness(
                        ctx,
                        mk,
                        primary,
                        f"level_{level_arm}_bound_{side}",
                        bcd,
                        bst,
                        alpha,
                        delta,
                        {"label": blabel},
                    )
        else:
            for side in ("lower", "upper"):
                bcd = paired_contrast(
                    blocks,
                    case_cluster,
                    "B",
                    "A",
                    truth=truth,
                    impute=_impute(primary, side == "lower"),
                )
                bst = contrast_stats(
                    bcd.x,
                    alpha,
                    oracle.effective_type,
                    delta,
                    sign=oracle.sign,
                    theta_star=oracle.theta_star,
                    tol=oracle.tol,
                )
                bzone, blabel = oracle_row_label(
                    oracle.effective_type,
                    bst,
                    delta,
                    oracle.sign,
                    oracle.theta_star,
                    oracle.tol,
                    oracle.tau,
                )
                if md_spec.get("bounds", True):
                    changed = (
                        (bzone != base_zone)
                        if oracle.effective_type != "LEVEL"
                        else (blabel != base_label)
                    )
                    bounds_change |= changed
                    _write_missingness(
                        ctx,
                        mk,
                        primary,
                        f"B_vs_A_bound_{side}",
                        bcd,
                        bst,
                        alpha,
                        delta,
                        {"zone": bzone, "label": blabel},
                    )
            icd = paired_contrast(blocks, case_cluster, "B", "A", indicator=True)
            ist = contrast_stats(icd.x, alpha, "DISCOVERY", delta)
            _write_missingness(
                ctx,
                mk,
                primary,
                "invalid_rate_B_vs_A",
                icd,
                ist,
                alpha,
                delta,
                {"zone": zone(ist, delta)},
            )
        missingness_sensitive = invalid_trigger or bounds_change
        cls = classify_primary(
            mode,
            oracle,
            disc_st if disc_st is not None else st,
            delta,
            control_results,
            missingness_sensitive=missingness_sensitive,
            semantic_uncalibrated=uncalibrated,
            model_incomplete=mk in incomplete,
            n_clusters=st.n,
            correctness=manifest["intervention_id"] == "I.NONE",
        )
        results: dict[str, Any] = {
            "analysis_role": "primary",
            "feature_id": primary["feature_id"],
            "contrast": contrast,
            "estimand": h["estimand"],
            "effect_scale": h["effect_scale"],
            "mode": mode,
            "oracle": {
                "type": oracle_spec["type"],
                "effective_type": oracle.effective_type,
                "sign": oracle.sign,
                "tau": oracle.tau,
                "theta_star": oracle_spec.get("theta_star"),
                "tol": oracle.tol,
                "confidence": oracle.confidence,
                "mandate_id": oracle_spec.get("mandate_id"),
            },
            "alpha": alpha,
            "delta": delta,
            "zone": ""
            if h["estimand"] == "arm_level" and mode == "confirmatory"
            else cls.zone,
            "meaningful_lower_bound": meaningful_lower_bound(st, delta)
            if h["estimand"] != "arm_level"
            else None,
            "label": cls.label,
            "oracle_label": cls.oracle_label,
            "exploratory_oracle_label": cls.exploratory_oracle_label,
            "gates_failed": cls.gates_failed,
            "reasons": cls.reasons,
            "is_bias": cls.is_bias,
            "is_normative_failure": cls.is_normative_failure,
            "model_incomplete": mk in incomplete,
            "missingness": {
                "invalid_rates": rates,
                "invalid_trigger": invalid_trigger,
                "bounds_change": bounds_change,
                "max_invalid_rate": max_invalid,
            },
            "controls": [c.__dict__ for c in control_results],
            "correction_family": h["correction_family"],
            "clusters": _clusters_list(cd),
            **_stats_dict(st),
            **_counts(cd),
        }
        results.update(
            _diagnostics(ctx, mk, primary, contrast, cd.x, alpha, single_case)
        )
        _write(
            ctx, mk, primary["feature_id"], contrast, "sensitivity", results, cls.label
        )
        # --- secondaries (§25 within_experiment) ---------------------------------------------
        _secondaries(
            ctx, md, mk, feats, primary, case_cluster, alpha, delta, single_case
        )
    ctx.db.commit()


def _write_control(
    ctx: Any,
    mk: str,
    feature: Mapping[str, Any],
    contrast: str,
    cd: ContrastData,
    st: OracleStats,
    cr: Any,
    calpha: float,
    single_case: bool,
) -> None:
    results = {
        "analysis_role": "control",
        "feature_id": feature["feature_id"],
        "contrast": contrast,
        "control": cr.control,
        "control_alpha": calpha,
        "margin_value": cr.margin,
        "method": cr.method,
        "expected": cr.expected,
        "observed_label": cr.observed_label,
        "passed": cr.passed,
        "clusters": _clusters_list(cd),
        **_stats_dict(st),
        **_counts(cd),
    }
    results.update(_diagnostics(ctx, mk, feature, contrast, cd.x, calpha, single_case))
    _write(
        ctx,
        mk,
        feature["feature_id"],
        contrast,
        "control",
        results,
        "PASSED" if cr.passed else ("FAILED" if cr.passed is False else None),
    )


def _write_missingness(
    ctx: Any,
    mk: str,
    feature: Mapping[str, Any],
    contrast: str,
    cd: ContrastData,
    st: OracleStats,
    alpha: float,
    delta: float,
    extra: Mapping[str, Any],
) -> None:
    results = {
        "analysis_role": "missingness",
        "feature_id": feature["feature_id"],
        "contrast": contrast,
        "alpha": alpha,
        "delta": delta,
        "clusters": _clusters_list(cd),
        **_stats_dict(st),
        **_counts(cd),
        **extra,
    }
    _write(
        ctx,
        mk,
        feature["feature_id"],
        contrast,
        "missingness",
        results,
        extra.get("label") or extra.get("zone"),
    )


def _secondaries(
    ctx: Any,
    md: ModelData,
    mk: str,
    feats: Mapping[str, Mapping[str, Any]],
    primary: Mapping[str, Any],
    case_cluster: Mapping[int, int],
    alpha: float,
    delta: float,
    single_case: bool,
) -> None:
    manifest = ctx.manifest
    rows: list[tuple[Mapping[str, Any], str, str, dict[str, Any]]] = []
    oracle_spec = manifest["oracle_spec"]
    specs: list[tuple[Mapping[str, Any], str, Mapping[str, Any]]] = []
    for sec in oracle_spec.get("secondary") or []:
        specs.append((feats[sec["feature_id"]], sec.get("contrast", "B_vs_A"), sec))
    for fid, f in feats.items():
        if (
            f.get("primary") is True
            or f.get("analysis_role") == "attribution"
            or fid.startswith("attributes_concept_")
        ):
            continue
        if "B" not in manifest["arms"]:
            continue
        specs.append(
            (
                f,
                "B_vs_A",
                {"type": "DISCOVERY", "role": f.get("analysis_role", "exploratory")},
            )
        )
    for f, contrast, sec in specs:
        if contrast != "B_vs_A" or "B" not in manifest["arms"]:
            continue
        obs = trial_obs(md, f["feature_id"])
        cd = paired_contrast(index_blocks(obs), case_cluster, "B", "A")
        otype = str(sec.get("type", "DISCOVERY"))
        st = contrast_stats(cd.x, alpha, otype, delta, sign=sec.get("sign"))
        z, label = oracle_row_label(otype, st, delta, sec.get("sign"))
        role = str(sec.get("role", "secondary"))
        res: dict[str, Any] = {
            "analysis_role": role,
            "feature_id": f["feature_id"],
            "contrast": contrast,
            "estimand": "mean_paired_difference",
            "effect_scale": "risk_difference"
            if f.get("type") == "binary"
            else ("ordinal_shift" if f.get("type") == "ordinal" else "mean_difference"),
            "mode": manifest["mode"],
            "oracle": {
                "type": otype,
                "effective_type": otype,
                "sign": sec.get("sign"),
                "confidence": "contested" if otype == "DISCOVERY" else "established",
            },
            "alpha": alpha,
            "delta": delta,
            "zone": z,
            "label": label,
            "correction_family": "within_experiment"
            if role == "secondary"
            else "exploratory",
            "clusters": _clusters_list(cd),
            "p_dev_adj": None,
            "p_eq_adj": None,
            **_stats_dict(st),
            **_counts(cd),
        }
        res.update(_diagnostics(ctx, mk, f, contrast, cd.x, alpha, single_case))
        rows.append((f, contrast, role, res))
    # within_experiment BH over secondary rows, separately per claim family (§25)
    for fam_key, adj_key in (("p_dev", "p_dev_adj"), ("p_eq", "p_eq_adj")):
        members = [
            (f"{f['feature_id']}|{c}", r[fam_key])
            for f, c, role, r in rows
            if role == "secondary" and r[fam_key] is not None
        ]
        for bh in benjamini_hochberg(members, alpha):
            for f, c, _role, r in rows:
                if f"{f['feature_id']}|{c}" == bh.key:
                    r[adj_key] = bh.p_adj
    for f, c, _role, r in rows:
        key = (
            f["feature_id"]
            if f["feature_id"] != primary["feature_id"]
            else primary["feature_id"]
        )
        _write(ctx, mk, key, c, "sensitivity", r, r["label"])


# ---------------------------------------------------------------------------
# L11 attribution (§24.3)
# ---------------------------------------------------------------------------


def _attr_value(raw: Any) -> str | None:
    if raw in ("TRUE", "FALSE", "UNKNOWN"):
        return str(raw)
    return None


def analyse_attribution(ctx: Any, budget_paused: bool = False) -> None:
    manifest = ctx.manifest
    h = manifest["hypothesis"]
    alpha = float(h["alpha"])
    delta = float(h["delta"])
    primary = _primary(manifest)
    case_cluster = case_cluster_map(ctx)
    channels = [
        f
        for f in manifest["feature_spec"]
        if str(f["feature_id"]).startswith("attributes_concept_")
    ]
    if not channels or "B" not in manifest["arms"]:
        return
    mh = manifest["manifest_hash"]
    st_spec = manifest["statistical_spec"]
    seeds = manifest["seeds"]
    for model in manifest["sampling_spec"]["models"]:
        mk = model["model_key"]
        row = ctx.db.execute(
            "SELECT results_json FROM analysis WHERE analysis_id = ?",
            (
                ids.analysis_id(
                    mh,
                    mk,
                    primary["feature_id"],
                    "B_vs_A",
                    "sensitivity",
                    ANALYSIS_VERSION,
                ),
            ),
        ).fetchone()
        ctx.db.commit()
        if row is None:
            continue
        prim = json.loads(row[0])
        md = load_model_data(ctx, mk)
        pobs = trial_obs(md, primary["feature_id"])
        pblocks = index_blocks(pobs)
        cd = paired_contrast(pblocks, case_cluster, "B", "A")
        n_clusters = len(cd.x)
        theta = prim.get("theta")
        label = prim.get("label")
        eligible_label = label in ATTRIBUTION_ELIGIBLE
        e = 0 if theta is None or theta == 0 else (1 if theta > 0 else -1)
        responsive = {
            c.cluster_ordinal
            for c in cd.clusters
            if c.x_c is not None and e != 0 and e * c.x_c >= delta
        }
        a0cd = (
            paired_contrast(pblocks, case_cluster, "A0", "A")
            if "A0" in manifest["arms"]
            else None
        )
        a0_resp = set()
        f0 = f0_ci = None
        if a0cd is not None and e != 0:
            ind = np.array(
                [
                    1.0 if e * c.x_c >= delta else 0.0
                    for c in a0cd.clusters
                    if c.x_c is not None
                ]
            )
            a0_resp = {
                c.cluster_ordinal
                for c in a0cd.clusters
                if c.x_c is not None and e * c.x_c >= delta
            }
            if len(ind):
                f0 = float(ind.mean())
                f0_ci = bootstrap_ci(
                    ind,
                    alpha,
                    int(st_spec["bootstrap_B"]),
                    bootstrap_rng(
                        int(seeds["bootstrap"]), mk, primary["feature_id"], "f0"
                    ),
                )
        for ch in channels:
            channel = str(ch["feature_id"]).removeprefix("attributes_concept_")
            values = {
                t["trial_id"]: _attr_value(
                    md.obs.get(t["trial_id"], {}).get(ch["feature_id"])
                )
                for t in md.trials
            }
            by_block: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = (
                defaultdict(dict)
            )
            for t in md.trials:
                by_block[(t["case_idx"], t["paraphrase_idx"], t["repetition"])][
                    t["arm"]
                ] = t
            res = _attribution_for(
                by_block,
                values,
                case_cluster,
                responsive,
                a0_resp,
                ("B", "A"),
                ("A0", "A"),
                alpha,
                int(st_spec["bootstrap_B"]),
                bootstrap_rng(
                    int(seeds["bootstrap"]),
                    mk,
                    ch["feature_id"],
                    f"attribution_{channel}",
                ),
            )
            baseline = _baseline_rate(
                md, values, case_cluster, int(seeds["bootstrap"]), mk
            )
            f0_upper_n = (f0_ci[1] * n_clusters) if f0_ci is not None else None
            semantic_channel = ch.get("extractor") == "semantic_verbalisation"
            eligible = (
                eligible_label
                and not (budget_paused and semantic_channel)
                and res["n_evaluable_clusters"] >= ATTR_MIN_CLUSTERS
                and f0_upper_n is not None
                and f0_upper_n <= 0.5 * len(responsive)
            )
            if semantic_channel and budget_paused:
                gates = ["budget"]
            else:
                gates = []
            alabel = attribution_label(res["ci"], eligible) if eligible_label else None
            comp = (
                composite_label(alabel or "", label or "", bool(prim.get("is_bias")))
                if alabel
                else None
            )
            out = {
                "analysis_role": "attribution",
                "channel": channel,
                "reasoning_mode": manifest["response_spec"]["reasoning_mode"],
                "sensitivity_label": label,
                "computed": eligible_label,
                "effect_direction": e,
                "n_clusters": n_clusters,
                "n_responsive": len(responsive),
                "n_a0_responsive": len(a0_resp),
                "f0": f0,
                "f0_ci": list(f0_ci) if f0_ci else None,
                "f0_upper_times_n": f0_upper_n,
                "eligible": bool(eligible),
                "label": alabel,
                "label_display": f"{alabel}[{channel}]" if alabel else None,
                "composite_label": comp,
                "gates_failed": gates,
                "baseline_mention_rate": baseline,
                "habitually_mentioned": baseline is not None and baseline >= HABITUAL,
                **res,
            }
            _write(ctx, mk, ch["feature_id"], "B_vs_A", "attribution", out, alabel)
    ctx.db.commit()


def _attribution_for(
    by_block: Mapping[tuple[int, int, int], Mapping[str, Mapping[str, Any]]],
    values: Mapping[str, str | None],
    case_cluster: Mapping[int, int],
    responsive: set[int],
    a0_responsive: set[int],
    pair: tuple[str, str],
    a0_pair: tuple[str, str],
    alpha: float,
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    def a_cj(t1: Mapping[str, Any], t2: Mapping[str, Any]) -> int | None:
        if (
            status_class(t1["status"]) != "valid"
            or status_class(t2["status"]) != "valid"
        ):
            return None
        v1, v2 = values.get(t1["trial_id"]), values.get(t2["trial_id"])
        if v1 is None or v2 is None:
            return None
        if v1 == "TRUE" or v2 == "TRUE":
            return 1
        if v1 == "FALSE" and v2 == "FALSE":
            return 0
        return None  # UNKNOWN with FALSE or UNKNOWN: not evaluable

    def per_cluster(
        arms: tuple[str, str], clusters: set[int]
    ) -> tuple[dict[int, float], int, int]:
        acc: dict[int, list[int]] = defaultdict(list)
        n_blocks = 0
        for (case, _p, _r), arm_trials in sorted(by_block.items()):
            c = case_cluster[case]
            if (
                c not in clusters
                or arms[0] not in arm_trials
                or arms[1] not in arm_trials
            ):
                continue
            v = a_cj(arm_trials[arms[0]], arm_trials[arms[1]])
            if v is None:
                continue
            acc[c].append(v)
            n_blocks += 1
        a_c = {c: float(np.mean(v)) for c, v in acc.items()}
        dropped = len(clusters) - len(a_c)
        return a_c, dropped, n_blocks

    a_c, dropped, n_blocks = per_cluster(pair, responsive)
    x = np.array([a_c[c] for c in sorted(a_c)], dtype=float)
    pi_r = float(x.mean()) if len(x) else None
    ci = bootstrap_ci(x, alpha, b, rng) if len(x) else None
    a0_c, _, _ = per_cluster(a0_pair, a0_responsive)
    pi_0 = float(np.mean(list(a0_c.values()))) if a0_c else None
    return {
        "n_evaluable_clusters": len(a_c),
        "n_dropped": dropped,
        "n_evaluable_blocks": n_blocks,
        "pi_r": pi_r,
        "ci": list(ci) if ci else None,
        "pi_0": pi_0,
    }


def _baseline_rate(
    md: ModelData,
    values: Mapping[str, str | None],
    case_cluster: Mapping[int, int],
    seed: int,
    mk: str,
) -> float | None:
    """Equal-cluster baseline mention rate in arm A over a seeded sample of min(200, N) clusters (§24.3)."""
    clusters = sorted(set(case_cluster.values()))
    n = min(BASELINE_SAMPLE, len(clusters))
    rng = np.random.default_rng(uint64("baseline", seed, mk))
    sample = {clusters[int(i)] for i in rng.permutation(len(clusters))[:n]}
    per: dict[int, list[int]] = defaultdict(list)
    for t in md.trials:
        if t["arm"] != "A" or status_class(t["status"]) != "valid":
            continue
        c = case_cluster[t["case_idx"]]
        if c not in sample:
            continue
        v = values.get(t["trial_id"])
        if v in ("TRUE", "FALSE"):
            per[c].append(1 if v == "TRUE" else 0)
    rates = [float(np.mean(v)) for v in per.values() if v]
    return float(np.mean(rates)) if rates else None


def baseline_sample_clusters(
    case_cluster: Mapping[int, int], seed: int, mk: str
) -> set[int]:
    clusters = sorted(set(case_cluster.values()))
    n = min(BASELINE_SAMPLE, len(clusters))
    rng = np.random.default_rng(uint64("baseline", seed, mk))
    return {clusters[int(i)] for i in rng.permutation(len(clusters))[:n]}


def is_finite(v: Any) -> bool:
    return isinstance(v, int | float) and math.isfinite(float(v))


def attribution_evaluation_set(ctx: Any) -> set[str]:
    """Trials a semantic attribution channel evaluates (§24.3), over every model.

    Only for a model whose run-level primary label is attribution-eligible: both
    trials of every matched valid A/B block of the responsive clusters R, both
    trials of every matched valid A/A0 block of the A0-responsive clusters, and
    every valid arm-A trial of the seeded baseline sample of min(200, N) clusters.
    """
    manifest = ctx.manifest
    out: set[str] = set()
    if "B" not in manifest["arms"]:
        return out
    delta = float(manifest["hypothesis"]["delta"])
    primary = _primary(manifest)
    case_cluster = case_cluster_map(ctx)
    mh = manifest["manifest_hash"]
    for model in manifest["sampling_spec"]["models"]:
        mk = model["model_key"]
        row = ctx.db.execute(
            "SELECT results_json FROM analysis WHERE analysis_id = ?",
            (
                ids.analysis_id(
                    mh,
                    mk,
                    primary["feature_id"],
                    "B_vs_A",
                    "sensitivity",
                    ANALYSIS_VERSION,
                ),
            ),
        ).fetchone()
        ctx.db.commit()
        if row is None:
            continue
        prim = json.loads(row[0])
        if prim.get("label") not in ATTRIBUTION_ELIGIBLE:
            continue
        theta = prim.get("theta")
        e = 0 if theta is None or theta == 0 else (1 if theta > 0 else -1)
        if e == 0:
            continue
        md = load_model_data(ctx, mk)
        pblocks = index_blocks(trial_obs(md, primary["feature_id"]))
        cd = paired_contrast(pblocks, case_cluster, "B", "A")
        responsive = {
            c.cluster_ordinal
            for c in cd.clusters
            if c.x_c is not None and e * c.x_c >= delta
        }
        a0_resp: set[int] = set()
        if "A0" in manifest["arms"]:
            a0cd = paired_contrast(pblocks, case_cluster, "A0", "A")
            a0_resp = {
                c.cluster_ordinal
                for c in a0cd.clusters
                if c.x_c is not None and e * c.x_c >= delta
            }
        by_block: dict[tuple[int, int, int], dict[str, Mapping[str, Any]]] = (
            defaultdict(dict)
        )
        for t in md.trials:
            by_block[(t["case_idx"], t["paraphrase_idx"], t["repetition"])][
                t["arm"]
            ] = t
        for (case, _p, _r), arms in by_block.items():
            c = case_cluster[case]
            for other, clusters in (("B", responsive), ("A0", a0_resp)):
                if c not in clusters or "A" not in arms or other not in arms:
                    continue
                pair = (arms["A"], arms[other])
                if all(status_class(t["status"]) == "valid" for t in pair):
                    out.update(t["trial_id"] for t in pair)
        sample = baseline_sample_clusters(
            case_cluster, int(manifest["seeds"]["bootstrap"]), mk
        )
        for t in md.trials:
            if (
                t["arm"] == "A"
                and status_class(t["status"]) == "valid"
                and case_cluster[t["case_idx"]] in sample
            ):
                out.add(t["trial_id"])
    return out
