"""Spec and manifest structure (spec §4).

A spec has every manifest key except those marked "manifest only" and the keys
``generate`` fills; ``generate`` copies every other key verbatim, so nothing is
defaulted silently. ``validate`` exits 2 on a missing key, a type mismatch or an
inconsistency between the hypothesis and the oracle.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.schema import (
    as_bool,
    as_float,
    as_int,
    as_list,
    as_mapping,
    as_str,
    check_keys,
    fail,
)
from financial_llm_harness.errors import HarnessValidationError

SCHEMA_VERSION = 3

SPEC_REQUIRED = (
    "experiment_key",
    "variant_id",
    "mode",
    "parent_discovery_experiment_id",
    "faq_id",
    "intervention_id",
    "domain",
    "task_family",
    "task",
    "concept",
    "research_question",
    "data_classification",
    "mandate",
    "hypothesis",
    "scenario_spec",
    "intervention_spec",
    "controls",
    "prompt_spec",
    "paraphrase_spec",
    "response_spec",
    "feature_spec",
    "oracle_spec",
    "sampling_spec",
    "statistical_spec",
    "budget",
    "seeds",
)
SPEC_OPTIONAL = ("provenance", "battery", "correctness_context")
GENERATED_KEYS = (
    "schema_version",
    "experiment_id",
    "design_hash",
    "arms",
    "evaluators",
    "independence",
    "versions",
    "corpus",
    "code_ref",
    "created_at",
    "manifest_hash",
)

MANIFEST_ORDER = (
    "schema_version",
    "experiment_key",
    "variant_id",
    "experiment_id",
    "design_hash",
    "mode",
    "parent_discovery_experiment_id",
    "faq_id",
    "intervention_id",
    "domain",
    "task_family",
    "task",
    "concept",
    "research_question",
    "data_classification",
    "mandate",
    "hypothesis",
    "scenario_spec",
    "intervention_spec",
    "controls",
    "arms",
    "prompt_spec",
    "paraphrase_spec",
    "response_spec",
    "feature_spec",
    "oracle_spec",
    "sampling_spec",
    "statistical_spec",
    "budget",
    "evaluators",
    "independence",
    "seeds",
    "versions",
    "battery",
    "correctness_context",
    "provenance",
    "corpus",
    "code_ref",
    "created_at",
    "manifest_hash",
)

DOMAINS = ("fundamental", "quant_models", "ml", "code", "none")
ESTIMANDS = ("mean_paired_difference", "arm_level", "paired_difference_vs_truth")
DIRECTIONS = ("positive", "negative", "two_sided", "none")
EFFECT_SCALES = ("risk_difference", "mean_difference", "ordinal_shift")
CORRECTION_FAMILIES = ("within_experiment", "programme", "exploratory")
CLASSIFICATIONS = ("synthetic", "licensed", "confidential")
REASONING_MODES = ("none", "rationale_after", "reasoning_before")
ENFORCEMENTS = ("provider_schema", "prompt_only", "auto")
ORACLE_TYPES = ("INVARIANCE", "DIRECTIONAL", "MAGNITUDE", "LEVEL", "DISCOVERY")
CONFIDENCES = ("established", "conventional", "contested")
_KEY_RE = re.compile(r"^[a-z0-9_]+(__[a-z0-9_]+)*\.v[0-9]+$")


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml_io.load(path)
    if not isinstance(data, dict):
        raise HarnessValidationError(f"{path}: top level must be a mapping")
    return data


def is_manifest(data: Mapping[str, Any]) -> bool:
    """``run FILE`` treats FILE as a manifest iff its top level has ``manifest_hash`` (§32)."""
    return "manifest_hash" in data


def check_spec_keys(spec: Mapping[str, Any], where: str) -> None:
    for key in GENERATED_KEYS:
        if key in spec:
            raise fail(
                where, f"key {key!r} is filled by generate and must not be in a spec"
            )
    check_keys(spec, where, required=SPEC_REQUIRED, optional=SPEC_OPTIONAL)
    sampling = as_mapping(spec["sampling_spec"], where + ".sampling_spec")
    if "panel" not in sampling:
        raise fail(where, "sampling_spec.panel is missing (spec only)")


def order_manifest(m: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: m[k] for k in MANIFEST_ORDER if k in m}
    for k in m:
        if k not in out:
            raise HarnessValidationError(f"manifest: unknown key {k!r}")
    return out


def check_design(m: Mapping[str, Any], where: str) -> None:
    """Type and consistency checks shared by spec and manifest (§4, §10, §17, §23.3)."""
    as_str(m["experiment_key"], where + ".experiment_key")
    if not _KEY_RE.match(m["experiment_key"]) or len(m["experiment_key"]) > 60:
        raise fail(
            where,
            f"experiment_key {m['experiment_key']!r} is malformed or over 60 characters",
        )
    mode = m["mode"]
    if mode not in ("discovery", "confirmatory"):
        raise fail(where, f"mode {mode!r} invalid")
    if m["domain"] not in DOMAINS:
        raise fail(where, f"domain {m['domain']!r} invalid")
    if m["data_classification"] not in CLASSIFICATIONS:
        raise fail(where, f"data_classification {m['data_classification']!r} invalid")
    mandate = as_mapping(m["mandate"], where + ".mandate")
    check_keys(mandate, where + ".mandate", required=("id", "version", "text"))
    as_str(mandate["text"], where + ".mandate.text")
    h = as_mapping(m["hypothesis"], where + ".hypothesis")
    check_keys(
        h,
        where + ".hypothesis",
        required=(
            "source",
            "primary_feature",
            "estimand",
            "level_arm",
            "effect_scale",
            "delta",
            "direction",
            "alpha",
            "power_target",
            "equivalence_margin",
            "correction_family",
            "missing_data",
        ),
    )
    if h["source"] not in ("a_priori", "discovery"):
        raise fail(where, f"hypothesis.source {h['source']!r} invalid")
    if h["estimand"] not in ESTIMANDS:
        raise fail(where, f"hypothesis.estimand {h['estimand']!r} invalid")
    if h["effect_scale"] not in EFFECT_SCALES:
        raise fail(where, f"hypothesis.effect_scale {h['effect_scale']!r} invalid")
    if h["direction"] not in DIRECTIONS:
        raise fail(where, f"hypothesis.direction {h['direction']!r} invalid")
    if h["correction_family"] not in CORRECTION_FAMILIES:
        raise fail(
            where, f"hypothesis.correction_family {h['correction_family']!r} invalid"
        )
    for k in ("delta", "alpha", "power_target", "equivalence_margin"):
        v = as_float(h[k], f"{where}.hypothesis.{k}")
        if v <= 0:
            raise fail(where, f"hypothesis.{k} must be positive")
    if not 0 < float(h["alpha"]) < 0.5:
        raise fail(where, "hypothesis.alpha must be in (0, 0.5)")
    md = as_mapping(h["missing_data"], where + ".hypothesis.missing_data")
    check_keys(
        md,
        where + ".hypothesis.missing_data",
        required=("policy", "max_invalid_rate", "bounds"),
    )
    if md["policy"] != "complete_pairs":
        raise fail(where, "missing_data.policy must be complete_pairs")
    as_float(md["max_invalid_rate"], where + ".missing_data.max_invalid_rate")
    as_bool(md["bounds"], where + ".missing_data.bounds")
    if h["estimand"] == "arm_level":
        if not isinstance(h["level_arm"], str):
            raise fail(where, "hypothesis.level_arm is required for estimand arm_level")
    elif h["level_arm"] is not None:
        raise fail(
            where, "hypothesis.level_arm must be null unless estimand is arm_level"
        )
    oracle = as_mapping(m["oracle_spec"], where + ".oracle_spec")
    check_keys(
        oracle,
        where + ".oracle_spec",
        required=("type", "rule", "mandate_id", "provenance", "version", "confidence"),
        optional=(
            "sign",
            "item_sign",
            "basis",
            "nodes",
            "tau",
            "theta_star",
            "tol",
            "secondary",
            "truth_fn",
            "branch",
        ),
    )
    otype = oracle["type"]
    if otype not in ORACLE_TYPES:
        raise fail(where, f"oracle_spec.type {otype!r} invalid")
    if oracle["confidence"] not in CONFIDENCES:
        raise fail(where, f"oracle_spec.confidence {oracle['confidence']!r} invalid")
    est = h["estimand"]
    computed = otype == "MAGNITUDE" and oracle.get("theta_star") == "computed"
    expected_est = (
        "arm_level"
        if otype == "LEVEL"
        else ("paired_difference_vs_truth" if computed else "mean_paired_difference")
    )
    if est != expected_est:
        raise fail(
            where,
            f"estimand {est!r} does not match oracle type {otype} (expected {expected_est!r})",
        )
    if otype == "DIRECTIONAL":
        sign = oracle.get("sign")
        if sign not in (-1, 1) or isinstance(sign, bool):
            raise fail(where, "a DIRECTIONAL oracle needs sign ±1 (the sign of B − A)")
        want = "positive" if sign == 1 else "negative"
    elif otype in ("INVARIANCE", "LEVEL"):
        want = "none"
    else:
        want = "two_sided"
    if h["direction"] != want:
        raise fail(
            where,
            f"hypothesis.direction {h['direction']!r} disagrees with the oracle ({want!r})",
        )
    if otype == "LEVEL":
        tau = as_float(oracle.get("tau"), where + ".oracle_spec.tau")
        if not 0 < tau < 1:
            raise fail(where, "oracle_spec.tau must lie in (0, 1)")
    if otype == "MAGNITUDE":
        as_float(oracle.get("tol"), where + ".oracle_spec.tol")
    st = as_mapping(m["statistical_spec"], where + ".statistical_spec")
    check_keys(
        st,
        where + ".statistical_spec",
        required=(
            "test",
            "ci_method",
            "bootstrap_B",
            "planning_ref",
            "planning_sd",
            "planning_sd_source",
            "sequential",
        ),
    )
    if st["test"] != "cluster_mean_t" or st["ci_method"] != "student_t":
        raise fail(
            where,
            "statistical_spec.test must be cluster_mean_t and ci_method student_t (§23.3)",
        )
    as_int(st["bootstrap_B"], where + ".statistical_spec.bootstrap_B")
    if st["sequential"] != "none" and mode == "confirmatory":
        raise fail(
            where, "confirmatory experiments use a fixed N (sequential: none, §26)"
        )
    for i, ref in enumerate(as_list(st["planning_ref"], where + ".planning_ref")):
        rm = as_mapping(ref, f"{where}.planning_ref[{i}]")
        check_keys(
            rm,
            f"{where}.planning_ref[{i}]",
            required=(
                "run_id",
                "model_key",
                "feature_id",
                "contrast",
                "variance",
                "source_row_id",
                "source_sha256",
            ),
            optional=("derivation",),
        )
        as_float(rm["variance"], f"{where}.planning_ref[{i}].variance")
    rs = as_mapping(m["response_spec"], where + ".response_spec")
    check_keys(
        rs,
        where + ".response_spec",
        required=(
            "contract_id",
            "contract_version",
            "reasoning_mode",
            "enforcement",
            "max_output_tokens",
            "abstain_allowed",
        ),
        optional=("enforcement_resolved", "enforcement_strata"),
    )
    if rs["reasoning_mode"] not in REASONING_MODES:
        raise fail(where, f"reasoning_mode {rs['reasoning_mode']!r} invalid")
    if rs["enforcement"] not in ENFORCEMENTS:
        raise fail(where, f"enforcement {rs['enforcement']!r} invalid")
    as_int(rs["max_output_tokens"], where + ".response_spec.max_output_tokens")
    ss = as_mapping(m["sampling_spec"], where + ".sampling_spec")
    as_int(ss["R"], where + ".sampling_spec.R")
    if ss.get("order") != "randomised_blocks":
        raise fail(where, "sampling_spec.order must be randomised_blocks")
    tp = as_mapping(ss["temperature_policy"], where + ".temperature_policy")
    if tp.get("mode") not in ("fixed", "provider_default"):
        raise fail(where, "temperature_policy.mode must be fixed or provider_default")
    ps = as_mapping(m["paraphrase_spec"], where + ".paraphrase_spec")
    check_keys(
        ps, where + ".paraphrase_spec", required=("P", "template_files", "families")
    )
    if as_int(ps["P"], where + ".paraphrase_spec.P") != len(ps["template_files"]):
        raise fail(where, "paraphrase_spec.P must equal the number of template_files")
    seeds = as_mapping(m["seeds"], where + ".seeds")
    check_keys(
        seeds,
        where + ".seeds",
        required=("generation", "paraphrase", "order", "bootstrap", "trial_seed_rule"),
    )
    for k in ("generation", "paraphrase", "order", "bootstrap"):
        as_int(seeds[k], f"{where}.seeds.{k}")
    budget = as_mapping(m["budget"], where + ".budget")
    check_keys(
        budget,
        where + ".budget",
        required=(
            "max_calls",
            "max_input_tokens",
            "max_output_tokens",
            "max_cost_usd",
            "pricing_table_version",
        ),
    )
    features = as_list(m["feature_spec"], where + ".feature_spec")
    primaries = [
        f for f in features if isinstance(f, Mapping) and f.get("primary") is True
    ]
    if len(primaries) != 1:
        raise fail(where, "feature_spec needs exactly one primary feature (§2.3)")
    if primaries[0].get("feature_id") != h["primary_feature"]:
        raise fail(
            where,
            "hypothesis.primary_feature differs from the primary feature_spec entry",
        )
    ids = [f.get("feature_id") for f in features]
    if len(set(ids)) != len(ids):
        raise fail(where, "feature_spec repeats a feature_id")
    sc = as_mapping(m["scenario_spec"], where + ".scenario_spec")
    check_keys(
        sc,
        where + ".scenario_spec",
        required=(
            "schema_ref",
            "generator",
            "distributions",
            "correlation",
            "constraints",
            "n_cases",
            "seed_stream",
        ),
        optional=("strata",),
    )
    as_int(sc["n_cases"], where + ".scenario_spec.n_cases")
    if sc["seed_stream"] not in ("conf", "disc"):
        raise fail(where, "scenario_spec.seed_stream must be conf or disc (§27)")
    if mode == "confirmatory" and sc["seed_stream"] != "conf":
        raise fail(
            where, "a confirmatory experiment draws from the conf seed stream (§27)"
        )
    controls = as_mapping(m["controls"], where + ".controls")
    check_keys(
        controls,
        where + ".controls",
        required=(
            "null_arm",
            "placebo",
            "positive",
            "control_alpha",
            "null_margin",
            "subset_n",
        ),
        optional=("placebo_margin", "pilot"),
    )
    if controls.get("pilot") is not None:
        if not isinstance(controls["pilot"], bool):
            raise fail(where, "controls.pilot must be a boolean")
        if controls["pilot"] and mode != "discovery":
            raise fail(
                where, "controls.pilot is a discovery planning pilot (§22, CR-7)"
            )
    ispec = m["intervention_spec"]
    if ispec is None:
        if m["intervention_id"] != "I.NONE":
            raise fail(
                where,
                "intervention_spec is null only for I.NONE correctness experiments",
            )
    else:
        ism = as_mapping(ispec, where + ".intervention_spec")
        for bad in ("levels", "factorial", "dose_levels"):
            if bad in ism:
                raise fail(where, "factorial and continuous dose designs are V2 (§10)")
        if isinstance(ism.get("counterfactual"), list):
            raise fail(where, "continuous dose levels are V2 (§10)")
    if mode == "confirmatory":
        if h["source"] == "discovery" and not m["parent_discovery_experiment_id"]:
            raise fail(
                where,
                "a confirmatory hypothesis with source discovery needs parent_discovery_experiment_id (§27)",
            )
        if (
            h["source"] == "a_priori"
            and m["parent_discovery_experiment_id"] is not None
        ):
            raise fail(
                where, "source a_priori has parent_discovery_experiment_id null (§27)"
            )
