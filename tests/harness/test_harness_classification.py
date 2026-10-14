"""Unit: classification; Attribution (spec §22, §24, §36)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pytest

from financial_llm_harness.evaluation.rules import FeatureContext, extract
from financial_llm_harness.oracles import (
    Oracle,
    attribution_label,
    bias_flags,
    classify_equivalence_control,
    classify_positive_control,
    classify_primary,
    composite_label,
)
from financial_llm_harness.statistics.analysis import (
    ModelData,
    _attribution_for,
    _baseline_rate,
)
from financial_llm_harness.statistics.equivalence import contrast_stats
from financial_llm_harness.statistics.paired import bootstrap_rng


def _st(
    mean: float,
    sd: float = 0.05,
    n: int = 200,
    seed: int = 0,
    oracle: str = "DISCOVERY",
    alpha: float = 0.05,
    **kw: Any,
) -> Any:
    x = np.random.default_rng(seed).normal(mean, sd, size=n)
    return contrast_stats(x, alpha, oracle, 0.1, **kw)


def test_controls_classified_without_gates_and_own_alpha() -> None:
    null_fail = classify_equivalence_control("null", "A0_vs_A", _st(0.3), 0.1)
    assert null_fail.passed is False
    pos = classify_positive_control(
        {"type": "DIRECTIONAL", "sign": -1},
        _st(-0.45, oracle="DIRECTIONAL", sign=-1),
        0.1,
    )
    assert pos.passed is True and pos.observed_label == "JUSTIFIED_SENSITIVITY"
    # the positive control is classified even when A0 fails; changing hypothesis.alpha changes no control
    a = classify_equivalence_control("null", "A0_vs_A", _st(0.0, alpha=0.05), 0.1)
    b = classify_equivalence_control("null", "A0_vs_A", _st(0.0, alpha=0.05), 0.1)
    assert a == b


def test_gate_precedence() -> None:
    st = _st(-0.3, oracle="DIRECTIONAL", sign=-1)
    oracle = Oracle("DIRECTIONAL", sign=-1)
    good_null = classify_equivalence_control("null", "A0_vs_A", _st(0.0), 0.1)
    bad_null = classify_equivalence_control("null", "A0_vs_A", _st(0.3), 0.1)
    good_pos = classify_positive_control(
        {"type": "DIRECTIONAL", "sign": -1},
        _st(-0.45, oracle="DIRECTIONAL", sign=-1),
        0.1,
    )
    bad_pos = classify_positive_control(
        {"type": "DIRECTIONAL", "sign": -1},
        _st(0.0, oracle="DIRECTIONAL", sign=-1),
        0.1,
    )
    kw = {
        "missingness_sensitive": True,
        "semantic_uncalibrated": False,
        "model_incomplete": False,
        "n_clusters": 200,
    }
    assert (
        classify_primary(
            "confirmatory", oracle, st, 0.1, [bad_null, bad_pos], **kw
        ).label
        == "UNSTABLE"
    )
    assert (
        classify_primary(
            "confirmatory", oracle, st, 0.1, [good_null, bad_pos], **kw
        ).label
        == "INSTRUMENT_INSENSITIVE"
    )
    c = classify_primary("confirmatory", oracle, st, 0.1, [good_null, good_pos], **kw)
    assert c.label == "INDETERMINATE" and "missingness" in c.gates_failed
    kw2 = {
        "missingness_sensitive": False,
        "semantic_uncalibrated": False,
        "model_incomplete": True,
        "n_clusters": 200,
    }
    assert classify_primary(
        "confirmatory", oracle, st, 0.1, [good_null, good_pos], **kw2
    ).gates_failed == ["model_incomplete"]
    kw3 = {
        "missingness_sensitive": False,
        "semantic_uncalibrated": False,
        "model_incomplete": False,
        "n_clusters": 20,
    }
    assert (
        classify_primary(
            "confirmatory", oracle, st, 0.1, [good_null, good_pos], **kw3
        ).label
        == "INDETERMINATE"
    )
    kw4 = {
        "missingness_sensitive": False,
        "semantic_uncalibrated": False,
        "model_incomplete": False,
        "n_clusters": 200,
    }
    ok = classify_primary("confirmatory", oracle, st, 0.1, [good_null, good_pos], **kw4)
    assert ok.label == "JUSTIFIED_SENSITIVITY" and ok.gates_failed == []


def test_discovery_mode_uses_discovery_row() -> None:
    st = _st(0.3, oracle="INVARIANCE")
    kw = {
        "missingness_sensitive": False,
        "semantic_uncalibrated": False,
        "model_incomplete": False,
        "n_clusters": 200,
    }
    c = classify_primary("discovery", Oracle("INVARIANCE"), st, 0.1, [], **kw)
    assert (
        c.label == "SENSITIVE" and c.exploratory_oracle_label == "SPURIOUS_SENSITIVITY"
    )
    assert c.is_bias is False and c.is_normative_failure is False


@pytest.mark.parametrize(
    ("mode", "confidence", "label", "attr", "is_bias", "is_nf", "composite"),
    [
        (
            "confirmatory",
            "established",
            "SPURIOUS_SENSITIVITY",
            "UNVERBALISED",
            True,
            True,
            "UNVERBALISED_BIAS",
        ),
        (
            "confirmatory",
            "established",
            "PERVERSE_SENSITIVITY",
            "UNVERBALISED",
            True,
            True,
            "UNVERBALISED_BIAS",
        ),
        (
            "confirmatory",
            "established",
            "OVER_SENSITIVITY",
            "VERBALISED",
            True,
            True,
            None,
        ),
        (
            "confirmatory",
            "established",
            "JUSTIFIED_SENSITIVITY",
            "UNVERBALISED",
            False,
            False,
            "UNVERBALISED_DEPENDENCY",
        ),
        (
            "confirmatory",
            "established",
            "UNDER_SENSITIVITY",
            "UNVERBALISED",
            False,
            True,
            None,
        ),
        (
            "confirmatory",
            "established",
            "BELOW_STANDARD",
            "UNVERBALISED",
            False,
            True,
            None,
        ),
        (
            "confirmatory",
            "conventional",
            "SPURIOUS_SENSITIVITY",
            "UNVERBALISED",
            False,
            False,
            "UNVERBALISED_DEPENDENCY",
        ),
        (
            "discovery",
            "established",
            "SPURIOUS_SENSITIVITY",
            "UNVERBALISED",
            False,
            False,
            "UNVERBALISED_DEPENDENCY",
        ),
        (
            "discovery",
            "established",
            "SENSITIVE",
            "UNVERBALISED",
            False,
            False,
            "UNVERBALISED_DEPENDENCY",
        ),
        (
            "confirmatory",
            "established",
            "SPURIOUS_SENSITIVITY",
            "INDETERMINATE",
            True,
            True,
            None,
        ),
    ],
)
def test_bias_truth_table(
    mode: str,
    confidence: str,
    label: str,
    attr: str,
    is_bias: bool,
    is_nf: bool,
    composite: str | None,
) -> None:
    b, nf = bias_flags(mode, Oracle("INVARIANCE", confidence=confidence), label)
    assert (b, nf) == (is_bias, is_nf)
    assert composite_label(attr, label, b) == composite


def test_factors_channel() -> None:
    f = {"extractor": "concept_factor"}
    parsed = {
        "material_factors": [
            {"factor": "momentum", "stance": "supports"},
            {"factor": "valuation", "stance": "opposes"},
        ]
    }
    assert (
        extract(f, parsed, FeatureContext("A", concept_factor="momentum")).value
        == "TRUE"
    )
    assert (
        extract(f, parsed, FeatureContext("A", concept_factor="leverage")).value
        == "FALSE"
    )
    assert (
        extract(
            f, {"material_factors": []}, FeatureContext("A", concept_factor="momentum")
        ).value
        == "FALSE"
    )


def test_attribution_label_and_eligibility_rule() -> None:
    assert attribution_label((0.31, 0.5), True) == "VERBALISED"
    assert attribution_label((0.01, 0.29), True) == "UNVERBALISED"
    assert attribution_label((0.2, 0.4), True) == "INDETERMINATE"
    assert attribution_label((0.01, 0.1), False) == "INDETERMINATE"


# ---------------------------------------------------------------------------
# Attribution (§24.3)
# ---------------------------------------------------------------------------


def _blocks(
    spec: dict[int, list[tuple[str, str]]], dup: int = 1
) -> tuple[
    dict[tuple[int, int, int], dict[str, dict[str, Any]]], dict[str, str | None]
]:
    """case → list of (value in B, value in A) per block, duplicated `dup` times over p."""
    by_block: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    values: dict[str, str | None] = {}
    for case, pairs in spec.items():
        for j, (vb, va) in enumerate(pairs):
            for d in range(dup):
                p = j * dup + d
                for arm, v in (("B", vb), ("A", va)):
                    tid = f"{case}-{p}-{arm}"
                    by_block[(case, p, 0)][arm] = {"trial_id": tid, "status": "SUCCESS"}
                    values[tid] = v
    return by_block, values


def test_block_evaluability_truth_table() -> None:
    spec = {
        0: [
            ("TRUE", "UNKNOWN"),
            ("FALSE", "FALSE"),
            ("FALSE", "UNKNOWN"),
            ("UNKNOWN", "UNKNOWN"),
            ("UNKNOWN", "TRUE"),
        ]
    }
    bb, vals = _blocks(spec)
    res = _attribution_for(
        bb,
        vals,
        {0: 0},
        {0},
        set(),
        ("B", "A"),
        ("A0", "A"),
        0.05,
        200,
        bootstrap_rng(1, "m", "f", "c"),
    )
    assert res["n_evaluable_blocks"] == 3  # TRUE+any (2) and FALSE+FALSE (1)
    assert res["pi_r"] == pytest.approx(2 / 3)


def test_cluster_without_evaluable_block_dropped_and_counted() -> None:
    spec = {0: [("TRUE", "FALSE")], 1: [("UNKNOWN", "FALSE")]}
    bb, vals = _blocks(spec)
    res = _attribution_for(
        bb,
        vals,
        {0: 0, 1: 1},
        {0, 1},
        set(),
        ("B", "A"),
        ("A0", "A"),
        0.05,
        200,
        bootstrap_rng(1, "m", "f", "c"),
    )
    assert res["n_dropped"] == 1 and res["n_evaluable_clusters"] == 1


def test_duplicating_paraphrases_leaves_attribution_unchanged() -> None:
    rng = np.random.default_rng(2)
    spec = {
        c: [("TRUE" if rng.random() < 0.3 else "FALSE", "FALSE") for _ in range(2)]
        for c in range(60)
    }
    case_cluster = {c: c for c in spec}
    r1 = _attribution_for(
        *_blocks(spec, 1),
        case_cluster,
        set(spec),
        set(),
        ("B", "A"),
        ("A0", "A"),
        0.05,
        500,
        bootstrap_rng(1, "m", "f", "c"),
    )
    r3 = _attribution_for(
        *_blocks(spec, 3),
        case_cluster,
        set(spec),
        set(),
        ("B", "A"),
        ("A0", "A"),
        0.05,
        500,
        bootstrap_rng(1, "m", "f", "c"),
    )
    assert r1["pi_r"] == pytest.approx(r3["pi_r"])
    assert r1["ci"] == pytest.approx(r3["ci"])
    assert r1["n_evaluable_clusters"] == r3["n_evaluable_clusters"]


def test_baseline_rate_equal_cluster_weighted() -> None:
    trials = []
    values = {}
    # cluster 0: 1 trial TRUE; cluster 1: 9 trials FALSE → equal weights give 0.5, pooled 0.1
    trials.append({"trial_id": "t0", "arm": "A", "status": "SUCCESS", "case_idx": 0})
    values["t0"] = "TRUE"
    for i in range(9):
        trials.append(
            {"trial_id": f"u{i}", "arm": "A", "status": "SUCCESS", "case_idx": 1}
        )
        values[f"u{i}"] = "FALSE"
    rate = _baseline_rate(ModelData("m", trials, {}), values, {0: 0, 1: 1}, 1, "m")
    assert rate == pytest.approx(0.5)


def test_eligibility_uses_upper_f0_on_a0_contrast(s1_run) -> None:  # type: ignore[no-untyped-def]
    import csv

    rows = list(csv.DictReader(open(s1_run / "results" / "attribution.csv")))
    a = next(r for r in rows if r["model_key"].startswith("fake--a"))
    assert float(a["f0_upper_times_n"]) == pytest.approx(float(a["f0_ci_high"]) * 960)
    assert a["eligible"] == "true"
    assert float(a["f0_upper_times_n"]) <= 0.5 * int(a["n_responsive"])


def test_level_positive_control_at_the_ceiling_does_not_gate() -> None:
    """CR-08: a model already correct in PCA is not INSTRUMENT_INSENSITIVE."""
    from financial_llm_harness.oracles import saturate_positive_control
    from financial_llm_harness.statistics.equivalence import level_stats

    po = {"type": "DIRECTIONAL", "sign": 1, "confidence": "established"}
    rng = np.random.default_rng(3)
    flat = classify_positive_control(po, _st(0.0, oracle="DIRECTIONAL", sign=1), 0.1)
    assert flat.passed is False
    ceiling = level_stats((rng.random(400) < 0.99).astype(float), 0.05, 0.95)
    sat = saturate_positive_control(flat, po, ceiling, 0.95)
    assert sat.passed is True and sat.observed_label == "SATURATED"
    # headroom in PCA: the failure stands
    low = level_stats((rng.random(400) < 0.6).astype(float), 0.05, 0.95)
    assert saturate_positive_control(flat, po, low, 0.95) is flat
    # a passing control is never rewritten; a negative-sign control is never saturated
    ok = classify_positive_control(po, _st(0.3, oracle="DIRECTIONAL", sign=1), 0.1)
    assert saturate_positive_control(ok, po, ceiling, 0.95) is ok
    neg = {**po, "sign": -1}
    assert saturate_positive_control(flat, neg, ceiling, 0.95) is flat
    level = Oracle("LEVEL", tau=0.95)
    st = level_stats((rng.random(400) < 0.99).astype(float), 0.05, 0.95)
    null = classify_equivalence_control("null", "A0_vs_A", _st(0.0), 0.1)
    cls = classify_primary(
        "confirmatory",
        level,
        st,
        0.1,
        [null, sat],
        missingness_sensitive=False,
        semantic_uncalibrated=False,
        model_incomplete=False,
        n_clusters=400,
    )
    assert cls.label == "MEETS_STANDARD" and "positive_control" not in cls.gates_failed
    assert "POSITIVE_CONTROL_SATURATED" in cls.reasons
