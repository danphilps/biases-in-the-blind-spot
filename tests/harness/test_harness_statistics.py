"""Unit: statistics; Pairing; Planning variance (spec §22, §23, §25, §36)."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pytest
from scipy import stats

from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.oracles import (
    NEEDS_DEV,
    NEEDS_EQ,
    oracle_row_label,
    programme_label,
    programme_zone,
    zone,
)
from financial_llm_harness.statistics import paired, power
from financial_llm_harness.statistics.correction import benjamini_hochberg
from financial_llm_harness.statistics.equivalence import contrast_stats, level_stats
from financial_llm_harness.statistics.paired import TrialObs

# ---------------------------------------------------------------------------
# Cluster-mean t procedures (§23.3)
# ---------------------------------------------------------------------------


def test_cluster_mean_t_hand_fixture() -> None:
    x = np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2])
    n = len(x)
    se = x.std(ddof=1) / math.sqrt(n)
    ref = float(stats.ttest_1samp(x, 0.0).pvalue)  # pyright: ignore[reportAttributeAccessIssue]
    assert paired.p_two_sided(x) == pytest.approx(ref, rel=1e-12)
    for a in (0.05, 0.10):
        t = stats.t.ppf(1 - a / 2, n - 1)
        lo, hi = paired.ci(x, a)  # type: ignore[misc]
        assert lo == pytest.approx(x.mean() - t * se, rel=1e-12)
        assert hi == pytest.approx(x.mean() + t * se, rel=1e-12)


@pytest.mark.parametrize("seed", range(40))
def test_tost_ci_equivalence_and_under_rule(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(
        rng.uniform(-0.15, 0.15), rng.uniform(0.05, 0.4), size=int(rng.integers(5, 80))
    )
    for a in (0.05, 0.1):
        lo, hi = -0.1, 0.12
        p = paired.tost(x, lo, hi)
        c2 = paired.ci(x, 2 * a)
        assert p is not None and c2 is not None
        assert (p < a) == (c2[0] > lo and c2[1] < hi)
    alpha, delta = 0.05, 0.1
    for s in (1, -1):
        st = contrast_stats(x, alpha, "DIRECTIONAL", delta, sign=s)
        sx_ci = paired.ci(s * x, 2 * alpha)
        assert sx_ci is not None and st.p_under is not None
        assert (st.p_under < alpha) == (sx_ci[1] < delta)


@pytest.mark.parametrize("seed", range(30))
def test_level_bounds_match_one_sided_p(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    tau = 0.9
    x = np.clip(
        rng.normal(rng.uniform(0.8, 1.0), 0.1, size=int(rng.integers(5, 60))), 0, 1
    )
    st = level_stats(x, 0.05, tau)
    _, label = oracle_row_label("LEVEL", st, 0.1, tau=tau)
    assert st.p_meets is not None and st.p_below is not None
    assert (label == "MEETS_STANDARD") == (st.p_meets <= 0.05)
    assert (label == "BELOW_STANDARD") == (st.p_below < 0.05)
    se = x.std(ddof=1) / math.sqrt(len(x))
    tq = stats.t.ppf(0.95, len(x) - 1)
    assert st.ci2a is not None and st.ci2a[0] == pytest.approx(x.mean() - tq * se)


def test_magnitude_p_dev() -> None:
    x = np.array([0.31, 0.29, 0.35, 0.40, 0.28, 0.33])
    st = contrast_stats(x, 0.05, "MAGNITUDE", 0.1, theta_star=0.1, tol=0.05)
    assert st.p_over is not None and st.p_short is not None
    assert st.p_dev == pytest.approx(min(1.0, 2 * min(st.p_over, st.p_short)))
    assert st.margin == (0.05, pytest.approx(0.15))


def test_paired_difference_vs_truth() -> None:
    obs = []
    truth = {}
    for case in range(6):
        truth[case] = 0.2 * case
        obs.append(TrialObs(case, "A", 0, 0, "valid", 1.0))
        obs.append(TrialObs(case, "B", 0, 0, "valid", 1.0 + 0.2 * case + 0.05))
    cd = paired.paired_contrast(
        paired.index_blocks(obs), {c: c for c in range(6)}, "B", "A", truth=truth
    )
    assert np.allclose(cd.x, 0.05)


def test_ordinal_outcomes_on_declared_scores() -> None:
    from financial_llm_harness.evaluation.rules import FeatureContext, extract

    f = {
        "extractor": "contract_category",
        "field": "rating",
        "allowed_values": ["sell", "hold", "buy"],
        "scores": {"sell": -1, "hold": 0, "buy": 1},
    }
    assert extract(f, {"rating": "buy"}, FeatureContext("A")).value == 1.0
    f2 = {
        "extractor": "contract_category",
        "field": "rating",
        "allowed_values": ["sell", "hold", "buy"],
    }
    assert extract(f2, {"rating": "hold"}, FeatureContext("A")).value == 1.0


# ---------------------------------------------------------------------------
# Degenerate samples (§23.3)
# ---------------------------------------------------------------------------


def test_degenerate_samples_never_call_ttest(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("ttest_1samp called on a degenerate sample")

    monkeypatch.setattr(paired.stats, "ttest_1samp", boom)
    zero = np.zeros(10)
    assert paired.ci(zero, 0.05) == (0.0, 0.0)
    assert paired.p_two_sided(zero) == 1.0
    assert paired.tost(zero, -0.1, 0.1) == 0.0
    small = np.full(10, 0.05)
    assert paired.p_two_sided(small) == 0.0
    assert paired.tost(small, -0.1, 0.1) == 0.0
    big = np.full(10, 0.2)
    assert paired.tost(big, -0.1, 0.1) == 1.0
    one = np.array([0.3])
    assert (
        paired.ci(one, 0.05) is None
        and paired.p_two_sided(one) is None
        and paired.tost(one, -0.1, 0.1) is None
    )
    st = contrast_stats(one, 0.05, "INVARIANCE", 0.1)
    assert st.theta == 0.3 and st.ci is None and st.p_raw is None


# ---------------------------------------------------------------------------
# Diagnostics (§23.3, E-2, E-19)
# ---------------------------------------------------------------------------


def test_exact_mcnemar_matches_statsmodels_and_legacy() -> None:
    from statsmodels.stats.contingency_tables import mcnemar

    from biases_in_the_blind_spot.concept_pipeline.statistics import compute_mcnemar

    for total in range(0, 201):
        for n_plus in range(0, total + 1, max(1, total // 7)):
            n_minus = total - n_plus
            x = np.array([1.0] * n_plus + [-1.0] * n_minus + [0.0] * 3)
            ours = paired.mcnemar_exact(x)
            if total == 0:
                assert ours == 1.0
                continue
            ref = float(
                mcnemar(np.array([[0, n_plus], [n_minus, 0]]), exact=True).pvalue  # pyright: ignore[reportAttributeAccessIssue]
            )
            assert ours == pytest.approx(ref, rel=1e-12)
            if total < 25:
                assert ours == pytest.approx(
                    compute_mcnemar(n_plus, n_minus)[1], rel=1e-12
                )


def test_wilcoxon_diagnostic() -> None:
    x = np.array([0.5, -0.2, 0.0, 1.0, 0.5, -1.0, 0.25, 0.75])
    assert paired.wilcoxon_pratt(x) == pytest.approx(
        stats.wilcoxon(x, zero_method="pratt").pvalue  # pyright: ignore[reportAttributeAccessIssue]
    )
    assert paired.wilcoxon_pratt(np.zeros(5)) == 1.0


def test_diagnostic_never_drives_zone() -> None:
    # 13 clusters at +1, 0 at -1, many zeros: McNemar significant, t-test CI wide relative to delta
    x = np.array([1.0] * 6 + [0.0] * 4)
    st = contrast_stats(x, 0.05, "INVARIANCE", 0.7)
    assert paired.mcnemar_exact(x) < 0.05
    assert zone(st, 0.7) != "Z+"  # zone follows the t-test (θ̂ 0.6 < δ)


def test_bootstrap_determinism() -> None:
    x = np.random.default_rng(0).normal(size=200)
    a = paired.bootstrap_ci(x, 0.05, 1000, paired.bootstrap_rng(7, "m", "f", "B_vs_A"))
    b = paired.bootstrap_ci(x, 0.05, 1000, paired.bootstrap_rng(7, "m", "f", "B_vs_A"))
    c = paired.bootstrap_ci(x, 0.05, 1000, paired.bootstrap_rng(7, "m", "f", "A0_vs_A"))
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# BH and the programme rules (§25)
# ---------------------------------------------------------------------------


def test_bh_hand_computed() -> None:
    rows = benjamini_hochberg(
        [("e1", 0.01), ("e2", 0.04), ("e3", 0.03), ("e4", 0.20)], 0.05
    )
    assert [r.key for r in rows] == ["e1", "e3", "e2", "e4"]
    assert [r.rank for r in rows] == [1, 2, 3, 4]
    assert [r.bh_threshold for r in rows] == pytest.approx(
        [0.0125, 0.025, 0.0375, 0.05]
    )
    assert [r.p_adj for r in rows] == pytest.approx(
        [0.04, 0.0533333333, 0.0533333333, 0.2]
    )
    assert [r.rejected for r in rows] == [True, False, False, False]
    # ties sort by key
    tied = benjamini_hochberg([("b", 0.01), ("a", 0.01)], 0.05)
    assert [r.key for r in tied] == ["a", "b"]


def test_bh_families_independent() -> None:
    dev = [("x", 0.01), ("y", 0.03)]
    eq = [("x", 0.2), ("y", 0.001)]
    before = benjamini_hochberg(eq, 0.05)
    benjamini_hochberg([*dev, ("z", 0.04)], 0.05)
    assert benjamini_hochberg(eq, 0.05) == before


@pytest.mark.parametrize(
    "otype", ["INVARIANCE", "DIRECTIONAL", "MAGNITUDE", "LEVEL", "DISCOVERY"]
)
def test_programme_rules_every_oracle(otype: str) -> None:
    labels = sorted(
        NEEDS_DEV[otype]
        | NEEDS_EQ[otype]
        | {"INDETERMINATE", "UNSTABLE", "INSTRUMENT_INSENSITIVE"}
    )
    for label in labels:
        for dev in (True, False):
            for eq in (True, False):
                pl = programme_label(otype, label, dev, eq)
                assert pl in (label, "INDETERMINATE")
                if label in NEEDS_DEV[otype]:
                    assert (pl == label) == dev
                if label in NEEDS_EQ[otype]:
                    assert (pl == label) == eq
                if label in ("INDETERMINATE", "UNSTABLE", "INSTRUMENT_INSENSITIVE"):
                    assert pl == label
    if otype in ("MAGNITUDE", "LEVEL"):
        assert programme_zone(otype, "Z+", True, True) == ""
    else:
        assert programme_zone(otype, "Z-", True, False) == "Z-"
        assert programme_zone(otype, "Z-", False, True) == "Zi"
        assert programme_zone(otype, "Z0", False, True) == "Z0"
        assert programme_zone(otype, "Z0", True, False) == "Zi"
        assert programme_zone(otype, "Zs", True, False) == "Zs"


def test_programme_m1_equals_run_label() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        x = rng.normal(rng.uniform(-0.3, 0.3), 0.3, size=40)
        st = contrast_stats(x, 0.05, "INVARIANCE", 0.1)
        z, label = oracle_row_label("INVARIANCE", st, 0.1)
        dev = benjamini_hochberg([("e", st.p_dev or 1.0)], 0.05)[0].rejected
        eq = benjamini_hochberg([("e", st.p_eq or 1.0)], 0.05)[0].rejected
        assert programme_label("INVARIANCE", label, dev, eq) == label
        assert programme_zone("INVARIANCE", z, dev, eq) == z


# ---------------------------------------------------------------------------
# Pairing (§23.2, §23.4)
# ---------------------------------------------------------------------------


def test_unmatched_blocks_give_no_cluster_outcome() -> None:
    obs = [
        TrialObs(0, "A", 1, 0, "valid", 1.0),
        TrialObs(0, "B", 1, 0, "invalid", None),
        TrialObs(0, "A", 2, 0, "invalid", None),
        TrialObs(0, "B", 2, 0, "valid", 0.0),
        TrialObs(1, "A", 1, 0, "valid", 1.0),
        TrialObs(1, "B", 1, 0, "valid", 0.0),
        TrialObs(1, "A", 2, 0, "valid", 0.0),
        TrialObs(1, "B", 2, 0, "valid", 0.0),
    ]
    cd = paired.paired_contrast(paired.index_blocks(obs), {0: 0, 1: 1}, "B", "A")
    c0, c1 = cd.clusters
    assert c0.x_c is None and c0.n_blocks_scheduled == 2 and c0.n_blocks_invalid == 2
    assert c1.x_c == -0.5
    assert cd.mean_b is not None and cd.mean_a is not None
    assert float(cd.x.mean()) == pytest.approx(cd.mean_b - cd.mean_a)


def test_block_status_precedence_partitions() -> None:
    rng = np.random.default_rng(11)
    classes = ["valid", "invalid", "missing", "excluded"]
    obs = []
    for case in range(40):
        for p in range(3):
            for arm in ("A", "B"):
                cls = classes[int(rng.integers(0, 4))]
                obs.append(
                    TrialObs(case, arm, p, 0, cls, 1.0 if cls == "valid" else None)
                )
    blocks = paired.index_blocks(obs)
    cd = paired.paired_contrast(blocks, {c: c // 2 for c in range(40)}, "B", "A")
    t = cd.totals()
    assert t["n_blocks_scheduled"] == 120
    assert (
        t["n_blocks_matched"]
        + t["n_blocks_invalid"]
        + t["n_blocks_missing"]
        + t["n_blocks_excluded"]
        == 120
    )
    for arms in blocks.values():
        s = paired.block_status(arms["A"], arms["B"])
        cls = {arms["A"].cls, arms["B"].cls}
        if "excluded" in cls:
            assert s == "excluded"
        elif "missing" in cls:
            assert s == "missing"
        elif "invalid" in cls:
            assert s == "invalid"
        else:
            assert s == "matched_valid"
    # bounds and the invalid-rate analysis use the grid without excluded or missing blocks
    ind = paired.paired_contrast(
        blocks, {c: c // 2 for c in range(40)}, "B", "A", indicator=True
    )
    lower = paired.paired_contrast(
        blocks,
        {c: c // 2 for c in range(40)},
        "B",
        "A",
        impute=lambda o, s: 0.0 if s == "x" else 1.0,
    )
    grid = t["n_blocks_matched"] + t["n_blocks_invalid"]
    for cdx in (ind, lower):
        assert cdx.totals()["n_blocks_scheduled"] == 120
    used = sum(
        1
        for (_c, _p, _r), arms in blocks.items()
        if paired.block_status(arms["A"], arms["B"]) in ("matched_valid", "invalid")
    )
    assert used == grid


# ---------------------------------------------------------------------------
# Planning variance (§22, §23.6)
# ---------------------------------------------------------------------------


def test_variance_bound_on_unit_interval() -> None:
    rng = np.random.default_rng(5)
    for _ in range(2000):
        k = int(rng.integers(2, 6))
        support = rng.uniform(-1, 1, size=k)
        probs = rng.dirichlet(np.ones(k))
        mean = float((support * probs).sum())
        var = float(((support - mean) ** 2 * probs).sum())
        assert var <= 1.0 + 1e-12
    assert float(np.var(np.array([-1.0, 1.0]))) == 1.0


def test_worked_check() -> None:
    assert power.superiority_n(1.0, 0.1, 0.05, 0.8) == 785
    assert power.equivalence_n(1.0, 0.1, 0.05, 0.8) == 857
    assert power.superiority_n(0.3, 0.1, 0.05, 0.8) == 236
    assert power.equivalence_n(0.3, 0.1, 0.05, 0.8) == 257
    assert power.superiority_n(1.0, 0.2, 0.05, 0.8) == 197
    assert power.level_n(0.25, 0.15, 0.05, 0.8) == 69
    assert power.level_n(0.25, 0.10, 0.05, 0.8) == 155
    assert power.equivalence_n(1.0, 0.15, 0.05, 0.8) == 381
    assert power.superiority_n(1.0, 0.30, 0.05, 0.8) == 88


def _design() -> dict[str, Any]:
    from tests.harness.helpers import load_spec

    return load_spec()


def test_planning_ref_reproduces_n_and_subset() -> None:
    d = _design()
    st = copy.deepcopy(d["statistical_spec"])
    primary = d["feature_spec"][0]
    st["planning_ref"] = [
        {
            "run_id": "r",
            "model_key": "m",
            "feature_id": "decision_add",
            "contrast": c,
            "variance": 0.3,
            "source_row_id": "x",
            "source_sha256": "y",
        }
        for c in ("B_vs_A", "A0_vs_A", "PL_vs_A", "PCB_vs_PCA")
    ]
    req = power.primary_requirement("DIRECTIONAL", d["hypothesis"], st, primary)
    assert req.n_required == 257 and req.variance.source == "planning_ref"
    ctrl = power.control_requirements(
        ["A", "B", "A0", "PL", "PCA", "PCB"],
        d["controls"],
        d["hypothesis"],
        st,
        primary,
    )
    assert power.computed_subset_n(ctrl) == 257
    nopilot = power.control_requirements(
        ["A", "B", "A0", "PL", "PCA", "PCB"],
        d["controls"],
        d["hypothesis"],
        d["statistical_spec"],
        primary,
    )
    assert power.computed_subset_n(nopilot) == 857


def test_non_fake_model_planned_at_no_pilot_bound() -> None:
    from financial_llm_harness.generation.validate_bundle import (
        _fake_planning_refs,
        _panel_planning_check,
    )

    d = _design()
    d["arms"] = ["A", "B", "A0", "PL", "PCA", "PCB"]
    d["sampling_spec"]["models"] = [
        {"model_key": "local--m--12345678", "provider": "local"}
    ]
    with pytest.raises(HarnessValidationError, match="local--m--12345678") as exc:
        _panel_planning_check(d, n_clusters=300, subset_n=300)
    assert exc.value.exit_code == 2
    _panel_planning_check(d, n_clusters=900, subset_n=857)
    d["statistical_spec"]["planning_ref"] = [
        {
            "run_id": None,
            "model_key": "local--m--12345678",
            "feature_id": "decision_add",
            "contrast": "B_vs_A",
            "variance": 0.1,
            "source_row_id": None,
            "source_sha256": None,
            "derivation": "x",
        }
    ]
    with pytest.raises(HarnessValidationError, match="fake-design"):
        _fake_planning_refs(d)


def test_power_formulas_by_simulation() -> None:
    rng = np.random.default_rng(9)
    sigma2, delta, alpha = 0.3, 0.1, 0.05
    n = power.superiority_n(sigma2, delta, alpha, 0.8)
    hits = 0
    sims = 600
    for _ in range(sims):
        x = rng.normal(delta, math.sqrt(sigma2), size=n)
        c = paired.ci(x, alpha)
        assert c is not None
        hits += c[0] > 0
    assert 0.74 <= hits / sims <= 0.86
