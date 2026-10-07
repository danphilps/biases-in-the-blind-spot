"""Statistical calibration by fake-mechanism simulation (spec §23, §36).

Outcomes are simulated with the §20.11 coupling (arms share U₁ with a given
probability) and analysed with the harness's matched-block cluster-mean
procedure; the trial-pooled IID analysis is shown to fail under clustering.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from financial_llm_harness.statistics import paired
from financial_llm_harness.statistics.paired import TrialObs

ALPHA = 0.05
N_EXP = 1000


def _simulate(
    rng: np.random.Generator,
    n_clusters: int,
    blocks: int,
    base: float,
    theta: float,
    couple: float,
    rho: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cluster outcomes x_c, pooled A, pooled B) for one experiment."""
    cluster_effect = rng.normal(0, 1, size=n_clusters) * rho
    pa = np.clip(base + cluster_effect, 0.02, 0.98)[:, None] * np.ones((1, blocks))
    pb = np.clip(pa + theta, 0, 1)
    u1 = rng.random((n_clusters, blocks))
    ub = np.where(
        rng.random((n_clusters, blocks)) < couple, u1, rng.random((n_clusters, blocks))
    )
    ya = (u1 < pa).astype(float)
    yb = (ub < pb).astype(float)
    x = (yb - ya).mean(axis=1)
    return x, ya.ravel(), yb.ravel()


def _mc_se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


@pytest.mark.parametrize("blocks", [1, 6])
def test_type_one_error(blocks: int) -> None:
    rng = np.random.default_rng(42 + blocks)
    rejections = 0
    for _ in range(N_EXP):
        x, _, _ = _simulate(rng, 120, blocks, 0.5, 0.0, couple=0.5)
        p = paired.p_two_sided(x)
        rejections += p is not None and p < ALPHA
    rate = rejections / N_EXP
    assert rate <= ALPHA + 2 * _mc_se(ALPHA, N_EXP)


def test_coverage_at_delta_plus() -> None:
    rng = np.random.default_rng(7)
    theta = -(0.10 + 0.05)
    covered = 0
    for _ in range(N_EXP):
        x, _, _ = _simulate(rng, 150, 3, 0.6, theta, couple=0.0)
        c = paired.ci(x, ALPHA)
        assert c is not None
        covered += c[0] <= theta <= c[1]
    assert 0.93 <= covered / N_EXP <= 0.97


def test_cluster_effects_break_trial_pooling() -> None:
    rng = np.random.default_rng(11)
    covered_cluster = covered_pooled = 0
    sims = 600
    for _ in range(sims):
        x, ya, yb = _simulate(rng, 80, 6, 0.5, 0.0, couple=0.95, rho=0.3)
        c = paired.ci(x, ALPHA)
        assert c is not None
        covered_cluster += c[0] <= 0.0 <= c[1]
        # trial-pooled IID analysis on the unmatched means: Welch interval over trials
        diff = yb.mean() - ya.mean()
        se = math.sqrt(ya.var(ddof=1) / len(ya) + yb.var(ddof=1) / len(yb))
        z = stats.norm.ppf(1 - ALPHA / 2)
        covered_pooled += diff - z * se <= 0.0 <= diff + z * se
    assert 0.93 <= covered_cluster / sims <= 0.975
    assert abs(covered_pooled / sims - 0.95) > abs(covered_cluster / sims - 0.95)


def test_vectorised_simulation_matches_matched_block_estimator() -> None:
    rng = np.random.default_rng(3)
    x, _, _ = _simulate(rng, 5, 2, 0.5, 0.2, couple=0.5)
    del x
    obs = []
    rng2 = np.random.default_rng(4)
    ya = (rng2.random((5, 2)) < 0.5).astype(float)
    yb = (rng2.random((5, 2)) < 0.7).astype(float)
    for c in range(5):
        for p in range(2):
            obs.append(TrialObs(c, "A", p, 0, "valid", float(ya[c, p])))
            obs.append(TrialObs(c, "B", p, 0, "valid", float(yb[c, p])))
    cd = paired.paired_contrast(
        paired.index_blocks(obs), {c: c for c in range(5)}, "B", "A"
    )
    assert np.allclose(cd.x, (yb - ya).mean(axis=1))
