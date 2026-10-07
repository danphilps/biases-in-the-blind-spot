"""Matched-block cluster outcomes and cluster-mean t inference (spec §23.2–§23.4, E-26, E-27).

Pairing happens before aggregation: a block difference exists only when both
trials of that exact (case, p, r) block are valid. Cluster outcomes x_c enter a
one-sample Student-t analysis with equal cluster weights.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from financial_llm_harness.core.ids import uint64

STATUS_CLASS = {
    "SUCCESS": "valid",
    "COMPILE_FAILURE": "valid",
    "TEST_FAILURE": "valid",
    "RUNTIME_FAILURE": "valid",
    "REFUSAL": "invalid",
    "EMPTY_RESPONSE": "invalid",
    "TRUNCATED": "invalid",
    "CONTENT_FILTERED": "invalid",
    "INVALID_RESPONSE": "invalid",
    "PARSE_FAILURE": "invalid",
    "API_ERROR": "missing",
    "RATE_LIMIT": "missing",
    "TIMEOUT": "missing",
    "SANDBOX_FAILURE": "missing",
    "BUDGET_EXHAUSTED": "missing",
    "PENDING": "missing",
    "SUBMITTED": "missing",
    "EXCLUDED_CONTEXT": "excluded",
    "EXCLUDED_MEMORISED": "excluded",
}
BLOCK_PRECEDENCE = ("excluded", "missing", "invalid", "matched_valid")


@dataclass(frozen=True)
class TrialObs:
    """One trial's analysis input for one feature."""

    case_idx: int
    arm: str
    p: int
    r: int
    cls: str  # valid | invalid | missing | excluded
    value: float | None  # None for a valid trial whose feature is missing


def status_class(status: str) -> str:
    return STATUS_CLASS.get(status, "missing")


Block = tuple[int, int, int]  # (case_idx, p, r)


def index_blocks(obs: Sequence[TrialObs]) -> dict[Block, dict[str, TrialObs]]:
    out: dict[Block, dict[str, TrialObs]] = defaultdict(dict)
    for o in obs:
        out[(o.case_idx, o.p, o.r)][o.arm] = o
    return out


def _trial_state(o: TrialObs) -> str:
    if o.cls == "valid" and o.value is None:
        return "missing"  # feature missing on a valid trial (e.g. EVALUATOR_UNKNOWN)
    return o.cls


def block_status(x: TrialObs, y: TrialObs) -> str:
    states = {_trial_state(x), _trial_state(y)}
    for s in ("excluded", "missing", "invalid"):
        if s in states:
            return s
    return "matched_valid"


@dataclass
class ClusterOutcome:
    cluster_ordinal: int
    n_cases: int
    n_blocks_scheduled: int = 0
    n_blocks_matched: int = 0
    n_blocks_invalid: int = 0
    n_blocks_missing: int = 0
    n_blocks_excluded: int = 0
    x_c: float | None = None
    mean_a: float | None = None
    mean_b: float | None = None
    truth_c: float | None = None


@dataclass
class ContrastData:
    """Per-cluster outcomes of one contrast, in cluster_ordinal order."""

    clusters: list[ClusterOutcome]
    n_trials_x: int = 0
    n_trials_y: int = 0
    case_counts: dict[str, int] = field(default_factory=dict)

    @property
    def x(self) -> np.ndarray:
        return np.array(
            [c.x_c for c in self.clusters if c.x_c is not None], dtype=float
        )

    def totals(self) -> dict[str, int]:
        keys = (
            "n_blocks_scheduled",
            "n_blocks_matched",
            "n_blocks_invalid",
            "n_blocks_missing",
            "n_blocks_excluded",
        )
        return {k: sum(getattr(c, k) for c in self.clusters) for k in keys}

    @property
    def mean_a(self) -> float | None:
        vals = [
            c.mean_a
            for c in self.clusters
            if c.x_c is not None and c.mean_a is not None
        ]
        return float(np.mean(vals)) if vals else None

    @property
    def mean_b(self) -> float | None:
        vals = [
            c.mean_b
            for c in self.clusters
            if c.x_c is not None and c.mean_b is not None
        ]
        return float(np.mean(vals)) if vals else None


def paired_contrast(
    blocks: Mapping[Block, Mapping[str, TrialObs]],
    case_cluster: Mapping[int, int],
    x_arm: str,
    y_arm: str,
    truth: Mapping[int, float] | None = None,
    impute: Callable[[TrialObs, str], float] | None = None,
    indicator: bool = False,
) -> ContrastData:
    """Matched-block contrast X − Y aggregated case → cluster (§23.2, §23.4).

    ``impute`` turns the primary analysis into a worst-case bound on the grid
    without excluded or missing blocks; ``indicator`` analyses the invalid rate.
    """
    per_case: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    trials_x = trials_y = 0
    for (case, _p, _r), arms in sorted(blocks.items()):
        if x_arm not in arms or y_arm not in arms:
            continue
        xo, yo = arms[x_arm], arms[y_arm]
        st = block_status(xo, yo)
        c = counts[case]
        c["scheduled"] += 1
        c[st] += 1
        if indicator or impute is not None:
            if st in ("excluded", "missing"):
                continue
            if indicator:
                xv = 1.0 if _trial_state(xo) == "invalid" else 0.0
                yv = 1.0 if _trial_state(yo) == "invalid" else 0.0
            else:
                assert impute is not None
                xv = xo.value if _trial_state(xo) == "valid" else impute(xo, "x")
                yv = yo.value if _trial_state(yo) == "valid" else impute(yo, "y")
                assert xv is not None and yv is not None
            per_case[case].append((float(xv) - float(yv), float(yv), float(xv)))
            continue
        if st != "matched_valid":
            continue
        trials_x += 1
        trials_y += 1
        assert xo.value is not None and yo.value is not None
        per_case[case].append(
            (float(xo.value) - float(yo.value), float(yo.value), float(xo.value))
        )
    return _aggregate(per_case, counts, case_cluster, truth, trials_x, trials_y)


def _aggregate(
    per_case: Mapping[int, list[tuple[float, float, float]]],
    counts: Mapping[int, Mapping[str, int]],
    case_cluster: Mapping[int, int],
    truth: Mapping[int, float] | None,
    trials_x: int,
    trials_y: int,
) -> ContrastData:
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for case in counts:
        by_cluster[case_cluster[case]].append(case)
    clusters: list[ClusterOutcome] = []
    for ordinal in sorted(by_cluster):
        cases = sorted(by_cluster[ordinal])
        co = ClusterOutcome(cluster_ordinal=ordinal, n_cases=len(cases))
        for case in cases:
            c = counts[case]
            co.n_blocks_scheduled += c.get("scheduled", 0)
            co.n_blocks_matched += c.get("matched_valid", 0)
            co.n_blocks_invalid += c.get("invalid", 0)
            co.n_blocks_missing += c.get("missing", 0)
            co.n_blocks_excluded += c.get("excluded", 0)
        d_i, a_i, b_i, t_i = [], [], [], []
        for case in cases:
            vals = per_case.get(case)
            if not vals:
                continue
            d_i.append(float(np.mean([v[0] for v in vals])))
            a_i.append(float(np.mean([v[1] for v in vals])))
            b_i.append(float(np.mean([v[2] for v in vals])))
            if truth is not None:
                t_i.append(float(truth[case]))
        if d_i:
            co.mean_a = float(np.mean(a_i))
            co.mean_b = float(np.mean(b_i))
            d_c = float(np.mean(d_i))
            if truth is not None:
                co.truth_c = float(np.mean(t_i))
                co.x_c = d_c - co.truth_c
            else:
                co.x_c = d_c
        clusters.append(co)
    return ContrastData(clusters, trials_x, trials_y)


def arm_level(
    obs: Sequence[TrialObs],
    case_cluster: Mapping[int, int],
    arm: str,
    impute_value: float | None = None,
) -> ContrastData:
    """λ = mean_c Ȳ_c over the valid observations of the level arm (§23.2)."""
    per_case: dict[int, list[float]] = defaultdict(list)
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_trials = 0
    for o in obs:
        if o.arm != arm:
            continue
        st = _trial_state(o)
        c = counts[o.case_idx]
        c["scheduled"] += 1
        c["matched_valid" if st == "valid" else st] += 1
        if st == "valid":
            n_trials += 1
            assert o.value is not None
            per_case[o.case_idx].append(float(o.value))
        elif st == "invalid" and impute_value is not None:
            per_case[o.case_idx].append(float(impute_value))
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for case in counts:
        by_cluster[case_cluster[case]].append(case)
    clusters: list[ClusterOutcome] = []
    for ordinal in sorted(by_cluster):
        cases = sorted(by_cluster[ordinal])
        co = ClusterOutcome(cluster_ordinal=ordinal, n_cases=len(cases))
        for case in cases:
            c = counts[case]
            co.n_blocks_scheduled += c.get("scheduled", 0)
            co.n_blocks_matched += c.get("matched_valid", 0)
            co.n_blocks_invalid += c.get("invalid", 0)
            co.n_blocks_missing += c.get("missing", 0)
            co.n_blocks_excluded += c.get("excluded", 0)
        means = [float(np.mean(per_case[c])) for c in cases if per_case.get(c)]
        if means:
            co.x_c = float(np.mean(means))
            co.mean_a = co.x_c
        clusters.append(co)
    return ContrastData(clusters, n_trials, 0)


# ---------------------------------------------------------------------------
# One-sample t procedures (§23.3)
# ---------------------------------------------------------------------------


def _degenerate(x: np.ndarray) -> bool:
    return len(x) >= 2 and float(np.ptp(x)) == 0.0


def p_two_sided(x: np.ndarray, mu: float = 0.0) -> float | None:
    if len(x) < 2:
        return None
    if _degenerate(x):
        return 1.0 if float(x[0]) == mu else 0.0
    return float(stats.ttest_1samp(x, mu).pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def p_greater(x: np.ndarray, mu: float) -> float | None:
    """One-sided p for H₁: θ > μ."""
    if len(x) < 2:
        return None
    if _degenerate(x):
        return 0.0 if float(x[0]) > mu else 1.0
    return float(stats.ttest_1samp(x, mu, alternative="greater").pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def p_less(x: np.ndarray, mu: float) -> float | None:
    """One-sided p for H₁: θ < μ."""
    if len(x) < 2:
        return None
    if _degenerate(x):
        return 0.0 if float(x[0]) < mu else 1.0
    return float(stats.ttest_1samp(x, mu, alternative="less").pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def ci(x: np.ndarray, a: float) -> tuple[float, float] | None:
    """Student-t CI_{1−a}."""
    if len(x) < 2:
        return None
    theta = float(x.mean())
    if _degenerate(x):
        return theta, theta
    se = float(x.std(ddof=1)) / math.sqrt(len(x))
    t = float(stats.t.ppf(1 - a / 2, len(x) - 1))
    return theta - t * se, theta + t * se


def tost(x: np.ndarray, low: float, high: float) -> float | None:
    """p_TOST = max(p(H₁: θ > L), p(H₁: θ < U))."""
    pg = p_greater(x, low)
    pl = p_less(x, high)
    if pg is None or pl is None:
        return None
    return max(pg, pl)


@dataclass(frozen=True)
class TResult:
    n: int
    theta: float | None
    sd_c: float | None
    ci: tuple[float, float] | None
    ci2a: tuple[float, float] | None
    p_raw: float | None


def t_summary(x: np.ndarray, alpha: float) -> TResult:
    n = len(x)
    theta = float(x.mean()) if n else None
    sd = float(x.std(ddof=1)) if n >= 2 else None
    return TResult(n, theta, sd, ci(x, alpha), ci(x, 2 * alpha), p_two_sided(x, 0.0))


# ---------------------------------------------------------------------------
# Diagnostics (§23.3)
# ---------------------------------------------------------------------------


def mcnemar_exact(x: np.ndarray) -> float:
    n_plus = int(np.sum(x == 1.0))
    n_minus = int(np.sum(x == -1.0))
    if n_plus + n_minus == 0:
        return 1.0
    return float(
        stats.binomtest(
            k=n_plus, n=n_plus + n_minus, p=0.5, alternative="two-sided"
        ).pvalue  # pyright: ignore[reportAttributeAccessIssue]
    )


def wilcoxon_pratt(x: np.ndarray) -> float:
    if len(x) == 0 or bool(np.all(x == 0)):
        return 1.0
    return float(stats.wilcoxon(x, zero_method="pratt").pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def bootstrap_rng(
    bootstrap_seed: int, model_key: str, feature_id: str, contrast: str
) -> np.random.Generator:
    return np.random.default_rng(
        uint64(bootstrap_seed, model_key, feature_id, contrast)
    )


def bootstrap_ci(
    x: np.ndarray, alpha: float, b: int, rng: np.random.Generator
) -> tuple[float, float] | None:
    """Cluster-bootstrap percentile CI_{1−α} of the mean (E-19)."""
    n = len(x)
    if n == 0:
        return None
    idx = rng.integers(0, n, size=(b, n))
    theta_star = x[idx].mean(axis=1)
    lo, hi = np.quantile(theta_star, [alpha / 2, 1 - alpha / 2], method="linear")
    return float(lo), float(hi)


def jsonable(v: Any) -> Any:
    """Convert numpy scalars and tuples for results_json."""
    if isinstance(v, np.floating | np.integer):
        return v.item()
    if isinstance(v, tuple):
        return [jsonable(i) for i in v]
    if isinstance(v, list):
        return [jsonable(i) for i in v]
    if isinstance(v, dict):
        return {k: jsonable(i) for k, i in v.items()}
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v
