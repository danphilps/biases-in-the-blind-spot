"""Group-sequential boundaries for discovery mode (spec §26).

Lan–DeMets O'Brien–Fleming-type spending with information fractions t_k;
nominal boundaries from the joint normal with corr(Z_j, Z_k) = √(t_j/t_k),
K ≤ 5. Confirmatory experiments use a fixed N and a single analysis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import optimize, stats


def obf_spending(t: float, alpha: float) -> float:
    """Cumulative α spent at information fraction t (two-sided OBF-type)."""
    if t <= 0:
        return 0.0
    return float(2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / math.sqrt(t))))


def obf_boundaries(fractions: Sequence[float], alpha: float) -> list[float]:
    """Nominal two-sided |Z| boundaries for each look."""
    ts = list(fractions)
    if not 1 <= len(ts) <= 5:
        raise ValueError("K must be between 1 and 5")
    if (
        any(b <= a for a, b in zip(ts, ts[1:], strict=False))
        or ts[-1] > 1
        or ts[0] <= 0
    ):
        raise ValueError("information fractions must increase within (0, 1]")
    bounds: list[float] = []
    for k in range(len(ts)):
        spend = obf_spending(ts[k], alpha) - (
            obf_spending(ts[k - 1], alpha) if k else 0.0
        )
        if k == 0:
            bounds.append(float(stats.norm.ppf(1 - spend / 2)))
            continue
        cov = np.array(
            [
                [math.sqrt(min(a, b) / max(a, b)) for b in ts[: k + 1]]
                for a in ts[: k + 1]
            ]
        )
        prev = bounds[:]

        def excess(
            c: float,
            prev: list[float] = prev,
            cov: np.ndarray = cov,
            spend: float = spend,
        ) -> float:
            lower = np.array([-b for b in prev] + [-np.inf])
            upper = np.array(prev + [np.inf])
            mvn = stats.multivariate_normal(mean=np.zeros(len(prev) + 1), cov=cov)  # pyright: ignore[reportArgumentType]
            # P(continue to k, then cross): P(|Z_j| < b_j ∀ j<k) − P(|Z_j| < b_j ∀ j<k, |Z_k| < c)
            inside_prev = _box_prob(mvn, lower, upper)
            upper_c = np.array(prev + [c])
            lower_c = np.array([-b for b in prev] + [-c])
            inside_all = _box_prob(mvn, lower_c, upper_c)
            return (inside_prev - inside_all) - spend

        root = optimize.brentq(excess, 0.5, 10.0)
        bounds.append(float(root))  # pyright: ignore[reportArgumentType]
    return bounds


def _box_prob(
    mvn: stats._multivariate.multivariate_normal_frozen,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    lo = np.where(np.isinf(lower), -40.0, lower)
    hi = np.where(np.isinf(upper), 40.0, upper)
    return float(mvn.cdf(hi, lower_limit=lo))
