"""Oracle-specific test statistics: TOST, under-sensitivity, magnitude and level (spec §23.3, §25).

Each claim-family p-value is defined so that, with m = 1, p < α reproduces the
§24.2 rule of the labels that need that family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from financial_llm_harness.statistics.paired import (
    ci,
    p_greater,
    p_less,
    p_two_sided,
    tost,
)


@dataclass(frozen=True)
class OracleStats:
    """Every p-value and bound an oracle row or claim family needs."""

    n: int
    theta: float | None
    sd_c: float | None
    ci: tuple[float, float] | None
    ci2a: tuple[float, float] | None
    p_raw: float | None
    p_tost: float | None
    margin: tuple[float, float]
    p_dev: float | None
    p_eq: float | None
    test_dev: str
    test_eq: str
    p_under: float | None = None
    p_over: float | None = None
    p_short: float | None = None
    p_meets: float | None = None
    p_below: float | None = None


def _sd(x: np.ndarray) -> float | None:
    return float(x.std(ddof=1)) if len(x) >= 2 else None


def contrast_stats(
    x: np.ndarray,
    alpha: float,
    oracle_type: str,
    delta: float,
    sign: int | None = None,
    theta_star: float | None = None,
    tol: float | None = None,
    margin: float | None = None,
) -> OracleStats:
    """Stats of a paired contrast under an oracle type (DISCOVERY for secondaries and controls)."""
    n = len(x)
    theta = float(x.mean()) if n else None
    c1, c2 = ci(x, alpha), ci(x, 2 * alpha)
    p_raw = p_two_sided(x, 0.0)
    if oracle_type == "MAGNITUDE":
        ts = 0.0 if theta_star is None else float(theta_star)
        t = float(tol if tol is not None else delta)
        lo, hi = ts - t, ts + t
        p_t = tost(x, lo, hi)
        p_over = p_greater(x, hi)
        p_short = p_less(x, lo)
        p_dev = (
            None
            if p_over is None or p_short is None
            else min(1.0, 2 * min(p_over, p_short))
        )
        return OracleStats(
            n,
            theta,
            _sd(x),
            c1,
            c2,
            p_raw,
            p_t,
            (lo, hi),
            p_dev,
            p_t,
            "t_outside_tol",
            "tost",
            p_over=p_over,
            p_short=p_short,
        )
    m = float(margin if margin is not None else delta)
    p_t = tost(x, -m, m)
    if oracle_type == "DIRECTIONAL":
        s = int(sign or 1)
        p_under = p_less(s * x, delta)
        return OracleStats(
            n,
            theta,
            _sd(x),
            c1,
            c2,
            p_raw,
            p_t,
            (-m, m),
            p_raw,
            p_under,
            "t_two_sided",
            "t_under",
            p_under=p_under,
        )
    return OracleStats(
        n, theta, _sd(x), c1, c2, p_raw, p_t, (-m, m), p_raw, p_t, "t_two_sided", "tost"
    )


def level_stats(x: np.ndarray, alpha: float, tau: float) -> OracleStats:
    """LEVEL(τ): one-sided (1 − α) bounds = the ends of CI_{1−2α}; p_meets, p_below."""
    n = len(x)
    lam = float(x.mean()) if n else None
    bounds = ci(x, 2 * alpha)
    p_meets = p_greater(x, tau)
    p_below = p_less(x, tau)
    return OracleStats(
        n,
        lam,
        _sd(x),
        ci(x, alpha),
        bounds,
        None,
        None,
        (tau, tau),
        p_below,
        p_meets,
        "t_below",
        "t_meets",
        p_meets=p_meets,
        p_below=p_below,
    )
