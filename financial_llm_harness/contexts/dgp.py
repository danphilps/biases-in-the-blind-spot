"""C7: synthetic panel with a known data-generating process (spec §T3, §T9.2).

    r_{i,t+1} = Σ_k β_ik f_{k,t+1} + Σ_j γ_j(s_t) z_{ij,t} + ε_{i,t+1}

Planted true characteristics have γ_j ≠ 0, spurious ones γ_j = 0;
cross-sectional correlation comes from the factors; ε is Student-t (fat
tails). Optional: a Markov regime path s_t switching γ, delistings with
delisting returns, and publication lags on characteristics (the panel then
shows z lagged by L dates). The known truth returned with the data — the
population IC per signal and regime, the true covariance, the regime path and
which signals the factors span — is computed from the parameters, never
estimated from the draws. Every parameter is the caller's (§T11, D-23), and
draws use the caller's dataset generator (§12).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from financial_llm_harness.errors import HarnessValidationError


@dataclass(frozen=True)
class DGPParams:
    n_assets: int
    n_dates: int
    factor_vol: Sequence[float]  # K factor volatilities per period
    beta_sd: float  # loadings β_ik ~ N(0, beta_sd²)
    gamma: Sequence[Sequence[float]]  # [regime][signal] γ_j(s)
    resid_vol: float
    t_dof: float  # Student-t degrees of freedom of ε (> 2)
    transition: Sequence[Sequence[float]] | None = (
        None  # regime Markov matrix; None: one regime
    )
    delist_prob: float = 0.0  # per asset and date
    delist_return: float = 0.0
    publication_lag: int = 0
    spanned: Sequence[
        int
    ] = ()  # signals equal to a loading (spanned by the factors), by index


@dataclass
class Panel:
    returns: np.ndarray  # (T, N), NaN after delisting; returns[t] realised over t → t+1
    signals: np.ndarray  # (T, N, J), as shown (publication lag applied)
    regimes: np.ndarray  # (T,)
    delisted_at: np.ndarray  # (N,) date index of delisting, −1 if never
    truth: dict[str, Any]


def _check(p: DGPParams) -> None:
    if p.t_dof <= 2:
        raise HarnessValidationError(
            "C7: Student-t ε needs more than 2 degrees of freedom"
        )
    n_reg = len(p.gamma)
    if p.transition is not None:
        m = np.asarray(p.transition, dtype=float)
        if m.shape != (n_reg, n_reg) or not np.allclose(m.sum(axis=1), 1.0):
            raise HarnessValidationError(
                "C7: the transition matrix must be stochastic, one row per regime"
            )
    elif n_reg != 1:
        raise HarnessValidationError("C7: several γ regimes need a transition matrix")
    if len({len(g) for g in p.gamma}) != 1:
        raise HarnessValidationError("C7: every regime has one γ per signal")
    if any(not 0 <= j < len(p.gamma[0]) for j in p.spanned):
        raise HarnessValidationError("C7: a spanned signal index is out of range")
    if len(p.spanned) > len(p.factor_vol):
        raise HarnessValidationError("C7: each spanned signal needs its own factor")


def simulate(p: DGPParams, rng: np.random.Generator) -> Panel:
    _check(p)
    n, t_len, k = p.n_assets, p.n_dates, len(p.factor_vol)
    j_sig = len(p.gamma[0])
    beta = p.beta_sd * rng.standard_normal((n, k))
    z = rng.standard_normal((t_len, n, j_sig))
    for idx, j in enumerate(p.spanned):
        z[:, :, j] = (beta[:, idx] / p.beta_sd)[
            None, :
        ]  # the signal is the factor loading
    regimes = np.zeros(t_len, dtype=int)
    if p.transition is not None:
        m = np.asarray(p.transition, dtype=float)
        for t in range(1, t_len):
            regimes[t] = int(rng.choice(len(m), p=m[regimes[t - 1]]))
    gamma = np.asarray(p.gamma, dtype=float)
    f = rng.standard_normal((t_len, k)) * np.asarray(p.factor_vol, dtype=float)
    scale = p.resid_vol / math.sqrt(p.t_dof / (p.t_dof - 2))
    eps = scale * rng.standard_t(p.t_dof, size=(t_len, n))
    r = f @ beta.T + np.einsum("tnj,tj->tn", z, gamma[regimes]) + eps
    delisted = np.full(n, -1)
    if p.delist_prob > 0:
        hit = rng.random((t_len, n)) < p.delist_prob
        for i in range(n):
            ts = np.flatnonzero(hit[:, i])
            if ts.size:
                d = int(ts[0])
                delisted[i] = d
                r[d, i] = p.delist_return
                r[d + 1 :, i] = np.nan
    shown = z.copy()
    if p.publication_lag > 0:
        shown[p.publication_lag :] = z[: -p.publication_lag]
        shown[: p.publication_lag] = np.nan
    return Panel(r, shown, regimes, delisted, truth(p, beta))


def truth(p: DGPParams, beta: np.ndarray) -> dict[str, Any]:
    """Population quantities from the parameters (§T9.2 'known truth')."""
    fv = np.asarray(p.factor_vol, dtype=float)
    cov = beta @ np.diag(fv**2) @ beta.T + np.eye(p.n_assets) * p.resid_vol**2
    gamma = np.asarray(p.gamma, dtype=float)
    # with unit-variance, independent signals, cov(z_j, r) = γ_j across the cross-section;
    # var(r | date) = Σ γ² + β-part + σ²; the IC is the cross-sectional Pearson correlation
    factor_part = float(p.beta_sd**2 * (fv**2).sum())
    ic = []
    for g in gamma:
        total = float((g**2).sum()) + factor_part + p.resid_vol**2
        ic.append([float(gj / math.sqrt(total)) for gj in g])
    return {
        "covariance": cov,
        "population_ic": ic,  # [regime][signal]; spanned signals also load on the factors
        "true_signals": sorted({j for g in gamma for j, v in enumerate(g) if v != 0.0}),
        "spurious_signals": sorted(
            {j for j in range(gamma.shape[1]) if not gamma[:, j].any()}
        ),
        "spanned_signals": sorted(p.spanned),
        "publication_lag": p.publication_lag,
    }


def rank_ic(signal: np.ndarray, forward: np.ndarray) -> float:
    """Spearman correlation of one date's signal and forward returns, NaNs dropped."""
    ok = ~(np.isnan(signal) | np.isnan(forward))
    if ok.sum() < 3:
        return float("nan")
    a = stats.rankdata(signal[ok])
    b = stats.rankdata(forward[ok])
    return float(np.corrcoef(a, b)[0, 1])
