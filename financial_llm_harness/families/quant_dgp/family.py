"""quant_dgp: Q1–Q6 items on a synthetic dataset with a known DGP (spec §34, §T9.2).

``dgp_params`` reads the C7 parameters from the state and ``dataset`` draws
the panel with the §12 dataset seed; the truth (population IC, true and
spurious signals, true covariance) comes from the parameters, never from the
draws. ``mp_upper_edge`` is the Marchenko–Pastur upper edge σ²(1 + √(N/T))²
(B.mp1967), the identity Q2.2.b's planted edge rests on. Parameter values are
the base design's (D-23).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from financial_llm_harness.contexts import dgp as C7
from financial_llm_harness.core.ids import uint64
from financial_llm_harness.errors import HarnessValidationError

FAMILY_ID = "quant_dgp"


def mp_upper_edge(n_assets: float, n_obs: float, variance: float) -> float:
    """λ₊ = σ²(1 + √(N/T))² for N assets and T observations (B.mp1967)."""
    if n_assets <= 0 or n_obs <= 0 or variance <= 0:
        raise HarnessValidationError("Marchenko–Pastur edge needs positive N, T and σ²")
    return variance * (1.0 + math.sqrt(n_assets / n_obs)) ** 2


def dgp_params(state: Mapping[str, Any]) -> C7.DGPParams:
    k = int(state["n_factors"])
    j = int(state["n_signals"])
    gamma = [[float(state[f"gamma_{i}"]) for i in range(1, j + 1)]]
    return C7.DGPParams(
        n_assets=int(state["n_assets"]),
        n_dates=int(state["n_dates"]),
        factor_vol=[float(state[f"factor_vol_{i}"]) for i in range(1, k + 1)],
        beta_sd=float(state["beta_sd"]),
        gamma=gamma,
        resid_vol=float(state["resid_vol"]),
        t_dof=float(state["t_dof"]),
        publication_lag=int(state.get("publication_lag", 0)),
    )


def dataset(
    state: Mapping[str, Any], seed_stream: str, generation_seed: int, case_idx: int
) -> C7.Panel:
    """The case's panel, seeded per §12 (never from a hash-derived seed)."""
    rng = np.random.default_rng(
        uint64("dataset", seed_stream, generation_seed, case_idx)
    )
    return C7.simulate(dgp_params(state), rng)
