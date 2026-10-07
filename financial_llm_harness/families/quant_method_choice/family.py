"""quant_method_choice: derived nodes, bank draws and the fake response (spec §6, §20.11, §35 S3)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from financial_llm_harness.families import bank_entries

FAMILY_ID = "quant_method_choice"
WELL_CONDITIONED = ("ledoit_wolf_linear", "nonlinear_shrinkage", "factor_model")
OTHERS = ("sample", "ewma", "other")


def _draw(name: str, rng: np.random.Generator) -> str:
    entries = bank_entries(FAMILY_ID, name)
    return str(entries[int(rng.integers(0, len(entries)))])


def draw_universe_label(rng: np.random.Generator, drawn: Mapping[str, Any]) -> str:
    return _draw("universe_labels", rng)


def n_obs(t_over_n: float, n_assets: float) -> float:
    """T = round(t_over_n · N), the number of return observations per asset."""
    return float(round(float(t_over_n) * float(n_assets)))


_JUSTIFICATION = (
    "The choice follows from the number of observations relative to the number of assets.",
    "The estimator suits the stated use and the length of the return history.",
    "This balances estimation error against the needs of the optimiser.",
)


def fake_response(
    contract_id: str,
    y: int | float,
    mention: bool,
    rng: np.random.Generator,
    concept_factor: str | None = None,
    reasoning_mode: str = "rationale_after",
) -> str:
    """A contract-valid response whose primary feature equals ``y``."""
    del mention, concept_factor
    if contract_id != "covariance_choice_v1":
        raise ValueError(f"fake_response: unknown contract {contract_id!r}")
    pool = WELL_CONDITIONED if int(y) == 1 else OTHERS
    estimator = pool[int(rng.integers(0, len(pool)))]
    text = _JUSTIFICATION[int(rng.integers(0, len(_JUSTIFICATION)))]
    if reasoning_mode == "reasoning_before":
        body: dict[str, Any] = {"justification": text, "estimator": estimator}
    elif reasoning_mode == "rationale_after":
        body = {"estimator": estimator, "justification": text}
    else:
        body = {"estimator": estimator}
    return json.dumps(body)
