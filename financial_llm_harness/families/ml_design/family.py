"""ml_design: bank draws and the fake response (spec §6, §20.11, §35 S4)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from financial_llm_harness.families import bank_entries

FAMILY_ID = "ml_design"
POINT_IN_TIME = ("expanding", "rolling", "cross_sectional_at_t")
OTHERS = ("full_sample", "none")


def draw_vendor_label(rng: np.random.Generator, drawn: Mapping[str, Any]) -> str:
    entries = bank_entries(FAMILY_ID, "vendor_labels")
    return str(entries[int(rng.integers(0, len(entries)))])


_RATIONALE = (
    "The window follows from how the normalised features will be used.",
    "This keeps the feature scales comparable across assets and dates.",
    "The choice balances stability of the estimates against responsiveness.",
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
    if contract_id != "temporal_norm_v1":
        raise ValueError(f"fake_response: unknown contract {contract_id!r}")
    pool = POINT_IN_TIME if int(y) == 1 else OTHERS
    window = pool[int(rng.integers(0, len(pool)))]
    future = window == "full_sample"
    text = _RATIONALE[int(rng.integers(0, len(_RATIONALE)))]
    if reasoning_mode == "reasoning_before":
        body: dict[str, Any] = {
            "rationale": text,
            "normalisation_window": window,
            "uses_future_data": future,
        }
    elif reasoning_mode == "rationale_after":
        body = {
            "normalisation_window": window,
            "uses_future_data": future,
            "rationale": text,
        }
    else:
        body = {"normalisation_window": window, "uses_future_data": future}
    return json.dumps(body)
