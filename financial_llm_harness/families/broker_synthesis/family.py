"""broker_synthesis: F7 summarisation and discrepancy items on C5 and C5k (spec §34, §T9.1).

``note_set`` rebuilds a case's planted note set from its state (the set
parameters and a seed node), so the truth is a function of the state; the
D-type features score a parsed ``findings`` list against it by the
deterministic matching of §T9.1 step 4. The consensus, history, metric and
observation pools are the base design's banks (D-23), passed in by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from financial_llm_harness.contexts import broker_notes as C5
from financial_llm_harness.core.ids import uint64

FAMILY_ID = "broker_synthesis"
SET_NODES = (
    "k_notes",
    "noise",
    "d",
    "k",
    "a",
    "n",
    "difference_offset",
    "inaccuracy_offset",
    "rounding",
)


def set_params(state: Mapping[str, Any]) -> C5.SetParams:
    return C5.SetParams(
        k_notes=int(state["k_notes"]),
        eps=float(state["noise"]),
        d=int(state["d"]),
        k=int(state["k"]),
        a=int(state["a"]),
        n=int(state["n"]),
        difference_offset=float(state["difference_offset"]),
        inaccuracy_offset=float(state["inaccuracy_offset"]),
        rounding=float(state["rounding"]),
    )


def note_set(
    state: Mapping[str, Any],
    consensus: Mapping[str, float],
    ratings: Sequence[str],
    thesis_ids: Sequence[str],
    history: Mapping[str, float],
    metrics: Sequence[str],
    novel_pool: Sequence[str],
    quote: Any,
) -> tuple[list[C5.Note], list[C5.Planted]]:
    rng = np.random.default_rng(uint64("note_set", int(state["set_seed"])))
    return C5.plant_set(
        consensus,
        ratings,
        thesis_ids,
        history,
        metrics,
        novel_pool,
        set_params(state),
        rng,
        quote,
    )


def score_findings(
    parsed: Mapping[str, Any], planted: Sequence[C5.Planted], field: str = "findings"
) -> dict[str, dict[str, float | None]]:
    """Precision, recall and F1 by type of the response's findings (§T9.1 step 5)."""
    findings = [f for f in parsed.get(field) or [] if isinstance(f, Mapping)]
    return C5.score(findings, planted)
