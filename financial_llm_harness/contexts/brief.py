"""C6: quant research brief (spec §T3; S3 and S4 are C6 slices, §35).

A brief is rendered from state fields; the values that determine the
methodological oracle (e.g. T/N, the label horizon, a publication lag) are
the planted truth. ``determining_values`` reads them from a state and
``unrendered`` names any that a brief would not show, which §9.1
answerability forbids.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from financial_llm_harness.errors import HarnessValidationError

# §T3 C6: what a brief carries
BRIEF_FIELDS = (
    "universe",
    "n_assets",
    "n_obs",
    "frequency",
    "objective",
    "constraints",
    "costs",
    "data_availability",
    "intended_use",
)


def determining_values(
    state: Mapping[str, Any], determining: Sequence[str]
) -> dict[str, Any]:
    """The planted truth: the state's values of the oracle-determining fields."""
    missing = [n for n in determining if n not in state]
    if missing:
        raise HarnessValidationError(f"C6: the state has no {missing}")
    return {n: state[n] for n in determining}


def unrendered(rendered: Iterable[str], determining: Sequence[str]) -> list[str]:
    shown = set(rendered)
    return [n for n in determining if n not in shown]
