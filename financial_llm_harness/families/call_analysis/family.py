"""call_analysis: the S2 transcript block and the fake response (spec §20.11, §35 S2).

The call is a block of items rendered from the state: the eleven items other
than the guidance cut in a per-case order (from the latent ``order_draw``),
the two immaterial items swapped when ``immaterial_order`` is ``swapped`` (the
placebo), and the guidance cut inserted at ``guidance_position`` (1-based)
unless ``guidance_included`` is ``no`` (the positive control's REMOVE). Each
item is one ``block`` span, so arms differ only in item order or presence
(``corpus.check_block_isolation``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from financial_llm_harness.core.ids import uint64
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.families import NodeDef, bank_entries
from financial_llm_harness.generation.render import format_value

FAMILY_ID = "call_analysis"
BLOCK = "items"
KEY_ITEM = "guidance_cut"


def item_bank() -> list[dict[str, Any]]:
    """The bank's item types in materiality-rank order (rank 1 first)."""
    entries = [dict(e) for e in bank_entries(FAMILY_ID, "transcript_items")]
    return sorted(entries, key=lambda e: int(e["rank"]))


def _types() -> list[str]:
    return [str(e["type"]) for e in item_bank()]


def immaterial_types() -> tuple[str, str]:
    """The two lowest-ranked item types, which the placebo swaps."""
    types = _types()
    return types[-1], types[-2]


TYPES = (
    "guidance_cut",
    "segment_results",
    "gross_margin",
    "capex",
    "buyback",
    "dividend",
    "debt_refinancing",
    "cost_programme",
    "management_change",
    "acquisition",
    "office_relocation",
    "sustainability_report",
)
# state nodes the block shows (§9.1 answerability and non-triviality, §9.2 near-duplicates)
BLOCK_NODES = {
    BLOCK: [
        *(f"{p}_{t}" for t in TYPES for p in ("s", "v1", "v2")),
        "guidance_position",
        "immaterial_order",
        "guidance_included",
    ]
}


def item_order(state: Mapping[str, Any]) -> list[str]:
    """The arm's item types, in the order the call presents them."""
    others = [t for t in _types() if t != KEY_ITEM]
    rng = np.random.default_rng(
        uint64("transcript_order", repr(float(state["order_draw"])))
    )
    order = [others[int(i)] for i in rng.permutation(len(others))]
    if state["immaterial_order"] == "swapped":
        a, b = immaterial_types()
        i, j = order.index(a), order.index(b)
        order[i], order[j] = order[j], order[i]
    elif state["immaterial_order"] != "as_listed":
        raise HarnessValidationError(f"immaterial_order {state['immaterial_order']!r}")
    if state["guidance_included"] == "yes":
        pos = int(state["guidance_position"])
        if not 1 <= pos <= len(others) + 1:
            raise HarnessValidationError(
                f"guidance_position {pos} outside 1..{len(others) + 1}"
            )
        order.insert(pos - 1, KEY_ITEM)
    elif state["guidance_included"] != "no":
        raise HarnessValidationError(
            f"guidance_included {state['guidance_included']!r}"
        )
    return order


def render_blocks(
    state: Mapping[str, Any], nodes: Mapping[str, NodeDef]
) -> dict[str, list[tuple[str, str | None]]]:
    """The ``items`` block: one bullet line per item, each its own span (§35 S2)."""
    bank = {str(e["type"]): e for e in item_bank()}
    order = item_order(state)
    lines: list[tuple[str, str | None]] = []
    for k, t in enumerate(order):
        sentences = list(bank[t]["sentences"])
        idx = int(state[f"s_{t}"])
        if not 0 <= idx < len(sentences):
            raise HarnessValidationError(
                f"s_{t} = {idx} outside the bank's {len(sentences)} sentences"
            )
        text = str(sentences[idx]).format(
            v1=format_value(nodes[f"v1_{t}"], state[f"v1_{t}"]),
            v2=format_value(nodes[f"v2_{t}"], state[f"v2_{t}"]),
        )
        end = "\n" if k < len(order) - 1 else ""
        lines.append((f"- {text}{end}", f"s_{t}"))
    return {BLOCK: lines}


# ---------------------------------------------------------------------------
# Fake response (§20.11)
# ---------------------------------------------------------------------------

_GUIDANCE_POINTS = (
    "Full-year revenue guidance was cut.",
    "Management lowered its guidance for the year.",
    "The revenue guidance was reduced.",
)
_OTHER_POINTS = (
    "Segment revenue grew.",
    "Gross margin improved.",
    "A share buyback was announced.",
    "The dividend was raised.",
    "Capital spending plans were set out.",
    "Debt was refinanced.",
)
_SUMMARIES = (
    "Management reviewed the quarter's results and the main corporate developments.",
    "The call covered trading, capital allocation and several corporate updates.",
    "Results and capital plans were discussed, together with management updates.",
)


def fake_response(
    contract_id: str,
    y: int | float,
    mention: bool,
    rng: np.random.Generator,
    concept_factor: str | None = None,
    reasoning_mode: str = "none",
) -> str:
    """A ``summary_v1`` response whose key points report the guidance cut iff ``y`` is 1."""
    del mention, concept_factor, reasoning_mode
    if contract_id != "summary_v1":
        raise ValueError(f"fake_response: unknown contract {contract_id!r}")
    picks = [int(i) for i in rng.permutation(len(_OTHER_POINTS))[:3]]
    points = [_OTHER_POINTS[i] for i in picks]
    if int(y) == 1:
        points.insert(0, _GUIDANCE_POINTS[int(rng.integers(0, len(_GUIDANCE_POINTS)))])
    summary = _SUMMARIES[int(rng.integers(0, len(_SUMMARIES)))]
    return json.dumps({"summary": summary, "key_points": points})
