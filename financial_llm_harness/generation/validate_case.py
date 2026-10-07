"""Case-level validation (spec §9.1).

A failing case is rejected with VALIDATION_REJECTED and a reason; the reason
counts towards the 30% rejection limit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.families import NodeDef
from financial_llm_harness.generation.render import (
    Rendered,
    displayed_number,
    numbers_in,
)


class CaseRejected(Exception):
    """A case failed a §9.1 check; ``reason`` is the rejection category."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def check_round_trip(
    rendered: Rendered,
    nodes: Mapping[str, NodeDef],
    state: Mapping[str, Any],
    display_overrides: Mapping[str, str] | None = None,
    precision_overrides: Mapping[str, int] | None = None,
) -> None:
    """Numbers parsed from each state slot equal the displayed state value (§9.1)."""
    display_overrides = display_overrides or {}
    precision_overrides = precision_overrides or {}
    for span in rendered.spans:
        if span.kind not in ("state", "block") or span.node is None:
            continue
        node = nodes.get(span.node)
        if node is None:
            continue
        text = rendered.user[span.start : span.end]
        got = numbers_in(text)
        if node.type == "categorical" or node.display == "text":
            if span.kind == "state" and got:
                raise CaseRejected(
                    "paraphrase_fidelity", f"text slot {span.node} shows digits"
                )
            continue
        if span.kind == "block":
            continue
        disp = display_overrides.get(span.node)
        prec = precision_overrides.get(span.node, node.precision)
        expected = displayed_number(node, state[span.node], disp, prec)
        if expected is None or len(got) != 1:
            raise CaseRejected("paraphrase_fidelity", f"slot {span.node} shows {got}")
        tol = 0.5 * 10 ** (-prec) + 1e-9
        shown = -abs(got[0]) if text.lstrip().startswith("-") else abs(got[0])
        if abs(shown - expected) > tol:
            raise CaseRejected(
                "paraphrase_fidelity",
                f"slot {span.node}: shown {shown}, state {expected}",
            )


def masked(rendered: Rendered, nodes_allowed: set[str], kinds_allowed: set[str]) -> str:
    """The user text with every allowed span replaced by a marker.

    Adjacent allowed spans (the items of one block, §35 S2) share one marker, so
    a block whose items are reordered or removed masks alike; what the block may
    hold is checked separately (``corpus.check_block_isolation``).
    """
    text = rendered.user
    out: list[str] = []
    pos = 0
    last_end = -1
    for span in sorted(rendered.spans, key=lambda s: s.start):
        if span.kind in kinds_allowed or (
            span.node is not None and span.node in nodes_allowed
        ):
            if span.start == last_end and span.start == pos:
                pos = span.end
                last_end = span.end
                continue
            out.append(text[pos : span.start])
            out.append("\x00")
            pos = span.end
            last_end = span.end
    out.append(text[pos:])
    return "".join(out)


def check_text_isolation(
    rx: Rendered,
    ry: Rendered,
    diff_nodes: set[str],
    extra_kinds: Iterable[str] = (),
) -> None:
    """The rendered-text diff of two arms is limited to the spans of diff fields (§9.1)."""
    kinds = set(extra_kinds)
    if rx.system != ry.system:
        raise CaseRejected("isolation", "system prompts differ")
    if masked(rx, diff_nodes, kinds) != masked(ry, diff_nodes, kinds):
        raise CaseRejected("isolation", "rendered text differs outside the diff spans")


def check_state_isolation(diff: set[str], declared: set[str], label: str) -> None:
    if diff != declared:
        raise CaseRejected(
            "isolation",
            f"{label}: state diff {sorted(diff)} != declared closure {sorted(declared)}",
        )


def check_neutrality(rendered: Rendered, leak_terms: Sequence[str]) -> None:
    """No leak term appears outside the mandate text (§9.1)."""
    text = masked(rendered, set(), {"mandate"}).lower()
    for term in leak_terms:
        if re.search(r"(?<![a-z])" + re.escape(term.lower()) + r"(?![a-z])", text):
            raise CaseRejected("neutrality", f"leak term {term!r}")


def check_answerability(rendered_nodes: set[str], referenced: Iterable[str]) -> None:
    missing = [n for n in referenced if n not in rendered_nodes]
    if missing:
        raise HarnessValidationError(
            f"answerability: oracle-referenced nodes are not rendered: {missing} (§9.1)"
        )


def check_measurability(feature: Mapping[str, Any]) -> None:
    ftype = feature.get("type")
    if ftype not in ("binary", "categorical", "ordinal", "scalar"):
        raise HarnessValidationError(
            f"primary feature type {ftype!r} is not measurable (§9.1)"
        )
    if ftype == "scalar":
        rng = feature.get("range")
        if not (isinstance(rng, list) and len(rng) == 2):
            raise HarnessValidationError(
                "a scalar primary feature needs a bounded range (§9.1)"
            )
    if ftype in ("categorical", "ordinal") and not feature.get("allowed_values"):
        raise HarnessValidationError(
            "a categorical primary feature needs allowed_values (§9.1)"
        )
