"""Rendering: template p filled with state(i, a), with recorded slot spans (spec §6.3, §8, §19).

Rendering only formats state values; it never computes. Field order is the
template's declared order, permuted per case (identical across a case's arms)
under ``field_order_policy: randomised_across_cases``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from financial_llm_harness.core.ids import uint64
from financial_llm_harness.core.schema import as_list, as_mapping, as_str, check_keys
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.families import NodeDef

CURRENCY = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_NUMBER_IN_TEXT = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+")
FIXED_SLOTS = ("preamble", "mandate", "fields", "contract", "note")


@dataclass(frozen=True)
class FieldSlot:
    slot: str
    label: str


@dataclass(frozen=True)
class BaseTemplate:
    template_family_id: str
    version: int
    system: str
    field_line: str
    fields: tuple[FieldSlot, ...]
    literal_numerals: tuple[str, ...]
    path: str


@dataclass(frozen=True)
class Paraphrase:
    paraphrase_id: str
    template_family_id: str
    paraphrase_family: str
    user: str
    literal_numerals: tuple[str, ...]
    path: str


def parse_base_template(data: Mapping[str, Any], path: str) -> BaseTemplate:
    check_keys(
        data,
        path,
        required=(
            "provenance",
            "template_family_id",
            "version",
            "system",
            "field_line",
            "fields",
        ),
        optional=("literal_numerals", "description"),
    )
    fl = as_str(data["field_line"], path + ".field_line")
    if fl.count("{value}") != 1:
        raise HarnessValidationError(f"{path}: field_line needs exactly one {{value}}")
    fields = []
    for i, f in enumerate(as_list(data["fields"], path + ".fields")):
        fm = as_mapping(f, f"{path}.fields[{i}]")
        check_keys(fm, f"{path}.fields[{i}]", required=("slot", "label"))
        fields.append(FieldSlot(as_str(fm["slot"], path), as_str(fm["label"], path)))
    return BaseTemplate(
        template_family_id=as_str(data["template_family_id"], path),
        version=int(data["version"]),
        system=as_str(data["system"], path + ".system"),
        field_line=fl,
        fields=tuple(fields),
        literal_numerals=tuple(str(x) for x in data.get("literal_numerals") or ()),
        path=path,
    )


def parse_paraphrase(data: Mapping[str, Any], path: str) -> Paraphrase:
    check_keys(
        data,
        path,
        required=(
            "provenance",
            "paraphrase_id",
            "template_family_id",
            "paraphrase_family",
            "user",
        ),
        optional=("literal_numerals", "description"),
    )
    return Paraphrase(
        paraphrase_id=as_str(data["paraphrase_id"], path),
        template_family_id=as_str(data["template_family_id"], path),
        paraphrase_family=as_str(data["paraphrase_family"], path),
        user=as_str(data["user"], path + ".user"),
        literal_numerals=tuple(str(x) for x in data.get("literal_numerals") or ()),
        path=path,
    )


# ---------------------------------------------------------------------------
# Display formats (§6.1)
# ---------------------------------------------------------------------------


def display_scale(display: str) -> float:
    return {"pct": 100.0, "money_m": 1e-6, "money_bn": 1e-9}.get(display, 1.0)


def format_value(
    node: NodeDef, value: Any, display: str | None = None, precision: int | None = None
) -> str:
    """Format a state value; never computes beyond unit scaling."""
    disp = display or node.display
    prec = node.precision if precision is None else precision
    if disp == "text" or node.type == "categorical":
        return str(value)
    v = float(value)
    if disp == "int":
        return f"{int(round(v)):,}"
    shown = v * display_scale(disp)
    if disp in ("money", "money_m", "money_bn"):
        unit = node.unit or "USD"
        sym = CURRENCY.get(unit, unit + " ")
        sign = "-" if shown < 0 else ""
        body = f"{abs(shown):,.{prec}f}"
        suffix = {"money": "", "money_m": "m", "money_bn": "bn"}[disp]
        return f"{sign}{sym}{body}{suffix}"
    if disp == "pct":
        return f"{shown:.{prec}f}%"
    if disp == "multiple":
        return f"{shown:.{prec}f}×"
    return f"{shown:,.{prec}f}"


def displayed_number(
    node: NodeDef, value: Any, display: str | None = None, precision: int | None = None
) -> float | None:
    """The number a reader sees, in display units, rounded to display precision."""
    disp = display or node.display
    if disp == "text" or node.type == "categorical":
        return None
    prec = node.precision if precision is None else precision
    v = float(value)
    if disp == "int":
        return float(int(round(v)))
    return round(v * display_scale(disp), prec)


def numbers_in(text: str) -> list[float]:
    return [float(m.group(0).replace(",", "")) for m in _NUMBER_IN_TEXT.finditer(text)]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    slot: str
    kind: str  # state | mandate | preamble | contract | block
    node: str | None
    start: int
    end: int


@dataclass
class RenderContext:
    state: Mapping[str, Any]
    nodes: Mapping[str, NodeDef]
    mandate: str
    contract: str
    preamble: str = ""
    field_order: Sequence[str] | None = None
    display_overrides: dict[str, str] = field(default_factory=dict)
    precision_overrides: dict[str, int] = field(default_factory=dict)
    blocks: dict[str, list[tuple[str, str | None]]] = field(default_factory=dict)
    note: str = (
        ""  # the positive control's PCB-only clause (§35 S3–S5); empty elsewhere
    )


@dataclass
class Rendered:
    system: str
    user: str
    spans: list[Span]

    def state_spans(self) -> list[Span]:
        return [s for s in self.spans if s.kind == "state"]


class _Builder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.pos = 0
        self.spans: list[Span] = []

    def lit(self, text: str) -> None:
        self.parts.append(text)
        self.pos += len(text)

    def slot(self, text: str, slot: str, kind: str, node: str | None) -> None:
        start = self.pos
        self.lit(text)
        self.spans.append(Span(slot, kind, node, start, self.pos))

    def text(self) -> str:
        return "".join(self.parts)


def field_order(
    base: BaseTemplate, policy: str, paraphrase_seed: int, case_idx: int
) -> list[str]:
    slots = [f.slot for f in base.fields]
    if policy == "fixed":
        return slots
    if policy != "randomised_across_cases":
        raise HarnessValidationError(f"field_order_policy {policy!r} invalid")
    perm = np.random.default_rng(
        uint64("field_order", paraphrase_seed, case_idx, base.template_family_id)
    ).permutation(len(slots))
    return [slots[int(i)] for i in perm]


def render(base: BaseTemplate, para: Paraphrase, ctx: RenderContext) -> Rendered:
    """Fill paraphrase ``user`` with the context; record every slot span."""
    labels = {f.slot: f.label for f in base.fields}
    order = (
        list(ctx.field_order)
        if ctx.field_order is not None
        else [f.slot for f in base.fields]
    )
    pre_line, _, post_line = base.field_line.partition("{value}")
    b = _Builder()
    pos = 0
    for m in _PLACEHOLDER.finditer(para.user):
        b.lit(para.user[pos : m.start()])
        name = m.group(1)
        if name == "preamble":
            if ctx.preamble:
                b.slot(ctx.preamble, "preamble", "preamble", None)
        elif name == "mandate":
            b.slot(ctx.mandate, "mandate", "mandate", None)
        elif name == "contract":
            b.slot(ctx.contract, "contract", "contract", None)
        elif name == "note":
            # always a span, possibly empty, so arms with and without the note mask alike
            b.slot(ctx.note, "note", "note", None)
        elif name == "fields":
            for i, slot in enumerate(order):
                if slot not in labels:
                    raise HarnessValidationError(
                        f"{base.path}: field slot {slot!r} undeclared"
                    )
                if i:
                    b.lit("\n")
                b.lit(pre_line.replace("{label}", labels[slot]))
                b.slot(_slot_text(slot, ctx), slot, "state", slot)
                b.lit(post_line.replace("{label}", labels[slot]))
        elif name in ctx.blocks:
            for text, node in ctx.blocks[name]:
                if node is None:
                    b.lit(text)
                else:
                    b.slot(text, name, "block", node)
        elif name in ctx.nodes:
            b.slot(_slot_text(name, ctx), name, "state", name)
        else:
            raise HarnessValidationError(f"{para.path}: unknown placeholder {{{name}}}")
        pos = m.end()
    b.lit(para.user[pos:])
    return Rendered(system=base.system, user=b.text(), spans=b.spans)


def _slot_text(node_name: str, ctx: RenderContext) -> str:
    if node_name not in ctx.nodes:
        raise HarnessValidationError(f"slot {node_name!r} is not a node of the world")
    node = ctx.nodes[node_name]
    if node.kind == "latent":
        raise HarnessValidationError(f"latent node {node_name!r} is never rendered")
    return format_value(
        node,
        ctx.state[node_name],
        ctx.display_overrides.get(node_name),
        ctx.precision_overrides.get(node_name),
    )


def rendered_nodes(
    base: BaseTemplate,
    para: Paraphrase,
    extra_blocks: Mapping[str, Sequence[str]] | None = None,
) -> set[str]:
    """Nodes a (base, paraphrase) pair renders."""
    out: set[str] = set()
    for m in _PLACEHOLDER.finditer(para.user):
        name = m.group(1)
        if name == "fields":
            out |= {f.slot for f in base.fields}
        elif name not in FIXED_SLOTS:
            if extra_blocks and name in extra_blocks:
                out |= set(extra_blocks[name])
            else:
                out.add(name)
    return out


def literal_text(base: BaseTemplate, para: Paraphrase) -> str:
    """Template literal text: everything outside slots, labels included."""
    user_lit = _PLACEHOLDER.sub(" ", para.user)
    field_lit = base.field_line.replace("{value}", " ").replace("{label}", " ")
    labels = " ".join(f.label for f in base.fields)
    return "\n".join([base.system, user_lit, field_lit, labels])


def check_literal_numerals(base: BaseTemplate, para: Paraphrase) -> None:
    """Template literal text holds no digit except its declared ``literal_numerals`` (§9.1)."""
    text = literal_text(base, para)
    allowed = sorted(
        {*base.literal_numerals, *para.literal_numerals}, key=len, reverse=True
    )
    for tok in allowed:
        text = text.replace(tok, " ")
    if re.search(r"\d", text):
        bad = re.findall(r"\S*\d\S*", text)
        raise HarnessValidationError(
            f"{para.path}: literal text has undeclared digits {bad[:5]} (§9.1 paraphrase fidelity)"
        )
