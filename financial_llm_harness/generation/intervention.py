"""Interventions with declared closure, and control-arm states (spec §6.4, §6.5, §22).

do(X := x′) is applied, then the derived descendants of X are recomputed, except
nodes held fixed, each of which needs a declared compensating primitive. The
structural graph for closure is the derived-node ``parents`` graph; a primitive's
``depends_on`` is a joint-sampling dependency, so such a primitive is exogenous
once drawn. The validator asserts
``diff(A, B) == {X} ∪ (desc(X) \\ held_fixed) ∪ compensate``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.generation.scenario import (
    Value,
    World,
    parse_dist,
    quantise,
    sample_dist,
)


@dataclass(frozen=True)
class InterventionSpec:
    op: str
    node: str
    via: str | None
    form: str
    baseline: Any
    counterfactual: Any
    delta: float | None
    relative_delta: float | None
    recompute: tuple[str, ...]
    held_fixed: tuple[str, ...]
    compensate: tuple[str, ...]
    counterbalance: bool

    @property
    def target(self) -> str:
        """X: the node set directly (the ``via`` primitive when one is declared)."""
        return self.via or self.node


def parse_intervention_spec(d: Mapping[str, Any]) -> InterventionSpec:
    allowed = {
        "op",
        "node",
        "via",
        "form",
        "baseline",
        "counterfactual",
        "delta",
        "relative_delta",
        "closure",
        "counterbalance",
    }
    extra = set(d) - allowed
    if extra:
        raise HarnessValidationError(f"intervention_spec: unknown keys {sorted(extra)}")
    for key in ("op", "node", "form", "closure"):
        if key not in d:
            raise HarnessValidationError(f"intervention_spec: missing key {key!r}")
    form = d["form"]
    if form not in ("set", "shift"):
        raise HarnessValidationError(f"intervention_spec.form {form!r} invalid")
    counterbalance = bool(d.get("counterbalance", False))
    if counterbalance and form == "set":
        raise HarnessValidationError(
            "counterbalance: true is allowed only with form: shift (§6.5)"
        )
    if form == "shift" and d.get("delta") is None and d.get("relative_delta") is None:
        raise HarnessValidationError("form: shift needs delta or relative_delta")
    c = d["closure"]
    if not isinstance(c, Mapping) or set(c) != {
        "recompute",
        "held_fixed",
        "compensate",
    }:
        raise HarnessValidationError(
            "intervention_spec.closure needs recompute, held_fixed, compensate"
        )
    return InterventionSpec(
        op=d["op"],
        node=d["node"],
        via=d.get("via"),
        form=form,
        baseline=d.get("baseline"),
        counterfactual=d.get("counterfactual"),
        delta=d.get("delta"),
        relative_delta=d.get("relative_delta"),
        recompute=tuple(c["recompute"] or ()),
        held_fixed=tuple(c["held_fixed"] or ()),
        compensate=tuple(c["compensate"] or ()),
        counterbalance=counterbalance,
    )


def structural_descendants(world: World, name: str) -> set[str]:
    """Derived descendants of ``name`` through ``parents`` edges, within the world."""
    out: set[str] = set()
    stack = [name]
    while stack:
        cur = stack.pop()
        for n in world.nodes.values():
            if n.kind == "derived" and cur in n.parents and n.name not in out:
                out.add(n.name)
                stack.append(n.name)
    return out


def _invert(
    world: World, node: str, via: str, state: Mapping[str, Value], target: Any
) -> Value:
    fname = f"invert_{node}_via_{via}"
    fn = world.fns(fname)
    value = fn(state, target)
    return quantise(world.nodes[via], value)


def set_node(
    world: World,
    base: Mapping[str, Value],
    node: str,
    via: str | None,
    value: Any,
    held_fixed: tuple[str, ...] = (),
    compensate: tuple[str, ...] = (),
) -> dict[str, Value]:
    """A copy of ``base`` with do(node := value), closure recomputed."""
    if node not in world.nodes:
        raise HarnessValidationError(
            f"intervention node {node!r} is not in the experiment's world"
        )
    s = dict(copy.deepcopy(dict(base)))
    if via is not None:
        if via not in world.nodes or world.nodes[via].kind != "primitive":
            raise HarnessValidationError(
                f"via {via!r} must be a primitive of the world"
            )
        if world.nodes[node].kind != "derived":
            raise HarnessValidationError(
                f"intervention via {via!r} needs a derived node, got {node!r}"
            )
        s[via] = _invert(world, node, via, base, value)
        x = via
    else:
        n = world.nodes[node]
        if n.kind == "latent":
            raise HarnessValidationError("a latent node cannot be intervened on")
        if n.kind == "derived":
            raise HarnessValidationError(f"derived node {node!r} needs a via primitive")
        s[node] = quantise(n, value)
        x = node
    desc = structural_descendants(world, x)
    held = set(held_fixed)
    if len(held_fixed) != len(compensate):
        raise HarnessValidationError(
            "each held_fixed node needs one compensating primitive (§6.4)"
        )
    for h, c in zip(held_fixed, compensate, strict=True):
        if h not in desc:
            raise HarnessValidationError(
                f"held_fixed {h!r} is not a descendant of {x!r}"
            )
        s[c] = _invert(world, h, c, s, base[h])
    to_recompute = desc | structural_descendants_many(world, compensate)
    for name in world.order:
        nd = world.nodes[name]
        if nd.kind != "derived" or name in held or name not in to_recompute:
            continue
        s[name] = world.fns(nd.fn or "")(**{p: s[p] for p in nd.parents})
    for h in held:
        nd = world.nodes[h]
        recomputed = world.fns(nd.fn or "")(**{p: s[p] for p in nd.parents})
        if not _close(recomputed, base[h]):
            raise HarnessValidationError(f"compensation failed to hold {h!r} fixed")
        s[h] = base[h]
    return s


def structural_descendants_many(world: World, names: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for n in names:
        out |= structural_descendants(world, n)
    return out


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def diff_nodes(
    world: World, a: Mapping[str, Value], b: Mapping[str, Value]
) -> set[str]:
    """Nodes (latent excluded) whose values differ between two states."""
    return {
        n
        for n, nd in world.nodes.items()
        if nd.kind != "latent" and a.get(n) != b.get(n)
    }


def declared_diff(world: World, spec: InterventionSpec) -> set[str]:
    x = spec.target
    desc = structural_descendants(world, x)
    return {x} | (desc - set(spec.held_fixed)) | set(spec.compensate)


def check_closure_declaration(world: World, spec: InterventionSpec) -> None:
    """The declared ``recompute`` list equals desc(X) \\ held_fixed exactly (§6.4)."""
    x = spec.target
    desc = structural_descendants(world, x)
    expected = desc - set(spec.held_fixed)
    if set(spec.recompute) != expected:
        raise HarnessValidationError(
            f"closure.recompute {sorted(spec.recompute)} != desc({x}) \\ held_fixed {sorted(expected)}"
        )
    if spec.via is not None and spec.node not in desc:
        raise HarnessValidationError(
            f"{spec.node!r} is not a descendant of via {spec.via!r}"
        )


def arm_values(
    world: World, spec: InterventionSpec, base: Mapping[str, Value], case_idx: int
) -> tuple[Any, Any, bool]:
    """(value in A, value in B, counterbalanced) for the intervened node (§6.5)."""
    if spec.form == "set":
        return spec.baseline, spec.counterfactual, False
    x = base[spec.node]
    if not isinstance(x, int | float):
        raise HarnessValidationError("form: shift needs a numeric node")
    xf = float(x)
    flipped = spec.counterbalance and case_idx % 2 == 1
    if spec.delta is not None:
        d = float(spec.delta)
        return (xf - d, xf, True) if flipped else (xf, xf + d, False)
    r = float(spec.relative_delta or 0.0)
    return (xf / (1 + r), xf, True) if flipped else (xf, xf * (1 + r), False)


def apply_intervention(
    world: World, spec: InterventionSpec, base: Mapping[str, Value], case_idx: int
) -> tuple[dict[str, Value], dict[str, Value], bool]:
    """Arm A and arm B states, and whether the case is counterbalanced."""
    va, vb, flipped = arm_values(world, spec, base, case_idx)
    if spec.form == "shift" and not flipped:
        sa = dict(base)  # arm A keeps the sampled value
    else:
        sa = set_node(
            world, base, spec.node, spec.via, va, spec.held_fixed, spec.compensate
        )
    if spec.form == "shift" and flipped:
        sb = dict(base)  # counterbalanced: arm B keeps the sampled value
    else:
        sb = set_node(
            world, base, spec.node, spec.via, vb, spec.held_fixed, spec.compensate
        )
    return sa, sb, flipped


def placebo_state(
    world: World,
    a_state: Mapping[str, Value],
    node: str,
    values: Any,
    rng: np.random.Generator,
) -> dict[str, Value]:
    """Arm A with the placebo node changed to a different value (§22)."""
    if node not in world.nodes:
        raise HarnessValidationError(f"placebo node {node!r} is not in the world")
    if isinstance(values, list):
        candidates = [v for v in values if v != a_state.get(node)]
        if not candidates:
            raise HarnessValidationError("placebo values offer no alternative")
        new = candidates[int(rng.integers(0, len(candidates)))]
    else:
        dist = parse_dist(str(values))
        new = a_state.get(node)
        for _ in range(200):
            new = sample_dist(dist, rng, a_state, world.fns)
            if new != a_state.get(node):
                break
        else:
            raise HarnessValidationError("placebo values never differ from arm A")
    return set_node(world, a_state, node, None, new)
