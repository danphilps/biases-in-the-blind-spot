"""Structured worlds: distribution and constraint grammars, joint sampling (spec §6.1, §6.2).

Distributions and constraints are strings parsed by fixed grammars, never
evaluated. Draw attempt k uses ``default_rng(uint64(seed_stream|generation|k))``
and draws nodes in topological order; a Gaussian copula couples the continuous
non-``fn`` primitives listed in ``correlation.nodes``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from financial_llm_harness.core.ids import uint64
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.families import NodeDef, ScenarioSchema

DIST_KINDS = (
    "normal",
    "lognormal",
    "uniform",
    "truncnormal",
    "beta",
    "categorical",
    "fixed",
    "fn",
)
CONTINUOUS = ("normal", "lognormal", "uniform", "truncnormal", "beta")
_PARAMS: dict[str, tuple[str, ...]] = {
    "normal": ("mu", "sigma"),
    "lognormal": ("mu", "sigma"),
    "uniform": ("low", "high"),
    "truncnormal": ("mu", "sigma", "low", "high"),
    "beta": ("a", "b", "low", "high"),
}
_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
Value = float | int | str


@dataclass(frozen=True)
class Dist:
    kind: str
    params: tuple[tuple[str, float], ...] = ()
    values: tuple[Value, ...] = ()
    probs: tuple[float, ...] = ()
    fn: str | None = None
    text: str = ""

    def p(self, name: str) -> float:
        for k, v in self.params:
            if k == name:
                return v
        raise KeyError(name)


class _Parser:
    def __init__(self, text: str) -> None:
        self.s = text
        self.i = 0

    def err(self, msg: str) -> HarnessValidationError:
        return HarnessValidationError(f"distribution {self.s!r}: {msg}")

    def ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def peek(self) -> str:
        self.ws()
        return self.s[self.i] if self.i < len(self.s) else ""

    def expect(self, ch: str) -> None:
        if self.peek() != ch:
            raise self.err(f"expected {ch!r} at {self.i}")
        self.i += 1

    def ident(self) -> str:
        self.ws()
        m = _IDENT_RE.match(self.s, self.i)
        if not m:
            raise self.err(f"expected a name at {self.i}")
        self.i = m.end()
        return m.group(0)

    def number(self) -> float:
        self.ws()
        m = _NUM_RE.match(self.s, self.i)
        if not m:
            raise self.err(f"expected a number at {self.i}")
        self.i = m.end()
        return float(m.group(0))

    def scalar(self) -> Value:
        """A number, a quoted string or a bare word."""
        ch = self.peek()
        if ch in ("'", '"'):
            end = self.s.find(ch, self.i + 1)
            if end < 0:
                raise self.err("unterminated string")
            out = self.s[self.i + 1 : end]
            self.i = end + 1
            return out
        m = _NUM_RE.match(self.s, self.i)
        if m:
            nxt = self.s[m.end() : m.end() + 1]
            if nxt in ("", ",", "]", ")", " "):
                rest = self.s[m.end() :].lstrip()
                if rest[:1] in ("", ",", "]", ")"):
                    self.i = m.end()
                    text = m.group(0)
                    return int(text) if re.fullmatch(r"[-+]?\d+", text) else float(text)
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in ",[]()=\"'":
            self.i += 1
        word = self.s[start : self.i].strip()
        if not word:
            raise self.err(f"expected a value at {start}")
        return word

    def scalar_list(self) -> list[Value]:
        self.expect("[")
        out: list[Value] = []
        if self.peek() == "]":
            self.i += 1
            return out
        while True:
            out.append(self.scalar())
            ch = self.peek()
            if ch == ",":
                self.i += 1
                continue
            if ch == "]":
                self.i += 1
                return out
            raise self.err(f"expected ',' or ']' at {self.i}")

    def at_end(self) -> None:
        self.ws()
        if self.i != len(self.s):
            raise self.err(f"unexpected text at {self.i}")


def parse_dist(text: str) -> Dist:
    """Parse a distribution string; anything outside the grammar exits 2."""
    if not isinstance(text, str):
        raise HarnessValidationError(f"distribution must be a string, got {text!r}")
    p = _Parser(text)
    kind = p.ident()
    if kind not in DIST_KINDS:
        raise p.err(f"unknown distribution {kind!r}")
    p.expect("(")
    if kind == "fn":
        name = p.ident()
        p.expect(")")
        p.at_end()
        return Dist("fn", fn=name, text=text)
    if kind == "fixed":
        save = p.i
        try:
            key = p.ident()
            if p.peek() == "=" and key == "value":
                p.i += 1
            else:
                p.i = save
        except HarnessValidationError:
            p.i = save
        value = p.scalar()
        p.expect(")")
        p.at_end()
        return Dist("fixed", values=(value,), text=text)
    if kind == "categorical":
        values: list[Value] | None = None
        probs: list[float] | None = None
        while True:
            key = p.ident()
            p.expect("=")
            if key == "values" and values is None:
                values = p.scalar_list()
            elif key == "probs" and probs is None:
                raw = p.scalar_list()
                if not all(isinstance(x, int | float) for x in raw):
                    raise p.err("probs must be numbers")
                probs = [float(x) for x in raw]
            else:
                raise p.err(f"unexpected argument {key!r}")
            ch = p.peek()
            if ch == ",":
                p.i += 1
                continue
            p.expect(")")
            break
        p.at_end()
        if not values:
            raise p.err("categorical needs a non-empty values list")
        if probs is None:
            probs = [1.0 / len(values)] * len(values)
        if len(probs) != len(values):
            raise p.err("probs and values differ in length")
        if any(x < 0 for x in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-9):
            raise p.err("probs must be non-negative and sum to 1")
        if len(set(map(str, values))) != len(values):
            raise p.err("categorical values repeat")
        return Dist("categorical", values=tuple(values), probs=tuple(probs), text=text)
    names = _PARAMS[kind]
    got: dict[str, float] = {}
    positional = 0
    while True:
        save = p.i
        key: str | None = None
        p.ws()
        m = _IDENT_RE.match(p.s, p.i)
        if m:
            p.i = m.end()
            if p.peek() == "=":
                p.i += 1
                key = m.group(0)
            else:
                p.i = save
        if key is None:
            if got and positional < len(got):
                raise p.err("positional argument after keyword")
            if positional >= len(names):
                raise p.err("too many arguments")
            key = names[positional]
            positional += 1
        if key not in names or key in got:
            raise p.err(f"unexpected argument {key!r}")
        got[key] = p.number()
        ch = p.peek()
        if ch == ",":
            p.i += 1
            continue
        p.expect(")")
        break
    p.at_end()
    if set(got) != set(names):
        raise p.err(f"{kind} needs exactly {', '.join(names)}")
    d = Dist(kind, params=tuple((n, got[n]) for n in names), text=text)
    _check_params(d, p)
    return d


def _check_params(d: Dist, p: _Parser) -> None:
    if d.kind in ("normal", "lognormal", "truncnormal") and d.p("sigma") <= 0:
        raise p.err("sigma must be positive")
    if d.kind in ("uniform", "truncnormal", "beta") and not d.p("low") < d.p("high"):
        raise p.err("low must be below high")
    if d.kind == "beta" and (d.p("a") <= 0 or d.p("b") <= 0):
        raise p.err("beta a and b must be positive")


def _ppf(d: Dist, u: np.ndarray | float) -> Any:
    """Inverse CDF of a continuous distribution."""
    if d.kind == "normal":
        return d.p("mu") + d.p("sigma") * stats.norm.ppf(u)
    if d.kind == "lognormal":
        return np.exp(d.p("mu") + d.p("sigma") * stats.norm.ppf(u))
    if d.kind == "uniform":
        return d.p("low") + u * (d.p("high") - d.p("low"))
    if d.kind == "truncnormal":
        mu, sg = d.p("mu"), d.p("sigma")
        a = stats.norm.cdf((d.p("low") - mu) / sg)
        b = stats.norm.cdf((d.p("high") - mu) / sg)
        return mu + sg * stats.norm.ppf(a + u * (b - a))
    if d.kind == "beta":
        lo, hi = d.p("low"), d.p("high")
        return lo + (hi - lo) * stats.beta.ppf(u, d.p("a"), d.p("b"))
    raise HarnessValidationError(f"{d.kind} has no inverse CDF")


def sample_dist(
    d: Dist,
    rng: np.random.Generator,
    drawn: Mapping[str, Value],
    fns: Callable[[str], Callable[..., Value]],
) -> Value:
    """One draw from a parsed distribution."""
    if d.kind == "fixed":
        return d.values[0]
    if d.kind == "categorical":
        idx = int(rng.choice(len(d.values), p=np.asarray(d.probs)))
        return d.values[idx]
    if d.kind == "fn":
        assert d.fn is not None
        return fns(d.fn)(rng, drawn)
    if d.kind == "normal":
        return float(rng.normal(d.p("mu"), d.p("sigma")))
    if d.kind == "lognormal":
        return float(rng.lognormal(d.p("mu"), d.p("sigma")))
    if d.kind == "uniform":
        return float(rng.uniform(d.p("low"), d.p("high")))
    if d.kind == "beta":
        return float(
            d.p("low") + (d.p("high") - d.p("low")) * rng.beta(d.p("a"), d.p("b"))
        )
    if d.kind == "truncnormal":
        return float(_ppf(d, float(rng.uniform())))
    raise HarnessValidationError(f"cannot sample {d.kind}")


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

_OP = r"(<=|>=|<|>)"
_NUM = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
_CON_RE = re.compile(
    rf"^\s*(?:{_NUM}\s*{_OP}\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*{_OP}\s*{_NUM}\s*$"
)
_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


@dataclass(frozen=True)
class Constraint:
    text: str
    node: str
    left: tuple[float, str] | None  # (number, op): number op node
    right: tuple[str, float]  # (op, number): node op number

    def holds(self, value: float) -> bool:
        if self.left is not None and not _OPS[self.left[1]](self.left[0], value):
            return False
        return _OPS[self.right[0]](value, self.right[1])


def parse_constraint(text: str) -> Constraint:
    if not isinstance(text, str):
        raise HarnessValidationError(f"constraint must be a string, got {text!r}")
    m = _CON_RE.match(text)
    if not m:
        raise HarnessValidationError(
            f"constraint {text!r} is outside the grammar (§6.1)"
        )
    left = (float(m.group(1)), m.group(2)) if m.group(1) is not None else None
    return Constraint(text, m.group(3), left, (m.group(4), float(m.group(5))))


# ---------------------------------------------------------------------------
# The experiment's world
# ---------------------------------------------------------------------------


@dataclass
class World:
    """The nodes an experiment uses, with their distributions and order."""

    schema: ScenarioSchema
    nodes: dict[str, NodeDef]
    dists: dict[str, Dist]
    order: list[str]
    corr_nodes: tuple[str, ...]
    corr_chol: np.ndarray | None
    constraints: list[Constraint]
    fns: Callable[[str], Callable[..., Any]]
    rendered_hint: set[str] = field(default_factory=set)

    def is_numeric(self, name: str) -> bool:
        return self.nodes[name].type == "numeric"


def build_world(
    schema: ScenarioSchema,
    scenario_spec: Mapping[str, Any],
    fns: Callable[[str], Callable[..., Any]],
) -> World:
    """Validate the scenario spec against the family schema and fix the draw order."""
    raw = scenario_spec.get("distributions")
    if not isinstance(raw, Mapping) or not raw:
        raise HarnessValidationError(
            "scenario_spec.distributions must be a non-empty mapping"
        )
    dists: dict[str, Dist] = {}
    for name, text in raw.items():
        if name not in schema.nodes:
            raise HarnessValidationError(
                f"scenario_spec.distributions: unknown node {name!r}"
            )
        if schema.nodes[name].kind == "derived":
            raise HarnessValidationError(
                f"derived node {name!r} cannot have a distribution"
            )
        dists[name] = parse_dist(text)
    included: dict[str, NodeDef] = {n: schema.nodes[n] for n in dists}
    changed = True
    while changed:
        changed = False
        for n in schema.nodes.values():
            if (
                n.kind == "derived"
                and n.name not in included
                and all(p in included for p in n.parents)
            ):
                included[n.name] = n
                changed = True
    for name, node in included.items():
        if node.kind != "derived":
            for dep in node.depends_on:
                if dep not in included:
                    raise HarnessValidationError(
                        f"node {name!r} depends on {dep!r}, which has no distribution"
                    )
        if (
            node.type == "categorical"
            and name in dists
            and dists[name].kind in CONTINUOUS
        ):
            raise HarnessValidationError(
                f"categorical node {name!r} has a continuous distribution"
            )
    order = _topo_order(included)
    corr = scenario_spec.get("correlation")
    corr_nodes: tuple[str, ...] = ()
    chol: np.ndarray | None = None
    if corr:
        corr_nodes = tuple(corr.get("nodes") or ())
        matrix = np.asarray(corr.get("matrix"), dtype=float)
        k = len(corr_nodes)
        if matrix.shape != (k, k) or k < 2:
            raise HarnessValidationError("correlation.matrix must be k×k with k ≥ 2")
        if not np.allclose(matrix, matrix.T):
            raise HarnessValidationError("correlation.matrix is not symmetric")
        if not np.allclose(np.diag(matrix), 1.0):
            raise HarnessValidationError("correlation.matrix diagonal must be 1")
        try:
            chol = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise HarnessValidationError(
                "correlation.matrix is not positive definite"
            ) from exc
        for n in corr_nodes:
            node = included.get(n)
            if (
                node is None
                or node.kind != "primitive"
                or dists[n].kind not in CONTINUOUS
            ):
                raise HarnessValidationError(
                    f"correlation node {n!r} must be a continuous non-fn primitive"
                )
    constraints = [parse_constraint(c) for c in scenario_spec.get("constraints") or []]
    for c in constraints:
        if c.node not in included or included[c.node].type != "numeric":
            raise HarnessValidationError(
                f"constraint {c.text!r} names an unknown or non-numeric node"
            )
    return World(schema, included, dists, order, corr_nodes, chol, constraints, fns)


def _topo_order(nodes: Mapping[str, NodeDef]) -> list[str]:
    names = list(nodes)
    deps = {
        n: [p for p in (*nodes[n].parents, *nodes[n].depends_on) if p in nodes]
        for n in names
    }
    out: list[str] = []
    state: dict[str, int] = {}

    def visit(n: str, stack: tuple[str, ...]) -> None:
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise HarnessValidationError(
                f"scenario graph has a cycle through {' → '.join((*stack, n))}"
            )
        state[n] = 1
        for d in deps[n]:
            visit(d, (*stack, n))
        state[n] = 2
        out.append(n)

    for n in names:
        visit(n, ())
    return out


def state_precision(node: NodeDef) -> int:
    """Decimal places a primitive's state value keeps, given its display format.

    State values are in base units (e.g. USD); ``pct`` shows value × 100,
    ``money_m`` value / 10⁶ and ``money_bn`` value / 10⁹.
    """
    if node.display == "pct":
        return node.precision + 2
    if node.display == "money_m":
        return node.precision - 6
    if node.display == "money_bn":
        return node.precision - 9
    return node.precision


def quantise(node: NodeDef, value: Value) -> Value:
    """Primitive numeric values are held at their display precision."""
    if node.type != "numeric" or node.kind != "primitive":
        return value
    v = float(value)  # type: ignore[arg-type]
    if not math.isfinite(v):
        return v
    if node.display == "int":
        return int(round(v))
    out = round(v, state_precision(node))
    return float(out)


def recompute_derived(
    world: World, state: dict[str, Value], only: Sequence[str] | None = None
) -> None:
    """Recompute derived nodes in topological order (all, or those listed)."""
    for name in world.order:
        node = world.nodes[name]
        if node.kind != "derived":
            continue
        if only is not None and name not in only:
            continue
        fn = world.fns(node.fn or "")
        state[name] = fn(**{p: state[p] for p in node.parents})


def attempt_rng(seed_stream: str, generation_seed: int, k: int) -> np.random.Generator:
    return np.random.default_rng(uint64(seed_stream, generation_seed, k))


def draw_state(world: World, rng: np.random.Generator) -> dict[str, Value]:
    """One joint draw of every node, in topological order."""
    state: dict[str, Value] = {}
    corr_values: dict[str, float] = {}
    if world.corr_chol is not None:
        z = world.corr_chol @ rng.standard_normal(len(world.corr_nodes))
        u = stats.norm.cdf(z)
        for name, ui in zip(world.corr_nodes, u, strict=True):
            corr_values[name] = float(_ppf(world.dists[name], float(ui)))
    for name in world.order:
        node = world.nodes[name]
        if node.kind == "derived":
            state[name] = world.fns(node.fn or "")(
                **{p: state[p] for p in node.parents}
            )
            continue
        if name in corr_values:
            value: Value = corr_values[name]
        else:
            value = sample_dist(world.dists[name], rng, state, world.fns)
        state[name] = quantise(node, value)
    return state


def violated_constraints(world: World, state: Mapping[str, Value]) -> list[str]:
    out: list[str] = []
    for c in world.constraints:
        v = state.get(c.node)
        if v is None or not isinstance(v, int | float) or not math.isfinite(float(v)):
            out.append(c.text)
            continue
        if not c.holds(float(v)):
            out.append(c.text)
    return out
