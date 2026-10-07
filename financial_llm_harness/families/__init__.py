"""Task families (spec §8, §8.1, §34) and their file loaders.

Each family directory holds ``family.yaml`` (scenario schema, interventions,
contracts, features, oracles, ``faq_ids``, ``battery_slots``), ``family.py``
(derived-node functions, scorers, ``fake_response``), ``defaults.yaml``,
``mandates/``, ``contracts/``, ``templates/``, ``paraphrases/``, ``banks/`` and
``fixtures/``. Prompt-visible and oracle-bearing files carry a provenance block.
"""

from __future__ import annotations

import contextvars
import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.code_ref import package_dir
from financial_llm_harness.core.schema import (
    as_list,
    as_mapping,
    as_str,
    check_keys,
    fail,
)
from financial_llm_harness.errors import HarnessValidationError

FAMILY_IDS = (
    "fundamental_judgement",
    "fundamental_extraction",
    "corporate_action",
    "call_analysis",
    "broker_synthesis",
    "quant_method_choice",
    "quant_dgp",
    "ml_design",
    "code_task",
)

OPERATIONS = (
    "ADD",
    "REMOVE",
    "REVERSE",
    "NUMERIC_SHIFT",
    "CATEGORY_CHANGE",
    "FRAMING_CHANGE",
    "CONSTRAINT_ADD",
    "CONSTRAINT_REMOVE",
    "POSITION_CHANGE",
)


def families_root() -> Path:
    return package_dir() / "families"


def family_dir(family_id: str) -> Path:
    if family_id not in FAMILY_IDS:
        raise HarnessValidationError(f"unknown task family {family_id!r}")
    return families_root() / family_id


def rel_to_package(path: Path) -> str:
    """POSIX path relative to the package directory (an asset logical name)."""
    return path.relative_to(package_dir()).as_posix()


@dataclass(frozen=True)
class NodeDef:
    name: str
    kind: str  # primitive | latent | derived
    type: str  # numeric | categorical
    unit: str | None
    precision: int
    display: str
    fn: str | None
    parents: tuple[str, ...]
    depends_on: tuple[str, ...]
    relevant: bool


@dataclass(frozen=True)
class ScenarioSchema:
    nodes: dict[str, NodeDef]

    def children(self, name: str) -> list[str]:
        return [
            n.name
            for n in self.nodes.values()
            if name in n.parents or name in n.depends_on
        ]

    def descendants(self, name: str) -> set[str]:
        out: set[str] = set()
        stack = [name]
        while stack:
            cur = stack.pop()
            for child in self.children(cur):
                if child not in out:
                    out.add(child)
                    stack.append(child)
        return out


@dataclass(frozen=True)
class Closure:
    recompute: tuple[str, ...]
    held_fixed: tuple[str, ...]
    compensate: tuple[str, ...]


@dataclass(frozen=True)
class InterventionDef:
    intervention_id: str
    op: str
    node: str
    via: str | None
    form: str
    baseline: Any
    counterfactual: Any
    delta: float | None
    relative_delta: float | None
    closure: Closure
    counterbalance: bool
    concept_direction: int
    concept_factor: str | None
    concept_lexicon: tuple[str, ...]


@dataclass(frozen=True)
class FeatureDef:
    feature_id: str
    data: dict[str, Any]


@dataclass(frozen=True)
class TaskFamily:
    family_id: str
    scenario_schema: ScenarioSchema
    interventions: dict[str, InterventionDef]
    response_contracts: tuple[str, ...]
    features: dict[str, FeatureDef]
    oracles: dict[str, Any]
    faq_ids: tuple[str, ...]
    battery_slots: dict[str, str]
    raw: dict[str, Any] = field(repr=False)

    @property
    def directory(self) -> Path:
        return family_dir(self.family_id)

    @property
    def module(self) -> ModuleType:
        return family_module(self.family_id)


def _node(name: str, d: Any, where: str) -> NodeDef:
    m = as_mapping(d, where)
    check_keys(
        m,
        where,
        required=("kind", "type"),
        optional=(
            "unit",
            "precision",
            "display",
            "fn",
            "parents",
            "depends_on",
            "relevant",
        ),
    )
    kind = as_str(m["kind"], where)
    if kind not in ("primitive", "latent", "derived"):
        raise fail(where, f"kind {kind!r} invalid")
    typ = as_str(m["type"], where)
    if typ not in ("numeric", "categorical"):
        raise fail(where, f"type {typ!r} invalid")
    display = m.get("display", "text" if typ == "categorical" else "number")
    if display not in (
        "money",
        "money_m",
        "money_bn",
        "pct",
        "multiple",
        "int",
        "number",
        "text",
    ):
        raise fail(where, f"display {display!r} invalid")
    if kind == "derived" and not m.get("fn"):
        raise fail(where, "a derived node names a fn")
    return NodeDef(
        name=name,
        kind=kind,
        type=typ,
        unit=m.get("unit"),
        precision=int(m.get("precision", 0)),
        display=display,
        fn=m.get("fn"),
        parents=tuple(m.get("parents") or ()),
        depends_on=tuple(m.get("depends_on") or ()),
        relevant=bool(m.get("relevant", False)),
    )


def parse_intervention(iid: str, d: Any, where: str) -> InterventionDef:
    m = as_mapping(d, where)
    check_keys(
        m,
        where,
        required=(
            "op",
            "node",
            "via",
            "form",
            "baseline",
            "counterfactual",
            "closure",
            "counterbalance",
            "concept_direction",
            "concept_factor",
        ),
        optional=("intervention_id", "delta", "relative_delta", "concept_lexicon"),
    )
    if m["op"] not in OPERATIONS:
        raise fail(where, f"op {m['op']!r} invalid")
    if m["form"] not in ("set", "shift"):
        raise fail(where, f"form {m['form']!r} invalid")
    if m["counterbalance"] and m["form"] == "set":
        raise fail(
            where, "counterbalance: true is allowed only with form: shift (§6.5)"
        )
    c = as_mapping(m["closure"], where + ".closure")
    check_keys(
        c, where + ".closure", required=("recompute", "held_fixed", "compensate")
    )
    cd = m["concept_direction"]
    if cd not in (-1, 0, 1) or isinstance(cd, bool):
        raise fail(where, "concept_direction must be -1, 0 or 1")
    return InterventionDef(
        intervention_id=m.get("intervention_id", iid),
        op=m["op"],
        node=m["node"],
        via=m["via"],
        form=m["form"],
        baseline=m["baseline"],
        counterfactual=m["counterfactual"],
        delta=m.get("delta"),
        relative_delta=m.get("relative_delta"),
        closure=Closure(
            tuple(as_list(c["recompute"], where)),
            tuple(as_list(c["held_fixed"], where)),
            tuple(as_list(c["compensate"], where)),
        ),
        counterbalance=bool(m["counterbalance"]),
        concept_direction=int(cd),
        concept_factor=m["concept_factor"],
        concept_lexicon=tuple(m.get("concept_lexicon") or ()),
    )


@lru_cache(maxsize=32)
def _load_family_cached(family_id: str, mtime_ns: int) -> TaskFamily:
    path = family_dir(family_id) / "family.yaml"
    data = yaml_io.load(path)
    where = str(path)
    check_keys(
        data,
        where,
        required=(
            "provenance",
            "family_id",
            "scenario_schema",
            "interventions",
            "response_contracts",
            "features",
            "oracles",
            "faq_ids",
            "battery_slots",
        ),
        optional=("contexts", "scorers", "description"),
    )
    if data["family_id"] != family_id:
        raise fail(where, f"family_id {data['family_id']!r} != directory {family_id!r}")
    nodes_raw = as_mapping(
        as_mapping(data["scenario_schema"], where).get("nodes"), where + ".nodes"
    )
    nodes = {
        name: _node(name, d, f"{where}.nodes.{name}") for name, d in nodes_raw.items()
    }
    for n in nodes.values():
        for p in (*n.parents, *n.depends_on):
            if p not in nodes:
                raise fail(where, f"node {n.name!r} references unknown node {p!r}")
    interventions = {
        iid: parse_intervention(iid, d, f"{where}.interventions.{iid}")
        for iid, d in (data["interventions"] or {}).items()
    }
    features = {
        fid: FeatureDef(fid, dict(as_mapping(d, f"{where}.features.{fid}")))
        for fid, d in (data["features"] or {}).items()
    }
    return TaskFamily(
        family_id=family_id,
        scenario_schema=ScenarioSchema(nodes),
        interventions=interventions,
        response_contracts=tuple(data["response_contracts"] or ()),
        features=features,
        oracles=dict(data["oracles"] or {}),
        faq_ids=tuple(data["faq_ids"] or ()),
        battery_slots=dict(data["battery_slots"] or {}),
        raw=dict(data),
    )


def load_family(family_id: str) -> TaskFamily:
    path = family_dir(family_id) / "family.yaml"
    if not path.exists():
        raise HarnessValidationError(f"family {family_id!r} has no family.yaml")
    return _load_family_cached(family_id, path.stat().st_mtime_ns)


def family_module(family_id: str) -> ModuleType:
    family_dir(family_id)
    return importlib.import_module(f"financial_llm_harness.families.{family_id}.family")


def load_yaml_file(family_id: str, rel: str) -> dict[str, Any]:
    path = family_dir(family_id) / rel
    if not path.exists():
        raise HarnessValidationError(f"missing family file {rel_to_package(path)}")
    data = yaml_io.load(path)
    return dict(as_mapping(data, str(path)))


def load_defaults(family_id: str) -> dict[str, Any]:
    path = family_dir(family_id) / "defaults.yaml"
    if not path.exists():
        return {"provenance": None, "entries": {}}
    return dict(as_mapping(yaml_io.load(path), str(path)))


def load_mandate(family_id: str, mandate_id: str) -> dict[str, Any]:
    data = load_yaml_file(family_id, f"mandates/{mandate_id}.yaml")
    check_keys(
        data,
        f"mandates/{mandate_id}.yaml",
        required=("provenance", "id", "version", "text", "country_neutral"),
    )
    if data["id"] != mandate_id:
        raise HarnessValidationError(f"mandate file {mandate_id} has id {data['id']!r}")
    return data


def load_clause(family_id: str, clause_id: str) -> dict[str, Any]:
    data = load_yaml_file(family_id, f"mandates/clauses/{clause_id}.yaml")
    check_keys(
        data,
        f"mandates/clauses/{clause_id}.yaml",
        required=("provenance", "id", "text"),
    )
    return data


def load_contract(family_id: str, contract_id: str) -> dict[str, Any]:
    return load_yaml_file(family_id, f"contracts/{contract_id}.yaml")


# ---------------------------------------------------------------------------
# Banks, with consumption tracking for assets.json (§0.10)
# ---------------------------------------------------------------------------

_BANK_LOG: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "_BANK_LOG", default=None
)


class BankTracker:
    """Records every bank file read while generation runs."""

    def __init__(self) -> None:
        self.used: set[str] = set()
        self._token: contextvars.Token[set[str] | None] | None = None

    def __enter__(self) -> BankTracker:
        self._token = _BANK_LOG.set(self.used)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _BANK_LOG.reset(self._token)


@lru_cache(maxsize=256)
def _bank_cached(path_str: str, mtime_ns: int) -> dict[str, Any]:
    return dict(as_mapping(yaml_io.load(Path(path_str)), path_str))


def bank_file(path: Path) -> dict[str, Any]:
    """Load a bank file (family or shared taxonomy bank) and record its use."""
    if not path.exists():
        raise HarnessValidationError(f"missing bank {rel_to_package(path)}")
    log = _BANK_LOG.get()
    if log is not None:
        log.add(rel_to_package(path))
    return _bank_cached(str(path), path.stat().st_mtime_ns)


def family_bank(family_id: str, name: str) -> dict[str, Any]:
    return bank_file(family_dir(family_id) / "banks" / f"{name}.yaml")


def bank_entries(family_id: str, name: str) -> list[Any]:
    data = family_bank(family_id, name)
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise HarnessValidationError(f"bank {family_id}/{name} has no entries")
    return entries


def provenance_of(data: Mapping[str, Any]) -> dict[str, Any] | None:
    prov = data.get("provenance")
    return dict(prov) if isinstance(prov, Mapping) else None
