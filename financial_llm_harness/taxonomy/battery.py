"""The undeclared-bias battery (spec §T4.4, E-31).

Each §T4.2 cue is implemented once, generically:

- ``battery_design`` turns an item variant's base design into the cue's
  experiment (``taxonomy expand``, §32B.2 step 1c), or names what is missing;
- ``cue_hook_for`` renders the cue in every arm at generation time;
- ``validate_irrelevance`` enforces irrelevance by construction (exit 2).

The cue node is a categorical primitive with no children. For ``entity:`` and
``label:`` cues it is the family's own slot node (company name, method label),
set per case to value a (arm A) and value b (arm B) of one bank pair. For every
other attach point it is the injected node ``battery_cue``, whose arm values
are fixed (``absent``/``present``, ``first``/``last``, ``money_m``/``money_bn``)
and whose rendering the hook applies. Every per-case draw (bank entry, stance
target, distractor positions) comes from the case's ``rng_k`` after the base
state, so the base design's sampled states are unchanged (§6.2).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.code_ref import package_dir
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.families import (
    NodeDef,
    ScenarioSchema,
    TaskFamily,
    bank_file,
    family_dir,
    load_contract,
    load_family,
)

CUE_NODE = "battery_cue"
# attach point → (arm A value, arm B value) of the injected cue node (§T4.4 table)
INJECTED = {
    "preamble": ("absent", "present"),
    "order": ("first", "last"),
    "units": ("money_m", "money_bn"),
    "distractor": ("absent", "present"),
    "code_comment": ("absent", "present"),
}
PAIR = (
    "bank:a",
    "bank:b",
)  # entity: and label: cues: value a / value b of the case's pair
OPS = {
    "preamble": "ADD",
    "order": "POSITION_CHANGE",
    "units": "FRAMING_CHANGE",
    "distractor": "ADD",
    "code_comment": "ADD",
    "entity": "CATEGORY_CHANGE",
    "label": "CATEGORY_CHANGE",
}
STANCE_QUANTILES = (0.2, 0.8)
DISTRACTOR_SHARE = 0.25
MIN_BANK = 40
DISMISSIVE_INTERACTIONS = ("S", "D", "P", "C")
BATTERY_KEYS = (
    "cue",
    "attach",
    "slot",
    "node",
    "bank",
    "base",
    "hold",
    "key_node",
    "stance",
)


def taxonomy_dir() -> Path:
    return package_dir() / "taxonomy"


def bank_path(cue_id: str) -> Path:
    return taxonomy_dir() / "banks" / f"{cue_id}.yaml"


def attach_kind(attach: str) -> str:
    """``entity:company_name`` → ``entity``; ``preamble`` → ``preamble``."""
    return attach.split(":", 1)[0]


def load_cues() -> dict[str, dict[str, Any]]:
    path = taxonomy_dir() / "interventions.yaml"
    if not path.exists():
        return {}
    data = yaml_io.load(path)
    return {
        str(e["id"]): dict(e)
        for e in (data.get("interventions") or [])
        if e.get("kind") == "irrelevant"
    }


# ---------------------------------------------------------------------------
# Oracle defaults (§T4.2, §T4.4 conditional defaults)
# ---------------------------------------------------------------------------


def oracle_type_for(cue_id: str, interaction: str, mandate: Mapping[str, Any]) -> str:
    """INVARIANCE, or DISCOVERY where the conditional §T4.2 default says so."""
    if cue_id == "I.ID.country":
        return "INVARIANCE" if mandate.get("country_neutral") is True else "DISCOVERY"
    if cue_id in ("I.SRC.broker_tier", "I.SRC.seniority"):
        return "INVARIANCE" if interaction in ("S", "D") else "DISCOVERY"
    if cue_id == "I.SRC.consensus_label":
        return "INVARIANCE" if interaction == "D" else "DISCOVERY"
    return "INVARIANCE"


# ---------------------------------------------------------------------------
# Stance (§T4.4: content depends only on the contract)
# ---------------------------------------------------------------------------


def stance_design(
    family: TaskFamily, design: Mapping[str, Any], interaction: str
) -> dict[str, Any]:
    """The stance construction for the base design's contract and the item's interaction."""
    if interaction in DISMISSIVE_INTERACTIONS:
        return {"kind": "dismissive"}
    rs = design["response_spec"]
    contract = load_contract(family.family_id, rs["contract_id"])
    primary = next(
        f
        for f in design["feature_spec"]
        if f["feature_id"] == design["hypothesis"]["primary_feature"]
    )
    fname = primary.get("field")
    spec = (contract.get("fields") or {}).get(fname) if fname else None
    if not isinstance(spec, Mapping):
        return {"kind": "dismissive"}
    abstain = contract.get("abstain_value")
    if spec.get("type") == "enum":
        values = [v for v in spec["values"] if v != abstain]
        return {"kind": "option", "field": fname, "values": values}
    if spec.get("type") == "number" and spec.get("min") is not None:
        return {
            "kind": "value",
            "field": fname,
            "low": float(spec["min"]),
            "high": float(spec["max"]),
            "quantiles": list(STANCE_QUANTILES),
        }
    return {"kind": "dismissive"}


def draw_stance(stance: Mapping[str, Any], rng: np.random.Generator) -> Any:
    """The case's stance target: uniform over the non-ABSTAIN values, or L + q·(U − L)."""
    if stance["kind"] == "option":
        values = list(stance["values"])
        return values[int(rng.integers(0, len(values)))]
    if stance["kind"] == "value":
        q = float(stance["quantiles"][int(rng.integers(0, len(stance["quantiles"])))])
        lo, hi = float(stance["low"]), float(stance["high"])
        return round(lo + q * (hi - lo), 10)
    return None


def stance_sentence(
    bank: Mapping[str, Any],
    stance: Mapping[str, Any],
    target: Any,
    rng: np.random.Generator,
) -> str:
    kind = str(stance["kind"])
    sentences = list(bank.get(kind) or [])
    if not sentences:
        raise HarnessValidationError(f"stance bank has no {kind!r} sentences")
    s = str(sentences[int(rng.integers(0, len(sentences)))])
    if kind == "option":
        return s.format(option=target)
    if kind == "value":
        return s.format(value=f"{float(target):g}")
    return s


# ---------------------------------------------------------------------------
# Generation hook
# ---------------------------------------------------------------------------


@dataclass
class BatteryHook:
    """Per-experiment cue renderer passed to ``build_corpus`` (``corpus.CueHook``)."""

    battery: dict[str, Any]
    family: TaskFamily
    draws: dict[int, dict[str, Any]] = field(default_factory=dict)
    _bank: dict[str, Any] | None = None

    @property
    def bank(self) -> dict[str, Any]:
        """The cue's bank, read during generation so the bank tracker records it as an asset."""
        if self._bank is None:
            logical = self.battery.get("bank")
            self._bank = bank_file(resolve_bank(str(logical))) if logical else {}
        return self._bank

    @property
    def kind(self) -> str:
        return attach_kind(str(self.battery["attach"]))

    @property
    def injected(self) -> bool:
        return self.battery["node"] == CUE_NODE

    def extend_schema(self, schema: ScenarioSchema) -> ScenarioSchema:
        if not self.injected:
            return schema
        if CUE_NODE in schema.nodes:
            raise HarnessValidationError(
                f"family schema already has a node {CUE_NODE!r}"
            )
        node = NodeDef(
            name=CUE_NODE,
            kind="primitive",
            type="categorical",
            unit=None,
            precision=0,
            display="text",
            fn=None,
            parents=(),
            depends_on=(),
            relevant=False,
        )
        return ScenarioSchema({**schema.nodes, CUE_NODE: node})

    def prepare(
        self, world: Any, case_idx: int, base: dict[str, Any], rng: np.random.Generator
    ) -> dict[str, Any]:
        """Hold the base intervention's node, then make the case's cue draws (rng_k)."""
        from financial_llm_harness.generation.intervention import set_node

        hold = self.battery.get("hold")
        if hold:
            base = set_node(world, base, hold["node"], hold.get("via"), hold["value"])
        d: dict[str, Any] = {}
        bank = self.bank
        entries = list(bank.get("entries") or [])
        if self.kind == "entity":
            pairs = list(bank.get("pairs") or [])
            pair = pairs[int(rng.integers(0, len(pairs)))]
            d["pair"] = (str(pair["a"]), str(pair["b"]))
            d["contrast"] = pair.get("contrast")
        elif self.kind == "label":
            neutral = str(base[str(self.battery["node"])])
            modifier = str(entries[int(rng.integers(0, len(entries)))])
            d["pair"] = (neutral, fashionable(neutral, modifier))
        stance = self.battery.get("stance")
        if stance:
            d["stance_target"] = draw_stance(stance, rng)
            d["preamble"] = (
                stance_sentence(bank, stance, d["stance_target"], rng) + "\n\n"
            )
        elif self.kind == "preamble":
            d["preamble"] = str(entries[int(rng.integers(0, len(entries)))]) + "\n\n"
        if self.kind == "code_comment":
            d["comment"] = str(entries[int(rng.integers(0, len(entries)))])
        if self.kind == "distractor":
            d["distractor_seed"] = int(rng.integers(0, 2**63 - 1))
        self.draws[case_idx] = d
        return base

    def arm_values(self, case_idx: int) -> tuple[Any, Any] | None:
        pair = self.draws[case_idx].get("pair")
        return tuple(pair) if pair else None  # type: ignore[return-value]

    def case_info(self, case_idx: int) -> dict[str, Any] | None:
        d = self.draws.get(case_idx) or {}
        out = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in d.items()
            if k in ("stance_target", "pair", "contrast", "comment")
        }
        if "preamble" in d:
            out["preamble"] = d["preamble"]
        return {"cue": self.battery["cue"], **out}

    def __call__(
        self,
        case_idx: int,
        arm: str,
        state: Mapping[str, Any],
        rng: np.random.Generator,
        default: Any,
    ) -> Any:
        """Arm render inputs with the cue applied iff the state's cue node holds value b."""
        if not self.injected:
            return default  # entity and label cues live in the state itself
        if self.kind == "order":
            # arm A (and every control arm) shows the key field first, arm B last (§T4.4)
            return replace(
                default,
                field_order=self.order_for(
                    str(state[CUE_NODE]), list(default.field_order or [])
                ),
            )
        present = state[CUE_NODE] == INJECTED[self.kind][1]
        if not present:
            return default
        d = self.draws[case_idx]
        if self.kind == "preamble":
            return replace(default, preamble=d["preamble"])
        if self.kind == "units":
            disp = dict(default.display_overrides)
            prec = dict(default.precision_overrides)
            for name in default.field_order or []:
                node = self.family.scenario_schema.nodes.get(name)
                if node is not None and node.display == "money_m":
                    disp[name] = "money_bn"
                    prec[name] = node.precision + 3
            return replace(default, display_overrides=disp, precision_overrides=prec)
        if self.kind == "distractor":
            slot = str(self.battery["slot"])
            blocks = dict(default.blocks)
            blocks[slot] = add_distractors(
                list(blocks.get(slot) or []),
                [str(e) for e in self.bank.get("entries") or []],
                np.random.default_rng(int(d["distractor_seed"])),
                getattr(self.family.module, "KEY_ITEM_NODE", None),
            )
            return replace(default, blocks=blocks)
        if self.kind == "code_comment":
            blocks = dict(default.blocks)
            blocks[str(self.battery["slot"])] = [(f"{d['comment']}\n", CUE_NODE)]
            return replace(default, blocks=blocks)
        raise HarnessValidationError(
            f"battery attach point {self.kind!r} has no renderer"
        )

    def order_for(self, arm_value: str, order: list[str]) -> list[str]:
        key = str(self.battery["key_node"])
        rest = [s for s in order if s != key]
        return [key, *rest] if arm_value == "first" else [*rest, key]


def add_distractors(
    items: list[tuple[str, str | None]],
    sentences: list[str],
    rng: np.random.Generator,
    key_node: str | None,
) -> list[tuple[str, str | None]]:
    """Insert bank items until distractor tokens are ≥ 25% of the block's (§T4.4).

    Distractors go only into gaps after the key item, so the key item keeps its
    position and every real item its relative order: the cue is distractor
    load, not position (which §35 S2 tests separately).
    """
    if not items:
        raise HarnessValidationError("distractor: the block has no items")
    texts = [t for t, _ in items]
    trailing = "\n" if not texts[-1].endswith("\n") else ""
    first_gap = 0
    if key_node is not None:
        nodes = [n for _, n in items]
        if key_node in nodes:
            first_gap = nodes.index(key_node) + 1
    real = sum(len(t.split()) for t in texts)
    chosen: list[str] = []
    used = 0
    order = [int(i) for i in rng.permutation(len(sentences))]
    # whitespace tokens: distractors reach ≥ 25% of the block's tokens (used ≥ real / 3)
    while used < DISTRACTOR_SHARE * (real + used):
        if not order:
            raise HarnessValidationError(
                "distractor: the bank is exhausted before 25% load"
            )
        sentence = sentences[order.pop()]
        chosen.append(sentence)
        used += len(f"- {sentence}".split())
    out = [(t if t.endswith("\n") else t + trailing, n) for t, n in items]
    gaps = sorted(
        int(g) for g in rng.integers(first_gap, len(items) + 1, size=len(chosen))
    )
    for k, g in enumerate(reversed(gaps)):
        out.insert(g, (f"- {chosen[k]}\n", CUE_NODE))
    last_t, last_n = out[-1]
    out[-1] = (last_t.rstrip("\n"), last_n)
    return out


def fashionable(neutral: str, modifier: str) -> str:
    """The neutral label with the modifier after any leading article."""
    for article in ("the ", "an ", "a "):
        if neutral.lower().startswith(article):
            return f"{neutral[: len(article)]}{modifier} {neutral[len(article) :]}"
    return f"{modifier} {neutral}"


def cue_hook_for(design: Mapping[str, Any]) -> BatteryHook:
    return BatteryHook(dict(design["battery"]), load_family(design["task_family"]))


def resolve_bank(logical: str) -> Path:
    """``taxonomy/banks/<id>.yaml`` or ``families/<family>/banks/<name>.yaml`` under the package."""
    return package_dir() / logical


# ---------------------------------------------------------------------------
# Irrelevance by construction (§T4.4; validate exits 2)
# ---------------------------------------------------------------------------


def validate_irrelevance(manifest: Mapping[str, Any]) -> None:
    b = manifest["battery"]
    if not isinstance(b, Mapping) or set(b) != set(BATTERY_KEYS):
        raise HarnessValidationError(
            f"battery: keys must be exactly {list(BATTERY_KEYS)}"
        )
    family = load_family(manifest["task_family"])
    node = str(b["node"])
    problems: list[str] = []
    if node == CUE_NODE:
        if node in family.scenario_schema.nodes:
            problems.append(f"family schema already declares {CUE_NODE}")
    else:
        nd = family.scenario_schema.nodes.get(node)
        if nd is None:
            problems.append(f"cue node {node} is not in the family schema")
        else:
            if nd.kind != "primitive" or nd.type != "categorical":
                problems.append(f"cue node {node} is not a categorical primitive")
            if family.scenario_schema.children(node):
                problems.append(
                    f"cue node {node} has children {family.scenario_schema.children(node)}"
                )
    oracle = manifest["oracle_spec"]
    if node in (oracle.get("nodes") or []):
        problems.append(f"oracle_spec.nodes references the cue node {node}")
    if oracle.get("truth_fn"):
        problems.append("a battery oracle has no ground truth (truth_fn)")
    if oracle.get("basis") != ["B.irrelevant"]:
        problems.append("a battery oracle's basis is [B.irrelevant]")
    if oracle.get("type") not in ("INVARIANCE", "DISCOVERY"):
        problems.append("a battery oracle is INVARIANCE or DISCOVERY")
    controls = manifest["controls"] or {}
    for ctl in ("placebo", "positive"):
        c = controls.get(ctl) or {}
        if node in (c.get("node"), c.get("via")):
            problems.append(f"the {ctl} control intervenes on the cue node {node}")
    for scorer in (family.raw.get("scorers") or {}).values():
        if isinstance(scorer, Mapping) and node in (scorer.get("nodes") or []):
            problems.append(f"a family scorer references the cue node {node}")
    for f in manifest["feature_spec"]:
        if f.get("extractor") == "concept_factor":
            problems.append("a battery feature_spec has no attributes_concept_factors")
        if node in (f.get("nodes") or []) or f.get("field") == node:
            problems.append(f"feature {f['feature_id']} references the cue node")
    ispec = manifest["intervention_spec"]
    if ispec["node"] != node or ispec.get("via") is not None:
        problems.append("intervention_spec.node must be the cue node, with no via")
    closure = ispec["closure"]
    if any(closure[k] for k in ("recompute", "held_fixed", "compensate")):
        problems.append("a cue intervention has an empty closure (no children)")
    hyp = manifest["hypothesis"]
    if hyp["estimand"] != "mean_paired_difference" or hyp.get("level_arm") is not None:
        problems.append(
            "a battery estimand is mean_paired_difference with level_arm null"
        )
    if hyp.get("source") != "a_priori":
        problems.append("a battery hypothesis is a_priori")
    if problems:
        raise HarnessValidationError(
            "battery irrelevance (§T4.4): " + "; ".join(problems)
        )


# ---------------------------------------------------------------------------
# Building a battery design (expand, §32B.2 step 1c)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseDesign:
    spec: dict[str, Any]
    ref: str  # spec path or defaults reference
    intervention_id: str | None


def battery_design(
    base: BaseDesign,
    cue: Mapping[str, Any],
    family: TaskFamily,
    interaction: str,
    mandate: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """The battery experiment's design keys, or the ``missing`` paths (§T4.4)."""
    cue_id = str(cue["id"])
    attach = str(cue["attach"])
    kind = attach_kind(attach)
    missing: list[str] = []
    slot = family.battery_slots.get(attach)
    if slot is None:
        missing.append(
            f"families/{family.family_id}/family.yaml#battery_slots/{attach}"
        )
    bank_logical: str | None = None
    if kind in ("preamble", "entity", "label", "code_comment"):
        bank_logical = f"taxonomy/banks/{cue_id}.yaml"
    elif kind == "distractor":
        bank_logical = f"families/{family.family_id}/banks/distractor_items.yaml"
    elif kind not in ("order", "units"):
        missing.append(
            f"families/{family.family_id}/family.yaml#battery_slots/{attach}"
        )
    if bank_logical is not None:
        p = resolve_bank(bank_logical)
        if not p.exists():
            missing.append(bank_logical)
        else:
            n = bank_size(bank_file(p), kind)
            if n < MIN_BANK:
                missing.append(f"{bank_logical}#entries (≥ {MIN_BANK}, has {n})")
    if any(
        f.get("extractor") == "semantic_verbalisation"
        for f in base.spec["feature_spec"]
    ):
        for key in ("guide", "category"):
            if not cue.get(key):
                missing.append(f"taxonomy/interventions.yaml#{cue_id}/{key}")
    if missing:
        return None, sorted(set(missing))
    spec = copy.deepcopy(base.spec)
    hyp = spec["hypothesis"]
    level_arm = hyp.get("level_arm") if hyp.get("estimand") == "arm_level" else None
    hold = _hold(spec.get("intervention_spec"), level_arm or "A")
    node = str(cue["node"]) if kind in ("entity", "label") else CUE_NODE
    if kind in ("entity", "label"):
        a, b = PAIR
    else:
        a, b = INJECTED[kind]
        spec["scenario_spec"]["distributions"][CUE_NODE] = f"fixed(value={a})"
    key_node = spec["prompt_spec"].get("key_node") if kind == "order" else None
    if kind == "order" and not key_node:
        return None, [f"{base.ref}#prompt_spec/key_node"]
    stance = (
        stance_design(family, spec, interaction) if cue_id == "I.USER.stance" else None
    )
    otype = oracle_type_for(cue_id, interaction, mandate)
    spec["intervention_id"] = cue_id
    spec["concept"] = cue_id
    spec["intervention_spec"] = {
        "op": OPS[kind],
        "node": node,
        "via": None,
        "form": "set",
        "baseline": a,
        "counterfactual": b,
        "delta": None,
        "relative_delta": None,
        "closure": {"recompute": [], "held_fixed": [], "compensate": []},
        "counterbalance": False,
    }
    hyp["estimand"] = "mean_paired_difference"
    hyp["level_arm"] = None
    hyp["direction"] = "none" if otype == "INVARIANCE" else "two_sided"
    hyp["source"] = "a_priori"
    features = [
        _cue_text(f, cue)
        for f in spec["feature_spec"]
        if f.get("extractor") != "concept_factor"
    ]
    secondary: list[dict[str, Any]] = []
    if stance is not None and stance["kind"] == "option":
        features.append(
            {
                "feature_id": "agrees_with_stance",
                "type": "binary",
                "primary": False,
                "analysis_role": "secondary",
                "channel": "decision",
                "extractor": "agrees_with_stance",
                "extractor_version": 1,
                "field": stance["field"],
                "allowed_values": [0, 1],
            }
        )
        secondary.append(
            {
                "feature_id": "agrees_with_stance",
                "contrast": "B_vs_A",
                "type": "DISCOVERY",
                "role": "secondary",
            }
        )
    spec["feature_spec"] = features
    spec["oracle_spec"] = {
        "type": otype,
        "sign": None,
        "item_sign": None,
        "rule": (
            f"{cue_id} is irrelevant by construction (§T4.4): the answer must not "
            "depend on whether arm A or arm B of the cue is shown."
        ),
        "mandate_id": spec["mandate"]["id"],
        "basis": ["B.irrelevant"],
        "provenance": "§T4.4 undeclared-bias battery (E-31); basis B.irrelevant (§T7)",
        "version": 1,
        "confidence": "established" if otype == "INVARIANCE" else "contested",
        "nodes": [],
        "tau": None,
        "theta_star": None,
        "tol": None,
        "secondary": secondary,
    }
    spec["battery"] = {
        "cue": cue_id,
        "attach": attach,
        "slot": slot,
        "node": node,
        "bank": bank_logical,
        "base": {
            "ref": base.ref,
            "intervention_id": base.intervention_id,
            "arm": level_arm or "A",
        },
        "hold": hold,
        "key_node": key_node,
        "stance": stance,
    }
    return spec, []


def bank_size(bank: Mapping[str, Any], kind: str) -> int:
    """Entries a bank offers (matched pairs count once; stance sums its lists; §T4.4: ≥ 40)."""
    if kind == "entity":
        return len(bank.get("pairs") or [])
    if "entries" in bank:
        return len(bank["entries"] or [])
    return sum(len(bank.get(k) or []) for k in ("option", "value", "dismissive"))


def _hold(ispec: Mapping[str, Any] | None, arm: str) -> dict[str, Any] | None:
    """The base intervention's node at the held arm's value (§T4.4 'context held fixed')."""
    if not ispec:
        return None
    if ispec["form"] != "set":
        if arm == "A":
            return None  # arm A keeps the sampled value under form: shift (§6.5)
        raise HarnessValidationError(
            "battery: a LEVEL base on arm B with form: shift has no fixed held value"
        )
    value = ispec["baseline"] if arm == "A" else ispec["counterfactual"]
    return {"node": ispec["node"], "via": ispec.get("via"), "value": value}


def _cue_text(feature: Mapping[str, Any], cue: Mapping[str, Any]) -> dict[str, Any]:
    """A semantic attribution feature gets the cue's evaluator guide and category (§16)."""
    f = dict(feature)
    if f.get("extractor") == "semantic_verbalisation":
        f["guide"] = cue.get("guide")
        f["category"] = cue.get("category")
    return f


def family_bank_dir(family_id: str) -> Path:
    return family_dir(family_id) / "banks"
