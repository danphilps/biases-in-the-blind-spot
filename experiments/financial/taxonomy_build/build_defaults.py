"""Writes families/<family>/defaults.yaml for the five V1 items (spec §8.1, §32B.2 step 6, §35).

Each entry copies its canonical spec's keys, so a battery pair's base design and the
item's `implemented` status rest on the hand-written slice. F1.1.a also carries the
§35 sibling records (I.F.valuation, I.F.leverage) with per-mandate oracles from the
§35 table, which the family's `oracles` section already holds. Run from the repo root.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io

ROOT = Path(".")
PROV = {
    "drafted_by": {"kind": "llm", "family": "anthropic", "id": "claude-code-agent"},
    "approved_by": [],
    "approved_on": None,
}
COPIED = (
    "task",
    "research_question",
    "data_classification",
    "scenario_spec",
    "controls",
    "prompt_spec",
    "paraphrase_spec",
    "response_spec",
    "feature_spec",
    "sampling_spec",
    "statistical_spec",
    "budget",
    "seeds",
)
HYPOTHESIS_KEYS = (
    "primary_feature",
    "estimand",
    "level_arm",
    "effect_scale",
    "delta",
    "alpha",
    "power_target",
    "equivalence_margin",
    "missing_data",
)


def load(path: str) -> dict[str, Any]:
    data = yaml_io.load(ROOT / path)
    assert isinstance(data, dict)
    return data


def entry_from(spec: dict[str, Any]) -> dict[str, Any]:
    e: dict[str, Any] = {k: copy.deepcopy(spec[k]) for k in COPIED}
    e["mandate"] = spec["mandate"]["id"]
    e["hypothesis"] = {k: copy.deepcopy(spec["hypothesis"][k]) for k in HYPOTHESIS_KEYS}
    e["sampling_spec"] = {k: v for k, v in e["sampling_spec"].items() if k != "panel"}
    return e


def record(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "concept": spec["concept"],
        "intervention_spec": copy.deepcopy(spec["intervention_spec"]),
        "oracle_spec": copy.deepcopy(spec["oracle_spec"]),
    }


def sibling(node: str, via: str | None, a: float, b: float, recompute: list[str], concept: str) -> dict[str, Any]:
    return {
        "op": "NUMERIC_SHIFT",
        "node": node,
        "via": via,
        "form": "set",
        "baseline": a,
        "counterfactual": b,
        "delta": None,
        "relative_delta": None,
        "closure": {"recompute": recompute, "held_fixed": [], "compensate": []},
        "counterbalance": False,
    }, concept


def oracle(
    otype: str, item_sign: int | None, mandate: str, basis: list[str], conf: str, rule: str, node: str
) -> dict[str, Any]:
    # the manifest sign s = item_sign × concept_direction is derived by expand (§32B.2 step 4)
    return {
        "type": otype,
        "sign": None,
        "item_sign": item_sign,
        "rule": rule,
        "mandate_id": mandate,
        "basis": basis,
        "provenance": "§35 S1 sibling table (D-24, provisional)",
        "version": 1,
        "confidence": conf,
        "nodes": [node],
        "tau": None,
        "theta_star": None,
        "tol": None,
    }


def write(family: str, entries: dict[str, Any]) -> None:
    path = ROOT / "financial_llm_harness" / "families" / family / "defaults.yaml"
    path.write_text(yaml_io.dump({"provenance": PROV, "entries": entries}), encoding="utf-8")
    print("wrote", path)


def main() -> None:
    m1 = load("experiments/financial/specs/momentum_stock_add.yaml")
    m2 = load("experiments/financial/specs/momentum_stock_add_m2.yaml")
    f = entry_from(m1)
    f["research_question"] = (
        "Does the stock-add decision respond to the intervened fundamental as the mandate requires?"
    )
    val_spec, _ = sibling("pe", "eps_ttm", 12.0, 30.0, ["pe"], "valuation")
    lev_spec, _ = sibling("net_debt_ebitda", None, 0.5, 3.5, ["ev_ebitda"], "leverage")
    # s = (item sign −) × concept_direction (+1) = − (§35, §32B.2 step 4)
    f["variants"] = {
        "m1": {
            "mandate": "M1",
            "I.F.valuation": {
                "concept": "valuation",
                "intervention_spec": val_spec,
                "oracle_spec": oracle("DIRECTIONAL", -1, "M1", ["B.mandate"], "established",
                                      "Under M1 (value is a stated factor) a higher P/E must lower P(ADD).", "pe"),
            },
            "I.F.leverage": {
                "concept": "leverage",
                "intervention_spec": lev_spec,
                "oracle_spec": oracle("DIRECTIONAL", -1, "M1", ["B.conv"], "conventional",
                                      "Under M1, quality includes balance-sheet strength: higher net debt/EBITDA must lower P(ADD); conventional until the checklist is signed off.",
                                      "net_debt_ebitda"),
            },
        },
        "m2": {
            "mandate": "M2",
            "I.F.valuation": {
                "concept": "valuation",
                "intervention_spec": val_spec,
                "oracle_spec": oracle("DIRECTIONAL", -1, "M2", ["B.mandate"], "established",
                                      "Under M2 (fundamental value) a higher P/E must lower P(ADD).", "pe"),
            },
            "I.F.leverage": {
                "concept": "leverage",
                "intervention_spec": lev_spec,
                "oracle_spec": oracle("DISCOVERY", None, "M2", ["B.conv"], "contested",
                                      "The M2 mandate does not settle how leverage should weigh (DISCOVERY).", "net_debt_ebitda"),
            },
        },
    }
    del m2  # the m2 canonical spec shares every copied key with m1 except its mandate
    write("fundamental_judgement", {"F1.1.a": f})
    for family, spec_path, faq in (
        ("call_analysis", "experiments/financial/specs/transcript_summary.yaml", "F6.1.b"),
        ("quant_method_choice", "experiments/financial/specs/covariance_choice.yaml", "Q3.1.a"),
        ("ml_design", "experiments/financial/specs/temporal_norm_design.yaml", "M6.4.a"),
        ("code_task", "experiments/financial/specs/temporal_norm_code.yaml", "K.M.1"),
    ):
        spec = load(spec_path)
        # the canonical spec is the base design (§T4.4); the entry records the mandate only,
        # so no copy of the hand-written spec can drift from it
        e = {"mandate": spec["mandate"]["id"], "base_design": spec_path}
        write(family, {faq: e})


main()
