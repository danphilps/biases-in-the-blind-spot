"""Writes family.yaml for the four §34 families M9a adds (spec §34, §38 M9a, §T11).

Each family lists its V1.1 items (§T11), its §34 contexts and the attach slots its
templates will support (only the universal preamble until templates exist), and a
scenario schema whose primitives are the parameters of the context generator it
renders (C1, C3, C5k, C7). No distribution, intervention, oracle, contract or
tolerance is written: those are the base design's (D-23) and stay
`unimplemented_content` in expand_report.json until the panel supplies them.
Run from the repo root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io

ROOT = Path("financial_llm_harness/families")
PROV = {
    "drafted_by": {"kind": "llm", "family": "anthropic", "id": "claude-code-agent"},
    "approved_by": [],
    "approved_on": None,
}


def num(
    display: str, precision: int, unit: str | None = None, relevant: bool = True
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "kind": "primitive",
        "type": "numeric",
        "precision": precision,
        "display": display,
    }
    if unit:
        d["unit"] = unit
    if relevant:
        d["relevant"] = True
    return d


def cat(relevant: bool = True) -> dict[str, Any]:
    d: dict[str, Any] = {"kind": "primitive", "type": "categorical", "display": "text"}
    if relevant:
        d["relevant"] = True
    return d


MONEY_M = ("money_m", 0, "USD")
PCT = ("pct", 2, None)


def c1_nodes() -> dict[str, Any]:
    nodes: dict[str, Any] = {
        "years": num("int", 0),
        "first_year": num("int", 0, relevant=False),
        "revenue0": num(*MONEY_M),
        "dep_rate": num(*PCT),
        "tax_rate": num(*PCT),
        "interest_rate": num(*PCT),
        "cash0": num(*MONEY_M),
        "ppe0": num(*MONEY_M),
        "debt0": num(*MONEY_M),
        "shares0": num("int", 0),
    }
    per_year = {
        "growth": PCT,
        "gross_margin": PCT,
        "opex_ratio": PCT,
        "capex_ratio": PCT,
        "dso": ("number", 1, None),
        "dio": ("number", 1, None),
        "dpo": ("number", 1, None),
        "accruals": MONEY_M,
        "unusual": MONEY_M,
        "impairment": MONEY_M,
        "payout": PCT,
        "buyback": MONEY_M,
        "issuance": MONEY_M,
        "repayment": MONEY_M,
        "share_change": PCT,
    }
    for name, (disp, prec, unit) in per_year.items():
        for k in range(1, 6):
            nodes[f"{name}_{k}"] = num(disp, prec, unit)
    return nodes


def c3_rights_nodes() -> dict[str, Any]:
    return {
        "share_price": num("money", 2, "USD"),
        "shares_outstanding": num("int", 0),
        "new_shares": num("int", 0),
        "subscription_price": num("money", 2, "USD"),
    }


def c3_deal_nodes() -> dict[str, Any]:
    return {
        "net_income": num(*MONEY_M),
        "shares_outstanding": num("int", 0),
        "buyback_amount": num(*MONEY_M),
        "buyback_price": num("money", 2, "USD"),
        "intrinsic_value": num("money", 2, "USD"),
        "funding": cat(),
        "cash_yield": num(*PCT),
        "tax_rate": num(*PCT),
        "acquirer_net_income": num(*MONEY_M),
        "acquirer_shares": num("int", 0),
        "target_net_income": num(*MONEY_M),
        "offer_value": num(*MONEY_M),
        "stock_fraction": num(*PCT),
        "acquirer_price": num("money", 2, "USD"),
        "debt_rate": num(*PCT),
        "cash_fraction": num(*PCT),
        "synergies": num(*MONEY_M),
    }


def c5k_nodes() -> dict[str, Any]:
    return {
        "k_notes": num("int", 0),
        "noise": num(*PCT, relevant=False),
        "d": num("int", 0, relevant=False),
        "k": num("int", 0, relevant=False),
        "a": num("int", 0, relevant=False),
        "n": num("int", 0, relevant=False),
        "difference_offset": num(*PCT, relevant=False),
        "inaccuracy_offset": num(*PCT, relevant=False),
        "rounding": num("pct", 3, None, relevant=False),
        "set_seed": num("int", 0, relevant=False),
    }


def c7_nodes() -> dict[str, Any]:
    nodes: dict[str, Any] = {
        "n_assets": num("int", 0),
        "n_dates": num("int", 0),
        "n_factors": num("int", 0),
        "n_signals": num("int", 0),
        "beta_sd": num("number", 3),
        "resid_vol": num("pct", 2),
        "t_dof": num("number", 1),
        "publication_lag": num("int", 0),
    }
    for i in range(1, 4):
        nodes[f"factor_vol_{i}"] = num("pct", 2)
    for i in range(1, 6):
        nodes[f"gamma_{i}"] = num(
            "number", 4, relevant=False
        )  # planted: never shown in the prompt
    return nodes


FAMILIES = {
    "fundamental_extraction": {
        "description": "X items in F1–F5 (spec §34); contexts C1, C3, rendered by the C1 and C3 generators.",
        "contexts": ["C1", "C3"],
        "faq_ids": ["F3.4.a", "F3.6.a", "F3.7.a", "F5.2.a"],
        "nodes": {**c1_nodes(), **c3_rights_nodes()},
        "scorers": {
            "roe": {"fn": "score_roe", "args": ["field", "convention", "tolerance"]},
            "roa": {"fn": "score_roa", "args": ["field", "convention", "tolerance"]},
            "roic": {"fn": "score_roic", "args": ["field", "convention", "tolerance"]},
            "dupont": {
                "fn": "score_dupont",
                "args": ["fields", "convention", "tolerance"],
            },
            "terp": {"fn": "score_terp", "args": ["field", "tolerance"]},
            "dilution": {
                "fn": "score_dilution",
                "args": ["field", "definition", "tolerance"],
            },
        },
    },
    "corporate_action": {
        "description": "F4 and F5 judgement items (spec §34); contexts C1, C3.",
        "contexts": ["C1", "C3"],
        "faq_ids": ["F4.2.a"],
        "nodes": c3_deal_nodes(),
        "scorers": {},
    },
    "broker_synthesis": {
        "description": "F7 S and D items (spec §34); contexts C1, C5, C5k (§T9.1).",
        "contexts": ["C1", "C5", "C5k"],
        "faq_ids": ["F7.3.b", "F7.3.d"],
        "nodes": c5k_nodes(),
        "scorers": {},
    },
    "quant_dgp": {
        "description": "Q1–Q6 items on a dataset with a known DGP (spec §34); contexts C6, C7 (§T9.2).",
        "contexts": ["C6", "C7"],
        "faq_ids": ["Q1.1.c", "Q2.2.b"],
        "nodes": c7_nodes(),
        "scorers": {},
    },
}


def main() -> None:
    for fid, f in FAMILIES.items():
        doc = {
            "provenance": PROV,
            "family_id": fid,
            "description": f["description"],
            "contexts": f["contexts"],
            "faq_ids": f["faq_ids"],
            "battery_slots": {"preamble": "preamble"},
            "scenario_schema": {"nodes": f["nodes"]},
            "interventions": {},
            "response_contracts": [],
            "features": {},
            "oracles": {},
            "scorers": f["scorers"],
        }
        d = ROOT / fid
        d.mkdir(parents=True, exist_ok=True)
        (d / "family.yaml").write_text(yaml_io.dump(doc), encoding="utf-8")
        init = d / "__init__.py"
        if not init.exists():
            init.write_text(f'"""Family ``{fid}`` (spec §34)."""\n', encoding="utf-8")
        print("wrote", d / "family.yaml")


main()
