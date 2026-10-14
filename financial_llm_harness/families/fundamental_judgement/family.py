"""fundamental_judgement: derived nodes, inversions, bank draws and the fake response (spec §6, §20.11, §35 S1)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from financial_llm_harness.families import bank_entries

FAMILY_ID = "fundamental_judgement"
FACTORS = (
    "valuation",
    "growth",
    "quality",
    "leverage",
    "momentum",
    "size",
    "sector",
    "other",
)

# --- sampled primitives (fn(name) distributions, §6.1) -----------------------


def sample_price_12m_ago(rng: np.random.Generator, drawn: Mapping[str, Any]) -> float:
    """price_12m_ago from the latent momentum draw: price_now / (1 + momentum_draw)."""
    return float(drawn["price_now"]) / (1.0 + float(drawn["momentum_draw"]))


def _draw(name: str, rng: np.random.Generator) -> str:
    entries = bank_entries(FAMILY_ID, name)
    return str(entries[int(rng.integers(0, len(entries)))])


def draw_company_name(rng: np.random.Generator, drawn: Mapping[str, Any]) -> str:
    return _draw("company_names", rng)


def draw_ceo_name(rng: np.random.Generator, drawn: Mapping[str, Any]) -> str:
    return _draw("ceo_names", rng)


def draw_hq_city(rng: np.random.Generator, drawn: Mapping[str, Any]) -> str:
    return _draw("hq_cities", rng)


# --- derived nodes -------------------------------------------------------------


def momentum_12m(price_now: float, price_12m_ago: float) -> float:
    return float(price_now) / float(price_12m_ago) - 1.0


def pe(price_now: float, eps_ttm: float) -> float:
    eps = float(eps_ttm)
    return float(price_now) / eps if eps != 0 else math.inf


def ev_ebitda(
    market_cap: float, net_debt_ebitda: float, revenue: float, ebit_margin: float
) -> float:
    """EV / EBITDA with EBITDA proxied by revenue × EBIT margin; EV includes net debt."""
    ebitda = float(revenue) * float(ebit_margin)
    if ebitda <= 0:
        return math.inf
    return (float(market_cap) + float(net_debt_ebitda) * ebitda) / ebitda


# --- inversions for interventions on derived nodes (§6.4) ----------------------


def invert_momentum_12m_via_price_12m_ago(
    state: Mapping[str, Any], target: float
) -> float:
    return float(state["price_now"]) / (1.0 + float(target))


def invert_pe_via_eps_ttm(state: Mapping[str, Any], target: float) -> float:
    return float(state["price_now"]) / float(target)


# --- fake adapter response (§20.11) ---------------------------------------------

_REASONING = (
    "The decision weighs the company's fundamentals against the mandate.",
    "Considering the profile as a whole relative to the stated mandate.",
    "The overall balance of the figures provided drives this view.",
    "Taking the reported metrics together under the mandate's criteria.",
)


def fake_response(
    contract_id: str,
    y: int | float,
    mention: bool,
    rng: np.random.Generator,
    concept_factor: str | None = None,
    reasoning_mode: str = "reasoning_before",
) -> str:
    """A contract-valid response whose primary feature equals ``y``."""
    if contract_id == "stock_attractive_v1":
        return _fake_attractive(y, rng, reasoning_mode)
    if contract_id != "stock_add_v1":
        raise ValueError(f"fake_response: unknown contract {contract_id!r}")
    decision = "ADD" if int(y) == 1 else "DO_NOT_ADD"
    stance = "supports" if decision == "ADD" else "opposes"
    others = [f for f in FACTORS if f not in ("other", concept_factor)]
    picks = [others[int(i)] for i in rng.choice(len(others), size=2, replace=False)]
    factors = [{"factor": f, "stance": stance} for f in picks]
    if mention and concept_factor is not None:
        factors.insert(
            int(rng.integers(0, len(factors) + 1)),
            {"factor": concept_factor, "stance": stance},
        )
    text = _REASONING[int(rng.integers(0, len(_REASONING)))]
    confidence = round(float(rng.uniform(0.5, 0.9)), 2)
    if reasoning_mode == "reasoning_before":
        body: dict[str, Any] = {
            "reasoning": text,
            "decision": decision,
            "confidence": confidence,
            "material_factors": factors,
        }
    elif reasoning_mode == "rationale_after":
        body = {
            "decision": decision,
            "confidence": confidence,
            "material_factors": factors,
            "rationale": text,
        }
    else:
        body = {
            "decision": decision,
            "confidence": confidence,
            "material_factors": factors,
        }
    return json.dumps(body)


def _fake_attractive(
    y: int | float, rng: np.random.Generator, reasoning_mode: str
) -> str:
    """stock_attractive_v1: decision, confidence and free text, with no factor list."""
    decision = "ATTRACTIVE" if int(y) == 1 else "NOT_ATTRACTIVE"
    text = _REASONING[int(rng.integers(0, len(_REASONING)))]
    confidence = round(float(rng.uniform(0.5, 0.9)), 2)
    if reasoning_mode == "reasoning_before":
        body: dict[str, Any] = {
            "reasoning": text,
            "decision": decision,
            "confidence": confidence,
        }
    elif reasoning_mode == "rationale_after":
        body = {"decision": decision, "confidence": confidence, "rationale": text}
    else:
        body = {"decision": decision, "confidence": confidence}
    return json.dumps(body)
