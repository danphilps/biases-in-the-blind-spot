"""corporate_action: F4 and F5 judgement items on C3 announcements (spec §34, §T3).

The state carries the deal or buyback terms; the truth functions compute the
C3 identities (accretion, buyback EPS effect, premium, TERP) from it. Every
value and tolerance is the base design's (D-23).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from financial_llm_harness.contexts import corporate_action as C3

FAMILY_ID = "corporate_action"


def truth_buyback_eps(state: Mapping[str, Any]) -> float:
    """EPS effect of a cash-funded buyback at the stated price (§T4.1 I.F.capital_return)."""
    return C3.buyback_eps(
        float(state["net_income"]),
        float(state["shares_outstanding"]),
        float(state["buyback_amount"]),
        float(state["buyback_price"]),
        float(state["cash_yield"]),
        float(state["tax_rate"]),
    )


def truth_price_vs_value(state: Mapping[str, Any]) -> float:
    """Buyback price relative to the stated intrinsic value (> 0: paying above value)."""
    return float(state["buyback_price"]) / float(state["intrinsic_value"]) - 1.0


def truth_accretion(state: Mapping[str, Any]) -> dict[str, float]:
    return C3.accretion(
        C3.Deal(**{f: float(state[f]) for f in C3.Deal.__dataclass_fields__})
    )
