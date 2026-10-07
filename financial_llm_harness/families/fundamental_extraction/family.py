"""fundamental_extraction: X items in F1–F5 on C1 statements and C3 deal terms (spec §34, §T3).

The state carries the C1 drivers (one node per driver and year) and the C3
rights-issue terms; ``statement_pack_of`` rebuilds the statements from it, and
``render_blocks`` shows them as tables. The scorers compute the truth of the
correctness checks the registry names (F3.6.a: roe, roa, roic, dupont; F5.2.a:
terp, dilution) from the state by identity (B.identity). Every convention a
check depends on (ending or average balances, the dilution definition) and
every tolerance is a required argument from the base design (D-23): nothing
here has a default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from financial_llm_harness.contexts import corporate_action as C3
from financial_llm_harness.contexts import statements as C1
from financial_llm_harness.errors import HarnessValidationError

FAMILY_ID = "fundamental_extraction"
MAX_YEARS = 5
PER_YEAR = (
    "growth",
    "gross_margin",
    "opex_ratio",
    "capex_ratio",
    "dso",
    "dio",
    "dpo",
    "accruals",
    "unusual",
    "impairment",
    "payout",
    "buyback",
    "issuance",
    "repayment",
    "share_change",
)
SCALARS = (
    "revenue0",
    "dep_rate",
    "tax_rate",
    "interest_rate",
    "cash0",
    "ppe0",
    "debt0",
    "shares0",
)
CONVENTIONS = ("ending", "average")
DILUTION = ("ownership", "value")


def drivers_of(state: Mapping[str, Any]) -> C1.Drivers:
    years = int(state["years"])
    if not 3 <= years <= MAX_YEARS:
        raise HarnessValidationError(f"years {years} outside 3–{MAX_YEARS} (§T3 C1)")

    def seq(name: str) -> list[float]:
        return [float(state[f"{name}_{k}"]) for k in range(1, years + 1)]

    def val(name: str) -> float:
        return float(state[name])

    return C1.Drivers(
        years=years,
        revenue0=val("revenue0"),
        growth=seq("growth"),
        gross_margin=seq("gross_margin"),
        opex_ratio=seq("opex_ratio"),
        dep_rate=val("dep_rate"),
        capex_ratio=seq("capex_ratio"),
        tax_rate=val("tax_rate"),
        interest_rate=val("interest_rate"),
        dso=seq("dso"),
        dio=seq("dio"),
        dpo=seq("dpo"),
        accruals=seq("accruals"),
        unusual=seq("unusual"),
        impairment=seq("impairment"),
        payout=seq("payout"),
        buyback=seq("buyback"),
        issuance=seq("issuance"),
        repayment=seq("repayment"),
        share_change=seq("share_change"),
        cash0=val("cash0"),
        ppe0=val("ppe0"),
        debt0=val("debt0"),
        shares0=val("shares0"),
    )


def statement_pack_of(state: Mapping[str, Any]) -> C1.StatementPack:
    return C1.statement_pack(drivers_of(state))


BLOCK_NODES = {
    "statements": [
        "years",
        "first_year",
        *SCALARS,
        *(f"{n}_{k}" for n in PER_YEAR for k in range(1, MAX_YEARS + 1)),
    ]
}


def render_blocks(
    state: Mapping[str, Any], nodes: Mapping[str, Any]
) -> dict[str, list[tuple[str, str | None]]]:
    """The statements tables, one block span (§T3 C1)."""
    del nodes
    if "years" not in state:
        return {}
    text = C1.render_pack(statement_pack_of(state), first_year=int(state["first_year"]))
    return {"statements": [(text, "years")]}


def _balance(pack: C1.StatementPack, key: str, convention: str) -> float:
    last = pack.years[-1].balance[key]
    if convention == "ending":
        return last
    if convention == "average":
        prev = pack.years[-2].balance[key] if len(pack.years) > 1 else pack.opening[key]
        return 0.5 * (prev + last)
    raise HarnessValidationError(
        f"balance convention {convention!r} not in {CONVENTIONS}"
    )


def truth_ratios(state: Mapping[str, Any], convention: str) -> dict[str, float]:
    """Last-year ROE, ROA, ROIC and the DuPont factors, by identity."""
    pack = statement_pack_of(state)
    y = pack.years[-1]
    ni, rev = y.income["net_income"], y.income["revenue"]
    equity = _balance(pack, "equity", convention)
    assets = _balance(pack, "total_assets", convention)
    debt = _balance(pack, "debt", convention)
    cash = _balance(pack, "cash", convention)
    nopat = y.income["ebit"] * (1.0 - float(state["tax_rate"]))
    margin, turnover, leverage = ni / rev, rev / assets, assets / equity
    return {
        "roe": ni / equity,
        "roa": ni / assets,
        "roic": nopat / (debt + equity - cash),
        "net_margin": margin,
        "asset_turnover": turnover,
        "equity_multiplier": leverage,
        "dupont_product": margin * turnover * leverage,
    }


def _within(value: Any, want: float, tolerance: float) -> bool:
    try:
        return abs(float(value) - want) <= tolerance
    except (TypeError, ValueError):
        return False


def score_roe(
    state: Mapping[str, Any],
    parsed: Mapping[str, Any],
    *,
    field: str,
    convention: str,
    tolerance: float,
) -> bool:
    return _within(parsed.get(field), truth_ratios(state, convention)["roe"], tolerance)


def score_roa(
    state: Mapping[str, Any],
    parsed: Mapping[str, Any],
    *,
    field: str,
    convention: str,
    tolerance: float,
) -> bool:
    return _within(parsed.get(field), truth_ratios(state, convention)["roa"], tolerance)


def score_roic(
    state: Mapping[str, Any],
    parsed: Mapping[str, Any],
    *,
    field: str,
    convention: str,
    tolerance: float,
) -> bool:
    return _within(
        parsed.get(field), truth_ratios(state, convention)["roic"], tolerance
    )


def score_dupont(
    state: Mapping[str, Any],
    parsed: Mapping[str, Any],
    *,
    fields: Mapping[str, str],
    convention: str,
    tolerance: float,
) -> bool:
    """Each DuPont factor right, and their product equal to ROE (the identity)."""
    t = truth_ratios(state, convention)
    parts = ("net_margin", "asset_turnover", "equity_multiplier")
    if not all(_within(parsed.get(fields[p]), t[p], tolerance) for p in parts):
        return False
    prod = 1.0
    for p in parts:
        prod *= float(parsed[fields[p]])
    return abs(prod - t["roe"]) <= tolerance * max(1.0, abs(t["roe"]))


def truth_terp(state: Mapping[str, Any]) -> float:
    return C3.terp(
        float(state["share_price"]),
        float(state["shares_outstanding"]),
        float(state["new_shares"]),
        float(state["subscription_price"]),
    )


def truth_dilution(state: Mapping[str, Any], definition: str) -> float:
    """Ownership dilution n/(N + n) of a non-participating holder, or value dilution (P − TERP)/P."""
    n, big_n = float(state["new_shares"]), float(state["shares_outstanding"])
    if definition == "ownership":
        return n / (big_n + n)
    if definition == "value":
        p = float(state["share_price"])
        return (p - truth_terp(state)) / p
    raise HarnessValidationError(
        f"dilution definition {definition!r} not in {DILUTION}"
    )


def score_terp(
    state: Mapping[str, Any], parsed: Mapping[str, Any], *, field: str, tolerance: float
) -> bool:
    return _within(parsed.get(field), truth_terp(state), tolerance)


def score_dilution(
    state: Mapping[str, Any],
    parsed: Mapping[str, Any],
    *,
    field: str,
    definition: str,
    tolerance: float,
) -> bool:
    return _within(parsed.get(field), truth_dilution(state, definition), tolerance)
