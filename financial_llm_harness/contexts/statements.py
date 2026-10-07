"""C1: financial statements pack with planted ground truth (spec §T3).

``statement_pack`` builds 3–5 fiscal years of income statement, balance sheet
and cash-flow statement from explicit drivers. Every flow is booked twice, so
the accounting identities hold by construction and ``identity_violations``
checks them. Planted, and returned as ground truth: the accrual level (non-cash
income booked to other current assets), the cash conversion cycle, unusual
items, impairments, share-count changes and the debt-maturity schedule.

No driver has a default: every value comes from the caller (a family's
scenario state), so this module plants nothing the experiment did not declare.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from financial_llm_harness.errors import HarnessValidationError

DAYS = 365.0
TOL = 1e-6


@dataclass(frozen=True)
class Drivers:
    """Per-year drivers (sequences of length ``years``) and opening balances."""

    years: int
    revenue0: float
    growth: Sequence[float]
    gross_margin: Sequence[float]
    opex_ratio: Sequence[float]
    dep_rate: float
    capex_ratio: Sequence[float]
    tax_rate: float
    interest_rate: float
    dso: Sequence[float]
    dio: Sequence[float]
    dpo: Sequence[float]
    accruals: Sequence[float]  # planted non-cash income per year
    unusual: Sequence[float]  # planted unusual or infrequent charges (+) per year
    impairment: Sequence[float]  # planted impairments of PP&E per year
    payout: Sequence[float]
    buyback: Sequence[float]
    issuance: Sequence[float]
    repayment: Sequence[float]  # the debt-maturity schedule
    share_change: Sequence[float]  # fractional change in share count per year
    cash0: float
    ppe0: float
    debt0: float
    shares0: float
    other_ca0: float = 0.0


@dataclass
class Year:
    income: dict[str, float] = field(default_factory=dict)
    balance: dict[str, float] = field(default_factory=dict)
    cash_flow: dict[str, float] = field(default_factory=dict)
    shares: float = 0.0


@dataclass
class StatementPack:
    years: list[Year]
    opening: dict[str, float]
    truth: dict[str, Any]


def _check(d: Drivers) -> None:
    if not 3 <= d.years <= 5:
        raise HarnessValidationError("C1: a statements pack spans 3–5 years (§T3)")
    for name in (
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
    ):
        if len(getattr(d, name)) != d.years:
            raise HarnessValidationError(f"C1: driver {name} needs {d.years} values")


def statement_pack(d: Drivers) -> StatementPack:
    _check(d)
    rev_prev = d.revenue0
    cogs_prev = d.revenue0 * (1.0 - d.gross_margin[0])
    ar = rev_prev * d.dso[0] / DAYS
    inv = cogs_prev * d.dio[0] / DAYS
    ap = cogs_prev * d.dpo[0] / DAYS
    cash, ppe, debt, oca, shares = d.cash0, d.ppe0, d.debt0, d.other_ca0, d.shares0
    equity = cash + ar + inv + oca + ppe - ap - debt
    if equity <= 0:
        raise HarnessValidationError("C1: opening equity must be positive")
    opening = {
        "cash": cash,
        "receivables": ar,
        "inventory": inv,
        "other_current_assets": oca,
        "ppe": ppe,
        "payables": ap,
        "debt": debt,
        "equity": equity,
    }
    years: list[Year] = []
    for t in range(d.years):
        rev = rev_prev * (1.0 + d.growth[t])
        cogs = rev * (1.0 - d.gross_margin[t])
        sga = rev * d.opex_ratio[t]
        da = ppe * d.dep_rate
        ebit = rev + d.accruals[t] - cogs - sga - da - d.unusual[t] - d.impairment[t]
        interest = debt * d.interest_rate
        ebt = ebit - interest
        tax = max(ebt, 0.0) * d.tax_rate
        ni = ebt - tax
        ar1 = rev * d.dso[t] / DAYS
        inv1 = cogs * d.dio[t] / DAYS
        ap1 = cogs * d.dpo[t] / DAYS
        oca1 = oca + d.accruals[t]
        cfo = (
            ni
            + da
            + d.impairment[t]
            - (ar1 - ar)
            - (inv1 - inv)
            + (ap1 - ap)
            - (oca1 - oca)
        )
        capex = rev * d.capex_ratio[t]
        cfi = -capex
        dividends = max(ni, 0.0) * d.payout[t]
        cff = d.issuance[t] - d.repayment[t] - dividends - d.buyback[t]
        cash1 = cash + cfo + cfi + cff
        ppe1 = ppe + capex - da - d.impairment[t]
        debt1 = debt + d.issuance[t] - d.repayment[t]
        equity1 = equity + ni - dividends - d.buyback[t]
        shares1 = shares * (1.0 + d.share_change[t])
        y = Year(
            income={
                "revenue": rev,
                "other_income_accrued": d.accruals[t],
                "cogs": cogs,
                "gross_profit": rev - cogs,
                "sga": sga,
                "depreciation": da,
                "unusual_items": d.unusual[t],
                "impairment": d.impairment[t],
                "ebit": ebit,
                "interest": interest,
                "pretax_income": ebt,
                "tax": tax,
                "net_income": ni,
            },
            balance={
                "cash": cash1,
                "receivables": ar1,
                "inventory": inv1,
                "other_current_assets": oca1,
                "ppe": ppe1,
                "total_assets": cash1 + ar1 + inv1 + oca1 + ppe1,
                "payables": ap1,
                "debt": debt1,
                "total_liabilities": ap1 + debt1,
                "equity": equity1,
            },
            cash_flow={
                "net_income": ni,
                "depreciation": da,
                "impairment": d.impairment[t],
                "change_receivables": -(ar1 - ar),
                "change_inventory": -(inv1 - inv),
                "change_payables": ap1 - ap,
                "change_other_current_assets": -(oca1 - oca),
                "cfo": cfo,
                "capex": -capex,
                "cfi": cfi,
                "debt_issued": d.issuance[t],
                "debt_repaid": -d.repayment[t],
                "dividends": -dividends,
                "buybacks": -d.buyback[t],
                "cff": cff,
                "net_change_in_cash": cfo + cfi + cff,
            },
            shares=shares1,
        )
        years.append(y)
        rev_prev, ar, inv, ap, oca = rev, ar1, inv1, ap1, oca1
        cash, ppe, debt, equity, shares = cash1, ppe1, debt1, equity1, shares1
    return StatementPack(years, opening, planted_truth(years, opening, d))


def planted_truth(
    years: list[Year], opening: Mapping[str, float], d: Drivers
) -> dict[str, Any]:
    """The ground truth the oracle reads (never the prose)."""
    out: dict[str, Any] = {
        "accruals_ratio": [],
        "cfo_to_net_income": [],
        "ccc_days": [],
    }
    prev_assets = sum(
        opening[k]
        for k in ("cash", "receivables", "inventory", "other_current_assets", "ppe")
    )
    for t, y in enumerate(years):
        avg_assets = 0.5 * (prev_assets + y.balance["total_assets"])
        ni, cfo = y.income["net_income"], y.cash_flow["cfo"]
        out["accruals_ratio"].append((ni - cfo) / avg_assets)
        out["cfo_to_net_income"].append(cfo / ni if ni != 0 else None)
        out["ccc_days"].append(d.dso[t] + d.dio[t] - d.dpo[t])
        prev_assets = y.balance["total_assets"]
    out["unusual_items"] = [(t, v) for t, v in enumerate(d.unusual) if v]
    out["impairments"] = [(t, v) for t, v in enumerate(d.impairment) if v]
    out["share_count"] = [y.shares for y in years]
    out["debt_maturities"] = [(t, v) for t, v in enumerate(d.repayment) if v]
    return out


def identity_violations(pack: StatementPack, tol: float = TOL) -> list[str]:
    """Accounting identities: A = L + E, cash roll-forward, equity roll-forward, CFO bridge."""
    bad: list[str] = []
    cash_prev = pack.opening["cash"]
    eq_prev = pack.opening["equity"]
    for t, y in enumerate(pack.years):
        b, c, i = y.balance, y.cash_flow, y.income
        scale = max(1.0, abs(b["total_assets"]))
        if abs(b["total_assets"] - b["total_liabilities"] - b["equity"]) > tol * scale:
            bad.append(f"year {t}: assets != liabilities + equity")
        if abs(cash_prev + c["cfo"] + c["cfi"] + c["cff"] - b["cash"]) > tol * scale:
            bad.append(f"year {t}: cash roll-forward")
        if (
            abs(
                eq_prev + i["net_income"] + c["dividends"] + c["buybacks"] - b["equity"]
            )
            > tol * scale
        ):
            bad.append(f"year {t}: equity roll-forward")
        bridge = sum(
            c[k]
            for k in (
                "net_income",
                "depreciation",
                "impairment",
                "change_receivables",
                "change_inventory",
                "change_payables",
                "change_other_current_assets",
            )
        )
        if abs(bridge - c["cfo"]) > tol * scale:
            bad.append(f"year {t}: CFO bridge")
        cash_prev, eq_prev = b["cash"], b["equity"]
    return bad


# ---------------------------------------------------------------------------
# Rendering and the deterministic round-trip scorer (§36: planted truth recoverable from text)
# ---------------------------------------------------------------------------

SECTIONS = (
    ("Income statement", "income"),
    ("Balance sheet", "balance"),
    ("Cash flow statement", "cash_flow"),
)
_ROW = re.compile(r"^\| ([a-z_]+) \| (.+) \|$")


def render_pack(
    pack: StatementPack, first_year: int, scale: float = 1e6, decimals: int = 1
) -> str:
    """Markdown tables, one per statement, in units of ``scale`` (default millions)."""
    years = [str(first_year + t) for t in range(len(pack.years))]
    out: list[str] = []
    for title, attr in SECTIONS:
        out += [
            f"### {title}",
            "",
            "| line | " + " | ".join(years) + " |",
            "|---|" + "---|" * len(years),
        ]
        for key in getattr(pack.years[0], attr):
            vals = [f"{getattr(y, attr)[key] / scale:.{decimals}f}" for y in pack.years]
            out.append(f"| {key} | " + " | ".join(vals) + " |")
        out.append("")
    out += ["### Opening balances", "", "| line | value |", "|---|---|"]
    out += [f"| {k} | {v / scale:.{decimals}f} |" for k, v in pack.opening.items()]
    return "\n".join(out) + "\n"


def parse_pack(text: str) -> dict[str, dict[str, list[float]]]:
    """Section → line → values, read back from ``render_pack`` text."""
    out: dict[str, dict[str, list[float]]] = {}
    section = ""
    for line in text.splitlines():
        if line.startswith("### "):
            section = line[4:]
            out[section] = {}
            continue
        m = _ROW.match(line)
        if m and m.group(1) != "line":
            out[section][m.group(1)] = [float(x) for x in m.group(2).split(" | ")]
    return out


def ratios_from_text(text: str) -> dict[str, list[float]]:
    """The accruals ratio and CFO/NI recomputed from the rendered tables alone."""
    t = parse_pack(text)
    inc, bal, cf = t["Income statement"], t["Balance sheet"], t["Cash flow statement"]
    opening = t["Opening balances"]
    prev = sum(
        opening[k][0]
        for k in ("cash", "receivables", "inventory", "other_current_assets", "ppe")
    )
    acc: list[float] = []
    conv: list[float] = []
    for i, ta in enumerate(bal["total_assets"]):
        acc.append((inc["net_income"][i] - cf["cfo"][i]) / (0.5 * (prev + ta)))
        conv.append(cf["cfo"][i] / inc["net_income"][i])
        prev = ta
    return {"accruals_ratio": acc, "cfo_to_net_income": conv}
