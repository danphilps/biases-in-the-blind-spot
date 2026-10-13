"""C3: corporate-action announcements and their computable truth (spec §T3, B.identity).

Pure functions of the planted deal terms: the theoretical ex-rights price
(TERP), EPS accretion or dilution of an acquisition, the offer premium,
buyback EPS effect and dividend cover. These are identities, so the oracle
computes them from the state, never from the prose.
"""

from __future__ import annotations

from dataclasses import dataclass

from financial_llm_harness.errors import HarnessValidationError


def terp(
    price: float, shares: float, new_shares: float, subscription_price: float
) -> float:
    """TERP = (N·P + n·S) / (N + n) for n new shares at S on N existing at P."""
    if shares <= 0 or new_shares < 0:
        raise HarnessValidationError("TERP: share counts must be positive")
    return (shares * price + new_shares * subscription_price) / (shares + new_shares)


def premium(offer_price: float, undisturbed_price: float) -> float:
    return offer_price / undisturbed_price - 1.0


@dataclass(frozen=True)
class Deal:
    acquirer_net_income: float
    acquirer_shares: float
    target_net_income: float
    offer_value: float  # total consideration
    stock_fraction: float  # share of the consideration paid in new acquirer shares
    acquirer_price: float  # issue price of the new shares
    debt_rate: float  # pre-tax cost of the debt-funded part
    cash_yield: float  # pre-tax return forgone on the cash-funded part
    cash_fraction: (
        float  # share paid from cash on hand (the rest of the non-stock part is debt)
    )
    synergies: float  # pre-tax run-rate synergies counted in the year
    tax_rate: float


def accretion(deal: Deal) -> dict[str, float]:
    """Pro-forma EPS against standalone EPS (accretive when > 0)."""
    if (
        not 0 <= deal.stock_fraction <= 1
        or not 0 <= deal.cash_fraction <= 1 - deal.stock_fraction
    ):
        raise HarnessValidationError(
            "accretion: financing fractions must sum to at most 1"
        )
    new_shares = deal.offer_value * deal.stock_fraction / deal.acquirer_price
    debt = deal.offer_value * (1.0 - deal.stock_fraction - deal.cash_fraction)
    cash = deal.offer_value * deal.cash_fraction
    after_tax = 1.0 - deal.tax_rate
    pro_forma_ni = (
        deal.acquirer_net_income
        + deal.target_net_income
        + deal.synergies * after_tax
        - debt * deal.debt_rate * after_tax
        - cash * deal.cash_yield * after_tax
    )
    standalone = deal.acquirer_net_income / deal.acquirer_shares
    pro_forma = pro_forma_ni / (deal.acquirer_shares + new_shares)
    return {
        "standalone_eps": standalone,
        "pro_forma_eps": pro_forma,
        "accretion": pro_forma / standalone - 1.0,
        "new_shares": new_shares,
    }


def buyback_eps(
    net_income: float,
    shares: float,
    amount: float,
    price: float,
    cash_yield: float,
    tax_rate: float,
) -> float:
    """EPS change of a buyback funded from cash: fewer shares, less after-tax interest income."""
    bought = amount / price
    new_ni = net_income - amount * cash_yield * (1.0 - tax_rate)
    return (new_ni / (shares - bought)) / (net_income / shares) - 1.0


def dividend_cover(eps: float, dps: float) -> float:
    if dps <= 0:
        raise HarnessValidationError("dividend cover needs a positive dividend")
    return eps / dps
