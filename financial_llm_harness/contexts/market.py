"""C2: market data panel with planted momentum, volatility, beta and events (spec §T3).

``market_panel`` simulates daily log returns r = α + β·m + σ_e·ε for one stock
against a simulated market path, then adds a constant drift so the realised
trailing return over the momentum window equals the planted momentum
exactly. Event days carry planted surprise returns; estimate revisions are
planted per event. Every magnitude is a caller-supplied parameter; the
random draws come from the caller's generator (§6.2, §12).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from financial_llm_harness.errors import HarnessValidationError


@dataclass(frozen=True)
class MarketParams:
    n_days: int
    momentum_window: int  # trading days, e.g. 252 for 12-month momentum
    momentum: float  # planted simple return over the window ending on the last day
    beta: float
    market_mu: float  # daily log drift of the market
    market_sigma: float  # daily volatility of the market
    idio_sigma: float  # daily idiosyncratic volatility
    price_now: float
    volume_mean: float
    event_days: Sequence[int]  # indices of planted events
    event_surprise: Sequence[float]  # planted event-day log returns
    revision: Sequence[float]  # planted estimate revision at each event (fraction)


@dataclass
class MarketPanel:
    prices: np.ndarray
    market: np.ndarray
    returns: np.ndarray
    market_returns: np.ndarray
    volume: np.ndarray
    truth: dict[str, Any]


def market_panel(p: MarketParams, rng: np.random.Generator) -> MarketPanel:
    if not 0 < p.momentum_window < p.n_days:
        raise HarnessValidationError(
            "C2: the momentum window must lie inside the panel"
        )
    if len(p.event_days) != len(p.event_surprise) or len(p.event_days) != len(
        p.revision
    ):
        raise HarnessValidationError("C2: one surprise and one revision per event day")
    if any(not 0 < d < p.n_days for d in p.event_days):
        raise HarnessValidationError("C2: event days must lie inside the panel")
    m = p.market_mu + p.market_sigma * rng.standard_normal(p.n_days)
    eps = p.idio_sigma * rng.standard_normal(p.n_days)
    r = p.beta * m + eps
    for d, s in zip(p.event_days, p.event_surprise, strict=True):
        r[d] += s
    # constant drift over the window so its realised simple return is the planted momentum
    w = slice(p.n_days - p.momentum_window, p.n_days)
    target = math.log1p(p.momentum)
    r[w] += (target - float(r[w].sum())) / p.momentum_window
    log_p = math.log(p.price_now) - np.concatenate(([0.0], np.cumsum(r[::-1])))[::-1]
    prices = np.exp(log_p)  # n_days + 1 prices, the last equal to price_now
    market = np.exp(np.concatenate(([0.0], np.cumsum(m))))
    volume = p.volume_mean * np.exp(0.5 * rng.standard_normal(p.n_days))
    truth = {
        "momentum": float(prices[-1] / prices[-1 - p.momentum_window] - 1.0),
        "beta": p.beta,
        "idio_vol_daily": p.idio_sigma,
        "events": [
            {"day": int(d), "surprise": float(s), "revision": float(v)}
            for d, s, v in zip(p.event_days, p.event_surprise, p.revision, strict=True)
        ],
    }
    return MarketPanel(prices, market, r, m, volume, truth)


def realised_beta(panel: MarketPanel, exclude: Sequence[int] = ()) -> float:
    """OLS beta of the stock's daily log returns on the market's, event days excluded."""
    keep = np.ones(len(panel.returns), dtype=bool)
    keep[list(exclude)] = False
    x, y = panel.market_returns[keep], panel.returns[keep]
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())
