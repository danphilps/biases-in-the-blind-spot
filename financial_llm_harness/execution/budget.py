"""Run control and cost: pricing, ``plan`` and the budget scheduler (spec §30, §31A L4).

Before submitting a block the runner checks spent + reserved + the block's upper
bound ≤ cap for every cap; an upper bound uses ``max_output_tokens`` and the
input estimate (chars / 3 without a provider tokeniser).
"""

from __future__ import annotations

import importlib.resources
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from financial_llm_harness.core import yaml_io
from financial_llm_harness.errors import HarnessValidationError

PRICE_STALE_DAYS = 90
CAPS = ("max_calls", "max_input_tokens", "max_output_tokens", "max_cost_usd")


def estimate_tokens(text: str) -> int:
    """chars / 3 upper-bound estimate (§20.8, §30)."""
    return int(math.ceil(len(text) / 3))


def load_pricing() -> dict[str, Any]:
    path = importlib.resources.files("financial_llm_harness") / "pricing.yaml"
    data = yaml_io.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "prices" not in data:
        raise HarnessValidationError(
            "pricing.yaml needs pricing_table_version and prices"
        )
    return data


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    batch_discount: float | None
    checked_on: str | None

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
        batch: bool = False,
    ) -> float:
        c = (
            input_tokens * self.input_per_mtok
            + (output_tokens + reasoning_tokens) * self.output_per_mtok
        ) / 1e6
        if batch and self.batch_discount:
            c *= 1 - float(self.batch_discount)
        return c


def price_for(pricing: Mapping[str, Any], provider: str, model: str) -> Price | None:
    entry = (pricing.get("prices") or {}).get(f"{provider}/{model}")
    if entry is None:
        return None
    return Price(
        float(entry["input_per_mtok_usd"]),
        float(entry["output_per_mtok_usd"]),
        entry.get("batch_discount"),
        entry.get("checked_on"),
    )


def stale_prices(pricing: Mapping[str, Any], keys: list[str], today: date) -> list[str]:
    out = []
    for k in keys:
        entry = (pricing.get("prices") or {}).get(k) or {}
        checked = entry.get("checked_on")
        if (
            checked
            and (today - date.fromisoformat(str(checked))).days > PRICE_STALE_DAYS
        ):
            out.append(k)
    return out


@dataclass
class Spend:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: Spend) -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd

    def sub(self, other: Spend) -> None:
        self.calls -= other.calls
        self.input_tokens -= other.input_tokens
        self.output_tokens -= other.output_tokens
        self.cost_usd -= other.cost_usd


@dataclass
class BudgetTracker:
    """Caps from the manifest budget, optionally replaced by a resume override (§30)."""

    caps: dict[str, float | None]
    spent: Spend = field(default_factory=Spend)
    reserved: Spend = field(default_factory=Spend)
    breached: bool = False

    @staticmethod
    def from_manifest(
        budget: Mapping[str, Any], override: Mapping[str, Any] | None = None
    ) -> BudgetTracker:
        caps: dict[str, float | None] = {k: budget.get(k) for k in CAPS}
        for k, v in (override or {}).items():
            if v is not None:
                caps[k] = v
        return BudgetTracker(caps)

    def fits(self, bound: Spend) -> bool:
        total = Spend()
        total.add(self.spent)
        total.add(self.reserved)
        total.add(bound)
        checks = (
            ("max_calls", total.calls),
            ("max_input_tokens", total.input_tokens),
            ("max_output_tokens", total.output_tokens),
            ("max_cost_usd", total.cost_usd),
        )
        for key, value in checks:
            cap = self.caps.get(key)
            if cap is not None and value > float(cap) + 1e-12:
                return False
        return True

    def reserve(self, bound: Spend) -> bool:
        if self.breached or not self.fits(bound):
            self.breached = True
            return False
        self.reserved.add(bound)
        return True

    def release(self, bound: Spend) -> None:
        self.reserved.sub(bound)

    def record(self, actual: Spend) -> None:
        self.spent.add(actual)


@dataclass
class PlanRow:
    model_key: str
    n_clusters: int
    trials: int
    calls: int
    input_tokens_est: int
    output_tokens_max: int
    cost_usd_low: float
    cost_usd_high: float


def plan_rows(
    manifest: Mapping[str, Any],
    renderings: list[dict[str, Any]],
    pricing: Mapping[str, Any],
) -> list[PlanRow]:
    """Per-model counts, token bounds and cost range from the dated pricing table (§30)."""
    R = int(manifest["sampling_spec"]["R"])
    max_out = int(manifest["response_spec"]["max_output_tokens"])
    n_trials = len(renderings) * R
    in_tokens = (
        sum(
            estimate_tokens(r["system"]) + estimate_tokens(r["user"])
            for r in renderings
        )
        * R
    )
    n_clusters = int(
        (manifest.get("_corpus_report") or {})
        .get("near_duplicates", {})
        .get("n_clusters", 0)
    )
    rows = []
    for m in manifest["sampling_spec"]["models"]:
        price = price_for(pricing, m["provider"], m["model"])
        if price is None:
            raise HarnessValidationError(
                f"pricing.yaml has no entry for {m['provider']}/{m['model']} (§30)"
            )
        low = price.cost(in_tokens, 0, 0, bool(m.get("batch")))
        high = price.cost(in_tokens, n_trials * max_out, 0, bool(m.get("batch")))
        rows.append(
            PlanRow(
                m["model_key"],
                n_clusters,
                n_trials,
                n_trials,
                in_tokens,
                n_trials * max_out,
                low,
                high,
            )
        )
    return rows
