"""Sample size and planning variance (spec §22, §23.6).

Normal-approximation formulas; every N is rounded up. With no pilot the
planning variance is the bound for a variable confined to an interval of width w:
w²/4, so paired binary d_c ∈ [−1, 1] gives 1.0 and arm-level binary 0.25.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scipy import stats

CONTROL_CONTRASTS = ("A0_vs_A", "PL_vs_A", "PCB_vs_PCA")


def z(p: float) -> float:
    return float(stats.norm.ppf(p))


def _ceil(x: float) -> int:
    # guard against 784.9999999 → 785 style float noise
    return int(math.ceil(x - 1e-9))


def superiority_n(sigma2: float, delta: float, alpha: float, power: float) -> int:
    return _ceil((z(1 - alpha / 2) + z(power)) ** 2 * sigma2 / delta**2)


def equivalence_n(sigma2: float, margin: float, alpha: float, power: float) -> int:
    beta = 1 - power
    return _ceil((z(1 - alpha) + z(1 - beta / 2)) ** 2 * sigma2 / margin**2)


def level_n(sigma2_y: float, delta: float, alpha: float, power: float) -> int:
    return _ceil((z(1 - alpha) + z(power)) ** 2 * sigma2_y / delta**2)


def power_superiority(n: int, sigma2: float, delta: float, alpha: float) -> float:
    if n <= 0:
        return 0.0
    return float(stats.norm.cdf(math.sqrt(n * delta**2 / sigma2) - z(1 - alpha / 2)))


def power_equivalence(n: int, sigma2: float, margin: float, alpha: float) -> float:
    if n <= 0:
        return 0.0
    return float(
        max(
            0.0,
            2 * stats.norm.cdf(math.sqrt(n * margin**2 / sigma2) - z(1 - alpha)) - 1,
        )
    )


def power_level(n: int, sigma2_y: float, delta: float, alpha: float) -> float:
    if n <= 0:
        return 0.0
    return float(stats.norm.cdf(math.sqrt(n * delta**2 / sigma2_y) - z(1 - alpha)))


def feature_range(feature: Mapping[str, Any]) -> tuple[float, float]:
    """[L, U] of a feature's numeric scale (binary [0, 1]; ordinal its scores)."""
    ftype = feature.get("type")
    if ftype == "binary":
        return 0.0, 1.0
    if ftype == "scalar":
        lo, hi = feature["range"]
        return float(lo), float(hi)
    if ftype in ("ordinal", "categorical"):
        scores = feature.get("scores")
        if scores:
            vals = (
                [float(v) for v in scores.values()]
                if isinstance(scores, Mapping)
                else [float(v) for v in scores]
            )
            return min(vals), max(vals)
        n = len(feature.get("allowed_values") or [])
        return 0.0, float(max(n - 1, 1))
    return 0.0, 1.0


def no_pilot_variance(feature: Mapping[str, Any], arm_level: bool) -> float:
    lo, hi = feature_range(feature)
    w = hi - lo
    return w * w / 4 if arm_level else w * w


@dataclass(frozen=True)
class PlannedVariance:
    variance: float
    source: str  # planning_ref | planning_sd | no_pilot


def planning_variance(
    statistical_spec: Mapping[str, Any],
    feature: Mapping[str, Any],
    contrast: str,
    arm_level: bool,
    model_key: str | None = None,
) -> PlannedVariance:
    """The §23.6 planning variance for (feature, contrast): planning_ref, planning_sd, or the bound."""
    refs = [
        r
        for r in statistical_spec.get("planning_ref") or []
        if r.get("feature_id") == feature.get("feature_id")
        and r.get("contrast") == contrast
        and (model_key is None or r.get("model_key") == model_key)
    ]
    if refs:
        return PlannedVariance(max(float(r["variance"]) for r in refs), "planning_ref")
    sd = statistical_spec.get("planning_sd")
    if (
        sd is not None
        and statistical_spec.get("planning_sd_source")
        and feature.get("type") in ("ordinal", "scalar")
    ):
        return PlannedVariance(float(sd) ** 2, "planning_sd")
    return PlannedVariance(no_pilot_variance(feature, arm_level), "no_pilot")


@dataclass(frozen=True)
class PrimaryRequirement:
    n_required: int
    superiority: int | None
    equivalence: int | None
    level: int | None
    variance: PlannedVariance
    contrast: str


def primary_requirement(
    oracle_type: str,
    hypothesis: Mapping[str, Any],
    statistical_spec: Mapping[str, Any],
    feature: Mapping[str, Any],
    tol: float | None = None,
    model_key: str | None = None,
) -> PrimaryRequirement:
    """Binding N for the primary (§23.6)."""
    alpha = float(hypothesis["alpha"])
    power = float(hypothesis["power_target"])
    delta = float(hypothesis["delta"])
    estimand = hypothesis["estimand"]
    if estimand == "arm_level":
        contrast = f"level_{hypothesis['level_arm']}"
        pv = planning_variance(statistical_spec, feature, contrast, True, model_key)
        n = level_n(pv.variance, delta, alpha, power)
        return PrimaryRequirement(n, None, None, n, pv, contrast)
    contrast = "B_vs_A"
    pv = planning_variance(statistical_spec, feature, contrast, False, model_key)
    margin = float(hypothesis.get("equivalence_margin") or delta)
    if oracle_type == "MAGNITUDE":
        eq = equivalence_n(
            pv.variance, float(tol if tol is not None else margin), alpha, power
        )
        return PrimaryRequirement(eq, None, eq, None, pv, contrast)
    sup = superiority_n(pv.variance, delta, alpha, power)
    eq = equivalence_n(pv.variance, margin, alpha, power)
    return PrimaryRequirement(max(sup, eq), sup, eq, None, pv, contrast)


@dataclass(frozen=True)
class ControlRequirement:
    contrast: str
    n_required: int
    variance: PlannedVariance


def control_requirements(
    arms: Sequence[str],
    controls: Mapping[str, Any],
    hypothesis: Mapping[str, Any],
    statistical_spec: Mapping[str, Any],
    feature: Mapping[str, Any],
    model_key: str | None = None,
) -> list[ControlRequirement]:
    """n_A0, n_PL and n_PC for the control arms present (§22)."""
    alpha = float(controls.get("control_alpha", hypothesis["alpha"]))
    power = float(hypothesis["power_target"])
    out: list[ControlRequirement] = []
    if "A0" in arms:
        pv = planning_variance(statistical_spec, feature, "A0_vs_A", False, model_key)
        out.append(
            ControlRequirement(
                "A0_vs_A",
                equivalence_n(
                    pv.variance, float(controls["null_margin"]), alpha, power
                ),
                pv,
            )
        )
    if "PL" in arms:
        pv = planning_variance(statistical_spec, feature, "PL_vs_A", False, model_key)
        out.append(
            ControlRequirement(
                "PL_vs_A",
                equivalence_n(
                    pv.variance, float(controls["placebo_margin"]), alpha, power
                ),
                pv,
            )
        )
    if "PCB" in arms:
        pv = planning_variance(
            statistical_spec, feature, "PCB_vs_PCA", False, model_key
        )
        pos = controls.get("positive") or {}
        effect = pos.get("planning_effect")
        effect = float(effect) if effect is not None else 2 * float(hypothesis["delta"])
        out.append(
            ControlRequirement(
                "PCB_vs_PCA", superiority_n(pv.variance, effect, alpha, power), pv
            )
        )
    return out


def computed_subset_n(requirements: Sequence[ControlRequirement]) -> int:
    return max([30, *(r.n_required for r in requirements)])
