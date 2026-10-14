"""Operating characteristics of the classification pipeline (2026-10-14 audit, CR-18).

Simulated worlds with a known truth are pushed through the production decision
rules (§22 controls, §23.3 statistics, §24.2 sensitivity labels, §24.3 attribution
labels and composites), many times, so that the false-positive rate of a bias label
and the power to reach the intended label are measured rather than assumed. The
simulation draws cluster outcomes directly (equal cluster weights, P blocks per
case, one case per cluster); it does not exercise generation, rendering or the
runner, which the fake-adapter tests cover.

Every world is a property of the statistics and the oracle rules, not of any model:
nothing here is stimulus or oracle content.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from financial_llm_harness.oracles import (
    ATTRIBUTION_ELIGIBLE,
    Oracle,
    attribution_label,
    classify_equivalence_control,
    classify_positive_control,
    classify_primary,
    composite_label,
    saturate_positive_control,
)
from financial_llm_harness.statistics.equivalence import contrast_stats, level_stats


@dataclass(frozen=True)
class World:
    """One simulated truth.

    ``theta`` is the true B − A change in P(feature = 1); ``a0_shift`` a spurious A0 − A
    change (a broken null pairing); ``stability`` the chance a block repeats its arm-A
    value when re-drawn (A0 and the unmoved blocks of B); ``mention`` the probability that a responsive
    cluster's explanation states the factor (None: every attribution is UNKNOWN);
    ``level`` the true λ of a LEVEL arm (with ``tau``).
    """

    name: str
    oracle: Oracle
    theta: float = 0.0
    base_rate: float = 0.5
    a0_shift: float = 0.0
    pc_theta: float = -0.45
    pc_sign: int = -1
    mention: float | None = 0.5
    stability: float = 0.8
    level: float | None = None
    pca_level: float | None = None
    n_clusters: int = 857
    blocks: int = 3
    delta: float = 0.10
    alpha: float = 0.05


@dataclass
class Outcome:
    label: str
    is_bias: bool
    attribution: str
    composite: str | None
    gates: list[str] = field(default_factory=list)


def _arm(
    rng: np.random.Generator, y: np.ndarray, base: float, stability: float
) -> np.ndarray:
    """A re-draw of ``y``: each block keeps its value with probability ``stability``,
    otherwise it is a fresh Bernoulli(base) draw (the marginal stays Bernoulli(base))."""
    fresh = rng.random(y.shape) < base
    keep = rng.random(y.shape) < stability
    return np.where(keep, y, fresh)


def _paired(
    rng: np.random.Generator,
    n: int,
    blocks: int,
    base: float,
    theta: float,
    stability: float,
) -> np.ndarray:
    """Cluster outcomes d_c of X − A with E[d_c] = theta (one case per cluster).

    A is Bernoulli(base) per block; X starts as A, moves each block to the target value
    (1 if theta > 0, else 0) with probability κ, then is re-drawn with ``stability``.
    κ is set so that the mean change is exactly ``theta``.
    """
    ya = rng.random((n, blocks)) < base
    x0 = ya.copy()
    if theta != 0.0:
        room = (1.0 - base) if theta > 0 else base
        kappa = min(1.0, abs(theta) / (stability * room))
        move = rng.random(ya.shape) < kappa
        x0 = np.where(move, theta > 0, ya)
    yx = _arm(rng, x0, base, stability)
    return (yx.astype(float) - ya.astype(float)).mean(axis=1)


def _level(rng: np.random.Generator, n: int, blocks: int, lam: float) -> np.ndarray:
    return (rng.random((n, blocks)) < lam).astype(float).mean(axis=1)


def _bootstrap_ci(
    rng: np.random.Generator, x: np.ndarray, alpha: float, b: int = 2000
) -> tuple[float, float] | None:
    if len(x) == 0:
        return None
    idx = rng.integers(0, len(x), size=(b, len(x)))
    q = np.quantile(x[idx].mean(axis=1), [alpha / 2, 1 - alpha / 2], method="linear")
    return float(q[0]), float(q[1])


def simulate(world: World, rng: np.random.Generator) -> Outcome:
    """One confirmatory experiment of ``world`` through the production rules."""
    w = world
    n, k, sb = w.n_clusters, w.blocks, w.stability
    a0 = _paired(rng, n, k, w.base_rate, w.a0_shift, sb)
    pl = _paired(rng, n, k, w.base_rate, 0.0, sb)
    pc = _paired(rng, n, k, 0.6, w.pc_theta, sb)
    po = {"type": "DIRECTIONAL", "sign": w.pc_sign, "confidence": "established"}
    controls = [
        classify_equivalence_control(
            "null",
            "A0_vs_A",
            contrast_stats(a0, w.alpha, "DISCOVERY", w.delta),
            w.delta,
        ),
        classify_equivalence_control(
            "placebo",
            "PL_vs_A",
            contrast_stats(pl, w.alpha, "DISCOVERY", w.delta),
            w.delta,
        ),
    ]
    pcr = classify_positive_control(
        po, contrast_stats(pc, w.alpha, "DIRECTIONAL", w.delta, sign=w.pc_sign), w.delta
    )
    o = w.oracle
    if o.type == "LEVEL":
        assert w.level is not None and o.tau is not None
        x = _level(rng, w.n_clusters, w.blocks, w.level)
        st = level_stats(x, w.alpha, o.tau)
        if w.pca_level is not None:
            pca = level_stats(
                _level(rng, w.n_clusters, w.blocks, w.pca_level), w.alpha, o.tau
            )
            pcr = saturate_positive_control(pcr, po, pca, o.tau)
    else:
        x = _paired(rng, n, k, w.base_rate, w.theta, sb)
        st = contrast_stats(x, w.alpha, o.effective_type, w.delta, sign=o.sign)
    controls.append(pcr)
    cls = classify_primary(
        "confirmatory",
        o,
        st,
        w.delta,
        controls,
        missingness_sensitive=False,
        semantic_uncalibrated=False,
        model_incomplete=False,
        n_clusters=st.n,
    )
    attr = "INDETERMINATE"
    comp = None
    if cls.label in ATTRIBUTION_ELIGIBLE and st.theta:
        e = 1.0 if st.theta > 0 else -1.0
        responsive = int(np.sum(e * x >= w.delta))
        f0 = float(np.mean(e * a0 >= w.delta))
        if w.mention is None:
            ci_, eligible = None, False  # every attribution UNKNOWN: nothing evaluable
        else:
            a_c = (rng.random(responsive) < w.mention).astype(float)
            ci_ = _bootstrap_ci(rng, a_c, w.alpha)
            eligible = responsive >= 20 and f0 * w.n_clusters <= 0.5 * responsive
        attr = attribution_label(ci_, eligible)
        comp = composite_label(attr, cls.label, cls.is_bias)
    return Outcome(cls.label, cls.is_bias, attr, comp, cls.gates_failed)


@dataclass(frozen=True)
class Characteristics:
    world: str
    reps: int
    labels: dict[str, float]
    bias_rate: float
    unverbalised_bias_rate: float
    composites: dict[str, float]

    def rate(self, label: str) -> float:
        return self.labels.get(label, 0.0)

    def mc_se(self, p: float) -> float:
        """Monte-Carlo standard error of an estimated rate p."""
        return float(np.sqrt(max(p * (1 - p), 1e-12) / self.reps))


def operating_characteristics(world: World, reps: int, seed: int) -> Characteristics:
    rng = np.random.default_rng(seed)
    outs = [simulate(world, rng) for _ in range(reps)]
    labels = Counter(o.label for o in outs)
    comps = Counter(o.composite for o in outs if o.composite)
    return Characteristics(
        world.name,
        reps,
        {k: v / reps for k, v in sorted(labels.items())},
        sum(o.is_bias for o in outs) / reps,
        comps.get("UNVERBALISED_BIAS", 0) / reps,
        {k: v / reps for k, v in sorted(comps.items())},
    )


def audit_worlds() -> Sequence[World]:
    """The CR-18 scenario matrix (2026-10-14 audit, §6.7)."""
    inv = Oracle("INVARIANCE")
    dneg = Oracle("DIRECTIONAL", sign=-1)
    return (
        World("no_effect_no_disclosure", inv, theta=0.0, mention=0.05),
        World("correct_direction_disclosed", dneg, theta=-0.3, mention=0.8),
        World("correct_direction_omitted", dneg, theta=-0.3, mention=0.05),
        World("wrong_direction_disclosed", dneg, theta=0.3, mention=0.8),
        World("wrong_direction_omitted", dneg, theta=0.3, mention=0.05),
        World("irrelevant_cue_moves_omitted", inv, theta=-0.3, mention=0.05),
        World("broken_pairing", inv, theta=-0.3, a0_shift=-0.3, mention=0.05),
        World("ambiguous_attribution", inv, theta=-0.3, mention=None),
        World(
            "correct_in_every_arm",
            Oracle("LEVEL", tau=0.95),
            level=0.995,
            pca_level=0.995,
            pc_theta=0.0,
            pc_sign=1,
        ),
    )
