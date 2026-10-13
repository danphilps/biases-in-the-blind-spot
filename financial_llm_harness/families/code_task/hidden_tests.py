"""§T6.1 hidden-test engines (spec §13, §T6, M9a).

Each engine is a pure check over a candidate function and fixture data, and
returns the list of its violations (empty when the check passes). A task's
hidden test calls the engine on the solution and asserts the list is empty.
The engines carry no task content: tolerances, seeds and data come from the
task's fixture (D-23), and each engine is exercised here by a known-good and a
known-bad reference (§36). Inside the sandbox a fixture's ``tests_hidden/``
holds a byte-identical copy of this module (``_engines.py``), checked by test.

Engines: HT.ALIGN, HT.PERM, HT.MONO, HT.REF, HT.PSD, HT.ANTI, HT.TRANS,
HT.SURV, HT.PURGE, HT.ANNUAL, HT.COMPOUND, HT.NAN. HT.LEAK, HT.NONTRIV,
HT.SHAPE and HT.FINITE are K.M.1's (S5) and live in its fixture.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Violations = list[str]


def _close(a: Any, b: Any, rtol: float, atol: float) -> bool:
    return bool(
        np.allclose(
            np.asarray(a, dtype=float),
            np.asarray(b, dtype=float),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    )


def ht_ref(candidate: Any, reference: Any, rtol: float, atol: float) -> Violations:
    """HT.REF: the output equals the reference implementation's within tolerance."""
    a, b = np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape:
        return [f"shape {a.shape} != reference {b.shape}"]
    if not _close(a, b, rtol, atol):
        err = float(np.nanmax(np.abs(a - b)))
        return [f"max abs deviation {err:.3g} from the reference"]
    return []


def ht_psd(matrix: Any, max_condition: float, tol: float = 1e-10) -> Violations:
    """HT.PSD: symmetric, positive semidefinite, condition number bounded."""
    m = np.asarray(matrix, dtype=float)
    out: Violations = []
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return [f"not square: {m.shape}"]
    if not np.allclose(m, m.T, atol=tol * max(1.0, float(np.abs(m).max()))):
        out.append("not symmetric")
    eig = np.linalg.eigvalsh(0.5 * (m + m.T))
    if eig.min() < -tol * max(1.0, float(eig.max())):
        out.append(f"negative eigenvalue {eig.min():.3g}")
    cond = float(eig.max() / eig.min()) if eig.min() > 0 else float("inf")
    if cond > max_condition:
        out.append(f"condition number {cond:.3g} > {max_condition:.3g}")
    return out


def ht_perm(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    perm: np.ndarray,
    axis: int,
    out_axes: Sequence[int],
    rtol: float,
    atol: float,
) -> Violations:
    """HT.PERM: permuting the assets (``axis`` of the input) permutes the output's ``out_axes`` alike."""
    base = np.asarray(fn(x), dtype=float)
    moved = np.asarray(fn(np.take(x, perm, axis=axis)), dtype=float)
    expected = base
    for ax in out_axes:
        expected = np.take(expected, perm, axis=ax)
    return (
        []
        if _close(moved, expected, rtol, atol)
        else ["output is not permuted with the assets"]
    )


def ht_mono(
    ic_fn: Callable[[np.ndarray, np.ndarray], Any],
    signal: np.ndarray,
    returns: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    rtol: float,
    atol: float,
) -> Violations:
    """HT.MONO: a rank IC is unchanged under a strictly increasing transform of the signal."""
    a = ic_fn(signal, returns)
    b = ic_fn(transform(signal), returns)
    return (
        []
        if _close(a, b, rtol, atol)
        else [
            "IC changes under a monotone transform (Pearson where rank was specified?)"
        ]
    )


def ht_align(
    fn: Callable[[np.ndarray, np.ndarray], Any],
    signal: np.ndarray,
    prices: np.ndarray,
    horizon: int,
    rtol: float,
    atol: float,
) -> Violations:
    """HT.ALIGN: a signal known at t is paired only with the return over (t, t+h].

    ``fn(signal, prices)`` gives one output per date. Replacing every price
    outside [t, t + h] with noise must leave output[t] unchanged: a same-period
    pairing (prices before t) or an overlapping window (after t + h) changes it.
    """
    base = np.asarray(fn(signal, prices), dtype=float)
    rng = np.random.default_rng(0)
    out: Violations = []
    n_t = prices.shape[0]
    for t in range(0, n_t - horizon - 1, max(1, (n_t - horizon) // 5)):
        noisy = prices.copy()
        mask = np.ones(n_t, dtype=bool)
        mask[t : t + horizon + 1] = False
        noisy[mask] = noisy[mask] * np.exp(0.1 * rng.standard_normal(noisy[mask].shape))
        again = np.asarray(fn(signal, noisy), dtype=float)
        if not _close(again[t], base[t], rtol, atol):
            out.append(f"date {t}: the output uses prices outside (t, t+{horizon}]")
    return out


def ht_anti(
    score: Callable[[Any, Any], float], pairs: Sequence[tuple[Any, Any]], atol: float
) -> Violations:
    """HT.ANTI: s(a, b) = −s(b, a)."""
    bad = [i for i, (a, b) in enumerate(pairs) if abs(score(a, b) + score(b, a)) > atol]
    return [f"{len(bad)} of {len(pairs)} pairs are not antisymmetric"] if bad else []


def ht_trans(
    score: Callable[[Any, Any], float], triads: Sequence[tuple[Any, Any, Any]]
) -> Violations:
    """HT.TRANS: no preference cycles a ≻ b ≻ c ≻ a on the triads."""
    cycles = 0
    for a, b, c in triads:
        ab, bc, ca = score(a, b) > 0, score(b, c) > 0, score(c, a) > 0
        if ab == bc == ca:
            cycles += 1
    return (
        [f"{cycles} of {len(triads)} triads form a preference cycle"] if cycles else []
    )


def ht_surv(
    returns_out: np.ndarray,
    delisted_at: np.ndarray,
    delisting_returns: np.ndarray,
    atol: float,
) -> Violations:
    """HT.SURV: delisted assets are kept, with their delisting return on the delisting date."""
    out: Violations = []
    r = np.asarray(returns_out, dtype=float)
    for i, d in enumerate(delisted_at):
        if d < 0:
            continue
        if r.shape[1] <= i:
            out.append(f"asset {i} dropped")
            continue
        if not np.isfinite(r[d, i]) or abs(r[d, i] - delisting_returns[i]) > atol:
            out.append(
                f"asset {i}: delisting return {r[d, i]} != {delisting_returns[i]}"
            )
    return out


def ht_purge(
    train_label_end: Sequence[Sequence[int]],
    test_start: Sequence[int],
    test_end: Sequence[int],
    train_start: Sequence[Sequence[int]],
    embargo: int,
) -> Violations:
    """HT.PURGE: no training label window [start, end] overlaps a test window or its embargo."""
    out: Violations = []
    for k, (ts, te) in enumerate(zip(test_start, test_end, strict=True)):
        for s, e in zip(train_start[k], train_label_end[k], strict=True):
            if s <= te + embargo and e >= ts:
                out.append(
                    f"fold {k}: a training label window [{s}, {e}] touches the test window [{ts}, {te}] + embargo {embargo}"
                )
                break
    return out


def ht_annual(
    value: float, per_period: float, periods_per_year: float, kind: str, rtol: float
) -> Violations:
    """HT.ANNUAL: annualisation matches the data frequency (√252 vs √12)."""
    if kind == "volatility":
        want = per_period * np.sqrt(periods_per_year)
    elif kind == "sharpe":
        want = per_period * np.sqrt(periods_per_year)
    elif kind == "mean":
        want = per_period * periods_per_year
    else:
        return [f"unknown annualisation kind {kind!r}"]
    return (
        []
        if abs(value - want) <= rtol * abs(want)
        else [f"annualised {value:.6g}, expected {want:.6g}"]
    )


def ht_compound(
    aggregated: Any,
    period_returns: np.ndarray,
    log_returns: bool,
    rtol: float,
    atol: float,
) -> Violations:
    """HT.COMPOUND: simple returns compound as a product, log returns as a sum."""
    r = np.asarray(period_returns, dtype=float)
    want = r.sum(axis=0) if log_returns else np.prod(1.0 + r, axis=0) - 1.0
    return (
        []
        if _close(aggregated, want, rtol, atol)
        else ["multi-period aggregation is wrong"]
    )


def ht_nan(
    output: np.ndarray, missing: np.ndarray, delisted_at: np.ndarray
) -> Violations:
    """HT.NAN: missing inputs stay missing (no forward fill from the future or across a delisting)."""
    out: Violations = []
    o = np.asarray(output, dtype=float)
    filled = missing & np.isfinite(o)
    if filled.any():
        out.append(f"{int(filled.sum())} missing input values were filled")
    for i, d in enumerate(delisted_at):
        if d >= 0 and np.isfinite(o[d + 1 :, i]).any():
            out.append(f"asset {i} has values after its delisting")
    return out
