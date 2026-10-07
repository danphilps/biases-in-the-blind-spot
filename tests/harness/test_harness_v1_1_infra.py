"""V1.1 infrastructure (spec §38 M9a, §36): the §T6.1 hidden-test engines, each with a
known-good and a known-bad reference solution; the bad one MUST fail and the good one
MUST pass."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from sklearn.covariance import LedoitWolf

from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.families.code_task import hidden_tests as HT

RNG = np.random.default_rng(20261008)


# --- reference solutions (code, not task content) ------------------------------------


def lw_good(x: np.ndarray) -> np.ndarray:
    """Ledoit–Wolf (2004) linear shrinkage towards the scaled identity."""
    t, n = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / t
    mu = np.trace(s) / n
    d2 = float(((s - mu * np.eye(n)) ** 2).sum())
    b2 = sum(float(((np.outer(r, r) - s) ** 2).sum()) for r in xc) / t**2
    b2 = min(b2, d2)
    return (b2 / d2) * mu * np.eye(n) + (1 - b2 / d2) * s


def sample_cov(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=0)
    return xc.T @ xc / x.shape[0]


def spearman_ic(sig: np.ndarray, ret: np.ndarray) -> float:
    return float(np.corrcoef(stats.rankdata(sig), stats.rankdata(ret))[0, 1])


def pearson_ic(sig: np.ndarray, ret: np.ndarray) -> float:
    return float(np.corrcoef(sig, ret)[0, 1])


H = 5


def ic_forward(signal: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Good: signal at t against the return over (t, t + h]."""
    out = np.full(signal.shape[0], np.nan)
    for t in range(signal.shape[0] - H):
        out[t] = spearman_ic(signal[t], prices[t + H] / prices[t] - 1)
    return out


def ic_same_period(signal: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Bad: signal at t against the return over (t − h, t]."""
    out = np.full(signal.shape[0], np.nan)
    for t in range(H, signal.shape[0] - H):
        out[t] = spearman_ic(signal[t], prices[t] / prices[t - H] - 1)
    return out


def test_ht_ref_and_psd() -> None:
    x = RNG.standard_normal((40, 60))  # T < N: the sample covariance is singular
    ref = LedoitWolf(assume_centered=False).fit(x).covariance_
    assert HT.ht_ref(lw_good(x), ref, rtol=1e-8, atol=1e-10) == []
    assert HT.ht_ref(sample_cov(x), ref, rtol=1e-8, atol=1e-10)
    assert HT.ht_psd(lw_good(x), max_condition=1e6) == []
    assert HT.ht_psd(sample_cov(x), max_condition=1e6)


def test_ht_perm() -> None:
    x = RNG.standard_normal((50, 8))
    perm = RNG.permutation(8)

    def zscore(a: np.ndarray) -> np.ndarray:
        return (a - a.mean(axis=0)) / a.std(axis=0)

    def order_dependent(a: np.ndarray) -> np.ndarray:
        return np.cumsum(a, axis=1)

    assert (
        HT.ht_perm(zscore, x, perm, axis=1, out_axes=[1], rtol=1e-12, atol=1e-12) == []
    )
    assert HT.ht_perm(
        order_dependent, x, perm, axis=1, out_axes=[1], rtol=1e-12, atol=1e-12
    )


def test_ht_mono() -> None:
    sig, ret = RNG.standard_normal(300), RNG.standard_normal(300)
    assert HT.ht_mono(spearman_ic, sig, ret, np.exp, rtol=1e-12, atol=1e-12) == []
    assert HT.ht_mono(pearson_ic, sig, ret, np.exp, rtol=1e-12, atol=1e-12)


def test_ht_align() -> None:
    t, n = 60, 30
    prices = 100 * np.exp(np.cumsum(0.01 * RNG.standard_normal((t, n)), axis=0))
    signal = RNG.standard_normal((t, n))
    assert HT.ht_align(ic_forward, signal, prices, H, rtol=1e-12, atol=1e-12) == []
    assert HT.ht_align(ic_same_period, signal, prices, H, rtol=1e-12, atol=1e-12)


def test_ht_anti_and_trans() -> None:
    items = list(RNG.standard_normal(12))
    pairs = [(a, b) for a in items for b in items if a != b]
    triads = [
        (items[i], items[j], items[k])
        for i in range(12)
        for j in range(i + 1, 12)
        for k in range(j + 1, 12)
    ]

    def good(a: float, b: float) -> float:
        return a - b

    def asymmetric(a: float, b: float) -> float:
        return a - 0.5 * b

    labels = {v: i % 3 for i, v in enumerate(items)}

    def cyclic(a: float, b: float) -> float:  # rock-paper-scissors on the label classes
        la, lb = labels[a], labels[b]
        return 0.0 if la == lb else (1.0 if (la - lb) % 3 == 1 else -1.0)

    assert HT.ht_anti(good, pairs, atol=1e-12) == []
    assert HT.ht_anti(asymmetric, pairs, atol=1e-12)
    assert HT.ht_trans(good, triads) == []
    by_class = [
        (a, b, c)
        for a in items
        for b in items
        for c in items
        if (labels[a], labels[b], labels[c]) == (1, 0, 2)
    ]
    assert HT.ht_trans(cyclic, by_class)


def test_ht_surv_and_nan() -> None:
    t, n = 20, 5
    r = 0.01 * RNG.standard_normal((t, n))
    delisted = np.array([-1, 7, -1, 12, -1])
    dl_ret = np.array([0.0, -0.3, 0.0, -0.5, 0.0])
    good = r.copy()
    for i, d in enumerate(delisted):
        if d >= 0:
            good[d, i] = dl_ret[i]
            good[d + 1 :, i] = np.nan
    bad = good.copy()
    bad[:, 1] = np.nan  # the delisted asset is dropped
    assert HT.ht_surv(good, delisted, dl_ret, atol=1e-12) == []
    assert HT.ht_surv(bad, delisted, dl_ret, atol=1e-12)
    missing = np.zeros((t, n), dtype=bool)
    missing[3, 2] = True
    raw = good.copy()
    raw[3, 2] = np.nan
    import pandas as pd

    ffilled = pd.DataFrame(raw).ffill().to_numpy()
    assert HT.ht_nan(raw, missing, delisted) == []
    assert HT.ht_nan(ffilled, missing, delisted)


def _folds(
    n: int, k: int, horizon: int, embargo: int, purge: bool
) -> tuple[list[list[int]], list[list[int]], list[int], list[int]]:
    size = n // k
    starts, ends, tr_s, tr_e = [], [], [], []
    for f in range(k):
        ts, te = f * size, (f + 1) * size - 1
        s_list, e_list = [], []
        for i in range(n):
            if ts <= i <= te:
                continue
            label_end = i + horizon
            if purge and i <= te + embargo and label_end >= ts:
                continue
            s_list.append(i)
            e_list.append(label_end)
        starts.append(ts)
        ends.append(te)
        tr_s.append(s_list)
        tr_e.append(e_list)
    return tr_s, tr_e, starts, ends


def test_ht_purge() -> None:
    tr_s, tr_e, ts, te = _folds(100, 5, horizon=4, embargo=2, purge=True)
    assert HT.ht_purge(tr_e, ts, te, tr_s, embargo=2) == []
    tr_s, tr_e, ts, te = _folds(100, 5, horizon=4, embargo=2, purge=False)
    assert HT.ht_purge(tr_e, ts, te, tr_s, embargo=2)


def test_ht_annual_and_compound() -> None:
    monthly_vol = 0.04
    assert (
        HT.ht_annual(
            monthly_vol * np.sqrt(12), monthly_vol, 12, "volatility", rtol=1e-12
        )
        == []
    )
    assert HT.ht_annual(
        monthly_vol * np.sqrt(252), monthly_vol, 12, "volatility", rtol=1e-12
    )
    r = 0.02 * RNG.standard_normal((12, 4))
    assert (
        HT.ht_compound(
            np.prod(1 + r, axis=0) - 1, r, log_returns=False, rtol=1e-12, atol=1e-12
        )
        == []
    )
    assert HT.ht_compound(r.sum(axis=0), r, log_returns=False, rtol=1e-12, atol=1e-12)
    assert (
        HT.ht_compound(r.sum(axis=0), r, log_returns=True, rtol=1e-12, atol=1e-12) == []
    )


# ---------------------------------------------------------------------------
# Families, scorers and the V1.1 expansion (§34, §T11, §38 M9a)
# ---------------------------------------------------------------------------

V1_1_FAMILIES = {
    "fundamental_judgement": {"F3.2.a", "F3.6.c"},
    "corporate_action": {"F4.2.a"},
    "fundamental_extraction": {"F3.4.a", "F3.6.a", "F3.7.a", "F5.2.a"},
    "call_analysis": {"F6.1.a", "F6.2.c"},
    "broker_synthesis": {"F7.3.b", "F7.3.d"},
    "quant_method_choice": {"Q1.3.a", "Q5.1.b"},
    "quant_dgp": {"Q1.1.c", "Q2.2.b"},
    "ml_design": {"M3.2.a", "M4.3.a", "M5.1.a", "M5.1.b", "M5.2.a", "M6.1.a"},
    "code_task": {"K.Q.1", "K.Q.4", "K.Q.7", "K.M.2", "K.M.3", "K.M.4", "K.M.7"},
}


def test_every_v1_1_item_is_listed_by_its_t11_family() -> None:
    from financial_llm_harness.families import load_family
    from financial_llm_harness.taxonomy.registry import family_listing, load_registry

    listing = family_listing()
    for fid, items in V1_1_FAMILIES.items():
        assert items <= set(load_family(fid).faq_ids), fid
        for i in items:
            assert listing[i] == [fid]
    reg = load_registry()
    v11 = {i.id for i in reg.items.values() if i.wave == "V1.1"}
    assert v11 == set().union(*V1_1_FAMILIES.values())


def test_v1_1_correctness_scorers_exist_where_the_spec_defines_them() -> None:
    from financial_llm_harness.families import load_family
    from financial_llm_harness.taxonomy.registry import load_registry

    reg = load_registry()
    scorers = {f: set(load_family(f).raw.get("scorers") or {}) for f in V1_1_FAMILIES}
    assert {"roe", "roa", "roic", "dupont", "terp", "dilution"} <= scorers[
        "fundamental_extraction"
    ]
    for kid in V1_1_FAMILIES["code_task"]:
        assert set(reg.items[kid].correctness_checks) <= scorers["code_task"], kid
    assert "correct_answer" in scorers["ml_design"]


def _c1_state() -> dict[str, float]:
    from financial_llm_harness.families.fundamental_extraction import family as FX

    st: dict[str, float] = {
        "years": 4,
        "first_year": 2022,
        "revenue0": 1e9,
        "dep_rate": 0.1,
        "tax_rate": 0.25,
        "interest_rate": 0.05,
        "cash0": 1.5e8,
        "ppe0": 6e8,
        "debt0": 3e8,
        "shares0": 1e8,
    }
    vals = {
        "growth": 0.05,
        "gross_margin": 0.4,
        "opex_ratio": 0.2,
        "capex_ratio": 0.06,
        "dso": 50.0,
        "dio": 60.0,
        "dpo": 40.0,
        "accruals": 1e7,
        "unusual": 0.0,
        "impairment": 0.0,
        "payout": 0.3,
        "buyback": 0.0,
        "issuance": 0.0,
        "repayment": 1e7,
        "share_change": 0.0,
    }
    for n in FX.PER_YEAR:
        for k in range(1, 6):
            st[f"{n}_{k}"] = vals[n]
    return st


def test_fundamental_extraction_scorers_follow_the_identities() -> None:
    from financial_llm_harness.families.fundamental_extraction import family as FX

    st = _c1_state()
    for conv in FX.CONVENTIONS:
        t = FX.truth_ratios(st, conv)
        assert t["dupont_product"] == pytest.approx(
            t["roe"], rel=1e-12
        )  # DuPont identity
        ok = {"roe": t["roe"]}
        assert FX.score_roe(st, ok, field="roe", convention=conv, tolerance=1e-9)
        assert not FX.score_roe(
            st, {"roe": t["roe"] * 1.1}, field="roe", convention=conv, tolerance=1e-9
        )
        fields = {p: p for p in ("net_margin", "asset_turnover", "equity_multiplier")}
        parsed = {p: t[p] for p in fields}
        assert FX.score_dupont(
            st, parsed, fields=fields, convention=conv, tolerance=1e-9
        )
    with pytest.raises(HarnessValidationError):
        FX.truth_ratios(
            st, "median"
        )  # a convention the design must state, never defaulted
    rights = {
        "share_price": 10.0,
        "shares_outstanding": 4.0,
        "new_shares": 1.0,
        "subscription_price": 8.0,
    }
    assert FX.truth_terp(rights) == pytest.approx(9.6)
    assert FX.truth_dilution(rights, "ownership") == pytest.approx(0.2)
    assert FX.truth_dilution(rights, "value") == pytest.approx(0.04)
    assert FX.score_terp(rights, {"terp": 9.6}, field="terp", tolerance=1e-9)


def test_marchenko_pastur_edge_and_d_type_scoring() -> None:
    from financial_llm_harness.contexts import broker_notes as C5
    from financial_llm_harness.families.broker_synthesis import family as BS
    from financial_llm_harness.families.quant_dgp import family as QD

    assert QD.mp_upper_edge(100, 400, 1.0) == pytest.approx((1 + 0.5) ** 2)
    planted = [
        C5.Planted("inaccuracy", ("N1",), "revenue_2025", "revenue_2025: 1296.00")
    ]
    good = {
        "findings": [
            {
                "type": "inaccuracy",
                "note_ids": ["N1"],
                "field": "revenue_2025",
                "quote": "Revenue_2025: 1296.00",
            }
        ]
    }
    assert BS.score_findings(good, planted)["inaccuracy"]["f1"] == 1.0
    assert BS.score_findings({"findings": []}, planted)["inaccuracy"]["recall"] == 0.0


def test_v1_1_expansion_reports_every_pair_with_its_missing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from financial_llm_harness.cli import main
    from financial_llm_harness.taxonomy import expand as X
    from financial_llm_harness.taxonomy import registry as R

    monkeypatch.setattr(R, "generated_dir", lambda: tmp_path)
    monkeypatch.setattr(X, "generated_dir", lambda: tmp_path)
    assert main(["taxonomy", "expand", "--wave", "V1.1", "--allow-partial"]) == 0
    rows = [
        r
        for r in json.loads((tmp_path / "expand_report.json").read_text())
        if r["wave"] == "V1.1"
    ]
    reg = R.load_registry()
    want = {
        (i.id, v, p)
        for i in reg.items.values()
        if i.wave == "V1.1"
        for v, p in i.pairs()
    }
    assert {(r["faq_id"], r["variant_id"], r["intervention_id"]) for r in rows} == want
    for r in rows:
        assert r["reason"] == "item_not_implemented" and r["missing"], r
        fam = R.family_listing()[r["faq_id"]][0]
        # the content pack names the family's own paths, not a wildcard (§T11)
        assert any(m.startswith(f"families/{fam}/") for m in r["missing"]), r
    assert not list(
        tmp_path.glob("*/*/*.yaml")
    )  # nothing generated without D-23 content
