"""Part II registry and ``taxonomy validate`` (spec §T1, §T4, §T7, §T8, §34, M8)."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from financial_llm_harness import families
from financial_llm_harness.cli import main
from financial_llm_harness.core import yaml_io
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.taxonomy import registry as R

V1 = {"F1.1.a", "F6.1.b", "Q3.1.a", "M6.4.a", "K.M.1"}


def _with(reg: R.Registry, faq_id: str, **changes: Any) -> R.Registry:
    items = dict(reg.items)
    data = copy.deepcopy(items[faq_id].data)
    data.update(changes)
    items[faq_id] = R.Item(data)
    return R.Registry(items, reg.interventions, reg.bases, reg.provenance)


def _errors(reg: R.Registry) -> str:
    with pytest.raises(HarnessValidationError) as exc:
        R.validate_registry(reg)
    return str(exc.value)


@pytest.fixture(scope="module")
def reg() -> R.Registry:
    return R.load_registry()


def test_registry_validates_with_the_section_t8_waves(reg: R.Registry) -> None:
    R.validate_registry(reg)
    waves: dict[str, set[str]] = {}
    for item in reg.items.values():
        waves.setdefault(item.wave, set()).add(item.id)
    assert waves["V1"] == V1
    assert len(waves["V1.1"]) == 28
    assert {"F3.2.a", "K.Q.4", "K.M.7"} <= waves["V1.1"]
    # §T1 testability: the three §T5 rows with neither a relevant entry nor a check are not registered
    unregistered = yaml_io.load(R.taxonomy_dir() / "unregistered.yaml")
    assert {u["id"] for u in unregistered["unregistered"]} == {
        "Q2.3.a",
        "Q5.2.a",
        "M2.2.b",
    }
    assert not {"Q2.3.a", "Q5.2.a", "M2.2.b"} & set(reg.items)


def test_every_item_is_testable_and_cue_free_rows_get_the_stance_cue(
    reg: R.Registry,
) -> None:
    for item in reg.items.values():
        assert R.is_testable(reg, item), item.id
    # M6.4.a lists no cue in §T5 (§35 S4)
    assert reg.items["M6.4.a"].irrelevant == ["I.USER.stance"]
    # F6.1.b's "order" is S2's own intervention (§35 S2)
    assert reg.items["F6.1.b"].irrelevant == ["I.PRES.distractor"]
    # an INVARIANCE relevant entry counts as irrelevant for testability (§T1, F6.2.c)
    f62c = reg.items["F6.2.c"]
    bare = _with(
        reg, "F6.2.c", interventions={**f62c.data["interventions"], "irrelevant": []}
    )
    assert R.is_testable(bare, bare.items["F6.2.c"])
    untestable = _with(
        reg,
        "F6.2.c",
        interventions={
            "relevant": [{"id": "I.F.guidance", "oracle_type": "DIRECTIONAL"}],
            "irrelevant": [],
        },
    )
    assert not R.is_testable(untestable, untestable.items["F6.2.c"])
    assert "not testable" in _errors(untestable)


def test_keys_follow_the_slug_rule_and_fit_sixty_characters(reg: R.Registry) -> None:
    assert (
        R.experiment_key("F3.2.a", "base", "I.F.accruals", 1)
        == "f3_2_a__base__i_f_accruals.v1"
    )
    assert (
        R.experiment_key("F1.1.a", "m2", "I.F.momentum", 1)
        == "f1_1_a__m2__i_f_momentum.v1"
    )
    assert (
        R.experiment_key("F6.1.b", "base", "I.ITEM.F6.1.b.guidance_cut_position", 1)
        == "f6_1_b__base__i_item_f6_1_b_guidance_cut_position.v1"
    )
    for item in reg.items.values():
        for vid, iid in item.pairs():
            assert len(R.experiment_key(item.id, vid, iid, item.version)) <= R.MAX_KEY


def test_pairs_include_variants_and_correctness(reg: R.Registry) -> None:
    f = reg.items["F1.1.a"]
    assert [v["variant_id"] for v in f.variants] == ["m0", "m1", "m2"]
    assert len(f.pairs()) == 3 * 8
    k = reg.items["K.M.1"]
    assert ("base", "I.NONE") in k.pairs()  # correctness_checks → an I.NONE pair (D-19)
    assert [v["variant_id"] for v in reg.items["F6.1.b"].variants] == ["base"]


def test_v1_items_are_canonical_and_implemented(reg: R.Registry) -> None:
    for fid in V1:
        impl = reg.items[fid].data["implementation"]
        assert impl["kind"] == "canonical_spec" and impl["specs"]
        assert reg.items[fid].data["status"] == "implemented"
    assert "V1 item must be canonical_spec" in _errors(
        _with(reg, "Q3.1.a", implementation={"kind": "generated"})
    )


def test_canonical_spec_key_and_claim_rules(reg: R.Registry) -> None:
    # incrementing the version without updating the canonical spec's key fails
    assert "experiment_key" in _errors(_with(reg, "Q3.1.a", version=3))
    # a spec that claims a pair but is not listed fails
    impl = {
        "kind": "canonical_spec",
        "specs": ["experiments/financial/specs/stock_attractive_momentum_m1.yaml"],
    }
    assert "not in its implementation.specs" in _errors(
        _with(reg, "F1.1.a", implementation=impl)
    )
    # a listed spec whose pair is not the item's fails
    assert "is no pair of" in _errors(
        _with(reg, "F1.1.a", variants=[{"variant_id": "m1", "mandate": "M1"}])
    )


def test_references_resolve(reg: R.Registry) -> None:
    assert "basis B.nope unknown" in _errors(_with(reg, "F3.2.a", bases=["B.nope"]))
    assert "context 'C99' unknown" in _errors(_with(reg, "F3.2.a", context=["C99"]))
    item = reg.items["F3.2.a"]
    bad = {**item.data["interventions"], "irrelevant": ["I.F.growth"]}
    assert "is not a §T4.2 cue" in _errors(_with(reg, "F3.2.a", interventions=bad))
    bad_rel = {
        **item.data["interventions"],
        "relevant": [{"id": "I.F.nothing", "oracle_type": "DIRECTIONAL"}],
    }
    assert "resolves to neither" in _errors(_with(reg, "F3.2.a", interventions=bad_rel))


def test_variants_schema(reg: R.Registry) -> None:
    assert "malformed variant_id" in _errors(
        _with(reg, "F3.2.a", variants=[{"variant_id": "M1"}])
    )
    assert "repeated" in _errors(
        _with(reg, "F3.2.a", variants=[{"variant_id": "a"}, {"variant_id": "a"}])
    )
    assert "keys outside" in _errors(
        _with(reg, "F3.2.a", variants=[{"variant_id": "a", "tau": 0.9}])
    )


def test_status_must_match_implementation(reg: R.Registry) -> None:
    assert "but declared specified" in _errors(_with(reg, "Q3.1.a", status="specified"))
    assert "needs a family listing it" in _errors(
        _with(reg, "F3.2.a", status="implemented")
    )


def test_family_listing_and_contexts(
    reg: R.Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "contexts ['C6'] outside family call_analysis" in _errors(
        _with(reg, "F6.1.b", context=["C6"])
    )
    real = R.family_listing()
    monkeypatch.setattr(
        R, "family_listing", lambda: {**real, "F6.1.b": ["call_analysis", "ml_design"]}
    )
    assert "listed by two families" in _errors(reg)
    monkeypatch.setattr(R, "family_listing", lambda: real)
    monkeypatch.setitem(R.FAMILY_CONTEXTS, "call_analysis", {"C3"})
    assert "families/call_analysis/family.yaml: contexts ['C4']" in _errors(reg)


def test_defaults_oracle_type_must_agree(
    reg: R.Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = families.load_defaults

    def patched(fid: str) -> dict[str, Any]:
        d = copy.deepcopy(real(fid))
        if fid == "fundamental_judgement":
            rec = d["entries"]["F1.1.a"]["variants"]["m2"]["I.F.valuation"]
            rec["oracle_spec"]["mandate_id"] = "M1"
        return d

    monkeypatch.setattr(families, "load_defaults", patched)
    assert "per_mandate record names M1, variant mandate is M2" in _errors(reg)


def test_orphan_generated_files_fail(
    reg: R.Registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(R, "generated_dir", lambda: tmp_path)
    R.validate_registry(reg)
    claimed = (
        tmp_path / "F1.1.a" / "m1" / "I.F.momentum.yaml"
    )  # claimed by the S1 canonical spec
    claimed.parent.mkdir(parents=True)
    claimed.write_text("{}\n", encoding="utf-8")
    assert "orphan generated file" in _errors(reg)
    claimed.unlink()
    stray = tmp_path / "F1.1.a" / "m3" / "I.F.valuation.yaml"  # no such variant
    stray.parent.mkdir(parents=True)
    stray.write_text("{}\n", encoding="utf-8")
    assert "orphan generated file" in _errors(reg)


def test_catalogues_cover_section_t4_and_t7(reg: R.Registry) -> None:
    irrelevant = {i for i, e in reg.interventions.items() if e["kind"] == "irrelevant"}
    # §T4.2: the 19 original cues, I.ID.ceo split into gender and ethnicity cues and
    # I.HYPE.label split into rhetorical and architecture labels (CR-8)
    assert len(irrelevant) == 21 and "I.CODE.comment" in irrelevant
    assert {"I.ID.ceo_ethnicity", "I.HYPE.arch"} <= irrelevant
    for iid in irrelevant:
        e = reg.interventions[iid]
        assert e["attach"] and e["guide"] and e["category"]
        # architecture labels describe a method class, so they are not irrelevant by construction
        want = "DISCOVERY" if iid == "I.HYPE.arch" else "INVARIANCE"
        assert e["oracle_default"] == want, iid
    assert reg.bases["B.irrelevant"]["confidence"] == "established"
    assert reg.bases["B.lw2004"]["confidence"] == "established"
    assert reg.bases["B.bmt1997"]["confidence"] == "contested"
    assert (
        reg.bases["B.syc2023"]["oracle"] is False
    )  # LLM-behaviour papers are never oracles


def test_cli_validate_and_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["taxonomy", "validate"]) == 0
    assert "118 items OK" in capsys.readouterr().out
    assert main(["taxonomy", "list", "--domain", "K", "--wave", "V1"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert [ln.split("\t")[0] for ln in out[1:]] == ["K.M.1"]


def test_expert_panel_sign_off_rule() -> None:
    from financial_llm_harness.taxonomy import expert_panel as E

    panel = {
        "members": [
            {"id": "a", "role": "fundamental_analyst", "conflicts": []},
            {"id": "b", "role": "fundamental_pm", "conflicts": []},
            {"id": "c", "role": "fundamental_pm", "conflicts": ["F1.1.a"]},
            {"id": "d", "role": "ml_architect", "conflicts": []},
        ]
    }
    base = {
        "basis": "B.conv",
        "faq_id": "F1.1.a",
        "domain": "fundamental",
        "items": [{"id": "1", "text": "t", "rationale": "r"}],
    }
    signed = {**base, "signoffs": [{"member": "a"}, {"member": "b"}]}
    assert E.is_signed_off(signed, panel)
    # two approvals from one role, a conflicted member, or an unmapped role do not count (§T10.6)
    assert not E.is_signed_off(
        {**base, "signoffs": [{"member": "b"}, {"member": "b"}]}, panel
    )
    assert not E.is_signed_off(
        {**base, "signoffs": [{"member": "a"}, {"member": "c"}]}, panel
    )
    assert not E.is_signed_off(
        {**base, "signoffs": [{"member": "a"}, {"member": "d"}]}, panel
    )
    assert not E.is_signed_off({**signed, "items": []}, panel)
    assert not E.is_signed_off(
        {**signed, "items": [{"id": "1", "text": "t", "rationale": ""}]}, panel
    )
    # the shipped panel names no member (D-13): nothing is signed off
    assert E.load_panel()["members"] == []
    assert not E.checklist_signed_off("B.conv", "F1.1.a")


# ---------------------------------------------------------------------------
# Context generators (§T3, §T9): planted truth recoverable, identities exact
# ---------------------------------------------------------------------------


def _drivers(**over: Any) -> Any:
    from financial_llm_harness.contexts.statements import Drivers

    n = 4
    base: dict[str, Any] = {
        "years": n,
        "revenue0": 1.0e9,
        "growth": [0.05, 0.07, 0.04, 0.06],
        "gross_margin": [0.40, 0.41, 0.39, 0.40],
        "opex_ratio": [0.20] * n,
        "dep_rate": 0.10,
        "capex_ratio": [0.06] * n,
        "tax_rate": 0.25,
        "interest_rate": 0.05,
        "dso": [45.0, 50.0, 55.0, 60.0],
        "dio": [60.0] * n,
        "dpo": [40.0] * n,
        "accruals": [0.0, 2.0e7, 4.0e7, 6.0e7],
        "unusual": [0.0, 1.5e7, 0.0, 0.0],
        "impairment": [0.0, 0.0, 3.0e7, 0.0],
        "payout": [0.3] * n,
        "buyback": [0.0, 1.0e7, 0.0, 2.0e7],
        "issuance": [0.0, 5.0e7, 0.0, 0.0],
        "repayment": [2.0e7, 0.0, 6.0e7, 0.0],
        "share_change": [0.0, -0.01, 0.02, 0.0],
        "cash0": 1.5e8,
        "ppe0": 6.0e8,
        "debt0": 3.0e8,
        "shares0": 1.0e8,
    }
    base.update(over)
    return Drivers(**base)


def test_c1_identities_hold_and_planted_truth_round_trips() -> None:
    from financial_llm_harness.contexts import statements as C1

    pack = C1.statement_pack(_drivers())
    assert C1.identity_violations(pack) == []
    truth = pack.truth
    assert truth["ccc_days"] == [65.0, 70.0, 75.0, 80.0]  # DSO + DIO − DPO
    assert truth["unusual_items"] == [(1, 1.5e7)] and truth["impairments"] == [
        (2, 3.0e7)
    ]
    assert truth["debt_maturities"] == [(0, 2.0e7), (2, 6.0e7)]
    # recovered from the rendered tables alone (rounded to $0.1m)
    text = C1.render_pack(pack, first_year=2022)
    back = C1.ratios_from_text(text)
    for got, want in zip(back["accruals_ratio"], truth["accruals_ratio"], strict=True):
        assert abs(got - want) < 1e-3
    # planted accruals push net income above cash flow: the ratio rises with them
    assert truth["accruals_ratio"][3] > truth["accruals_ratio"][0]
    bad = C1.statement_pack(_drivers())
    bad.years[1].balance["cash"] += 1.0e6
    assert C1.identity_violations(bad)
    with pytest.raises(HarnessValidationError):
        C1.statement_pack(_drivers(years=2))


def test_c2_planted_momentum_is_exact_and_beta_recovered() -> None:
    from financial_llm_harness.contexts.market import (
        MarketParams,
        market_panel,
        realised_beta,
    )

    p = MarketParams(
        n_days=756,
        momentum_window=252,
        momentum=-0.25,
        beta=1.3,
        market_mu=0.0003,
        market_sigma=0.01,
        idio_sigma=0.004,
        price_now=50.0,
        volume_mean=1e6,
        event_days=[300, 600],
        event_surprise=[0.05, -0.04],
        revision=[0.02, -0.03],
    )
    panel = market_panel(p, np.random.default_rng(1))
    assert panel.prices[-1] == pytest.approx(50.0)
    assert panel.prices[-1] / panel.prices[-253] - 1 == pytest.approx(-0.25, abs=1e-12)
    assert abs(realised_beta(panel, exclude=[300, 600]) - 1.3) < 0.05
    again = market_panel(p, np.random.default_rng(1))
    assert np.array_equal(again.prices, panel.prices)


def test_c3_terp_and_accretion_are_exact() -> None:
    from financial_llm_harness.contexts import corporate_action as C3

    # 1-for-4 rights issue at 8.00 on shares trading at 10.00: (4·10 + 1·8) / 5 = 9.60
    assert C3.terp(10.0, 4.0, 1.0, 8.0) == pytest.approx(9.6, abs=1e-12)
    deal = C3.Deal(
        acquirer_net_income=100.0,
        acquirer_shares=100.0,
        target_net_income=20.0,
        offer_value=300.0,
        stock_fraction=0.5,
        acquirer_price=15.0,
        debt_rate=0.06,
        cash_yield=0.02,
        cash_fraction=0.2,
        synergies=10.0,
        tax_rate=0.25,
    )
    out = C3.accretion(deal)
    ni = 100 + 20 + 10 * 0.75 - 90 * 0.06 * 0.75 - 60 * 0.02 * 0.75
    assert out["new_shares"] == pytest.approx(10.0)
    assert out["pro_forma_eps"] == pytest.approx(ni / 110.0, abs=1e-12)
    assert out["accretion"] == pytest.approx((ni / 110.0) / 1.0 - 1.0, abs=1e-12)
    assert C3.premium(13.0, 10.0) == pytest.approx(0.3)


def test_c4_tone_dial_moves_tone_only() -> None:
    from financial_llm_harness.contexts import transcript as C4

    lex = {
        "positive": ["strong", "improved"],
        "negative": ["weak", "declined"],
        "neutral": ["stable"],
    }
    s = "Demand was {tone} and margins {tone}."
    pos = C4.fill_tone(s, "positive", lex, np.random.default_rng(0))
    neg = C4.fill_tone(s, "negative", lex, np.random.default_rng(0))
    assert C4.net_tone(pos, lex) == 1.0 and C4.net_tone(neg, lex) == -1.0
    assert (
        C4.net_tone(C4.fill_tone(s, "neutral", lex, np.random.default_rng(0)), lex)
        is None
    )
    assert C4.guidance_direction(100.0, 95.0) == "cut"
    assert C4.guidance_direction(100.0, 100.0) == "maintained"


def _note_set() -> tuple[list[Any], list[Any]]:
    from financial_llm_harness.contexts import broker_notes as C5

    params = C5.SetParams(
        k_notes=4,
        eps=0.02,
        d=2,
        k=2,
        a=1,
        n=1,
        difference_offset=0.2,
        inaccuracy_offset=0.08,
        rounding=0.005,
    )
    return C5.plant_set(
        consensus={"2026": 3.0, "2027": 3.4},
        ratings=["buy", "hold"],
        thesis_ids=["T1", "T2", "T3"],
        history={"revenue_2025": 1200.0, "ebit_2025": 180.0},
        metrics=["adjusted EBITDA"],
        novel_pool=["inventory build in the second half"],
        p=params,
        rng=np.random.default_rng(3),
        quote=lambda t, f, v: f"{f}: {v:.2f}" if v is not None else f"{f}",
    )


def test_c5k_d_type_scoring() -> None:
    from financial_llm_harness.contexts import broker_notes as C5

    notes, planted = _note_set()
    assert {p.type for p in planted} == {
        "difference",
        "inaccuracy",
        "ambiguity",
        "novel",
    }
    texts = {
        n.note_id: C5.render_note(
            n, lambda t, f, v: f"{f}: {v:.2f}" if v is not None else f"{f}"
        )
        for n in notes
    }
    for p in planted:  # every planted quote is in its note's text (round trip)
        assert all(p.quote in texts[i] for i in p.note_ids)
    perfect = [
        {
            "type": p.type,
            "note_ids": list(p.note_ids),
            "field": p.field,
            "quote": p.quote,
        }
        for p in planted
    ]
    scores = C5.score(perfect, planted)
    assert all(scores[t]["f1"] == 1.0 for t in C5.TYPES)
    empty = C5.score([], planted)
    assert all(empty[t]["recall"] == 0.0 for t in C5.TYPES)
    stray = C5.score(
        [
            *perfect,
            {
                "type": "inaccuracy",
                "note_ids": ["N1"],
                "field": "capex_2025",
                "quote": "capex",
            },
        ],
        planted,
    )
    precision = stray["inaccuracy"]["precision"]
    assert stray["inaccuracy"]["fp"] == 1 and precision is not None and precision < 1.0
    with pytest.raises(HarnessValidationError):
        C5.plant_set(
            {"2026": 3.0},
            ["buy"],
            ["T1"],
            {},
            [],
            [],
            C5.SetParams(2, 0.06, 1, 0, 0, 0, 0.2, 0.1, 0.01),
            np.random.default_rng(0),
        )


def test_c7_dgp_parameters_are_recovered() -> None:
    from financial_llm_harness.contexts import dgp as C7

    p = C7.DGPParams(
        n_assets=400,
        n_dates=240,
        factor_vol=[0.04, 0.02],
        beta_sd=0.5,
        gamma=[[0.01, 0.0, -0.006]],
        resid_vol=0.06,
        t_dof=5.0,
        publication_lag=1,
    )
    panel = C7.simulate(p, np.random.default_rng(11))
    z = panel.signals
    est = []
    for j in range(3):
        coefs = []
        for t in range(p.publication_lag, p.n_dates):
            x, y = (
                z[t, :, j],
                panel.returns[t - p.publication_lag],
            )  # unlag what the panel shows
            coefs.append(float(np.polyfit(x, y, 1)[0]))
        est.append(float(np.mean(coefs)))
    assert est[0] == pytest.approx(0.01, abs=0.002)
    assert est[1] == pytest.approx(0.0, abs=0.002)
    assert est[2] == pytest.approx(-0.006, abs=0.002)
    assert panel.truth["true_signals"] == [0, 2] and panel.truth[
        "spurious_signals"
    ] == [1]
    ic = panel.truth["population_ic"][0]
    assert ic[0] > 0 > ic[2] and ic[1] == 0.0
    assert np.isnan(z[0]).all()  # the publication lag hides the first date
    with pytest.raises(HarnessValidationError):
        C7.simulate(
            C7.DGPParams(10, 10, [0.1], 0.5, [[0.1], [0.2]], 0.1, 5.0),
            np.random.default_rng(0),
        )


# ---------------------------------------------------------------------------
# Survey and weights (§T10.2, §T10.3)
# ---------------------------------------------------------------------------


def _responses(
    path: Path, per_role: int, roles: list[str], dup: str | None = None
) -> Path:
    import csv as _csv

    from financial_llm_harness.taxonomy.survey import instrument, items_for

    inst = instrument()
    rows = []
    rng = np.random.default_rng(5)
    freq = ["never", "yearly", "quarterly", "monthly", "weekly", "daily"]
    for role in roles:
        items = items_for(role, inst)[:12]
        for r in range(per_role):
            for it in items:
                rows.append(
                    {
                        "respondent_id": f"{role}-{r}",
                        "role": role,
                        "faq_id": it.id,
                        "frequency": freq[int(rng.integers(0, 6))],
                        "stakes": int(rng.integers(1, 6)),
                        "channel": "chat LLM",
                    }
                )
    if dup is not None:
        rows += [
            {**x, "respondent_id": x["respondent_id"] + "-copy"}
            for x in rows
            if x["role"] == dup
        ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_survey_routing_exports_only_mapped_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from financial_llm_harness.taxonomy import survey as S

    monkeypatch.setattr(S, "export_dir", lambda: tmp_path)
    inst = S.instrument()
    ids = {i.id for i in S.items_for("pension_cio", inst)}
    assert ids and all(i.split(".")[0] in {"F1", "F4", "Q3", "Q5", "Q6"} for i in ids)
    dev = {i.id for i in S.items_for("ml_developer", inst)}
    assert "K.M.1" in dev and all(i.startswith(("M", "K.")) for i in dev)
    paths = S.export_survey("pension_cio")
    rows = (paths[0]).read_text(encoding="utf-8").splitlines()
    assert {r.split(",")[3] for r in rows[1:] if ",closed," in r} == ids


def test_weights_threshold_balancing_and_determinism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from financial_llm_harness.taxonomy import registry as Rg
    from financial_llm_harness.taxonomy import weights as W

    out = tmp_path / "tax"
    out.mkdir()
    monkeypatch.setattr(W, "taxonomy_dir", lambda: out)
    roles = sorted(W.instrument()["routing"])
    few = _responses(tmp_path / "few.csv", 3, roles)
    doc = Rg.yaml_io.load(W.compute_weights(few))
    assert doc["threshold"]["met"] is False
    assert W.used_weights() is None  # below D-12: every weight stays 1
    full = _responses(tmp_path / "full.csv", 5, roles[:6])
    a = W.compute_weights(full).read_bytes()
    assert W.compute_weights(full).read_bytes() == a  # deterministic under its seed
    doc = Rg.yaml_io.load(out / "weights.yaml")
    assert doc["threshold"]["met"] is True
    for dom in ("F", "Q", "M", "K"):
        ws = [
            v["weight"]
            for f, v in doc["items"].items()
            if W.DOMAIN_LETTER[Rg.load_registry().items[f].data["domain"]] == dom
        ]
        if ws:
            assert sum(ws) == pytest.approx(1.0)
    dup = _responses(tmp_path / "dup.csv", 5, roles[:6], dup=roles[0])
    doc_dup = Rg.yaml_io.load(W.compute_weights(dup))
    assert {f: v["weight"] for f, v in doc_dup["items"].items()} == {
        f: v["weight"] for f, v in doc["items"].items()
    }  # role balancing: duplicating one role's respondents changes no weight


def test_coverage_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from financial_llm_harness.taxonomy import coverage as Cv

    monkeypatch.setattr(Cv, "taxonomy_dir", lambda: tmp_path)
    text = Cv.write_coverage().read_text(encoding="utf-8")
    assert "## fundamental" in text and "| J |" in text
    assert "B.conv" in text  # conventional bases still unsigned are listed
    assert "Fleiss' κ is not available" in text
    assert Cv.fleiss_kappa([["a", "a"], ["b", "b"], ["a", "a"]]) == pytest.approx(1.0)
    assert Cv.fleiss_kappa([["a"]]) is None


def test_references_cover_the_literature_bases(reg: R.Registry) -> None:
    text = (R.taxonomy_dir() / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@misc\{([^,]+),", text))
    construction = {"B.identity", "B.planted", "B.mandate", "B.irrelevant", "B.conv"}
    assert keys == set(reg.bases) - construction


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("HARNESS_LIVE") != "1", reason="HARNESS_LIVE=1 not set"
)
def test_every_reference_doi_resolves() -> None:
    import urllib.request

    text = (R.taxonomy_dir() / "references.bib").read_text(encoding="utf-8")
    for doi in re.findall(r"doi\s*=\s*\{([^}]+)\}", text):
        req = urllib.request.Request(f"https://doi.org/{doi}", method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https host
            assert resp.status < 400, doi
