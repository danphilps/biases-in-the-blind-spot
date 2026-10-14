"""The undeclared-bias battery (spec §T4.4, §32B.2 step 1c, E-31, M8)."""

from __future__ import annotations

import copy
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.evaluation.rules import FeatureContext, extract
from financial_llm_harness.families import load_family, load_mandate
from financial_llm_harness.generation.corpus import build_corpus
from financial_llm_harness.taxonomy import battery as BT
from financial_llm_harness.taxonomy import expand as X
from financial_llm_harness.taxonomy.registry import family_listing, load_registry
from tests.harness.helpers import FAKE_A, ROOT, read_csv, run_slice

# I.ID.name and I.ID.ceo_ethnicity are listed on F1.1.a but have no bank until the
# validated data of D-25 exist, so they build no spec (test_identity_cues_await_d25).
V1_BATTERY = {
    ("F1.1.a", "m0"): ["I.ID.ceo", "I.USER.stance", "I.PRES.order"],
    ("F1.1.a", "m1"): ["I.ID.ceo", "I.USER.stance", "I.PRES.order"],
    ("F1.1.a", "m2"): ["I.ID.ceo", "I.USER.stance", "I.PRES.order"],
    ("F6.1.b", "base"): ["I.PRES.distractor"],
    ("Q3.1.a", "base"): ["I.HYPE.label", "I.HYPE.arch", "I.USER.stance"],
    ("M6.4.a", "base"): ["I.USER.stance"],
    ("K.M.1", "base"): ["I.CODE.comment", "I.HYPE.label", "I.HYPE.arch"],
}
# Cues that are not irrelevant by construction (§T4.2): DISCOVERY, so discovery mode.
# I.HYPE.arch is not irrelevant by construction; the others are disputed until validated
# (D-27). Only I.PRES.distractor (S2) stays a confirmatory INVARIANCE cue in V1.
DISCOVERY_CUES = {
    "I.HYPE.arch",
    "I.ID.ceo",
    "I.USER.stance",
    "I.PRES.order",
    "I.HYPE.label",
    "I.CODE.comment",
}


@pytest.fixture(scope="module")
def battery_specs() -> dict[tuple[str, str, str], dict[str, Any]]:
    reg = load_registry()
    listing = family_listing()
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for faq in {f for f, _ in V1_BATTERY}:
        for r in X.expand_item(reg, reg.items[faq], listing):
            if r.spec is not None and r.spec.get("battery"):
                out[(r.faq_id, r.variant_id, r.intervention_id)] = r.spec
    return out


def _small(spec: dict[str, Any], n: int = 24) -> dict[str, Any]:
    """The battery design in discovery mode with a few cases: arms A and B only, for rendering checks."""
    s = copy.deepcopy(spec)
    s["mode"] = "discovery"
    s["scenario_spec"]["n_cases"] = n
    s["scenario_spec"]["seed_stream"] = "disc"
    return s


def test_every_v1_cue_pair_is_built_from_its_base(
    battery_specs: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    expected = {(f, v, c) for (f, v), cues in V1_BATTERY.items() for c in cues}
    assert set(battery_specs) == expected
    base_paths = {
        "F1.1.a": "experiments/financial/specs/stock_attractive_momentum_{vid}.yaml",
        "F6.1.b": "experiments/financial/specs/transcript_summary.yaml",
        "Q3.1.a": "experiments/financial/specs/covariance_choice.yaml",
        "M6.4.a": "experiments/financial/specs/temporal_norm_design.yaml",
        "K.M.1": "experiments/financial/specs/temporal_norm_code.yaml",
    }
    for (faq, vid, cue), spec in battery_specs.items():
        b = spec["battery"]
        assert b["base"]["ref"] == base_paths[faq].format(vid=vid)
        base = X.yaml_io.load(X.repo_root() / b["base"]["ref"])
        # copied unchanged: controls, sizing, seeds, statistics, contract (§T4.4)
        assert spec["controls"] == base["controls"]
        assert spec["scenario_spec"]["n_cases"] == base["scenario_spec"]["n_cases"]
        assert spec["seeds"] == base["seeds"]
        assert spec["statistical_spec"] == base["statistical_spec"]
        assert spec["response_spec"] == base["response_spec"]
        # set by the battery
        assert spec["intervention_id"] == cue and spec["concept"] == cue
        assert spec["oracle_spec"]["basis"] == ["B.irrelevant"]
        assert spec["hypothesis"]["source"] == "a_priori"
        assert spec["hypothesis"]["estimand"] == "mean_paired_difference"
        assert spec["hypothesis"]["level_arm"] is None
        if cue in DISCOVERY_CUES:
            assert spec["oracle_spec"]["type"] == "DISCOVERY"
            assert spec["hypothesis"]["direction"] == "two_sided"
            assert spec["mode"] == "discovery"
        else:
            assert spec["oracle_spec"]["type"] == "INVARIANCE"
            assert spec["hypothesis"]["direction"] == "none"
            assert spec["mode"] == "confirmatory"  # P1 and B.irrelevant is established
        assert not any(
            f.get("extractor") == "concept_factor" for f in spec["feature_spec"]
        )
        BT.validate_irrelevance(spec)


def test_context_is_held_at_the_level_arm(
    battery_specs: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    holds = {k: v["battery"]["hold"] for k, v in battery_specs.items()}
    assert holds[("F1.1.a", "m1", "I.USER.stance")] == {
        "node": "momentum_12m",
        "via": "price_12m_ago",
        "value": 0.25,
    }
    assert (
        holds[("Q3.1.a", "base", "I.HYPE.label")]["value"] == 0.5
    )  # T/N = 0.5 (§35 S3)
    assert holds[("M6.4.a", "base", "I.USER.stance")]["value"] == (
        "signal for a walk-forward backtest"
    )
    assert holds[("F6.1.b", "base", "I.PRES.distractor")]["value"] == 2
    corpus = build_corpus(
        _small(battery_specs[("M6.4.a", "base", "I.USER.stance")]),
        BT.cue_hook_for(battery_specs[("M6.4.a", "base", "I.USER.stance")]),
    )
    for case in corpus.cases:
        for arm in ("A", "B"):
            assert case["arms"][arm]["purpose"] == "signal for a walk-forward backtest"


def _renders(corpus: Any) -> dict[tuple[int, str, int], str]:
    return {
        (r["case_idx"], r["arm"], r["paraphrase_idx"]): r["user"]
        for r in corpus.renderings
    }


@pytest.mark.parametrize(
    "key",
    [
        ("F1.1.a", "m1", "I.USER.stance"),
        ("F1.1.a", "m0", "I.ID.ceo"),
        ("F1.1.a", "m2", "I.ID.ceo"),
        ("F1.1.a", "m1", "I.PRES.order"),
        ("F6.1.b", "base", "I.PRES.distractor"),
        ("Q3.1.a", "base", "I.HYPE.label"),
        ("Q3.1.a", "base", "I.HYPE.arch"),
        ("K.M.1", "base", "I.CODE.comment"),
    ],
)
def test_arms_differ_only_by_the_cue(
    battery_specs: dict[tuple[str, str, str], dict[str, Any]], key: tuple[str, str, str]
) -> None:
    spec = battery_specs[key]
    corpus = build_corpus(_small(spec), BT.cue_hook_for(spec))
    node = spec["battery"]["node"]
    users = _renders(corpus)
    for case in corpus.cases:
        a, b = case["arms"]["A"], case["arms"]["B"]
        assert {k for k in a if a[k] != b.get(k)} == {
            node
        }  # one state node differs (§9.1)
        assert case["cue"]["cue"] == key[2]
        for p in range(3):
            ua, ub = (
                users[(case["case_idx"], "A", p)],
                users[(case["case_idx"], "B", p)],
            )
            assert ua != ub
            if key[2] == "I.USER.stance":
                pre = ub[: len(ub) - len(ua)]
                assert ub.endswith(ua) and pre.endswith("\n\n")
                assert str(case["cue"]["stance_target"]) in pre
            elif key[2] in ("I.ID.name", "I.ID.ceo"):
                va, vb = case["cue"]["pair"]
                assert ua.replace(va, vb) == ub
                if key[2] == "I.ID.ceo":
                    # gender cue only: the surname is shared (§T4.2)
                    assert va.split()[-1] == vb.split()[-1]
                    assert case["cue"]["contrast"] == "gender"
            elif key[2] in ("I.HYPE.label", "I.HYPE.arch"):
                va, vb = case["cue"]["pair"]
                assert ua.replace(va, vb) == ub
                assert set(va.split()) < set(
                    vb.split()
                )  # the same label plus a modifier
            elif key[2] == "I.CODE.comment":
                comment = case["cue"]["comment"]
                assert ub.replace(comment + "\n", "", 1) == ua
                assert f"{comment}\ndef normalise(" in ub
            elif key[2] == "I.PRES.order":
                la = [ln for ln in ua.splitlines() if ln.startswith("- ")]
                lb = [ln for ln in ub.splitlines() if ln.startswith("- ")]
                assert sorted(la) == sorted(lb)
                assert lb == la[1:] + la[:1]  # the key field moves from first to last
                assert "12 months" in la[0]  # the key field, momentum_12m
            elif key[2] == "I.PRES.distractor":
                la = [ln for ln in ua.splitlines() if ln.startswith("- ")]
                lb = [ln for ln in ub.splitlines() if ln.startswith("- ")]
                assert [ln for ln in lb if ln in la] == la  # every real item, in order
                assert la[1] == lb[1]  # the guidance cut keeps position 2
                extra = [ln for ln in lb if ln not in la]
                tokens = sum(len(x.split()) for x in extra)
                assert tokens >= 0.25 * sum(len(x.split()) for x in lb)


def test_stance_targets_are_balanced_and_numeric_quantiles() -> None:
    stance = {"kind": "option", "values": ["a", "b", "c", "d", "e"]}
    rng = np.random.default_rng(20261007)
    counts = Counter(BT.draw_stance(stance, rng, i) for i in range(10_000))
    assert counts == dict.fromkeys(stance["values"], 2000)  # exactly balanced
    # any run of consecutive cases is balanced to within one
    window = Counter(BT.draw_stance(stance, rng, i) for i in range(37, 37 + 23))
    assert max(window.values()) - min(window.values()) <= 1
    numeric = {"kind": "value", "low": 0.0, "high": 1.0, "quantiles": [0.2, 0.8]}
    values = {BT.draw_stance(numeric, rng, i) for i in range(200)}
    assert values == {0.2, 0.8}


def test_stance_construction_follows_the_contract_and_interaction() -> None:
    fam = load_family("fundamental_judgement")
    s1 = X.yaml_io.load(
        X.repo_root() / "experiments/financial/specs/momentum_stock_add.yaml"
    )
    assert BT.stance_design(fam, s1, "J") == {
        "kind": "option",
        "field": "decision",
        "values": ["ADD", "DO_NOT_ADD"],
    }
    for interaction in ("S", "D", "P", "C"):
        assert BT.stance_design(fam, s1, interaction) == {"kind": "dismissive"}
    s1_v2 = X.yaml_io.load(
        X.repo_root() / "experiments/financial/specs/stock_attractive_momentum_m0.yaml"
    )
    assert BT.stance_design(fam, s1_v2, "J") == {
        "kind": "option",
        "field": "decision",
        "values": ["ATTRACTIVE", "NOT_ATTRACTIVE"],
    }
    # M6.4.a: the stance targets an enum value of normalisation_window (§35 S4)
    s4 = X.yaml_io.load(
        X.repo_root() / "experiments/financial/specs/temporal_norm_design.yaml"
    )
    st = BT.stance_design(load_family("ml_design"), s4, "M")
    assert st["field"] == "normalisation_window" and "full_sample" in st["values"]


def test_agrees_with_stance_is_a_discovery_secondary(
    battery_specs: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    spec = battery_specs[("F1.1.a", "m1", "I.USER.stance")]
    feat = next(
        f for f in spec["feature_spec"] if f["feature_id"] == "agrees_with_stance"
    )
    assert feat["field"] == "decision" and feat["analysis_role"] == "secondary"
    assert {
        "feature_id": "agrees_with_stance",
        "contrast": "B_vs_A",
        "type": "DISCOVERY",
        "role": "secondary",
    } in spec["oracle_spec"]["secondary"]
    ctx = FeatureContext(
        arm="B",
        concept_factor=None,
        stance_target="ATTRACTIVE",
        truth=None,
        code_outcomes=None,
        state=None,
    )
    module = load_family("fundamental_judgement").module
    assert extract(feat, {"decision": "ATTRACTIVE"}, ctx, module).value == 1
    assert extract(feat, {"decision": "NOT_ATTRACTIVE"}, ctx, module).value == 0
    # a dismissive stance (S2) has no categorical target, so no agreement feature
    s2 = battery_specs[("F6.1.b", "base", "I.PRES.distractor")]
    assert not any(f["feature_id"] == "agrees_with_stance" for f in s2["feature_spec"])


def test_conditional_defaults() -> None:
    cues = BT.load_cues()

    def otype(cue_id: str, interaction: str, mandate: dict[str, Any]) -> str:
        return BT.oracle_type_for(cues[cue_id], interaction, mandate)

    assert otype("I.ID.country", "J", {"country_neutral": True}) == "INVARIANCE"
    assert otype("I.ID.country", "J", {}) == "DISCOVERY"
    assert otype("I.SRC.broker_tier", "S", {}) == "INVARIANCE"
    assert otype("I.SRC.seniority", "D", {}) == "INVARIANCE"
    assert otype("I.SRC.broker_tier", "E", {}) == "DISCOVERY"
    assert otype("I.SRC.consensus_label", "D", {}) == "INVARIANCE"
    assert otype("I.SRC.consensus_label", "S", {}) == "DISCOVERY"
    assert otype("I.HYPE.arch", "M", {}) == "DISCOVERY"
    assert otype("I.PRES.distractor", "S", {}) == "INVARIANCE"


DISPUTED = (
    "I.ID.name",
    "I.ID.ceo",
    "I.ID.ceo_ethnicity",
    "I.PRES.order",
    "I.USER.stance",
    "I.HYPE.label",
    "I.CODE.comment",
)


def test_disputed_cues_are_discovery_until_validated() -> None:
    """CR-07, D-27: a cue whose irrelevance is disputed never yields a confirmatory bias."""
    cues = BT.load_cues()
    assert {
        i for i, c in cues.items() if c.get("irrelevance_status") == "disputed"
    } == set(DISPUTED)
    for cue_id in DISPUTED:
        assert BT.oracle_type_for(cues[cue_id], "J", {}) == "DISCOVERY", cue_id
    # the status, not the stored default alone, decides
    validated = {
        **cues["I.USER.stance"],
        "irrelevance_status": None,
        "oracle_default": "INVARIANCE",
    }
    assert BT.oracle_type_for(validated, "J", {}) == "INVARIANCE"


def test_generated_disputed_cue_specs_are_discovery() -> None:
    from financial_llm_harness.core import yaml_io

    root = ROOT / "experiments" / "financial" / "generated"
    seen = 0
    for path in sorted(root.glob("*/*/I.*.yaml")):
        spec = yaml_io.load(path)
        if spec["intervention_id"] in DISPUTED:
            seen += 1
            assert spec["mode"] == "discovery", path
            assert spec["oracle_spec"]["type"] == "DISCOVERY", path
            assert spec["oracle_spec"]["confidence"] == "contested", path
    assert seen > 0


def test_missing_slot_or_bank_is_named_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    fam = load_family("ml_design")
    base = BT.BaseDesign(
        X.yaml_io.load(
            X.repo_root() / "experiments/financial/specs/temporal_norm_design.yaml"
        ),
        "experiments/financial/specs/temporal_norm_design.yaml",
        "I.Q.purpose",
    )
    cues = BT.load_cues()
    mandate = load_mandate("ml_design", "MLD1")
    spec, missing = BT.battery_design(base, cues["I.ID.ceo"], fam, "M", mandate)
    assert spec is None
    assert missing == ["families/ml_design/family.yaml#battery_slots/entity:ceo_name"]
    real = BT.resolve_bank
    monkeypatch.setattr(
        BT, "resolve_bank", lambda logical: real(logical).with_name("absent.yaml")
    )
    spec, missing = BT.battery_design(base, cues["I.USER.stance"], fam, "M", mandate)
    assert spec is None and missing == ["taxonomy/banks/I.USER.stance.yaml"]


def test_irrelevance_by_construction_is_enforced(
    battery_specs: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    spec = battery_specs[("F1.1.a", "m1", "I.ID.ceo")]
    bad = copy.deepcopy(spec)
    bad["oracle_spec"]["nodes"] = ["ceo_name"]
    with pytest.raises(HarnessValidationError, match="references the cue node"):
        BT.validate_irrelevance(bad)
    bad = copy.deepcopy(spec)
    bad["controls"]["placebo"] = {"node": "ceo_name", "values": ["x", "y"]}
    with pytest.raises(HarnessValidationError, match="placebo control intervenes"):
        BT.validate_irrelevance(bad)
    bad = copy.deepcopy(spec)
    bad["battery"]["node"] = bad["intervention_spec"]["node"] = "revenue"
    with pytest.raises(HarnessValidationError, match="categorical primitive"):
        BT.validate_irrelevance(bad)
    bad = copy.deepcopy(spec)
    bad["battery"]["node"] = bad["intervention_spec"]["node"] = "price_now"
    with pytest.raises(HarnessValidationError, match="has children"):
        BT.validate_irrelevance(bad)


def test_bank_sizes_and_placeholder_provenance() -> None:
    for cue in (
        "I.USER.stance",
        "I.ID.ceo",
        "I.HYPE.label",
        "I.HYPE.arch",
        "I.CODE.comment",
    ):
        bank = X.yaml_io.load(BT.bank_path(cue))
        kind = BT.attach_kind(BT.load_cues()[cue]["attach"])
        assert BT.bank_size(bank, kind) >= BT.MIN_BANK, cue
        assert (
            bank["provenance"]["drafted_by"]["kind"] == "llm"
        )  # refused for live runs (D-14)
        assert bank["provenance"]["approved_by"] == []
    pairs = X.yaml_io.load(BT.bank_path("I.ID.ceo"))["pairs"]
    assert len({(e["a"], e["b"]) for e in pairs}) == len(pairs)
    assert all(not re.search(r"\d", e["a"] + e["b"]) for e in pairs)
    # gender cue only: same surname, contrast recorded (§T4.2)
    assert all(e["a"].split()[-1] == e["b"].split()[-1] for e in pairs)
    assert {e["contrast"] for e in pairs} == {"gender"}
    # rhetorical and architecture labels never overlap (§T4.2)
    hype = set(X.yaml_io.load(BT.bank_path("I.HYPE.label"))["entries"])
    arch = set(X.yaml_io.load(BT.bank_path("I.HYPE.arch"))["entries"])
    assert not hype & arch


def test_identity_cues_await_d25() -> None:
    """I.ID.name and I.ID.ceo_ethnicity need validated banks (D-25): no file, a report row."""
    for cue in ("I.ID.name", "I.ID.ceo_ethnicity"):
        assert not BT.bank_path(cue).exists()
    reg = load_registry()
    rows = [
        r
        for r in X.expand_item(reg, reg.items["F1.1.a"], family_listing())
        if r.intervention_id in ("I.ID.name", "I.ID.ceo_ethnicity")
    ]
    assert len(rows) == 6  # three variants × two cues
    for r in rows:
        assert r.spec is None and r.reason == "unimplemented_content"
        assert r.missing == [f"taxonomy/banks/{r.intervention_id}.yaml"]


def test_a_battery_experiment_runs_end_to_end(
    battery_specs: dict[tuple[str, str, str], dict[str, Any]],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    spec = battery_specs[("F1.1.a", "m1", "I.USER.stance")]
    path = tmp_path_factory.mktemp("spec") / "stance.yaml"
    path.write_text(X.yaml_io.dump(spec), encoding="utf-8")
    run = run_slice(path, tmp_path_factory.mktemp("home"))
    labels = {
        (r["model_key"], r["feature_id"]): r
        for r in read_csv(run / "results" / "labels.csv")
    }
    primary = labels[(FAKE_A, "decision_attractive")]
    # a disputed cue (D-27): discovery, so never a bias label
    assert primary["oracle_type"] == "DISCOVERY" and primary["experiment_key"] == (
        "f1_1_a__m1__i_user_stance.v3"
    )
    assert primary["is_bias"] == "false"
    assert labels[(FAKE_A, "agrees_with_stance")]["oracle_type"] == "DISCOVERY"
    # discovery mode runs no control gates (§24.2), as for m0
    assert read_csv(run / "results" / "controls.csv") == []
    assert Path(run / "manifest").is_dir()
