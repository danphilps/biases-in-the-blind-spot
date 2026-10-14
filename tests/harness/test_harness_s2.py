"""Slices: S2 `transcript_summary` (spec §35 S2, §36 Slices, M6)."""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from financial_llm_harness.core import bundle as B
from financial_llm_harness.families.call_analysis import family as CA
from tests.harness.helpers import FAKE_A, FAKE_B, ROOT, read_csv, run_slice

S2_SPEC = ROOT / "experiments" / "financial" / "specs" / "transcript_summary.yaml"
GUIDANCE = "s_guidance_cut"


@pytest.fixture(scope="module")
def s2_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return run_slice(S2_SPEC, tmp_path_factory.mktemp("s2home"))


def test_s2_invariance_primary_and_controls(s2_run: Path) -> None:
    manifest = B.load_manifest(s2_run / "manifest")
    assert manifest["scenario_spec"]["n_cases"] == 960
    assert manifest["statistical_spec"]["planning_ref"] == []
    assert manifest["controls"]["subset_n"] == 857
    assert manifest["oracle_spec"]["type"] == "INVARIANCE"
    labels = {r["model_key"]: r for r in read_csv(s2_run / "results" / "labels.csv")}
    assert labels[FAKE_A]["oracle_type"] == "INVARIANCE"
    # fake--a has theta -0.3 (§20.11): spurious sensitivity to the position of the cut
    assert labels[FAKE_A]["sensitivity_label"] == "SPURIOUS_SENSITIVITY"
    assert labels[FAKE_B]["sensitivity_label"] == "INVARIANT_AS_REQUIRED"
    controls = read_csv(s2_run / "results" / "controls.csv")
    assert {c["control"] for c in controls} == {"null", "placebo", "positive"}
    assert all(c["passed"] == "true" for c in controls)


def test_s2_rule_decides_true_and_only_non_matches_reach_the_evaluator(
    s2_run: Path,
) -> None:
    conn = sqlite3.connect(s2_run / "db" / "run.sqlite")
    rows = conn.execute(
        "SELECT trial_id, evaluator_version, value_json FROM observation "
        "WHERE feature_id = 'includes_guidance_cut'"
    ).fetchall()
    rule = {t: v for t, ev, v in rows if ev == "output_inclusion_rule@2"}
    semantic = {t: v for t, ev, v in rows if ev != "output_inclusion_rule@2"}
    assert rule and semantic
    assert set(rule.values()) == {"1"}  # the rule only ever decides TRUE
    assert not set(rule) & set(semantic)  # a decided trial never reaches the evaluator
    eval_obs = {
        r[0] for r in conn.execute("SELECT DISTINCT observation_id FROM eval_call")
    }
    assert len(eval_obs) == len(semantic)
    valid = {
        r[0]
        for r in conn.execute("SELECT trial_id FROM trial WHERE status = 'SUCCESS'")
    }
    assert (
        set(rule) | set(semantic) == valid
    )  # every parsed response has one observation
    conn.close()


def test_s2_arms_differ_only_by_position_order_or_removal(s2_run: Path) -> None:
    rows = B.read_jsonl(s2_run / "manifest" / B.RENDERINGS_FILE)
    by = {(r["case_idx"], r["arm"], r["paraphrase_idx"]): r["user"] for r in rows}

    def items(user: str) -> list[str]:
        return [ln for ln in user.splitlines() if ln.startswith("- ")]

    checked = Counter[str]()
    for (case, arm, p), user in by.items():
        if arm != "A" or p != 0:
            continue
        a = items(user)
        assert len(a) == 12
        g = a[1]  # the guidance cut, position 2 in arm A
        others = [x for x in a if x != g]
        b = items(by[(case, "B", 0)])
        assert (
            b == others[:10] + [g] + others[10:]
        )  # position 11 in arm B, all else equal
        checked["B"] += 1
        if (case, "PL", 0) in by:
            pl = items(by[(case, "PL", 0)])
            assert sorted(pl) == sorted(a) and pl != a
            assert sum(x != y for x, y in zip(a, pl, strict=True)) == 2  # one swap
            pcb = items(by[(case, "PCB", 0)])
            assert pcb == others  # the cut removed, nothing else
            assert items(by[(case, "PCA", 0)]) == a
            checked["controls"] += 1
    assert checked["B"] == 960 and checked["controls"] == 857


def test_s2_bank_and_block_nodes_agree() -> None:
    bank = CA.item_bank()
    assert [e["type"] for e in bank] == list(CA.TYPES)
    assert [int(e["rank"]) for e in bank] == list(range(1, 13))
    assert bank[0]["type"] == "guidance_cut"
    for e in bank:
        assert len(e["sentences"]) >= 10  # §35 S2: at least ten sentences per item type
        for s in e["sentences"]:
            assert s.count("{v1}") == 1 and s.count("{v2}") == 1
            literal = s.replace("{v1}", "").replace("{v2}", "")
            assert not any(ch.isdigit() for ch in literal)  # every figure is a slot
            if e["type"] != "guidance_cut":
                assert "guidance" not in s.lower()
    assert set(CA.immaterial_types()) == {"office_relocation", "sustainability_report"}
    assert GUIDANCE in CA.BLOCK_NODES[CA.BLOCK]


def test_s2_inclusion_rule_reads_the_whole_requested_output() -> None:
    """CR-9: a cut reported only in the summary is included, not an omission."""
    from financial_llm_harness.evaluation.rules import FeatureContext, extract

    feature = {
        "extractor": "output_inclusion_rule",
        "fields": ["summary", "key_points"],
    }
    ctx = FeatureContext(arm="B")

    def run(summary: str, points: list[str]) -> tuple[object, str]:
        ob = extract(feature, {"summary": summary, "key_points": points}, ctx)
        return ob.value, ob.status

    in_summary = "Results were mixed. Management cut full-year guidance."
    assert run(in_summary, ["Segment revenue grew."]) == (1, "OK")
    assert run("Results were mixed.", ["Revenue guidance was lowered."]) == (1, "OK")
    # guidance and a cut word in different sentences is not a match: the evaluator decides
    split = "Guidance was discussed. Capex was cut."
    assert run(split, ["Segment revenue grew."]) == (None, "DEFERRED")
    assert run("Results were mixed.", ["Segment revenue grew."]) == (None, "DEFERRED")
    # the version-1 rule still reads key_points only, so stored runs re-evaluate unchanged
    v1 = {"extractor": "key_points_rule", "field": "key_points"}
    ob = extract(v1, {"summary": in_summary, "key_points": ["x"]}, ctx)
    assert (ob.value, ob.status) == (None, "DEFERRED")


def test_s2_semantic_channel_text_holds_the_whole_requested_output() -> None:
    from financial_llm_harness.evaluation.semantic import _channel_text

    feature = {"extractor": "semantic_inclusion", "fields": ["summary", "key_points"]}
    parsed = {"summary": "S.", "key_points": ["k"]}
    text = _channel_text(feature, parsed, None, None, "none")
    assert text is not None and '"summary": "S."' in text and '"k"' in text
