"""Slices: S3 `covariance_choice` (spec §35, §36, D-21)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.harness.helpers import (
    ROOT,
    assert_level_slice,
    pcb_note_only_in_pcb,
    read_csv,
    run_slice,
)

S3_SPEC = ROOT / "experiments" / "financial" / "specs" / "covariance_choice.yaml"


@pytest.fixture(scope="module")
def s3_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return run_slice(S3_SPEC, tmp_path_factory.mktemp("s3home"))


def test_s3_level_primary_controls_and_size(s3_run: Path) -> None:
    # §35 sizes: primary LEVEL at δ 0.15 needs 69; A0 and PL at margin 0.15 need 381; n_cases 420
    assert_level_slice(s3_run, "well_conditioned_choice", 0.9, 420, 381)
    report = json.loads((s3_run / "manifest" / "corpus_report.json").read_text())
    power = report["power"]
    assert power["primary"]["n_required"] == 69
    assert {c["contrast"]: c["n_required"] for c in power["controls"]} == {
        "A0_vs_A": 381,
        "PL_vs_A": 381,
        "PCB_vs_PCA": 88,
    }
    assert power["n_clusters"] >= 381


def test_s3_intervention_and_positive_control(s3_run: Path) -> None:
    from financial_llm_harness.core import bundle as B

    cases = B.read_jsonl(s3_run / "manifest" / B.CASES_FILE)
    for c in cases[:50]:
        a, b = c["arms"]["A"], c["arms"]["B"]
        assert (a["t_over_n"], b["t_over_n"]) == (5.0, 0.5)
        assert a["n_obs"] == round(5.0 * a["n_assets"]) and b["n_obs"] == round(
            0.5 * b["n_assets"]
        )
        assert sorted(c["diff"]) == ["n_obs", "t_over_n"]
        if "PCA" in c["arms"]:
            # PCA and PCB hold T/N = 0.5 as in B; PCB alone adds the rank-deficiency sentence
            assert c["arms"]["PCA"]["t_over_n"] == c["arms"]["PCB"]["t_over_n"] == 0.5
    pcb_note_only_in_pcb(
        s3_run, "The sample covariance matrix of these returns is rank-deficient."
    )


def test_s3_secondary_is_discovery(s3_run: Path) -> None:
    effects = read_csv(s3_run / "results" / "effects.csv")
    sec = [e for e in effects if e["analysis_role"] == "secondary"]
    assert sec and all(e["zone"] in ("Z+", "Z-", "Z0", "Zs", "Zi") for e in sec)
    assert all(e["delta"] == "0.15" for e in sec)
