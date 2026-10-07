"""Slices: S4 `temporal_norm_design` (spec §35, §36, D-21)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.helpers import (
    ROOT,
    assert_level_slice,
    pcb_note_only_in_pcb,
    run_slice,
)

S4_SPEC = ROOT / "experiments" / "financial" / "specs" / "temporal_norm_design.yaml"
DESCRIPTIVE = "descriptive study of historical relationships"
BACKTEST = "signal for a walk-forward backtest"


@pytest.fixture(scope="module")
def s4_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return run_slice(S4_SPEC, tmp_path_factory.mktemp("s4home"))


def test_s4_level_primary_controls_and_size(s4_run: Path) -> None:
    assert_level_slice(s4_run, "point_in_time", 0.95, 960, 857)


def test_s4_framing_change_and_positive_control(s4_run: Path) -> None:
    from financial_llm_harness.core import bundle as B

    cases = B.read_jsonl(s4_run / "manifest" / B.CASES_FILE)
    for c in cases[:50]:
        assert (c["arms"]["A"]["purpose"], c["arms"]["B"]["purpose"]) == (
            DESCRIPTIVE,
            BACKTEST,
        )
        assert c["diff"] == ["purpose"]
        if "PCA" in c["arms"]:
            assert c["arms"]["PCA"]["purpose"] == DESCRIPTIVE
            assert c["arms"]["PCB"]["purpose"] == BACKTEST
    rows = B.read_jsonl(s4_run / "manifest" / B.RENDERINGS_FILE)
    a = next(r for r in rows if r["arm"] == "A" and r["paraphrase_idx"] == 0)
    b = next(
        r
        for r in rows
        if r["arm"] == "B"
        and r["case_idx"] == a["case_idx"]
        and r["paraphrase_idx"] == 0
    )
    assert a["user"].replace(DESCRIPTIVE, BACKTEST) == b["user"]
    pcb_note_only_in_pcb(
        s4_run, "Every statistic must use only data available on its date."
    )
