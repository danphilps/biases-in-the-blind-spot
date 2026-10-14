"""Comparison (spec §32A.7, §36)."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.reporting.compare import compare_runs
from tests.harness.helpers import FAKE_A, FAKE_B, run_small

RID1 = "11111111-0000-4000-8000-000000000001"
RID2 = "22222222-0000-4000-8000-000000000002"
RID3 = "33333333-0000-4000-8000-000000000003"


def day(d: int) -> datetime:
    return datetime(2026, 1, d, tzinfo=UTC)


@pytest.fixture
def two_runs(tmp_path: Path, harness_home: Path) -> tuple[Path, Path]:
    r1 = run_small(tmp_path, RID1, name="a.yaml")
    r2 = run_small(
        tmp_path, RID2, name="b.yaml", experiment_key="f1_1_a__m1__i_f_momentum.v9"
    )
    return r1, r2


def test_matrix_rows_resolve_into_labels(
    two_runs: tuple[Path, Path], harness_home: Path
) -> None:
    out = compare_runs([RID1, RID2], "fake-pair", clock=lambda: day(1))
    assert out.name == "2026-01-01_fake-pair"
    assert sorted(p.name for p in out.iterdir()) == [
        "comparison.html",
        "comparison.md",
        "matrix.csv",
        "matrix.svg",
    ]
    with open(out / "matrix.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["model_key"], r["experiment_key"]) for r in rows] == [
        (FAKE_A, "calibration_s1__m1__i_f_momentum.v1"),
        (FAKE_A, "f1_1_a__m1__i_f_momentum.v9"),
        (FAKE_B, "calibration_s1__m1__i_f_momentum.v1"),
        (FAKE_B, "f1_1_a__m1__i_f_momentum.v9"),
    ]
    run_dirs = {RID1: two_runs[0], RID2: two_runs[1]}
    for r in rows:
        with open(run_dirs[r["run_id"]] / "results" / "labels.csv", newline="") as fh:
            labels = {x["row_id"]: x for x in csv.DictReader(fh)}
        assert labels[r["labels_row_id"]]["model_key"] == r["model_key"]
        assert labels[r["labels_row_id"]]["sensitivity_label"] == r["sensitivity_label"]
    html = (out / "comparison.html").read_text()
    assert "http" not in html.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "<details><summary>Data</summary>" in html


def test_same_name_same_date_exits_2_and_rerun_is_byte_identical(
    two_runs: tuple[Path, Path], harness_home: Path
) -> None:
    first = compare_runs([RID1, RID2], "repeat", clock=lambda: day(1))
    with pytest.raises(HarnessValidationError) as exc:
        compare_runs([RID1, RID2], "repeat", clock=lambda: day(1))
    assert exc.value.exit_code == 2
    second = compare_runs([RID1, RID2], "repeat", clock=lambda: day(2))
    for name in ("matrix.csv", "matrix.svg", "comparison.md", "comparison.html"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_runs_differing_in_delta_exit_2(tmp_path: Path, harness_home: Path) -> None:
    run_small(tmp_path, RID1, name="a.yaml")
    run_small(tmp_path, RID3, name="c.yaml", hypothesis__delta=0.15)
    with pytest.raises(HarnessValidationError) as exc:
        compare_runs([RID1, RID3], "delta", clock=lambda: day(1))
    assert exc.value.exit_code == 2 and "not comparable" in str(exc.value)
    assert not (harness_home / "comparisons").exists() or not any(
        (harness_home / "comparisons").iterdir()
    )


def test_bad_name_exits_2_before_writing(
    two_runs: tuple[Path, Path], harness_home: Path
) -> None:
    for bad in ("Has_Upper", "x" * 49, "under_score"):
        with pytest.raises(HarnessValidationError):
            compare_runs([RID1, RID2], bad, clock=lambda: day(1))
    assert not (harness_home / "comparisons").exists()
