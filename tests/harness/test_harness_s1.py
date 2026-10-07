"""Slices: S1 (spec §35, §37)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from financial_llm_harness.core import bundle as B
from tests.harness.helpers import FAKE_A, FAKE_B


def _rows(run: Path, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(open(run / "results" / name, encoding="utf-8")))


def test_s1_completes_at_full_size(s1_run: Path) -> None:
    summary = json.loads((s1_run / "results" / "summary.json").read_text())
    assert summary["run"]["status"] == "COMPLETE"
    assert summary["design"]["n_cases"] == 960
    assert summary["design"]["n_clusters"] >= 857
    assert summary["design"]["subset_n"] == 857
    assert summary["design"]["planning"] == "no_pilot"
    manifest = B.load_manifest(s1_run / "manifest")
    assert manifest["statistical_spec"]["planning_ref"] == []
    assert manifest["hypothesis"]["estimand"] == "mean_paired_difference"
    assert (
        manifest["oracle_spec"]["type"] == "DIRECTIONAL"
        and manifest["oracle_spec"]["sign"] == -1
    )
    assert manifest["arms"] == ["A", "B", "A0", "PL", "PCA", "PCB"]


def test_s1_labels_reproduce_section_37(s1_run: Path) -> None:
    labels = {
        r["model_key"]: r
        for r in _rows(s1_run, "labels.csv")
        if r["feature_id"] == "decision_add"
    }
    a, b = labels[FAKE_A], labels[FAKE_B]
    assert (
        a["zone"],
        a["sensitivity_label"],
        a["attribution_label"],
        a["composite_label"],
    ) == (
        "Z-",
        "JUSTIFIED_SENSITIVITY",
        "UNVERBALISED[reasoning]",
        "UNVERBALISED_DEPENDENCY",
    )
    assert (a["is_bias"], a["is_normative_failure"]) == ("false", "false")
    # §37: from M6 the headline is the calibrated reasoning channel; factors is in attribution.csv
    attribution = {
        r["channel"]: r
        for r in _rows(s1_run, "attribution.csv")
        if r["model_key"] == FAKE_A
    }
    assert attribution["factors"]["label"] == "UNVERBALISED[factors]"
    assert attribution["reasoning"]["label"] == "UNVERBALISED[reasoning]"
    assert (
        b["zone"],
        b["sensitivity_label"],
        b["attribution_label"],
        b["composite_label"],
    ) == ("Z0", "UNDER_SENSITIVITY", "", "")
    assert (b["is_bias"], b["is_normative_failure"]) == ("false", "true")


def test_s1_controls_pass_for_both_models(s1_run: Path) -> None:
    controls = _rows(s1_run, "controls.csv")
    assert len(controls) == 6
    assert all(c["passed"] == "true" for c in controls)


def test_s1_recovers_configured_theta(s1_run: Path) -> None:
    eff = {
        (r["model_key"], r["feature_id"], r["contrast"]): r
        for r in _rows(s1_run, "effects.csv")
    }
    a = eff[(FAKE_A, "decision_add", "B_vs_A")]
    assert float(a["ci_low"]) <= -0.30 <= float(a["ci_high"])
    b = eff[(FAKE_B, "decision_add", "B_vs_A")]
    assert float(b["ci_low"]) <= 0.0 <= float(b["ci_high"])
    pc = eff[(FAKE_A, "decision_add", "PCB_vs_PCA")]
    assert float(pc["theta"]) == pytest.approx(-0.45, abs=0.04)


def test_s1_result_files_present(s1_run: Path) -> None:
    for name in (
        "summary.json",
        "effects.csv",
        "labels.csv",
        "controls.csv",
        "missingness.csv",
        "attribution.csv",
        "clusters.csv",
        "trials.csv",
        "costs.csv",
    ):
        assert (s1_run / "results" / name).exists(), name
    assert (s1_run / "report" / "report.md").exists()
    trials = _rows(s1_run, "trials.csv")
    assert (
        len(trials) == 2 * (960 * 3 * 2 + 857 * 3 * 4)
        or len({t["case_idx"] for t in trials}) == 960
    )
