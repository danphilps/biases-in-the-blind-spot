"""Replay (spec §31, §36): ``--reevaluate`` is byte-identical; ``--reexecute`` is a replication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from financial_llm_harness.cli import main
from financial_llm_harness.core.code_ref import package_dir
from financial_llm_harness.errors import FrozenInputDrift
from financial_llm_harness.reporting.replay import reevaluate, reexecute
from financial_llm_harness.storage.db import db_path
from tests.harness.helpers import (
    FIXED_CODE_REF,
    GOLDEN_RUN_ID,
    assert_same_files,
    fixed_clock,
    output_files,
    run_small,
)

PARSER = package_dir() / "evaluation" / "parse.py"


def test_reevaluate_is_byte_identical(tmp_path: Path, harness_home: Path) -> None:
    run_dir = run_small(tmp_path)
    before = output_files(run_dir)
    reevaluate(GOLDEN_RUN_ID, code_ref=FIXED_CODE_REF, clock=fixed_clock)
    assert_same_files(output_files(run_dir), before)


def test_cli_reevaluate_with_computed_code_identity(
    tmp_path: Path, harness_home: Path
) -> None:
    run_dir = run_small(tmp_path, inject=False)
    before = output_files(run_dir)
    assert main(["replay", GOLDEN_RUN_ID[:8], "--reevaluate"]) == 0
    assert_same_files(output_files(run_dir), before)


def test_reevaluate_refuses_after_parser_edit(
    tmp_path: Path, harness_home: Path
) -> None:
    run_dir = run_small(tmp_path, inject=False)
    before = output_files(run_dir)
    db_before = db_path(run_dir).read_bytes()
    original = PARSER.read_bytes()
    try:
        PARSER.write_bytes(original + b"\n# edited for a replay drift test\n")
        with pytest.raises(FrozenInputDrift) as exc:
            reevaluate(GOLDEN_RUN_ID)
        assert exc.value.exit_code == 2
        assert any("harness_tree_sha256" in m for m in exc.value.mismatches)
        assert main(["replay", GOLDEN_RUN_ID, "--reevaluate"]) == 2
    finally:
        PARSER.write_bytes(original)
    # refused before writing anything
    assert db_path(run_dir).read_bytes() == db_before
    assert_same_files(output_files(run_dir), before)


def test_reexecute_reports_replication(tmp_path: Path, harness_home: Path) -> None:
    run_small(tmp_path)
    new_id = "11111111-0000-4000-8000-000000000002"
    res = reexecute(
        GOLDEN_RUN_ID,
        yes=True,
        new_run_id=new_id,
        clock=fixed_clock,
        code_ref=FIXED_CODE_REF,
    )
    assert res.status == "COMPLETE" and res.run_dir is not None
    assert res.run_dir.name.endswith("_11111111")
    rep = json.loads((res.run_dir / "replication.json").read_text())
    assert rep["original_run_id"] == GOLDEN_RUN_ID
    assert rep["replication_run_id"] == new_id
    for m in rep["models"]:
        # the fake adapter is a pure function of the trial seed, so the replication is exact
        assert m["theta_difference"] == 0.0
        assert m["ci"] == [0.0, 0.0]
        assert m["n_clusters_paired"] > 0
        assert m["zone_agrees"] and m["sensitivity_label_agrees"]
        assert m["attribution_label_agrees"]
    assert "Replication of run" in (res.run_dir / "replication.md").read_text()


def test_reexecute_needs_yes(tmp_path: Path, harness_home: Path) -> None:
    run_small(tmp_path)
    res = reexecute(GOLDEN_RUN_ID, yes=False, code_ref=FIXED_CODE_REF)
    assert res.run_id is None and res.exit_code == 0
    assert len([p for p in (harness_home / "runs").iterdir() if p.is_dir()]) == 1
