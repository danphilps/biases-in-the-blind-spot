"""Data policy (spec §0.9, §30.1, §36); the programme clause arrives at M10."""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pytest

from financial_llm_harness.execution import runner
from financial_llm_harness.storage.db import db_path
from tests.harness.helpers import run_small

EXPERIMENT_KEY = "f1_1_a__m1__i_f_momentum.v1"


def _summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "results" / "summary.json").read_text())


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )


def test_confidential_run_in_git_work_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git(repo, "init", "-q").returncode == 0
    monkeypatch.setenv("HARNESS_HOME", str(repo))
    with caplog.at_level(
        logging.WARNING, logger="financial_llm_harness.execution.runner"
    ):
        run_dir = run_small(tmp_path, data_classification="confidential")
    assert _summary(run_dir)["output_policy"] == {
        "publishable": False,
        "reason": "non_synthetic_default",
    }
    assert (run_dir / ".gitignore").read_text() == "*\n"
    readme = (run_dir / "README.md").read_text()
    assert "Data classification: confidential." in readme
    assert "Not publishable: non_synthetic_default." in readme
    for page in ("report.md", "report.html"):
        assert (
            "Not publishable: non_synthetic_default."
            in (run_dir / "report" / page).read_text()
        )
    # the work-tree warning is printed and recorded
    assert any("inside a git work tree" in r.getMessage() for r in caplog.records)
    conn = sqlite3.connect(str(db_path(run_dir)))
    try:
        kinds = [r[0] for r in conn.execute("SELECT kind FROM event").fetchall()]
    finally:
        conn.close()
    assert "non_synthetic_in_git_work_tree" in kinds
    # git ignores every file of the run directory: raw responses, models/, manifest/, db/, reports
    files = [p for p in run_dir.rglob("*") if p.is_file()]
    assert any("models" in p.parts for p in files)
    assert any("manifest" in p.parts for p in files)
    assert any("artifacts" in p.parts for p in files)
    for p in files:
        rel = p.relative_to(repo).as_posix()
        assert _git(repo, "check-ignore", "-q", rel).returncode == 0, rel
    # the index row of a non-publishable run shows no labels
    index = json.loads((repo / "runs" / "index.json").read_text())
    row = next(
        r for r in index["runs"] if r["run_id"] == _summary(run_dir)["run"]["run_id"]
    )
    assert row["publishable"] is False
    assert all(m["headline_label"] is None for m in row["models"])
    line = next(
        x
        for x in (repo / "runs" / "INDEX.md").read_text().splitlines()
        if "00000000" in x
    )
    assert "SENSITIVITY" not in line and "UNVERBALISED" not in line


def test_user_output_policy_makes_it_publishable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harness_home: Path
) -> None:
    policy = runner.load_provider_policy()
    policy["publish"] = {"licensed": [], "confidential": [EXPERIMENT_KEY]}
    monkeypatch.setattr(runner, "load_provider_policy", lambda: policy)
    run_dir = run_small(tmp_path, data_classification="confidential")
    assert _summary(run_dir)["output_policy"] == {
        "publishable": True,
        "reason": "user_output_policy",
    }
    assert (run_dir / ".gitignore").read_text() == "*\n"
    readme = (run_dir / "README.md").read_text()
    assert "Data classification: confidential." in readme
    assert "Not publishable" not in readme
    # the recorded decision, not the live policy, drives rendering (§2.15)
    monkeypatch.setattr(runner, "load_provider_policy", lambda: {"publish": {}})
    from financial_llm_harness.reporting.render import render_run

    render_run(run_dir)
    assert _summary(run_dir)["output_policy"]["publishable"] is True


def test_synthetic_run_has_no_gitignore(tmp_path: Path, harness_home: Path) -> None:
    run_dir = run_small(tmp_path)
    assert not (run_dir / ".gitignore").exists()
    assert _summary(run_dir)["output_policy"] == {
        "publishable": True,
        "reason": "synthetic",
    }
    assert "Data classification" not in (run_dir / "README.md").read_text()


def test_programme_with_a_confidential_run(tmp_path: Path, harness_home: Path) -> None:
    import shutil

    from financial_llm_harness.core import yaml_io
    from financial_llm_harness.execution.programme import run_programme
    from tests.harness.helpers import (
        FIXED_CODE_REF,
        ROOT,
        S1_M2_SPEC,
        fixed_clock,
        small_spec,
    )

    s1 = small_spec(tmp_path, name="m1.yaml")
    s2 = small_spec(
        tmp_path, name="m2.yaml", base=S1_M2_SPEC, data_classification="confidential"
    )
    panel = tmp_path / "panel.yaml"
    shutil.copyfile(
        ROOT / "experiments" / "financial" / "panels" / "ci_fake.yaml", panel
    )
    prog = tmp_path / "prog.yaml"
    prog.write_text(
        yaml_io.dump(
            {
                "schema_version": 1,
                "programme_key": "policy_mix",
                "panel": str(panel),
                "select": {"specs": [str(s1), str(s2)], "generated": None},
                "budget": {"max_cost_usd": 1.0, "max_calls": 100_000},
                "on_experiment_failure": "continue",
            }
        )
    )
    res = run_programme(prog, yes=True, clock=fixed_clock, code_ref=FIXED_CODE_REF)
    assert res.exit_code == 0 and res.programme_dir is not None
    assert (res.programme_dir / ".gitignore").read_text() == "*\n"
    summary = json.loads((res.programme_dir / "results" / "summary.json").read_text())
    assert summary["output_policy"] == {
        "publishable": False,
        "reason": "non_synthetic_default",
    }
    assert (
        "Not publishable" in (res.programme_dir / "report" / "programme.md").read_text()
    )
