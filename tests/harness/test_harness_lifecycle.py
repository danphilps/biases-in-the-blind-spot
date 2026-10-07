"""Lifecycle (spec §31A, §32, §0.7)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from financial_llm_harness.errors import HarnessError, HarnessValidationError, LockHeld
from financial_llm_harness.execution.runner import resolve_run, run_experiment
from financial_llm_harness.storage.archive import WriterLock
from tests.harness.helpers import ROOT, small_spec


def test_dry_run_and_missing_yes_create_no_run_dir(
    tmp_path: Path, harness_home: Path
) -> None:
    spec = small_spec(tmp_path)
    res = run_experiment(spec, "ci_fake", yes=False)
    assert res.exit_code == 0 and res.run_dir is None and "plan:" in res.plan_text
    res = run_experiment(spec, "ci_fake", yes=True, dry_run=True)
    assert res.run_dir is None
    assert not (harness_home / "runs").exists()


def test_run_id_resolution_and_lock(tmp_path: Path, harness_home: Path) -> None:
    spec = small_spec(tmp_path)
    spec2 = small_spec(
        tmp_path, name="other.yaml", experiment_key="f1_1_a__m1__i_f_momentum.v9"
    )
    r1 = run_experiment(
        spec, "ci_fake", yes=True, run_id="abcdef01-0000-4000-8000-000000000001"
    )
    r2 = run_experiment(
        spec2, "ci_fake", yes=True, run_id="abcdef01-1111-4000-8000-000000000002"
    )
    assert r1.status == "COMPLETE" and r1.exit_code == 0
    assert resolve_run("abcdef01-0000-4000-8000-000000000001") == r1.run_dir
    assert resolve_run("abcdef0100") == r1.run_dir
    with pytest.raises(HarnessValidationError) as exc:
        resolve_run("abcdef01")
    assert exc.value.exit_code == 2
    with pytest.raises(HarnessError) as exc2:
        resolve_run("ffffffff")
    assert exc2.value.exit_code == 1
    assert r1.run_dir is not None and r2.run_dir is not None
    run_json = json.loads((r1.run_dir / "run.json").read_text())
    assert list(run_json) == [
        "schema_version",
        "run_id",
        "experiment_key",
        "manifest_hash",
        "started_at",
    ]
    # a second process on the same run exits 1 on the lock
    with WriterLock(r1.run_dir):
        code = subprocess.run(
            [
                sys.executable,
                "-m",
                "financial_llm_harness.cli",
                "render",
                "abcdef01-0000-4000-8000-000000000001",
            ],
            cwd=ROOT,
            env={**_env(harness_home)},
            capture_output=True,
            text=True,
        )
        assert code.returncode == 1
        with pytest.raises(LockHeld):
            WriterLock(r1.run_dir).acquire()


def _env(home: Path) -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["HARNESS_HOME"] = str(home)
    return env


def test_fake_run_does_not_import_torch_or_chat_limiter(
    tmp_path: Path, harness_home: Path
) -> None:
    spec = small_spec(tmp_path)
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from financial_llm_harness.execution.runner import run_experiment\n"
        f"r = run_experiment(Path({str(spec)!r}), 'ci_fake', yes=True)\n"
        "assert r.status == 'COMPLETE', r.status\n"
        "print('torch' in sys.modules, 'chat_limiter' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=_env(harness_home),
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False False"


def test_cli_run_spec(tmp_path: Path, harness_home: Path) -> None:
    from financial_llm_harness.cli import main

    spec = small_spec(tmp_path)
    assert main(["run", str(spec), "--panel", "ci_fake"]) == 0
    assert not (harness_home / "runs").exists()
    assert main(["run", str(spec), "--panel", "ci_fake", "--yes"]) == 0
    runs = list((harness_home / "runs").iterdir())
    assert any(r.is_dir() for r in runs)


def test_submitted_trial_without_batch_id_is_resubmitted_on_resume(
    tmp_path: Path, harness_home: Path
) -> None:
    import sqlite3

    from financial_llm_harness.core import ids
    from financial_llm_harness.execution.runner import (
        InjectedInterruption,
        resume_experiment,
    )
    from financial_llm_harness.storage.db import db_path
    from tests.harness.helpers import FIXED_CODE_REF, GOLDEN_RUN_ID, fixed_clock

    spec = small_spec(tmp_path)
    with pytest.raises(InjectedInterruption):
        run_experiment(
            spec,
            "ci_fake",
            yes=True,
            run_id=GOLDEN_RUN_ID,
            clock=fixed_clock,
            code_ref=FIXED_CODE_REF,
            interrupt_after=25,
        )
    run_dir = resolve_run(GOLDEN_RUN_ID)
    conn = sqlite3.connect(str(db_path(run_dir)))
    try:
        trial_id = conn.execute(
            "SELECT trial_id FROM trial WHERE status = 'PENDING' ORDER BY trial_id LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            "UPDATE trial SET status = 'SUBMITTED' WHERE trial_id = ?", (trial_id,)
        )
        conn.execute(
            "INSERT INTO attempt(attempt_id, trial_id, attempt_no, started_at, transport_status, batch_id) "
            "VALUES (?, ?, 1, '2026-01-01T00:00:00Z', 'in_flight', NULL)",
            (ids.attempt_id(trial_id, 1), trial_id),
        )
        conn.commit()
    finally:
        conn.close()
    res = resume_experiment(
        GOLDEN_RUN_ID, yes=True, clock=fixed_clock, code_ref=FIXED_CODE_REF
    )
    assert res.status == "COMPLETE"
    conn = sqlite3.connect(str(db_path(run_dir)))
    try:
        rows = conn.execute(
            "SELECT attempt_no, transport_status FROM attempt WHERE trial_id = ? ORDER BY attempt_no",
            (trial_id,),
        ).fetchall()
        used = conn.execute(
            "SELECT used_attempt_id, status FROM trial WHERE trial_id = ?", (trial_id,)
        ).fetchone()
    finally:
        conn.close()
    assert [tuple(r) for r in rows] == [(1, "abandoned"), (2, "complete")]
    assert used[0] == ids.attempt_id(trial_id, 2) and used[1] in (
        "SUCCESS",
        "PARSE_FAILURE",
    )
