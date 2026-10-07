"""Resume (spec §29, §31A, §36): scientific-state equivalence through ``resume_experiment``."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from financial_llm_harness.core import ids
from financial_llm_harness.execution.runner import (
    InjectedInterruption,
    resume_experiment,
    run_experiment,
)
from financial_llm_harness.storage.artifacts import ArtifactStore
from financial_llm_harness.storage.db import db_path
from tests.harness.helpers import (
    FIXED_CODE_REF,
    GOLDEN_RUN_ID,
    assert_same_files,
    fixed_clock,
    output_files,
    small_spec,
)


def _rows(run_dir: Path, sql: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(str(db_path(run_dir)))
    try:
        return [tuple(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def _selected_response_bytes(run_dir: Path) -> dict[str, bytes | None]:
    store = ArtifactStore(run_dir / "artifacts")
    out: dict[str, bytes | None] = {}
    for trial_id, text in _rows(
        run_dir,
        "SELECT t.trial_id, r.text_artifact FROM trial t LEFT JOIN response r ON r.attempt_id = t.used_attempt_id",
    ):
        out[trial_id] = store.get(text) if text else None
    return out


def _scientific_state(run_dir: Path) -> dict[str, Any]:
    return {
        "trials": sorted(_rows(run_dir, "SELECT trial_id, status FROM trial")),
        "responses": _selected_response_bytes(run_dir),
        "observations": sorted(
            _rows(
                run_dir,
                "SELECT observation_id, trial_id, feature_id, evaluator_version, value_json, status, confidence FROM observation",
            )
        ),
        "analysis": sorted(
            _rows(
                run_dir,
                "SELECT analysis_id, manifest_hash, model_key, feature_id, contrast, analysis_kind, analysis_version, "
                "results_json, label, code_sha FROM analysis",
            )
        ),
    }


def _run(spec: Path, home: Path, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> Any:
    monkeypatch.setenv("HARNESS_HOME", str(home))
    return run_experiment(
        spec,
        "ci_fake",
        yes=True,
        run_id=GOLDEN_RUN_ID,
        clock=fixed_clock,
        code_ref=FIXED_CODE_REF,
        **kw,
    )


def test_resume_equals_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = small_spec(tmp_path)
    full = _run(spec, tmp_path / "full", monkeypatch)
    assert full.status == "COMPLETE" and full.run_dir is not None

    with pytest.raises(InjectedInterruption):
        _run(spec, tmp_path / "cut", monkeypatch, interrupt_after=137)
    cut_dir = next((tmp_path / "cut" / "runs").glob("*_00000000"))
    statuses = dict(
        _rows(cut_dir, "SELECT status, COUNT(*) FROM trial GROUP BY status")
    )
    assert statuses.get("PENDING", 0) > 0, statuses
    assert _rows(cut_dir, "SELECT status FROM run") == [("RUNNING",)]

    # A request in flight when the process died: SUBMITTED, an in-flight attempt, no batch_id.
    conn = sqlite3.connect(str(db_path(cut_dir)))
    try:
        trial_id = conn.execute(
            "SELECT trial_id FROM trial WHERE status = 'PENDING' ORDER BY trial_id LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            "UPDATE trial SET status = 'SUBMITTED' WHERE trial_id = ?", (trial_id,)
        )
        n_prev = conn.execute(
            "SELECT COUNT(*) FROM attempt WHERE trial_id = ?", (trial_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO attempt(attempt_id, trial_id, attempt_no, started_at, transport_status) VALUES (?, ?, ?, ?, 'in_flight')",
            (
                ids.attempt_id(trial_id, n_prev + 1),
                trial_id,
                n_prev + 1,
                "2026-01-01T00:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    res = resume_experiment(
        GOLDEN_RUN_ID, yes=True, clock=fixed_clock, code_ref=FIXED_CODE_REF
    )
    assert res.status == "COMPLETE" and res.exit_code == 0
    assert res.run_dir == cut_dir

    # the in-flight attempt was abandoned and the trial resubmitted with a new attempt
    attempts = _rows(
        cut_dir,
        f"SELECT attempt_no, transport_status FROM attempt WHERE trial_id = '{trial_id}' ORDER BY attempt_no",
    )
    assert attempts[-2:] == [(n_prev + 1, "abandoned"), (n_prev + 2, "complete")]
    used = _rows(
        cut_dir, f"SELECT used_attempt_id FROM trial WHERE trial_id = '{trial_id}'"
    )
    assert used == [(ids.attempt_id(trial_id, n_prev + 2),)]

    # no duplicate trials; equal scientific state; byte-identical outputs
    n, distinct = _rows(
        cut_dir, "SELECT COUNT(*), COUNT(DISTINCT trial_id) FROM trial"
    )[0]
    assert n == distinct
    assert _scientific_state(cut_dir) == _scientific_state(full.run_dir)
    assert_same_files(output_files(cut_dir), output_files(full.run_dir))
    assert _rows(cut_dir, "SELECT resume_count FROM run") == [(1,)]


def test_resume_without_yes_does_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = small_spec(tmp_path)
    with pytest.raises(InjectedInterruption):
        _run(spec, tmp_path / "cut", monkeypatch, interrupt_after=50)
    res = resume_experiment(
        GOLDEN_RUN_ID, yes=False, clock=fixed_clock, code_ref=FIXED_CODE_REF
    )
    assert res.exit_code == 0 and res.status is None
    cut_dir = next((tmp_path / "cut" / "runs").glob("*_00000000"))
    assert _rows(cut_dir, "SELECT resume_count, status FROM run") == [(0, "RUNNING")]
