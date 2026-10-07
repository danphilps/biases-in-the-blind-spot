"""Budget, cap override and pricing (spec §30, §36)."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.execution import budget as BU
from financial_llm_harness.execution import runner
from financial_llm_harness.storage.db import db_path
from tests.harness.helpers import (
    FAKE_A,
    FIXED_CODE_REF,
    GOLDEN_RUN_ID,
    fixed_clock,
    small_spec,
)


def _rows(run_dir: Path, sql: str, *args: Any) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(str(db_path(run_dir)))
    try:
        return [tuple(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def _run(spec: Path) -> Any:
    return runner.run_experiment(
        spec,
        "ci_fake",
        yes=True,
        run_id=GOLDEN_RUN_ID,
        clock=fixed_clock,
        code_ref=FIXED_CODE_REF,
    )


def test_cap_breach_pauses_with_whole_blocks_and_resume_override(
    tmp_path: Path, harness_home: Path
) -> None:
    spec = small_spec(tmp_path, budget__max_calls=200)
    res = _run(spec)
    assert res.status == "PAUSED_BUDGET" and res.exit_code == 3
    run_dir = res.run_dir
    n_calls = _rows(run_dir, "SELECT COUNT(*) FROM response")[0][0]
    assert 0 < n_calls <= 200
    # blocks are atomic: every block is either fully executed or not started
    blocks: dict[str, set[str]] = defaultdict(set)
    for block_id, status in _rows(run_dir, "SELECT block_id, status FROM trial"):
        blocks[block_id].add("PENDING" if status == "PENDING" else "DONE")
    assert all(len(v) == 1 for v in blocks.values())
    assert _rows(run_dir, "SELECT status FROM run") == [("PAUSED_BUDGET",)]
    # resume with a cap override: stored on the run row and as an event, then completes
    res2 = runner.resume_experiment(
        GOLDEN_RUN_ID,
        yes=True,
        clock=fixed_clock,
        code_ref=FIXED_CODE_REF,
        max_calls=100_000,
        max_cost_usd=10.0,
    )
    assert res2.status == "COMPLETE" and res2.exit_code == 0
    override = json.loads(_rows(run_dir, "SELECT budget_override_json FROM run")[0][0])
    assert override == {"max_calls": 100_000, "max_cost_usd": 10.0}
    events = [
        json.loads(p)
        for (p,) in _rows(
            run_dir, "SELECT payload_json FROM event WHERE kind = 'budget_override'"
        )
    ]
    assert events == [override]
    assert "Budget override" in (run_dir / "report" / "report.md").read_text()


def test_cli_resume_flags(tmp_path: Path, harness_home: Path) -> None:
    from financial_llm_harness.cli import main

    # the CLI computes the code identity, so this run is made without injection
    res = runner.run_experiment(
        small_spec(tmp_path, budget__max_calls=100),
        "ci_fake",
        yes=True,
        run_id=GOLDEN_RUN_ID,
    )
    assert res.exit_code == 3
    assert (
        main(["run", "--resume", GOLDEN_RUN_ID[:8], "--max-calls", "120", "--yes"]) == 3
    )
    assert res.run_dir is not None
    override = json.loads(
        _rows(res.run_dir, "SELECT budget_override_json FROM run")[0][0]
    )
    assert override["max_calls"] == 120


def test_missing_usage_is_estimated_and_flagged(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    real_make = runner.make_adapter

    class NoUsage:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def capabilities(self) -> Any:
            return self.inner.capabilities()

        async def run(self, request: Any) -> Any:
            return replace(await self.inner.run(request), usage=None)

    monkeypatch.setattr(
        runner,
        "make_adapter",
        lambda m, e: NoUsage(real_make(m, e))
        if e["model_key"] == FAKE_A
        else real_make(m, e),
    )
    res = _run(small_spec(tmp_path))
    rows = _rows(
        res.run_dir,
        "SELECT r.usage_estimated, r.input_tokens, r.output_tokens FROM response r JOIN trial t ON t.trial_id = r.trial_id WHERE t.model_key = ?",
        FAKE_A,
    )
    assert rows and all(r[0] == 1 and r[1] > 0 and r[2] > 0 for r in rows)


def _bundle(tmp_path: Path) -> Path:
    from financial_llm_harness.generation.generate import generate

    out = tmp_path / "bundle"
    generate(small_spec(tmp_path), out, code_ref=FIXED_CODE_REF)
    return out


def test_plan_exits_2_for_a_missing_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from financial_llm_harness.execution.plan import render_plan

    out = _bundle(tmp_path)
    real = BU.load_pricing

    def pricing() -> dict[str, Any]:
        d = real()
        d["prices"] = {k: v for k, v in d["prices"].items() if k != "fake/a"}
        return d

    monkeypatch.setattr(BU, "load_pricing", pricing)
    with pytest.raises(HarnessValidationError) as exc:
        render_plan(out)
    assert exc.value.exit_code == 2


def test_plan_warns_on_a_stale_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from financial_llm_harness.execution.plan import render_plan

    out = _bundle(tmp_path)
    real = BU.load_pricing

    def pricing() -> dict[str, Any]:
        d = real()
        d["prices"] = {
            **d["prices"],
            "fake/a": {**d["prices"]["fake/a"], "checked_on": "2020-01-01"},
        }
        return d

    monkeypatch.setattr(BU, "load_pricing", pricing)
    with caplog.at_level(logging.WARNING):
        text = render_plan(out)
    assert "fake/a was checked more than 90 days ago" in text
    assert any("more than 90 days" in r.getMessage() for r in caplog.records)


def test_plan_exits_2_for_an_evaluator_without_price(tmp_path: Path) -> None:
    from financial_llm_harness.execution.plan import plan_data

    out = _bundle(tmp_path)
    from financial_llm_harness.core import yaml_io

    m = yaml_io.load(out / "manifest.yaml")
    m["evaluators"] = [
        {
            "feature_id": "x",
            "model_key": "acme--judge--00000000",
            "provider": "acme",
            "model": "judge",
            "family": "acme",
        }
    ]
    (out / "manifest.yaml").write_text(yaml_io.dump(m))
    with pytest.raises(HarnessValidationError) as exc:
        plan_data(out)
    assert exc.value.exit_code == 2


def test_price_arithmetic_and_batch_discount() -> None:
    p = BU.Price(1.0, 4.0, 0.5, "2026-10-01")
    assert p.cost(1_000_000, 0) == pytest.approx(1.0)
    assert p.cost(0, 1_000_000, 1_000_000) == pytest.approx(
        8.0
    )  # reasoning billed as output
    assert p.cost(1_000_000, 1_000_000, batch=True) == pytest.approx(2.5)
