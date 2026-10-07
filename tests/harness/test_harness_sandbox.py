"""Sandbox and slice S5 `temporal_norm_code` (spec §13, §35 S5, §36 Sandbox, M5).

Tests marked ``sandbox`` start containers. With no usable Docker or Podman
backend they skip, unless ``CI=true``, in which case they fail (§36). The other
tests need no backend: they inject the process runner, so no code runs.
"""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.canonical_json import tree_sha256
from financial_llm_harness.core.ids import uint64
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.evaluation import sandbox as SB
from financial_llm_harness.evaluation.parse import MAX_CODE_BYTES, parse_code_response
from financial_llm_harness.families.code_task import family as CT
from tests.harness.helpers import (
    FAKE_A,
    FAKE_B,
    FIXED_CODE_REF,
    ROOT,
    fixed_clock,
    read_csv,
)

S5_SPEC = ROOT / "experiments" / "financial" / "specs" / "temporal_norm_code.yaml"
FX = CT.load_fixture("temporal_norm_code")
CHECKS = (
    "normalises_nontrivially",
    "shape_and_index_valid",
    "numeric_valid",
    "leak_free",
)
BACKEND = SB.find_backend()
IN_CI = os.environ.get("CI") == "true"
needs_backend = pytest.mark.skipif(
    BACKEND is None and not IN_CI,
    reason="no usable Docker or Podman backend (skips locally, fails in CI; §36)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def s5_state(
    n_assets: int = 25, n_dates: int = 800, n_features: int = 4
) -> dict[str, Any]:
    names = ["book_to_price", "earnings_yield", "roe", "accruals", "leverage"][
        :n_features
    ]
    return {
        "n_assets": n_assets,
        "n_dates": n_dates,
        "n_features": n_features,
        "feature_names": ", ".join(names),
        "burn_in": 252,
    }


MANIFEST_SEEDS = {"scenario_spec": {"seed_stream": "conf"}, "seeds": {"generation": 1}}


def s5_case(code: str, state: dict[str, Any] | None = None) -> SB.CodeCase:
    data = CT.case_data(MANIFEST_SEEDS, 0, state or s5_state(), FX)
    return SB.CodeCase(
        code=code,
        allowed_file=FX.allowed_file,
        data=data,
        tests_hidden=FX.tests_hidden,
        tests_hidden_sha256=tree_sha256(FX.tests_hidden),
        image=FX.sandbox_image,
        tests=list(FX.checks.values()),
    )


def probe_case(tmp: Path, code: str, test_body: str) -> SB.CodeCase:
    """A one-test case whose hidden test imports ``solution`` from /work in a fixture,
    as the S5 hidden tests do, so an import failure is an ERROR."""
    hidden = tmp / "tests_hidden"
    hidden.mkdir()
    (hidden / "test_probe.py").write_text(
        "import importlib\n\nimport pytest\n\n\n@pytest.fixture\ndef solution():\n"
        '    return importlib.import_module("solution")\n\n\ndef test_probe(solution):\n'
        + test_body,
        encoding="utf-8",
    )
    return SB.CodeCase(
        code=code,
        allowed_file="solution.py",
        data={"meta.json": b"{}\n"},
        tests_hidden=hidden,
        tests_hidden_sha256=tree_sha256(hidden),
        image="base",
        tests=["test_probe"],
    )


def summary(outcomes: dict[str, str]) -> bytes:
    lines = ["..F.", "=" * 20 + " short test summary info " + "=" * 20]
    lines += [
        f"{o} ../tests_hidden/test_normalise.py::{t}" for t, o in outcomes.items()
    ]
    return ("\n".join(lines) + "\n").encode()


def mount_source(argv: list[str], destination: str) -> Path:
    spec = next(a for a in argv if f"destination={destination}," in a)
    return Path(spec.split("source=", 1)[1].split(",", 1)[0])


def s5_spec(
    tmp: Path,
    n_cases: int = 40,
    overrides: dict[str, Any] | None = None,
    **more: Any,
) -> Path:
    """S5 planned with fake-design variances (§22), for fast pipeline tests."""
    spec = yaml_io.load(S5_SPEC)
    assert isinstance(spec, dict)
    spec = copy.deepcopy(spec)
    spec["scenario_spec"]["n_cases"] = n_cases
    spec["statistical_spec"]["bootstrap_B"] = 500
    spec["statistical_spec"]["planning_ref"] = [
        {
            "run_id": None,
            "model_key": mk,
            "feature_id": "valid_point_in_time_normaliser",
            "contrast": contrast,
            "variance": 0.002,
            "source_row_id": None,
            "source_sha256": None,
            "derivation": "test-only bound for the fake mechanism",
        }
        for mk in (FAKE_A, FAKE_B)
        for contrast in ("level_B", "A0_vs_A", "PL_vs_A", "PCB_vs_PCA")
    ]
    for key, value in {**(overrides or {}), **more}.items():
        cur: Any = spec
        parts = key.split("__")
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = value
    path = tmp / "s5_small.yaml"
    path.write_text(yaml_io.dump(spec), encoding="utf-8")
    return path


# smaller generated panels for the pipeline tests (the §35 ranges are kept by S5_SPEC)
SMALL_PANELS = {
    "scenario_spec__distributions__n_assets": "uniform(low=20, high=30)",
    "scenario_spec__distributions__n_dates": "uniform(low=800, high=900)",
    "scenario_spec__distributions__n_features": "uniform(low=3, high=5)",
}


def simulated_spawn(calls: list[list[str]]) -> Any:
    """A stand-in for the container: reads the solution's source and reports the
    outcomes its reference would get (§35 S5); it executes nothing."""

    def run(argv: list[str], timeout: float) -> SB.Proc:
        calls.append(argv)
        code = (
            mount_source(argv, "/case_ro") / "starter" / FX.allowed_file
        ).read_text()
        if "expanding()" in code or "rolling(" in code:
            failing: tuple[str, ...] = ()
        elif 'transform("mean")' in code:
            failing = ("leak_free",)
        else:
            failing = ("normalises_nontrivially",)
        outcomes = {
            FX.checks[c]: ("FAILED" if c in failing else "PASSED") for c in CHECKS
        }
        return SB.Proc(1 if failing else 0, summary(outcomes), b"", False, False, False)

    return run


# ---------------------------------------------------------------------------
# No backend needed
# ---------------------------------------------------------------------------


class _Recorder:
    calls: list[tuple[Any, dict[str, Any]]] = []

    def __init__(self, args: Any, **kwargs: Any) -> None:
        _Recorder.calls.append((args, kwargs))
        self.stdout = io.BytesIO(summary({"test_probe": "PASSED"}))
        self.stderr = io.BytesIO(b"")
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        pass


def test_backend_called_with_argv_list_and_shell_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Recorder.calls = []
    monkeypatch.setattr(SB.subprocess, "Popen", _Recorder)
    case = probe_case(tmp_path, "X = 1\n", "    assert solution.X == 1\n")
    res = SB.run_case(case, "a" * 32, backend="/usr/bin/docker", image="sha256:img")
    assert res.status == "SUCCESS"
    ((argv, kwargs),) = _Recorder.calls
    assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)
    assert kwargs["shell"] is False
    assert argv[:5] == [
        "/usr/bin/docker",
        "run",
        "--rm",
        "--name",
        "harness-aaaaaaaaaaaa",
    ]
    assert argv[-4:] == ["sha256:img", "sh", "-c", SB.SCRIPT]
    for flag in (
        "--network=none",
        "--read-only",
        "--memory=512m",
        "--memory-swap=512m",
        "--cpus=1",
        "--pids-limit=64",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
    ):
        assert flag in argv
    assert argv[argv.index("--ulimit") + 1] == "nofile=256"
    envs = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert envs == ["PYTHONDONTWRITEBYTECODE=1", "HOME=/work", "PYTHONPATH=/work"]
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--mount"]
    assert (
        mounts[0] == "type=tmpfs,destination=/work,tmpfs-size=67108864,tmpfs-mode=1777"
    )
    assert (
        mounts[1] == "type=tmpfs,destination=/tmp,tmpfs-size=67108864,tmpfs-mode=1777"
    )
    assert mounts[2].endswith(",destination=/case_ro,readonly")
    assert mounts[3].endswith(",destination=/tests_hidden,readonly")
    assert all(m.startswith("type=bind,source=") for m in mounts[2:])
    # hidden tests are mounted, never copied into the writable /work (§13.4)
    assert "/tests_hidden" not in SB.SCRIPT.split("&&")[0]


def test_hidden_tests_hash_checked_before_and_after(tmp_path: Path) -> None:
    case = probe_case(tmp_path, "X = 1\n", "    assert solution.X == 1\n")

    def tamper(argv: list[str], timeout: float) -> SB.Proc:
        (mount_source(argv, "/tests_hidden") / "test_probe.py").write_text("tampered")
        return SB.Proc(0, summary({"test_probe": "PASSED"}), b"", False, False, False)

    res = SB.run_case(case, "b" * 32, backend="docker", image="img", runner=tamper)
    assert (res.status, res.hash_check) == ("SANDBOX_FAILURE", "after")

    def never(argv: list[str], timeout: float) -> SB.Proc:
        raise AssertionError("no container may start after a failed pre-check")

    bad = dataclasses.replace(case, tests_hidden_sha256="0" * 64)
    res = SB.run_case(bad, "c" * 32, backend="docker", image="img", runner=never)
    assert (res.status, res.hash_check) == ("SANDBOX_FAILURE", "before")


def test_oversized_code_is_parse_failure_and_no_container(tmp_path: Path) -> None:
    big = "x = 1\n" * (MAX_CODE_BYTES // 6 + 10)
    assert len(big.encode()) > MAX_CODE_BYTES
    assert parse_code_response(f"```python\n{big}```\n").status == "PARSE_FAILURE"

    def never(argv: list[str], timeout: float) -> SB.Proc:
        raise AssertionError("an oversized code block must never run")

    case = probe_case(tmp_path, big, "    pass\n")
    res = SB.run_case(case, "d" * 32, backend="docker", image="img", runner=never)
    assert res.status == "PARSE_FAILURE"


def test_no_backend_is_sandbox_failure(tmp_path: Path) -> None:
    case = probe_case(tmp_path, "X = 1\n", "    pass\n")
    res = SB.run_case(case, "e" * 32, backend=None)
    assert (res.status, res.failure) == ("SANDBOX_FAILURE", "no_backend")


def test_per_test_outcomes_and_status_mapping() -> None:
    tests = ["test_a", "test_b"]
    spoof = b"PASSED x.py::test_b\n" + summary({"test_a": "PASSED", "test_b": "FAILED"})
    got = SB.parse_outcomes(spoof.decode(), tests)
    assert got == {"test_a": "PASSED", "test_b": "FAILED"}
    # absent from the summary counts as failed; a duplicate takes the worse outcome
    got = SB.parse_outcomes(
        summary({"test_a": "PASSED"}).decode() + "ERROR ../t.py::test_a\n", tests
    )
    assert got == {"test_a": "ERROR", "test_b": "ABSENT"}

    def proc(
        code: int | None, out: bytes = b"", err: bytes = b"", timeout: bool = False
    ) -> SB.Proc:
        return SB.Proc(code, out, err, False, False, timeout)

    ok = {"test_a": "PASSED", "test_b": "PASSED"}
    assert SB.classify(proc(0), ok)[0] == "SUCCESS"
    assert (
        SB.classify(proc(1), {"test_a": "PASSED", "test_b": "FAILED"})[0]
        == "TEST_FAILURE"
    )
    assert (
        SB.classify(proc(None, timeout=True), dict.fromkeys(tests, "ABSENT"))[0]
        == "TEST_FAILURE"
    )
    err = dict.fromkeys(tests, "ERROR")
    compile_out = summary(err) + b"E   SyntaxError: '(' was never closed\n"
    assert SB.classify(proc(1, compile_out), err)[0] == "COMPILE_FAILURE"
    assert SB.classify(proc(1, summary(err)), err)[0] == "RUNTIME_FAILURE"
    assert (
        SB.classify(proc(137), dict.fromkeys(tests, "ABSENT"))[0] == "RUNTIME_FAILURE"
    )
    assert SB.classify(
        proc(125, b"", b"docker: error"), dict.fromkeys(tests, "ABSENT")
    ) == (
        "SANDBOX_FAILURE",
        "backend_error",
    )
    assert SB.classify(
        proc(1, b"", b"cp: cannot create"), dict.fromkeys(tests, "ABSENT")
    ) == (
        "SANDBOX_FAILURE",
        "setup_failed",
    )


def test_output_is_read_through_a_1_mib_cap() -> None:
    script = (
        "import sys; sys.stdout.write('x' * 3_000_000); "
        "sys.stdout.write('\\n=== short test summary info ===\\nPASSED t.py::test_a\\n'); "
        "sys.stderr.write('y' * 10)"
    )
    proc = SB.spawn(["python3", "-c", script], 30)
    assert proc.exit_code == 0
    assert len(proc.stdout) == SB.OUTPUT_CAP and proc.stdout_truncated
    assert proc.stderr == b"y" * 10 and not proc.stderr_truncated
    # the summary beyond the cap still decides the outcome: truncation never changes a status
    assert SB.parse_outcomes(proc.parse_text(), ["test_a"]) == {"test_a": "PASSED"}
    assert SB.classify(proc, {"test_a": "PASSED"})[0] == "SUCCESS"


def test_s5_dataset_follows_the_section_12_seed() -> None:
    state = s5_state(n_assets=7, n_dates=30, n_features=3)
    a = CT.case_data(MANIFEST_SEEDS, 4, state, FX)
    b = CT.case_data(MANIFEST_SEEDS, 4, state, FX)
    assert a == b  # deterministic bytes, so the artefact is content-addressed
    assert a != CT.case_data(MANIFEST_SEEDS, 5, state, FX)
    assert CT.dataset_seed(MANIFEST_SEEDS, 4) == uint64("dataset", "conf", 1, 4)
    with np.load(io.BytesIO(a["panel.npz"]), allow_pickle=False) as z:
        assert z["values"].shape == (30, 7, 3)
        assert len(np.unique(z["dates"])) == 30
        assert list(z["columns"]) == ["book_to_price", "earnings_yield", "roe"]
        assert not np.allclose(z["values"].mean(axis=0), 0.0, atol=0.5)
    assert json.loads(a["meta.json"]) == {"burn_in": 252}


def test_s5_fake_response_uses_the_reference_solutions() -> None:
    good, bad = CT.reference_sources("temporal_norm_code")
    assert len(good) == 2 and len(bad) == 2
    for y, pool in ((1, good), (0, bad)):
        for seed in range(6):
            text = CT.fake_response(
                "normalise_code_v1", y, False, np.random.default_rng(seed)
            )
            parsed = parse_code_response(text)
            assert parsed.status == "SUCCESS" and parsed.code in pool


def test_s5_generates_validates_and_freezes_at_section_35_size(tmp_path: Path) -> None:
    from financial_llm_harness.core import bundle as B
    from financial_llm_harness.generation.generate import generate
    from financial_llm_harness.generation.provenance import live_violations
    from financial_llm_harness.generation.validate_bundle import (
        file_provenances,
        freeze,
    )

    out = tmp_path / "bundle"
    res = generate(S5_SPEC, out, code_ref=FIXED_CODE_REF, clock=fixed_clock)
    assert res.manifest["scenario_spec"]["n_cases"] == 960
    assert res.manifest["statistical_spec"]["planning_ref"] == []
    freeze(out, code_ref=FIXED_CODE_REF, clock=fixed_clock)
    assert res.manifest["controls"]["subset_n"] == 857
    roles = {(a["role"], a["logical_name"]) for a in B.load_assets(out)}
    base = "families/code_task/fixtures/temporal_norm_code"
    assert ("hidden_tests", f"{base}/tests_hidden") in roles
    assert ("template", f"{base}/starter") in roles
    assert ("family_config", f"{base}/fixture.yaml") in roles
    assert ("sandbox_recipe", "families/code_task/sandbox/base") in roles
    provs = file_provenances(B.load_assets(out), res.manifest)
    sources = {p.source for p in provs}
    assert {
        f"{base}/tests_hidden",
        f"{base}/starter",
        f"{base}/fixture.yaml",
    } <= sources
    assert any("tests_hidden" in v for v in live_violations(provs))
    for r in B.read_jsonl(out / B.RENDERINGS_FILE)[:12]:
        assert FX.starter_text().strip() in r["user"]


def test_fixture_with_fewer_than_6_unique_dates_exits_2(tmp_path: Path) -> None:
    from financial_llm_harness.generation.generate import generate
    from financial_llm_harness.generation.validate_bundle import validate_bundle

    spec = s5_spec(tmp_path, scenario_spec__distributions__n_dates="fixed(value=5)")
    out = tmp_path / "bundle"
    generate(spec, out, code_ref=FIXED_CODE_REF, clock=fixed_clock)
    with pytest.raises(HarnessValidationError, match="5 unique dates") as exc:
        validate_bundle(out, code_ref=FIXED_CODE_REF, clock=fixed_clock)
    assert exc.value.exit_code == 2
    assert CT.cut_points(CT.dataset_dates({"n_dates": 6}))  # 6 dates are enough
    with pytest.raises(HarnessValidationError):
        CT.cut_points(CT.dataset_dates({"n_dates": 5}))


@pytest.fixture
def simulated(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[list[str]]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(SB, "find_backend", lambda: "/fake/docker")
    monkeypatch.setattr(SB, "ensure_image", lambda name, backend: "sha256:fakeimage")
    monkeypatch.setattr(SB, "spawn", simulated_spawn(calls))
    yield calls


def test_s5_pipeline_records_code_outcomes(
    tmp_path: Path, harness_home: Path, simulated: list[list[str]]
) -> None:
    from financial_llm_harness.execution.runner import run_experiment
    from financial_llm_harness.storage.db import connect, db_path

    res = run_experiment(
        s5_spec(tmp_path, overrides=SMALL_PANELS),
        "ci_fake",
        yes=True,
        run_id="00000000-0000-4000-8000-000000000051",
        clock=fixed_clock,
        code_ref=FIXED_CODE_REF,
    )
    assert res.status == "COMPLETE" and res.run_dir is not None
    conn = connect(db_path(res.run_dir))
    statuses = dict(
        conn.execute("SELECT status, COUNT(*) FROM trial GROUP BY status").fetchall()
    )
    assert set(statuses) <= {"SUCCESS", "TEST_FAILURE", "PARSE_FAILURE"}
    events = [
        json.loads(r[0])
        for r in conn.execute(
            "SELECT payload_json FROM event WHERE kind = 'code_outcomes'"
        )
    ]
    executed = statuses.get("SUCCESS", 0) + statuses.get("TEST_FAILURE", 0)
    assert len(events) == len(simulated) == executed > 0
    for ev in events:
        assert set(ev["outcomes"]) == set(CHECKS)
        assert ev["image_id"] == "sha256:fakeimage" and ev["dataset_artifact"]
        assert ev["status"] == (
            "SUCCESS" if all(ev["outcomes"].values()) else "TEST_FAILURE"
        )
    ids = {
        r[0]
        for r in conn.execute(
            "SELECT sandbox_image_id FROM trial WHERE status IN ('SUCCESS', 'TEST_FAILURE')"
        )
    }
    assert ids == {"sha256:fakeimage"}
    obs = conn.execute("SELECT COUNT(DISTINCT feature_id) FROM observation").fetchone()[
        0
    ]
    assert obs == 5
    labels = read_csv(res.run_dir / "results" / "labels.csv")
    # labels.csv is model × feature (§32A): the primary carries the LEVEL label
    assert {r["feature_id"] for r in labels} == {
        "valid_point_in_time_normaliser",
        *CHECKS,
    }
    primary = [r for r in labels if r["feature_id"] == "valid_point_in_time_normaliser"]
    assert len(primary) == 2 and all(r["oracle_type"] == "LEVEL" for r in primary)
    assert all(r["zone"] == "" for r in primary)
    effects = read_csv(res.run_dir / "results" / "effects.csv")
    secondary = {e["feature_id"] for e in effects if e["analysis_role"] == "secondary"}
    assert secondary == {"valid_point_in_time_normaliser", *CHECKS}
    conn.close()

    # re-evaluation reads the stored outcomes and never executes code (§31)
    from financial_llm_harness.reporting.replay import reevaluate

    n = len(simulated)
    before = (res.run_dir / "results" / "effects.csv").read_bytes()
    reevaluate(res.run_id or "", code_ref=FIXED_CODE_REF, clock=fixed_clock)
    assert len(simulated) == n
    assert (res.run_dir / "results" / "effects.csv").read_bytes() == before


def test_s5_pipeline_without_backend_marks_sandbox_failure(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from financial_llm_harness.execution.runner import run_experiment
    from financial_llm_harness.storage.db import connect, db_path

    monkeypatch.setattr(SB, "find_backend", lambda: None)
    res = run_experiment(
        s5_spec(tmp_path, overrides=SMALL_PANELS),
        "ci_fake",
        yes=True,
        run_id="00000000-0000-4000-8000-000000000052",
        clock=fixed_clock,
        code_ref=FIXED_CODE_REF,
    )
    assert res.status != "COMPLETE" and res.run_dir is not None
    conn = connect(db_path(res.run_dir))
    statuses = {r[0] for r in conn.execute("SELECT DISTINCT status FROM trial")}
    assert (
        statuses <= {"SANDBOX_FAILURE", "PARSE_FAILURE"}
        and "SANDBOX_FAILURE" in statuses
    )
    failures = conn.execute(
        "SELECT COUNT(*) FROM event WHERE kind = 'sandbox_failure'"
    ).fetchone()[0]
    assert failures > 0
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM event WHERE kind = 'code_outcomes'"
        ).fetchone()[0]
        == 0
    )
    conn.close()


# ---------------------------------------------------------------------------
# Containers (marker ``sandbox``)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def image() -> str:
    assert BACKEND is not None, "CI=true: the sandbox tests need Docker or Podman (§36)"
    return SB.ensure_image("base", BACKEND)


def run_probe(
    tmp: Path, image: str, code: str, test_body: str, **kw: Any
) -> SB.SandboxResult:
    case = probe_case(tmp, code, test_body)
    return SB.run_case(case, os.urandom(16).hex(), backend=BACKEND, image=image, **kw)


@pytest.mark.sandbox
@needs_backend
def test_network_is_blocked(tmp_path: Path, image: str) -> None:
    code = (
        "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    RESULT = 'connected'\nexcept OSError:\n    RESULT = 'blocked'\n"
    )
    res = run_probe(tmp_path, image, code, "    assert solution.RESULT == 'blocked'\n")
    assert res.status == "SUCCESS", res.stdout[-2000:]


@pytest.mark.sandbox
@needs_backend
def test_memory_limit(tmp_path: Path, image: str) -> None:
    code = (
        "try:\n    block = bytearray(1 << 30)\n    block[:: 4096] = b'x' * len(block[:: 4096])\n"
        "    RESULT = 'allocated'\nexcept MemoryError:\n    RESULT = 'failed'\n"
    )
    res = run_probe(tmp_path, image, code, "    assert solution.RESULT == 'failed'\n")
    # either the allocation raises inside the limit, or the kernel kills the container
    assert res.status in ("SUCCESS", "RUNTIME_FAILURE"), res.stdout[-2000:]


@pytest.mark.sandbox
@needs_backend
def test_timeout_kills_and_removes_the_container(tmp_path: Path, image: str) -> None:
    attempt = os.urandom(16).hex()
    case = probe_case(tmp_path, "import time\ntime.sleep(300)\n", "    pass\n")
    res = SB.run_case(case, attempt, backend=BACKEND, image=image, timeout=10)
    assert (res.status, res.timeout) == ("TEST_FAILURE", True)
    assert BACKEND is not None
    left = subprocess.run(
        [
            BACKEND,
            "ps",
            "-a",
            "--filter",
            f"name=harness-{attempt[:12]}",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert left.stdout.strip() == ""


@pytest.mark.sandbox
@needs_backend
def test_pid_limit_contains_a_fork_bomb(tmp_path: Path, image: str) -> None:
    code = (
        "import os, time\nCOUNT = 0\nfor _ in range(500):\n    try:\n        pid = os.fork()\n"
        "    except OSError:\n        break\n    if pid == 0:\n        time.sleep(5)\n        os._exit(0)\n"
        "    COUNT += 1\n"
    )
    res = run_probe(tmp_path, image, code, "    assert solution.COUNT < 64\n")
    assert res.status == "SUCCESS", res.stdout[-2000:]


@pytest.mark.sandbox
@needs_backend
def test_root_is_read_only_and_work_is_writable(tmp_path: Path, image: str) -> None:
    code = (
        "import pathlib\nRESULT = {}\nfor p in ('/usr/local/x', '/etc/x', '/work/x', '/tmp/x'):\n"
        "    try:\n        pathlib.Path(p).write_text('x')\n        RESULT[p] = 'ok'\n"
        "    except OSError:\n        RESULT[p] = 'denied'\n"
    )
    body = (
        "    assert solution.RESULT == {'/usr/local/x': 'denied', '/etc/x': 'denied', "
        "'/work/x': 'ok', '/tmp/x': 'ok'}\n"
    )
    res = run_probe(tmp_path, image, code, body)
    assert res.status == "SUCCESS", res.stdout[-2000:]


@pytest.mark.sandbox
@needs_backend
def test_output_truncation(tmp_path: Path, image: str) -> None:
    res = run_probe(
        tmp_path, image, "X = 1\n", "    print('z' * 3_000_000)\n    assert False\n"
    )
    assert res.status == "TEST_FAILURE"
    assert res.stdout_truncated and len(res.stdout) == SB.OUTPUT_CAP


@pytest.mark.sandbox
@needs_backend
def test_hidden_tests_cannot_be_changed(tmp_path: Path, image: str) -> None:
    code = (
        "import os, pathlib\nERRORS = []\nfor action in ('overwrite', 'delete', 'add'):\n    try:\n"
        "        if action == 'overwrite':\n            pathlib.Path('/tests_hidden/test_probe.py').write_text('x')\n"
        "        elif action == 'delete':\n            os.remove('/tests_hidden/test_probe.py')\n"
        "        else:\n            pathlib.Path('/tests_hidden/test_new.py').write_text('x')\n"
        "        ERRORS.append(None)\n    except OSError as exc:\n        ERRORS.append(type(exc).__name__)\n"
    )
    res = run_probe(tmp_path, image, code, "    assert None not in solution.ERRORS\n")
    assert res.status == "SUCCESS", res.stdout[-2000:]
    assert res.hash_check is None


@pytest.mark.sandbox
@needs_backend
def test_hidden_tests_import_the_solution_from_work(tmp_path: Path, image: str) -> None:
    res = run_probe(
        tmp_path,
        image,
        "X = 1\n",
        "    assert solution.__file__ == '/work/solution.py'\n",
    )
    assert res.status == "SUCCESS", res.stdout[-2000:]


@pytest.mark.sandbox
@needs_backend
def test_compile_and_runtime_failures(tmp_path: Path, image: str) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    res = run_probe(tmp_path / "a", image, "def broken(:\n    pass\n", "    pass\n")
    assert res.status == "COMPILE_FAILURE", res.stdout[-2000:]
    res = run_probe(tmp_path / "b", image, "import no_such_module\n", "    pass\n")
    assert res.status == "RUNTIME_FAILURE", res.stdout[-2000:]


@pytest.mark.sandbox
@needs_backend
@pytest.mark.parametrize("reference", sorted(FX.references))
def test_s5_reference_solutions(reference: str, image: str) -> None:
    """§36: a no-op fails only normalises_nontrivially (so the primary); a full-sample
    z-score fails only leak_free; expanding and rolling z-scores pass all four."""
    code = (FX.directory / reference).read_text(encoding="utf-8")
    res = SB.run_case(s5_case(code), os.urandom(16).hex(), backend=BACKEND, image=image)
    failing = {c for c, t in FX.checks.items() if res.tests.get(t) != "PASSED"}
    assert failing == set(FX.references[reference]), res.stdout[-3000:]
    assert res.status == ("TEST_FAILURE" if failing else "SUCCESS")
