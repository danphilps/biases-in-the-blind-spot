from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def harness_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated HARNESS_HOME; the repository root stays the working directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HARNESS_HOME", str(home))
    return home


@pytest.fixture(scope="session")
def s1_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One full-size S1 (M1) fake run with the E-23 injected identity, shared by the session."""
    import os

    from financial_llm_harness.execution.runner import run_experiment
    from tests.harness.helpers import (
        FIXED_CODE_REF,
        GOLDEN_RUN_ID,
        S1_SPEC,
        fixed_clock,
    )

    home = tmp_path_factory.mktemp("s1home")
    old = os.environ.get("HARNESS_HOME")
    os.environ["HARNESS_HOME"] = str(home)
    try:
        res = run_experiment(
            S1_SPEC,
            "ci_fake",
            yes=True,
            run_id=GOLDEN_RUN_ID,
            clock=fixed_clock,
            code_ref=FIXED_CODE_REF,
        )
    finally:
        if old is None:
            os.environ.pop("HARNESS_HOME", None)
        else:
            os.environ["HARNESS_HOME"] = old
    assert res.run_dir is not None
    return res.run_dir
