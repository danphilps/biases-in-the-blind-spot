"""Policy (spec §20.5, §2.14, §36): a disallowed provider exits 5 before any call."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from financial_llm_harness.errors import PolicyRefusal
from financial_llm_harness.execution import runner
from tests.harness.helpers import (
    FIXED_CODE_REF,
    small_spec,
    write_panel,
    write_probe,
)
from tests.harness.test_harness_adapters import make_entry


def _live_ready(monkeypatch: pytest.MonkeyPatch, provider: str, model: str) -> None:
    """Test-only: let agent-drafted S1 content validate for a non-fake target, give it a key and price it.

    A live target also needs the §7 deny-list; the test supplies an empty one.
    """
    import tempfile

    from financial_llm_harness.core import yaml_io
    from financial_llm_harness.execution import budget as BU
    from financial_llm_harness.generation import deny_list as DL
    from financial_llm_harness.generation import validate_bundle as VB
    from financial_llm_harness.models.panels import key_env_of

    monkeypatch.setattr(VB, "live_violations", lambda provs: [])
    deny = Path(tempfile.mkdtemp()) / "deny_list.yaml"
    deny.write_text(
        yaml_io.dump(
            {"version": "test", "source": "empty test list", "names": [], "tickers": []}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(DL, "deny_list_path", lambda: deny)
    env = key_env_of(provider)
    assert env is not None
    monkeypatch.setenv(env, "test-key")  # only providers with a key are run (§20.9)
    real = BU.load_pricing

    def pricing() -> dict[str, Any]:
        data = real()
        data["prices"] = {
            **data["prices"],
            f"{provider}/{model}": {
                "input_per_mtok_usd": 1.0,
                "output_per_mtok_usd": 2.0,
                "reasoning_billed_as": "output",
                "batch_discount": None,
                "source_url": "test",
                "checked_on": "2026-10-01",
            },
        }
        return data

    monkeypatch.setattr(BU, "load_pricing", pricing)


@pytest.mark.parametrize("classification", ["licensed", "confidential"])
def test_non_synthetic_manifest_to_external_provider_exits_5(
    tmp_path: Path,
    harness_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    classification: str,
) -> None:
    entry = make_entry("deepseek")
    write_probe(harness_home, entry, date="2026-10-01")
    panel = write_panel(tmp_path, [entry])
    _live_ready(monkeypatch, "deepseek", "deepseek-test")
    calls: list[str] = []
    monkeypatch.setattr(
        runner, "make_adapter", lambda manifest, e: calls.append(e["model_key"])
    )
    spec = small_spec(
        tmp_path,
        n_cases=960,
        statistical_spec__planning_ref=[],
        data_classification=classification,
    )
    with pytest.raises(PolicyRefusal) as exc:
        runner.run_experiment(spec, str(panel), yes=True, code_ref=FIXED_CODE_REF)
    assert exc.value.exit_code == 5
    assert calls == []  # refused before any adapter existed
    assert not (harness_home / "runs").exists()


def test_synthetic_manifest_to_external_provider_passes_policy(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = make_entry("deepseek")
    write_probe(harness_home, entry, date="2026-10-01")
    panel = write_panel(tmp_path, [entry])
    _live_ready(monkeypatch, "deepseek", "deepseek-test")
    res = runner.run_experiment(
        small_spec(tmp_path, n_cases=960, statistical_spec__planning_ref=[]),
        str(panel),
        yes=False,
        code_ref=FIXED_CODE_REF,
    )
    assert res.run_dir is None and res.exit_code == 0  # stopped at confirm, after L5


def test_denied_region_exits_5(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = make_entry("qwen")
    entry = {**entry, "region": "cn-beijing"}
    from financial_llm_harness.models.panels import derive_model_key

    entry = {k: v for k, v in entry.items() if k != "model_key"}
    entry = {"model_key": derive_model_key(entry), **entry}
    write_probe(harness_home, entry, date="2026-10-01")
    panel = write_panel(tmp_path, [entry])
    _live_ready(monkeypatch, "qwen", "qwen-test")
    policy = runner.load_provider_policy()
    policy["deny_regions"] = ["qwen:cn-beijing"]
    monkeypatch.setattr(runner, "load_provider_policy", lambda: policy)
    with pytest.raises(PolicyRefusal):
        runner.run_experiment(
            small_spec(tmp_path, n_cases=960, statistical_spec__planning_ref=[]),
            str(panel),
            yes=True,
            code_ref=FIXED_CODE_REF,
        )
