"""API-key gating and the two-provider path (spec §20.9, §20.10, §23, §24; AGENTS.md).

Only providers with a populated key are run; each panel model is analysed on its
own; the Anthropic and OpenAI adapters feed the same analysis and outputs as the
fake path; and a live manifest is still refused until its text is human-authored.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from financial_llm_harness.core import yaml_io
from financial_llm_harness.errors import HarnessValidationError, MissingApiKey
from financial_llm_harness.execution import runner
from financial_llm_harness.generation.generate import generate
from financial_llm_harness.models.base import build_adapter, load_providers
from financial_llm_harness.models.panels import load_panel, split_by_api_key
from tests.harness.helpers import (
    FAKE_A,
    FAKE_B,
    FIXED_CODE_REF,
    S1_M2_SPEC,
    ci_fake_entries,
    fake_planning_refs,
    fixed_clock,
    local_entry,
    small_spec,
    write_panel,
    write_probe,
)
from tests.harness.test_harness_adapters import make_caps

LIVE_ANTHROPIC = "anthropic--claude-opus-5-5--4740b1e8"
LIVE_OPENAI = "openai--gpt-6-luna--5cfae6f9"
KEY_ENVS = sorted(
    {p["key_env"] for p in load_providers()["providers"].values() if p.get("key_env")}
)


def _keys(monkeypatch: pytest.MonkeyPatch, **present: str) -> None:
    """Exactly the named provider keys are set; every other key variable is removed."""
    for env in KEY_ENVS:
        monkeypatch.delenv(env, raising=False)
    for env, value in present.items():
        monkeypatch.setenv(env, value)


def _rows(run: Path, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(open(run / "results" / name, encoding="utf-8")))


def _no_semantic(spec: Path) -> Path:
    """The spec without its semantic feature: the live panel has no outside-family evaluator (§2.7)."""
    data = yaml_io.load(spec)
    assert isinstance(data, dict)
    data["feature_spec"] = [
        f
        for f in data["feature_spec"]
        if not str(f.get("extractor", "")).startswith("semantic_")
    ]
    spec.write_text(yaml_io.dump(data), encoding="utf-8")
    return spec


def _pilot_refs(model_key: str) -> list[dict[str, Any]]:
    return [
        {
            "run_id": "pilot-run",
            "model_key": model_key,
            "feature_id": "decision_add",
            "contrast": c,
            "variance": 0.05,
            "source_row_id": "row",
            "source_sha256": "0" * 64,
        }
        for c in ("B_vs_A", "A0_vs_A", "PL_vs_A", "PCB_vs_PCA")
    ]


def _live_spec(tmp: Path, home: Path) -> Path:
    """A small S1 spec planned for the live panel, with a probe file for each live model."""
    live = load_panel("live")
    for e in live.models:
        write_probe(home, e)
    refs = fake_planning_refs() + _pilot_refs(LIVE_ANTHROPIC) + _pilot_refs(LIVE_OPENAI)
    return _no_semantic(small_spec(tmp, statistical_spec__planning_ref=refs))


# ---------------------------------------------------------------------------
# Key gating
# ---------------------------------------------------------------------------


def test_split_by_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = load_panel("live").models + ci_fake_entries() + [local_entry()]
    _keys(monkeypatch, OPENAI_API_KEY="sk-test")
    keyed, skipped = split_by_api_key(entries)
    # fake and local providers need no key
    assert [e["model_key"] for e in keyed] == [
        LIVE_OPENAI,
        FAKE_A,
        FAKE_B,
        entries[-1]["model_key"],
    ]
    assert skipped == [
        {
            "model_key": LIVE_ANTHROPIC,
            "provider": "anthropic",
            "model": "claude-opus-5-5",
            "key_env": "ANTHROPIC_API_KEY",
            "reason": "no_api_key",
        }
    ]
    _keys(monkeypatch, OPENAI_API_KEY="sk-test", ANTHROPIC_API_KEY="   ")
    assert [s["model_key"] for s in split_by_api_key(entries)[1]] == [LIVE_ANTHROPIC]


def test_live_panel_models_and_prices() -> None:
    from financial_llm_harness.execution import budget as BU

    panel = load_panel("live")
    assert [(m["provider"], m["family"], m["model"]) for m in panel.models] == [
        ("anthropic", "anthropic", "claude-opus-5-5"),
        ("openai", "openai", "gpt-6-luna"),
    ]
    assert [m["model_key"] for m in panel.models] == [LIVE_ANTHROPIC, LIVE_OPENAI]
    assert panel.evaluator is None
    pricing = BU.load_pricing()
    for m in panel.models:
        assert BU.price_for(pricing, m["provider"], m["model"]) is not None


def test_generate_runs_only_models_with_a_key(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from financial_llm_harness.execution.plan import render_plan

    spec = _live_spec(tmp_path, harness_home)
    _keys(monkeypatch, OPENAI_API_KEY="sk-test")
    m = generate(spec, tmp_path / "b1", "live", code_ref=FIXED_CODE_REF).manifest
    ss = m["sampling_spec"]
    assert [e["model_key"] for e in ss["models"]] == [LIVE_OPENAI]
    assert ss["models_skipped"] == [
        {
            "model_key": LIVE_ANTHROPIC,
            "provider": "anthropic",
            "model": "claude-opus-5-5",
            "key_env": "ANTHROPIC_API_KEY",
            "reason": "no_api_key",
        }
    ]
    assert m["independence"]["target_families"] == ["openai"]
    plan = render_plan(tmp_path / "b1")
    assert f"{LIVE_ANTHROPIC}: skipped, ANTHROPIC_API_KEY is not set" in plan
    assert f"{LIVE_OPENAI}: trials" in plan

    _keys(monkeypatch, OPENAI_API_KEY="sk-test", ANTHROPIC_API_KEY="sk-ant-test")
    m = generate(spec, tmp_path / "b2", "live", code_ref=FIXED_CODE_REF).manifest
    assert [e["model_key"] for e in m["sampling_spec"]["models"]] == [
        LIVE_ANTHROPIC,
        LIVE_OPENAI,
    ]
    assert "models_skipped" not in m["sampling_spec"]
    # the skipped set is panel data: it moves neither design_hash nor the trial seeds (§4, §18)
    one = yaml_io.load(tmp_path / "b1" / "manifest.yaml")
    assert isinstance(one, dict)
    assert one["design_hash"] == m["design_hash"]

    _keys(monkeypatch)
    with pytest.raises(MissingApiKey, match="no target model has its API key") as exc:
        generate(spec, tmp_path / "b3", "live", code_ref=FIXED_CODE_REF)
    assert exc.value.exit_code == 2


def test_live_manifest_still_needs_human_text(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Key gating never opens the D-14 gate: agent-drafted text cannot reach a live model (§2.7, §8.1)."""
    from datetime import UTC, datetime

    from financial_llm_harness.generation.validate_bundle import validate_bundle

    spec = _live_spec(tmp_path, harness_home)
    _keys(monkeypatch, OPENAI_API_KEY="sk-test", ANTHROPIC_API_KEY="sk-ant-test")
    out = tmp_path / "bundle"
    generate(spec, out, "live", code_ref=FIXED_CODE_REF)
    with pytest.raises(HarnessValidationError, match="human-authored") as exc:
        validate_bundle(
            out,
            code_ref=FIXED_CODE_REF,
            clock=lambda: datetime(2026, 1, 2, tzinfo=UTC),
        )
    assert exc.value.exit_code == 2


def test_frozen_model_without_key_is_refused_not_dropped(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _live_spec(tmp_path, harness_home)
    _keys(monkeypatch, OPENAI_API_KEY="sk-test", ANTHROPIC_API_KEY="sk-ant-test")
    m = generate(spec, tmp_path / "b", "live", code_ref=FIXED_CODE_REF).manifest
    runner.check_api_keys(m)
    _keys(monkeypatch, OPENAI_API_KEY="sk-test")
    with pytest.raises(
        MissingApiKey, match=f"{LIVE_ANTHROPIC} \\(ANTHROPIC_API_KEY\\)"
    ):
        runner.check_api_keys(m)


def test_evaluator_without_key_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    google = local_entry(model="gemini-test", family="google", provider="google")
    panel = write_panel(tmp_path, ci_fake_entries(), google)
    _keys(monkeypatch)
    with pytest.raises(MissingApiKey, match="GEMINI_API_KEY") as exc:
        generate(
            small_spec(tmp_path), tmp_path / "b", str(panel), code_ref=FIXED_CODE_REF
        )
    assert exc.value.exit_code == 2


def test_probe_needs_a_key_and_panel_probe_skips_keyless(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from financial_llm_harness import cli
    from financial_llm_harness.models import probe

    probed: list[str] = []
    monkeypatch.setattr(
        probe, "probe_entry", lambda e, **kw: probed.append(e["model_key"]) or Path("p")
    )
    _keys(monkeypatch)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    assert cli.main(["probe", LIVE_ANTHROPIC, "--panel", "live"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert cli.main(["models", "list", "--provider", "openai"]) == 2
    _keys(monkeypatch, OPENAI_API_KEY="sk-test")
    assert cli.main(["probe", "--panel", "live"]) == 0
    out = capsys.readouterr().out
    assert probed == [LIVE_OPENAI]
    assert f"skipped {LIVE_ANTHROPIC}: ANTHROPIC_API_KEY is not set" in out


# ---------------------------------------------------------------------------
# Two providers end to end: Anthropic Messages and OpenAI Chat Completions
# ---------------------------------------------------------------------------

_MOMENTUM = re.compile(r"over the past 12 months: (-?[0-9.]+)%")
_LEVERAGE = re.compile(r"Net debt to EBITDA: (-?[0-9.]+)×")
_COMPANY = re.compile(r"Company: (.+)")


def scripted_answer(user: str, names_momentum: bool) -> str:
    """Test-only respondent with a planted momentum dependence (θ ≈ −0.4) and a leverage response.

    Its baseline draw is a function of the company name, so A, A0 and PL agree and
    the B − A contrast carries only the planted effect.
    """
    m = _MOMENTUM.search(user)
    lev = _LEVERAGE.search(user)
    co = _COMPANY.search(user)
    assert m and lev and co
    mom = float(m.group(1))
    u = int(hashlib.sha256(co.group(1).encode()).hexdigest()[:8], 16) / 2**32
    add = u < (0.3 if mom < 0 else 0.7) and float(lev.group(1)) < 5.0
    factors = [{"factor": "valuation", "stance": "supports" if add else "opposes"}]
    if names_momentum and mom < 0:
        factors.append({"factor": "momentum", "stance": "opposes"})
    return json.dumps(
        {
            "reasoning": "The figures are weighed against the mandate.",
            "decision": "ADD" if add else "DO_NOT_ADD",
            "confidence": 0.6,
            "material_factors": factors,
        }
    )


def anthropic_handler(seen: list[dict[str, Any]], names_momentum: bool) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.path == "/v1/messages"
        text = scripted_answer(body["messages"][-1]["content"], names_momentum)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 600, "output_tokens": 80},
            },
            headers={"request-id": "req_1"},
        )

    return handler


def openai_handler(seen: list[dict[str, Any]], names_momentum: bool) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.path == "/v1/chat/completions"
        text = scripted_answer(body["messages"][-1]["content"], names_momentum)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 600, "completion_tokens": 80},
            },
            headers={"x-request-id": "r1"},
        )

    return handler


def _two_provider_run(
    tmp: Path,
    monkeypatch: pytest.MonkeyPatch,
    models: list[dict[str, Any]],
    run_id: str,
) -> tuple[Any, dict[str, list[dict[str, Any]]]]:
    """S1 (M2) with fake--a served by the Anthropic adapter and fake--b by the OpenAI adapter.

    The fake panel keeps the bundle valid on the fake path (D-14); the adapters, the
    request bodies and everything after L7 are those of a live run.
    """
    live = load_panel("live")
    seen: dict[str, list[dict[str, Any]]] = {"anthropic": [], "openai": []}
    routes = {
        FAKE_A: (
            live.entry(LIVE_ANTHROPIC),
            anthropic_handler(seen["anthropic"], True),
        ),
        FAKE_B: (live.entry(LIVE_OPENAI), openai_handler(seen["openai"], False)),
    }
    real_make = runner.make_adapter

    def make(manifest: Any, entry: Any) -> Any:
        if entry["model_key"] not in routes:
            return real_make(manifest, entry)
        e, handler = routes[entry["model_key"]]
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return build_adapter(e, make_caps(e), http_client=client, key="test-key")

    monkeypatch.setattr(runner, "make_adapter", make)
    keys = {m["model_key"] for m in models}
    refs = [r for r in fake_planning_refs() if r["model_key"] in keys]
    spec = _no_semantic(
        small_spec(
            tmp,
            n_cases=60,
            base=S1_M2_SPEC,
            name=f"{run_id[:8]}.yaml",
            statistical_spec__planning_ref=refs,
        )
    )
    panel = write_panel(
        tmp, copy.deepcopy(models), None, name=f"{run_id[:8]}.panel.yaml"
    )
    res = runner.run_experiment(
        spec,
        str(panel),
        yes=True,
        run_id=run_id,
        clock=fixed_clock,
        code_ref=FIXED_CODE_REF,
    )
    return res, seen


def test_two_providers_end_to_end(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = ci_fake_entries()
    res, seen = _two_provider_run(
        tmp_path, monkeypatch, [a, b], "0000a100-0000-4000-8000-000000000000"
    )
    assert res.status == "COMPLETE" and res.run_dir is not None
    run = res.run_dir
    # each provider received only its own model's trials, under its own protocol
    assert seen["anthropic"] and seen["openai"]
    assert {r["model"] for r in seen["anthropic"]} == {"claude-opus-5-5"}
    assert {r["model"] for r in seen["openai"]} == {"gpt-6-luna"}
    trials = _rows(run, "trials.csv")
    assert len(seen["anthropic"]) == sum(t["model_key"] == FAKE_A for t in trials)
    assert len(seen["openai"]) == sum(t["model_key"] == FAKE_B for t in trials)
    for name in (
        "summary.json",
        "effects.csv",
        "labels.csv",
        "attribution.csv",
        "controls.csv",
    ):
        assert (run / "results" / name).is_file(), name
    assert (run / "report" / "report.md").is_file()
    # the planted momentum dependence violates the M2 invariance oracle for both models;
    # only the respondent that never names momentum is UNVERBALISED_BIAS (§24.2, §24.3)
    labels = {
        r["model_key"]: r
        for r in _rows(run, "labels.csv")
        if r["feature_id"] == "decision_add"
    }
    for mk in (FAKE_A, FAKE_B):
        assert labels[mk]["sensitivity_label"] == "SPURIOUS_SENSITIVITY", labels[mk]
        assert labels[mk]["is_bias"] == "true"
    assert labels[FAKE_B]["composite_label"] == "UNVERBALISED_BIAS"
    assert labels[FAKE_A]["attribution_label"].startswith("VERBALISED")
    assert labels[FAKE_A]["composite_label"] == ""


def test_each_model_is_analysed_independently(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model's estimates and labels do not depend on which other models share the panel."""
    a, b = ci_fake_entries()
    both, _ = _two_provider_run(
        tmp_path, monkeypatch, [a, b], "0000b100-0000-4000-8000-000000000000"
    )
    alone, seen = _two_provider_run(
        tmp_path, monkeypatch, [b], "0000b200-0000-4000-8000-000000000000"
    )
    assert both.status == alone.status == "COMPLETE"
    assert seen["anthropic"] == []
    cols = ("feature_id", "contrast", "theta", "ci_low", "ci_high", "p_dev", "p_eq")

    def effects(run: Path) -> list[tuple[str, ...]]:
        return sorted(
            tuple(r[c] for c in cols)
            for r in _rows(run, "effects.csv")
            if r["model_key"] == FAKE_B
        )

    assert both.run_dir is not None and alone.run_dir is not None
    assert effects(both.run_dir) == effects(alone.run_dir)
    lab = (
        "feature_id",
        "zone",
        "sensitivity_label",
        "attribution_label",
        "composite_label",
    )

    def labels(run: Path) -> list[tuple[str, ...]]:
        return sorted(
            tuple(r[c] for c in lab)
            for r in _rows(run, "labels.csv")
            if r["model_key"] == FAKE_B
        )

    assert labels(both.run_dir) == labels(alone.run_dir)


# ---------------------------------------------------------------------------
# Programmes
# ---------------------------------------------------------------------------


def test_live_programme_plan_lists_tested_and_skipped_models(
    harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from financial_llm_harness.execution.programme import cli_plan, load_programme

    path = Path("experiments/financial/programmes/live.yaml")
    assert load_programme(path)["panel"] == "live"
    _keys(monkeypatch, OPENAI_API_KEY="sk-test")
    code, text = cli_plan(
        path
    )  # no probe yet: every experiment is SKIPPED_INVALID, no call made
    assert code == 0
    assert f"models tested: {LIVE_OPENAI}" in text
    assert f"model skipped: {LIVE_ANTHROPIC} (ANTHROPIC_API_KEY is not set)" in text
    assert text.count("SKIPPED_INVALID") == 5


def test_programme_report_lists_only_tested_models() -> None:
    from types import SimpleNamespace

    from financial_llm_harness.reporting.programme import tested_models

    def exp(models: list[str], skipped: list[str]) -> Any:
        ss = {
            "models": [{"model_key": k} for k in models],
            "models_skipped": [{"model_key": k} for k in skipped],
        }
        return SimpleNamespace(manifest={"sampling_spec": ss})

    panel = [LIVE_ANTHROPIC, LIVE_OPENAI]
    assert tested_models(panel, [exp([LIVE_OPENAI], [LIVE_ANTHROPIC])]) == [LIVE_OPENAI]
    assert tested_models(panel, [exp(panel, [])]) == panel
    # an experiment that never generated (SKIPPED_INVALID) skips no model
    invalid: Any = SimpleNamespace(manifest={})
    assert tested_models(panel, [invalid]) == panel
