"""Panels and provenance (spec §2.7, §8.1, §14, §20.9, §36)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from financial_llm_harness.core.code_ref import CodeRef
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.generation.generate import generate, resolve_enforcement
from financial_llm_harness.generation.validate_bundle import validate_bundle
from financial_llm_harness.models.panels import derive_model_key, load_panel
from tests.harness.helpers import (
    ci_fake_entries,
    local_entry,
    small_spec,
    write_panel,
    write_probe,
)

CLEAN = CodeRef("a" * 40, "0" * 64, None, None, False)


def _reject(
    tmp: Path, models: list[dict[str, Any]], evaluator: dict[str, Any] | None = None
) -> None:
    path = write_panel(tmp, models, evaluator)
    with pytest.raises(HarnessValidationError) as exc:
        load_panel(str(path))
    assert exc.value.exit_code == 2


def _rekey(e: dict[str, Any]) -> dict[str, Any]:
    e = dict(e)
    e["model_key"] = derive_model_key(e)
    return e


def test_panel_rejection_rules(tmp_path: Path) -> None:
    a, b = ci_fake_entries()
    _reject(tmp_path, [a, a])  # repeated key
    _reject(
        tmp_path, [dict(a, model_key="fake--a--00000000")]
    )  # key differs from derived
    _reject(tmp_path, [_rekey(dict(a, provider="nope"))])  # unknown provider
    _reject(tmp_path, [_rekey(dict(a, family=""))])  # empty family
    _reject(
        tmp_path, [_rekey(dict(a, request_profile="openai/chat_completions"))]
    )  # another provider's profile
    _reject(
        tmp_path,
        [
            _rekey(
                dict(
                    local_entry(),
                    request_profile="local/chat_completions",
                    transport="openai_compat",
                )
            )
        ],
    )
    google = dict(local_entry(), provider="google", model="g", family="google")
    _reject(
        tmp_path, [_rekey(dict(google, request_profile="google/openai_compat"))]
    )  # protocol ≠ adapter for native
    _reject(
        tmp_path,
        [
            _rekey(
                dict(
                    google,
                    request_profile="google/generate_content",
                    transport="openai_compat",
                )
            )
        ],
    )
    load_panel(
        str(
            write_panel(
                tmp_path,
                [_rekey(dict(google, transport="openai_compat"))],
                name="ok.yaml",
            )
        )
    )
    _reject(
        tmp_path,
        [_rekey(dict(local_entry(), provider="openrouter", model="x/y", family="x"))],
    )  # openrouter without pin
    _reject(
        tmp_path,
        [_rekey(dict(a, upstream_routing={"provider": "z", "quantization": None}))],
    )  # pin on non-openrouter
    _reject(
        tmp_path, [_rekey(dict(local_entry(), batch=True))]
    )  # batch where provider has none
    ev = _rekey(dict(a, model="eval", family="fake_a"))
    _reject(tmp_path, [a, b], ev)  # evaluator family equals a target family
    load_panel("ci_fake")


def test_enforcement_auto_resolution() -> None:
    yes = {"capabilities": {"supports_schema": True}}
    no = {"capabilities": {"supports_schema": False}}
    assert resolve_enforcement("auto", [yes, yes]) == "provider_schema"
    assert resolve_enforcement("auto", [yes, no]) == "prompt_only"
    assert resolve_enforcement("prompt_only", [yes]) == "prompt_only"


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


def _live_bundle(
    tmp_path: Path, harness_home: Path, family: str = "local_family"
) -> Path:
    from tests.harness.helpers import fake_planning_refs

    entry = local_entry(family=family)
    write_probe(harness_home, entry)
    panel = write_panel(tmp_path, [entry])
    spec = small_spec(
        tmp_path,
        statistical_spec__planning_ref=fake_planning_refs()
        + _pilot_refs(entry["model_key"]),
    )
    out = tmp_path / "bundle"
    generate(spec, out, str(panel), code_ref=CLEAN)
    return out


def test_live_target_refuses_llm_drafted_files(
    tmp_path: Path, harness_home: Path
) -> None:
    out = _live_bundle(tmp_path, harness_home)
    from datetime import UTC, datetime

    with pytest.raises(HarnessValidationError, match="human-authored") as exc:
        validate_bundle(
            out, code_ref=CLEAN, clock=lambda: datetime(2026, 1, 2, tzinfo=UTC)
        )
    assert exc.value.exit_code == 2


def test_independence_overlap_exits_2(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _live_bundle(tmp_path, harness_home, family="anthropic")
    from datetime import UTC, datetime

    from financial_llm_harness.generation import validate_bundle as vb

    monkeypatch.setattr(vb, "live_violations", lambda provs: [])
    with pytest.raises(HarnessValidationError, match="independence") as exc:
        validate_bundle(
            out, code_ref=CLEAN, clock=lambda: datetime(2026, 1, 2, tzinfo=UTC)
        )
    assert exc.value.exit_code == 2
    import yaml

    m = yaml.safe_load((out / "manifest.yaml").read_text())
    assert "anthropic" in m["independence"]["oracle_author_families"]
    assert "anthropic" in m["independence"]["text_author_families"]


def test_fake_targets_exempt_from_independence(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    generate(small_spec(tmp_path), out, code_ref=CLEAN)
    validate_bundle(out, code_ref=CLEAN)


def test_stale_or_missing_probe_exits_6(tmp_path: Path, harness_home: Path) -> None:
    from datetime import UTC, datetime

    from financial_llm_harness.errors import StaleProbe

    entry = local_entry()
    panel = write_panel(tmp_path, [entry])
    with pytest.raises(StaleProbe) as exc:
        generate(small_spec(tmp_path), tmp_path / "b0", str(panel), code_ref=CLEAN)
    assert exc.value.exit_code == 6
    write_probe(harness_home, entry, date="2025-01-01")
    out = tmp_path / "b1"
    generate(small_spec(tmp_path), out, str(panel), code_ref=CLEAN)
    with pytest.raises(StaleProbe):
        validate_bundle(
            out, code_ref=CLEAN, clock=lambda: datetime(2026, 1, 2, tzinfo=UTC)
        )


def test_semantic_feature_needs_evaluator(tmp_path: Path) -> None:
    spec = small_spec(tmp_path)
    import yaml

    data = yaml.safe_load(spec.read_text())
    data["feature_spec"] = [
        f
        for f in data["feature_spec"]
        if f["feature_id"] != "attributes_concept_reasoning"
    ]
    data["feature_spec"].append(
        {
            "feature_id": "attributes_concept_reasoning",
            "type": "binary",
            "primary": False,
            "analysis_role": "attribution",
            "channel": "reasoning",
            "extractor": "semantic_verbalisation",
            "extractor_version": 1,
            "allowed_values": ["TRUE", "FALSE", "UNKNOWN"],
            "unknown_allowed": True,
            "confidence_threshold": 0.7,
            "guide": "g",
            "category": "c",
        }
    )
    from financial_llm_harness.core import yaml_io

    spec.write_text(yaml_io.dump(data))
    panel = write_panel(tmp_path, copy.deepcopy(ci_fake_entries()), None)
    with pytest.raises(HarnessValidationError, match="evaluator"):
        generate(spec, tmp_path / "b", str(panel), code_ref=CLEAN)
