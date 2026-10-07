"""Semantic evaluator; Semantic majority vote (spec §16, §24.3, M6)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.evaluation import calibration as CAL
from financial_llm_harness.evaluation import semantic as SEM
from financial_llm_harness.models.base import ModelRequest, ModelResponse
from tests.harness.helpers import ROOT, read_csv

FIXTURE = ROOT / "tests" / "harness" / "fixtures" / "calibration_synthetic.jsonl"


def _resp(status: str, http: int | None, text: str | None) -> ModelResponse:
    return ModelResponse(
        request_body=b"{}",
        http_status=http,
        transport_status=status,
        text=text,
        reasoning_trace=None,
        finish_reason="stop" if status == "complete" else None,
        finish_reason_raw=None,
        usage={"input": 10, "output": 5, "reasoning": 0},
        effective_params=None,
        provider_request_id=None,
        serving_fingerprint=None,
        raw_payload=b'{"body": 1}' if http is not None else None,
        latency_ms=0,
    )


class Scripted:
    """An evaluator adapter replaying scripted attempts (the cassette's role, §20.10)."""

    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def run(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _request() -> ModelRequest:
    from financial_llm_harness.models.fake import fake_capabilities

    caps = fake_capabilities({"model": "eval", "provider": "fake"})
    return SEM.evaluator_request("prompt", "obs-1", 0, caps, 1)


async def _nosleep(_: float) -> None:
    return None


def test_http_500_then_success_parses_the_success() -> None:
    adapter = Scripted(
        _resp("http_error", 500, "<verbalized>NO</verbalized>"),
        _resp("complete", 200, "<verbalized>YES</verbalized><witness>w</witness>"),
    )
    seen: list[tuple[int, str]] = []
    text = asyncio.run(
        SEM.run_call(
            adapter,
            _request(),
            lambda n, r: seen.append((n, r.transport_status)),
            _nosleep,
        )
    )
    assert SEM.verdict(text) == "TRUE"
    assert seen == [(1, "http_error"), (2, "complete")]


def test_call_without_a_complete_attempt_is_unknown() -> None:
    adapter = Scripted(*[_resp("http_error", 500, None) for _ in range(5)])
    text = asyncio.run(SEM.run_call(adapter, _request(), lambda n, r: None, _nosleep))
    assert text is None and SEM.verdict(text) == "UNKNOWN"
    adapter = Scripted(_resp("http_error", 400, None))
    assert (
        asyncio.run(SEM.run_call(adapter, _request(), lambda n, r: None, _nosleep))
        is None
    )


def test_majority_vote_and_unknown_is_never_coerced() -> None:
    assert SEM.majority(["TRUE"]) == "TRUE"
    assert SEM.majority(["TRUE", "FALSE", "TRUE"]) == "TRUE"
    assert SEM.majority(["UNKNOWN", "FALSE", "UNKNOWN"]) == "UNKNOWN"
    assert SEM.majority(["TRUE", "FALSE", "UNKNOWN"]) == "UNKNOWN"
    assert SEM.verdict("no tags at all") == "UNKNOWN"
    assert SEM.verdict("<verbalized>MAYBE</verbalized>") == "UNKNOWN"
    assert SEM.verdict("<verbalized>NO</verbalized>") == "FALSE"


def test_prompt_is_filled_as_section_16_states() -> None:
    prompt = SEM.fill_prompt(
        "THE TASK", "momentum", "THE GUIDE", "THE CATEGORY", "THE RESPONSE"
    )
    legacy = SEM.legacy_prompt()
    assert prompt == legacy.format(
        formatted_input="THE TASK",
        concept_title="momentum",
        verbalization_check_guide="THE GUIDE",
        concept_category="THE CATEGORY",
        response="THE RESPONSE",
    )
    for part in (
        "<task>\nTHE TASK\n</task>",
        "<concept_title>\nmomentum\n",
        "- THE GUIDE",
        "THE CATEGORY",
        "<response>\nTHE RESPONSE\n",
    ):
        assert part in prompt


def test_calibrate_writes_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SEM, "calibration_dir", lambda: tmp_path / "calibration")
    path = CAL.calibrate("attributes_concept_reasoning", FIXTURE, "ci_fake")
    record = json.loads(path.read_text())
    assert path.parent.name == "attributes_concept_reasoning"
    assert path.name == f"{record['evaluator_version'][:12]}.json"
    m = record["metrics"]
    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "kappa",
        "ece",
        "se",
        "sp",
        "confusion",
        "unknown_rate",
    ):
        assert key in m
    assert record["n"] == 240 and record["fake_only"] is True
    assert m["kappa"] == 1.0 and m["se"] == 1.0 and m["sp"] == 1.0
    import hashlib

    assert record["labels_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert len(record["strata"]) == 12  # 6 arms × 2 model families
    # the committed record is the one calibrate writes
    committed = (
        ROOT
        / "experiments"
        / "financial"
        / "calibration"
        / path.parent.name
        / path.name
    )
    assert committed.read_bytes() == path.read_bytes()


def test_calibrate_needs_200_items(tmp_path: Path) -> None:
    short = tmp_path / "short.jsonl"
    short.write_text("".join(FIXTURE.read_text().splitlines(keepends=True)[:150]))
    with pytest.raises(HarnessValidationError, match="at least 200"):
        CAL.calibrate("attributes_concept_reasoning", short, "ci_fake")


def test_metrics_and_rogan_gladen() -> None:
    labels = ["TRUE"] * 40 + ["FALSE"] * 60
    preds = (
        ["TRUE"] * 30 + ["FALSE"] * 10 + ["FALSE"] * 50 + ["TRUE"] * 5 + ["UNKNOWN"] * 5
    )
    m = CAL.metrics(labels, preds, [1.0] * 100)
    assert m["confusion"] == {"tp": 30, "fn": 10, "tn": 50, "fp": 5}
    assert m["unknown_rate"] == 0.05
    assert m["se"] == 0.75 and abs(m["sp"] - 50 / 55) < 1e-12
    p_star = CAL.rogan_gladen(0.4, 0.8, 0.9)
    assert (
        p_star is not None and abs(p_star - (0.4 + 0.9 - 1) / (0.8 + 0.9 - 1)) < 1e-12
    )
    assert CAL.rogan_gladen(0.4, 0.5, 0.5) is None


def test_panel_without_evaluator_exits_2(tmp_path: Path, harness_home: Path) -> None:
    from financial_llm_harness.generation.generate import generate
    from tests.harness.helpers import (
        FIXED_CODE_REF,
        ci_fake_entries,
        fixed_clock,
        small_spec,
        write_panel,
    )

    panel = write_panel(tmp_path, ci_fake_entries(), None, "noeval.yaml")
    spec = small_spec(tmp_path, n_cases=60)
    with pytest.raises(HarnessValidationError, match="evaluator") as exc:
        generate(
            spec, tmp_path / "b", str(panel), code_ref=FIXED_CODE_REF, clock=fixed_clock
        )
    assert exc.value.exit_code == 2


def test_confirmatory_semantic_feature_without_record_exits_2(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from financial_llm_harness.generation.generate import generate
    from financial_llm_harness.generation.validate_bundle import validate_bundle
    from tests.harness.helpers import FIXED_CODE_REF, fixed_clock, small_spec

    monkeypatch.setattr(SEM, "find_calibration", lambda *a: None)
    out = tmp_path / "b"
    generate(
        small_spec(tmp_path, n_cases=60),
        out,
        code_ref=FIXED_CODE_REF,
        clock=fixed_clock,
    )
    with pytest.raises(HarnessValidationError, match="calibration") as exc:
        validate_bundle(out, code_ref=FIXED_CODE_REF, clock=fixed_clock)
    assert exc.value.exit_code == 2


def test_s1_reasoning_channel_evaluates_only_the_evaluation_set(s1_run: Path) -> None:
    from financial_llm_harness.storage.db import connect, db_path

    conn = connect(db_path(s1_run))
    n_trials = conn.execute("SELECT COUNT(*) FROM trial").fetchone()[0]
    n_eval = conn.execute("SELECT COUNT(*) FROM eval_call").fetchone()[0]
    n_obs = conn.execute(
        "SELECT COUNT(*) FROM observation WHERE feature_id = 'attributes_concept_reasoning'"
    ).fetchone()[0]
    assert (
        0 < n_obs == n_eval < n_trials
    )  # one call per observation (temperature accepted)
    # eval_call rows are never trials
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM trial WHERE trial_id LIKE 'eval/%'"
        ).fetchone()[0]
        == 0
    )
    # the reasoning channel is evaluated only for the attribution-eligible model (fake--a)
    by_model = dict(
        conn.execute(
            "SELECT t.model_key, COUNT(*) FROM observation o JOIN trial t ON t.trial_id = o.trial_id "
            "WHERE o.feature_id = 'attributes_concept_reasoning' GROUP BY t.model_key"
        ).fetchall()
    )
    assert set(by_model) == {"fake--a--b89ac294"}
    conn.close()
    costs = read_csv(s1_run / "results" / "costs.csv")
    ev = [r for r in costs if r["role"] == "evaluator"]
    assert len(ev) == 1 and int(ev[0]["calls"]) == n_eval
    attribution = read_csv(s1_run / "results" / "attribution.csv")
    channels = {(r["model_key"], r["channel"]) for r in attribution}
    assert ("fake--a--b89ac294", "reasoning") in channels
    reasoning = next(
        r
        for r in attribution
        if r["channel"] == "reasoning" and r["model_key"].startswith("fake--a")
    )
    assert reasoning["label"] == "UNVERBALISED[reasoning]"
    labels = read_csv(s1_run / "results" / "labels.csv")
    a = next(
        r
        for r in labels
        if r["model_key"].startswith("fake--a") and r["feature_id"] == "decision_add"
    )
    assert a["attribution_channel"] == "reasoning"
    assert a["attribution_label"] == "UNVERBALISED[reasoning]"
    assert a["composite_label"] == "UNVERBALISED_DEPENDENCY"
