"""Adapters (spec §20, §36): cassettes, conformance, request profiles, request artefacts,
OpenRouter pinning, clients, parameter support, retry bounds, live smoke tests."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from typing import Any

import pytest

from financial_llm_harness.core.canonical_json import canonical
from financial_llm_harness.execution import retries as RT
from financial_llm_harness.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    build_adapter,
    load_providers,
    profile_sha256,
    resolve_profile_id,
)
from financial_llm_harness.models.panels import derive_model_key
from tests.harness.helpers import Cassette

SYSTEM = "You are a careful assistant. Use only the information provided."
USER = "Reply with a JSON object with the key decision, YES or NO."
CONTRACT = {
    "contract_id": "test_v1",
    "schema": {
        "type": "object",
        "properties": {"decision": {"type": "string", "enum": ["YES", "NO"]}},
        "required": ["decision"],
        "additionalProperties": False,
    },
}
ANSWER = '{"decision": "YES"}'
SECRET = "sk-test-secret-value-0123456789"
THINK_DEFAULT = {"mode": "provider_default", "effort": None, "budget_tokens": None}

# cassette directory → (provider, model, transport, profile, upstream_routing)
TARGETS: dict[str, tuple[str, str, str, str | None, dict[str, Any] | None]] = {
    "openai": ("openai", "gpt-test", "native", None, None),
    "anthropic": ("anthropic", "claude-test", "native", None, None),
    "google": ("google", "gemini-test", "native", None, None),
    "google_compat": ("google", "gemini-test", "openai_compat", None, None),
    "deepseek": ("deepseek", "deepseek-test", "native", None, None),
    "qwen": ("qwen", "qwen-test", "native", None, None),
    "moonshot": ("moonshot", "kimi-test", "native", None, None),
    "openrouter": (
        "openrouter",
        "vendor/open-model",
        "native",
        None,
        {"provider": "acme-gpu", "quantization": "fp8"},
    ),
    "local": ("local", "local-model", "native", None, None),
}


def make_entry(
    target: str,
    thinking: dict[str, Any] | None = None,
    profile: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider, model, transport, prof, routing = TARGETS[target]
    e: dict[str, Any] = {
        "provider": provider,
        "family": f"{provider}_family",
        "model": model,
        "region": None,
        "transport": transport,
        "thinking": thinking or dict(THINK_DEFAULT),
        "request_profile": profile or prof,
        "batch": False,
        "upstream_routing": routing,
        "params": params or {},
    }
    return {"model_key": derive_model_key(e), **e}


def make_caps(entry: dict[str, Any], **overrides: Any) -> ModelCapabilities:
    pid = resolve_profile_id(
        entry["provider"], entry["transport"], entry.get("request_profile")
    )
    profile = load_providers()["providers"][entry["provider"]]["request_profiles"][pid]
    caps = ModelCapabilities(
        provider=entry["provider"],
        model=entry["model"],
        region=entry["region"],
        transport=entry["transport"],
        serving="first_party",
        revision=None,
        request_profile_id=pid,
        request_profile_sha256=profile_sha256(entry["provider"], pid),
        context_tokens=128000,
        max_output_tokens=8192,
        temperature_range=(0.0, 2.0),
        seed_parameter="accepted" if profile.get("seed_field") else "unavailable",
        seed_reproducibility_rate=1.0 if profile.get("seed_field") else None,
        supports_system=True,
        supports_schema=True,
        thinking_effective="not_controllable",
        trace_visibility="none",
        stream_required=False,
        upstream_routing=None,
        probe_serving_fingerprint=None,
        param_support=(),
    )
    return replace(caps, **overrides)


def make_request(
    caps: ModelCapabilities, enforcement: str = "prompt_only", **kw: Any
) -> ModelRequest:
    seed = 12345 if caps.seed_parameter in ("accepted", "ignored") else None
    base: dict[str, Any] = {
        "system": SYSTEM,
        "user": USER,
        "response_contract": CONTRACT,
        "max_output_tokens": 256,
        "temperature": 0.0,
        "seed": seed,
        "enforcement": enforcement,
        "trial_id": "trial-test",
    }
    base.update(kw)
    return ModelRequest(**base)


def run_case(
    target: str,
    case: str,
    *,
    entry: dict[str, Any] | None = None,
    caps: ModelCapabilities | None = None,
    enforcement: str = "prompt_only",
    n: int = 1,
) -> tuple[list[ModelResponse], Cassette]:
    entry = entry or make_entry(target)
    caps = caps or make_caps(entry)
    cas = Cassette(target, case)
    adapter = build_adapter(entry, caps, http_client=cas.client(), key=SECRET)

    async def go() -> list[ModelResponse]:
        return [await adapter.run(make_request(caps, enforcement)) for _ in range(n)]

    out = asyncio.run(go())
    cas.assert_exhausted()
    return out, cas


def check_invariants(resp: ModelResponse, cas: Cassette, i: int) -> None:
    """The conformance invariants every adapter satisfies (§20.7, §15)."""
    sent = cas.requests[i]
    assert resp.request_body == sent.content
    assert SECRET.encode() not in resp.request_body
    for h in (b"authorization", b"x-goog-api-key", b"x-api-key", b"bearer"):
        assert h not in resp.request_body.lower()
    assert resp.raw_payload == cas.response_bytes[i]
    assert resp.latency_ms >= 0
    if resp.transport_status == "complete":
        assert resp.finish_reason in (
            "stop",
            "length",
            "content_filter",
            "refusal",
            "tool",
            "other",
        )
        assert resp.usage is not None and set(resp.usage) == {
            "input",
            "output",
            "reasoning",
        }
        assert resp.serving_fingerprint is not None
        assert resp.effective_params is not None
    else:
        # a transport-class body is provenance only: no parsed model field (§15)
        for f in (
            resp.text,
            resp.reasoning_trace,
            resp.finish_reason,
            resp.usage,
            resp.effective_params,
            resp.serving_fingerprint,
        ):
            assert f is None


ALL = sorted(TARGETS)


@pytest.mark.parametrize("target", ALL)
def test_plain_response(target: str) -> None:
    (resp,), cas = run_case(target, "plain")
    check_invariants(resp, cas, 0)
    assert resp.transport_status == "complete" and resp.http_status == 200
    assert resp.text == ANSWER
    assert resp.finish_reason == "stop"
    assert resp.provider_request_id is not None
    assert resp.usage is not None and resp.usage["input"] and resp.usage["output"]
    fp = resp.serving_fingerprint
    assert fp is not None and fp.provider_model_id is not None
    assert resp.effective_params is not None
    assert resp.effective_params["system_mode"] == "native"


@pytest.mark.parametrize("target", ALL)
def test_structured_output_request_profile(target: str) -> None:
    """Each profile maps the fixed frozen messages and entry to the cassette's expected body."""
    (resp,), cas = run_case(target, "structured", enforcement="provider_schema")
    check_invariants(resp, cas, 0)
    assert resp.text is not None and json.loads(resp.text) == {"decision": "YES"}


TRACE: dict[str, tuple[str, dict[str, Any], str | None]] = {
    "anthropic": (
        "trace",
        {"mode": "on", "effort": "high", "budget_tokens": None},
        None,
    ),
    "google": ("trace", {"mode": "on", "effort": None, "budget_tokens": 1024}, None),
    "deepseek": ("trace", dict(THINK_DEFAULT), None),
    "qwen": ("trace", dict(THINK_DEFAULT), None),
    "moonshot": ("trace", dict(THINK_DEFAULT), None),
    "openrouter": ("trace", dict(THINK_DEFAULT), None),
    "local": ("trace", dict(THINK_DEFAULT), None),
}


@pytest.mark.parametrize("target", sorted(TRACE))
def test_reasoning_trace_extraction(target: str) -> None:
    case, thinking, profile = TRACE[target]
    entry = make_entry(target, thinking=thinking, profile=profile)
    (resp,), cas = run_case(target, case, entry=entry)
    check_invariants(resp, cas, 0)
    assert resp.reasoning_trace
    assert resp.text == ANSWER  # traces and text are never joined


def test_anthropic_extended_thinking_profile() -> None:
    entry = make_entry(
        "anthropic",
        thinking={"mode": "on", "effort": None, "budget_tokens": 2048},
        profile="anthropic/extended_thinking",
    )
    (resp,), cas = run_case("anthropic", "trace_extended", entry=entry)
    check_invariants(resp, cas, 0)
    assert resp.reasoning_trace == "Summarised reasoning."


def test_qwen_enable_thinking_profile() -> None:
    entry = make_entry(
        "qwen",
        thinking={"mode": "on", "effort": None, "budget_tokens": 512},
        profile="qwen/enable_thinking",
    )
    (resp,), cas = run_case("qwen", "enable_thinking", entry=entry)
    check_invariants(resp, cas, 0)
    assert resp.reasoning_trace


@pytest.mark.parametrize("target", ALL)
def test_silently_ignored_parameter_is_recorded(target: str) -> None:
    entry = make_entry(target)
    caps = make_caps(entry, param_support=(("temperature", "ignored"),))
    (resp,), cas = run_case(target, "ignored_param", entry=entry, caps=caps)
    check_invariants(resp, cas, 0)
    assert resp.effective_params is not None
    assert resp.effective_params["temperature"] == {
        "requested": 0.0,
        "sent": True,
        "effective": "provider_default",
    }


def _finish_cases(target: str) -> list[str]:
    return sorted(
        p.stem.removeprefix("finish_")
        for p in Cassette.dir(target).glob("finish_*.json")
    )


FINISH = [(t, f) for t in ALL for f in _finish_cases(t)]


@pytest.mark.parametrize(("target", "raw"), FINISH)
def test_each_finish_reason(target: str, raw: str) -> None:
    entry = make_entry(target)
    caps = make_caps(entry)
    (resp,), cas = run_case(target, f"finish_{raw}", entry=entry, caps=caps)
    check_invariants(resp, cas, 0)
    profile = load_providers()["providers"][entry["provider"]]["request_profiles"][
        caps.request_profile_id
    ]
    expected = profile["finish_reason_map"].get(raw, "other")
    assert resp.finish_reason_raw == raw
    assert resp.finish_reason == expected


@pytest.mark.parametrize("target", ALL)
def test_429_then_success(target: str) -> None:
    (first, second), cas = run_case(target, "rate_limit_then_success", n=2)
    check_invariants(first, cas, 0)
    check_invariants(second, cas, 1)
    assert first.transport_status == "http_error" and first.http_status == 429
    assert first.retry_after_s == 0.0
    assert RT.classify(first.transport_status, first.http_status) == "retry_429"
    assert second.transport_status == "complete" and second.text == ANSWER
    assert cas.requests[0].content == cas.requests[1].content  # the identical request


@pytest.mark.parametrize("target", ALL)
def test_500_then_success(target: str) -> None:
    (first, second), cas = run_case(target, "server_error_then_success", n=2)
    check_invariants(first, cas, 0)
    assert first.transport_status == "http_error" and first.http_status == 500
    assert RT.classify("http_error", 500) == "retry_transport"
    assert second.transport_status == "complete"


@pytest.mark.parametrize("target", ALL)
def test_system_prompt_inlined_only_when_unsupported(target: str) -> None:
    entry = make_entry(target)
    caps = make_caps(entry, supports_system=False)
    (resp,), cas = run_case(target, "inlined_system", entry=entry, caps=caps)
    check_invariants(resp, cas, 0)
    assert resp.effective_params is not None
    assert resp.effective_params["system_mode"] == "inlined"


@pytest.mark.parametrize("target", ALL)
def test_rejected_temperature_is_not_sent(target: str) -> None:
    """E-21: a parameter the frozen probe marks rejected is omitted and recorded as provider_default."""
    entry = make_entry(target)
    caps = make_caps(entry, param_support=(("temperature", "rejected"),))
    (resp,), cas = run_case(target, "temperature_omitted", entry=entry, caps=caps)
    check_invariants(resp, cas, 0)
    assert b'"temperature"' not in cas.requests[0].content
    assert resp.effective_params is not None
    assert resp.effective_params["temperature"] == {
        "requested": 0.0,
        "sent": False,
        "effective": "provider_default",
    }


def test_openai_refusal_field() -> None:
    (resp,), cas = run_case("openai", "refusal")
    check_invariants(resp, cas, 0)
    assert resp.refusal is True and resp.finish_reason == "refusal"


def test_openrouter_pinning_in_body_and_fingerprint() -> None:
    (resp,), cas = run_case("openrouter", "plain")
    body = json.loads(cas.requests[0].content)
    assert body["provider"] == {
        "only": ["acme-gpu"],
        "order": ["acme-gpu"],
        "allow_fallbacks": False,
        "quantizations": ["fp8"],
    }
    assert resp.serving_fingerprint is not None
    assert resp.serving_fingerprint.upstream_provider == "acme-gpu"
    (other,), _ = run_case("openrouter", "other_upstream")
    assert other.serving_fingerprint is not None
    assert other.serving_fingerprint.upstream_provider == "other-host"


@pytest.mark.parametrize("target", ["openai", "anthropic", "deepseek", "openrouter"])
def test_clients_are_built_without_sdk_retries(target: str) -> None:
    entry = make_entry(target)
    adapter = build_adapter(entry, make_caps(entry), key=SECRET)
    assert adapter.client.max_retries == 0  # type: ignore[attr-defined]


def test_every_registry_provider_has_cassettes() -> None:
    providers = set(load_providers()["providers"]) - {"fake"}
    covered = {TARGETS[t][0] for t in TARGETS}
    assert providers <= covered
    assert {t for t in TARGETS if TARGETS[t][2] == "openai_compat"} == {"google_compat"}


def test_cassette_bodies_are_canonical_json() -> None:
    for target in ALL:
        for path in Cassette.dir(target).glob("*.json"):
            for e in json.loads(path.read_text()):
                canonical(e["request"]["json"])


def test_retry_bounds() -> None:
    state = RT.RetryState()
    actions = [RT.next_action(state, "retry_429", 0.0)[0] for _ in range(20)]
    assert actions[:19] == ["retry"] * 19 and actions[19] == "RATE_LIMIT"
    state = RT.RetryState()
    actions = [RT.next_action(state, "retry_transport", None)[0] for _ in range(5)]
    assert actions == ["retry"] * 4 + ["TRANSPORT_EXHAUSTED"]
    # 30 minutes of cumulative 429 waiting also ends the trial
    state = RT.RetryState()
    assert RT.next_action(state, "retry_429", 1000.0)[0] == "retry"
    assert RT.next_action(state, "retry_429", 1000.0)[0] == "RATE_LIMIT"
    # a batch item that errors three times ends API_ERROR (two resubmissions, §15)
    assert RT.batch_item_action(errors=1, expired=False) == "resubmit"
    assert RT.batch_item_action(errors=2, expired=False) == "resubmit"
    assert RT.batch_item_action(errors=3, expired=False) == "API_ERROR"
    assert RT.batch_item_action(errors=3, expired=True) == "TIMEOUT"


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("HARNESS_LIVE") != "1", reason="HARNESS_LIVE=1 not set"
)
@pytest.mark.parametrize(
    ("provider", "env"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("google", "GEMINI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("qwen", "DASHSCOPE_API_KEY"),
        ("moonshot", "MOONSHOT_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
)
def test_live_models_list_smoke(provider: str, env: str) -> None:
    if not os.environ.get(env):
        pytest.skip(f"{env} not set")
    from financial_llm_harness.models.base import list_models

    assert list_models(provider)


THINKING_CASES = [
    (
        "openai",
        "reasoning_effort",
        {"mode": "on", "effort": "low", "budget_tokens": None},
    ),
    (
        "qwen",
        "reasoning_effort",
        {"mode": "on", "effort": "high", "budget_tokens": None},
    ),
    (
        "openrouter",
        "reasoning_effort",
        {"mode": "on", "effort": "medium", "budget_tokens": None},
    ),
    (
        "anthropic",
        "thinking_off",
        {"mode": "off", "effort": None, "budget_tokens": None},
    ),
    ("google", "thinking_off", {"mode": "off", "effort": None, "budget_tokens": None}),
    (
        "google",
        "thinking_level",
        {"mode": "on", "effort": "low", "budget_tokens": None},
    ),
]


@pytest.mark.parametrize(("target", "case", "thinking"), THINKING_CASES)
def test_thinking_settings_map_to_profile_fields(
    target: str, case: str, thinking: dict[str, Any]
) -> None:
    entry = make_entry(target, thinking=thinking)
    (resp,), cas = run_case(target, case, entry=entry)
    check_invariants(resp, cas, 0)
    assert resp.transport_status == "complete"


# ---------------------------------------------------------------------------
# Serving drift, capability drift and OpenRouter pinning through the runner
# ---------------------------------------------------------------------------

S1_ANSWER = json.dumps(
    {
        "reasoning": "The figures are weighed against the mandate.",
        "decision": "ADD",
        "confidence": 0.6,
        "material_factors": [{"factor": "valuation", "stance": "supports"}],
    }
)


def _s1_handler(revision: Any, trace: bool = False, upstream: str | None = None) -> Any:
    """Answers every chat request with a contract-valid S1 response; the fingerprint is a
    deterministic function of the request body, so it does not depend on submission order."""
    import hashlib

    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        h = hashlib.sha256(request.content).digest()[0]
        msg: dict[str, Any] = {"role": "assistant", "content": S1_ANSWER}
        if trace:
            msg["reasoning_content"] = "weighing"
        body: dict[str, Any] = {
            "id": "c",
            "object": "chat.completion",
            "model": "deepseek-test",
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
        rev = revision(h) if callable(revision) else revision
        if rev is not None:
            body["system_fingerprint"] = rev
        if upstream is not None:
            body["provider"] = upstream
        return httpx.Response(200, json=body, headers={"x-request-id": "r"})

    return handler


def _cassette_run(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
    *,
    pin: dict[str, Any] | None = None,
    interrupt_after: int | None = None,
    name: str = "small.yaml",
) -> Any:
    """A small S1 fake run in which fake--a is served by the openai_compatible adapter over a mock transport."""
    import httpx

    from financial_llm_harness.execution import runner
    from tests.harness.helpers import (
        FAKE_A,
        FIXED_CODE_REF,
        GOLDEN_RUN_ID,
        fixed_clock,
        small_spec,
    )

    real_make = runner.make_adapter
    real_open = runner.open_context
    ds_entry = make_entry("deepseek")
    ds_caps = make_caps(ds_entry)

    def make(manifest: Any, entry: Any) -> Any:
        if entry["model_key"] != FAKE_A:
            return real_make(manifest, entry)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return build_adapter(ds_entry, ds_caps, http_client=client, key="k")

    def open_ctx(*a: Any, **kw: Any) -> Any:
        ctx = real_open(*a, **kw)
        if pin is not None:
            for m in ctx.manifest["sampling_spec"]["models"]:
                if m["model_key"] == FAKE_A:
                    m["upstream_routing"] = pin
        return ctx

    monkeypatch.setattr(runner, "make_adapter", make)
    monkeypatch.setattr(runner, "open_context", open_ctx)
    spec = small_spec(tmp_path, name=name)
    return runner.run_experiment(
        spec,
        "ci_fake",
        yes=True,
        run_id=GOLDEN_RUN_ID,
        clock=fixed_clock,
        code_ref=FIXED_CODE_REF,
        interrupt_after=interrupt_after,
    )


def _events(run_dir: Any, kind: str) -> list[dict[str, Any]]:
    import sqlite3

    from financial_llm_harness.storage.db import db_path

    conn = sqlite3.connect(str(db_path(run_dir)))
    try:
        return [
            json.loads(r[0])
            for r in conn.execute(
                "SELECT payload_json FROM event WHERE kind = ? ORDER BY seq", (kind,)
            ).fetchall()
        ]
    finally:
        conn.close()


def _model_status(run_dir: Any, model_key: str) -> dict[str, Any]:
    s = json.loads((run_dir / "results" / "summary.json").read_text())
    return next(m for m in s["models"] if m["model_key"] == model_key)


def test_serving_drift_makes_model_and_run_incomplete(
    tmp_path: Any, harness_home: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.harness.helpers import FAKE_A, FAKE_B

    res = _cassette_run(
        tmp_path, monkeypatch, _s1_handler(lambda h: "rev-a" if h % 2 else "rev-b")
    )
    assert res.status == "INCOMPLETE" and res.exit_code == 4
    ev = _events(res.run_dir, "serving_drift")
    assert [(e["model_key"], e["field"]) for e in ev] == [(FAKE_A, "model_revision")]
    assert _model_status(res.run_dir, FAKE_A)["drift"]["serving"] is True
    prim = _model_status(res.run_dir, FAKE_A)["primary"]
    # no confirmatory label: the gate rows of §24.2 apply, with model_incomplete among the gates
    assert "model_incomplete" in prim["gates_failed"]
    assert prim["sensitivity_label"] in (
        "UNSTABLE",
        "INSTRUMENT_INSENSITIVE",
        "INDETERMINATE",
    )
    assert prim["is_bias"] is False and prim["is_normative_failure"] is False
    assert _model_status(res.run_dir, FAKE_B)["drift"]["serving"] is False


def test_serving_drift_is_order_and_resume_independent(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from financial_llm_harness.execution.runner import (
        InjectedInterruption,
        resume_experiment,
    )
    from tests.harness.helpers import FIXED_CODE_REF, GOLDEN_RUN_ID, fixed_clock

    handler = _s1_handler(lambda h: "rev-a" if h % 2 else "rev-b")
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "full"))
    full = _cassette_run(tmp_path, monkeypatch, handler)
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "cut"))
    with pytest.raises(InjectedInterruption):
        _cassette_run(tmp_path, monkeypatch, handler, interrupt_after=97)
    res = resume_experiment(
        GOLDEN_RUN_ID, yes=True, clock=fixed_clock, code_ref=FIXED_CODE_REF
    )
    strip = [
        {k: e[k] for k in ("model_key", "field", "expected", "observed")}
        for e in _events(full.run_dir, "serving_drift")
    ]
    assert strip == [
        {k: e[k] for k in ("model_key", "field", "expected", "observed")}
        for e in _events(res.run_dir, "serving_drift")
    ]
    assert res.status == full.status == "INCOMPLETE"


def test_null_fingerprint_field_is_not_drift(
    tmp_path: Any, harness_home: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    res = _cassette_run(tmp_path, monkeypatch, _s1_handler(None))
    assert _events(res.run_dir, "serving_drift") == []
    assert res.status == "COMPLETE"


def test_trace_appearing_is_capability_drift(
    tmp_path: Any, harness_home: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.harness.helpers import FAKE_A

    res = _cassette_run(tmp_path, monkeypatch, _s1_handler("rev-a", trace=True))
    ev = _events(res.run_dir, "capability_drift")
    assert [(e["model_key"], e["field"]) for e in ev] == [(FAKE_A, "trace_visibility")]
    assert res.status == "INCOMPLETE" and res.exit_code == 4
    assert _model_status(res.run_dir, FAKE_A)["drift"]["capability"] is True


def test_openrouter_upstream_breaking_the_pin_is_serving_drift(
    tmp_path: Any, harness_home: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.harness.helpers import FAKE_A

    pin = {"provider": "acme-gpu", "quantization": "fp8"}
    res = _cassette_run(
        tmp_path, monkeypatch, _s1_handler("rev-a", upstream="other-host"), pin=pin
    )
    fields = {
        (e["model_key"], e["field"]) for e in _events(res.run_dir, "serving_drift")
    }
    assert (FAKE_A, "upstream_provider") in fields
    assert res.status == "INCOMPLETE"


def test_openrouter_entry_without_upstream_routing_exits_2(tmp_path: Any) -> None:
    from financial_llm_harness.errors import HarnessValidationError
    from financial_llm_harness.models.panels import load_panel
    from tests.harness.helpers import write_panel

    e = make_entry("openrouter")
    e["upstream_routing"] = None
    e = {k: v for k, v in e.items() if k != "model_key"}
    e = {"model_key": derive_model_key(e), **e}
    with pytest.raises(HarnessValidationError) as exc:
        load_panel(str(write_panel(tmp_path, [e])))
    assert exc.value.exit_code == 2


def test_profile_protocol_must_match_transport(tmp_path: Any) -> None:
    from financial_llm_harness.errors import HarnessValidationError
    from financial_llm_harness.models.panels import load_panel
    from tests.harness.helpers import write_panel

    e = make_entry(
        "google", profile="google/openai_compat"
    )  # native transport, chat profile
    with pytest.raises(HarnessValidationError) as exc:
        load_panel(str(write_panel(tmp_path, [e])))
    assert exc.value.exit_code == 2


def test_request_artefacts_hash_and_hold_no_secret(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every attempt's stored request body hashes to attempt.request_artifact and holds no key or header."""
    import hashlib
    import sqlite3

    from financial_llm_harness.storage.artifacts import ArtifactStore
    from financial_llm_harness.storage.db import db_path

    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
    res = _cassette_run(tmp_path, monkeypatch, _s1_handler("rev-a"))
    store = ArtifactStore(res.run_dir / "artifacts")
    conn = sqlite3.connect(str(db_path(res.run_dir)))
    try:
        rows = conn.execute("SELECT request_artifact FROM attempt").fetchall()
    finally:
        conn.close()
    assert rows
    for (sha,) in rows:
        data = store.get(sha)
        assert hashlib.sha256(data).hexdigest() == sha
        low = data.lower()
        assert SECRET.encode() not in data
        assert b"authorization" not in low and b"x-goog-api-key" not in low


def test_editing_a_used_profile_is_frozen_input_drift(
    tmp_path: Any, harness_home: Any
) -> None:
    from financial_llm_harness.core import bundle as B
    from financial_llm_harness.core.code_ref import package_dir
    from financial_llm_harness.generation.generate import generate
    from financial_llm_harness.models import base as MB
    from tests.harness.helpers import (
        FIXED_CODE_REF,
        local_entry,
        small_spec,
        write_panel,
        write_probe,
    )

    entry = local_entry()
    write_probe(harness_home, entry, date="2026-10-01")
    out = tmp_path / "bundle"
    generate(
        small_spec(tmp_path, statistical_spec__planning_ref=[], n_cases=960),
        out,
        str(write_panel(tmp_path, [entry])),
        code_ref=FIXED_CODE_REF,
    )
    (out / B.LOCK_FILE).write_text(B.load_manifest(out)["manifest_hash"] + "\n")
    rows = [r for r in B.load_assets(out) if r["role"] == "request_profile"]
    assert len(rows) == 1 and "local/chat_completions" in rows[0]["logical_name"]
    assert B.frozen_input_mismatches(out, FIXED_CODE_REF) == []
    path = package_dir() / "models" / "providers.yaml"
    original = path.read_bytes()
    try:
        # an unused profile changes no hash
        path.write_bytes(
            original.replace(
                b"      openai/chat_completions:\n        profile_version: 1",
                b"      openai/chat_completions:\n        profile_version: 2",
            )
        )
        MB._registry_text.cache_clear()
        assert B.frozen_input_mismatches(out, FIXED_CODE_REF) == []
        # the used one is drift
        path.write_bytes(
            original.replace(
                b"      local/chat_completions:\n        profile_version: 1",
                b"      local/chat_completions:\n        profile_version: 2",
            )
        )
        MB._registry_text.cache_clear()
        mism = B.frozen_input_mismatches(out, FIXED_CODE_REF)
        assert any("local/chat_completions" in m for m in mism), mism
    finally:
        path.write_bytes(original)
        MB._registry_text.cache_clear()
