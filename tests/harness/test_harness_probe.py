"""Adapters: probe logic (spec §20.4, §36), tested against hand-written cassettes."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from financial_llm_harness.errors import HarnessValidationError, StaleProbe
from financial_llm_harness.models import probe as PR
from financial_llm_harness.models.base import (
    ModelCapabilities,
    build_adapter,
    profile_sha256,
)
from tests.harness.helpers import Cassette
from tests.harness.test_harness_adapters import make_entry, make_request

DS_URL = "https://api.deepseek.com/chat/completions"
DS_MODELS = "https://api.deepseek.com/models"
A_URL = "https://api.anthropic.com/v1/messages"
A_MODELS = "https://api.anthropic.com/v1/models"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OR_MODELS = "https://openrouter.ai/api/v1/models"
H = {"content-type": "application/json", "x-request-id": "req-probe"}
OK = '{"answer": "ok"}'


def _e(method: str, url: str, req: Any, status: int, body: Any) -> dict[str, Any]:
    return {
        "request": {"method": method, "url": url, "json": req},
        "response": {"status": status, "headers": H, "body": body},
    }


def chat_req(
    model: str,
    user: str = PR.USER,
    temperature: float | None = 0.0,
    seed: int | None = PR.PROBE_SEED,
    response_format: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": PR.SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": 256,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = seed
    if response_format is not None:
        body["response_format"] = response_format
    body.update(extra or {})
    return body


def chat_resp(
    model: str,
    text: str,
    fingerprint: str = "fp_1",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "id": "chatcmpl-probe",
        "object": "chat.completion",
        "created": 1767225600,
        "model": model,
        "system_fingerprint": fingerprint,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }
    out.update(extra or {})
    return out


def listing(model: str, url: str = DS_MODELS) -> dict[str, Any]:
    return _e(
        "GET",
        url,
        None,
        200,
        {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "t"}]},
    )


def deepseek_entries(
    t0_texts: tuple[str, str, str] = (OK, OK, OK),
    seed_outputs: tuple[tuple[str, str], ...] = (
        ("a b c", "a b c"),
        ("1, 2", "3, 4"),
        ("Xa\nXb", "Xa\nXb"),
        ("qwer", "asdf"),
    ),
) -> list[dict[str, Any]]:
    m = "deepseek-test"
    entries = [listing(m)]
    entries += [_e("POST", DS_URL, chat_req(m), 200, chat_resp(m, t)) for t in t0_texts]
    for k, (o1, o2) in enumerate(seed_outputs):
        for o in (o1, o2):
            entries.append(
                _e(
                    "POST",
                    DS_URL,
                    chat_req(
                        m,
                        user=PR.ENTROPY_PROMPTS[k],
                        temperature=PR.PROBE_TEMPERATURE_SEED_CHECK,
                        seed=PR.PROBE_SEED + k,
                    ),
                    200,
                    chat_resp(m, o),
                )
            )
    entries.append(
        _e(
            "POST",
            DS_URL,
            chat_req(m, response_format={"type": "json_object"}),
            200,
            chat_resp(m, OK),
        )
    )
    return entries


def run_probe(
    entry: dict[str, Any], cas: Cassette, tmp_path: Path
) -> tuple[dict[str, Any], PR.Prober]:
    path = PR.probe_entry(
        entry,
        http_client=cas.client(),
        key="sk-probe-secret",
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        out_root=tmp_path / "probes",
    )
    cas.assert_exhausted()
    data = json.loads(path.read_text())
    assert path == tmp_path / "probes" / entry["model_key"] / "2026-01-01.json"
    return data, PR.Prober(entry)


def test_determinism_and_seed_reproducibility(tmp_path: Path) -> None:
    entry = make_entry("deepseek")
    data, _ = run_probe(entry, Cassette.from_entries(deepseek_entries()), tmp_path)
    caps = data["capabilities"]
    assert data["checks"]["listed"] is True
    assert data["checks"]["deterministic_at_t0"] is True
    assert caps["seed_parameter"] == "accepted"
    # 8 calls on 4 prompts; 2 of 4 reproduced
    assert caps["seed_reproducibility_rate"] == 0.5
    assert (
        sum(1 for c in data["checks"]["calls"] if c["label"].startswith("seed_")) == 8
    )
    assert caps["supports_schema"] is True and data["checks"]["schema_enforced"] is True
    assert data["calls"] == 13 <= PR.MAX_CALLS
    # the probe records its request profile and the serving fingerprint
    assert data["request_profile_id"] == "deepseek/chat_completions"
    assert data["request_profile_sha256"] == profile_sha256(
        "deepseek", "deepseek/chat_completions"
    )
    assert caps["probe_serving_fingerprint"]["model_revision"] == "fp_1"
    assert data["checks"]["serving_fingerprint_stable"] is True
    assert "sk-probe-secret" not in json.dumps(data)


def test_nondeterminism_is_detected(tmp_path: Path) -> None:
    entry = make_entry("deepseek")
    cas = Cassette.from_entries(deepseek_entries(t0_texts=(OK, '{"answer":"ok"}', OK)))
    data, _ = run_probe(entry, cas, tmp_path)
    assert data["checks"]["deterministic_at_t0"] is False


def test_rejected_seed_is_never_sent_later(tmp_path: Path) -> None:
    m = "deepseek-test"
    err = {"error": {"message": "Unrecognized request argument supplied: seed"}}
    entries = [
        listing(m),
        _e("POST", DS_URL, chat_req(m), 400, err),
        _e("POST", DS_URL, chat_req(m, seed=None), 200, chat_resp(m, OK)),
        _e("POST", DS_URL, chat_req(m, seed=None), 200, chat_resp(m, OK)),
        _e("POST", DS_URL, chat_req(m, seed=None), 200, chat_resp(m, OK)),
        _e(
            "POST",
            DS_URL,
            chat_req(m, seed=None, response_format={"type": "json_object"}),
            200,
            chat_resp(m, OK),
        ),
    ]
    entry = make_entry("deepseek")
    data, _ = run_probe(entry, Cassette.from_entries(entries), tmp_path)
    caps = ModelCapabilities.from_dict(data["capabilities"])
    assert caps.seed_parameter == "rejected"
    assert caps.seed_reproducibility_rate is None
    assert caps.param_status("seed") == "rejected"
    # a later request from that model carries no seed
    req = make_request(caps)
    assert req.seed is None


def test_profile_without_seed_field_is_unavailable(tmp_path: Path) -> None:
    m = "claude-test"
    body = {
        "model": m,
        "max_tokens": 256,
        "system": PR.SYSTEM,
        "messages": [{"role": "user", "content": PR.USER}],
        "temperature": 0.0,
    }
    resp = {
        "id": "msg_p",
        "type": "message",
        "role": "assistant",
        "model": m,
        "content": [{"type": "text", "text": OK}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    schema_body = {
        **body,
        "output_config": {
            "format": {"type": "json_schema", "schema": PR.SCHEMA_CONTRACT["schema"]}
        },
    }
    entries = [
        _e(
            "GET",
            A_MODELS,
            None,
            200,
            {"data": [{"id": m, "type": "model"}], "has_more": False},
        ),
        _e("POST", A_URL, body, 200, resp),
        _e("POST", A_URL, body, 200, resp),
        _e("POST", A_URL, body, 200, resp),
        _e("POST", A_URL, schema_body, 200, resp),
    ]
    data, _ = run_probe(
        make_entry("anthropic"), Cassette.from_entries(entries), tmp_path
    )
    assert data["capabilities"]["seed_parameter"] == "unavailable"
    assert data["capabilities"]["seed_reproducibility_rate"] is None
    assert data["calls"] == 5


def test_rejected_temperature_is_recorded_and_omitted_later(tmp_path: Path) -> None:
    m = "deepseek-test"
    err = {
        "error": {
            "message": "Unsupported parameter: 'temperature' is not supported with this model."
        }
    }
    no_t = {"temperature": None}
    entries = [
        listing(m),
        _e("POST", DS_URL, chat_req(m), 400, err),
        *[
            _e("POST", DS_URL, chat_req(m, **no_t), 200, chat_resp(m, OK))  # type: ignore[arg-type]
            for _ in range(3)
        ],
    ]
    for k in range(4):
        for _ in range(2):
            entries.append(
                _e(
                    "POST",
                    DS_URL,
                    chat_req(
                        m,
                        user=PR.ENTROPY_PROMPTS[k],
                        temperature=None,
                        seed=PR.PROBE_SEED + k,
                    ),
                    200,
                    chat_resp(m, f"out {k}"),
                )
            )
    entries.append(
        _e(
            "POST",
            DS_URL,
            chat_req(m, temperature=None, response_format={"type": "json_object"}),
            200,
            chat_resp(m, OK),
        )
    )
    entry = make_entry("deepseek")
    data, _ = run_probe(entry, Cassette.from_entries(entries), tmp_path)
    caps = ModelCapabilities.from_dict(data["capabilities"])
    assert caps.param_status("temperature") == "rejected"
    assert data["checks"]["temperature_effective"] == "provider_default"
    assert caps.seed_reproducibility_rate == 1.0
    # a later request omits temperature and records effective: provider_default
    later = Cassette.from_entries(
        [
            _e(
                "POST",
                DS_URL,
                {**chat_req(m, user="u", temperature=None, seed=1)},
                200,
                chat_resp(m, OK),
            )
        ]
    )
    adapter = build_adapter(entry, caps, http_client=later.client(), key="k")
    from dataclasses import replace

    req = replace(
        make_request(caps), system=PR.SYSTEM, user="u", seed=1, max_output_tokens=256
    )
    resp = asyncio.run(adapter.run(req))
    later.assert_exhausted()
    assert resp.effective_params is not None
    assert resp.effective_params["temperature"]["effective"] == "provider_default"
    assert resp.effective_params["temperature"]["sent"] is False


def test_probe_never_exceeds_20_calls() -> None:
    prober = PR.Prober(make_entry("deepseek"))
    prober.state.calls = PR.MAX_CALLS
    with pytest.raises(PR.ProbeBudgetExceeded):
        asyncio.run(
            prober.call(PR.USER, temperature=0.0, seed=1, schema=False, label="x")
        )


def test_openrouter_probe_with_another_upstream_exits_2(tmp_path: Path) -> None:
    m = "vendor/open-model"
    pin = {
        "provider": {
            "only": ["acme-gpu"],
            "order": ["acme-gpu"],
            "allow_fallbacks": False,
            "quantizations": ["fp8"],
        }
    }
    resp = chat_resp(m, OK, extra={"provider": "other-host"})
    entries = [
        listing(m, OR_MODELS),
        *[_e("POST", OR_URL, chat_req(m, extra=pin), 200, resp) for _ in range(3)],
    ]
    with pytest.raises(HarnessValidationError) as exc:
        run_probe(make_entry("openrouter"), Cassette.from_entries(entries), tmp_path)
    assert exc.value.exit_code == 2 and "pin" in str(exc.value)


def test_validate_exits_6_when_profile_sha_differs_from_probe(
    tmp_path: Path, harness_home: Path
) -> None:
    from financial_llm_harness.generation.generate import generate
    from financial_llm_harness.generation.validate_bundle import validate_bundle
    from tests.harness.helpers import (
        FIXED_CODE_REF,
        local_entry,
        small_spec,
        write_panel,
        write_probe,
    )

    entry = local_entry()
    panel = write_panel(tmp_path, [entry])
    probe = write_probe(harness_home, entry, date="2026-01-01")
    data = json.loads(probe.read_text())
    data["request_profile_sha256"] = "f" * 64
    data["capabilities"]["request_profile_sha256"] = "f" * 64
    probe.write_text(json.dumps(data))
    out = tmp_path / "bundle"
    with pytest.raises(StaleProbe) as exc:
        generate(small_spec(tmp_path), out, str(panel), code_ref=FIXED_CODE_REF)
        validate_bundle(
            out, code_ref=FIXED_CODE_REF, clock=lambda: datetime(2026, 1, 2, tzinfo=UTC)
        )
    assert exc.value.exit_code == 6
