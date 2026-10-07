"""Request profiles: the deterministic mapping between frozen messages and wire bodies (spec §20.2, §20.10).

A profile is data in ``providers.yaml``. This module interprets it: it builds the
request body from the frozen message object, the panel entry and the frozen
capabilities, and reads a response body at the profile's paths. Adapters branch
on ``protocol`` only.

Path syntax (dotted): ``a.b`` (keys), ``0`` (list index), ``*`` (every element),
``name[key=value]`` (the elements of list ``name`` whose ``key`` equals ``value``;
``true``/``false`` compare as booleans).
"""

from __future__ import annotations

import contextvars
import copy
import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ServingFingerprint,
)

_FILTER = re.compile(r"^([A-Za-z_][\w-]*)\[([\w-]+)=([^\]]*)\]$")


def _split(path: str) -> list[str]:
    return [p for p in path.split(".") if p != ""]


def _filter_value(raw: str) -> Any:
    if raw == "true":
        return True
    if raw == "false":
        return False
    return raw


def get_all(obj: Any, path: str) -> list[Any]:
    """Every value at ``path`` (an empty list where nothing matches)."""
    cur: list[Any] = [obj]
    for seg in _split(path):
        nxt: list[Any] = []
        m = _FILTER.match(seg)
        for c in cur:
            if m is not None:
                name, key, val = m.group(1), m.group(2), _filter_value(m.group(3))
                lst = c.get(name) if isinstance(c, Mapping) else None
                if isinstance(lst, list):
                    nxt.extend(
                        x for x in lst if isinstance(x, Mapping) and x.get(key) == val
                    )
            elif seg == "*":
                if isinstance(c, list):
                    nxt.extend(c)
            elif seg.isdigit():
                if isinstance(c, list) and int(seg) < len(c):
                    nxt.append(c[int(seg)])
            elif isinstance(c, Mapping) and seg in c:
                nxt.append(c[seg])
        cur = nxt
    return cur


def get_first(obj: Any, path: str | None) -> Any:
    if not path:
        return None
    vals = get_all(obj, path)
    return vals[0] if vals else None


def set_path(body: dict[str, Any], path: str, value: Any) -> None:
    """Set ``path`` (keys only) in ``body``, creating mappings; a mapping value merges deeply."""
    parts = _split(path)
    cur = body
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    last = parts[-1]
    if isinstance(value, Mapping) and isinstance(cur.get(last), dict):
        deep_merge(cur[last], value)
    else:
        cur[last] = copy.deepcopy(value)


def deep_merge(dst: dict[str, Any], src: Mapping[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuiltRequest:
    body: dict[str, Any]
    effective_params: dict[str, Any]
    system_mode: str  # native | inlined


def _messages(
    protocol: str, system: str, user: str, inline: bool
) -> tuple[Any, str | None]:
    """(messages or contents, system text for a protocol with a top-level system field)."""
    user_text = f"{system}\n\n{user}" if inline else user
    if protocol == "chat_completions":
        msgs = [] if inline else [{"role": "system", "content": system}]
        msgs.append({"role": "user", "content": user_text})
        return msgs, None
    if protocol == "anthropic_messages":
        return [{"role": "user", "content": user_text}], None if inline else system
    if protocol == "gemini_generate_content":
        return [{"role": "user", "parts": [{"text": user_text}]}], (
            None if inline else system
        )
    raise HarnessValidationError(f"protocol {protocol!r} has no wire body")


def thinking_values(
    protocol: str, profile: Mapping[str, Any], thinking: Mapping[str, Any]
) -> dict[str, Any]:
    """Request path → value expressing the panel's ``thinking`` under the profile (§20.2, §20.3)."""
    spec = profile.get("thinking") or {}
    kind = spec.get("kind", "none")
    fields = list(spec.get("fields") or [])
    mode = thinking.get("mode", "provider_default")
    effort = thinking.get("effort")
    budget = thinking.get("budget_tokens")
    if kind in ("none", "model_variant") or mode == "provider_default":
        return {}
    out: dict[str, Any] = {}
    if kind == "reasoning_effort":
        if mode == "on" and effort is not None:
            out[fields[0]] = effort
        elif mode == "off" and spec.get("off_value") is not None:
            out[fields[0]] = spec["off_value"]
        return out
    if kind == "adaptive":  # anthropic_messages: thinking + output_config.effort
        if mode == "off":
            out[fields[0]] = {"type": "disabled"}
            return out
        out[fields[0]] = {"type": "adaptive"}
        if effort is not None and len(fields) > 1:
            out[fields[1]] = effort
        return out
    if kind == "thinking_config":
        if protocol == "anthropic_messages":
            if mode == "off":
                out[fields[0]] = {"type": "disabled"}
            elif budget is not None:
                out[fields[0]] = {"type": "enabled", "budget_tokens": int(budget)}
            return out
        if protocol == "gemini_generate_content":
            if mode == "off":
                out[fields[0]] = {"thinkingBudget": 0}
            elif budget is not None:
                out[fields[0]] = {"thinkingBudget": int(budget)}
            elif effort is not None:
                out[fields[0]] = {"thinkingLevel": effort}
            return out
    if kind == "enable_thinking":
        out[fields[0]] = mode == "on"
        if mode == "on" and budget is not None and len(fields) > 1:
            out[fields[1]] = int(budget)
        return out
    raise HarnessValidationError(
        f"thinking kind {kind!r} is not defined for protocol {protocol!r}"
    )


def structured_output_values(
    protocol: str, profile: Mapping[str, Any], contract: Mapping[str, Any] | None
) -> dict[str, Any]:
    so = profile.get("structured_output") or {}
    kind, field = so.get("kind", "none"), so.get("field")
    if kind == "none" or not field or contract is None:
        return {}
    schema = contract.get("schema")
    name = str(contract.get("contract_id", "response"))
    if kind == "json_object":
        return {field: {"type": "json_object"}}
    if kind == "json_schema":
        return {
            field: {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            }
        }
    if kind == "native_schema":
        if protocol == "anthropic_messages":
            return {field: {"type": "json_schema", "schema": schema}}
        if protocol == "gemini_generate_content":
            return {
                "generationConfig.responseMimeType": "application/json",
                field: schema,
            }
    raise HarnessValidationError(
        f"structured output {kind!r} is not defined for protocol {protocol!r}"
    )


def routing_values(
    profile: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    """OpenRouter hard pinning (§20.8): allowlist and order hold only the pinned provider."""
    rf = profile.get("routing_fields")
    pin = entry.get("upstream_routing")
    if not rf or not pin:
        return {}
    out: dict[str, Any] = {
        rf["allowlist"]: [pin["provider"]],
        rf["order"]: [pin["provider"]],
        rf["allow_fallbacks"]: False,
    }
    if pin.get("quantization"):
        out[rf["quantizations"]] = [pin["quantization"]]
    return out


def build_body(
    profile: Mapping[str, Any],
    entry: Mapping[str, Any],
    caps: ModelCapabilities,
    request: ModelRequest,
) -> BuiltRequest:
    """The request body for one attempt, by the profile's deterministic mapping only (§20.10)."""
    protocol = profile["protocol"]
    inline = not caps.supports_system
    messages, system = _messages(protocol, request.system, request.user, inline)
    body: dict[str, Any] = {}
    effective: dict[str, Any] = {}
    if protocol == "gemini_generate_content":
        body["contents"] = messages
        if system is not None:
            body["systemInstruction"] = {"parts": [{"text": system}]}
    else:
        body["model"] = entry["model"]
        if protocol == "anthropic_messages" and system is not None:
            body["system"] = system
        body["messages"] = messages

    def put(name: str, path: str | None, value: Any) -> None:
        """Send a parameter unless its frozen support is ``rejected`` (E-21); record requested and effective."""
        status = caps.param_status(name)
        if path is None or status == "rejected":
            effective[name] = {
                "requested": value,
                "sent": False,
                "effective": "provider_default",
            }
            return
        set_path(body, path, value)
        effective[name] = {
            "requested": value,
            "sent": True,
            "effective": "provider_default" if status == "ignored" else value,
        }

    put("max_output_tokens", profile["max_output_field"], request.max_output_tokens)
    if request.temperature is not None:
        put(
            "temperature",
            "generationConfig.temperature"
            if protocol == "gemini_generate_content"
            else "temperature",
            request.temperature,
        )
    if request.seed is not None:
        # seed is None when seed_parameter is rejected or unavailable (§20.8)
        put("seed", profile.get("seed_field"), request.seed)
    if request.enforcement == "provider_schema":
        for path, value in structured_output_values(
            protocol, profile, request.response_contract
        ).items():
            put(
                "response_schema"
                if path == (profile.get("structured_output") or {}).get("field")
                else path,
                path,
                value,
            )
    for path, value in thinking_values(
        protocol, profile, entry.get("thinking") or {}
    ).items():
        put(path, path, value)
    for path, value in routing_values(profile, entry).items():
        set_path(body, path, value)
        effective[path] = {"requested": value, "sent": True, "effective": value}
    for key, value in sorted((entry.get("params") or {}).items()):
        put(key, key, value)
    return BuiltRequest(body, effective, "inlined" if inline else "native")


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Extracted:
    text: str | None
    reasoning_trace: str | None
    finish_reason: str | None
    finish_reason_raw: str | None
    usage: dict[str, int | None]
    serving_fingerprint: ServingFingerprint
    refusal: bool


def _join(values: list[Any]) -> str | None:
    parts = [v for v in values if isinstance(v, str)]
    return "\n".join(parts) if parts else None


def _int(v: Any) -> int | None:
    return int(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def extract(
    profile: Mapping[str, Any],
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    region: str | None,
) -> Extracted:
    """Model fields of a complete response body, at the profile's paths (§20.7, §20.10)."""
    protocol = profile["protocol"]
    refusal = False
    if protocol == "chat_completions":
        text = get_first(payload, "choices.0.message.content")
        raw_finish = get_first(payload, "choices.0.finish_reason")
        refusal_text = get_first(payload, "choices.0.message.refusal")
        if isinstance(refusal_text, str) and refusal_text:
            refusal = True
            text = text if isinstance(text, str) else None
        usage = {
            "input": _int(get_first(payload, "usage.prompt_tokens")),
            "output": _int(get_first(payload, "usage.completion_tokens")),
            "reasoning": _int(
                get_first(payload, "usage.completion_tokens_details.reasoning_tokens")
            ),
        }
    elif protocol == "anthropic_messages":
        # text blocks are concatenated in order; thinking blocks go to the trace, never joined (§20.10)
        texts = [
            b.get("text")
            for b in payload.get("content") or []
            if isinstance(b, Mapping) and b.get("type") == "text"
        ]
        text = "".join(t for t in texts if isinstance(t, str)) if texts else None
        raw_finish = payload.get("stop_reason")
        refusal = raw_finish == "refusal"
        usage = {
            "input": _int(get_first(payload, "usage.input_tokens")),
            "output": _int(get_first(payload, "usage.output_tokens")),
            "reasoning": None,
        }
    elif protocol == "gemini_generate_content":
        parts = get_all(payload, "candidates.0.content.parts")
        plist = parts[0] if parts and isinstance(parts[0], list) else []
        texts = [
            p.get("text")
            for p in plist
            if isinstance(p, Mapping) and not p.get("thought") and "text" in p
        ]
        text = "".join(t for t in texts if isinstance(t, str)) if texts else None
        raw_finish = get_first(payload, "candidates.0.finishReason")
        if raw_finish is None and get_first(payload, "promptFeedback.blockReason"):
            raw_finish = "SAFETY"
        usage = {
            "input": _int(get_first(payload, "usageMetadata.promptTokenCount")),
            "output": _int(get_first(payload, "usageMetadata.candidatesTokenCount")),
            "reasoning": _int(get_first(payload, "usageMetadata.thoughtsTokenCount")),
        }
    else:
        raise HarnessValidationError(f"protocol {protocol!r} has no response body")
    trace = None
    for path in profile.get("trace_paths") or []:
        trace = _join(get_all(payload, path))
        if trace:
            break
    fmap = profile.get("finish_reason_map") or {}
    finish = fmap.get(str(raw_finish), "other") if raw_finish is not None else None
    if refusal:
        finish = "refusal"
    fp_paths = profile.get("fingerprint_paths") or {}
    lower = {k.lower(): v for k, v in headers.items()}

    def fp(name: str) -> str | None:
        path = fp_paths.get(name)
        if not path:
            return None
        v = get_first(payload, path)
        if v is None:
            v = lower.get(str(path).lower())
        return None if v is None else str(v)

    fingerprint = ServingFingerprint(
        provider_model_id=fp("provider_model_id"),
        model_revision=fp("model_revision"),
        upstream_provider=fp("upstream_provider"),
        quantization=fp("quantization"),
        region=fp("region") if fp_paths.get("region") else region,
    )
    return Extracted(
        text if isinstance(text, str) else None,
        trace,
        finish,
        None if raw_finish is None else str(raw_finish),
        usage,
        fingerprint,
        refusal,
    )


def request_id(profile: Mapping[str, Any], headers: Mapping[str, str]) -> str | None:
    lower = {k.lower(): v for k, v in headers.items()}
    for h in profile.get("request_id_headers") or []:
        v = lower.get(str(h).lower())
        if v:
            return str(v)
    return None


# ---------------------------------------------------------------------------
# Wire helpers shared by the HTTP adapters (§20.10)
# ---------------------------------------------------------------------------

_CAPTURE: contextvars.ContextVar[list[bytes] | None] = contextvars.ContextVar(
    "fh_request_capture", default=None
)


async def capture_request_hook(request: httpx.Request) -> None:
    """httpx request event hook: keep the exact body bytes sent (never headers)."""
    slot = _CAPTURE.get()
    if slot is not None:
        slot.append(bytes(request.content))


def attach_capture(client: httpx.AsyncClient) -> httpx.AsyncClient:
    hooks = dict(client.event_hooks)
    req = list(hooks.get("request", []))
    if capture_request_hook not in req:
        req.append(capture_request_hook)
    hooks["request"] = req
    client.event_hooks = hooks
    return client


def new_capture() -> list[bytes]:
    slot: list[bytes] = []
    _CAPTURE.set(slot)
    return slot


def retry_after(headers: Mapping[str, str]) -> float | None:
    lower = {k.lower(): v for k, v in headers.items()}
    v = lower.get("retry-after")
    if v is None:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        return None


def api_key(provider_cfg: Mapping[str, Any]) -> str | None:
    """The provider's key from the environment (or ``.env`` loaded by the CLI); never stored."""
    env = provider_cfg.get("key_env")
    return os.environ.get(env) if env else None


def failure_response(
    sent: bytes,
    transport_status: str,
    *,
    http_status: int | None = None,
    body: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    request_id_value: str | None = None,
    latency_ms: int = 0,
    error: str | None = None,
) -> ModelResponse:
    """A transport-class attempt: the body is provenance only, every model field None (§15, §20.7)."""
    return ModelResponse(
        request_body=sent,
        http_status=http_status,
        transport_status=transport_status,
        text=None,
        reasoning_trace=None,
        finish_reason=None,
        finish_reason_raw=None,
        usage=None,
        effective_params=None,
        provider_request_id=request_id_value,
        serving_fingerprint=None,
        raw_payload=body,
        latency_ms=latency_ms,
        retry_after_s=retry_after(headers or {}),
        error=error,
    )


def complete_response(
    profile: Mapping[str, Any],
    built: BuiltRequest,
    sent: bytes,
    status: int,
    body: bytes,
    headers: Mapping[str, str],
    region: str | None,
    latency_ms: int,
    *,
    stream: bool = False,
    payload: Mapping[str, Any] | None = None,
) -> ModelResponse:
    """A 2xx body read at the profile's paths; an unreadable body is transport-class."""
    rid = request_id(profile, headers)
    if payload is None:
        try:
            parsed = json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            return failure_response(
                sent,
                "http_error",
                http_status=status,
                body=body,
                headers=headers,
                request_id_value=rid,
                latency_ms=latency_ms,
                error=f"unreadable response body: {exc}",
            )
        if not isinstance(parsed, Mapping):
            return failure_response(
                sent,
                "http_error",
                http_status=status,
                body=body,
                headers=headers,
                request_id_value=rid,
                latency_ms=latency_ms,
                error="response body is not a JSON object",
            )
        payload = parsed
    ex = extract(profile, payload, headers, region)
    if rid is None:
        rid_body = payload.get("id") or payload.get("responseId")
        rid = str(rid_body) if rid_body else None
    return ModelResponse(
        request_body=sent,
        http_status=status,
        transport_status="complete",
        text=ex.text,
        reasoning_trace=ex.reasoning_trace,
        finish_reason=ex.finish_reason,
        finish_reason_raw=ex.finish_reason_raw,
        usage=ex.usage,
        effective_params={**built.effective_params, "system_mode": built.system_mode},
        provider_request_id=rid,
        serving_fingerprint=ex.serving_fingerprint,
        raw_payload=body,
        latency_ms=latency_ms,
        refusal=ex.refusal,
        stream=stream,
    )


def elapsed_ms(t0: float) -> int:
    return int(round((time.monotonic() - t0) * 1000))
