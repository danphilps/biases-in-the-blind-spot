"""``openai_compatible`` adapter: Chat Completions via the ``openai`` SDK (spec §20.1, §20.10).

One class serves OpenAI, DeepSeek, Qwen, Kimi, Gemini compatibility mode,
OpenRouter and self-hosted engines; their differences are request-profile data.
The SDK runs with ``max_retries=0`` (the harness owns retries, §15); each call
uses ``with_raw_response``, stores the unmodified body and reads fields at the
profile's paths. The exact request bytes come from an ``httpx`` request hook.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import httpx
import openai

from financial_llm_harness.models import profile as P
from financial_llm_harness.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    base_url_for,
    provider_entry,
    provider_limits,
    request_profile,
)


class OpenAICompatibleAdapter:
    """Chat Completions adapter driven by the model's frozen request profile."""

    def __init__(
        self,
        entry: Mapping[str, Any],
        caps: ModelCapabilities,
        *,
        http_client: httpx.AsyncClient | None = None,
        key: str | None = None,
    ) -> None:
        self.entry = dict(entry)
        self.caps = caps
        cfg = provider_entry(entry["provider"])
        self.profile = request_profile(entry["provider"], caps.request_profile_id)
        if self.profile["protocol"] != "chat_completions":
            raise ValueError(
                f"profile {caps.request_profile_id} is not chat_completions"
            )
        timeout = float(
            provider_limits(entry["provider"]).get("request_timeout_s", 600)
        )
        base_url = base_url_for(
            entry["provider"], entry.get("region"), entry["transport"]
        )
        client = http_client or httpx.AsyncClient(timeout=timeout)
        self.client = openai.AsyncOpenAI(
            api_key=key if key is not None else (P.api_key(cfg) or "not-needed"),
            base_url=base_url,
            max_retries=0,
            timeout=timeout,
            http_client=P.attach_capture(client),
        )

    def capabilities(self) -> ModelCapabilities:
        return self.caps

    async def run(self, request: ModelRequest) -> ModelResponse:
        built = P.build_body(self.profile, self.entry, self.caps, request)
        body = dict(built.body)
        stream = self.caps.stream_required
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        extra = {k: v for k, v in body.items() if k not in ("model", "messages")}
        sent = P.new_capture()
        t0 = time.monotonic()
        try:
            raw = await self.client.chat.completions.with_raw_response.create(
                model=body["model"],
                messages=body["messages"],
                extra_body=extra,
            )
        except openai.APIStatusError as exc:
            resp = exc.response
            return P.failure_response(
                _sent(sent),
                "http_error",
                http_status=resp.status_code,
                body=resp.content,
                headers=resp.headers,
                request_id_value=P.request_id(self.profile, resp.headers),
                latency_ms=P.elapsed_ms(t0),
                error=type(exc).__name__,
            )
        except openai.APITimeoutError as exc:
            return P.failure_response(
                _sent(sent), "timeout", latency_ms=P.elapsed_ms(t0), error=str(exc)
            )
        except openai.APIConnectionError as exc:
            return P.failure_response(
                _sent(sent),
                "connection_error",
                latency_ms=P.elapsed_ms(t0),
                error=str(exc),
            )
        http = raw.http_response
        content = await http.aread() if not http.is_closed else http.content
        payload = _accumulate_stream(content) if stream else None
        return P.complete_response(
            self.profile,
            built,
            _sent(sent),
            http.status_code,
            content,
            http.headers,
            self.entry.get("region"),
            P.elapsed_ms(t0),
            stream=stream,
            payload=payload,
        )

    async def list_model_ids(self) -> list[str]:
        page = await self.client.models.list()
        return sorted(m.id for m in page.data)


def _sent(slot: list[bytes]) -> bytes:
    return slot[-1] if slot else b""


def _accumulate_stream(content: bytes) -> dict[str, Any]:
    """Fold server-sent chat-completion chunks into one response object (§20.3: streaming providers)."""
    text: list[str] = []
    reasoning: list[str] = []
    out: dict[str, Any] = {"choices": [{"message": {}, "finish_reason": None}]}
    for line in content.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        for key in ("id", "model", "system_fingerprint", "provider"):
            if chunk.get(key) is not None:
                out[key] = chunk[key]
        if chunk.get("usage"):
            out["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text.append(delta["content"])
            for rk in ("reasoning_content", "reasoning"):
                if isinstance(delta.get(rk), str):
                    reasoning.append(delta[rk])
            if choice.get("finish_reason") is not None:
                out["choices"][0]["finish_reason"] = choice["finish_reason"]
    msg = out["choices"][0]["message"]
    msg["content"] = "".join(text) if text else None
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
        msg["reasoning"] = msg["reasoning_content"]
    return out
