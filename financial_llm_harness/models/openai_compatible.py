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

    # -- OpenAI Batch (batch: openai_batch, §20.10) --------------------------

    async def batch_submit(
        self, items: list[tuple[str, ModelRequest]]
    ) -> P.BatchSubmission:
        """Upload one JSONL file (``custom_id = trial_id``) and create the batch."""
        lines: dict[str, bytes] = {}
        built: dict[str, P.BuiltRequest] = {}
        for custom_id, req in items:
            b = P.build_body(self.profile, self.entry, self.caps, req)
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": b.body,
            }
            lines[custom_id] = json.dumps(line, ensure_ascii=False).encode("utf-8")
            built[custom_id] = b
        data = b"\n".join(lines[c] for c, _ in items) + b"\n"
        f = await self.client.files.create(
            file=("batch.jsonl", data, "application/jsonl"), purpose="batch"
        )
        batch = await self.client.batches.create(
            input_file_id=f.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return P.BatchSubmission(batch.id, float(batch.created_at), lines, built)

    async def batch_state(self, batch_id: str) -> P.BatchState:
        b = await self.client.batches.retrieve(batch_id)
        ended = b.status in ("completed", "failed", "expired", "cancelled")
        return P.BatchState(ended, float(b.created_at), str(b.status))

    async def batch_cancel(self, batch_id: str) -> None:
        await self.client.batches.cancel(batch_id)

    async def batch_results(
        self, batch_id: str, sub: P.BatchSubmission
    ) -> dict[str, ModelResponse]:
        """Read both the output and the error file; an item in neither is expired."""
        b = await self.client.batches.retrieve(batch_id)
        out: dict[str, ModelResponse] = {}
        for file_id in (b.output_file_id, b.error_file_id):
            if not file_id:
                continue
            content = (await self.client.files.content(file_id)).content
            for raw in content.splitlines():
                if not raw.strip():
                    continue
                item = json.loads(raw)
                cid = str(item.get("custom_id"))
                if cid not in sub.lines:
                    continue
                resp = item.get("response") or {}
                status = resp.get("status_code")
                body = resp.get("body")
                if (
                    item.get("error") is None
                    and status == 200
                    and isinstance(body, dict)
                ):
                    out[cid] = P.complete_response(
                        self.profile,
                        sub.built[cid],
                        sub.lines[cid],
                        200,
                        raw,
                        {"x-request-id": str(resp.get("request_id") or "")},
                        self.entry.get("region"),
                        0,
                        payload=body,
                    )
                else:
                    out[cid] = P.batch_item_failure(
                        sub.lines[cid], "batch_errored", raw, status
                    )
        for cid, line in sub.lines.items():
            if cid not in out:
                out[cid] = P.batch_item_failure(line, "batch_expired", None)
        return out


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
