"""``anthropic`` adapter: the Messages API via the ``anthropic`` SDK (spec §20.1, §20.10).

``max_retries=0``; ``with_raw_response``; text blocks concatenated in order into
``text`` and thinking blocks into ``reasoning_trace``, never joined. Fields the
SDK does not model (thinking, ``output_config``) travel in ``extra_body``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import anthropic
import httpx

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


class AnthropicAdapter:
    """Messages API adapter driven by the model's frozen request profile."""

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
        if self.profile["protocol"] != "anthropic_messages":
            raise ValueError(
                f"profile {caps.request_profile_id} is not anthropic_messages"
            )
        timeout = float(
            provider_limits(entry["provider"]).get("request_timeout_s", 600)
        )
        client = http_client or httpx.AsyncClient(timeout=timeout)
        self.client = anthropic.AsyncAnthropic(
            api_key=key if key is not None else (P.api_key(cfg) or ""),
            base_url=base_url_for(
                entry["provider"], entry.get("region"), entry["transport"]
            ),
            max_retries=0,
            timeout=timeout,
            http_client=P.attach_capture(client),
        )

    def capabilities(self) -> ModelCapabilities:
        return self.caps

    async def run(self, request: ModelRequest) -> ModelResponse:
        built = P.build_body(self.profile, self.entry, self.caps, request)
        body = dict(built.body)
        kwargs: dict[str, Any] = {
            "model": body.pop("model"),
            "messages": body.pop("messages"),
        }
        max_field = self.profile["max_output_field"]
        kwargs["max_tokens"] = body.pop(max_field, request.max_output_tokens)
        if "system" in body:
            kwargs["system"] = body.pop("system")
        sent = P.new_capture()
        t0 = time.monotonic()
        try:
            raw = await self.client.messages.with_raw_response.create(
                **kwargs, extra_body=body
            )
        except anthropic.APIStatusError as exc:
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
        except anthropic.APITimeoutError as exc:
            return P.failure_response(
                _sent(sent), "timeout", latency_ms=P.elapsed_ms(t0), error=str(exc)
            )
        except anthropic.APIConnectionError as exc:
            return P.failure_response(
                _sent(sent),
                "connection_error",
                latency_ms=P.elapsed_ms(t0),
                error=str(exc),
            )
        http = raw.http_response
        content = await http.aread() if not http.is_closed else http.content
        return P.complete_response(
            self.profile,
            built,
            _sent(sent),
            http.status_code,
            content,
            http.headers,
            self.entry.get("region"),
            P.elapsed_ms(t0),
        )

    async def list_model_ids(self) -> list[str]:
        page = await self.client.models.list(limit=1000)
        return sorted(m.id for m in page.data)

    # -- Anthropic Message Batches (batch: anthropic_message_batches, §20.10) --

    async def batch_submit(
        self, items: list[tuple[str, ModelRequest]]
    ) -> P.BatchSubmission:
        """One batch, ``custom_id = trial_id``; each item's params are the profile's body."""
        lines: dict[str, bytes] = {}
        built: dict[str, P.BuiltRequest] = {}
        requests = []
        for custom_id, req in items:
            b = P.build_body(self.profile, self.entry, self.caps, req)
            item = {"custom_id": custom_id, "params": b.body}
            requests.append(item)
            lines[custom_id] = json.dumps(item, ensure_ascii=False).encode("utf-8")
            built[custom_id] = b
        resp = await self.client.post(
            "/v1/messages/batches",
            body={"requests": requests},
            cast_to=httpx.Response,
        )
        data = resp.json()
        return P.BatchSubmission(
            str(data["id"]), _epoch(data.get("created_at")), lines, built
        )

    async def batch_state(self, batch_id: str) -> P.BatchState:
        resp = await self.client.get(
            f"/v1/messages/batches/{batch_id}", cast_to=httpx.Response
        )
        data = resp.json()
        status = str(data.get("processing_status"))
        return P.BatchState(status == "ended", _epoch(data.get("created_at")), status)

    async def batch_cancel(self, batch_id: str) -> None:
        await self.client.post(
            f"/v1/messages/batches/{batch_id}/cancel", cast_to=httpx.Response
        )

    async def batch_results(
        self, batch_id: str, sub: P.BatchSubmission
    ) -> dict[str, ModelResponse]:
        meta = (
            await self.client.get(
                f"/v1/messages/batches/{batch_id}", cast_to=httpx.Response
            )
        ).json()
        out: dict[str, ModelResponse] = {}
        url = meta.get("results_url")
        if url:
            content = (await self.client.get(url, cast_to=httpx.Response)).content
            for raw in content.splitlines():
                if not raw.strip():
                    continue
                item = json.loads(raw)
                cid = str(item.get("custom_id"))
                if cid not in sub.lines:
                    continue
                result = item.get("result") or {}
                kind = result.get("type")
                msg = result.get("message")
                if kind == "succeeded" and isinstance(msg, dict):
                    out[cid] = P.complete_response(
                        self.profile,
                        sub.built[cid],
                        sub.lines[cid],
                        200,
                        raw,
                        {"request-id": str(msg.get("id") or "")},
                        self.entry.get("region"),
                        0,
                        payload=msg,
                    )
                elif kind == "errored":
                    out[cid] = P.batch_item_failure(
                        sub.lines[cid], "batch_errored", raw
                    )
                else:  # expired or canceled
                    out[cid] = P.batch_item_failure(
                        sub.lines[cid], "batch_expired", raw
                    )
        for cid, line in sub.lines.items():
            if cid not in out:
                out[cid] = P.batch_item_failure(line, "batch_expired", None)
        return out


def _epoch(value: Any) -> float:
    """RFC 3339 → epoch seconds; numbers pass through."""
    from datetime import datetime

    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    return 0.0


def _sent(slot: list[bytes]) -> bytes:
    return slot[-1] if slot else b""
