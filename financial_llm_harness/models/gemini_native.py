"""``gemini_native`` adapter: ``POST {base_url}/models/{model}:generateContent`` via ``httpx`` (spec §20.1, §20.10).

The key travels only in the ``x-goog-api-key`` header, never in the URL or body,
so no request artefact contains it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

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


class GeminiNativeAdapter:
    """generateContent adapter driven by the model's frozen request profile."""

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
        if self.profile["protocol"] != "gemini_generate_content":
            raise ValueError(
                f"profile {caps.request_profile_id} is not gemini_generate_content"
            )
        self.timeout = float(
            provider_limits(entry["provider"]).get("request_timeout_s", 600)
        )
        self.base_url = str(
            base_url_for(entry["provider"], entry.get("region"), entry["transport"])
        ).rstrip("/")
        self.key = key if key is not None else (P.api_key(cfg) or "")
        self.client = P.attach_capture(
            http_client or httpx.AsyncClient(timeout=self.timeout)
        )

    def capabilities(self) -> ModelCapabilities:
        return self.caps

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.key, "content-type": "application/json"}

    async def run(self, request: ModelRequest) -> ModelResponse:
        built = P.build_body(self.profile, self.entry, self.caps, request)
        data = json.dumps(built.body, ensure_ascii=False).encode("utf-8")
        url = f"{self.base_url}/models/{self.entry['model']}:generateContent"
        sent = P.new_capture()
        t0 = time.monotonic()
        try:
            resp = await self.client.post(
                url, content=data, headers=self._headers(), timeout=self.timeout
            )
        except httpx.TimeoutException as exc:
            return P.failure_response(
                _sent(sent, data),
                "timeout",
                latency_ms=P.elapsed_ms(t0),
                error=str(exc),
            )
        except httpx.TransportError as exc:
            return P.failure_response(
                _sent(sent, data),
                "connection_error",
                latency_ms=P.elapsed_ms(t0),
                error=str(exc),
            )
        if resp.status_code >= 400:
            return P.failure_response(
                _sent(sent, data),
                "http_error",
                http_status=resp.status_code,
                body=resp.content,
                headers=resp.headers,
                request_id_value=P.request_id(self.profile, resp.headers),
                latency_ms=P.elapsed_ms(t0),
            )
        return P.complete_response(
            self.profile,
            built,
            _sent(sent, data),
            resp.status_code,
            resp.content,
            resp.headers,
            self.entry.get("region"),
            P.elapsed_ms(t0),
        )

    async def list_model_ids(self) -> list[str]:
        ids: list[str] = []
        token: str | None = None
        while True:
            params: dict[str, Any] = {"pageSize": 1000}
            if token:
                params["pageToken"] = token
            resp = await self.client.get(
                f"{self.base_url}/models", params=params, headers=self._headers()
            )
            resp.raise_for_status()
            data = resp.json()
            ids += [
                str(m["name"]).removeprefix("models/") for m in data.get("models") or []
            ]
            token = data.get("nextPageToken")
            if not token:
                return sorted(ids)


def _sent(slot: list[bytes], fallback: bytes) -> bytes:
    return slot[-1] if slot else fallback
