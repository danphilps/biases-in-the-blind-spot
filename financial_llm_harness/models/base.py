"""Adapter interface and provider registry (spec §20.1, §20.2, §20.7).

Provider differences are data in ``providers.yaml``: every request and response
mapping is a named, versioned request profile that the adapter interprets.
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.canonical_json import sha256_canonical
from financial_llm_harness.errors import HarnessValidationError

PROTOCOL_FOR_ADAPTER: dict[str, tuple[str, ...]] = {
    "openai_compatible": ("chat_completions",),
    "anthropic": ("anthropic_messages",),
    "gemini_native": ("gemini_generate_content",),
    "fake": ("in_process",),
}

FINISH_REASONS = ("stop", "length", "content_filter", "refusal", "tool", "other")


@dataclass(frozen=True)
class UpstreamRouting:
    provider: str
    quantization: str | None


@dataclass(frozen=True)
class ServingFingerprint:
    provider_model_id: str | None
    model_revision: str | None
    upstream_provider: str | None
    quantization: str | None
    region: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "provider_model_id": self.provider_model_id,
            "model_revision": self.model_revision,
            "upstream_provider": self.upstream_provider,
            "quantization": self.quantization,
            "region": self.region,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any] | None) -> ServingFingerprint | None:
        if d is None:
            return None
        return ServingFingerprint(
            d.get("provider_model_id"),
            d.get("model_revision"),
            d.get("upstream_provider"),
            d.get("quantization"),
            d.get("region"),
        )


@dataclass(frozen=True)
class ModelCapabilities:
    provider: str
    model: str
    region: str | None
    transport: str
    serving: str
    revision: str | None
    request_profile_id: str
    request_profile_sha256: str
    context_tokens: int
    max_output_tokens: int
    temperature_range: tuple[float, float] | None
    seed_parameter: str
    seed_reproducibility_rate: float | None
    supports_system: bool
    supports_schema: bool
    thinking_effective: str
    trace_visibility: str
    stream_required: bool
    upstream_routing: UpstreamRouting | None
    probe_serving_fingerprint: ServingFingerprint | None
    param_support: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "region": self.region,
            "transport": self.transport,
            "serving": self.serving,
            "revision": self.revision,
            "request_profile_id": self.request_profile_id,
            "request_profile_sha256": self.request_profile_sha256,
            "context_tokens": self.context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "temperature_range": list(self.temperature_range)
            if self.temperature_range is not None
            else None,
            "seed_parameter": self.seed_parameter,
            "seed_reproducibility_rate": self.seed_reproducibility_rate,
            "supports_system": self.supports_system,
            "supports_schema": self.supports_schema,
            "thinking_effective": self.thinking_effective,
            "trace_visibility": self.trace_visibility,
            "stream_required": self.stream_required,
            "upstream_routing": None
            if self.upstream_routing is None
            else {
                "provider": self.upstream_routing.provider,
                "quantization": self.upstream_routing.quantization,
            },
            "probe_serving_fingerprint": None
            if self.probe_serving_fingerprint is None
            else self.probe_serving_fingerprint.to_dict(),
            "param_support": [list(p) for p in self.param_support],
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> ModelCapabilities:
        ur = d.get("upstream_routing")
        tr = d.get("temperature_range")
        return ModelCapabilities(
            provider=d["provider"],
            model=d["model"],
            region=d.get("region"),
            transport=d["transport"],
            serving=d["serving"],
            revision=d.get("revision"),
            request_profile_id=d["request_profile_id"],
            request_profile_sha256=d["request_profile_sha256"],
            context_tokens=int(d["context_tokens"]),
            max_output_tokens=int(d["max_output_tokens"]),
            temperature_range=None if tr is None else (float(tr[0]), float(tr[1])),
            seed_parameter=d["seed_parameter"],
            seed_reproducibility_rate=d.get("seed_reproducibility_rate"),
            supports_system=bool(d["supports_system"]),
            supports_schema=bool(d["supports_schema"]),
            thinking_effective=d["thinking_effective"],
            trace_visibility=d["trace_visibility"],
            stream_required=bool(d["stream_required"]),
            upstream_routing=None
            if ur is None
            else UpstreamRouting(ur["provider"], ur.get("quantization")),
            probe_serving_fingerprint=ServingFingerprint.from_dict(
                d.get("probe_serving_fingerprint")
            ),
            param_support=tuple(
                (str(p[0]), str(p[1])) for p in d.get("param_support") or []
            ),
        )

    def param_status(self, name: str) -> str:
        for key, status in self.param_support:
            if key == name:
                return status
        return "accepted"


@dataclass(frozen=True)
class TestContext:
    """Fake-adapter context (§20.11); real adapters ignore it and never send it."""

    __test__ = False  # not a pytest class

    arm: str
    case_idx: int
    paraphrase_idx: int
    repetition: int
    seed: int
    block_seed: int


@dataclass
class ModelRequest:
    """One attempt's input: the frozen message object plus the execution envelope."""

    system: str
    user: str
    response_contract: dict[str, Any] | None
    max_output_tokens: int
    temperature: float | None
    seed: int | None
    enforcement: str  # provider_schema | prompt_only
    trial_id: str
    test_context: TestContext | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Returned for every attempt, including a transport failure (§20.7)."""

    request_body: bytes
    http_status: int | None
    transport_status: str  # §29 transport_status
    text: str | None
    reasoning_trace: str | None
    finish_reason: str | None
    finish_reason_raw: str | None
    usage: dict[str, int | None] | None
    effective_params: dict[str, Any] | None
    provider_request_id: str | None
    serving_fingerprint: ServingFingerprint | None
    raw_payload: bytes | None
    latency_ms: int
    refusal: bool = False
    usage_estimated: bool = False
    stream: bool = False
    retry_after_s: float | None = None
    error: str | None = None


class ModelAdapter(Protocol):
    def capabilities(self) -> ModelCapabilities: ...

    async def run(self, request: ModelRequest) -> ModelResponse: ...


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _registry_text() -> str:
    path = (
        importlib.resources.files("financial_llm_harness") / "models" / "providers.yaml"
    )
    return path.read_text(encoding="utf-8")


def load_providers() -> dict[str, Any]:
    """The parsed ``models/providers.yaml``."""
    data = yaml_io.loads(_registry_text())
    if not isinstance(data, dict) or "providers" not in data:
        raise HarnessValidationError("providers.yaml: missing 'providers'")
    return data


def provider_entry(provider: str) -> dict[str, Any]:
    providers = load_providers()["providers"]
    if provider not in providers:
        raise HarnessValidationError(f"provider {provider!r} is not in providers.yaml")
    return providers[provider]


def resolve_profile_id(
    provider: str, transport: str, request_profile: str | None
) -> str:
    """The panel entry's profile, or the provider's default for its transport (§20.2)."""
    entry = provider_entry(provider)
    if request_profile is not None:
        return request_profile
    if transport != "native":
        alt = (entry.get("alt_transports") or {}).get(transport)
        if alt is None:
            raise HarnessValidationError(
                f"provider {provider!r} has no transport {transport!r}"
            )
        return alt["default_profile"]
    return entry["default_profile"]


def request_profile(provider: str, profile_id: str) -> dict[str, Any]:
    profiles = provider_entry(provider).get("request_profiles") or {}
    if profile_id not in profiles:
        raise HarnessValidationError(
            f"request profile {profile_id!r} is not a profile of provider {provider!r}"
        )
    return profiles[profile_id]


def profile_sha256(provider: str, profile_id: str) -> str:
    return sha256_canonical(request_profile(provider, profile_id))


def base_url_for(provider: str, region: str | None, transport: str) -> str | None:
    entry = provider_entry(provider)
    if transport != "native":
        return (entry.get("alt_transports") or {})[transport]["base_url"]
    if region is not None:
        regions = entry.get("regions") or {}
        if region not in regions:
            raise HarnessValidationError(
                f"provider {provider!r} has no region {region!r}"
            )
        return regions[region]
    return entry.get("base_url")


def provider_limits(provider: str) -> dict[str, Any]:
    return dict(provider_entry(provider).get("limits") or {})


# ---------------------------------------------------------------------------
# Adapter construction and model listing (§20.1, §32 `models list`)
# ---------------------------------------------------------------------------


def build_adapter(
    entry: Mapping[str, Any],
    caps: ModelCapabilities,
    *,
    http_client: Any = None,
    key: str | None = None,
) -> ModelAdapter:
    """The adapter for the entry's resolved request profile; branches on ``protocol`` only (§20.1).

    Gemini's ``openai_compat`` transport has a ``chat_completions`` profile and so
    uses the ``openai_compatible`` adapter.
    """
    protocol = request_profile(entry["provider"], caps.request_profile_id)["protocol"]
    kind = {
        "chat_completions": "openai_compatible",
        "anthropic_messages": "anthropic",
        "gemini_generate_content": "gemini_native",
    }.get(protocol, provider_entry(entry["provider"])["adapter"])
    if kind == "openai_compatible":
        from financial_llm_harness.models.openai_compatible import (
            OpenAICompatibleAdapter,
        )

        return OpenAICompatibleAdapter(entry, caps, http_client=http_client, key=key)
    if kind == "anthropic":
        from financial_llm_harness.models.anthropic import AnthropicAdapter

        return AnthropicAdapter(entry, caps, http_client=http_client, key=key)
    if kind == "gemini_native":
        from financial_llm_harness.models.gemini_native import GeminiNativeAdapter

        return GeminiNativeAdapter(entry, caps, http_client=http_client, key=key)
    raise HarnessValidationError(
        f"provider {entry['provider']!r}: adapter {kind!r} is built by the runner"
    )


def unprobed_capabilities(entry: Mapping[str, Any]) -> ModelCapabilities:
    """Neutral capabilities for calls made before a probe exists (``models list``, the probe itself)."""
    provider = entry["provider"]
    pid = resolve_profile_id(provider, entry["transport"], entry.get("request_profile"))
    return ModelCapabilities(
        provider=provider,
        model=entry["model"],
        region=entry.get("region"),
        transport=entry["transport"],
        serving=str(provider_entry(provider).get("serving", "first_party")),
        revision=None,
        request_profile_id=pid,
        request_profile_sha256=profile_sha256(provider, pid),
        context_tokens=0,
        max_output_tokens=0,
        temperature_range=None,
        seed_parameter="accepted",
        seed_reproducibility_rate=None,
        supports_system=True,
        supports_schema=True,
        thinking_effective="not_controllable",
        trace_visibility="none",
        stream_required=False,
        upstream_routing=None,
        probe_serving_fingerprint=None,
        param_support=(),
    )


def list_models(provider: str, region: str | None = None) -> list[str]:
    """Model IDs from the provider's models endpoint; model IDs are never hard-coded (§20.2)."""
    import asyncio

    cfg = provider_entry(provider)
    if cfg["adapter"] == "fake":
        return ["a", "b", "eval"]
    entry = {
        "provider": provider,
        "model": "",
        "region": region,
        "transport": "native",
        "request_profile": None,
        "thinking": {"mode": "provider_default", "effort": None, "budget_tokens": None},
        "params": {},
        "upstream_routing": None,
    }
    adapter = build_adapter(entry, unprobed_capabilities(entry))
    return list(asyncio.run(adapter.list_model_ids()))  # type: ignore[attr-defined]
