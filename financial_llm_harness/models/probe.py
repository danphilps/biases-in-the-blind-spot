"""``financial-harness probe MODEL_KEY --panel NAME`` (spec §20.4).

At most 20 calls with fixed, neutral prompts under the entry's request profile.
Writes ``$HARNESS_HOME/probes/<model_key>/<YYYY-MM-DD>.json`` (committed, §0.9),
recording the profile ID, ``profile_version`` and sha256 it ran under, every
check, the call count and its own cost.

The probe prompts below are capability checks, not experimental stimuli: they
never reach a trial, a case or an oracle, and no label depends on their wording.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core.paths import probes_dir
from financial_llm_harness.errors import HarnessError, HarnessValidationError
from financial_llm_harness.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ServingFingerprint,
    UpstreamRouting,
    build_adapter,
    request_profile,
    unprobed_capabilities,
)

MAX_CALLS = 20
PROBE_SCHEMA_VERSION = 1
PROBE_SEED = 20260101
PROBE_TEMPERATURE_SEED_CHECK = 0.7  # the D-5 temperature (§20.4 check 4)
MAX_OUTPUT = 256

SYSTEM = "You are a test endpoint. Follow the instruction exactly."
USER = 'Reply with the JSON object {"answer": "ok"} and nothing else.'
SCHEMA_CONTRACT = {
    "contract_id": "probe_v1",
    "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["ok"]}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}
# Four high-entropy prompts: each admits many plausible continuations (§20.4 check 4).
ENTROPY_PROMPTS = (
    "Write 30 arbitrary English words separated by single spaces.",
    "List 25 random integers between 1 and 1000, separated by commas.",
    "Invent 20 fictional town names, one per line.",
    "Write a sequence of 40 random lowercase letters.",
)


class ProbeBudgetExceeded(HarnessError):
    """The probe would exceed its 20-call bound."""


@dataclass
class ProbeState:
    caps: ModelCapabilities
    calls: int = 0
    cost_usd: float = 0.0
    rejected: dict[str, str] = field(default_factory=dict)  # param name → error excerpt
    fingerprints: list[ServingFingerprint] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)


def _wire_names(name: str) -> list[str]:
    """Names under which a provider error may cite a parameter."""
    parts = [p for p in re.split(r"[.:]", name) if p]
    names = {name, *parts}
    aliases = {
        "max_output_tokens": ["max_tokens", "max_completion_tokens", "maxOutputTokens"],
        "response_schema": [
            "response_format",
            "output_config",
            "format",
            "responseSchema",
            "responseJsonSchema",
        ],
    }
    names.update(aliases.get(name, []))
    return sorted(n for n in names if len(n) > 2)


def named_params(message: str, sent: Mapping[str, Any], skip: set[str]) -> list[str]:
    low = message.lower()
    out = []
    for name in sent:
        if name in skip or name == "system_mode":
            continue
        if any(
            re.search(rf"\b{re.escape(w.lower())}\b", low) for w in _wire_names(name)
        ):
            out.append(name)
    return out


def _error_text(resp: ModelResponse) -> str:
    body = resp.raw_payload or b""
    return body.decode("utf-8", errors="replace")


class Prober:
    def __init__(
        self,
        entry: Mapping[str, Any],
        *,
        http_client: Any = None,
        key: str | None = None,
        price: Callable[[int, int, int], float] | None = None,
    ) -> None:
        self.entry = dict(entry)
        self.http_client = http_client
        self.key = key
        self.price = price
        caps = unprobed_capabilities(entry)
        if entry.get("upstream_routing"):
            pin = entry["upstream_routing"]
            caps = replace(
                caps,
                upstream_routing=UpstreamRouting(
                    pin["provider"], pin.get("quantization")
                ),
            )
        self.profile = request_profile(entry["provider"], caps.request_profile_id)
        if not self.profile.get("seed_field"):
            caps = replace(caps, seed_parameter="unavailable")
        self.state = ProbeState(caps)

    def _adapter(self) -> Any:
        return build_adapter(
            self.entry, self.state.caps, http_client=self.http_client, key=self.key
        )

    async def call(
        self,
        user: str,
        *,
        temperature: float | None,
        seed: int | None,
        schema: bool,
        label: str,
    ) -> ModelResponse:
        """One request; on an HTTP 400 naming parameters, mark them rejected and retry without them (check 7)."""
        while True:
            if self.state.calls >= MAX_CALLS:
                raise ProbeBudgetExceeded(f"probe would exceed {MAX_CALLS} calls")
            caps = self.state.caps
            send_seed = seed if caps.seed_parameter in ("accepted", "ignored") else None
            req = ModelRequest(
                system=SYSTEM,
                user=user,
                response_contract=SCHEMA_CONTRACT if schema else None,
                max_output_tokens=MAX_OUTPUT,
                temperature=temperature,
                seed=send_seed,
                enforcement="provider_schema"
                if schema and caps.supports_schema
                else "prompt_only",
                trial_id=f"probe/{label}/{self.state.calls}",
            )
            resp = await self._adapter().run(req)
            self.state.calls += 1
            self._record(label, resp)
            if resp.transport_status == "complete":
                if resp.serving_fingerprint is not None:
                    self.state.fingerprints.append(resp.serving_fingerprint)
                return resp
            if resp.http_status != 400:
                return resp
            text = _error_text(resp)
            if "stream" in text.lower() and not caps.stream_required:
                self.state.caps = replace(caps, stream_required=True)
                continue
            sent = self._sent_params(req)
            named = named_params(text, sent, set(self.state.rejected))
            if not named:
                if "system" in text.lower() and caps.supports_system:
                    self.state.caps = replace(caps, supports_system=False)
                    continue
                return resp
            for name in named:
                self.state.rejected[name] = text[:200]
            self._apply_rejections()

    def _sent_params(self, req: ModelRequest) -> dict[str, Any]:
        from financial_llm_harness.models.profile import build_body

        return build_body(
            self.profile, self.entry, self.state.caps, req
        ).effective_params

    def _apply_rejections(self) -> None:
        caps = self.state.caps
        support = dict(caps.param_support)
        for name in self.state.rejected:
            support[name] = "rejected"
        new = replace(caps, param_support=tuple(sorted(support.items())))
        if "seed" in self.state.rejected and new.seed_parameter != "unavailable":
            new = replace(new, seed_parameter="rejected")
        if "response_schema" in self.state.rejected:
            new = replace(new, supports_schema=False)
        self.state.caps = new

    def _record(self, label: str, resp: ModelResponse) -> None:
        cost = 0.0
        if (
            resp.transport_status == "complete"
            and self.price is not None
            and resp.usage
        ):
            cost = self.price(
                int(resp.usage.get("input") or 0),
                int(resp.usage.get("output") or 0),
                int(resp.usage.get("reasoning") or 0),
            )
        self.state.cost_usd += cost
        self.state.log.append(
            {
                "label": label,
                "transport_status": resp.transport_status,
                "http_status": resp.http_status,
                "finish_reason": resp.finish_reason,
            }
        )

    async def run(self) -> dict[str, Any]:
        s = self.state
        checks: dict[str, Any] = {}
        # 1. listing
        listed: bool | None
        try:
            s.calls += 1
            ids = await self._adapter().list_model_ids()  # type: ignore[attr-defined]
            listed = self.entry["model"] in ids
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            listed = None
            checks["listing_error"] = type(exc).__name__
        checks["listed"] = listed
        # 2, 3 and 7: system role, determinism at T = 0, parameter support
        temperature = 0.0
        texts: list[str | None] = []
        first_trace: str | None = None
        for i in range(3):
            resp = await self.call(
                USER,
                temperature=temperature,
                seed=PROBE_SEED,
                schema=False,
                label=f"t0_{i}",
            )
            if resp.transport_status != "complete":
                raise HarnessValidationError(
                    f"probe: {self.entry['model_key']} returned {resp.transport_status} "
                    f"(HTTP {resp.http_status}): {_error_text(resp)[:200]}"
                )
            texts.append(resp.text)
            if i == 0:
                first_trace = resp.reasoning_trace
        temperature_effective = (
            "provider_default" if "temperature" in s.rejected else "fixed"
        )
        checks["system_role"] = "accepted" if s.caps.supports_system else "inlined"
        checks["deterministic_at_t0"] = len(set(texts)) == 1
        checks["temperature_effective"] = temperature_effective
        # 4. seed reproducibility
        rate: float | None = None
        if s.caps.seed_parameter in ("accepted", "ignored"):
            same = 0
            for k, prompt in enumerate(ENTROPY_PROMPTS):
                outs = []
                for j in range(2):
                    r = await self.call(
                        prompt,
                        temperature=PROBE_TEMPERATURE_SEED_CHECK,
                        seed=PROBE_SEED + k,
                        schema=False,
                        label=f"seed_{k}_{j}",
                    )
                    outs.append(r.text if r.transport_status == "complete" else None)
                if outs[0] is not None and outs[0] == outs[1]:
                    same += 1
            if s.caps.seed_parameter in ("accepted", "ignored"):
                rate = same / len(ENTROPY_PROMPTS)
        s.caps = replace(s.caps, seed_reproducibility_rate=rate)
        # 5. structured output
        so_kind = (self.profile.get("structured_output") or {}).get("kind", "none")
        schema_enforced = False
        if so_kind != "none":
            r = await self.call(
                USER,
                temperature=temperature,
                seed=PROBE_SEED,
                schema=True,
                label="schema",
            )
            if r.transport_status == "complete" and s.caps.supports_schema:
                schema_enforced = _validates(r.text)
        s.caps = replace(s.caps, supports_schema=schema_enforced)
        checks["schema_enforced"] = schema_enforced
        # 6. thinking and trace visibility
        visibility = "none"
        if first_trace:
            visibility = (
                "full" if self.profile["protocol"] == "chat_completions" else "summary"
            )
        thinking = self.entry.get("thinking") or {}
        kind = (self.profile.get("thinking") or {}).get("kind", "none")
        thinking_rejected = any(
            k.startswith(
                (
                    "thinking",
                    "output_config",
                    "generationConfig.thinking",
                    "reasoning",
                    "enable_thinking",
                )
            )
            for k in s.rejected
        )
        mode = thinking.get("mode", "provider_default")
        if (
            kind in ("none", "model_variant")
            or mode == "provider_default"
            or thinking_rejected
        ):
            effective = "not_controllable"
        elif mode == "off":
            effective = "off" if visibility == "none" else "not_controllable"
        elif visibility == "summary":
            effective = "summary_only"
        else:
            effective = "on"
        s.caps = replace(
            s.caps, trace_visibility=visibility, thinking_effective=effective
        )
        # 9. serving
        fps = s.fingerprints
        first_fp = fps[0] if fps else None
        stable = (
            all(
                (f.model_revision, f.upstream_provider, f.quantization)
                == (
                    first_fp.model_revision,
                    first_fp.upstream_provider,
                    first_fp.quantization,
                )
                for f in fps
            )
            if first_fp
            else True
        )
        pin = self.entry.get("upstream_routing")
        if pin:
            for f in fps:
                if (
                    f.upstream_provider is None
                    or f.upstream_provider != pin["provider"]
                ):
                    raise HarnessValidationError(
                        f"probe: upstream provider {f.upstream_provider!r} breaks the pin "
                        f"{pin['provider']!r}; the routing pin is not effective (§20.8)"
                    )
                if pin.get("quantization") and f.quantization not in (
                    None,
                    pin["quantization"],
                ):
                    raise HarnessValidationError(
                        f"probe: quantisation {f.quantization!r} breaks the pin {pin['quantization']!r} (§20.8)"
                    )
        s.caps = replace(
            s.caps,
            probe_serving_fingerprint=first_fp,
            revision=first_fp.model_revision if first_fp else None,
        )
        checks["serving_fingerprint_stable"] = stable
        checks["stream_required"] = s.caps.stream_required
        checks["param_errors"] = dict(sorted(s.rejected.items()))
        checks["calls"] = list(s.log)
        # 7. final parameter support: every parameter the entry would send
        sent = self._sent_params(
            ModelRequest(
                system=SYSTEM,
                user=USER,
                response_contract=SCHEMA_CONTRACT,
                max_output_tokens=MAX_OUTPUT,
                temperature=0.0,
                seed=PROBE_SEED if self.profile.get("seed_field") else None,
                enforcement="provider_schema" if so_kind != "none" else "prompt_only",
                trial_id="probe/final",
            )
        )
        support = {
            name: ("rejected" if name in s.rejected else "accepted")
            for name in sent
            if name != "system_mode"
        }
        for name in s.rejected:
            support[name] = "rejected"
        s.caps = replace(s.caps, param_support=tuple(sorted(support.items())))
        return checks


def _validates(text: str | None) -> bool:
    if text is None:
        return False
    try:
        obj = json.loads(text)
    except ValueError:
        return False
    return (
        isinstance(obj, dict) and obj.get("answer") == "ok" and set(obj) == {"answer"}
    )


def probe_entry(
    entry: Mapping[str, Any],
    *,
    http_client: Any = None,
    key: str | None = None,
    clock: Callable[[], datetime] | None = None,
    out_root: Path | None = None,
) -> Path:
    """Run the probe for one panel entry and write its JSON file; returns the path."""
    from financial_llm_harness.execution import budget as BU
    from financial_llm_harness.generation.generate import utc_now

    if entry["provider"] == "fake":
        raise HarnessValidationError("fake models need no probe (§20.4)")
    pricing = BU.load_pricing()
    price = BU.price_for(pricing, entry["provider"], entry["model"])
    prober = Prober(
        entry,
        http_client=http_client,
        key=key,
        price=price.cost if price is not None else None,
    )
    checks = asyncio.run(prober.run())
    now = (clock or utc_now)()
    date = now.strftime("%Y-%m-%d")
    caps = prober.state.caps
    payload = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "model_key": entry["model_key"],
        "date": date,
        "request_profile_id": caps.request_profile_id,
        "request_profile_version": prober.profile.get("profile_version"),
        "request_profile_sha256": caps.request_profile_sha256,
        "capabilities": caps.to_dict(),
        "checks": checks,
        "calls": prober.state.calls,
        "cost_usd": prober.state.cost_usd,
    }
    root = out_root or probes_dir()
    path = root / entry["model_key"] / f"{date}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    B.atomic_write_text(path, json.dumps(payload, indent=2) + "\n")
    return path


def run_probe(model_key: str, panel: str) -> Path:
    """CLI entry: the entry (target or evaluator) in the named panel; a key absent from it exits 2."""
    from financial_llm_harness.models.panels import load_panel

    return probe_entry(load_panel(panel).entry(model_key))
