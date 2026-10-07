"""Fake adapter (spec §20.11, E-18).

Deterministic, in-process, with a configurable effect. It draws U₁, U₂, U₃ from
``default_rng(test_context.seed)``; arm A0 first draws V and, if
V < ``a0_stability``, reuses arm A's U₁ from ``default_rng(block_seed)``. The
coupling of a block's arms through the matched requested seed is a property of
the fake, not an assumption about real providers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from financial_llm_harness.core.canonical_json import canonical
from financial_llm_harness.core.ids import uint31
from financial_llm_harness.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ServingFingerprint,
    profile_sha256,
)

FAKE_PROFILE = "fake/in_process"


def fake_capabilities(entry: Mapping[str, Any]) -> ModelCapabilities:
    """The fixed capabilities of a fake model (§20.11)."""
    params = sorted(
        {
            "temperature",
            "top_p",
            "seed",
            "max_output_tokens",
            *map(str, (entry.get("params") or {})),
        }
    )
    return ModelCapabilities(
        provider="fake",
        model=str(entry["model"]),
        region=entry.get("region"),
        transport=str(entry.get("transport", "native")),
        serving="in_process",
        revision="fake-1",
        request_profile_id=FAKE_PROFILE,
        request_profile_sha256=profile_sha256("fake", FAKE_PROFILE),
        context_tokens=1_000_000,
        max_output_tokens=1_000_000,
        temperature_range=(0.0, 2.0),
        seed_parameter="accepted",
        seed_reproducibility_rate=1.0,
        supports_system=True,
        supports_schema=True,
        thinking_effective="not_controllable",
        trace_visibility="none",
        stream_required=False,
        upstream_routing=None,
        probe_serving_fingerprint=ServingFingerprint(
            str(entry["model"]), "fake-1", None, None, None
        ),
        param_support=tuple((p, "accepted") for p in params),
    )


@dataclass(frozen=True)
class FakeTask:
    """What the fake needs about the experiment to render a contract-valid body."""

    contract_id: str
    reasoning_mode: str
    feature_type: str  # binary | scalar | ordinal | categorical
    feature_range: tuple[float, float]
    concept_factor: str | None
    fake_response: Callable[..., str]
    pc_sign: int | None = None  # sign of the positive control's DIRECTIONAL oracle


class FakeAdapter:
    """In-process adapter for tests and CI."""

    def __init__(self, entry: Mapping[str, Any], task: FakeTask | None) -> None:
        self.entry = dict(entry)
        self.params = dict(entry.get("params") or {})
        self.task = task
        self._caps = fake_capabilities(entry)

    def capabilities(self) -> ModelCapabilities:
        return self._caps

    def _rate(self, arm: str) -> float:
        base = float(self.params.get("base_rate", 0.5))
        if arm == "B":
            return base + float(self.params.get("theta", 0.0))
        if arm == "PCB":
            pc = float(self.params.get("pc_theta", 0.0))
            # §20.11 sets PCB at base_rate + pc_theta; the step is applied in the direction the
            # control's oracle requires, so a DIRECTIONAL(+) control (S3–S5) is not reversed.
            sign = self.task.pc_sign if self.task is not None else None
            return base + (abs(pc) * sign if sign in (-1, 1) else pc)
        return base

    def draws(self, request: ModelRequest) -> tuple[float, float, float]:
        ctx = request.test_context
        assert ctx is not None, "the runner always fills test_context"
        g = np.random.default_rng(ctx.seed)
        if ctx.arm == "A0":
            v = float(g.random())
            u1, u2, u3 = (float(x) for x in g.random(3))
            if v < float(self.params.get("a0_stability", 0.0)):
                u1 = float(np.random.default_rng(ctx.block_seed).random())
            return u1, u2, u3
        u1, u2, u3 = (float(x) for x in g.random(3))
        return u1, u2, u3

    def body_text(self, request: ModelRequest) -> str:
        assert self.task is not None and request.test_context is not None
        u1, u2, u3 = self.draws(request)
        if u2 < float(self.params.get("invalid_rate", 0.0)):
            return "I cannot provide a structured answer to this request."
        p = self._rate(request.test_context.arm)
        lo, hi = self.task.feature_range
        if self.task.feature_type == "binary":
            y: int | float = 1 if u1 < min(max(p, 0.0), 1.0) else 0
        else:
            sd = float(self.params.get("sd", 1.0))
            y = float(np.clip(p + sd * float(stats.norm.ppf(u1)), lo, hi))
        mention = u3 < float(self.params.get("factor_mention_rate", 0.5))
        rng = np.random.default_rng(request.test_context.seed + 1)
        return self.task.fake_response(
            self.task.contract_id,
            y,
            mention,
            rng,
            concept_factor=self.task.concept_factor,
            reasoning_mode=self.task.reasoning_mode,
        )

    def request_body(self, request: ModelRequest) -> bytes:
        """Canonical JSON of the frozen message, effective parameters and seed; never test_context."""
        return canonical(
            {
                "messages": {
                    "system": request.system,
                    "user": request.user,
                    "response_contract": request.response_contract,
                },
                "params": {
                    "max_output_tokens": request.max_output_tokens,
                    "temperature": request.temperature,
                },
                "seed": request.seed,
                "model": self.entry["model"],
            }
        )

    async def run(self, request: ModelRequest) -> ModelResponse:
        body = self.request_body(request)
        text = self.body_text(request)
        usage: dict[str, int | None] = {
            "input": (len(request.system) + len(request.user)) // 4,
            "output": len(text) // 4,
            "reasoning": 0,
        }
        raw = json.dumps(
            {
                "model": self.entry["model"],
                "text": text,
                "finish_reason": "stop",
                "usage": usage,
            }
        ).encode()
        assert request.test_context is not None
        return ModelResponse(
            request_body=body,
            http_status=200,
            transport_status="complete",
            text=text,
            reasoning_trace=None,
            finish_reason="stop",
            finish_reason_raw="stop",
            usage=usage,
            effective_params={
                "temperature": request.temperature,
                "seed": request.seed,
                "max_output_tokens": request.max_output_tokens,
            },
            provider_request_id=f"fake-{request.trial_id}",
            serving_fingerprint=ServingFingerprint(
                str(self.entry["model"]), "fake-1", None, None, None
            ),
            raw_payload=raw,
            latency_ms=0,
        )


class FakeEvaluator:
    """Fake semantic evaluator: YES when U₁ < verbalised_rate, U₁ from ``uint31(observation_id)``."""

    def __init__(self, entry: Mapping[str, Any]) -> None:
        self.entry = dict(entry)
        self.rate = float((entry.get("params") or {}).get("verbalised_rate", 0.5))
        self._caps = fake_capabilities(entry)

    def capabilities(self) -> ModelCapabilities:
        return self._caps

    def verdict(self, observation_id: str) -> str:
        u1 = float(np.random.default_rng(uint31(observation_id)).random())
        if u1 < self.rate:
            return "<verbalized>YES</verbalized><witness>fake</witness>"
        return "<verbalized>NO</verbalized>"

    async def run(self, request: ModelRequest) -> ModelResponse:
        obs_id = str(request.extra.get("observation_id"))
        # U₁ from the design-level observation key when the runner supplies one (semantic.py)
        text = self.verdict(str(request.extra.get("seed_key") or obs_id))
        body = canonical(
            {
                "messages": {"system": request.system, "user": request.user},
                "seed": request.seed,
                "model": self.entry["model"],
            }
        )
        usage: dict[str, int | None] = {
            "input": (len(request.system) + len(request.user)) // 4,
            "output": len(text) // 4,
            "reasoning": 0,
        }
        return ModelResponse(
            request_body=body,
            http_status=200,
            transport_status="complete",
            text=text,
            reasoning_trace=None,
            finish_reason="stop",
            finish_reason_raw="stop",
            usage=usage,
            effective_params={"temperature": request.temperature, "seed": request.seed},
            provider_request_id=f"fake-{obs_id}",
            serving_fingerprint=ServingFingerprint(
                str(self.entry["model"]), "fake-1", None, None, None
            ),
            raw_payload=json.dumps({"text": text}).encode(),
            latency_ms=0,
        )
