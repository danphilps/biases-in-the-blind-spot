"""Adapters: FakeAdapter (spec §20.11, E-18)."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from financial_llm_harness.core.ids import uint31
from financial_llm_harness.families import family_module
from financial_llm_harness.models.base import ModelRequest, TestContext
from financial_llm_harness.models.fake import (
    FakeAdapter,
    FakeEvaluator,
    FakeTask,
    fake_capabilities,
)
from tests.harness.helpers import ci_fake_entries

FAMILY = family_module("fundamental_judgement")


def _adapter(entry_idx: int = 0, **params: float) -> FakeAdapter:
    entry = dict(ci_fake_entries()[entry_idx])
    entry["params"] = {**entry["params"], **params}
    task = FakeTask(
        "stock_add_v1",
        "reasoning_before",
        "binary",
        (0.0, 1.0),
        "momentum",
        FAMILY.fake_response,
    )
    return FakeAdapter(entry, task)


def _req(arm: str, seed: int, block_seed: int | None = None) -> ModelRequest:
    return ModelRequest(
        system="s",
        user="u",
        response_contract=None,
        max_output_tokens=100,
        temperature=0.7,
        seed=seed,
        enforcement="provider_schema",
        trial_id="t",
        test_context=TestContext(
            arm, 0, 0, 0, seed, block_seed if block_seed is not None else seed
        ),
    )


def test_draws_follow_seed() -> None:
    ad = _adapter()
    u = np.random.default_rng(12345).random(3)
    assert ad.draws(_req("A", 12345)) == tuple(float(x) for x in u)


def test_binary_outcome_and_effect() -> None:
    ad = _adapter(invalid_rate=0.0)
    ya, yb = [], []
    for seed in (uint31("s", i) for i in range(2000)):
        ra = json.loads(ad.body_text(_req("A", seed)))
        rb = json.loads(ad.body_text(_req("B", seed)))
        ya.append(ra["decision"] == "ADD")
        yb.append(rb["decision"] == "ADD")
        if not ya[-1]:
            assert not yb[-1]  # shared U₁ with θ < 0: B never adds when A does not
    assert np.mean(ya) == pytest.approx(0.6, abs=0.04)
    assert np.mean(yb) == pytest.approx(0.3, abs=0.04)


def test_invalid_rate_gives_non_json() -> None:
    ad = _adapter(invalid_rate=1.0)
    with pytest.raises(json.JSONDecodeError):
        json.loads(ad.body_text(_req("A", 1)))


def test_a0_stability_reuses_arm_a_draw() -> None:
    ad = _adapter(a0_stability=1.0)
    u1_a = float(np.random.default_rng(777).random())
    assert ad.draws(_req("A0", 99, block_seed=777))[0] == u1_a
    ad0 = _adapter(a0_stability=0.0)
    g = np.random.default_rng(99)
    g.random()
    assert ad0.draws(_req("A0", 99, block_seed=777))[0] == float(g.random(3)[0])


def test_mention_rate() -> None:
    ad = _adapter(invalid_rate=0.0, factor_mention_rate=1.0)
    body = json.loads(ad.body_text(_req("A", 5)))
    assert any(f["factor"] == "momentum" for f in body["material_factors"])
    ad = _adapter(invalid_rate=0.0, factor_mention_rate=0.0)
    body = json.loads(ad.body_text(_req("A", 5)))
    assert all(f["factor"] != "momentum" for f in body["material_factors"])


def test_response_metadata_and_request_body() -> None:
    ad = _adapter()
    resp = asyncio.run(ad.run(_req("A", 3)))
    assert resp.finish_reason == "stop" and resp.latency_ms == 0
    assert resp.provider_request_id == "fake-t"
    assert (
        resp.serving_fingerprint is not None
        and resp.serving_fingerprint.model_revision == "fake-1"
    )
    assert (
        b"test_context" not in resp.request_body and b'"arm"' not in resp.request_body
    )
    assert resp.usage == {
        "input": 0,
        "output": len(resp.text or "") // 4,
        "reasoning": 0,
    }
    caps = fake_capabilities(ci_fake_entries()[0])
    assert caps.seed_parameter == "accepted" and caps.seed_reproducibility_rate == 1.0
    assert (
        caps.supports_schema
        and caps.supports_system
        and caps.trace_visibility == "none"
    )
    assert all(status == "accepted" for _, status in caps.param_support)


def test_fake_evaluator() -> None:
    ev = FakeEvaluator({"model": "eval", "params": {"verbalised_rate": 0.5}})
    yes = sum("YES" in ev.verdict(f"o{i}") for i in range(400))
    assert 150 < yes < 250
    u = float(np.random.default_rng(uint31("o1")).random())
    assert ("YES" in ev.verdict("o1")) == (u < 0.5)
