"""Unit: parsers, fuzzed with malformed JSON and multiple fences (spec §13.2, §14)."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from financial_llm_harness.evaluation.parse import (
    MAX_CODE_BYTES,
    contract_from_dict,
    parse_code_response,
    parse_json_response,
    strip_fence,
)
from financial_llm_harness.families import load_contract

CONTRACT = contract_from_dict(
    load_contract("fundamental_judgement", "stock_add_v1"), "stock_add_v1"
)
GOOD = {
    "reasoning": "x",
    "decision": "ADD",
    "confidence": 0.7,
    "material_factors": [{"factor": "momentum", "stance": "supports"}],
}


def _p(text: str, abstain: bool = False) -> str:
    return parse_json_response(CONTRACT, "reasoning_before", abstain, text).status


def test_valid_and_single_fence() -> None:
    assert _p(json.dumps(GOOD)) == "SUCCESS"
    assert _p("```json\n" + json.dumps(GOOD) + "\n```") == "SUCCESS"
    assert _p("```\n" + json.dumps(GOOD) + "\n```") == "SUCCESS"


def test_multiple_fences_and_prose_fail() -> None:
    doc = json.dumps(GOOD)
    assert _p(f"```json\n```json\n{doc}\n```\n```") == "PARSE_FAILURE"
    assert _p(f"Here you go:\n```json\n{doc}\n```") == "PARSE_FAILURE"
    assert _p(f"```json\n{doc}\n```\n```json\n{doc}\n```") == "PARSE_FAILURE"
    assert _p(doc + " trailing") == "PARSE_FAILURE"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("decision"),
        lambda d: d.update(extra=1),
        lambda d: d.update(decision="BUY"),
        lambda d: d.update(confidence=1.5),
        lambda d: d.update(confidence=True),
        lambda d: d.update(confidence="0.5"),
        lambda d: d.update(
            material_factors=[{"factor": "vibes", "stance": "supports"}]
        ),
        lambda d: d.update(material_factors=[{"factor": "momentum"}]),
        lambda d: d.update(
            material_factors={"factor": "momentum", "stance": "supports"}
        ),
        lambda d: d.update(reasoning=3),
    ],
)
def test_contract_violations_are_parse_failures(mutate) -> None:  # type: ignore[no-untyped-def]
    d = json.loads(json.dumps(GOOD))
    mutate(d)
    assert _p(json.dumps(d)) == "PARSE_FAILURE"


def test_empty_and_abstention() -> None:
    assert _p("") == "EMPTY_RESPONSE"
    assert _p("   ") == "EMPTY_RESPONSE"
    d = dict(GOOD, decision="ABSTAIN")
    assert _p(json.dumps(d)) == "REFUSAL"
    assert _p("[1, 2]") == "PARSE_FAILURE"
    assert _p("NaN") == "PARSE_FAILURE"


def test_strip_fence_at_most_one() -> None:
    assert strip_fence("```json\n{}\n```") == "{}"
    assert strip_fence("```\n```\n{}\n```\n```").startswith("```")


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=400))
def test_fuzz_never_crashes(text: str) -> None:
    assert _p(text) in ("SUCCESS", "PARSE_FAILURE", "EMPTY_RESPONSE", "REFUSAL")
    assert parse_code_response(text).status in (
        "SUCCESS",
        "PARSE_FAILURE",
        "EMPTY_RESPONSE",
    )


@settings(max_examples=200, deadline=None)
@given(
    st.recursive(
        st.none() | st.booleans() | st.floats(allow_nan=False) | st.text(max_size=10),
        lambda c: st.lists(c, max_size=3)
        | st.dictionaries(st.text(max_size=8), c, max_size=4),
        max_leaves=12,
    )
)
def test_fuzz_json_shapes(obj: object) -> None:
    assert _p(json.dumps(obj)) in ("SUCCESS", "PARSE_FAILURE", "REFUSAL")


def test_code_protocol() -> None:
    assert parse_code_response("```python\nx = 1\n```").status == "SUCCESS"
    assert parse_code_response("```python\nx = 1\n```").code == "x = 1\n"
    assert parse_code_response("no code").status == "PARSE_FAILURE"
    assert (
        parse_code_response("```python\na\n```\n```python\nb\n```").status
        == "PARSE_FAILURE"
    )
    assert parse_code_response("```py\na\n```").status == "PARSE_FAILURE"
    assert parse_code_response("```\na\n```").status == "PARSE_FAILURE"
    big = "```python\n" + "x" * (MAX_CODE_BYTES + 1) + "\n```"
    assert parse_code_response(big).status == "PARSE_FAILURE"
