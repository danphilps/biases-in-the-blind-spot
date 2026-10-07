"""YAML 1.2 core-schema semantics (spec §0.11, E-28)."""

from __future__ import annotations

import math

import pytest
import yaml

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.schema import as_int
from financial_llm_harness.errors import HarnessValidationError


def test_yaml11_booleans_and_dates_are_strings() -> None:
    data = yaml_io.loads("a: off\nb: on\nc: yes\nd: no\ne: 2026-10-06\ny: y\nn: n\n")
    assert data == {
        "a": "off",
        "b": "on",
        "c": "yes",
        "d": "no",
        "e": "2026-10-06",
        "y": "y",
        "n": "n",
    }


def test_core_types_kept() -> None:
    data = yaml_io.loads("t: true\nf: false\nT: True\nn: null\nm: ~\ne:\nN: NULL\n")
    assert data == {
        "t": True,
        "f": False,
        "T": True,
        "n": None,
        "m": None,
        "e": None,
        "N": None,
    }


def test_numbers() -> None:
    data = yaml_io.loads(
        "a: 1:20\nb: 1_000\nc: 010\nd: 0o17\ne: 0x1F\nf: 1e5\ng: -.5\nh: .inf\ni: .nan\nj: +3\n"
    )
    assert data["a"] == "1:20" and data["b"] == "1_000"
    assert data["c"] == 10 and isinstance(data["c"], int)
    assert data["d"] == 15 and data["e"] == 31
    assert data["f"] == 1e5 and isinstance(data["f"], float)
    assert data["g"] == -0.5 and data["h"] == math.inf and math.isnan(data["i"])
    assert data["j"] == 3


@pytest.mark.parametrize(
    "text",
    [
        "a: !!timestamp 2026-10-06",
        "a: !!python/object:os.system x",
        "a: !!binary aGk=",
        "a: !!set {x}",
        "<<: !!merge {a: 1}",
    ],
)
def test_forbidden_tags_exit_2(text: str) -> None:
    with pytest.raises(HarnessValidationError) as exc:
        yaml_io.loads(text)
    assert exc.value.exit_code == 2


def test_allowed_explicit_tags() -> None:
    data = yaml_io.loads(
        "a: !!str 10\nb: !!int '7'\nc: !!float '2'\nd: !!bool 'true'\ne: !!null ''\n"
    )
    assert data == {"a": "10", "b": 7, "c": 2.0, "d": True, "e": None}


def test_merge_key_not_resolved() -> None:
    assert yaml_io.loads("<<: x\n") == {"<<": "x"}


def test_validator_rejects_bool_for_int() -> None:
    value = yaml_io.loads("n: true\n")["n"]
    with pytest.raises(HarnessValidationError):
        as_int(value, "n")


def test_round_trip_under_both_loaders() -> None:
    data = {
        "s": ["off", "on", "2026-10-06", "1e5", "0o17", "yes", "010", "1:20"],
        "f": 1e20,
        "g": 1e-5,
        "n": None,
        "b": False,
    }
    text = yaml_io.dump(data)
    assert yaml_io.loads(text) == data
    assert yaml.safe_load(text) == data
