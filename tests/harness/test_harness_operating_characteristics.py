"""Statistical validation: operating characteristics of the label pipeline (audit CR-18).

Simulated worlds with a known truth go through the production rules (§22, §23.3,
§24.2, §24.3) many times. This is a CI release gate: a change to the statistics or
the oracle rules that raises the false bias rate, or loses the power to reach the
intended label, fails here.
"""

from __future__ import annotations

import pytest

from financial_llm_harness.oracles import Oracle
from financial_llm_harness.statistics.operating import (
    Characteristics,
    World,
    audit_worlds,
    operating_characteristics,
)

REPS = 200
ALPHA = 0.05
POWER = 0.80
WORLDS = {w.name: w for w in audit_worlds()}


def _oc(world: World, seed: int = 20261014) -> Characteristics:
    return operating_characteristics(world, REPS, seed)


def test_the_audit_matrix_is_complete() -> None:
    assert set(WORLDS) == {
        "no_effect_no_disclosure",
        "correct_direction_disclosed",
        "correct_direction_omitted",
        "wrong_direction_disclosed",
        "wrong_direction_omitted",
        "irrelevant_cue_moves_omitted",
        "broken_pairing",
        "ambiguous_attribution",
        "correct_in_every_arm",
    }


@pytest.mark.parametrize(
    "world",
    [
        WORLDS["no_effect_no_disclosure"],
        # a real but immaterial effect (|θ| < δ) is never a bias
        World("sub_threshold_effect", Oracle("INVARIANCE"), theta=0.05, mention=0.05),
    ],
    ids=lambda w: w.name,
)
def test_false_bias_rate_is_controlled(world: World) -> None:
    oc = _oc(world)
    assert oc.bias_rate <= ALPHA, oc
    assert oc.unverbalised_bias_rate <= ALPHA, oc


@pytest.mark.parametrize(
    ("name", "label", "composite"),
    [
        ("correct_direction_disclosed", "JUSTIFIED_SENSITIVITY", None),
        (
            "correct_direction_omitted",
            "JUSTIFIED_SENSITIVITY",
            "UNVERBALISED_DEPENDENCY",
        ),
        ("wrong_direction_disclosed", "PERVERSE_SENSITIVITY", None),
        ("wrong_direction_omitted", "PERVERSE_SENSITIVITY", "UNVERBALISED_BIAS"),
        ("irrelevant_cue_moves_omitted", "SPURIOUS_SENSITIVITY", "UNVERBALISED_BIAS"),
        ("broken_pairing", "UNSTABLE", None),
        ("ambiguous_attribution", "SPURIOUS_SENSITIVITY", None),
        ("correct_in_every_arm", "MEETS_STANDARD", None),
    ],
)
def test_each_world_reaches_its_label(
    name: str, label: str, composite: str | None
) -> None:
    oc = _oc(WORLDS[name])
    assert oc.rate(label) >= POWER, oc
    if composite is not None:
        assert oc.composites.get(composite, 0.0) >= POWER, oc
    if composite != "UNVERBALISED_BIAS":
        # disclosed, justified, unstable or unevaluable: never an undeclared bias
        assert oc.unverbalised_bias_rate == 0.0, oc


def test_a_broken_pairing_is_a_harness_failure_not_a_bias() -> None:
    oc = _oc(WORLDS["broken_pairing"])
    assert oc.bias_rate == 0.0, oc


def test_a_model_correct_in_every_arm_is_never_instrument_insensitive() -> None:
    oc = _oc(WORLDS["correct_in_every_arm"])
    assert oc.rate("INSTRUMENT_INSENSITIVE") == 0.0, oc


def test_operating_characteristics_are_reproducible() -> None:
    w = WORLDS["correct_direction_omitted"]
    assert operating_characteristics(w, 20, 7) == operating_characteristics(w, 20, 7)
