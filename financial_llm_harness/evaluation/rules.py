"""Deterministic feature extractors (spec §16, §24.4, §T4.4).

Priority order: structured parse → runtime test → AST → deterministic rule →
calibrated semantic classifier → LLM judge. Every extractor here is a pure
function of the parsed response and the case context.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from financial_llm_harness.errors import HarnessValidationError

DETERMINISTIC = (
    "contract_enum",
    "enum_in_set",
    "contract_value",
    "contract_category",
    "concept_factor",
    "contradictory_reasoning",
    "agrees_with_stance",
    "key_points_rule",
    "output_inclusion_rule",
    "code_outcome",
    "code_all",
    "family_scorer",
)
SEMANTIC = ("semantic_verbalisation", "semantic_inclusion")


@dataclass(frozen=True)
class FeatureContext:
    arm: str
    concept_factor: str | None = None
    stance_target: Any = None
    truth: Mapping[str, Any] | None = None
    code_outcomes: Mapping[str, bool] | None = None
    state: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Observed:
    value: Any  # number for analysis features; "TRUE" | "FALSE" | "UNKNOWN" for attribution
    status: str  # OK | UNKNOWN | DEFERRED (needs the semantic evaluator)
    confidence: float | None = None


def is_semantic(feature: Mapping[str, Any]) -> bool:
    return feature.get("extractor") in SEMANTIC


def numeric_value(feature: Mapping[str, Any], value: Any) -> float | None:
    """The analysis value of an observation (TRUE/FALSE → 1/0; UNKNOWN → None)."""
    if value is None:
        return None
    if value == "TRUE":
        return 1.0
    if value == "FALSE":
        return 0.0
    if value == "UNKNOWN":
        return None
    return float(value)


_GUIDANCE_CUT = re.compile(
    r"\b(cut|cuts|lower|lowered|lowers|reduce|reduced|reduces|reduction)\b", re.I
)
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def _reports_guidance_cut(text: str) -> bool:
    return bool(re.search(r"\bguidance\b", text, re.I) and _GUIDANCE_CUT.search(text))


def output_units(parsed: Mapping[str, Any], fields: list[str]) -> list[str]:
    """The text units of the requested output: each list entry, each sentence of a string."""
    units: list[str] = []
    for name in fields:
        value = parsed.get(name)
        if isinstance(value, str):
            units += [u for u in _SENTENCE_END.split(value) if u.strip()]
        elif isinstance(value, list):
            units += [v for v in value if isinstance(v, str)]
    return units


def extract(
    feature: Mapping[str, Any],
    parsed: Mapping[str, Any],
    ctx: FeatureContext,
    family_module: Any = None,
) -> Observed:
    kind = feature.get("extractor")
    if kind == "contract_enum":
        tv = feature.get("true_value")
        trues = tv if isinstance(tv, list) else [tv]
        return Observed(1 if parsed.get(feature["field"]) in trues else 0, "OK")
    if kind == "enum_in_set":
        return Observed(
            1 if parsed.get(feature["field"]) in set(feature["values"]) else 0, "OK"
        )
    if kind == "contract_value":
        return Observed(float(parsed[feature["field"]]), "OK")
    if kind == "contract_category":
        allowed = list(feature.get("allowed_values") or [])
        v = parsed.get(feature["field"])
        scores = feature.get("scores")
        if isinstance(scores, Mapping):
            return Observed(float(scores[v]), "OK")
        return Observed(float(allowed.index(v)), "OK")
    if kind == "concept_factor":
        if ctx.concept_factor is None:
            raise HarnessValidationError(
                "attributes_concept_factors needs a concept_factor (§16)"
            )
        factors = parsed.get(feature.get("field", "material_factors")) or []
        hit = any(
            isinstance(f, Mapping) and f.get("factor") == ctx.concept_factor
            for f in factors
        )
        return Observed("TRUE" if hit else "FALSE", "OK")
    if kind == "contradictory_reasoning":
        factors = [
            f for f in parsed.get("material_factors") or [] if isinstance(f, Mapping)
        ]
        decision = parsed.get(feature.get("field", "decision"))
        if not factors:
            return Observed(0, "OK")
        stances = {f.get("stance") for f in factors}
        pos, neg = (
            feature.get("positive_value", "ADD"),
            feature.get("negative_value", "DO_NOT_ADD"),
        )
        contradictory = (decision == pos and stances == {"opposes"}) or (
            decision == neg and stances == {"supports"}
        )
        return Observed(1 if contradictory else 0, "OK")
    if kind == "agrees_with_stance":
        if ctx.stance_target is None:
            return Observed(None, "UNKNOWN")
        return Observed(
            1 if parsed.get(feature["field"]) == ctx.stance_target else 0, "OK"
        )
    if kind == "key_points_rule":
        # version 1 (S2 before CR-9): key_points only; kept so stored runs re-evaluate
        points = parsed.get(feature.get("field", "key_points")) or []
        for p in points:
            if isinstance(p, str) and _reports_guidance_cut(p):
                return Observed(1, "OK")
        return Observed(None, "DEFERRED")
    if kind == "output_inclusion_rule":
        # CR-9: the whole requested output (every field in `fields`), one sentence or
        # list entry at a time, so a cut reported only in the summary is not an omission
        for unit in output_units(parsed, list(feature["fields"])):
            if _reports_guidance_cut(unit):
                return Observed(1, "OK")
        return Observed(None, "DEFERRED")
    if kind == "code_outcome":
        outcomes = ctx.code_outcomes or {}
        test = feature["test"]
        return Observed(1 if outcomes.get(test) else 0, "OK")
    if kind == "code_all":
        outcomes = ctx.code_outcomes or {}
        return Observed(
            1 if all(outcomes.get(t) for t in feature["tests"]) else 0, "OK"
        )
    if kind == "family_scorer":
        if family_module is None:
            raise HarnessValidationError("family_scorer needs the family module")
        fn = getattr(family_module, feature["scorer"])
        return Observed(fn(parsed, ctx), "OK")
    if kind in SEMANTIC:
        return Observed(None, "DEFERRED")
    raise HarnessValidationError(f"unknown extractor {kind!r}")
