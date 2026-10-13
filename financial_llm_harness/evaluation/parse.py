"""Response contracts and parsing (spec §13.2, §14).

JSON contracts: strip at most one surrounding code fence, ``json.loads`` the
rest, then validate keys, enums and ranges. Any failure is PARSE_FAILURE and
the raw response is kept; partial salvage is forbidden. ``code_v1``: exactly
one fenced block with info string ``python``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from financial_llm_harness.errors import HarnessValidationError

MAX_CODE_BYTES = 262_144
_FENCE = re.compile(r"^```[^\n`]*\n(.*)\n```\s*$", re.DOTALL)
_PY_BLOCK = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Contract:
    contract_id: str
    version: int
    kind: str  # json | code_v1
    fields: dict[str, Any]
    decision_fields: tuple[str, ...]
    free_text: dict[str, str | None]
    instructions: dict[str, str]
    abstain_value: str | None

    def keys(self, mode: str) -> list[str]:
        ft = self.free_text.get(mode)
        if mode == "reasoning_before":
            return ([ft] if ft else []) + list(self.decision_fields)
        if mode == "rationale_after":
            return list(self.decision_fields) + ([ft] if ft else [])
        return list(self.decision_fields)

    def free_text_key(self, mode: str) -> str | None:
        return self.free_text.get(mode)

    def instruction_text(self, mode: str, abstain_allowed: bool) -> str:
        key = (
            f"{mode}+abstain"
            if abstain_allowed and f"{mode}+abstain" in self.instructions
            else mode
        )
        if key not in self.instructions:
            raise HarnessValidationError(
                f"contract {self.contract_id}: no instructions for {key!r}"
            )
        if (
            abstain_allowed
            and key == mode
            and self.kind == "json"
            and self.abstain_value
        ):
            raise HarnessValidationError(
                f"contract {self.contract_id}: abstain_allowed needs '{mode}+abstain' instructions"
            )
        return self.instructions[key]

    def enum_values(self, name: str, abstain_allowed: bool) -> list[str]:
        spec = self.fields[name]
        values = list(spec["values"])
        if abstain_allowed and self.abstain_value and name == self.decision_fields[0]:
            values.append(self.abstain_value)
        return values

    def json_schema(self, mode: str, abstain_allowed: bool) -> dict[str, Any] | None:
        if self.kind != "json":
            return None
        props: dict[str, Any] = {}
        for k in self.keys(mode):
            spec = self.fields[k]
            values = (
                self.enum_values(k, abstain_allowed) if spec["type"] == "enum" else None
            )
            props[k] = self._field_schema(k, spec, values)
        return {
            "type": "object",
            "properties": props,
            "required": self.keys(mode),
            "additionalProperties": False,
        }

    def _field_schema(
        self, name: str, spec: Mapping[str, Any], enum_values: list[str] | None = None
    ) -> dict[str, Any]:
        t = spec["type"]
        if t == "string":
            return {"type": "string"}
        if t == "enum":
            return {
                "type": "string",
                "enum": enum_values
                if enum_values is not None
                else list(spec["values"]),
            }
        if t == "number":
            out: dict[str, Any] = {"type": "number"}
            if "min" in spec:
                out["minimum"] = spec["min"]
            if "max" in spec:
                out["maximum"] = spec["max"]
            return out
        if t == "integer":
            out = {"type": "integer"}
            if "min" in spec:
                out["minimum"] = spec["min"]
            if "max" in spec:
                out["maximum"] = spec["max"]
            return out
        if t == "boolean":
            return {"type": "boolean"}
        if t == "string_list":
            out = {"type": "array", "items": {"type": "string"}}
            if "max_items" in spec:
                out["maxItems"] = spec["max_items"]
            return out
        if t in ("object_list", "object"):
            item = spec["item"] if t == "object_list" else spec["fields"]
            inner = {
                "type": "object",
                "properties": {k: self._field_schema(k, v) for k, v in item.items()},
                "required": list(item),
                "additionalProperties": False,
            }
            if t == "object":
                return inner
            out = {"type": "array", "items": inner}
            if "max_items" in spec:
                out["maxItems"] = spec["max_items"]
            return out
        raise HarnessValidationError(f"contract field {name!r}: unknown type {t!r}")

    def response_contract(self, mode: str, abstain_allowed: bool) -> dict[str, Any]:
        """The frozen ``response_contract`` of a rendering (§0.10)."""
        return {
            "contract_id": self.contract_id,
            "version": self.version,
            "kind": self.kind,
            "reasoning_mode": mode,
            "abstain_allowed": abstain_allowed,
            "schema": self.json_schema(mode, abstain_allowed),
        }


def contract_from_dict(data: Mapping[str, Any], where: str) -> Contract:
    for key in ("contract_id", "version", "kind", "instructions"):
        if key not in data:
            raise HarnessValidationError(f"{where}: missing key {key!r}")
    kind = data["kind"]
    if kind not in ("json", "code_v1"):
        raise HarnessValidationError(f"{where}: contract kind {kind!r} invalid")
    return Contract(
        contract_id=str(data["contract_id"]),
        version=int(data["version"]),
        kind=kind,
        fields=dict(data.get("fields") or {}),
        decision_fields=tuple(data.get("decision_fields") or ()),
        free_text=dict(data.get("free_text") or {}),
        instructions=dict(data["instructions"]),
        abstain_value=data.get("abstain_value"),
    )


@dataclass(frozen=True)
class ParseResult:
    status: str  # SUCCESS | PARSE_FAILURE | REFUSAL | EMPTY_RESPONSE
    data: dict[str, Any] | None
    error: str | None = None
    code: str | None = None


def strip_fence(text: str) -> str:
    """Strip at most one surrounding code fence."""
    s = text.strip()
    m = _FENCE.match(s)
    return m.group(1) if m else s


def _check_value(
    name: str, spec: Mapping[str, Any], value: Any, enum_values: list[str] | None = None
) -> str | None:
    t = spec["type"]
    if t == "string":
        if not isinstance(value, str):
            return f"{name}: expected a string"
        if "max_words" in spec and len(value.split()) > int(spec["max_words"]):
            return f"{name}: more than {spec['max_words']} words"
        return None
    if t == "enum":
        allowed = enum_values if enum_values is not None else list(spec["values"])
        return (
            None
            if isinstance(value, str) and value in allowed
            else f"{name}: {value!r} not in {allowed}"
        )
    if t in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"{name}: expected a number"
        if t == "integer" and not isinstance(value, int):
            return f"{name}: expected an integer"
        if not math.isfinite(float(value)):
            return f"{name}: not finite"
        if "min" in spec and value < spec["min"]:
            return f"{name}: below {spec['min']}"
        if "max" in spec and value > spec["max"]:
            return f"{name}: above {spec['max']}"
        return None
    if t == "boolean":
        return None if isinstance(value, bool) else f"{name}: expected a boolean"
    if t == "string_list":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            return f"{name}: expected a list of strings"
        if "max_items" in spec and len(value) > spec["max_items"]:
            return f"{name}: more than {spec['max_items']} items"
        return None
    if t in ("object_list", "object"):
        items = value if t == "object_list" else [value]
        if t == "object_list" and not isinstance(value, list):
            return f"{name}: expected a list"
        if (
            t == "object_list"
            and "max_items" in spec
            and len(value) > spec["max_items"]
        ):
            return f"{name}: more than {spec['max_items']} items"
        sub = spec["item"] if t == "object_list" else spec["fields"]
        for it in items:
            if not isinstance(it, dict) or set(it) != set(sub):
                return f"{name}: item keys must be exactly {sorted(sub)}"
            for k, v in it.items():
                err = _check_value(f"{name}.{k}", sub[k], v)
                if err:
                    return err
        return None
    return f"{name}: unknown type {t}"


def parse_json_response(
    contract: Contract, mode: str, abstain_allowed: bool, text: str | None
) -> ParseResult:
    if text is None or text.strip() == "":
        return ParseResult("EMPTY_RESPONSE", None, "empty text")
    try:
        obj = json.loads(strip_fence(text))
    except (json.JSONDecodeError, ValueError) as exc:
        return ParseResult("PARSE_FAILURE", None, f"invalid JSON: {exc}")
    if not isinstance(obj, dict):
        return ParseResult("PARSE_FAILURE", None, "top level is not an object")
    keys = contract.keys(mode)
    if set(obj) != set(keys):
        return ParseResult(
            "PARSE_FAILURE", None, f"keys {sorted(obj)} != contract keys {sorted(keys)}"
        )
    decision_key = contract.decision_fields[0] if contract.decision_fields else None
    for k in keys:
        spec = contract.fields[k]
        if k == decision_key and spec["type"] == "enum" and contract.abstain_value:
            if obj[k] == contract.abstain_value and not abstain_allowed:
                return ParseResult(
                    "REFUSAL", None, "abstention outside abstain_allowed"
                )
            err = _check_value(
                k, spec, obj[k], contract.enum_values(k, abstain_allowed)
            )
        else:
            err = _check_value(k, spec, obj[k])
        if err:
            return ParseResult("PARSE_FAILURE", None, err)
    return ParseResult("SUCCESS", obj)


def parse_code_response(text: str | None) -> ParseResult:
    """``code_v1``: exactly one fenced block with info string ``python`` (§13.2)."""
    if text is None or text.strip() == "":
        return ParseResult("EMPTY_RESPONSE", None, "empty text")
    blocks = list(_PY_BLOCK.finditer(text))
    py = [m for m in blocks if m.group(1).strip() == "python"]
    if len(blocks) != 1 or len(py) != 1:
        return ParseResult(
            "PARSE_FAILURE", None, f"{len(blocks)} fenced blocks, {len(py)} python"
        )
    code = py[0].group(2)
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        return ParseResult("PARSE_FAILURE", None, "code block over 262,144 bytes")
    return ParseResult("SUCCESS", {"code": code}, code=code)


def parse_response(
    contract: Contract, mode: str, abstain_allowed: bool, text: str | None
) -> ParseResult:
    if contract.kind == "code_v1":
        return parse_code_response(text)
    return parse_json_response(contract, mode, abstain_allowed, text)
