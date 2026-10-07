"""Small typed validators for YAML-derived data (spec §0.11: types are validated).

``bool`` is never accepted where an ``int`` or ``float`` is declared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from financial_llm_harness.errors import HarnessValidationError


def fail(path: str, message: str) -> HarnessValidationError:
    return HarnessValidationError(f"{path}: {message}")


def req(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    """Return ``mapping[key]`` or raise naming the missing key."""
    if not isinstance(mapping, Mapping):
        raise fail(path, f"expected a mapping, got {type(mapping).__name__}")
    if key not in mapping:
        raise fail(path, f"missing key {key!r}")
    return mapping[key]


def as_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise fail(path, f"expected a mapping, got {type(value).__name__}")
    return value


def as_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise fail(path, f"expected a list, got {type(value).__name__}")
    return list(value)


def as_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise fail(path, f"expected a string, got {type(value).__name__}")
    return value


def as_opt_str(value: Any, path: str) -> str | None:
    return None if value is None else as_str(value, path)


def as_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise fail(path, f"expected a boolean, got {type(value).__name__}")
    return value


def as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise fail(path, f"expected an integer, got {type(value).__name__}")
    return value


def as_opt_int(value: Any, path: str) -> int | None:
    return None if value is None else as_int(value, path)


def as_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise fail(path, f"expected a number, got {type(value).__name__}")
    return float(value)


def as_opt_float(value: Any, path: str) -> float | None:
    return None if value is None else as_float(value, path)


def check_keys(
    mapping: Mapping[str, Any],
    path: str,
    required: Sequence[str] = (),
    optional: Sequence[str] = (),
) -> None:
    """Fail on a missing required key or any key outside required ∪ optional."""
    as_mapping(mapping, path)
    for key in required:
        if key not in mapping:
            raise fail(path, f"missing key {key!r}")
    allowed = set(required) | set(optional)
    for key in mapping:
        if key not in allowed:
            raise fail(path, f"unknown key {key!r}")
