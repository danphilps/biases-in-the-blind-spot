"""YAML 1.2 core-schema loading and dumping (spec §0.11, E-28).

All harness YAML is read through ``HarnessSafeLoader`` and written through
``HarnessSafeDumper``. Plain scalars resolve per the YAML 1.2 core schema, so
``off``, ``yes`` and ``2026-10-06`` load as strings; only a small set of
explicit tags is allowed, so no Python constructor is reachable.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, ScalarNode

from financial_llm_harness.errors import HarnessValidationError

_BOOL_RE = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
_INT_RE = re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$")
_FLOAT_RE = re.compile(
    r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
)
_NULL_RE = re.compile(r"^(?:null|Null|NULL|~|)$")

_TAG_STR = "tag:yaml.org,2002:str"
_TAG_INT = "tag:yaml.org,2002:int"
_TAG_FLOAT = "tag:yaml.org,2002:float"
_TAG_BOOL = "tag:yaml.org,2002:bool"
_TAG_NULL = "tag:yaml.org,2002:null"
_TAG_SEQ = "tag:yaml.org,2002:seq"
_TAG_MAP = "tag:yaml.org,2002:map"


class HarnessSafeLoader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 core-schema implicit resolvers and restricted tags."""


HarnessSafeLoader.yaml_implicit_resolvers = {}
HarnessSafeLoader.add_implicit_resolver(_TAG_BOOL, _BOOL_RE, list("tTfF"))
HarnessSafeLoader.add_implicit_resolver(_TAG_INT, _INT_RE, list("-+0123456789"))
HarnessSafeLoader.add_implicit_resolver(_TAG_FLOAT, _FLOAT_RE, list("-+.0123456789"))
HarnessSafeLoader.add_implicit_resolver(_TAG_NULL, _NULL_RE, ["n", "N", "~", ""])


def _construct_bool(loader: HarnessSafeLoader, node: ScalarNode) -> bool:
    value = str(loader.construct_scalar(node))
    if not _BOOL_RE.match(value):
        raise HarnessValidationError(f"invalid !!bool value {value!r}")
    return value.lower() == "true"


def _construct_int(loader: HarnessSafeLoader, node: ScalarNode) -> int:
    value = str(loader.construct_scalar(node))
    if not _INT_RE.match(value):
        raise HarnessValidationError(f"invalid !!int value {value!r}")
    if value.startswith("0o"):
        return int(value[2:], 8)
    if value.startswith("0x"):
        return int(value[2:], 16)
    return int(value, 10)


def _construct_float(loader: HarnessSafeLoader, node: ScalarNode) -> float:
    value = str(loader.construct_scalar(node))
    if not (_FLOAT_RE.match(value) or _INT_RE.match(value)):
        raise HarnessValidationError(f"invalid !!float value {value!r}")
    lowered = value.lower()
    if lowered.endswith(".inf"):
        return -math.inf if lowered.startswith("-") else math.inf
    if lowered == ".nan":
        return math.nan
    if _INT_RE.match(value) and value.startswith(("0o", "0x")):
        return float(_construct_int(loader, node))
    return float(value)


def _construct_null(loader: HarnessSafeLoader, node: ScalarNode) -> None:
    value = str(loader.construct_scalar(node))
    if not _NULL_RE.match(value):
        raise HarnessValidationError(f"invalid !!null value {value!r}")
    return None


def _construct_mapping(loader: HarnessSafeLoader, node: MappingNode) -> dict[Any, Any]:
    keys: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicate = key in keys
        except TypeError as exc:
            raise HarnessValidationError(f"unhashable mapping key {key!r}") from exc
        if duplicate:
            raise HarnessValidationError(f"duplicate mapping key {key!r}")
        keys.add(key)
    mapping = loader.construct_mapping(node, deep=True)
    return dict(mapping)


def _construct_undefined(loader: HarnessSafeLoader, node: Any) -> Any:
    raise HarnessValidationError(f"YAML tag {node.tag!r} is not allowed (§0.11)")


HarnessSafeLoader.yaml_constructors = {}
HarnessSafeLoader.add_constructor(_TAG_STR, yaml.SafeLoader.construct_yaml_str)
HarnessSafeLoader.add_constructor(_TAG_SEQ, yaml.SafeLoader.construct_yaml_seq)
HarnessSafeLoader.add_constructor(_TAG_MAP, _construct_mapping)
HarnessSafeLoader.add_constructor(_TAG_BOOL, _construct_bool)
HarnessSafeLoader.add_constructor(_TAG_INT, _construct_int)
HarnessSafeLoader.add_constructor(_TAG_FLOAT, _construct_float)
HarnessSafeLoader.add_constructor(_TAG_NULL, _construct_null)
HarnessSafeLoader.add_constructor(None, _construct_undefined)  # type: ignore[arg-type]
HarnessSafeLoader.yaml_multi_constructors = {}


class HarnessSafeDumper(yaml.SafeDumper):
    """SafeDumper whose resolvers also know the YAML 1.2 int and float forms.

    A string that either schema would read as another type is therefore quoted.
    """

    def ignore_aliases(self, data: Any) -> bool:
        return True


HarnessSafeDumper.add_implicit_resolver(_TAG_INT, _INT_RE, list("-+0123456789"))
HarnessSafeDumper.add_implicit_resolver(_TAG_FLOAT, _FLOAT_RE, list("-+.0123456789"))


def loads(text: str) -> Any:
    """Parse YAML text with ``HarnessSafeLoader``."""
    try:
        return yaml.load(text, Loader=HarnessSafeLoader)  # noqa: S506 - restricted loader
    except HarnessValidationError:
        raise
    except yaml.YAMLError as exc:
        raise HarnessValidationError(f"invalid YAML: {exc}") from exc


def load(path: str | Path) -> Any:
    """Read and parse a YAML file with ``HarnessSafeLoader``."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HarnessValidationError(f"file not found: {p}") from exc
    try:
        return loads(text)
    except HarnessValidationError as exc:
        raise HarnessValidationError(f"{p}: {exc}") from exc


def dump(data: Any) -> str:
    """Serialise ``data`` with ``HarnessSafeDumper`` (keys in insertion order)."""
    return yaml.dump(
        data, Dumper=HarnessSafeDumper, sort_keys=False, allow_unicode=True
    )
