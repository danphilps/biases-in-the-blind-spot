"""``results/DATA_DICTIONARY.md``, generated from ``reporting/schemas/*.json`` (spec §32A.3).

For every file and column: name, type, unit, definition and spec section.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"


def schema_path(file_name: str) -> Path:
    """``effects.csv`` → ``schemas/effects.v1.json`` (§33)."""
    return SCHEMA_DIR / f"{file_name.split('.', 1)[0]}.v1.json"


@cache
def load_schema(file_name: str) -> dict[str, Any]:
    """The JSON Schema (2020-12) of ``results/<file_name>``."""
    return json.loads(schema_path(file_name).read_text(encoding="utf-8"))


def schema_files() -> list[str]:
    """The results files that have a schema, by file name (each schema's ``title``)."""
    return sorted(
        json.loads(p.read_text(encoding="utf-8"))["title"]
        for p in SCHEMA_DIR.glob("*.v1.json")
    )


def _type_text(p: Mapping[str, Any]) -> str:
    if "enum" in p:
        vals = [v for v in p["enum"] if v is not None]
        nullable = None in p["enum"]
        return (
            "enum: "
            + ", ".join(f"`{v}`" for v in vals)
            + (" (nullable)" if nullable else "")
        )
    if "const" in p:
        return f"const `{p['const']}`"
    t = p.get("type", "object")
    types = t if isinstance(t, list) else [t]
    base = [x for x in types if x != "null"]
    text = " or ".join(base) if base else "null"
    xt = p.get("x-type")
    if xt in ("uuid", "sha256", "proportion"):
        text = {
            "uuid": "string (UUID)",
            "sha256": "string (SHA-256 hex)",
            "proportion": "number in [0, 1]",
        }[xt]
    if "null" in types:
        text += " (nullable)"
    return text


def _cell(s: Any) -> str:
    return str(s if s is not None else "—").replace("|", "\\|").replace("\n", " ")


def _walk(prefix: str, p: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    yield prefix, p
    props = p.get("properties")
    if isinstance(props, Mapping):
        for k, v in props.items():
            yield from _walk(f"{prefix}.{k}" if prefix else k, v)
    items = p.get("items")
    if isinstance(items, Mapping) and isinstance(items.get("properties"), Mapping):
        for k, v in items["properties"].items():
            yield from _walk(f"{prefix}[].{k}", v)


def render_dictionary(files: Sequence[str]) -> str:
    """Markdown dictionary for ``summary.json`` and every results file in ``files`` that has a schema."""
    names = [
        "summary.json",
        *[f for f in files if f != "summary.json" and schema_path(f).exists()],
    ]
    out = [
        "# Data dictionary",
        "",
        "Generated from `reporting/schemas/*.json` (JSON Schema 2020-12). CSV: RFC 4180, UTF-8, LF; missing = empty field; "
        "booleans `true`/`false`; floats via `repr`; proportions as decimals in [0, 1].",
        "",
    ]
    for i, name in enumerate(names, 1):
        s = load_schema(name)
        out += [f"## {i}. `{name}`", "", f"{s['description']} Spec {s['x-spec']}.", ""]
        out += ["| Field | Type | Unit | Definition | Spec |", "|---|---|---|---|---|"]
        if s.get("x-format") == "csv":
            for col in s["x-columns"]:
                p = s["properties"][col]
                out.append(
                    f"| `{col}` | {_cell(_type_text(p))} | {_cell(p.get('x-unit'))} | {_cell(p.get('description'))} | {_cell(p.get('x-spec'))} |"
                )
        else:
            for path, p in _walk("", s):
                if not path:
                    continue
                out.append(
                    f"| `{path}` | {_cell(_type_text(p))} | {_cell(p.get('x-unit'))} | {_cell(p.get('description'))} | {_cell(p.get('x-spec'))} |"
                )
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"
