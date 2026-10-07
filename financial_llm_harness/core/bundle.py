"""The frozen experiment bundle and its closure checks (spec §0.10, §4, E-24).

Bundle files: ``manifest.yaml``, ``manifest.lock``, ``cases.jsonl``,
``renderings.jsonl``, ``assets.json``, ``corpus_report.json``,
``probes/<model_key>.json``. Every consumed asset has one ``assets.json`` row;
shared registry files are hashed per entry, so editing one item is not drift for
an experiment of another.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.canonical_json import (
    sha256_bytes,
    sha256_canonical,
    tree_sha256,
)
from financial_llm_harness.core.code_ref import CodeRef
from financial_llm_harness.core.hashes import (
    ALL_ROLES,
    compute_design_hash,
    compute_manifest_hash,
)
from financial_llm_harness.core.paths import pkg, repo_root
from financial_llm_harness.errors import HarnessValidationError

MANIFEST_FILE = "manifest.yaml"
LOCK_FILE = "manifest.lock"
CASES_FILE = "cases.jsonl"
RENDERINGS_FILE = "renderings.jsonl"
ASSETS_FILE = "assets.json"
CORPUS_REPORT_FILE = "corpus_report.json"
PROBES_DIR = "probes"

SOURCES = ("package", "repository", "generated")


@dataclass(frozen=True)
class AssetRow:
    """One ``assets.json`` row."""

    role: str
    logical_name: str
    sha256: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "logical_name": self.logical_name,
            "sha256": self.sha256,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# File formats (§32A.2)
# ---------------------------------------------------------------------------


def json_text(obj: Any) -> str:
    """JSON file text: UTF-8, 2-space indent, trailing newline, NaN forbidden."""
    return json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def jsonl_line(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


def jsonl_text(rows: Iterable[Any]) -> str:
    return "".join(jsonl_line(r) + "\n" for r in rows)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write tmp → fsync → ``os.replace`` (§29)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


# ---------------------------------------------------------------------------
# Asset hashing (§0.10)
# ---------------------------------------------------------------------------

_TAXONOMY_LISTS = {
    "taxonomy/faq.yaml": "items",
    "taxonomy/bases.yaml": "bases",
    "taxonomy/interventions.yaml": "interventions",
}


def _find_by_id(items: Sequence[Mapping[str, Any]], ident: str, where: str) -> Any:
    for item in items:
        if isinstance(item, Mapping) and item.get("id") == ident:
            return item
    raise HarnessValidationError(f"{where}: no entry with id {ident!r}")


def entry_payload(file_part: str, data: Mapping[str, Any], selector: str) -> Any:
    """The object hashed for a per-entry asset ``file_part#selector``."""
    where = f"{file_part}#{selector}"
    provenance = data.get("provenance")
    if file_part == "models/providers.yaml":
        for prov in (data.get("providers") or {}).values():
            profiles = prov.get("request_profiles") or {}
            if selector in profiles:
                return profiles[selector]
        raise HarnessValidationError(f"{where}: no request profile {selector!r}")
    if file_part in _TAXONOMY_LISTS:
        items = data.get(_TAXONOMY_LISTS[file_part]) or []
        entry = _find_by_id(items, selector, where)
        return {"entry": entry, "provenance": provenance}
    if file_part.endswith("/defaults.yaml"):
        entries = data.get("entries") or {}
        faq_id, _, variant_id = selector.partition("/")
        if faq_id not in entries:
            raise HarnessValidationError(f"{where}: no entry {faq_id!r}")
        entry = dict(entries[faq_id])
        if variant_id:
            variants = entry.get("variants") or {}
            if variant_id not in variants:
                raise HarnessValidationError(
                    f"{where}: no variant record {variant_id!r}"
                )
            return {"entry": variants[variant_id], "provenance": provenance}
        entry.pop("variants", None)
        return {"entry": entry, "provenance": provenance}
    if file_part.endswith("/family.yaml"):
        if selector == "core":
            core = {
                k: v
                for k, v in data.items()
                if k not in ("interventions", "oracles", "provenance")
            }
            return {"entry": core, "provenance": provenance}
        section, _, key = selector.partition("/")
        if section not in ("interventions", "oracles"):
            raise HarnessValidationError(f"{where}: unknown selector")
        table = data.get(section) or {}
        if key not in table:
            raise HarnessValidationError(f"{where}: no {section} entry {key!r}")
        return {"entry": table[key], "provenance": provenance}
    raise HarnessValidationError(
        f"{where}: per-entry hashing not defined for this file"
    )


def _python_object_sha256(dotted: str) -> str:
    module_name, _, attr = dotted.rpartition(".")
    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        raise HarnessValidationError(f"cannot resolve Python asset {dotted!r}") from exc
    if isinstance(obj, str):
        return sha256_bytes(obj.encode("utf-8"))
    return sha256_bytes(inspect.getsource(obj).encode("utf-8"))


def source_root(source: str, bundle_dir: Path | None) -> Path:
    if source == "package":
        return pkg()
    if source == "repository":
        return repo_root()
    if source == "generated":
        if bundle_dir is None:
            raise HarnessValidationError("generated asset needs a bundle directory")
        return bundle_dir
    raise HarnessValidationError(f"unknown asset source {source!r}")


def path_sha256(path: Path) -> str:
    """File bytes, or the §0.10 tree hash for a directory."""
    if path.is_dir():
        return tree_sha256(path)
    if not path.is_file():
        raise HarnessValidationError(f"asset not found: {path}")
    return file_sha256(path)


def asset_sha256(logical_name: str, source: str, bundle_dir: Path | None = None) -> str:
    """Recompute an asset's sha256 from its current source."""
    if source == "package" and "/" not in logical_name and "#" not in logical_name:
        if not logical_name.endswith((".yaml", ".json", ".py", ".md")):
            return _python_object_sha256(logical_name)
    root = source_root(source, bundle_dir)
    if "#" in logical_name:
        file_part, _, selector = logical_name.partition("#")
        data = yaml_io.load(root / file_part)
        if not isinstance(data, Mapping):
            raise HarnessValidationError(f"{file_part}: expected a mapping")
        return sha256_canonical(entry_payload(file_part, data, selector))
    return path_sha256(root / logical_name)


def make_asset(
    role: str, logical_name: str, source: str, bundle_dir: Path | None = None
) -> AssetRow:
    if role not in ALL_ROLES:
        raise HarnessValidationError(f"unknown asset role {role!r}")
    if source not in SOURCES:
        raise HarnessValidationError(f"unknown asset source {source!r}")
    return AssetRow(
        role, logical_name, asset_sha256(logical_name, source, bundle_dir), source
    )


def sorted_assets(rows: Iterable[AssetRow]) -> list[AssetRow]:
    unique: dict[tuple[str, str], AssetRow] = {}
    for row in rows:
        key = (row.role, row.logical_name)
        prev = unique.get(key)
        if prev is not None and prev != row:
            raise HarnessValidationError(
                f"asset {key} listed twice with different hashes"
            )
        unique[key] = row
    return [unique[k] for k in sorted(unique)]


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------


def load_manifest(bundle_dir: Path) -> dict[str, Any]:
    data = yaml_io.load(bundle_dir / MANIFEST_FILE)
    if not isinstance(data, dict):
        raise HarnessValidationError(f"{bundle_dir / MANIFEST_FILE}: not a mapping")
    return data


def load_assets(bundle_dir: Path) -> list[dict[str, Any]]:
    data = json.loads((bundle_dir / ASSETS_FILE).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise HarnessValidationError("assets.json: expected a JSON array")
    return data


def read_lock(bundle_dir: Path) -> str | None:
    path = bundle_dir / LOCK_FILE
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Frozen-input checks (§4)
# ---------------------------------------------------------------------------


def frozen_input_mismatches(
    bundle_dir: Path, current_code_ref: CodeRef, require_lock: bool = True
) -> list[str]:
    """Every §4 frozen-input mismatch of a bundle; empty means it is intact."""
    out: list[str] = []
    manifest = load_manifest(bundle_dir)
    assets = load_assets(bundle_dir)
    stored_manifest_hash = manifest.get("manifest_hash")
    recomputed = compute_manifest_hash(manifest)
    if recomputed != stored_manifest_hash:
        out.append(
            f"manifest_hash: manifest says {stored_manifest_hash}, recomputed {recomputed}"
        )
    lock = read_lock(bundle_dir)
    if lock is None:
        if require_lock:
            out.append("manifest.lock: missing")
    elif lock != stored_manifest_hash:
        out.append(f"manifest.lock: {lock} != manifest_hash {stored_manifest_hash}")
    design = compute_design_hash(manifest, assets)
    if design != manifest.get("design_hash"):
        out.append(
            f"design_hash: manifest says {manifest.get('design_hash')}, recomputed {design}"
        )
    corpus = manifest.get("corpus") or {}
    for key, fname in (
        ("cases", CASES_FILE),
        ("renderings", RENDERINGS_FILE),
        ("assets", ASSETS_FILE),
    ):
        expected = (corpus.get(key) or {}).get("sha256")
        path = bundle_dir / fname
        actual = file_sha256(path) if path.exists() else None
        if actual != expected:
            out.append(f"corpus.{key}: {fname} sha256 {actual} != {expected}")
    for row in assets:
        try:
            actual = asset_sha256(row["logical_name"], row["source"], bundle_dir)
        except HarnessValidationError as exc:
            out.append(f"asset {row['role']}:{row['logical_name']}: {exc}")
            continue
        if actual != row["sha256"]:
            out.append(
                f"asset {row['role']}:{row['logical_name']}: sha256 {actual} != {row['sha256']}"
            )
    code_ref = manifest.get("code_ref") or {}
    if code_ref.get("harness_tree_sha256") != current_code_ref.harness_tree_sha256:
        out.append(
            "code_ref.harness_tree_sha256: "
            f"{code_ref.get('harness_tree_sha256')} != {current_code_ref.harness_tree_sha256}"
        )
    if code_ref.get("legacy_tree_sha256") is not None:
        if code_ref.get("legacy_tree_sha256") != current_code_ref.legacy_tree_sha256:
            out.append(
                "code_ref.legacy_tree_sha256: "
                f"{code_ref.get('legacy_tree_sha256')} != {current_code_ref.legacy_tree_sha256}"
            )
    return out
