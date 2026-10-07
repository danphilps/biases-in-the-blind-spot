"""Canonical JSON and hashing helpers (spec §4).

Canonical form: JSON, sorted keys, UTF-8, no insignificant whitespace, floats via
Python ``repr``, NaN and Inf forbidden.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from financial_llm_harness.errors import HarnessValidationError


def canonical(obj: Any) -> bytes:
    """Return the canonical JSON bytes of ``obj``."""
    try:
        text = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:  # NaN or Inf
        raise HarnessValidationError(f"canonical JSON forbids NaN/Inf: {exc}") from exc
    return text.encode("utf-8")


def canonical_str(obj: Any) -> str:
    """Canonical JSON as text."""
    return canonical(obj).decode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_canonical(obj: Any) -> str:
    """Hex sha256 of the canonical JSON of ``obj``."""
    return sha256_bytes(canonical(obj))


def sha256_text(text: str) -> str:
    """Hex sha256 of UTF-8 text."""
    return sha256_bytes(text.encode("utf-8"))


def tree_entries(
    root: Path, include: Callable[[str], bool] | None = None
) -> list[list[str]]:
    """Sorted ``[posix_path, sha256(bytes)]`` for every regular file under ``root``.

    ``__pycache__/`` directories are always excluded; ``include`` filters on the
    POSIX path relative to ``root``.
    """
    entries: list[list[str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if "__pycache__" in rel.split("/"):
            continue
        if include is not None and not include(rel):
            continue
        entries.append([rel, sha256_bytes(path.read_bytes())])
    entries.sort(key=lambda e: e[0])
    return entries


def tree_sha256(root: Path, include: Callable[[str], bool] | None = None) -> str:
    """The §0.10 tree hash: sha256 of the canonical JSON of the sorted file list."""
    return sha256_canonical(tree_entries(root, include))
