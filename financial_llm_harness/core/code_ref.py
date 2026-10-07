"""Executable-source identity (spec §0.10 ``code_ref``, E-24).

``harness_tree_sha256`` covers every ``*.py`` file under the package directory
and every file under ``reporting/``, excluding ``__pycache__/`` and
``families/code_task/fixtures/``. Other package files enter the closure through
``assets.json`` when an experiment uses them.
"""

from __future__ import annotations

import importlib.resources
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from financial_llm_harness.core.canonical_json import tree_sha256

CODE_FIXTURES_PREFIX = "families/code_task/fixtures/"


@dataclass(frozen=True)
class CodeRef:
    """The five ``code_ref`` fields of a manifest."""

    harness_git_sha: str | None
    harness_tree_sha256: str
    legacy_git_sha: str | None
    legacy_tree_sha256: str | None
    dirty: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CodeRef:
        return CodeRef(
            harness_git_sha=data["harness_git_sha"],
            harness_tree_sha256=data["harness_tree_sha256"],
            legacy_git_sha=data["legacy_git_sha"],
            legacy_tree_sha256=data["legacy_tree_sha256"],
            dirty=bool(data["dirty"]),
        )


def package_dir(name: str = "financial_llm_harness") -> Path:
    """Filesystem directory of an importable package."""
    return Path(str(importlib.resources.files(name)))


def _harness_include(rel: str) -> bool:
    if rel.startswith(CODE_FIXTURES_PREFIX):
        return False
    return rel.endswith(".py") or rel.startswith("reporting/")


def _legacy_include(rel: str) -> bool:
    return rel.endswith(".py")


def harness_tree_sha256() -> str:
    return tree_sha256(package_dir(), _harness_include)


def legacy_tree_sha256() -> str:
    return tree_sha256(package_dir("biases_in_the_blind_spot"), _legacy_include)


def _git(directory: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(directory), *args],
            capture_output=True,
            text=True,
            check=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout


def _git_identity(directory: Path) -> tuple[str | None, bool]:
    """(HEAD sha, dirty) for the package directory; unidentifiable → (None, True)."""
    head = _git(directory, "rev-parse", "HEAD")
    if head is None:
        return None, True
    status = _git(directory, "status", "--porcelain", "--", ".")
    if status is None:
        return None, True
    return head.strip(), status.strip() != ""


def compute_code_ref(uses_legacy: bool) -> CodeRef:
    """Compute the code identity; legacy fields are null without a semantic feature."""
    h_dir = package_dir()
    h_sha, h_dirty = _git_identity(h_dir)
    l_sha: str | None = None
    l_tree: str | None = None
    l_dirty = False
    if uses_legacy:
        l_dir = package_dir("biases_in_the_blind_spot")
        l_sha, l_dirty = _git_identity(l_dir)
        l_tree = legacy_tree_sha256()
    return CodeRef(
        harness_git_sha=h_sha,
        harness_tree_sha256=harness_tree_sha256(),
        legacy_git_sha=l_sha,
        legacy_tree_sha256=l_tree,
        dirty=h_dirty or l_dirty,
    )
