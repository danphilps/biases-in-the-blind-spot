"""Path roots (spec §0.8, E-9).

``HARNESS_HOME`` (default: ``experiments/financial/outputs/`` in the repository)
roots ``runs/``, ``comparisons/``, ``programmes/`` and ``probes/``. Author-written
inputs live in the repository under ``experiments/financial/``; the repository
root is the working directory, as every documented command runs from it.
``HARNESS_REPO_ROOT`` overrides it for tests that build a scratch repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from financial_llm_harness.core.code_ref import package_dir


def repo_root() -> Path:
    return Path(os.environ.get("HARNESS_REPO_ROOT") or os.getcwd())


def financial_dir() -> Path:
    return repo_root() / "experiments" / "financial"


def harness_home() -> Path:
    """Root of every harness output; outputs sit beside their inputs, never in them."""
    home = os.environ.get("HARNESS_HOME")
    return Path(home) if home else financial_dir() / "outputs"


def panels_dir() -> Path:
    return financial_dir() / "panels"


def runs_dir() -> Path:
    return harness_home() / "runs"


def probes_dir() -> Path:
    return harness_home() / "probes"


def programmes_dir() -> Path:
    return harness_home() / "programmes"


def comparisons_dir() -> Path:
    return harness_home() / "comparisons"


def pkg() -> Path:
    """The harness package directory."""
    return package_dir()
