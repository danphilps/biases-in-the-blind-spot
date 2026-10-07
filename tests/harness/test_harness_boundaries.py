"""Boundaries (spec §0.7): forbidden imports, module names, test basenames."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "financial_llm_harness"
FORBIDDEN = (
    "chat_limiter",
    "biases_in_the_blind_spot.concept_pipeline.responses_generator",
    "biases_in_the_blind_spot.concept_pipeline.api_llm_base",
    "biases_in_the_blind_spot.util",
)
FIXTURES = HARNESS / "families" / "code_task" / "fixtures"


def _harness_modules() -> list[Path]:
    return [
        p
        for p in sorted(HARNESS.rglob("*.py"))
        if "__pycache__" not in p.parts and FIXTURES not in p.parents
    ]


def _imported_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
        names = [node.module]
        names.extend(f"{node.module}.{a.name}" for a in node.names)
        return names
    return []


def _is_forbidden(name: str) -> bool:
    return any(name == f or name.startswith(f + ".") for f in FORBIDDEN)


def test_no_forbidden_import_in_any_harness_module() -> None:
    offenders: list[str] = []
    for path in _harness_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for name in _imported_names(node):
                if _is_forbidden(name):
                    offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert offenders == []


def test_no_top_level_legacy_import() -> None:
    offenders: list[str] = []
    for path in _harness_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            for name in _imported_names(node):
                if name == "biases_in_the_blind_spot" or name.startswith(
                    "biases_in_the_blind_spot."
                ):
                    offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert offenders == []


def test_no_harness_module_named_test() -> None:
    assert [p for p in _harness_modules() if p.name.startswith("test_")] == []


def test_no_shared_test_basenames() -> None:
    harness = {p.name for p in (ROOT / "tests" / "harness").glob("test_*.py")}
    legacy: set[str] = set()
    for d in ("concept_pipeline", "dataset_preparer", "legacy"):
        legacy |= {p.name for p in (ROOT / "tests" / d).glob("test_*.py")}
    assert harness & legacy == set()
    assert all(name.startswith("test_harness_") for name in harness)
