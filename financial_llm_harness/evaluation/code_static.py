"""L2 static code checks (spec §13.3): ``ast.parse`` plus an import scan.

These are recorded as features and in the ``code_outcomes`` event; they never
block execution, which is the sandbox's job (§13.4). They run on the host's
parser, so a construct valid only in a newer Python can parse here and still
fail to compile in the image; the sandbox's COMPILE_FAILURE is authoritative.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StaticResult:
    parses: bool
    imports: tuple[str, ...]
    syntax_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parses": self.parses,
            "imports": list(self.imports),
            "syntax_error": self.syntax_error,
        }


def static_check(code: str) -> StaticResult:
    """Parse ``code`` and list the top-level modules it imports, sorted."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        return StaticResult(False, (), f"{type(exc).__name__}: {exc}")
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return StaticResult(True, tuple(sorted(found)), None)
