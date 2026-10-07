"""Text and oracle provenance (spec §2.7, §8.1, D-14).

Every prompt-visible text file and every oracle-bearing file starts with a
``provenance`` block. A coding agent records ``kind: llm`` and its own family.
For any non-fake target, ``validate`` refuses a file with ``kind: llm`` or an
empty ``approved_by``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from financial_llm_harness.errors import HarnessValidationError

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Provenance:
    source: str  # logical name of the file
    kind: str  # human | llm
    family: str | None
    ident: str | None
    approved_by: tuple[str, ...]
    approved_on: str | None
    role: str  # text | oracle | both

    @property
    def live_ok(self) -> bool:
        return self.kind == "human" and len(self.approved_by) > 0


def parse_provenance(
    data: Mapping[str, Any] | None, source: str, role: str
) -> Provenance:
    if not isinstance(data, Mapping):
        raise HarnessValidationError(f"{source}: missing provenance block (§8.1)")
    allowed = {"drafted_by", "approved_by", "approved_on"}
    if set(data) - allowed or "drafted_by" not in data or "approved_by" not in data:
        raise HarnessValidationError(
            f"{source}: provenance needs drafted_by, approved_by, approved_on"
        )
    d = data["drafted_by"]
    if not isinstance(d, Mapping) or d.get("kind") not in ("human", "llm"):
        raise HarnessValidationError(
            f"{source}: provenance.drafted_by.kind must be human or llm"
        )
    family = d.get("family")
    if family is not None and not isinstance(family, str):
        raise HarnessValidationError(
            f"{source}: provenance.drafted_by.family must be a string or null"
        )
    if d["kind"] == "llm" and not family:
        raise HarnessValidationError(
            f"{source}: an llm-drafted file records its family"
        )
    approved = data["approved_by"]
    if not isinstance(approved, list) or not all(isinstance(a, str) for a in approved):
        raise HarnessValidationError(
            f"{source}: provenance.approved_by must be a list of names"
        )
    on = data.get("approved_on")
    if on is not None and (not isinstance(on, str) or not _DATE.match(on)):
        raise HarnessValidationError(
            f"{source}: approved_on must be a quoted ISO date or null"
        )
    return Provenance(source, d["kind"], family, d.get("id"), tuple(approved), on, role)


def author_families(provs: Iterable[Provenance], roles: tuple[str, ...]) -> list[str]:
    return sorted({p.family for p in provs if p.family and p.role in roles})


def live_violations(provs: Iterable[Provenance]) -> list[str]:
    """Files that block a live run: agent-drafted or unapproved (D-14)."""
    out = []
    for p in provs:
        if p.kind == "llm":
            out.append(f"{p.source}: drafted_by.kind is llm")
        elif not p.approved_by:
            out.append(f"{p.source}: approved_by is empty")
    return out
