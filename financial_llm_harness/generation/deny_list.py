"""Deny-list screening of synthetic entity names (spec §7, §T4.4).

The deny-list is a versioned list of real listed-company names and tickers,
supplied by the user (it is real-world reference data, never written by an
agent). Every name that can be rendered into an entity slot, from a family's
name bank or a battery pair bank, is screened against it. A live (non-fake)
manifest needs the list and a clean screen; on the fake path screening runs
whenever the list exists.

Matching is deliberately broad: a name is a hit if, after normalisation
(case-folded, punctuation removed, whitespace collapsed), it equals a listed
name, contains a listed name of two or more words as a word sequence, or has
an upper-case token equal to a listed ticker.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.paths import repo_root
from financial_llm_harness.errors import HarnessValidationError

_PUNCT = re.compile(r"[^\w\s]")


def deny_list_path() -> Path:
    return repo_root() / "experiments" / "financial" / "deny_list.yaml"


@dataclass(frozen=True)
class DenyList:
    version: str
    names: tuple[tuple[str, ...], ...]
    tickers: frozenset[str]


def normalise(text: str) -> tuple[str, ...]:
    return tuple(_PUNCT.sub(" ", text).casefold().split())


def load_deny_list(path: Path | None = None) -> DenyList | None:
    p = path or deny_list_path()
    if not p.exists():
        return None
    data = yaml_io.load(p)
    if not isinstance(data, Mapping) or not {
        "version",
        "source",
        "names",
        "tickers",
    } <= set(data):
        raise HarnessValidationError(
            f"{p}: needs version, source, names and tickers (§7)"
        )
    names = tuple(n for n in (normalise(str(x)) for x in data["names"] or []) if n)
    tickers = frozenset(
        str(t).strip().upper() for t in data["tickers"] or [] if str(t).strip()
    )
    return DenyList(str(data["version"]), names, tickers)


def _contains(tokens: tuple[str, ...], sub: tuple[str, ...]) -> bool:
    k = len(sub)
    return any(tokens[i : i + k] == sub for i in range(len(tokens) - k + 1))


def hits(name: str, deny: DenyList) -> list[str]:
    """Why ``name`` matches the deny-list (empty when it is clean)."""
    toks = normalise(name)
    out: list[str] = []
    for listed in deny.names:
        if toks == listed or (len(listed) >= 2 and _contains(toks, listed)):
            out.append("name " + " ".join(listed))
    for raw in _PUNCT.sub(" ", name).split():
        if raw.isupper() and raw in deny.tickers:
            out.append(f"ticker {raw}")
    return out


def screen(names: Iterable[str], deny: DenyList) -> dict[str, list[str]]:
    return {n: h for n in sorted(set(names)) if (h := hits(n, deny))}


def bank_names(bank: Mapping[str, Any]) -> list[str]:
    """Every string a name bank can render: plain entries and both members of pairs."""
    out: list[str] = []
    for e in bank.get("entries") or []:
        if isinstance(e, str):
            out.append(e)
        elif isinstance(e, Mapping):
            out += [str(v) for k, v in e.items() if k in ("a", "b", "name")]
    for p in bank.get("pairs") or []:
        out += (
            [str(p[k]) for k in ("a", "b")]
            if isinstance(p, Mapping)
            else [str(x) for x in p]
        )
    return out


# §T4.4 entity slots that carry names of firms or people (country and date are not names)
ENTITY_NODES = ("company_name", "ceo_name", "broker", "analyst_label")


def screen_cases(
    cases: Iterable[Mapping[str, Any]], *, live: bool, path: Path | None = None
) -> None:
    """Screen every rendered entity name of a corpus (§7); exit 2 on a hit, or live without a list."""
    deny = load_deny_list(path)
    if deny is None:
        if live:
            raise HarnessValidationError(
                f"§7: a live run needs the deny-list of real listed-company names and tickers at {path or deny_list_path()}"
            )
        return
    names = {
        str(state[node])
        for case in cases
        for state in (case.get("arms") or {}).values()
        for node in ENTITY_NODES
        if node in state
    }
    found = screen(names, deny)
    if found:
        shown = "; ".join(f"{n!r}: {', '.join(h)}" for n, h in list(found.items())[:10])
        raise HarnessValidationError(
            f"§7 deny-list v{deny.version}: {len(found)} entity name(s) match: {shown}"
        )
