"""Corpus-level diagnostics and near-duplicate clustering (spec §9.2, E-32).

Clusters are formed before any hash exists and numbered by ``cluster_ordinal``
in order of each cluster's lowest ``case_idx``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.families import NodeDef

SHINGLE = 5
TEXT_JACCARD = 0.8


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def structured_edges(
    states: Sequence[Mapping[str, Any]],
    nodes: Mapping[str, NodeDef],
    rendered: set[str],
    displayed: Sequence[Mapping[str, str]],
) -> list[tuple[int, int]]:
    """Structured near-duplicate pairs (§9.2)."""
    n = len(states)
    cat = sorted(
        name
        for name, nd in nodes.items()
        if nd.kind == "primitive"
        and (nd.type == "categorical" or nd.display == "text")
        and name in rendered
    )
    num_all = sorted(
        name
        for name, nd in nodes.items()
        if nd.kind == "primitive"
        and nd.type == "numeric"
        and nd.display != "text"
        and name in rendered
    )
    num: list[str] = []
    for name in num_all:
        vals = np.array([float(s[name]) for s in states], dtype=float)
        if n > 0 and float(np.std(vals)) > 0:
            num.append(name)
    k = len(num)
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, s in enumerate(states):
        groups[tuple(str(s[c]) for c in cat)].append(i)
    edges: list[tuple[int, int]] = []
    if k >= 3:
        mat = np.array([[float(s[f]) for f in num] for s in states], dtype=float)
        z = (mat - mat.mean(axis=0)) / mat.std(axis=0)
        eps = 0.05 * math.sqrt(k)
        for members in groups.values():
            if len(members) < 2:
                continue
            sub = z[members]
            d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(axis=2)
            ii, jj = np.where(np.triu(d2 < eps * eps, k=1))
            edges.extend(
                (members[a], members[b])
                for a, b in zip(ii.tolist(), jj.tolist(), strict=True)
            )
    else:
        keyed: dict[tuple[tuple[str, str], ...], list[int]] = defaultdict(list)
        for i, disp in enumerate(displayed):
            keyed[tuple(sorted(disp.items()))].append(i)
        for members in keyed.values():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    edges.append((members[a], members[b]))
    return edges


def shingles(text: str) -> set[tuple[str, ...]]:
    words = text.lower().split()
    if len(words) < SHINGLE:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def text_edges(slot_texts: Sequence[str]) -> list[tuple[int, int]]:
    """Text near-duplicate pairs: Jaccard of word 5-shingles ≥ 0.8, computed exactly."""
    sets = [shingles(t) for t in slot_texts]
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, s in enumerate(sets):
        for sh in s:
            index[sh].append(i)
    candidates: set[tuple[int, int]] = set()
    for members in index.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                candidates.add((members[a], members[b]))
    edges = []
    for a, b in sorted(candidates):
        sa, sb = sets[a], sets[b]
        union = len(sa | sb)
        if union and len(sa & sb) / union >= TEXT_JACCARD:
            edges.append((a, b))
    return edges


@dataclass(frozen=True)
class Clustering:
    ordinal_of_case: list[int]
    members: list[list[int]]
    n_structured_edges: int
    n_text_edges: int

    @property
    def n_clusters(self) -> int:
        return len(self.members)

    @property
    def largest(self) -> int:
        return max((len(m) for m in self.members), default=0)


def cluster_cases(
    n: int, structured: Sequence[tuple[int, int]], text: Sequence[tuple[int, int]]
) -> Clustering:
    uf = _UnionFind(n)
    for a, b in (*structured, *text):
        uf.union(a, b)
    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(i)
    members = sorted((sorted(m) for m in comps.values()), key=lambda m: m[0])
    ordinal = [0] * n
    for o, m in enumerate(members):
        for i in m:
            ordinal[i] = o
    return Clustering(ordinal, members, len(structured), len(text))


def chaining_guard(clustering: Clustering, n_cases: int) -> None:
    limit = max(10, math.ceil(0.02 * n_cases))
    if clustering.largest > limit:
        idx = max(
            range(clustering.n_clusters), key=lambda o: len(clustering.members[o])
        )
        raise HarnessValidationError(
            f"chaining guard: cluster {idx} holds {clustering.largest} cases (> {limit}); "
            "the generator varies too little for its n_cases (§9.2, E-32)"
        )


def template_concentration(
    families_per_case: Sequence[Sequence[str]],
) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for fams in families_per_case:
        for f in fams:
            counts[f] += 1
            total += 1
    return {f: c / total for f, c in sorted(counts.items())} if total else {}


def check_template_concentration(shares: Mapping[str, float]) -> None:
    n_fam = max(1, len(shares))
    limit = max(0.4, 1.0 / n_fam)
    for fam, share in shares.items():
        if share > limit + 1e-12:
            raise HarnessValidationError(
                f"template family {fam!r} holds {share:.3f} of renderings (> {limit:.3f}, §9.2)"
            )


def check_direction_balance(counterbalanced_by_cluster: Sequence[bool]) -> float:
    if not counterbalanced_by_cluster:
        return 0.0
    share = sum(counterbalanced_by_cluster) / len(counterbalanced_by_cluster)
    if not 0.45 <= share <= 0.55:
        raise HarnessValidationError(
            f"direction balance: counterbalanced share of clusters {share:.3f} outside [0.45, 0.55]"
        )
    return share


def realised_marginals(
    states: Sequence[Mapping[str, Any]], nodes: Mapping[str, NodeDef]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, nd in nodes.items():
        if nd.kind == "derived":
            continue
        vals = [s[name] for s in states]
        if nd.type == "numeric" and nd.display != "text":
            arr = np.array([float(v) for v in vals], dtype=float)
            q = (
                np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
                if len(arr)
                else [math.nan] * 5
            )
            out[name] = {
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                "min": float(arr.min()),
                "p05": float(q[0]),
                "p25": float(q[1]),
                "p50": float(q[2]),
                "p75": float(q[3]),
                "p95": float(q[4]),
                "max": float(arr.max()),
            }
        else:
            counts: dict[str, int] = defaultdict(int)
            for v in vals:
                counts[str(v)] += 1
            out[name] = {
                "shares": {k: c / len(vals) for k, c in sorted(counts.items())}
            }
    return out


def realised_correlation(
    states: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> list[list[float]] | None:
    if len(names) < 2 or len(states) < 3:
        return None
    mat = np.array([[float(s[n]) for n in names] for s in states], dtype=float)
    c = np.corrcoef(mat, rowvar=False)
    return [[float(x) for x in row] for row in c]
