"""Multiple-testing correction (spec §25, E-30).

BH adjusts p-values, never confidence levels. Members are sorted by
(p, experiment_key); p_adj of rank k = min over j ≥ k of min(1, m·p_(j)/j).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BHRow:
    key: str
    p_raw: float
    p_adj: float
    rank: int
    bh_threshold: float
    rejected: bool
    m: int
    q: float


def benjamini_hochberg(members: Sequence[tuple[str, float]], q: float) -> list[BHRow]:
    """BH step-up over ``(key, p)`` members; rows returned in rank order."""
    m = len(members)
    if m == 0:
        return []
    ordered = sorted(members, key=lambda kv: (kv[1], kv[0]))
    adj = [0.0] * m
    running = 1.0
    for j in range(m - 1, -1, -1):
        running = min(running, min(1.0, m * ordered[j][1] / (j + 1)))
        adj[j] = running
    return [
        BHRow(
            key=key,
            p_raw=p,
            p_adj=adj[k],
            rank=k + 1,
            bh_threshold=(k + 1) * q / m,
            rejected=adj[k] <= q,
            m=m,
            q=q,
        )
        for k, (key, p) in enumerate(ordered)
    ]


def holm(members: Sequence[tuple[str, float]], q: float) -> list[BHRow]:
    """Holm step-down, offered when a programme declares it."""
    m = len(members)
    if m == 0:
        return []
    ordered = sorted(members, key=lambda kv: (kv[1], kv[0]))
    adj = [0.0] * m
    running = 0.0
    for k, (_, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - k) * p))
        adj[k] = running
    return [
        BHRow(key, p, adj[k], k + 1, q / (m - k), adj[k] <= q, m, q)
        for k, (key, p) in enumerate(ordered)
    ]
