"""C5 and C5k: broker notes and note sets with planted discrepancies (spec §T3, §T9.1).

``plant_set`` builds K notes on one company from a consensus analyst state:
each note's estimates sit within ±ε of consensus (ordinary noise, never a
discrepancy) and carry a subset of the thesis points. It then plants

- d differences: one note's EPS estimate for a year off the consensus by more
  than 10% of the mean (ratings that would read as opposite are aligned first,
  so no difference arises by accident);
- k inaccuracies: a stated historical figure that differs from C1 by more
  than rounding;
- a ambiguities: a metric named without a definition, period or basis;
- n novel observations: supported by C1, in one note only.

Every magnitude (ε, the planted offsets, the pools of metrics and
observations) is the caller's. ``score`` matches findings
``{type, note_ids, field, quote}`` deterministically: type, field and the set
of note IDs must agree, and the normalised planted quote must occur in the
normalised finding quote (§T9.1 step 4); it returns precision, recall and F1
by type (step 5).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from financial_llm_harness.errors import HarnessValidationError

TYPES = ("difference", "inaccuracy", "ambiguity", "novel")
DIFFERENCE_THRESHOLD = 0.10  # §T9.1: more than 10% of the mean
OPPOSITE = {"buy": "sell", "sell": "buy"}


@dataclass
class Note:
    note_id: str
    rating: str
    estimates: dict[str, float]
    thesis: list[str]
    stated: dict[str, float] = field(
        default_factory=dict
    )  # historical figures the note states
    undefined_metrics: list[str] = field(default_factory=list)
    novel: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Planted:
    type: str
    note_ids: tuple[str, ...]
    field: str
    quote: str


@dataclass(frozen=True)
class SetParams:
    k_notes: int
    eps: float  # ordinary relative noise on estimates
    d: int
    k: int
    a: int
    n: int
    difference_offset: float  # relative EPS offset of a planted difference (> 10%)
    inaccuracy_offset: float  # relative error of a planted inaccuracy
    rounding: float  # relative rounding tolerance of stated figures


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.%-]", " ", text.lower())).strip()


def plant_set(
    consensus: Mapping[str, float],
    ratings: Sequence[str],
    thesis_ids: Sequence[str],
    history: Mapping[str, float],
    metrics: Sequence[str],
    novel_pool: Sequence[str],
    p: SetParams,
    rng: np.random.Generator,
    quote: Any = None,
) -> tuple[list[Note], list[Planted]]:
    """K notes and their planted findings. ``quote(type, field, value)`` gives the exact phrase a note renders."""
    if 2 * p.eps >= DIFFERENCE_THRESHOLD:
        raise HarnessValidationError(
            "C5k: ordinary noise 2ε must stay below the 10% difference threshold"
        )
    if p.difference_offset <= DIFFERENCE_THRESHOLD + p.eps:
        raise HarnessValidationError(
            "C5k: a planted difference must exceed 10% of the mean plus noise"
        )
    if p.inaccuracy_offset <= p.rounding:
        raise HarnessValidationError("C5k: a planted inaccuracy must exceed rounding")
    if (
        p.k > len(history)
        or p.a > len(metrics)
        or p.n > len(novel_pool)
        or p.d > len(consensus) * p.k_notes
    ):
        raise HarnessValidationError(
            "C5k: not enough fields, metrics or observations to plant"
        )
    say = quote or (lambda t, f, v: f"{f} {v}")
    notes: list[Note] = []
    for i in range(p.k_notes):
        est = {
            y: v * (1.0 + float(rng.uniform(-p.eps, p.eps)))
            for y, v in consensus.items()
        }
        keep = [t for t in thesis_ids if rng.random() < 0.5] or [thesis_ids[0]]
        rating = str(ratings[int(rng.integers(0, len(ratings)))])
        notes.append(Note(f"N{i + 1}", rating, est, keep, stated=dict(history)))
    # ratings that would read as opposite by accident are made equal before planting
    first = notes[0].rating
    for nt in notes[1:]:
        if OPPOSITE.get(nt.rating.lower()) == first.lower():
            nt.rating = first
    planted: list[Planted] = []
    years = list(consensus)
    cells = [(i, y) for i in range(len(notes)) for y in years]
    for c in rng.permutation(len(cells))[: p.d]:
        i, y = cells[int(c)]
        nt = notes[i]
        sign = 1.0 if rng.random() < 0.5 else -1.0
        nt.estimates[y] = consensus[y] * (1.0 + sign * p.difference_offset)
        planted.append(
            Planted(
                "difference",
                (nt.note_id,),
                f"eps_{y}",
                say("difference", f"eps_{y}", nt.estimates[y]),
            )
        )
    fields = list(history)
    for j in range(p.k):
        nt = notes[int(rng.integers(0, len(notes)))]
        f = fields[j]
        nt.stated[f] = history[f] * (1.0 + p.inaccuracy_offset)
        planted.append(
            Planted("inaccuracy", (nt.note_id,), f, say("inaccuracy", f, nt.stated[f]))
        )
    for j in range(p.a):
        nt = notes[int(rng.integers(0, len(notes)))]
        nt.undefined_metrics.append(metrics[j])
        planted.append(
            Planted(
                "ambiguity",
                (nt.note_id,),
                metrics[j],
                say("ambiguity", metrics[j], None),
            )
        )
    for j in range(p.n):
        nt = notes[int(rng.integers(0, len(notes)))]
        nt.novel.append(novel_pool[j])
        planted.append(
            Planted(
                "novel", (nt.note_id,), novel_pool[j], say("novel", novel_pool[j], None)
            )
        )
    return notes, planted


def score(
    findings: Sequence[Mapping[str, Any]], planted: Sequence[Planted]
) -> dict[str, dict[str, float | None]]:
    """Deterministic matching (§T9.1 step 4); precision, recall and F1 by type (step 5)."""
    used: set[int] = set()
    tp: dict[str, int] = dict.fromkeys(TYPES, 0)
    fp: dict[str, int] = dict.fromkeys(TYPES, 0)
    for f in findings:
        t = str(f.get("type"))
        if t not in TYPES:
            continue
        ids = tuple(sorted(str(x) for x in f.get("note_ids") or []))
        q = normalise(str(f.get("quote") or ""))
        hit = None
        for k, pl in enumerate(planted):
            if k in used or pl.type != t or pl.field != f.get("field"):
                continue
            if tuple(sorted(pl.note_ids)) != ids or normalise(pl.quote) not in q:
                continue
            hit = k
            break
        if hit is None:
            fp[t] += 1
        else:
            used.add(hit)
            tp[t] += 1
    out: dict[str, dict[str, float | None]] = {}
    for t in TYPES:
        n_planted = sum(pl.type == t for pl in planted)
        fn = n_planted - tp[t]
        prec = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else None
        rec = tp[t] / n_planted if n_planted else None
        if prec is None or rec is None:
            f1 = None
        else:
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        out[t] = {
            "tp": tp[t],
            "fp": fp[t],
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    return out


def render_note(note: Note, quote: Any = None) -> str:
    """A note's text. Every figure of a kind is phrased alike, so the text never marks the planted one."""
    say = quote or (lambda t, f, v: f"{f} {v}")
    lines = [f"Note {note.note_id}", f"Rating: {note.rating}"]
    lines += [say("difference", f"eps_{y}", v) for y, v in note.estimates.items()]
    lines += [say("inaccuracy", f, v) for f, v in note.stated.items()]
    lines += [f"Thesis: {t}" for t in note.thesis]
    lines += [say("ambiguity", m, None) for m in note.undefined_metrics]
    lines += [say("novel", o, None) for o in note.novel]
    return "\n".join(lines) + "\n"
