"""C4: earnings-call transcript assembly with a tone dial (spec §T3, §35 S2).

A transcript is prepared remarks (items from a family's sentence bank) and a
Q&A. ``fill_tone`` fills ``{tone}`` slots from Loughran–McDonald word classes
(B.lm2011; the word lists are the family's bank, never written here) and
``net_tone`` measures the result, so a tone intervention can be checked to
have moved tone and nothing else. Guidance items carry prior-quarter values,
so the planted direction (raised, cut, maintained) is a function of the state.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from financial_llm_harness.errors import HarnessValidationError

TONES = ("negative", "neutral", "positive")
_WORD = re.compile(r"[A-Za-z']+")


def fill_tone(
    sentence: str,
    tone: str,
    lexicon: Mapping[str, Sequence[str]],
    rng: np.random.Generator,
) -> str:
    """Fill every ``{tone}`` slot with a word of the requested class."""
    if tone not in TONES:
        raise HarnessValidationError(f"tone {tone!r} not in {TONES}")
    words = list(lexicon.get(tone) or [])
    if not words:
        raise HarnessValidationError(f"the tone lexicon has no {tone!r} words")
    out = sentence
    while "{tone}" in out:
        out = out.replace("{tone}", words[int(rng.integers(0, len(words)))], 1)
    return out


def net_tone(text: str, lexicon: Mapping[str, Sequence[str]]) -> float | None:
    """(positive − negative) / (positive + negative) word counts; None without tone words."""
    pos = {w.lower() for w in lexicon.get("positive") or []}
    neg = {w.lower() for w in lexicon.get("negative") or []}
    tokens = [t.lower() for t in _WORD.findall(text)]
    p = sum(t in pos for t in tokens)
    n = sum(t in neg for t in tokens)
    return None if p + n == 0 else (p - n) / (p + n)


def guidance_direction(prior: float, current: float, tolerance: float = 0.0) -> str:
    """The planted truth of a guidance item: raised, cut or maintained."""
    if current > prior + tolerance:
        return "raised"
    if current < prior - tolerance:
        return "cut"
    return "maintained"


@dataclass(frozen=True)
class QA:
    question: str
    answer: str
    evasive: bool  # planted: the answer does not address the question


def qa_truth(pairs: Sequence[QA]) -> list[int]:
    """Indices of the planted evasive answers."""
    return [i for i, qa in enumerate(pairs) if qa.evasive]
