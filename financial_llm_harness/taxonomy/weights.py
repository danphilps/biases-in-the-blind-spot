"""``taxonomy weights --responses PATH``: FAQ frequency × stakes weights (spec §T10.3, D-9, D-12).

Responses are a CSV kept outside the repository (§T10.5), one row per
(respondent, item) with columns ``respondent_id, role, faq_id, frequency,
stakes, channel``. Only aggregates are written, to ``taxonomy/weights.yaml``.

1. f̄ᵢᵣ, s̄ᵢᵣ: mean asks per month and mean stakes of item i in role r;
2. uᵢ = mean over the roles routed to i (with answers) of f̄ᵢᵣ·s̄ᵢᵣ;
3. wᵢ = uᵢ / Σⱼ uⱼ within each domain (F, Q, M, K);
4. 2,000 seeded bootstrap resamples of respondents within each role give a
   90% interval per weight and the stability of its tertile;
5. the weights are used only with ≥ 5 completed responses in ≥ 6 of the 8
   roles (D-12); otherwise every used weight stays 1.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.ids import uint64
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.taxonomy.registry import load_registry, taxonomy_dir
from financial_llm_harness.taxonomy.survey import instrument, routed

BOOTSTRAP_B = 2000
MIN_PER_ROLE = 5
MIN_ROLES = 6
DOMAIN_LETTER = {"fundamental": "F", "quant_models": "Q", "ml": "M", "code": "K"}


@dataclass(frozen=True)
class Answer:
    respondent: str
    role: str
    faq_id: str
    asks: float
    stakes: float
    channel: str | None


def read_responses(path: Path, inst: Mapping[str, Any]) -> list[Answer]:
    freq = {
        a["label"]: float(a["asks_per_month"])
        for a in inst["closed_questions"][0]["answers"]
    }
    roles = set(inst["routing"])
    out: list[Answer] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for n, row in enumerate(csv.DictReader(fh), start=2):
            role = row.get("role", "")
            if role not in roles:
                raise HarnessValidationError(
                    f"{path}:{n}: role {role!r} is not a §T10.1 role"
                )
            if row.get("frequency") not in freq:
                raise HarnessValidationError(
                    f"{path}:{n}: frequency {row.get('frequency')!r}"
                )
            stakes = float(row["stakes"])
            if stakes not in (1.0, 2.0, 3.0, 4.0, 5.0):
                raise HarnessValidationError(
                    f"{path}:{n}: stakes {row['stakes']!r} outside 1–5"
                )
            out.append(
                Answer(
                    str(row["respondent_id"]),
                    role,
                    str(row["faq_id"]),
                    freq[row["frequency"]],
                    stakes,
                    row.get("channel") or None,
                )
            )
    return out


def item_scores(
    answers: Sequence[Answer], routes: Mapping[str, set[str]]
) -> dict[str, float]:
    """uᵢ: the mean over routed roles of f̄ᵢᵣ·s̄ᵢᵣ."""
    acc: dict[tuple[str, str], list[Answer]] = defaultdict(list)
    for a in answers:
        if a.role in routes.get(a.faq_id, set()):
            acc[(a.faq_id, a.role)].append(a)
    per_item: dict[str, list[float]] = defaultdict(list)
    for (fid, _role), rows in acc.items():
        f = float(np.mean([r.asks for r in rows]))
        s = float(np.mean([r.stakes for r in rows]))
        per_item[fid].append(f * s)
    return {fid: float(np.mean(v)) for fid, v in per_item.items()}


def normalise(u: Mapping[str, float], domain_of: Mapping[str, str]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for fid, v in u.items():
        totals[domain_of[fid]] += v
    return {
        fid: (v / totals[domain_of[fid]] if totals[domain_of[fid]] > 0 else 0.0)
        for fid, v in u.items()
    }


def tertiles(w: Mapping[str, float], domain_of: Mapping[str, str]) -> dict[str, int]:
    """1 = top third of its domain, 3 = bottom third (ranks, ties by ID)."""
    out: dict[str, int] = {}
    by_domain: dict[str, list[str]] = defaultdict(list)
    for fid in w:
        by_domain[domain_of[fid]].append(fid)
    for ids in by_domain.values():
        ranked = sorted(ids, key=lambda f: (-w[f], f))
        n = len(ranked)
        for k, fid in enumerate(ranked):
            out[fid] = 1 + min(2, (3 * k) // max(n, 1))
    return out


def compute_weights(responses: Path, *, seed: int = 20261007) -> Path:
    inst = instrument()
    reg = load_registry()
    routes = {
        fid: {r for r, pre in inst["routing"].items() if routed(item, list(pre))}
        for fid, item in reg.items.items()
    }
    domain_of = {
        fid: DOMAIN_LETTER[str(i.data["domain"])] for fid, i in reg.items.items()
    }
    answers = [a for a in read_responses(responses, inst) if a.faq_id in reg.items]
    respondents: dict[str, set[str]] = defaultdict(set)
    for a in answers:
        respondents[a.role].add(a.respondent)
    n_per_role = {r: len(respondents.get(r, set())) for r in sorted(inst["routing"])}
    met = sum(1 for n in n_per_role.values() if n >= MIN_PER_ROLE) >= MIN_ROLES
    w = normalise(item_scores(answers, routes), domain_of)
    t = tertiles(w, domain_of)
    # bootstrap: resample respondents within each role (§T10.3 step 4)
    rng = np.random.default_rng(uint64("survey_weights", seed))
    by_resp: dict[tuple[str, str], list[Answer]] = defaultdict(list)
    for a in answers:
        by_resp[(a.role, a.respondent)].append(a)
    roles = {r: sorted(ids) for r, ids in respondents.items()}
    draws: dict[str, list[float]] = defaultdict(list)
    same_tertile: dict[str, int] = defaultdict(int)
    for _ in range(BOOTSTRAP_B):
        sample: list[Answer] = []
        for role, ids in roles.items():
            for k in rng.integers(0, len(ids), size=len(ids)):
                sample += by_resp[(role, ids[int(k)])]
        wb = normalise(item_scores(sample, routes), domain_of)
        tb = tertiles(wb, domain_of)
        for fid in w:
            draws[fid].append(wb.get(fid, 0.0))
            same_tertile[fid] += int(tb.get(fid) == t[fid])
    channels: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a in answers:
        if a.channel:
            channels[a.faq_id][a.channel] += 1
    items: dict[str, Any] = {}
    for fid in sorted(w):
        lo, hi = np.quantile(draws[fid], [0.05, 0.95])
        total = sum(channels[fid].values())
        items[fid] = {
            "weight": round(w[fid], 10),
            "ci90": [round(float(lo), 10), round(float(hi), 10)],
            "tertile": t[fid],
            "tertile_stability": round(same_tertile[fid] / BOOTSTRAP_B, 4),
            "channel_shares": {
                c: round(n / total, 4) for c, n in sorted(channels[fid].items())
            }
            if total
            else {},
        }
    doc = {
        "survey_version": str(inst["survey_version"]),
        "n_per_role": n_per_role,
        "threshold": {"min_per_role": MIN_PER_ROLE, "min_roles": MIN_ROLES, "met": met},
        "bootstrap": {"B": BOOTSTRAP_B, "seed": seed},
        "items": items,
    }
    path = taxonomy_dir() / "weights.yaml"
    path.write_text(yaml_io.dump(doc), encoding="utf-8")
    return path


def used_weights() -> dict[str, float] | None:
    """The item weights when the D-12 threshold is met, else None (every weight stays 1)."""
    path = taxonomy_dir() / "weights.yaml"
    if not path.exists():
        return None
    data = yaml_io.load(path)
    if not (data.get("threshold") or {}).get("met"):
        return None
    return {fid: float(v["weight"]) for fid, v in (data.get("items") or {}).items()}
