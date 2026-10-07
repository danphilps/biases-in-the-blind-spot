"""``taxonomy survey-export``: the practitioner survey instrument (spec §T10.1, §T10.2).

The instrument's wording, answer scales and role routing live in
``taxonomy/survey/instrument.yaml`` (transcribed from §T10.1–§T10.2); the item
list is the registry's, keyed by FAQ ID so weights map back without manual
matching. Output: one CSV and one JSON per export, for any survey tool.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.paths import repo_root
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.taxonomy.registry import Item, load_registry, taxonomy_dir

CSV_COLUMNS = (
    "survey_version",
    "role",
    "kind",
    "faq_id",
    "domain",
    "category",
    "question_id",
    "prompt",
    "answers",
)


def instrument() -> dict[str, Any]:
    path = taxonomy_dir() / "survey" / "instrument.yaml"
    if not path.exists():
        raise HarnessValidationError(f"{path} not found (§T10.2)")
    data = yaml_io.load(path)
    if not isinstance(data, Mapping):
        raise HarnessValidationError(f"{path}: not a mapping")
    return dict(data)


def routed(item: Item, prefixes: list[str]) -> bool:
    """§T10.1 routing: a domain letter (F, Q, M), a category (F1, M5) or §T6 (the code tasks)."""
    code = item.data.get("domain") == "code"
    head = item.id.split(".")[0]  # F1 for F1.1.a; K for K.M.1
    for p in prefixes:
        if p == "§T6":
            if code:
                return True
        elif code:
            continue
        elif len(p) == 1 and head.startswith(p):
            return True
        elif head == p:
            return True
    return False


def items_for(role: str, inst: Mapping[str, Any]) -> list[Item]:
    routing = inst["routing"]
    if role not in routing:
        raise HarnessValidationError(
            f"unknown role {role!r}; roles: {sorted(routing)} (§T10.1)"
        )
    reg = load_registry()
    return [i for i in reg.items.values() if routed(i, list(routing[role]))]


def rows(role: str, inst: Mapping[str, Any]) -> list[dict[str, Any]]:
    version = str(inst["survey_version"])
    out: list[dict[str, Any]] = []
    for item in items_for(role, inst):
        for q in inst["closed_questions"]:
            out.append(
                {
                    "survey_version": version,
                    "role": role,
                    "kind": "closed",
                    "faq_id": item.id,
                    "domain": item.data.get("domain"),
                    "category": item.data.get("category"),
                    "question_id": q["id"],
                    "prompt": f"{item.data.get('question')} — {q['text']}",
                    "answers": [a["label"] for a in q["answers"]],
                }
            )
    categories = sorted(
        {(i.data.get("domain"), i.data.get("category")) for i in items_for(role, inst)}
    )
    for domain, category in categories:
        for q in inst["open_questions"]:
            out.append(
                {
                    "survey_version": version,
                    "role": role,
                    "kind": "open",
                    "faq_id": None,
                    "domain": domain,
                    "category": category,
                    "question_id": q["id"],
                    "prompt": q["text"],
                    "answers": None,
                }
            )
    return out


def export_dir() -> Path:
    return repo_root() / "experiments" / "financial" / "survey"


def export_survey(role: str | None = None) -> list[Path]:
    inst = instrument()
    roles = [role] if role else sorted(inst["routing"])
    all_rows: list[dict[str, Any]] = []
    for r in roles:
        all_rows += rows(r, inst)
    out = export_dir() / str(inst["survey_version"])
    out.mkdir(parents=True, exist_ok=True)
    stem = f"instrument_{role}" if role else "instrument"
    doc = {
        "survey_version": inst["survey_version"],
        "purpose": inst["purpose"],
        "profile": inst["profile"],
        "closed_questions": inst["closed_questions"],
        "rows": all_rows,
    }
    jpath = out / f"{stem}.json"
    jpath.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    buf = io.StringIO()
    w: csv.DictWriter[str] = csv.DictWriter(
        buf, fieldnames=list(CSV_COLUMNS), lineterminator="\n"
    )
    w.writeheader()
    for r in all_rows:
        w.writerow({**r, "answers": "|".join(r["answers"]) if r["answers"] else ""})
    cpath = out / f"{stem}.csv"
    cpath.write_text(buf.getvalue(), encoding="utf-8")
    return [cpath, jpath]
