"""Calibration records for semantic features (spec §16, D-3).

``financial-harness calibrate --feature FEATURE_ID --labels PATH --panel NAME``
runs the panel's evaluator on a labelled JSONL file and writes
``experiments/financial/calibration/<feature_id>/<evaluator_version[:12]>.json``.

Each labelled item is ``{item_id, formatted_input, response_text, label: TRUE|FALSE,
arm, model_family}`` plus the prompt fields the evaluator needs, ``concept_title``,
``guide`` and ``category`` (§16 fills the legacy prompt with them; a labels file
may mix concepts). The item's ``item_id`` plays the role of the observation ID,
so call seeds are ``uint31(item_id | call_no)``.

A record made for a ``fake`` evaluator is marked ``fake_only`` and is valid only
for fake panels (§16).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from financial_llm_harness.core.bundle import atomic_write_text, read_jsonl
from financial_llm_harness.core.paths import repo_root
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.evaluation import semantic as SEM

MIN_ITEMS = 200
ITEM_KEYS = (
    "item_id",
    "formatted_input",
    "response_text",
    "label",
    "arm",
    "model_family",
    "concept_title",
    "guide",
    "category",
)
ECE_BINS = 10


def metrics(
    labels: Sequence[str], predictions: Sequence[str], confidences: Sequence[float]
) -> dict[str, Any]:
    """Accuracy, precision, recall, F1, Cohen's κ, ECE, confusion matrix, Se and Sp.

    UNKNOWN predictions are excluded from the metrics and reported as a rate.
    """
    pairs = [
        (y, p, c)
        for y, p, c in zip(labels, predictions, confidences, strict=True)
        if p != "UNKNOWN"
    ]
    n = len(pairs)
    tp = sum(1 for y, p, _ in pairs if y == "TRUE" and p == "TRUE")
    tn = sum(1 for y, p, _ in pairs if y == "FALSE" and p == "FALSE")
    fp = sum(1 for y, p, _ in pairs if y == "FALSE" and p == "TRUE")
    fn = sum(1 for y, p, _ in pairs if y == "TRUE" and p == "FALSE")
    acc = (tp + tn) / n if n else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    if n:
        p_yes = ((tp + fn) / n) * ((tp + fp) / n)
        p_no = ((tn + fp) / n) * ((tn + fn) / n)
        pe = p_yes + p_no
        kappa = 1.0 if pe == 1.0 else ((acc or 0.0) - pe) / (1 - pe)
    else:
        kappa = None
    spec = tn / (tn + fp) if tn + fp else None
    ece = 0.0
    for b in range(ECE_BINS):
        lo, hi = b / ECE_BINS, (b + 1) / ECE_BINS
        members = [
            (y, p, c) for y, p, c in pairs if (lo < c <= hi) or (b == 0 and c == 0.0)
        ]
        if members:
            conf = sum(c for _, _, c in members) / len(members)
            correct = sum(1 for y, p, _ in members if y == p) / len(members)
            ece += len(members) / n * abs(conf - correct)
    return {
        "n_evaluated": n,
        "unknown_rate": (len(labels) - n) / len(labels) if labels else None,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "ece": ece if n else None,
        "se": recall,
        "sp": spec,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def load_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise HarnessValidationError(f"labels file not found: {path}")
    items = read_jsonl(path)
    seen: set[str] = set()
    for i, item in enumerate(items):
        missing = [k for k in ITEM_KEYS if k not in item]
        if missing:
            raise HarnessValidationError(f"{path}: item {i} lacks {missing}")
        if item["label"] not in ("TRUE", "FALSE"):
            raise HarnessValidationError(
                f"{path}: item {i} label must be TRUE or FALSE"
            )
        if item["item_id"] in seen:
            raise HarnessValidationError(
                f"{path}: duplicate item_id {item['item_id']!r}"
            )
        seen.add(str(item["item_id"]))
    if len(items) < MIN_ITEMS:
        raise HarnessValidationError(
            f"{path}: {len(items)} labelled items; calibration needs at least {MIN_ITEMS} (§16, D-3)"
        )
    return items


async def _judge_items(
    adapter: Any,
    caps: Any,
    items: Sequence[Mapping[str, Any]],
    calls: Sequence[int],
) -> list[list[str]]:
    out: list[list[str]] = []
    for item in items:
        prompt = SEM.fill_prompt(
            str(item["formatted_input"]),
            str(item["concept_title"]),
            str(item["guide"]),
            str(item["category"]),
            str(item["response_text"]),
        )
        verdicts: list[str] = []
        for call_no in calls:
            request = SEM.evaluator_request(
                prompt, str(item["item_id"]), call_no, caps, len(calls)
            )
            text = await SEM.run_call(adapter, request, lambda n, r: None)
            verdicts.append(SEM.verdict(text))
        out.append(verdicts)
    return out


def calibrate(feature_id: str, labels: Path | str, panel_name: str) -> Path:
    """Run the panel's evaluator on the labelled items and write the calibration record."""
    from financial_llm_harness.generation.generate import resolve_model
    from financial_llm_harness.models.panels import load_panel

    labels_path = Path(labels)
    items = load_items(labels_path)
    panel = load_panel(panel_name)
    entry = panel.evaluator
    if entry is None:
        raise HarnessValidationError(
            f"panel {panel_name!r} has no evaluator entry (§16, §20.9)"
        )
    model, _ = resolve_model(entry)
    from financial_llm_harness.models.base import ModelCapabilities

    caps = ModelCapabilities.from_dict(model["capabilities"])
    calls = SEM.calls_for(caps)
    version = SEM.evaluator_version(model, len(calls))
    adapter = SEM.make_evaluator(entry, caps)
    verdicts = asyncio.run(_judge_items(adapter, caps, items, calls))
    predictions = [SEM.majority(v) for v in verdicts]
    confidences = [
        sum(1 for x in v if x == p) / len(v)
        for v, p in zip(verdicts, predictions, strict=True)
    ]
    m = metrics([str(i["label"]) for i in items], predictions, confidences)
    strata = Counter(f"{i['arm']}|{i['model_family']}" for i in items)
    record = {
        "feature_id": feature_id,
        "evaluator_version": version,
        "evaluator": {
            "model_key": entry["model_key"],
            "provider": entry["provider"],
            "model": entry["model"],
            "family": entry["family"],
            "revision": model.get("revision"),
            "n_calls": len(calls),
        },
        "fake_only": entry["provider"] == "fake",
        "prompt_hash": SEM.prompt_hash(),
        "parser_version": SEM.PARSER_VERSION,
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "n": len(items),
        "strata": dict(sorted(strata.items())),
        "kappa_min": SEM.KAPPA_MIN,
        "deployable": m["kappa"] is not None and m["kappa"] >= SEM.KAPPA_MIN,
        "metrics": m,
    }
    out = SEM.calibration_path(feature_id, version)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(record, indent=2, sort_keys=True) + "\n")
    return out


def rogan_gladen(p_hat: float, se: float, sp: float) -> float | None:
    """p* = (p̂ + Sp − 1)/(Se + Sp − 1), clipped to [0, 1]; None when Se + Sp ≤ 1 (§16)."""
    denom = se + sp - 1.0
    if denom <= 0:
        return None
    return min(1.0, max(0.0, (p_hat + sp - 1.0) / denom))


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(repo_root().resolve()).as_posix()
