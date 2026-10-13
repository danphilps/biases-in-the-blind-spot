"""Calibration-set tooling for the semantic evaluator (spec §16, D-3): export, then merge.

    export:  python export_calibration_items.py export RUN_DIR FEATURE_ID OUT_DIR [--n 200] [--seed S]
    merge:   python export_calibration_items.py merge OUT_DIR LABELS_CSV ITEMS_JSONL

`export` samples parsed responses of a run (a pilot run of the target panel, never
the fake path for a real calibration), stratified round-robin over (arm, model
family) with a fixed seed, and writes two files:

- `items_blind.csv`, for the human labellers: item_id, the prompt the model saw,
  the text the evaluator would judge, the concept, guide and category, and an empty
  `label` column (TRUE or FALSE). The arm and the model are withheld, so labels are
  not conditioned on them;
- `key.jsonl`: item_id → arm, model family, trial_id, kept away from the labellers.

`merge` joins the filled CSV with the key into the JSONL that
`financial-harness calibrate FEATURE --labels ITEMS_JSONL --panel PANEL` reads
(§16: at least 200 items; κ ≥ 0.70 to deploy).
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core.ids import uint64
from financial_llm_harness.evaluation.calibration import ITEM_KEYS
from financial_llm_harness.evaluation.parse import parse_response
from financial_llm_harness.evaluation.semantic import _channel_text, prompt_fields
from financial_llm_harness.families import load_family
from financial_llm_harness.generation.corpus import load_design_contract
from financial_llm_harness.storage.artifacts import ArtifactStore

BLIND_COLUMNS = ("item_id", "formatted_input", "response_text", "concept_title", "guide", "category", "label")


def export(run_dir: Path, feature_id: str, out_dir: Path, n: int = 200, seed: int = 20261008) -> None:
    manifest = B.load_manifest(run_dir / "manifest")
    feature = next(f for f in manifest["feature_spec"] if f["feature_id"] == feature_id)
    family = load_family(manifest["task_family"])
    fam_feature = family.features.get(feature_id)
    merged = {**(dict(fam_feature.data) if fam_feature else {}), **feature}
    guide, category = prompt_fields(manifest, feature)
    contract, _, _ = load_design_contract(manifest, family)
    rs = manifest["response_spec"]
    mode, abstain = rs["reasoning_mode"], bool(rs["abstain_allowed"])
    renderings = {
        (r["case_idx"], r["arm"], r["paraphrase_idx"]): r
        for r in B.read_jsonl(run_dir / "manifest" / B.RENDERINGS_FILE)
    }
    families = {m["model_key"]: m["family"] for m in manifest["sampling_spec"]["models"]}
    store = ArtifactStore(run_dir / "artifacts")
    db = sqlite3.connect(run_dir / "db" / "run.sqlite")
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT t.trial_id, t.case_idx, t.arm, t.paraphrase_idx, t.model_key, t.status, "
        "r.text_artifact, r.trace_artifact FROM trial t JOIN response r ON r.attempt_id = t.used_attempt_id "
        "WHERE t.used_attempt_id IS NOT NULL AND t.status = 'SUCCESS' ORDER BY t.trial_id"
    ).fetchall()
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        text = store.get_text(row["text_artifact"]) if row["text_artifact"] else None
        parsed = parse_response(contract, mode, abstain, text)
        if parsed.status != "SUCCESS" or parsed.data is None:
            continue
        trace = store.get_text(row["trace_artifact"]) if row["trace_artifact"] else None
        response = _channel_text(merged, parsed.data, trace, contract, mode)
        if response is None:
            continue
        rendering = renderings[(row["case_idx"], row["arm"], row["paraphrase_idx"])]
        fam = families.get(row["model_key"], "unknown")
        strata[(row["arm"], fam)].append(
            {"trial_id": row["trial_id"], "arm": row["arm"], "model_family": fam,
             "formatted_input": rendering["user"], "response_text": response}
        )
    rng = np.random.default_rng(uint64("calibration_export", seed))
    pools = {k: [v[int(i)] for i in rng.permutation(len(v))] for k, v in sorted(strata.items())}
    chosen: list[dict[str, Any]] = []
    while len(chosen) < n and any(pools.values()):
        for k in sorted(pools):
            if pools[k] and len(chosen) < n:
                chosen.append(pools[k].pop())
    if len(chosen) < n:
        print(f"warning: only {len(chosen)} eligible responses (< {n})", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = [int(i) for i in rng.permutation(len(chosen))]  # labellers see a shuffled order
    with (out_dir / "items_blind.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(BLIND_COLUMNS))
        w.writeheader()
        for k, idx in enumerate(order):
            c = chosen[idx]
            w.writerow({
                "item_id": f"{feature_id}-{k:04d}", "formatted_input": c["formatted_input"],
                "response_text": c["response_text"], "concept_title": manifest["concept"],
                "guide": guide, "category": category, "label": "",
            })
    with (out_dir / "key.jsonl").open("w", encoding="utf-8") as fh:
        for k, idx in enumerate(order):
            c = chosen[idx]
            fh.write(json.dumps({"item_id": f"{feature_id}-{k:04d}", "arm": c["arm"],
                                 "model_family": c["model_family"], "trial_id": c["trial_id"]}) + "\n")
    print(f"wrote {len(chosen)} items to {out_dir}")


def merge(out_dir: Path, labels_csv: Path, items_jsonl: Path) -> None:
    key = {r["item_id"]: r for r in B.read_jsonl(out_dir / "key.jsonl")}
    with labels_csv.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with items_jsonl.open("w", encoding="utf-8") as fh:
        for r in rows:
            label = (r.get("label") or "").strip().upper()
            if label not in ("TRUE", "FALSE"):
                raise SystemExit(f"{r['item_id']}: label must be TRUE or FALSE, got {r.get('label')!r}")
            k = key[r["item_id"]]
            item = {**{c: r[c] for c in BLIND_COLUMNS if c != "label"}, "label": label,
                    "arm": k["arm"], "model_family": k["model_family"]}
            assert set(ITEM_KEYS) <= set(item)
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} labelled items to {items_jsonl}")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "export":
        run, feat, out, *opt = rest
        n = int(opt[opt.index("--n") + 1]) if "--n" in opt else 200
        seed = int(opt[opt.index("--seed") + 1]) if "--seed" in opt else 20261008
        export(Path(run), feat, Path(out), n, seed)
    elif cmd == "merge":
        out, labels, items = rest
        merge(Path(out), Path(labels), Path(items))
    else:
        raise SystemExit(__doc__)
