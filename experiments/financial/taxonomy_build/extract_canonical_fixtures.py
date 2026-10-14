"""Writes generator-regression fixtures: the first three cases of each V1 canonical spec and of the S1-cal calibration specs (§7, §35).

Each fixture holds, per case, the sampled state, the arm A and arm B states and the
arm A and B renderings of paraphrase 0, from the spec's own seeds in discovery mode
(arms A and B, n_cases 3). They are extracted from generator output, not hand-
authored: they catch generator drift, while the hand-authored canonical cases of
§7 remain the user's (D-14). Run from the repo root.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io
from financial_llm_harness.generation.corpus import build_corpus

SPECS = (
    "momentum_stock_add.yaml",
    "momentum_stock_add_m2.yaml",
    "stock_attractive_momentum_m0.yaml",
    "stock_attractive_momentum_m1.yaml",
    "stock_attractive_momentum_m2.yaml",
    "transcript_summary.yaml",
    "covariance_choice.yaml",
    "temporal_norm_design.yaml",
    "temporal_norm_code.yaml",
)
OUT = Path("tests/harness/fixtures/canonical")


def small(spec: dict[str, Any]) -> dict[str, Any]:
    s = copy.deepcopy(spec)
    s["mode"] = "discovery"
    s["scenario_spec"]["n_cases"] = 3
    return s


def fixture(spec: dict[str, Any]) -> dict[str, Any]:
    corpus = build_corpus(small(spec))
    users = {
        (r["case_idx"], r["arm"], r["paraphrase_idx"]): r["user"]
        for r in corpus.renderings
    }
    return {
        "source": "generator_regression",
        "experiment_key": spec["experiment_key"],
        "cases": [
            {
                "sampled": c["sampled"],
                "arms": {a: c["arms"][a] for a in ("A", "B")},
                "user": {a: users[(c["case_idx"], a, 0)] for a in ("A", "B")},
            }
            for c in corpus.cases
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SPECS:
        spec = yaml_io.load(Path("experiments/financial/specs") / name)
        fx = fixture(spec)
        path = OUT / f"{spec['experiment_key']}.json"
        path.write_text(
            json.dumps(fx, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("wrote", path)


if __name__ == "__main__":
    main()
