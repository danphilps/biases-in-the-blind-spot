"""Legacy results check (spec §0.6, E-14)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "experiments/llm_bias/concept_pipeline/results/concept_pipeline_llm_bias_gpt-4.1_stage-0.json",
    "experiments/loan_approval/concept_pipeline/results/concept_pipeline_loan_approval_gpt-4.1_stage-0.json",
    "experiments/university_admission/concept_pipeline/results/concept_pipeline_university_admission_gpt-4.1_stage-0.json",
]
LFS_POINTER = "version https://git-lfs.github.com/spec/v1"


@pytest.mark.parametrize("rel", FILES)
def test_legacy_stage_results_load(rel: str) -> None:
    path = ROOT / rel
    with open(path, encoding="utf-8") as fh:
        head = fh.read(len(LFS_POINTER))
    if head.startswith(LFS_POINTER):
        if os.environ.get("CI") == "true":
            pytest.fail(f"{rel} is an LFS pointer in CI; CI checks out LFS")
        pytest.skip(f"{rel} is an LFS pointer that was never pulled")
    from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
        StageResults,
    )

    stage = StageResults.from_json(path.read_text(encoding="utf-8"))
    assert isinstance(stage, StageResults)
    assert stage.stage_idx == 0
    assert len(stage.get_stage_input_ids()) > 0
