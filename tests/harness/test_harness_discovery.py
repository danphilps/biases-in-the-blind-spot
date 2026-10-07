"""Discovery (spec §27, §27.1, M7)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from financial_llm_harness.core import yaml_io
from financial_llm_harness.discovery import import_legacy
from financial_llm_harness.errors import HarnessValidationError
from tests.harness.helpers import FIXED_CODE_REF, fixed_clock, small_spec

C1 = "00000000-0000-4000-8000-0000000000c1"
C2 = "00000000-0000-4000-8000-0000000000c2"
C3 = "00000000-0000-4000-8000-0000000000c3"


def _legacy_fixture(directory: Path, significant: list[str] | None) -> None:
    """A small legacy result and dataset, written with the legacy persistence code."""
    from biases_in_the_blind_spot.concept_pipeline.concept import Concept
    from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
    from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
        ConceptPipelineDataset,
    )
    from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
        ConceptPipelineResult,
    )
    from biases_in_the_blind_spot.concept_pipeline.pipeline_persistence import (
        get_result_path,
        save_result,
    )

    concepts = [
        Concept(
            title=f"concept {k}",
            verbalization_check_guide=f"guide {k}",
            removal_action=f"remove {k}",
            addition_action=f"add {k}",
            id=ConceptId(cid),
        )
        for k, cid in enumerate((C1, C2))
    ]
    ds = ConceptPipelineDataset(
        dataset_name="tiny",
        input_template="{text}",
        input_parameters={},
        varying_input_param_name="text",
        varying_inputs={},
        concepts=concepts,
    )
    ds.save(directory)
    result = ConceptPipelineResult(
        experiment_key="tiny_key",
        significant_unfaithful_concepts=[ConceptId(c) for c in significant]
        if significant is not None
        else None,
    )
    save_result(result, get_result_path("tiny_key", directory))


def test_import_legacy_writes_one_entry_per_concept(tmp_path: Path) -> None:
    _legacy_fixture(tmp_path, [C2, C1])
    out = tmp_path / "hypotheses.yaml"
    assert import_legacy(tmp_path, "tiny_key", "tiny", out) == 2
    data = yaml_io.load(out)
    assert isinstance(data, dict)
    entries = data["hypotheses"]
    assert [e["concept_id"] for e in entries] == [C2, C1]
    assert entries[0] == {
        "source": "legacy_concept_pipeline",
        "legacy_experiment_key": "tiny_key",
        "concept_id": C2,
        "title": "concept 1",
        "verbalization_check_guide": "guide 1",
        "addition_action": "add 1",
        "removal_action": "remove 1",
    }
    # it never writes a spec
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".yaml") == [
        "hypotheses.yaml"
    ]


@pytest.mark.parametrize("significant", [None, []])
def test_import_legacy_exits_2_without_significant_concepts(
    tmp_path: Path, significant: list[str] | None
) -> None:
    _legacy_fixture(tmp_path, significant)
    with pytest.raises(
        HarnessValidationError, match="significant_unfaithful_concepts"
    ) as exc:
        import_legacy(tmp_path, "tiny_key", "tiny", tmp_path / "h.yaml")
    assert exc.value.exit_code == 2
    assert not (tmp_path / "h.yaml").exists()


def test_import_legacy_exits_2_for_a_missing_concept(tmp_path: Path) -> None:
    _legacy_fixture(tmp_path, [C1, C3])
    with pytest.raises(HarnessValidationError, match="missing from dataset") as exc:
        import_legacy(tmp_path, "tiny_key", "tiny", tmp_path / "h.yaml")
    assert exc.value.exit_code == 2


def test_confirmatory_vector_within_epsilon_of_parent_exits_2(
    tmp_path: Path, harness_home: Path
) -> None:
    """§27: a confirmatory spec whose primitive vectors repeat its parent discovery corpus."""
    from financial_llm_harness.execution.runner import run_experiment
    from financial_llm_harness.generation.generate import generate
    from financial_llm_harness.generation.validate_bundle import validate_bundle

    disc = small_spec(tmp_path, n_cases=60, name="disc.yaml", mode="discovery")
    res = run_experiment(
        disc,
        "ci_fake",
        yes=True,
        clock=fixed_clock,
        code_ref=FIXED_CODE_REF,
        run_id="00000000-0000-4000-8000-0000000000d1",
    )
    assert res.run_dir is not None
    parent = yaml_io.load(res.run_dir / "manifest" / "manifest.yaml")
    assert isinstance(parent, dict)
    # same seed stream and generation seed: the confirmatory corpus repeats the parent's vectors
    spec = yaml_io.load(disc)
    assert isinstance(spec, dict)
    conf = copy.deepcopy(spec)
    conf["mode"] = "confirmatory"
    conf["hypothesis"]["source"] = "discovery"
    conf["parent_discovery_experiment_id"] = parent["experiment_id"]
    path = tmp_path / "conf.yaml"
    path.write_text(yaml_io.dump(conf), encoding="utf-8")
    out = tmp_path / "bundle"
    generate(path, out, code_ref=FIXED_CODE_REF, clock=fixed_clock)
    with pytest.raises(HarnessValidationError, match="within ε of parent") as exc:
        validate_bundle(out, code_ref=FIXED_CODE_REF, clock=fixed_clock)
    assert exc.value.exit_code == 2
    # a disjoint seed stream passes the check
    conf["scenario_spec"]["seed_stream"] = "conf"
    conf["seeds"]["generation"] = int(conf["seeds"]["generation"]) + 1000
    path.write_text(yaml_io.dump(conf), encoding="utf-8")
    out2 = tmp_path / "bundle2"
    generate(path, out2, code_ref=FIXED_CODE_REF, clock=fixed_clock)
    validate_bundle(out2, code_ref=FIXED_CODE_REF, clock=fixed_clock)
