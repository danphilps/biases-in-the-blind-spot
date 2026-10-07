"""Discovery → confirmation (spec §27, §27.1, E-12).

``import_legacy`` is the legacy discovery bridge: it turns the significant
unfaithful concepts of a legacy concept-pipeline result into hypothesis entries.
It never writes a spec; a person turns a hypothesis into a structured-world
intervention and a confirmatory spec (§27.1).

``check_parent_overlap`` is the §27 seed-stream check: no confirmatory
primitive vector may lie within ε of a vector of its parent discovery corpus,
by the §9.2 structured near-duplicate rule over the two corpora together.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import yaml_io
from financial_llm_harness.errors import HarnessValidationError

HYPOTHESIS_KEYS = (
    "source",
    "legacy_experiment_key",
    "concept_id",
    "title",
    "verbalization_check_guide",
    "addition_action",
    "removal_action",
)


def import_legacy(
    results_dir: Path, experiment_key: str, dataset: str, output: Path
) -> int:
    """Write one hypothesis per significant unfaithful concept; returns the count (§27.1)."""
    persistence = importlib.import_module(
        "biases_in_the_blind_spot.concept_pipeline.pipeline_persistence"
    )
    dataset_mod = importlib.import_module(
        "biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset"
    )
    result = persistence.load_pipeline_result_for_experiment(
        experiment_key, Path(results_dir)
    )
    if result is None:
        raise HarnessValidationError(
            f"no legacy result for experiment {experiment_key!r} under {results_dir}"
        )
    ids = list(getattr(result, "significant_unfaithful_concepts", None) or [])
    if not ids:
        raise HarnessValidationError(
            f"legacy result {experiment_key!r} has no significant_unfaithful_concepts (§27.1)"
        )
    ds = dataset_mod.ConceptPipelineDataset.load_by_name(dataset, Path(results_dir))
    if ds is None:
        raise HarnessValidationError(
            f"legacy dataset {dataset!r} not found under {results_dir}"
        )
    concepts = {
        str(c.id): c
        for c in (ds.concepts or [])
        + (ds.deduplicated_concepts or [])
        + (ds.final_concepts or [])
    }
    entries: list[dict[str, Any]] = []
    for cid in ids:
        concept = concepts.get(str(cid))
        if concept is None:
            raise HarnessValidationError(
                f"concept {cid} of {experiment_key!r} is missing from dataset {dataset!r} (§27.1)"
            )
        entries.append(
            {
                "source": "legacy_concept_pipeline",
                "legacy_experiment_key": experiment_key,
                "concept_id": str(cid),
                "title": concept.title,
                "verbalization_check_guide": concept.verbalization_check_guide,
                "addition_action": concept.addition_action,
                "removal_action": concept.removal_action,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    B.atomic_write_text(output, yaml_io.dump({"hypotheses": entries}))
    return len(entries)


# ---------------------------------------------------------------------------
# §27 seed-stream check
# ---------------------------------------------------------------------------


def find_bundle(experiment_id: str) -> Path | None:
    """The frozen bundle of an experiment under HARNESS_HOME (a run's ``manifest/``)."""
    from financial_llm_harness.core.paths import runs_dir

    runs = runs_dir()
    if not runs.exists():
        return None
    for manifest in sorted(runs.glob("*/manifest/manifest.yaml")):
        data = yaml_io.load(manifest)
        if isinstance(data, Mapping) and data.get("experiment_id") == experiment_id:
            return manifest.parent
    return None


def check_parent_overlap(
    manifest: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    nodes: Mapping[str, Any],
) -> None:
    """Exit 2 when a confirmatory primitive vector lies within ε of a parent discovery vector."""
    from financial_llm_harness.generation.validate_corpus import structured_edges

    parent = manifest.get("parent_discovery_experiment_id")
    if manifest.get("mode") != "confirmatory" or not parent:
        return
    bundle = find_bundle(str(parent))
    if bundle is None:
        raise HarnessValidationError(
            f"parent discovery experiment {parent} has no frozen bundle under HARNESS_HOME (§27)"
        )
    parent_cases = B.read_jsonl(bundle / B.CASES_FILE)
    mine = [c["arms"]["A"] for c in cases]
    theirs = [c["arms"]["A"] for c in parent_cases]
    shared = {
        name
        for name, nd in nodes.items()
        if nd.kind == "primitive" and all(name in s for s in mine + theirs)
    }
    states = mine + theirs
    displayed = [{k: str(s[k]) for k in sorted(shared)} for s in states]
    n_mine = len(mine)
    for a, b in structured_edges(states, nodes, shared, displayed):
        if (a < n_mine) != (b < n_mine):
            i, j = (a, b - n_mine) if a < n_mine else (b, a - n_mine)
            raise HarnessValidationError(
                f"confirmatory case {cases[i]['case_idx']} lies within ε of parent discovery case "
                f"{parent_cases[j]['case_idx']} (§27, §9.2)"
            )
