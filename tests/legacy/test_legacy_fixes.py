"""Regression tests for the two legacy fixes allowed in V1 (spec §0.5)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest

from biases_in_the_blind_spot.concept_pipeline.cluster_id import ClusterId
from biases_in_the_blind_spot.concept_pipeline.concept_bias_test_result import (
    ConceptBiasTestResult,
)
from biases_in_the_blind_spot.concept_pipeline.concept_id import ConceptId
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_dataset import (
    ConceptPipelineDataset,
)
from biases_in_the_blind_spot.concept_pipeline.concept_pipeline_result import (
    ConceptPipelineResult,
    StageResults,
)
from biases_in_the_blind_spot.concept_pipeline.input_clusterer import InputClusterer
from biases_in_the_blind_spot.concept_pipeline.input_id import InputId
from biases_in_the_blind_spot.concept_pipeline.response_id import ResponseId
from biases_in_the_blind_spot.concept_pipeline.unfaithful_concepts import (
    build_unfaithful_concepts_html,
)
from biases_in_the_blind_spot.concept_pipeline.variation_pair_id import VariationPairId
from biases_in_the_blind_spot.concept_pipeline.variation_pair_responses import (
    VariationPairResponses,
)
from biases_in_the_blind_spot.concept_pipeline.variation_pair_verbalization import (
    VariationPairVerbalization,
)
from biases_in_the_blind_spot.concept_pipeline.verbalization_check_result import (
    VerbalizationCheckResult,
)
from tests.concept_pipeline.helpers import make_dataset


def _iid(i: int) -> InputId:
    import uuid

    return InputId(str(uuid.UUID(int=i + 1)))


def _result(
    test: Literal["mcnemar", "fisher"],
) -> tuple[ConceptPipelineDataset, ConceptPipelineResult]:
    cid = ConceptId()
    iid = InputId()
    pid = VariationPairId()
    rp, rn, rb = ResponseId(), ResponseId(), ResponseId()
    flags = VerbalizationCheckResult(verbalized=False, witness="")
    stats_key = "mcnemar_p_value" if test == "mcnemar" else "fisher_p_value"
    bias = ConceptBiasTestResult(
        responses_by_input={
            iid: {
                pid: VariationPairResponses(
                    positive_responses={rp: "p"},
                    negative_responses={rn: "n"},
                    positive_acceptances={rp: 1},
                    negative_acceptances={rn: 0},
                )
            }
        },
        flipped_variation_pairs={iid: {pid: True}},
        statistics_positive_vs_negative={stats_key: 0.0123},
        statistics_positive={"acceptance_proportion": 1.0, "invalid_proportion": 0.0},
        statistics_negative={"acceptance_proportion": 0.0, "invalid_proportion": 0.0},
    )
    stage = StageResults(
        stage_idx=0,
        k_inputs_per_representative_cluster=1,
        seed=0,
        input_indices_by_representative_cluster={ClusterId(0): [iid]},
        concepts_at_stage_start=[cid],
        significant_concepts=[cid],
        concept_ids_unverbalized_on_baseline=[cid],
        concept_verbalization_on_baseline_responses={cid: {iid: {rb: flags}}},
        variation_bias_results={cid: bias},
        concept_verbalization_on_variation_responses={
            cid: {
                iid: {
                    pid: VariationPairVerbalization(
                        positive_variation_responses_verbalizations={rp: flags},
                        negative_variation_responses_verbalizations={rn: flags},
                    )
                }
            }
        },
    )
    result = ConceptPipelineResult(
        stages=[stage],
        baseline_verbalization_threshold=0.25,
        variations_verbalization_threshold=0.5,
        significance_test=test,
        significant_unfaithful_concepts=[cid],
        baseline_acceptances_by_input={iid: {rb: 1}},
    )
    return make_dataset(cid, iid), result


@pytest.mark.parametrize("test", ["mcnemar", "fisher"])
def test_unfaithful_html_reads_configured_p_value_key(
    tmp_path: Path, test: Literal["mcnemar", "fisher"]
) -> None:
    dataset, result = _result(test)
    build_unfaithful_concepts_html(tmp_path, dataset, result)
    html = (tmp_path / "unfaithful_concepts.html").read_text()
    assert "P_value (positive vs negative): 0.0123" in html


def test_kmeans_seed_default_none_and_seeded_is_reproducible() -> None:
    assert InputClusterer().seed is None
    emb = {_iid(i): [float(i % 5), float((i * 7) % 3), float(i % 2)] for i in range(40)}
    with patch(
        "biases_in_the_blind_spot.concept_pipeline.input_clusterer.KMeans"
    ) as km:
        km.return_value.fit_predict.return_value = [i % 3 for i in range(40)]
        InputClusterer._kmeans_cluster(emb, 3)
        assert km.call_args.kwargs["random_state"] is None
        InputClusterer._kmeans_cluster(emb, 3, 7)
        assert km.call_args.kwargs["random_state"] == 7
    a = InputClusterer._kmeans_cluster(emb, 3, 11)
    b = InputClusterer._kmeans_cluster(emb, 3, 11)
    assert a == b


def test_clustering_persists_seed_on_dataset(tmp_path: Path) -> None:
    iid = _iid(0)
    ds = make_dataset(ConceptId(), iid)
    ds.n_representative_input_clusters = 1
    assert ds.input_clustering_seed is None
    clusterer = InputClusterer(seed=5)

    async def fake_embed(texts: dict, max_concurrent: int = 100) -> dict:
        return {k: [1.0, 0.0] for k in texts}

    with patch.object(InputClusterer, "_embed_batch", side_effect=fake_embed):
        asyncio.run(clusterer.cluster_inputs_if_needed(ds, tmp_path))
    assert ds.input_clustering_seed == 5
    loaded = ConceptPipelineDataset.from_dict(
        {k: v for k, v in ds.to_dict().items() if k != "input_clustering_seed"}
    )
    assert loaded.input_clustering_seed is None


class _Stop(Exception):
    pass


def _run_hypothesis_sampling(seed: int | None) -> ConceptPipelineDataset:
    from types import SimpleNamespace

    from biases_in_the_blind_spot.concept_pipeline.implicit_concept_detector import (
        ImplicitConceptDetector,
    )

    ds = make_dataset(ConceptId(), _iid(0))
    ds.input_clusters = SimpleNamespace(clusters=[[_iid(0)]])  # type: ignore[assignment]
    ds.n_inputs_per_representative_cluster_for_concept_hypothesis = 1
    ids = [_iid(i) for i in range(30)]
    detector = ImplicitConceptDetector(
        llm_model_name="none", hypothesis_sampling_seed=seed
    )
    with (
        patch.object(
            ConceptPipelineDataset,
            "select_inputs_from_each_representative_cluster",
            return_value={ClusterId(0): list(ids)},
        ),
        patch.object(ConceptPipelineDataset, "get_input_params", side_effect=_Stop),
        pytest.raises(_Stop),
    ):
        asyncio.run(detector.hypothesize_concepts(ds, Path("."), "absence"))
    return ds


def test_hypothesis_shuffle_seed() -> None:
    import random

    from biases_in_the_blind_spot.concept_pipeline.implicit_concept_detector import (
        ImplicitConceptDetector,
    )

    assert (
        ImplicitConceptDetector.__dataclass_fields__["hypothesis_sampling_seed"].default
        is None
    )
    a = _run_hypothesis_sampling(9)
    b = _run_hypothesis_sampling(9)
    expected = [_iid(i) for i in range(30)]
    random.Random(9).shuffle(expected)
    assert a.concept_hypothesis_sampling_seed == 9
    assert (
        a.concept_hypothesis_input_indices
        == expected
        == b.concept_hypothesis_input_indices
    )
    unseeded = _run_hypothesis_sampling(None)
    assert unseeded.concept_hypothesis_sampling_seed is None
    assert sorted(unseeded.concept_hypothesis_input_indices or []) == sorted(expected)
