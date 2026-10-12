"""Design projection, ``design_hash`` and ``manifest_hash`` (spec §4, E-25).

``design_hash`` identifies the scientific stimuli independently of the model
panel; ``manifest_hash`` identifies one executable frozen envelope.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from financial_llm_harness.core.canonical_json import sha256_canonical

SCIENTIFIC_ROLES = frozenset(
    {
        "template",
        "paraphrase",
        "bank",
        "mandate",
        "oracle",
        "contract",
        "family_config",
        "hidden_tests",
        "sandbox_recipe",
    }
)

ALL_ROLES = SCIENTIFIC_ROLES | {
    "evaluator_prompt",
    "evaluator_parser",
    "panel",
    "probe",
    "request_profile",
}

DESIGN_KEYS: tuple[str, ...] = (
    "schema_version",
    "experiment_key",
    "variant_id",
    "mode",
    "parent_discovery_experiment_id",
    "faq_id",
    "intervention_id",
    "domain",
    "task_family",
    "task",
    "concept",
    "research_question",
    "mandate",
    "hypothesis",
    "scenario_spec",
    "intervention_spec",
    "controls",
    "arms",
    "prompt_spec",
    "paraphrase_spec",
    "response_spec",
    "feature_spec",
    "oracle_spec",
    "sampling_spec",
    "statistical_spec",
    "seeds",
    "versions",
)

# Keys excluded from the manifest-hash input: derived values never enter their own hash.
MANIFEST_HASH_EXCLUDED = ("manifest_hash", "experiment_id", "created_at")


def scientific_assets(assets: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """``{role, logical_name, sha256}`` of every scientific asset row, sorted."""
    rows = [
        {"role": a["role"], "logical_name": a["logical_name"], "sha256": a["sha256"]}
        for a in assets
        if a["role"] in SCIENTIFIC_ROLES
    ]
    rows.sort(key=lambda r: (r["role"], r["logical_name"]))
    return rows


def design_projection(
    manifest: Mapping[str, Any], assets: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The JSON object whose canonical sha256 is ``design_hash``."""
    proj: dict[str, Any] = {}
    for key in DESIGN_KEYS:
        if key not in manifest:
            continue
        value = copy.deepcopy(manifest[key])
        if key == "response_spec" and isinstance(value, dict):
            value.pop("enforcement_resolved", None)
        if key == "sampling_spec" and isinstance(value, dict):
            value.pop("panel_ref", None)
            value.pop("models", None)
            value.pop("models_skipped", None)  # key gating must not move seeds (§18)
            value.pop("panel", None)
        proj[key] = value
    corpus = manifest.get("corpus") or {}
    proj["corpus"] = {
        "cases": {"sha256": corpus.get("cases", {}).get("sha256")},
        "renderings": {"sha256": corpus.get("renderings", {}).get("sha256")},
    }
    proj["scientific_assets"] = scientific_assets(assets)
    return proj


def compute_design_hash(
    manifest: Mapping[str, Any], assets: Sequence[Mapping[str, Any]]
) -> str:
    return sha256_canonical(design_projection(manifest, assets))


def compute_manifest_hash(manifest: Mapping[str, Any]) -> str:
    body = {k: v for k, v in manifest.items() if k not in MANIFEST_HASH_EXCLUDED}
    return sha256_canonical(body)
