"""L8: deterministic evaluation (spec §31A L8, §16).

One observation per (trial with a selected response, deterministic feature).
Re-parsing a stored response is deterministic, so re-evaluation rewrites the
same rows (observation IDs depend only on trial, feature and evaluator version).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from financial_llm_harness.core import ids
from financial_llm_harness.evaluation.parse import parse_response
from financial_llm_harness.evaluation.rules import FeatureContext, extract, is_semantic
from financial_llm_harness.families import load_family
from financial_llm_harness.models.base import ModelResponse

VALID_FOR_FEATURES = ("SUCCESS", "COMPILE_FAILURE", "TEST_FAILURE", "RUNTIME_FAILURE")
MODEL_LEVEL_STATUSES = (
    "SUCCESS",
    "REFUSAL",
    "EMPTY_RESPONSE",
    "TRUNCATED",
    "CONTENT_FILTERED",
    "INVALID_RESPONSE",
    "PARSE_FAILURE",
)


def evaluator_version(feature: Mapping[str, Any]) -> str:
    return f"{feature['extractor']}@{feature.get('extractor_version', 1)}"


def concept_factor_for(manifest: Mapping[str, Any]) -> str | None:
    if manifest.get("battery"):
        return None
    family = load_family(manifest["task_family"])
    idef = family.interventions.get(manifest["intervention_id"])
    return idef.concept_factor if idef is not None else None


def evaluate_deterministic(ctx: Any, *, execute_code: bool = True) -> None:
    """Re-derive model-level statuses from the selected responses, then write observations.

    For a code task, ``execute_code`` first runs every pending parsed response in
    the sandbox (§13.4); re-evaluation passes False and reads the stored
    ``code_outcomes`` events, so it never executes code (§31).
    """
    from financial_llm_harness.execution.runner import model_status

    manifest = ctx.manifest
    if execute_code and manifest["task_family"] == "code_task":
        from financial_llm_harness.evaluation.sandbox import execute_code_trials

        execute_code_trials(ctx)
    rs = manifest["response_spec"]
    mode = rs["reasoning_mode"]
    abstain = bool(rs["abstain_allowed"])
    family = load_family(manifest["task_family"])
    concept_factor = concept_factor_for(manifest)
    features = [f for f in manifest["feature_spec"] if not is_semantic(f)]
    semantic_with_rule = [
        f for f in manifest["feature_spec"] if f.get("extractor") == "key_points_rule"
    ]
    features += [f for f in semantic_with_rule if f not in features]
    code_task = manifest["task_family"] == "code_task"
    rows = ctx.db.execute(
        "SELECT t.trial_id, t.arm, t.case_idx, t.status, t.used_attempt_id, r.text_artifact, r.finish_reason "
        "FROM trial t LEFT JOIN response r ON r.attempt_id = t.used_attempt_id WHERE t.used_attempt_id IS NOT NULL "
        "ORDER BY t.trial_id"
    ).fetchall()
    obs_rows: list[tuple[Any, ...]] = []
    status_updates: list[tuple[str, str]] = []
    for row in rows:
        text = (
            ctx.store.get_text(row["text_artifact"]) if row["text_artifact"] else None
        )
        if row["status"] in MODEL_LEVEL_STATUSES or (
            code_task and row["status"] in VALID_FOR_FEATURES
        ):
            probe = ModelResponse(
                b"",
                200,
                "complete",
                text,
                None,
                row["finish_reason"],
                None,
                None,
                None,
                None,
                None,
                None,
                0,
            )
            recomputed = model_status(probe, ctx.contract, mode, abstain)
            if not code_task and recomputed != row["status"]:
                status_updates.append((recomputed, row["trial_id"]))
            if code_task and recomputed != "SUCCESS" and row["status"] != recomputed:
                status_updates.append((recomputed, row["trial_id"]))
        status = row["status"]
        if status not in VALID_FOR_FEATURES:
            continue
        parsed = parse_response(ctx.contract, mode, abstain, text)
        if parsed.status != "SUCCESS" or parsed.data is None:
            continue
        case = ctx.cases[row["case_idx"]]
        code_outcomes = None
        if code_task:
            code_outcomes = _code_outcomes(ctx, row["trial_id"])
            if code_outcomes is None:
                continue
        cue = case.get("cue") or {}
        fctx = FeatureContext(
            arm=row["arm"],
            concept_factor=concept_factor,
            stance_target=cue.get("stance_target"),
            truth=case.get("truth"),
            code_outcomes=code_outcomes,
            state=case["arms"].get(row["arm"]),
        )
        for f in features:
            if f.get("extractor") == "concept_factor" and concept_factor is None:
                continue
            ob = extract(f, parsed.data, fctx, family.module)
            if ob.status == "DEFERRED":
                continue
            ev = evaluator_version(f)
            oid = ids.observation_id(row["trial_id"], f["feature_id"], ev)
            obs_rows.append(
                (
                    oid,
                    row["trial_id"],
                    f["feature_id"],
                    ev,
                    json.dumps(ob.value),
                    ob.status,
                    ob.confidence,
                )
            )
    for status, trial_id in status_updates:
        ctx.db.execute(
            "UPDATE trial SET status = ? WHERE trial_id = ?", (status, trial_id)
        )
    ctx.db.begin()
    ctx.db.conn.executemany(
        "INSERT OR REPLACE INTO observation(observation_id, trial_id, feature_id, evaluator_version, value_json, status, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
        obs_rows,
    )
    ctx.db.commit()


def _code_outcomes(ctx: Any, trial_id: str) -> dict[str, bool] | None:
    row = ctx.db.execute(
        "SELECT payload_json FROM event WHERE kind = 'code_outcomes' AND payload_json LIKE ? ORDER BY seq DESC LIMIT 1",
        (f'%"{trial_id}"%',),
    ).fetchone()
    if row is None:
        return None
    payload = json.loads(row[0])
    return {k: bool(v) for k, v in payload["outcomes"].items()}
