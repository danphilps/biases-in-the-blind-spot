"""Semantic evaluator (spec §16, §28, §29, E-3, D-2, D-3).

The evaluator is the panel's ``evaluator`` entry. Its prompt is the legacy
``DEFAULT_VERBALIZATION_PROMPT`` and its parser the legacy
``default_verbalization_parser``, both imported lazily (§0.7). Each request
attempt is one ``eval_call`` row, never a ``trial`` row, and counts towards the
budget (§30). One call at temperature 0 when the evaluator's frozen probe
accepts ``temperature``; otherwise three calls seeded ``uint31(observation_id |
call_no)``, and the observation takes the value that at least two return
(UNKNOWN included). An UNKNOWN observation has status EVALUATOR_UNKNOWN and is
never coerced.

Re-evaluation is byte-identical: an observation whose calls are all stored is
recomputed from the stored outputs; only ``re_judge`` calls the evaluator again.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from financial_llm_harness.core import ids
from financial_llm_harness.core.canonical_json import sha256_canonical
from financial_llm_harness.core.paths import repo_root
from financial_llm_harness.errors import BudgetExhausted, HarnessValidationError
from financial_llm_harness.execution import retries as RT
from financial_llm_harness.models.base import ModelCapabilities, ModelRequest

LEGACY_MODULE = "biases_in_the_blind_spot.concept_pipeline.verbalization_detector"
PROMPT_OBJECT = f"{LEGACY_MODULE}.DEFAULT_VERBALIZATION_PROMPT"
PARSER_OBJECT = f"{LEGACY_MODULE}.default_verbalization_parser"
PARSER_VERSION = 1
MAX_OUTPUT_TOKENS = 512
KAPPA_MIN = 0.70
SEMANTIC_EXTRACTORS = ("semantic_verbalisation", "semantic_inclusion")
VALID_STATUSES = ("SUCCESS",)


def legacy_prompt() -> str:
    module = importlib.import_module(LEGACY_MODULE)
    return str(module.DEFAULT_VERBALIZATION_PROMPT)


def legacy_parser() -> Callable[[str], Any]:
    module = importlib.import_module(LEGACY_MODULE)
    return module.default_verbalization_parser


def prompt_hash() -> str:
    return hashlib.sha256(legacy_prompt().encode("utf-8")).hexdigest()


def calls_for(caps: ModelCapabilities) -> list[int]:
    """One call at temperature 0 if accepted, otherwise three (§16)."""
    return [0] if caps.param_status("temperature") != "rejected" else [0, 1, 2]


def evaluator_version(model: Mapping[str, Any], n_calls: int) -> str:
    """(model, revision, prompt hash, parser version, decoding params, number of calls) (§16)."""
    return sha256_canonical(
        {
            "model_key": model["model_key"],
            "provider": model["provider"],
            "model": model["model"],
            "revision": model.get("revision"),
            "prompt_hash": prompt_hash(),
            "parser_version": PARSER_VERSION,
            "decoding": {
                "temperature": 0.0 if n_calls == 1 else None,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "seed": "uint31(observation_id|call_no)" if n_calls > 1 else None,
            },
            "n_calls": n_calls,
        }
    )


# ---------------------------------------------------------------------------
# Calibration records (§16)
# ---------------------------------------------------------------------------


def calibration_dir() -> Path:
    return repo_root() / "experiments" / "financial" / "calibration"


def calibration_path(feature_id: str, version: str) -> Path:
    return calibration_dir() / feature_id / f"{version[:12]}.json"


def find_calibration(
    feature_id: str, version: str, provider: str
) -> dict[str, Any] | None:
    """``{path, sha256, kappa}`` of the feature's record for this evaluator version, or None.

    A record made from the synthetic fixture is valid only for ``fake`` evaluators.
    """
    path = calibration_path(feature_id, version)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("evaluator_version") != version or data.get("feature_id") != feature_id:
        return None
    if data.get("fake_only") and provider != "fake":
        return None
    return {
        "path": path.relative_to(repo_root()).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kappa": data["metrics"]["kappa"],
    }


# ---------------------------------------------------------------------------
# Prompt and verdicts
# ---------------------------------------------------------------------------


def feature_text(feature: Mapping[str, Any], key: str, intervention_id: str) -> str:
    """A feature's ``guide`` or ``category``: a string, or a mapping by intervention ID."""
    raw = feature.get(key)
    if isinstance(raw, Mapping):
        raw = raw.get(intervention_id)
    if not isinstance(raw, str) or not raw.strip():
        raise HarnessValidationError(
            f"semantic feature {feature.get('feature_id')}: no {key} for {intervention_id} (§16)"
        )
    return raw


def prompt_fields(
    manifest: Mapping[str, Any], feature: Mapping[str, Any]
) -> tuple[str, str]:
    """The feature's ``guide`` and ``category`` for the manifest's intervention (exit 2 if absent)."""
    from financial_llm_harness.families import load_family

    family = load_family(manifest["task_family"])
    fam = family.features.get(feature["feature_id"])
    merged = {**(dict(fam.data) if fam is not None else {}), **feature}
    return (
        feature_text(merged, "guide", manifest["intervention_id"]),
        feature_text(merged, "category", manifest["intervention_id"]),
    )


def fill_prompt(
    formatted_input: str, concept_title: str, guide: str, category: str, response: str
) -> str:
    return legacy_prompt().format(
        formatted_input=formatted_input,
        concept_title=concept_title,
        verbalization_check_guide=guide,
        concept_category=category,
        response=response,
    )


def verdict(text: str | None) -> str:
    """TRUE | FALSE | UNKNOWN from one call's output; the parser's None is UNKNOWN."""
    if text is None:
        return "UNKNOWN"
    parsed = legacy_parser()(text)
    if parsed is None:
        return "UNKNOWN"
    return "TRUE" if bool(parsed.verbalized) else "FALSE"


def majority(values: Sequence[str]) -> str:
    """The value at least two calls return; one call decides alone (§16)."""
    if len(values) == 1:
        return values[0]
    for v in ("TRUE", "FALSE", "UNKNOWN"):
        if sum(1 for x in values if x == v) >= 2:
            return v
    return "UNKNOWN"


def call_seed(observation_id: str, call_no: int) -> int:
    return ids.uint31(observation_id, call_no)


# ---------------------------------------------------------------------------
# The evaluator adapter
# ---------------------------------------------------------------------------


def evaluator_entry(manifest: Mapping[str, Any]) -> dict[str, Any]:
    from financial_llm_harness.models.panels import load_panel

    panel = load_panel(manifest["sampling_spec"]["panel_ref"]["name"])
    ev = panel.evaluator
    if ev is None:
        raise HarnessValidationError("the panel has no evaluator entry (§16, §20.9)")
    return dict(ev)


def evaluator_capabilities(
    entry: Mapping[str, Any], bundle_dir: Path | None
) -> ModelCapabilities:
    from financial_llm_harness.models.fake import fake_capabilities

    if entry["provider"] == "fake":
        return fake_capabilities(entry)
    if bundle_dir is None:
        raise HarnessValidationError("a live evaluator needs its frozen probe")
    probe = bundle_dir / "probes" / f"{entry['model_key']}.json"
    if not probe.exists():
        raise HarnessValidationError(
            f"no frozen probe for evaluator {entry['model_key']} (§20.4)"
        )
    return ModelCapabilities.from_dict(
        json.loads(probe.read_text(encoding="utf-8"))["capabilities"]
    )


def make_evaluator(entry: Mapping[str, Any], caps: ModelCapabilities) -> Any:
    if entry["provider"] == "fake":
        from financial_llm_harness.models.fake import FakeEvaluator

        return FakeEvaluator(entry)
    from financial_llm_harness.models.base import build_adapter

    return build_adapter(entry, caps)


def evaluator_request(
    prompt: str,
    observation_id: str,
    call_no: int,
    caps: ModelCapabilities,
    n_calls: int,
    seed_key: str | None = None,
) -> ModelRequest:
    """One evaluator call. A live evaluator's requested seed is ``uint31(observation_id |
    call_no)`` (§16). The fake evaluator draws from ``seed_key``: the observation key
    with ``design_hash`` in place of ``manifest_hash``, so, like trial seeds (§3, §18),
    fake verdicts do not change with code identity or panel (§36 Golden and identity).
    """
    seeded = n_calls > 1 and caps.seed_parameter in ("accepted", "ignored")
    return ModelRequest(
        system="",
        user=prompt,
        response_contract=None,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0 if n_calls == 1 else None,
        seed=call_seed(observation_id, call_no) if seeded else None,
        enforcement="prompt_only",
        trial_id=f"eval/{observation_id}/{call_no}",
        extra={
            "observation_id": observation_id,
            "call_no": call_no,
            "seed_key": seed_key or observation_id,
        },
    )


async def run_call(
    adapter: Any,
    request: ModelRequest,
    on_attempt: Callable[[int, Any], None],
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> str | None:
    """One call through the §15 retry policy; the text of its ``complete`` attempt, or None."""
    state = RT.RetryState()
    attempt_no = 0
    while True:
        attempt_no += 1
        resp = await adapter.run(request)
        on_attempt(attempt_no, resp)
        kind = RT.classify(resp.transport_status, resp.http_status)
        if kind == "complete":
            return resp.text
        action, wait = RT.next_action(state, kind, getattr(resp, "retry_after_s", None))
        if action != "retry":
            return None
        if wait > 0:
            await sleep(wait)


# ---------------------------------------------------------------------------
# L9: semantic evaluation of a run
# ---------------------------------------------------------------------------


def _channel_text(
    feature: Mapping[str, Any],
    parsed: Mapping[str, Any],
    trace: str | None,
    contract: Any,
    mode: str,
) -> str | None:
    if feature.get("extractor") == "semantic_inclusion":
        fields = feature.get("fields")
        if fields:
            return json.dumps(
                {name: parsed.get(name) for name in fields}, ensure_ascii=False
            )
        field = feature.get("field")
        if field:
            value = parsed.get(field)
            return (
                json.dumps(value, ensure_ascii=False)
                if not isinstance(value, str)
                else value
            )
        return json.dumps(dict(parsed), ensure_ascii=False, sort_keys=True)
    channel = feature.get("channel")
    if channel == "trace":
        return trace
    key = contract.free_text_key(mode)
    if key is None:
        return None
    value = parsed.get(key)
    return value if isinstance(value, str) and value.strip() else None


@dataclass(frozen=True)
class Job:
    observation_id: str
    trial_id: str
    feature_id: str
    version: str
    prompt: str
    seed_key: (
        str  # design-level key that seeds a fake evaluator (see evaluator_request)
    )


def evaluate_semantic(
    ctx: Any, *, re_judge: bool = False, concurrency: int = 8
) -> None:
    """Write one observation per (trial with a parsed response, semantic feature) (§16).

    A trial the feature's deterministic rule already decided keeps that
    observation. An attribution channel evaluates only the §24.3 evaluation set
    of each model whose run-level label is attribution-eligible. Raises ``BudgetExhausted`` when the next call would breach a
    cap; finished observations are kept and resume continues with the rest.
    """
    from financial_llm_harness.evaluation.parse import parse_response
    from financial_llm_harness.families import load_family

    manifest = ctx.manifest
    features = [
        f for f in manifest["feature_spec"] if f.get("extractor") in SEMANTIC_EXTRACTORS
    ]
    if not features:
        return
    evaluators = {e["feature_id"]: e for e in manifest.get("evaluators") or []}
    entry = evaluator_entry(manifest)
    caps = evaluator_capabilities(entry, ctx.bundle_dir)
    calls = calls_for(caps)
    family = load_family(manifest["task_family"])
    rs = manifest["response_spec"]
    mode = rs["reasoning_mode"]
    abstain = bool(rs["abstain_allowed"])
    rows = ctx.db.execute(
        "SELECT t.trial_id, t.case_idx, t.arm, t.paraphrase_idx, t.repetition, t.model_key, t.status, r.text_artifact, r.trace_artifact "
        "FROM trial t JOIN response r ON r.attempt_id = t.used_attempt_id "
        "WHERE t.used_attempt_id IS NOT NULL ORDER BY t.trial_id"
    ).fetchall()
    ctx.db.commit()
    from financial_llm_harness.statistics.analysis import attribution_evaluation_set

    attribution_set: set[str] | None = None
    jobs: list[Job] = []
    for f in features:
        attribution = f.get("extractor") == "semantic_verbalisation"
        if attribution and attribution_set is None:
            attribution_set = attribution_evaluation_set(ctx)
        ev = evaluators.get(f["feature_id"])
        if ev is None:
            raise HarnessValidationError(
                f"semantic feature {f['feature_id']}: no evaluators entry (§16)"
            )
        version = str(ev["evaluator_version"])
        fam = family.features.get(f["feature_id"])
        merged = {**(dict(fam.data) if fam is not None else {}), **f}
        guide, category = prompt_fields(manifest, f)
        decided = {
            r[0]
            for r in ctx.db.execute(
                "SELECT trial_id FROM observation WHERE feature_id = ? AND evaluator_version != ?",
                (f["feature_id"], version),
            ).fetchall()
        }
        ctx.db.commit()
        for row in rows:
            if row["status"] not in VALID_STATUSES or row["trial_id"] in decided:
                continue
            if attribution and row["trial_id"] not in (attribution_set or set()):
                continue  # only the §24.3 evaluation set is evaluated
            text = (
                ctx.store.get_text(row["text_artifact"])
                if row["text_artifact"]
                else None
            )
            parsed = parse_response(ctx.contract, mode, abstain, text)
            if parsed.status != "SUCCESS" or parsed.data is None:
                continue
            trace = (
                ctx.store.get_text(row["trace_artifact"])
                if row["trace_artifact"]
                else None
            )
            response = _channel_text(merged, parsed.data, trace, ctx.contract, mode)
            if response is None:
                continue
            rendering = ctx.renderings[
                (row["case_idx"], row["arm"], row["paraphrase_idx"])
            ]
            prompt = fill_prompt(
                rendering["user"], str(manifest["concept"]), guide, category, response
            )
            obs_id = ids.observation_id(row["trial_id"], f["feature_id"], version)
            seed_key = "/".join(
                str(x)
                for x in (
                    manifest["design_hash"],
                    row["case_idx"],
                    row["arm"],
                    row["paraphrase_idx"],
                    row["repetition"],
                    row["model_key"],
                    f["feature_id"],
                    version,
                )
            )
            jobs.append(
                Job(obs_id, row["trial_id"], f["feature_id"], version, prompt, seed_key)
            )
    live: list[Job] = []
    for job in jobs:
        stored = None if re_judge else _stored_verdicts(ctx, job.observation_id, calls)
        if stored is None:
            live.append(job)
        else:
            _write_observation(ctx, job, stored)
    if live:
        adapter = make_evaluator(entry, caps)
        breached = asyncio.run(
            _run_jobs(ctx, adapter, entry, caps, live, calls, re_judge, concurrency)
        )
        if breached:
            raise BudgetExhausted("an evaluator call would breach the budget (§30)")


def _write_observation(ctx: Any, job: Job, verdicts: Sequence[str]) -> None:
    value = majority(verdicts)
    ctx.db.execute(
        "INSERT OR REPLACE INTO observation(observation_id, trial_id, feature_id, evaluator_version, value_json, status, confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            job.observation_id,
            job.trial_id,
            job.feature_id,
            job.version,
            json.dumps(value),
            "EVALUATOR_UNKNOWN" if value == "UNKNOWN" else "OK",
            sum(1 for v in verdicts if v == value) / len(verdicts),
        ),
    )
    ctx.db.commit()


def _stored_verdicts(ctx: Any, obs_id: str, calls: Sequence[int]) -> list[str] | None:
    """Verdicts from the stored calls, or None when a call has no stored attempt.

    A call whose stored attempts include no ``complete`` one yields UNKNOWN (§16).
    """
    out: list[str] = []
    for call_no in calls:
        rows = ctx.db.execute(
            "SELECT status, raw_artifact FROM eval_call WHERE observation_id = ? AND call_no = ? ORDER BY attempt_no",
            (obs_id, call_no),
        ).fetchall()
        ctx.db.commit()
        if not rows:
            return None
        complete = [r for r in rows if r["status"] == "complete"]
        if complete:
            out.append(
                verdict(json.loads(ctx.store.get(complete[0]["raw_artifact"]))["text"])
            )
        else:
            out.append("UNKNOWN")
    return out


async def _run_jobs(
    ctx: Any,
    adapter: Any,
    entry: Mapping[str, Any],
    caps: ModelCapabilities,
    jobs: Sequence[Job],
    calls: Sequence[int],
    re_judge: bool,
    concurrency: int,
) -> bool:
    """Run the jobs with bounded concurrency; True when the budget stopped them."""
    from financial_llm_harness.execution import budget as BU

    price = BU.price_for(ctx.pricing, entry["provider"], entry["model"])
    gate = asyncio.Semaphore(max(1, concurrency))
    stopped = False

    async def one(job: Job) -> None:
        nonlocal stopped
        async with gate:
            if stopped:
                return
            verdicts: list[str] = []
            for call_no in calls:
                n_in = BU.estimate_tokens(job.prompt)
                bound = BU.Spend(
                    1,
                    n_in,
                    MAX_OUTPUT_TOKENS,
                    price.cost(n_in, MAX_OUTPUT_TOKENS) if price else 0.0,
                )
                if not ctx.budget.reserve(bound):
                    stopped = True
                    return
                base = _max_attempt(ctx, job.observation_id, call_no) if re_judge else 0
                spent = BU.Spend()
                request = evaluator_request(
                    job.prompt,
                    job.observation_id,
                    call_no,
                    caps,
                    len(calls),
                    job.seed_key,
                )

                def record(
                    attempt_no: int,
                    resp: Any,
                    call_no: int = call_no,
                    base: int = base,
                    spent: Any = spent,
                ) -> None:
                    spent.add(
                        _record_attempt(
                            ctx,
                            entry,
                            price,
                            job.observation_id,
                            call_no,
                            base + attempt_no,
                            resp,
                        )
                    )

                text = await run_call(adapter, request, record, ctx.sleep)
                ctx.budget.release(bound)
                ctx.budget.record(spent)
                verdicts.append(verdict(text))
            _write_observation(ctx, job, verdicts)

    await asyncio.gather(*(one(j) for j in jobs))
    return stopped


def _max_attempt(ctx: Any, obs_id: str, call_no: int) -> int:
    row = ctx.db.execute(
        "SELECT MAX(attempt_no) FROM eval_call WHERE observation_id = ? AND call_no = ?",
        (obs_id, call_no),
    ).fetchone()
    ctx.db.commit()
    return int(row[0] or 0)


def _record_attempt(
    ctx: Any,
    entry: Mapping[str, Any],
    price: Any,
    obs_id: str,
    call_no: int,
    attempt_no: int,
    resp: Any,
) -> Any:
    """One ``eval_call`` row per request attempt (§29); returns its spend."""
    from financial_llm_harness.execution import budget as BU

    payload = ctx.store.put(resp.raw_payload) if resp.raw_payload is not None else None
    wrapper = json.dumps(
        {
            "text": resp.text,
            "transport_status": resp.transport_status,
            "payload_artifact": payload,
        },
        sort_keys=True,
    ).encode()
    usage = resp.usage or {}
    complete = resp.transport_status == "complete"
    it = int(usage.get("input") or 0)
    ot = int(usage.get("output") or 0) + int(usage.get("reasoning") or 0)
    cost = price.cost(it, ot) if (price is not None and complete) else 0.0
    now = ctx.db.now()
    ctx.db.execute(
        "INSERT OR REPLACE INTO eval_call(eval_call_id, observation_id, call_no, attempt_no, model_key, prompt_hash, request_artifact, raw_artifact, status, http_status, provider_request_id, input_tokens, output_tokens, cost_usd, started_at, ended_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            ids.eval_call_id(obs_id, call_no, attempt_no),
            obs_id,
            call_no,
            attempt_no,
            entry["model_key"],
            prompt_hash(),
            ctx.store.put(resp.request_body),
            ctx.store.put(wrapper) if complete or payload is not None else None,
            resp.transport_status,
            resp.http_status,
            resp.provider_request_id,
            it,
            ot,
            cost,
            now,
            now,
        ),
    )
    ctx.db.commit()
    return BU.Spend(1 if complete or payload is not None else 0, it, ot, cost)
