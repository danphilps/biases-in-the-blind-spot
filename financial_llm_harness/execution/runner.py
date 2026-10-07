"""The run lifecycle (spec §31A, L1–L13) and trial execution (§15, §18, §20.8, §29).

Every step after L6 is idempotent: IDs are deterministic (§3), so a resume
repeats the §4 frozen-input checks and restarts at L7.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import ids, yaml_io
from financial_llm_harness.core.canonical_json import canonical, sha256_canonical
from financial_llm_harness.core.code_ref import CodeRef
from financial_llm_harness.core.manifest import is_manifest, load_yaml_mapping
from financial_llm_harness.core.paths import runs_dir
from financial_llm_harness.errors import (
    BudgetExhausted,
    FrozenInputDrift,
    HarnessError,
    HarnessValidationError,
    PolicyRefusal,
)
from financial_llm_harness.evaluation.parse import (
    Contract,
    contract_from_dict,
    parse_response,
)
from financial_llm_harness.execution import budget as BU
from financial_llm_harness.execution import retries as RT
from financial_llm_harness.families import load_contract, load_family
from financial_llm_harness.generation.generate import iso_z, utc_now
from financial_llm_harness.generation.validate_bundle import (
    current_code_ref,
    freeze,
)
from financial_llm_harness.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    TestContext,
)
from financial_llm_harness.storage.archive import WriterLock
from financial_llm_harness.storage.artifacts import ArtifactStore
from financial_llm_harness.storage.db import connect, db_path

log = logging.getLogger("financial_llm_harness.execution.runner")

TERMINAL = {
    "SUCCESS",
    "REFUSAL",
    "EMPTY_RESPONSE",
    "TRUNCATED",
    "CONTENT_FILTERED",
    "INVALID_RESPONSE",
    "PARSE_FAILURE",
    "COMPILE_FAILURE",
    "TEST_FAILURE",
    "RUNTIME_FAILURE",
    "API_ERROR",
    "RATE_LIMIT",
    "TIMEOUT",
    "SANDBOX_FAILURE",
    "EXCLUDED_CONTEXT",
    "EXCLUDED_MEMORISED",
}
MODEL_LEVEL = {
    "SUCCESS",
    "REFUSAL",
    "EMPTY_RESPONSE",
    "TRUNCATED",
    "CONTENT_FILTERED",
    "INVALID_RESPONSE",
    "PARSE_FAILURE",
    "COMPILE_FAILURE",
    "TEST_FAILURE",
    "RUNTIME_FAILURE",
    "SANDBOX_FAILURE",
}
COMMIT_EVERY_TRIALS = 100
COMMIT_EVERY_S = 5.0
RUN_JSON_SCHEMA = 1


@dataclass
class RunResult:
    run_id: str | None
    run_dir: Path | None
    status: str | None
    exit_code: int
    plan_text: str = ""
    messages: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Run directories and identity (§31A, §32A.1)
# ---------------------------------------------------------------------------


def slug(key: str) -> str:
    return key.replace("_", "-").replace(".", "-")


def run_dir_name(started: datetime, experiment_key: str, run_id: str) -> str:
    return f"{started.strftime('%Y-%m-%d')}_{slug(experiment_key)}_{run_id[:8]}"


def resolve_run(run_id: str) -> Path:
    """A full run ID or a unique prefix of ≥ 8 hex characters (§31A)."""
    root = runs_dir()
    matches = []
    if root.is_dir():
        for d in sorted(root.iterdir()):
            rj = d / "run.json"
            if not rj.exists():
                continue
            rid = json.loads(rj.read_text(encoding="utf-8"))["run_id"]
            if rid == run_id or (
                len(run_id) >= 8
                and rid.replace("-", "").startswith(run_id.replace("-", ""))
            ):
                matches.append(d)
    if not matches:
        raise HarnessError(f"no run matches {run_id!r} under {root}")
    if len(matches) > 1:
        raise HarnessValidationError(
            f"run ID prefix {run_id!r} is ambiguous: {[m.name for m in matches]}"
        )
    return matches[0]


def output_policy(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """§30.1, evaluated once at L6."""
    cls = manifest["data_classification"]
    if cls == "synthetic":
        return {"publishable": True, "reason": "synthetic"}
    policy = load_provider_policy()
    allowed = set((policy.get("publish") or {}).get(cls) or [])
    if manifest["experiment_key"] in allowed:
        return {"publishable": True, "reason": "user_output_policy"}
    return {"publishable": False, "reason": "non_synthetic_default"}


def load_provider_policy() -> dict[str, Any]:
    import importlib.resources

    text = (
        importlib.resources.files("financial_llm_harness") / "provider_policy.yaml"
    ).read_text(encoding="utf-8")
    data = yaml_io.loads(text)
    if not isinstance(data, dict):
        raise HarnessValidationError("provider_policy.yaml must be a mapping")
    return data


def check_policy(manifest: Mapping[str, Any]) -> None:
    """Rule §2.14 (L5): exit 5 if the data classification is not allowed for a target."""
    policy = load_provider_policy()
    cls = manifest["data_classification"]
    allowed = set((policy.get("allow") or {}).get(cls) or [])
    denied = set(policy.get("deny_regions") or [])
    entries = list(manifest["sampling_spec"]["models"])
    for m in entries:
        if m["provider"] == "fake":
            continue
        if m["provider"] not in allowed:
            raise PolicyRefusal(
                f"{cls} data may not be sent to provider {m['provider']!r} (provider_policy.yaml, §2.14)"
            )
        if m.get("region") and f"{m['provider']}:{m['region']}" in denied:
            raise PolicyRefusal(
                f"region {m['provider']}:{m['region']} is denied by provider_policy.yaml"
            )


def in_git_work_tree(path: Path) -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
    except OSError:
        return False
    return out.stdout.strip() == "true"


# ---------------------------------------------------------------------------
# DB writer with commit batching (§29)
# ---------------------------------------------------------------------------


class RunDB:
    def __init__(self, run_dir: Path, clock: Callable[[], datetime]) -> None:
        self.conn = connect(db_path(run_dir))
        self.clock = clock
        self._in_tx = False
        self._terminal_since_commit = 0
        self._last_commit = time.monotonic()

    def now(self) -> str:
        return iso_z(self.clock())

    def begin(self) -> None:
        if not self._in_tx:
            self.conn.execute("BEGIN")
            self._in_tx = True

    def commit(self) -> None:
        if self._in_tx:
            self.conn.execute("COMMIT")
            self._in_tx = False
        self._terminal_since_commit = 0
        self._last_commit = time.monotonic()

    def terminal_written(self) -> None:
        self._terminal_since_commit += 1
        if (
            self._terminal_since_commit >= COMMIT_EVERY_TRIALS
            or time.monotonic() - self._last_commit >= COMMIT_EVERY_S
        ):
            self.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        self.begin()
        return self.conn.execute(sql, params)

    def event(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.execute(
            "INSERT INTO event(ts, kind, payload_json) VALUES (?, ?, ?)",
            (self.now(), kind, json.dumps(payload, sort_keys=True)),
        )

    def close(self) -> None:
        self.commit()
        self.conn.close()


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    manifest: dict[str, Any]
    db: RunDB
    store: ArtifactStore
    clock: Callable[[], datetime]
    code_ref: CodeRef
    contract: Contract
    renderings: dict[tuple[int, str, int], dict[str, Any]]
    cases: dict[int, dict[str, Any]]
    budget: BU.BudgetTracker
    pricing: dict[str, Any]
    adapters: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, ModelCapabilities] = field(default_factory=dict)
    submission_seq: int = 0
    sleep: Callable[[float], Any] = asyncio.sleep
    interrupt_after: int | None = None
    executed: int = 0

    @property
    def bundle_dir(self) -> Path:
        return self.run_dir / "manifest"

    @property
    def models(self) -> list[dict[str, Any]]:
        return list(self.manifest["sampling_spec"]["models"])


class InjectedInterruption(Exception):
    """Fault injection for the resume tests (§36)."""


def load_bundle_context(
    bundle_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[tuple[int, str, int], dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    manifest = B.load_manifest(bundle_dir)
    renderings = {
        (r["case_idx"], r["arm"], r["paraphrase_idx"]): r
        for r in B.read_jsonl(bundle_dir / B.RENDERINGS_FILE)
    }
    cases = {c["case_idx"]: c for c in B.read_jsonl(bundle_dir / B.CASES_FILE)}
    return manifest, renderings, cases


def manifest_contract(manifest: Mapping[str, Any]) -> Contract:
    cid = manifest["response_spec"]["contract_id"]
    return contract_from_dict(load_contract(manifest["task_family"], cid), cid)


def make_adapter(manifest: Mapping[str, Any], entry: Mapping[str, Any]) -> Any:
    """Build the adapter for a manifest ``models`` entry (§20.1)."""
    if entry["provider"] == "fake":
        from financial_llm_harness.models.fake import FakeAdapter, FakeTask

        family = load_family(manifest["task_family"])
        primary = next(f for f in manifest["feature_spec"] if f.get("primary") is True)
        from financial_llm_harness.statistics.power import feature_range

        concept_factor = None
        idef = family.interventions.get(manifest["intervention_id"])
        if idef is not None:
            concept_factor = idef.concept_factor
        task = FakeTask(
            contract_id=manifest["response_spec"]["contract_id"],
            reasoning_mode=manifest["response_spec"]["reasoning_mode"],
            feature_type=primary["type"],
            feature_range=feature_range(primary),
            concept_factor=concept_factor,
            fake_response=family.module.fake_response,
        )
        return FakeAdapter(entry, task)
    from financial_llm_harness.models.base import build_adapter

    return build_adapter(entry, ModelCapabilities.from_dict(entry["capabilities"]))


# ---------------------------------------------------------------------------
# L1–L5
# ---------------------------------------------------------------------------


def prepare_bundle(
    path: Path,
    panel: str | None,
    *,
    code_ref: CodeRef | None,
    clock: Callable[[], datetime],
    workdir: Path,
) -> Path:
    """L1 generate (spec input), L2 validate, L3 freeze; returns the bundle directory."""
    data = load_yaml_mapping(path)
    if is_manifest(data):
        bundle_dir = path.parent
        if B.read_lock(bundle_dir) is None:
            raise HarnessValidationError(
                "manifest input needs an existing manifest.lock (rule §2.1)"
            )
        freeze(bundle_dir, code_ref=code_ref, clock=clock)
        return bundle_dir
    from financial_llm_harness.generation.generate import generate

    bundle_dir = workdir / "bundle"
    res = generate(path, bundle_dir, panel, code_ref=code_ref, clock=clock)
    freeze(bundle_dir, code_ref=code_ref, regenerated=res.corpus, clock=clock)
    return bundle_dir


def plan_text(bundle_dir: Path) -> str:
    """L4: counts, tokens, cost range and power (§30)."""
    from financial_llm_harness.execution.plan import render_plan

    return render_plan(bundle_dir)


# ---------------------------------------------------------------------------
# L6 materialise
# ---------------------------------------------------------------------------


def materialise(
    bundle_dir: Path, run_id: str, clock: Callable[[], datetime], code_ref: CodeRef
) -> Path:
    manifest = B.load_manifest(bundle_dir)
    drift = B.frozen_input_mismatches(bundle_dir, code_ref, require_lock=True)
    if drift:
        raise FrozenInputDrift(drift)
    started = clock()
    run_dir = runs_dir() / run_dir_name(started, manifest["experiment_key"], run_id)
    if run_dir.exists():
        raise HarnessError(f"run directory {run_dir} already exists")
    run_dir.mkdir(parents=True)
    policy = output_policy(manifest)
    if manifest["data_classification"] != "synthetic":
        (run_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    shutil.copytree(bundle_dir, run_dir / "manifest")
    B.atomic_write_text(
        run_dir / "run.json",
        B.json_text(
            {
                "schema_version": RUN_JSON_SCHEMA,
                "run_id": run_id,
                "experiment_key": manifest["experiment_key"],
                "manifest_hash": manifest["manifest_hash"],
                "started_at": iso_z(started),
            }
        ),
    )
    store = ArtifactStore(run_dir / "artifacts")
    db = RunDB(run_dir, clock)
    try:
        _populate(db, store, run_dir, manifest, run_id, started, code_ref, policy)
        db.commit()
        _work_tree_warning(db, run_dir, manifest)
        db.commit()
    finally:
        db.close()
    return run_dir


def _work_tree_warning(db: RunDB, run_dir: Path, manifest: Mapping[str, Any]) -> None:
    if manifest["data_classification"] == "synthetic":
        return
    if in_git_work_tree(run_dir):
        log.warning(
            "run directory %s holds %s data and is inside a git work tree; it is git-ignored and not publishable "
            "unless provider_policy.yaml allows it (§0.9, §30.1)",
            run_dir,
            manifest["data_classification"],
        )
        db.event(
            "non_synthetic_in_git_work_tree",
            {
                "run_dir": str(run_dir),
                "classification": manifest["data_classification"],
            },
        )


def trial_seed(
    design_hash: str, case_idx: int, p: int, r: int, model_key: str, arm: str
) -> tuple[int, int]:
    """(requested seed, block seed): matched across a block's arms; A0 has its own stream (§18)."""
    block = ids.uint31(design_hash, case_idx, p, r, model_key)
    if arm == "A0":
        return ids.uint31("A0", design_hash, case_idx, p, r, model_key), block
    return block, block


def _populate(
    db: RunDB,
    store: ArtifactStore,
    run_dir: Path,
    manifest: dict[str, Any],
    run_id: str,
    started: datetime,
    code_ref: CodeRef,
    policy: Mapping[str, Any],
) -> None:
    bundle = run_dir / "manifest"
    mh = manifest["manifest_hash"]
    dh = manifest["design_hash"]
    now = iso_z(started)
    manifest_art = store.put((bundle / B.MANIFEST_FILE).read_bytes())
    db.execute(
        "INSERT INTO run(run_id, manifest_hash, status, started_at, completed_at, code_ref_json, resume_count, budget_override_json, output_policy_json) "
        "VALUES (?, ?, 'RUNNING', ?, NULL, ?, 0, NULL, ?)",
        (
            run_id,
            mh,
            now,
            json.dumps(code_ref.to_dict(), sort_keys=True),
            json.dumps(dict(policy), sort_keys=True),
        ),
    )
    db.execute(
        "INSERT INTO manifest(manifest_hash, design_hash, experiment_id, schema_version, manifest_artifact, frozen_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            mh,
            dh,
            manifest["experiment_id"],
            manifest["schema_version"],
            manifest_art,
            manifest["created_at"],
        ),
    )
    R = int(manifest["sampling_spec"]["R"])
    models = [m["model_key"] for m in manifest["sampling_spec"]["models"]]
    renderings = B.read_jsonl(bundle / B.RENDERINGS_FILE)
    by_case_arm: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in renderings:
        by_case_arm[(r["case_idx"], r["arm"])].append(r)
        frozen = {
            "system": r["system"],
            "user": r["user"],
            "response_contract": r["response_contract"],
        }
        data = canonical(frozen)
        key = store.put(data)
        if key != r["prompt_hash"]:
            raise HarnessValidationError(
                f"frozen message hash mismatch at case {r['case_idx']} arm {r['arm']}"
            )
    trial_rows = []
    for case in B.read_jsonl(bundle / B.CASES_FILE):
        ci = case["case_idx"]
        tc = ids.testcase_id(dh, ci)
        db.execute(
            "INSERT INTO testcase(testcase_id, design_hash, case_idx, cluster_id, cluster_ordinal, split, state_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                tc,
                dh,
                ci,
                ids.cluster_id(dh, case["cluster_ordinal"]),
                case["cluster_ordinal"],
                case["split"],
                sha256_canonical(case["sampled"]),
            ),
        )
        for arm in manifest["arms"]:
            if arm not in case["arms"]:
                continue
            av = ids.arm_variant_id(tc, arm)
            diff = sorted(
                k
                for k in case["arms"][arm]
                if case["arms"][arm][k] != case["arms"]["A"].get(k)
            )
            db.execute(
                "INSERT INTO variant(arm_variant_id, testcase_id, arm, state_hash, diff_json) VALUES (?, ?, ?, ?, ?)",
                (av, tc, arm, case["state_hashes"][arm], json.dumps(diff)),
            )
            for rr in sorted(by_case_arm[(ci, arm)], key=lambda x: x["paraphrase_idx"]):
                p = rr["paraphrase_idx"]
                rid = ids.rendering_id(av, p)
                n_est = BU.estimate_tokens(rr["system"]) + BU.estimate_tokens(
                    rr["user"]
                )
                db.execute(
                    "INSERT INTO rendering(rendering_id, arm_variant_id, paraphrase_idx, prompt_hash, n_tokens_est) VALUES (?, ?, ?, ?, ?)",
                    (rid, av, p, rr["prompt_hash"], n_est),
                )
                for r_ in range(R):
                    for mk in models:
                        seed, _ = trial_seed(dh, ci, p, r_, mk, arm)
                        trial_rows.append(
                            (
                                ids.trial_id(mh, ci, arm, p, r_, mk),
                                rid,
                                ids.block_id(mh, ci, p, r_, mk),
                                mk,
                                arm,
                                r_,
                                seed,
                                now,
                                ci,
                                p,
                            )
                        )
    db.conn.executemany(
        "INSERT INTO trial(trial_id, rendering_id, block_id, model_key, arm, repetition, seed, status, used_attempt_id, submission_seq, created_at, completed_at, case_idx, paraphrase_idx) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', NULL, NULL, ?, NULL, ?, ?)",
        trial_rows,
    )
    limits = {}
    from financial_llm_harness.models.base import provider_limits

    for m in manifest["sampling_spec"]["models"]:
        limits[m["provider"]] = provider_limits(m["provider"])
    db.event("provider_limits", {"limits": limits})


# ---------------------------------------------------------------------------
# L7 execute
# ---------------------------------------------------------------------------


def model_status(
    resp: ModelResponse, contract: Contract, mode: str, abstain: bool
) -> str:
    """The terminal model-level status of a complete response (§14, §28)."""
    if resp.refusal or resp.finish_reason == "refusal":
        return "REFUSAL"
    if resp.finish_reason == "content_filter":
        return "CONTENT_FILTERED"
    if resp.finish_reason == "length":
        return "TRUNCATED"
    if resp.text is None or resp.text.strip() == "":
        return "EMPTY_RESPONSE"
    return parse_response(contract, mode, abstain, resp.text).status


def _temperature(manifest: Mapping[str, Any], caps: ModelCapabilities) -> float | None:
    """The requested temperature; an adapter omits it where the frozen probe says rejected (E-21)."""
    del caps
    tp = manifest["sampling_spec"]["temperature_policy"]
    if tp.get("mode") != "fixed":
        return None
    return float(tp["temperature"])


def build_request(ctx: RunContext, trial: Mapping[str, Any]) -> ModelRequest:
    caps = ctx.capabilities[trial["model_key"]]
    r = ctx.renderings[(trial["case_idx"], trial["arm"], trial["paraphrase_idx"])]
    seed, block_seed = trial_seed(
        ctx.manifest["design_hash"],
        trial["case_idx"],
        trial["paraphrase_idx"],
        trial["repetition"],
        trial["model_key"],
        trial["arm"],
    )
    send_seed = seed if caps.seed_parameter in ("accepted", "ignored") else None
    return ModelRequest(
        system=r["system"],
        user=r["user"],
        response_contract=r["response_contract"],
        max_output_tokens=int(ctx.manifest["response_spec"]["max_output_tokens"]),
        temperature=_temperature(ctx.manifest, caps),
        seed=send_seed,
        enforcement=ctx.manifest["response_spec"]["enforcement_resolved"],
        trial_id=trial["trial_id"],
        test_context=TestContext(
            trial["arm"],
            trial["case_idx"],
            trial["paraphrase_idx"],
            trial["repetition"],
            seed,
            block_seed,
        ),
    )


def _price(ctx: RunContext, model_key: str) -> BU.Price | None:
    m = next(x for x in ctx.models if x["model_key"] == model_key)
    return BU.price_for(ctx.pricing, m["provider"], m["model"])


def _store_attempt_response(
    ctx: RunContext, trial: Mapping[str, Any], attempt_id: str, resp: ModelResponse
) -> BU.Spend:
    db, store = ctx.db, ctx.store
    req_art = store.put(resp.request_body)
    raw_art = store.put(resp.raw_payload) if resp.raw_payload is not None else None
    db.execute(
        "UPDATE attempt SET ended_at = ?, transport_status = ?, http_status = ?, provider_request_id = ?, request_artifact = ? WHERE attempt_id = ?",
        (
            db.now(),
            resp.transport_status,
            resp.http_status,
            resp.provider_request_id,
            req_art,
            attempt_id,
        ),
    )
    spend = BU.Spend(
        calls=1 if raw_art is not None or resp.transport_status == "complete" else 0
    )
    if resp.transport_status == "complete":
        text_art = store.put_text(resp.text) if resp.text is not None else None
        trace_art = (
            store.put_text(resp.reasoning_trace) if resp.reasoning_trace else None
        )
        usage = resp.usage or {}
        estimated = resp.usage_estimated
        it = usage.get("input")
        ot = usage.get("output")
        rt = usage.get("reasoning") or 0
        if it is None or ot is None:
            estimated = True
            it = (
                BU.estimate_tokens(
                    ctx.renderings[
                        (trial["case_idx"], trial["arm"], trial["paraphrase_idx"])
                    ]["user"]
                )
                if it is None
                else it
            )
            ot = BU.estimate_tokens(resp.text or "") if ot is None else ot
        price = _price(ctx, trial["model_key"])
        cost = price.cost(int(it), int(ot), int(rt)) if price is not None else 0.0
        db.execute(
            "INSERT OR REPLACE INTO response(attempt_id, trial_id, text_artifact, trace_artifact, raw_artifact, finish_reason, input_tokens, output_tokens, reasoning_tokens, usage_estimated, latency_ms, effective_params_json, serving_fingerprint_json, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                attempt_id,
                trial["trial_id"],
                text_art,
                trace_art,
                raw_art,
                resp.finish_reason,
                int(it),
                int(ot),
                int(rt),
                1 if estimated else 0,
                resp.latency_ms,
                json.dumps(resp.effective_params, sort_keys=True)
                if resp.effective_params is not None
                else None,
                json.dumps(resp.serving_fingerprint.to_dict(), sort_keys=True)
                if resp.serving_fingerprint
                else None,
                cost,
            ),
        )
        spend.input_tokens, spend.output_tokens, spend.cost_usd = (
            int(it),
            int(ot) + int(rt),
            cost,
        )
    elif raw_art is not None:
        db.execute(
            "INSERT OR REPLACE INTO response(attempt_id, trial_id, text_artifact, trace_artifact, raw_artifact, finish_reason, input_tokens, output_tokens, reasoning_tokens, usage_estimated, latency_ms, effective_params_json, serving_fingerprint_json, cost_usd) "
            "VALUES (?, ?, NULL, NULL, ?, NULL, NULL, NULL, NULL, NULL, ?, NULL, NULL, NULL)",
            (attempt_id, trial["trial_id"], raw_art, resp.latency_ms),
        )
    return spend


def _next_attempt_no(ctx: RunContext, trial_id: str) -> int:
    row = ctx.db.execute(
        "SELECT MAX(attempt_no) FROM attempt WHERE trial_id = ?", (trial_id,)
    ).fetchone()
    return 1 if row[0] is None else int(row[0]) + 1


def _finish_trial(
    ctx: RunContext, trial_id: str, status: str, used_attempt: str | None
) -> None:
    ctx.db.execute(
        "UPDATE trial SET status = ?, used_attempt_id = ?, completed_at = ? WHERE trial_id = ?",
        (status, used_attempt, ctx.db.now(), trial_id),
    )
    ctx.db.terminal_written()


async def run_trial(ctx: RunContext, trial: Mapping[str, Any]) -> BU.Spend:
    """Execute one trial to a terminal status, recording every attempt (§15, §29)."""
    caps = ctx.capabilities[trial["model_key"]]
    rendering = ctx.renderings[
        (trial["case_idx"], trial["arm"], trial["paraphrase_idx"])
    ]
    n_in = BU.estimate_tokens(rendering["system"]) + BU.estimate_tokens(
        rendering["user"]
    )
    if (
        n_in + int(ctx.manifest["response_spec"]["max_output_tokens"])
        > caps.context_tokens
    ):
        _finish_trial(ctx, trial["trial_id"], "EXCLUDED_CONTEXT", None)
        return BU.Spend()
    adapter = ctx.adapters[trial["model_key"]]
    request = build_request(ctx, trial)
    state = RT.RetryState()
    total = BU.Spend()
    ctx.submission_seq += 1
    ctx.db.execute(
        "UPDATE trial SET status = 'SUBMITTED', submission_seq = ? WHERE trial_id = ? AND status = 'PENDING'",
        (ctx.submission_seq, trial["trial_id"]),
    )
    contract = ctx.contract
    rs = ctx.manifest["response_spec"]
    last_kind = "api_error"
    while True:
        attempt_no = _next_attempt_no(ctx, trial["trial_id"])
        attempt_id = ids.attempt_id(trial["trial_id"], attempt_no)
        ctx.db.execute(
            "INSERT INTO attempt(attempt_id, trial_id, attempt_no, started_at, ended_at, transport_status, http_status, provider_request_id, batch_id, request_artifact) "
            "VALUES (?, ?, ?, ?, NULL, 'in_flight', NULL, NULL, NULL, NULL)",
            (attempt_id, trial["trial_id"], attempt_no, ctx.db.now()),
        )
        resp = await adapter.run(request)
        total.add(_store_attempt_response(ctx, trial, attempt_id, resp))
        kind = RT.classify(resp.transport_status, resp.http_status)
        if kind == "complete":
            _check_capability_drift(ctx, trial["model_key"], caps, resp)
            status = model_status(
                resp, contract, rs["reasoning_mode"], bool(rs["abstain_allowed"])
            )
            _finish_trial(ctx, trial["trial_id"], status, attempt_id)
            return total
        last_kind = "timeout" if resp.transport_status == "timeout" else kind
        action, wait = RT.next_action(state, kind, resp.retry_after_s)
        if action == "retry":
            if wait > 0:
                await ctx.sleep(wait)
            continue
        if action == "TRANSPORT_EXHAUSTED":
            final = "TIMEOUT" if last_kind == "timeout" else "API_ERROR"
        else:
            final = action
        _finish_trial(ctx, trial["trial_id"], final, None)
        return total


def _check_capability_drift(
    ctx: RunContext, model_key: str, caps: ModelCapabilities, resp: ModelResponse
) -> None:
    if caps.trace_visibility == "none" and resp.reasoning_trace:
        ctx.db.event(
            "capability_drift",
            {
                "model_key": model_key,
                "field": "trace_visibility",
                "expected": "none",
                "observed": "trace",
            },
        )


def pending_blocks(ctx: RunContext, model_key: str) -> list[list[dict[str, Any]]]:
    """Non-terminal trials grouped by block, in the shuffled block order with permuted arms (§18)."""
    rows = ctx.db.execute(
        "SELECT trial_id, model_key, arm, repetition, seed, status, case_idx, paraphrase_idx FROM trial WHERE model_key = ? ORDER BY case_idx, paraphrase_idx, repetition",
        (model_key,),
    ).fetchall()
    blocks: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        blocks[(r["case_idx"], r["paraphrase_idx"], r["repetition"])].append(dict(r))
    keys = sorted(blocks)
    perm = np.random.default_rng(
        ids.uint64("block_order", ctx.manifest["seeds"]["order"], model_key)
    ).permutation(len(keys))
    arm_rank = {a: i for i, a in enumerate(ctx.manifest["arms"])}
    out = []
    for i in perm:
        k = keys[int(i)]
        trials = sorted(blocks[k], key=lambda t: arm_rank[t["arm"]])
        ap = np.random.default_rng(
            ids.uint64(
                "arm_order", ctx.manifest["seeds"]["order"], k[0], k[1], k[2], model_key
            )
        ).permutation(len(trials))
        ordered = [trials[int(j)] for j in ap]
        todo = [t for t in ordered if t["status"] not in TERMINAL]
        if todo:
            out.append(todo)
    return out


def _block_bound(ctx: RunContext, block: list[dict[str, Any]]) -> BU.Spend:
    b = BU.Spend()
    max_out = int(ctx.manifest["response_spec"]["max_output_tokens"])
    for t in block:
        r = ctx.renderings[(t["case_idx"], t["arm"], t["paraphrase_idx"])]
        n_in = BU.estimate_tokens(r["system"]) + BU.estimate_tokens(r["user"])
        b.calls += 1
        b.input_tokens += n_in
        b.output_tokens += max_out
        price = _price(ctx, t["model_key"])
        if price is not None:
            b.cost_usd += price.cost(n_in, max_out)
    return b


async def execute(ctx: RunContext) -> None:
    """L7: every trial terminal, or a budget breach."""
    sems: dict[tuple[str, str | None], asyncio.Semaphore] = {}

    async def worker(model: Mapping[str, Any]) -> None:
        from financial_llm_harness.models.base import provider_limits

        key = (model["provider"], model.get("region"))
        if key not in sems:
            sems[key] = asyncio.Semaphore(
                int(provider_limits(model["provider"]).get("max_concurrency", 8))
            )
        sem = sems[key]
        tasks: list[asyncio.Task[BU.Spend]] = []
        for block in pending_blocks(ctx, model["model_key"]):
            bound = _block_bound(ctx, block)
            if not ctx.budget.reserve(bound):
                break
            block_tasks = []
            for trial in block:
                if (
                    ctx.interrupt_after is not None
                    and ctx.executed >= ctx.interrupt_after
                ):
                    raise InjectedInterruption()
                await sem.acquire()
                ctx.executed += 1

                async def one(t: Mapping[str, Any] = trial) -> BU.Spend:
                    try:
                        spend = await run_trial(ctx, t)
                        ctx.budget.record(spend)
                        return spend
                    finally:
                        sem.release()

                block_tasks.append(asyncio.ensure_future(one()))
            tasks.extend(block_tasks)

            async def settle(
                bt: list[asyncio.Task[BU.Spend]] = block_tasks, bd: BU.Spend = bound
            ) -> None:
                await asyncio.gather(*bt)
                ctx.budget.release(bd)

            tasks.append(asyncio.ensure_future(settle()))  # type: ignore[arg-type]
        if tasks:
            await asyncio.gather(*tasks)

    await asyncio.gather(*(worker(m) for m in ctx.models))
    ctx.db.commit()


def serving_drift_check(ctx: RunContext) -> None:
    """SERVING_DRIFT over the set of complete responses, order-independent (§20.8)."""
    for m in ctx.models:
        rows = ctx.db.execute(
            "SELECT r.serving_fingerprint_json, a.attempt_id FROM response r JOIN attempt a ON a.attempt_id = r.attempt_id "
            "JOIN trial t ON t.trial_id = r.trial_id WHERE t.model_key = ? AND a.transport_status = 'complete' ORDER BY a.attempt_id",
            (m["model_key"],),
        ).fetchall()
        revisions: dict[str, str] = {}
        pin = m.get("upstream_routing")
        for r in rows:
            fp = json.loads(r[0]) if r[0] else {}
            rev = fp.get("model_revision")
            if rev is not None and rev not in revisions:
                revisions[rev] = r[1]
            if pin:
                up = fp.get("upstream_provider")
                if up is None or up != pin["provider"]:
                    _drift_event(
                        ctx,
                        m["model_key"],
                        "upstream_provider",
                        pin["provider"],
                        up,
                        r[1],
                    )
                q = pin.get("quantization")
                if q is not None and fp.get("quantization") != q:
                    _drift_event(
                        ctx,
                        m["model_key"],
                        "quantization",
                        q,
                        fp.get("quantization"),
                        r[1],
                    )
        if len(revisions) > 1:
            revs = sorted(revisions)
            _drift_event(
                ctx,
                m["model_key"],
                "model_revision",
                revs[0],
                revs[1],
                revisions[revs[1]],
            )
    ctx.db.commit()


def _drift_event(
    ctx: RunContext,
    model_key: str,
    field_: str,
    expected: Any,
    observed: Any,
    attempt_id: str,
) -> None:
    payload = {
        "model_key": model_key,
        "field": field_,
        "expected": expected,
        "observed": observed,
        "attempt_id": attempt_id,
    }
    existing = ctx.db.execute(
        "SELECT 1 FROM event WHERE kind = 'serving_drift' AND payload_json = ?",
        (json.dumps(payload, sort_keys=True),),
    ).fetchone()
    if existing is None:
        ctx.db.event("serving_drift", payload)


# ---------------------------------------------------------------------------
# Lifecycle drivers
# ---------------------------------------------------------------------------


def open_context(
    run_dir: Path,
    clock: Callable[[], datetime],
    code_ref: CodeRef,
    override: Mapping[str, Any] | None = None,
    data_dir: Path | None = None,
) -> RunContext:
    """Open a run; ``data_dir`` holds ``db/`` and ``artifacts/`` when they are not in ``run_dir``."""
    manifest, renderings, cases = load_bundle_context(run_dir / "manifest")
    db = RunDB(data_dir or run_dir, clock)
    row = db.execute("SELECT run_id, budget_override_json FROM run").fetchone()
    db.commit()
    stored_override = (
        json.loads(row["budget_override_json"]) if row["budget_override_json"] else None
    )
    budget = BU.BudgetTracker.from_manifest(
        manifest["budget"], override or stored_override
    )
    spent = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) + COALESCE(SUM(reasoning_tokens), 0), COALESCE(SUM(cost_usd), 0) FROM response"
    ).fetchone()
    ev = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), COALESCE(SUM(cost_usd), 0) FROM eval_call"
    ).fetchone()
    db.commit()
    budget.spent = BU.Spend(
        int(spent[0]) + int(ev[0]),
        int(spent[1]) + int(ev[1]),
        int(spent[2]) + int(ev[2]),
        float(spent[3]) + float(ev[3]),
    )
    ctx = RunContext(
        run_id=row["run_id"],
        run_dir=run_dir,
        manifest=manifest,
        db=db,
        store=ArtifactStore((data_dir or run_dir) / "artifacts"),
        clock=clock,
        code_ref=code_ref,
        contract=manifest_contract(manifest),
        renderings=renderings,
        cases=cases,
        budget=budget,
        pricing=BU.load_pricing(),
    )
    for m in ctx.models:
        ctx.capabilities[m["model_key"]] = ModelCapabilities.from_dict(
            m["capabilities"]
        )
        ctx.adapters[m["model_key"]] = make_adapter(manifest, m)
    return ctx


def set_run_status(ctx: RunContext, status: str, completed: bool) -> None:
    ctx.db.execute(
        "UPDATE run SET status = ?, completed_at = ? WHERE run_id = ?",
        (status, ctx.db.now() if completed else None, ctx.run_id),
    )
    ctx.db.commit()


def post_execution(ctx: RunContext, *, semantic: bool = True) -> str:
    """L8–L13; returns the run status."""
    from financial_llm_harness.evaluation.deterministic import evaluate_deterministic
    from financial_llm_harness.reporting.render import render_run
    from financial_llm_harness.statistics.analysis import (
        analyse_attribution,
        analyse_sensitivity,
        model_incomplete_set,
    )

    evaluate_deterministic(ctx)
    analyse_sensitivity(ctx)
    paused = False
    if semantic and any(
        f.get("extractor") in ("semantic_verbalisation", "semantic_inclusion")
        for f in ctx.manifest["feature_spec"]
    ):
        from financial_llm_harness.evaluation.semantic import evaluate_semantic

        try:
            evaluate_semantic(ctx)
        except BudgetExhausted:
            paused = True
        analyse_sensitivity(ctx)
    analyse_attribution(ctx, budget_paused=paused)
    incomplete = model_incomplete_set(ctx)
    not_terminal = ctx.db.execute(
        "SELECT COUNT(*) FROM trial WHERE status NOT IN ({})".format(
            ",".join("?" * len(TERMINAL))
        ),
        tuple(TERMINAL),
    ).fetchone()[0]
    ctx.db.commit()
    if paused or ctx.budget.breached and not_terminal:
        status = "PAUSED_BUDGET"
    elif incomplete or not_terminal:
        status = "INCOMPLETE"
    else:
        status = "COMPLETE"
    set_run_status(ctx, status, completed=status in ("COMPLETE", "INCOMPLETE"))
    render_run(ctx.run_dir)
    from financial_llm_harness.reporting.index import write_index

    write_index()
    return status


EXIT_FOR_STATUS = {"COMPLETE": 0, "INCOMPLETE": 4, "PAUSED_BUDGET": 3, "FAILED": 1}


def _execute_and_finish(ctx: RunContext, interrupt_after: int | None = None) -> str:
    ctx.interrupt_after = interrupt_after
    asyncio.run(execute(ctx))
    serving_drift_check(ctx)
    if ctx.budget.breached:
        remaining = ctx.db.execute(
            "SELECT COUNT(*) FROM trial WHERE status IN ('PENDING', 'SUBMITTED')"
        ).fetchone()[0]
        ctx.db.commit()
        if remaining:
            from financial_llm_harness.evaluation.deterministic import (
                evaluate_deterministic,
            )
            from financial_llm_harness.reporting.render import render_run
            from financial_llm_harness.statistics.analysis import analyse_sensitivity

            evaluate_deterministic(ctx)
            analyse_sensitivity(ctx)
            set_run_status(ctx, "PAUSED_BUDGET", completed=False)
            render_run(ctx.run_dir)
            return "PAUSED_BUDGET"
    return post_execution(ctx)


def _fail(ctx: RunContext | None, run_dir: Path | None) -> None:
    if run_dir is None:
        return
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    with open(logs / "traceback.txt", "a", encoding="utf-8") as fh:
        fh.write(traceback.format_exc())
    if ctx is not None:
        try:
            ctx.db.commit()
            set_run_status(ctx, "FAILED", completed=False)
        except Exception:  # noqa: BLE001 - best effort while failing
            pass


def run_experiment(
    path: Path,
    panel: str | None,
    *,
    yes: bool,
    run_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
    code_ref: CodeRef | None = None,
    dry_run: bool = False,
    interrupt_after: int | None = None,
) -> RunResult:
    """Library entry point of ``run`` (§31A, E-23)."""
    clk = clock or utc_now
    rid = run_id or str(uuid.uuid4())
    with tempfile.TemporaryDirectory(prefix="fh-bundle-") as tmp:
        bundle_dir = prepare_bundle(
            path, panel, code_ref=code_ref, clock=clk, workdir=Path(tmp)
        )
        manifest = B.load_manifest(bundle_dir)
        ref = current_code_ref(manifest, code_ref)
        text = plan_text(bundle_dir)
        check_policy(manifest)
        if not yes or dry_run:
            return RunResult(None, None, None, 0, text)
        run_dir = materialise(bundle_dir, rid, clk, ref)
    ctx: RunContext | None = None
    lock = WriterLock(run_dir)
    lock.acquire()
    try:
        ctx = open_context(run_dir, clk, ref)
        status = _execute_and_finish(ctx, interrupt_after)
        return RunResult(rid, run_dir, status, EXIT_FOR_STATUS[status], text)
    except InjectedInterruption:
        if ctx is not None:
            ctx.db.commit()
        raise
    except HarnessError:
        _fail(ctx, run_dir)
        raise
    except Exception:
        _fail(ctx, run_dir)
        raise
    finally:
        if ctx is not None:
            ctx.db.close()
        lock.release()


def resume_experiment(
    run_id: str,
    *,
    yes: bool,
    clock: Callable[[], datetime] | None = None,
    code_ref: CodeRef | None = None,
    max_cost_usd: float | None = None,
    max_calls: int | None = None,
    interrupt_after: int | None = None,
) -> RunResult:
    """``run --resume`` (§29, §31A): frozen-input checks, then restart at L7."""
    clk = clock or utc_now
    run_dir = resolve_run(run_id)
    bundle_dir = run_dir / "manifest"
    manifest = B.load_manifest(bundle_dir)
    ref = current_code_ref(manifest, code_ref)
    drift = B.frozen_input_mismatches(bundle_dir, ref, require_lock=True)
    lock = WriterLock(run_dir)
    lock.acquire()
    ctx: RunContext | None = None
    try:
        if drift:
            db = RunDB(run_dir, clk)
            db.event("frozen_input_drift", {"mismatches": drift})
            db.close()
            raise FrozenInputDrift(drift)
        if not yes:
            return RunResult(None, run_dir, None, 0, "resume needs --yes")
        override = None
        if max_cost_usd is not None or max_calls is not None:
            override = {"max_cost_usd": max_cost_usd, "max_calls": max_calls}
        ctx = open_context(run_dir, clk, ref, override)
        ctx.db.execute(
            "UPDATE run SET resume_count = resume_count + 1, status = 'RUNNING' WHERE run_id = ?",
            (ctx.run_id,),
        )
        if override is not None:
            ctx.db.execute(
                "UPDATE run SET budget_override_json = ? WHERE run_id = ?",
                (json.dumps(override, sort_keys=True), ctx.run_id),
            )
            ctx.db.event("budget_override", override)
        _work_tree_warning(ctx.db, run_dir, manifest)
        ctx.db.execute(
            "UPDATE attempt SET transport_status = 'abandoned', ended_at = ? WHERE transport_status = 'in_flight' AND batch_id IS NULL",
            (ctx.db.now(),),
        )
        ctx.db.commit()
        status = _execute_and_finish(ctx, interrupt_after)
        return RunResult(ctx.run_id, run_dir, status, EXIT_FOR_STATUS[status])
    except InjectedInterruption:
        if ctx is not None:
            ctx.db.commit()
        raise
    except FrozenInputDrift:
        raise
    except Exception:
        _fail(ctx, run_dir)
        raise
    finally:
        if ctx is not None:
            ctx.db.close()
        lock.release()
