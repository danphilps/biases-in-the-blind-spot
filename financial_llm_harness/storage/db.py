"""Per-run SQLite database (spec §29).

One database per run at ``<run_dir>/db/run.sqlite``, WAL mode, single writer.
``attempt`` and ``eval_call`` are append-only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS run(
    run_id TEXT PRIMARY KEY,
    manifest_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    code_ref_json TEXT,
    resume_count INTEGER NOT NULL DEFAULT 0,
    budget_override_json TEXT,
    output_policy_json TEXT
);
CREATE TABLE IF NOT EXISTS manifest(
    manifest_hash TEXT PRIMARY KEY,
    design_hash TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    manifest_artifact TEXT NOT NULL,
    frozen_at TEXT
);
CREATE TABLE IF NOT EXISTS testcase(
    testcase_id TEXT PRIMARY KEY,
    design_hash TEXT NOT NULL,
    case_idx INTEGER NOT NULL,
    cluster_id TEXT NOT NULL,
    cluster_ordinal INTEGER NOT NULL,
    split TEXT,
    state_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS variant(
    arm_variant_id TEXT PRIMARY KEY,
    testcase_id TEXT NOT NULL,
    arm TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    diff_json TEXT
);
CREATE TABLE IF NOT EXISTS rendering(
    rendering_id TEXT PRIMARY KEY,
    arm_variant_id TEXT NOT NULL,
    paraphrase_idx INTEGER NOT NULL,
    prompt_hash TEXT NOT NULL,
    n_tokens_est INTEGER
);
CREATE TABLE IF NOT EXISTS trial(
    trial_id TEXT PRIMARY KEY,
    rendering_id TEXT NOT NULL,
    block_id TEXT NOT NULL,
    model_key TEXT NOT NULL,
    arm TEXT NOT NULL,
    repetition INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    used_attempt_id TEXT,
    submission_seq INTEGER,
    created_at TEXT,
    completed_at TEXT,
    case_idx INTEGER NOT NULL,
    paraphrase_idx INTEGER NOT NULL,
    sandbox_image_id TEXT
);
CREATE INDEX IF NOT EXISTS trial_model ON trial(model_key, case_idx, arm, paraphrase_idx, repetition);
CREATE TABLE IF NOT EXISTS attempt(
    attempt_id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    transport_status TEXT NOT NULL,
    http_status INTEGER,
    provider_request_id TEXT,
    batch_id TEXT,
    request_artifact TEXT
);
CREATE INDEX IF NOT EXISTS attempt_trial ON attempt(trial_id, attempt_no);
CREATE TABLE IF NOT EXISTS response(
    attempt_id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL,
    text_artifact TEXT,
    trace_artifact TEXT,
    raw_artifact TEXT,
    finish_reason TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    reasoning_tokens INTEGER,
    usage_estimated INTEGER,
    latency_ms INTEGER,
    effective_params_json TEXT,
    serving_fingerprint_json TEXT,
    cost_usd REAL
);
CREATE INDEX IF NOT EXISTS response_trial ON response(trial_id);
CREATE TABLE IF NOT EXISTS observation(
    observation_id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL,
    feature_id TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    value_json TEXT,
    status TEXT NOT NULL,
    confidence REAL
);
CREATE INDEX IF NOT EXISTS observation_trial ON observation(trial_id, feature_id);
CREATE TABLE IF NOT EXISTS eval_call(
    eval_call_id TEXT PRIMARY KEY,
    observation_id TEXT NOT NULL,
    call_no INTEGER NOT NULL,
    attempt_no INTEGER NOT NULL,
    model_key TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    request_artifact TEXT,
    raw_artifact TEXT,
    status TEXT NOT NULL,
    http_status INTEGER,
    provider_request_id TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    started_at TEXT,
    ended_at TEXT
);
CREATE INDEX IF NOT EXISTS eval_call_obs ON eval_call(observation_id, call_no, attempt_no);
CREATE TABLE IF NOT EXISTS analysis(
    analysis_id TEXT PRIMARY KEY,
    manifest_hash TEXT NOT NULL,
    model_key TEXT NOT NULL,
    feature_id TEXT NOT NULL,
    contrast TEXT NOT NULL,
    analysis_kind TEXT NOT NULL,
    analysis_version INTEGER NOT NULL,
    results_json TEXT NOT NULL,
    label TEXT,
    code_sha TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS event(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    kind TEXT NOT NULL,
    payload_json TEXT
);
"""

TRANSPORT_STATUSES = (
    "complete",
    "http_error",
    "connection_error",
    "timeout",
    "batch_errored",
    "batch_expired",
    "in_flight",
    "abandoned",
)


def connect(path: Path) -> sqlite3.Connection:
    """Open (and create) a run database in WAL mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def db_path(run_dir: Path) -> Path:
    return run_dir / "db" / "run.sqlite"
