"""Attempts (spec §15, §29, §36): attempt selection over transport failures."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from financial_llm_harness.core import ids
from financial_llm_harness.execution import runner
from financial_llm_harness.models.base import (
    ModelRequest,
    ModelResponse,
    load_providers,
)
from financial_llm_harness.models.profile import failure_response
from financial_llm_harness.storage.db import connect, db_path
from tests.harness.helpers import FAKE_A, GOLDEN_RUN_ID, run_small

# (case_idx, arm, paraphrase_idx) of fake--a → scripted outcomes of successive attempts
SCRIPT: dict[tuple[int, str, int], list[str]] = {
    (0, "A", 0): ["500", "ok"],
    (1, "A", 0): ["429", "429", "ok"],
    (2, "A", 0): ["refusal"],
    (3, "A", 0): ["400"],
    (4, "A", 0): ["429"] * 20,
    (5, "A", 0): ["timeout"] * 5,
}


class ScriptedAdapter:
    """Wraps the fake adapter; returns scripted transport outcomes for chosen trials."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.seen: dict[tuple[int, str, int], int] = defaultdict(int)

    def capabilities(self) -> Any:
        return self.inner.capabilities()

    async def run(self, request: ModelRequest) -> ModelResponse:
        tc = request.test_context
        assert tc is not None
        key = (tc.case_idx, tc.arm, tc.paraphrase_idx)
        n = self.seen[key]
        self.seen[key] += 1
        steps = SCRIPT.get(key, [])
        kind = steps[n] if n < len(steps) else "ok"
        sent = b'{"scripted": true}'
        if kind == "500":
            return failure_response(
                sent, "http_error", http_status=500, body=b'{"error": "boom"}'
            )
        if kind == "429":
            return failure_response(
                sent,
                "http_error",
                http_status=429,
                body=b'{"error": "slow down"}',
                headers={"retry-after": "0"},
            )
        if kind == "400":
            return failure_response(
                sent, "http_error", http_status=400, body=b'{"error": "bad"}'
            )
        if kind == "timeout":
            return failure_response(sent, "timeout")
        resp = await self.inner.run(request)
        if kind == "refusal":
            from dataclasses import replace

            return replace(
                resp,
                text="I can't help with that.",
                refusal=True,
                finish_reason="refusal",
            )
        return resp


@pytest.fixture
def scripted_run(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    real_make = runner.make_adapter
    real_open = runner.open_context

    def make(manifest: Any, entry: Any) -> Any:
        a = real_make(manifest, entry)
        return ScriptedAdapter(a) if entry["model_key"] == FAKE_A else a

    def open_ctx(*a: Any, **kw: Any) -> Any:
        ctx = real_open(*a, **kw)

        async def no_sleep(_s: float) -> None:
            return None

        ctx.sleep = no_sleep
        return ctx

    monkeypatch.setattr(runner, "make_adapter", make)
    monkeypatch.setattr(runner, "open_context", open_ctx)
    return run_small(tmp_path)


def _trial(conn: sqlite3.Connection, case: int) -> tuple[str, str, str | None]:
    row = conn.execute(
        "SELECT trial_id, status, used_attempt_id FROM trial WHERE model_key = ? AND case_idx = ? AND arm = 'A' AND paraphrase_idx = 0",
        (FAKE_A, case),
    ).fetchone()
    return row[0], row[1], row[2]


def _attempts(
    conn: sqlite3.Connection, trial_id: str
) -> list[tuple[int, str, int | None]]:
    return [
        tuple(r)
        for r in conn.execute(
            "SELECT attempt_no, transport_status, http_status FROM attempt WHERE trial_id = ? ORDER BY attempt_no",
            (trial_id,),
        ).fetchall()
    ]


def test_attempt_selection(scripted_run: Path) -> None:
    conn = sqlite3.connect(str(db_path(scripted_run)))
    try:
        # HTTP 500 with a body, then success: the success is selected
        tid, status, used = _trial(conn, 0)
        assert _attempts(conn, tid) == [(1, "http_error", 500), (2, "complete", 200)]
        assert used == ids.attempt_id(tid, 2) and status in ("SUCCESS", "PARSE_FAILURE")
        # the response row of an HTTP-error body keeps the body only, no parsed model field
        row = conn.execute(
            "SELECT text_artifact, trace_artifact, raw_artifact, finish_reason, input_tokens, output_tokens, reasoning_tokens, "
            "usage_estimated, effective_params_json, serving_fingerprint_json, cost_usd FROM response WHERE attempt_id = ?",
            (ids.attempt_id(tid, 1),),
        ).fetchone()
        assert row is not None and row[2] is not None
        assert [row[i] for i in (0, 1, 3, 4, 5, 6, 7, 8, 9, 10)] == [None] * 10
        # an HTTP 429 body is never selected
        tid, status, used = _trial(conn, 1)
        assert [a[2] for a in _attempts(conn, tid)][:2] == [429, 429]
        assert used == ids.attempt_id(tid, 3)
        # a behavioural refusal is terminal and selected, with no retry
        tid, status, used = _trial(conn, 2)
        assert status == "REFUSAL" and used == ids.attempt_id(tid, 1)
        assert len(_attempts(conn, tid)) == 1
        # any other 4xx: API_ERROR, no retry, no selected response
        tid, status, used = _trial(conn, 3)
        assert status == "API_ERROR" and used is None and len(_attempts(conn, tid)) == 1
        # 20 consecutive 429s: RATE_LIMIT, no selected response
        tid, status, used = _trial(conn, 4)
        assert (
            status == "RATE_LIMIT" and used is None and len(_attempts(conn, tid)) == 20
        )
        # 5 transport timeouts: TIMEOUT, no selected response
        tid, status, used = _trial(conn, 5)
        assert status == "TIMEOUT" and used is None and len(_attempts(conn, tid)) == 5
        # no selected attempt is ever a transport failure
        bad = conn.execute(
            "SELECT COUNT(*) FROM trial t JOIN attempt a ON a.attempt_id = t.used_attempt_id WHERE a.transport_status != 'complete'"
        ).fetchone()[0]
        assert bad == 0
    finally:
        conn.close()
    # costs.csv counts attempts with a response row: the HTTP-error bodies count as calls
    import csv

    with open(scripted_run / "results" / "costs.csv", newline="") as fh:
        rows = {r["model_key"]: r for r in csv.DictReader(fh)}
    conn = sqlite3.connect(str(db_path(scripted_run)))
    try:
        n_resp = conn.execute(
            "SELECT COUNT(*) FROM response r JOIN trial t ON t.trial_id = r.trial_id WHERE t.model_key = ?",
            (FAKE_A,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert int(rows[FAKE_A]["calls"]) == n_resp


def test_lower_attempt_no_of_two_complete_duplicates_is_selected(
    tmp_path: Path,
) -> None:
    conn = connect(tmp_path / "db" / "run.sqlite")
    tid = "trial-x"
    for no in (1, 2, 3):
        aid = ids.attempt_id(tid, no)
        status = "http_error" if no == 1 else "complete"
        conn.execute(
            "INSERT INTO attempt(attempt_id, trial_id, attempt_no, transport_status) VALUES (?, ?, ?, ?)",
            (aid, tid, no, status),
        )
        conn.execute(
            "INSERT INTO response(attempt_id, trial_id, raw_artifact) VALUES (?, ?, 'x')",
            (aid, tid),
        )
    assert runner.select_used_attempt(conn, tid) == ids.attempt_id(tid, 2)
    conn.close()


def test_registry_batch_values() -> None:
    providers = load_providers()["providers"]
    for name, cfg in providers.items():
        assert cfg["batch"] in ("openai_batch", "anthropic_message_batches", "none"), (
            name
        )
    assert {n for n, c in providers.items() if c["batch"] != "none"} == {
        "openai",
        "anthropic",
    }


def test_run_with_failures_stays_complete_and_balanced(scripted_run: Path) -> None:
    import json

    s = json.loads((scripted_run / "results" / "summary.json").read_text())
    assert s["run"]["run_id"] == GOLDEN_RUN_ID
    assert s["run"]["status"] == "COMPLETE"  # 3 missing trials of ~1,000 stay under 2%


# ---------------------------------------------------------------------------
# Batch flows (§20.10, E-4): OpenAI Batch and Anthropic Message Batches
# ---------------------------------------------------------------------------

S1_ANSWER = (
    '{"reasoning": "The figures are weighed against the mandate.", "decision": "ADD", '
    '"confidence": 0.6, "material_factors": [{"factor": "valuation", "stance": "supports"}]}'
)


class BatchServer:
    """A stateful mock of a provider's batch API; ``outcome(custom_id, n)`` scripts item n of each trial."""

    def __init__(
        self, kind: str, outcome: Any, *, old_first: bool = False, polls: int = 1
    ) -> None:
        import time

        self.kind = kind
        self.outcome = outcome
        self.old_first = old_first
        self.polls = polls
        self.files: dict[str, list[dict[str, Any]]] = {}
        self.batches: dict[str, dict[str, Any]] = {}
        self.seen: dict[str, int] = defaultdict(int)
        self.created: list[str] = []
        self.cancelled: list[str] = []
        self.now = time.time

    # -- shared -------------------------------------------------------------

    def _new_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        bid = f"batch_{len(self.batches) + 1}"
        created = self.now() - (26 * 3600 if self.old_first and not self.batches else 0)
        b = {
            "id": bid,
            "items": items,
            "created_at": created,
            "polls": 0,
            "status": "in_progress",
        }
        self.batches[bid] = b
        self.created.append(bid)
        return b

    def _results(self, b: dict[str, Any]) -> tuple[list[bytes], list[bytes]]:
        ok: list[bytes] = []
        err: list[bytes] = []
        for item in b["items"]:
            cid = item["custom_id"]
            n = self.seen[cid]
            self.seen[cid] += 1
            outcome = "expired" if b["status"] == "cancelled" else self.outcome(cid, n)
            if outcome == "expired":
                if self.kind == "anthropic":
                    ok.append(
                        json.dumps(
                            {"custom_id": cid, "result": {"type": "expired"}}
                        ).encode()
                    )
                continue
            if self.kind == "openai":
                if outcome == "errored":
                    err.append(
                        json.dumps(
                            {
                                "id": "r",
                                "custom_id": cid,
                                "response": {
                                    "status_code": 500,
                                    "request_id": "q",
                                    "body": {"error": {"message": "x"}},
                                },
                                "error": None,
                            }
                        ).encode()
                    )
                else:
                    body = {
                        "id": "chatcmpl-b",
                        "object": "chat.completion",
                        "model": "gpt-test",
                        "system_fingerprint": "fp_b",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": S1_ANSWER},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    }
                    ok.append(
                        json.dumps(
                            {
                                "id": "r",
                                "custom_id": cid,
                                "response": {
                                    "status_code": 200,
                                    "request_id": "q",
                                    "body": body,
                                },
                                "error": None,
                            }
                        ).encode()
                    )
            else:
                if outcome == "errored":
                    ok.append(
                        json.dumps(
                            {
                                "custom_id": cid,
                                "result": {
                                    "type": "errored",
                                    "error": {"type": "api_error", "message": "x"},
                                },
                            }
                        ).encode()
                    )
                else:
                    msg = {
                        "id": "msg_b",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-test",
                        "content": [{"type": "text", "text": S1_ANSWER}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 100, "output_tokens": 20},
                    }
                    ok.append(
                        json.dumps(
                            {
                                "custom_id": cid,
                                "result": {"type": "succeeded", "message": msg},
                            }
                        ).encode()
                    )
        return ok, err

    def _poll(self, b: dict[str, Any]) -> None:
        if b["status"] == "in_progress":
            b["polls"] += 1
            if b["polls"] > self.polls and not (
                self.old_first and b["id"] == "batch_1"
            ):
                b["status"] = "ended"
                b["ok"], b["err"] = self._results(b)

    # -- handler ------------------------------------------------------------

    def handler(self, request: Any) -> Any:
        import httpx

        path = request.url.path
        if self.kind == "openai":
            if path.endswith("/files") and request.method == "POST":
                lines = [
                    json.loads(x)
                    for x in request.content.split(b"\n")
                    if x.startswith(b'{"custom_id"')
                ]
                fid = f"file-in-{len(self.files) + 1}"
                self.files[fid] = lines
                return httpx.Response(
                    200,
                    json={
                        "id": fid,
                        "object": "file",
                        "bytes": 1,
                        "created_at": 0,
                        "filename": "batch.jsonl",
                        "purpose": "batch",
                        "status": "processed",
                    },
                )
            if path.endswith("/batches") and request.method == "POST":
                body = json.loads(request.content)
                b = self._new_batch(self.files[body["input_file_id"]])
                return httpx.Response(200, json=self._openai_batch(b))
            if "/batches/" in path and path.endswith("/cancel"):
                b = self.batches[path.split("/")[-2]]
                b["status"] = "cancelled"
                self.cancelled.append(b["id"])
                b["ok"], b["err"] = self._results(b)
                return httpx.Response(200, json=self._openai_batch(b))
            if "/batches/" in path:
                b = self.batches[path.split("/")[-1]]
                self._poll(b)
                return httpx.Response(200, json=self._openai_batch(b))
            if path.endswith("/content"):
                fid = path.split("/")[-2]
                bid, which = fid.split(":")
                data = self.batches[bid]["ok" if which == "out" else "err"]
                return httpx.Response(200, content=b"\n".join(data) + b"\n")
        else:
            if path.endswith("/messages/batches") and request.method == "POST":
                body = json.loads(request.content)
                b = self._new_batch(body["requests"])
                return httpx.Response(200, json=self._anthropic_batch(b))
            if path.endswith("/cancel"):
                b = self.batches[path.split("/")[-2]]
                b["status"] = "cancelled"
                self.cancelled.append(b["id"])
                b["ok"], b["err"] = self._results(b)
                return httpx.Response(200, json=self._anthropic_batch(b))
            if path.endswith("/results"):
                b = self.batches[path.split("/")[-2]]
                return httpx.Response(200, content=b"\n".join(b["ok"]) + b"\n")
            if "/messages/batches/" in path:
                b = self.batches[path.split("/")[-1]]
                self._poll(b)
                return httpx.Response(200, json=self._anthropic_batch(b))
        return httpx.Response(
            404, json={"error": f"unexpected {request.method} {path}"}
        )

    def _openai_batch(self, b: dict[str, Any]) -> dict[str, Any]:
        ended = b["status"] in ("ended", "cancelled")
        status = {
            "in_progress": "in_progress",
            "ended": "completed",
            "cancelled": "cancelled",
        }[b["status"]]
        return {
            "id": b["id"],
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "f",
            "completion_window": "24h",
            "status": status,
            "created_at": int(b["created_at"]),
            "output_file_id": f"{b['id']}:out" if ended else None,
            "error_file_id": f"{b['id']}:err" if ended else None,
        }

    def _anthropic_batch(self, b: dict[str, Any]) -> dict[str, Any]:
        from datetime import UTC, datetime

        ended = b["status"] in ("ended", "cancelled")
        return {
            "id": b["id"],
            "type": "message_batch",
            "processing_status": "ended" if ended else "in_progress",
            "created_at": datetime.fromtimestamp(b["created_at"], UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "results_url": f"https://api.anthropic.com/v1/messages/batches/{b['id']}/results"
            if ended
            else None,
        }


def _batch_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    server: BatchServer,
    *,
    interrupt_after: int | None = None,
) -> Any:
    import httpx

    from financial_llm_harness.models.base import build_adapter
    from tests.harness.helpers import FIXED_CODE_REF, fixed_clock, small_spec
    from tests.harness.test_harness_adapters import make_caps, make_entry

    target = "openai" if server.kind == "openai" else "anthropic"
    entry = {**make_entry(target), "batch": True}
    caps = make_caps(entry)
    real_make = runner.make_adapter
    real_open = runner.open_context

    def make(manifest: Any, e: Any) -> Any:
        if e["model_key"] != FAKE_A:
            return real_make(manifest, e)
        client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
        return build_adapter(entry, caps, http_client=client, key="k")

    def open_ctx(*a: Any, **kw: Any) -> Any:
        ctx = real_open(*a, **kw)
        for m in ctx.manifest["sampling_spec"]["models"]:
            if m["model_key"] == FAKE_A:
                m["batch"] = True

        async def no_sleep(_s: float) -> None:
            return None

        ctx.sleep = no_sleep
        return ctx

    monkeypatch.setattr(runner, "make_adapter", make)
    monkeypatch.setattr(runner, "open_context", open_ctx)
    return runner.run_experiment(
        small_spec(tmp_path, n_cases=50),
        "ci_fake",
        yes=True,
        run_id=GOLDEN_RUN_ID,
        clock=fixed_clock,
        code_ref=FIXED_CODE_REF,
        interrupt_after=interrupt_after,
    )


def _trial_ids(run_dir: Path) -> list[str]:
    conn = sqlite3.connect(str(db_path(run_dir)))
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT trial_id FROM trial WHERE model_key = ? ORDER BY trial_id",
                (FAKE_A,),
            ).fetchall()
        ]
    finally:
        conn.close()


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
def test_batch_items_error_expire_and_resubmit(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    flaky: dict[str, str] = {}

    def outcome(cid: str, n: int) -> str:
        role = flaky.setdefault(
            cid,
            ["always_err", "expire_once", "ok", "ok"][len(flaky) % 4]
            if len(flaky) < 8
            else "ok",
        )
        if role == "always_err":
            return "errored"
        if role == "expire_once" and n == 0:
            return "expired"
        return "ok"

    server = BatchServer(kind, outcome, polls=2)
    res = _batch_run(tmp_path, monkeypatch, server)
    assert res.status == "COMPLETE"
    conn = sqlite3.connect(str(db_path(res.run_dir)))
    try:
        err_trials = [c for c, r in flaky.items() if r == "always_err"]
        exp_trials = [c for c, r in flaky.items() if r == "expire_once"]
        assert err_trials and exp_trials
        for tid in err_trials:
            status, used = conn.execute(
                "SELECT status, used_attempt_id FROM trial WHERE trial_id = ?", (tid,)
            ).fetchone()
            attempts = conn.execute(
                "SELECT transport_status, batch_id FROM attempt WHERE trial_id = ? ORDER BY attempt_no",
                (tid,),
            ).fetchall()
            # errored three times: two resubmissions, then API_ERROR with no selected response
            assert status == "API_ERROR" and used is None
            assert [a[0] for a in attempts] == ["batch_errored"] * 3
            assert (
                len({a[1] for a in attempts}) == 3
            )  # each resubmission went into the next batch
        for tid in exp_trials:
            status, used = conn.execute(
                "SELECT status, used_attempt_id FROM trial WHERE trial_id = ?", (tid,)
            ).fetchone()
            attempts = conn.execute(
                "SELECT transport_status FROM attempt WHERE trial_id = ? ORDER BY attempt_no",
                (tid,),
            ).fetchall()
            assert [a[0] for a in attempts] == ["batch_expired", "complete"]
            assert used == ids.attempt_id(tid, 2) and status == "SUCCESS"
        # every batch ID was stored on the attempts before results were read
        n_batches = conn.execute(
            "SELECT COUNT(DISTINCT batch_id) FROM attempt a JOIN trial t ON t.trial_id = a.trial_id WHERE t.model_key = ?",
            (FAKE_A,),
        ).fetchone()[0]
        assert n_batches == len(server.created) == 3
        # the request artefact of a batch item is its request line, holding custom_id = trial_id
        from financial_llm_harness.storage.artifacts import ArtifactStore

        store = ArtifactStore(res.run_dir / "artifacts")
        art = conn.execute(
            "SELECT request_artifact FROM attempt WHERE trial_id = ? ORDER BY attempt_no LIMIT 1",
            (err_trials[0],),
        ).fetchone()[0]
        assert json.loads(store.get(art))["custom_id"] == err_trials[0]
    finally:
        conn.close()


def test_batch_cancelled_after_25_hours(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = BatchServer("openai", lambda cid, n: "ok", old_first=True)
    res = _batch_run(tmp_path, monkeypatch, server)
    assert server.cancelled == ["batch_1"]
    assert res.status == "COMPLETE"
    conn = sqlite3.connect(str(db_path(res.run_dir)))
    try:
        rows = conn.execute(
            "SELECT a.transport_status, COUNT(*) FROM attempt a JOIN trial t ON t.trial_id = a.trial_id WHERE t.model_key = ? GROUP BY a.transport_status",
            (FAKE_A,),
        ).fetchall()
    finally:
        conn.close()
    counts = dict(rows)
    assert counts["batch_expired"] == counts["complete"] == len(_trial_ids(res.run_dir))


def test_resume_repolls_live_batch_without_resubmitting(
    tmp_path: Path, harness_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from financial_llm_harness.execution.runner import (
        InjectedInterruption,
        resume_experiment,
    )
    from tests.harness.helpers import FIXED_CODE_REF, fixed_clock

    server = BatchServer("anthropic", lambda cid, n: "ok", polls=1)
    with pytest.raises(InjectedInterruption):
        _batch_run(tmp_path, monkeypatch, server, interrupt_after=1)
    assert server.created == ["batch_1"]
    res = resume_experiment(
        GOLDEN_RUN_ID, yes=True, clock=fixed_clock, code_ref=FIXED_CODE_REF
    )
    assert res.status == "COMPLETE" and res.run_dir is not None
    assert server.created == ["batch_1"]  # re-polled, never resubmitted
    conn = sqlite3.connect(str(db_path(res.run_dir)))
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM attempt a JOIN trial t ON t.trial_id = a.trial_id WHERE t.model_key = ? AND a.transport_status = 'abandoned'",
            (FAKE_A,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 0
