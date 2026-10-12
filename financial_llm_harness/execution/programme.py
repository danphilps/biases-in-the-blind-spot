"""Programmes: every selected question against every panel model (spec §32B, §25, D-17, D-18, D-20, E-30).

``programme run`` resolves the selection, generates, validates and freezes each
experiment against the programme panel, plans against the programme budget,
runs the valid experiments one at a time with cap = min(manifest cap, budget
remaining), then writes the programme report. ``programme report`` is a pure
function of ``state.json``, the programme file, each run's database and the
registry: it runs BH per (model, claim family) on the stored p-values and never
recomputes an estimate or a confidence interval.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import ids, yaml_io
from financial_llm_harness.core.canonical_json import sha256_bytes
from financial_llm_harness.core.code_ref import CodeRef
from financial_llm_harness.core.paths import programmes_dir, repo_root
from financial_llm_harness.errors import (
    BudgetExhausted,
    HarnessError,
    HarnessValidationError,
    PolicyRefusal,
    StaleProbe,
)

PROGRAMME_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
KEY_RE = re.compile(r"^[a-z0-9_]{1,48}$")
TERMINAL = ("COMPLETE", "INCOMPLETE", "FAILED", "SKIPPED_INVALID")
RUN_EXIT = {"COMPLETE": 0, "INCOMPLETE": 4, "PAUSED_BUDGET": 3, "FAILED": 1}


class InjectedProgrammeStop(Exception):
    """Fault injection between experiments for the programme resume test (§36)."""


@dataclass
class ProgrammeResult:
    exit_code: int
    message: str
    programme_dir: Path | None = None
    programme_run_id: str | None = None


# ---------------------------------------------------------------------------
# Programme file and selection (§32B.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Selected:
    path: Path
    rel: str
    spec: dict[str, Any]

    @property
    def experiment_key(self) -> str:
        return str(self.spec["experiment_key"])


def load_programme(path: Path) -> dict[str, Any]:
    data = yaml_io.load(path)
    where = str(path)
    if not isinstance(data, dict):
        raise HarnessValidationError(f"{where}: a programme file is a mapping")
    for key in ("schema_version", "programme_key", "panel", "select", "budget"):
        if key not in data:
            raise HarnessValidationError(f"{where}: missing {key!r} (§32B.1)")
    if data["schema_version"] != PROGRAMME_SCHEMA_VERSION:
        raise HarnessValidationError(f"{where}: schema_version must be 1")
    if not isinstance(data["programme_key"], str) or not KEY_RE.match(
        data["programme_key"]
    ):
        raise HarnessValidationError(
            f"{where}: programme_key must match [a-z0-9_]{{1,48}} (§32A.1)"
        )
    budget = data["budget"] or {}
    for key in ("max_cost_usd", "max_calls"):
        if budget.get(key) is None:
            raise HarnessValidationError(
                f"{where}: budget.{key} is required for a programme (§32B.1, D-2)"
            )
    policy = data.get("on_experiment_failure", "continue")
    if policy not in ("continue", "stop"):
        raise HarnessValidationError(
            f"{where}: on_experiment_failure must be continue or stop"
        )
    data["on_experiment_failure"] = policy
    return data


def _generated_files(sel: Mapping[str, Any]) -> list[Path]:
    """``select.generated``: files under ``experiments/financial/generated/`` matching wave, priority, mode and variants."""
    root = repo_root() / "experiments" / "financial" / "generated"
    faq_path = repo_root() / "financial_llm_harness" / "taxonomy" / "faq.yaml"
    if not faq_path.exists():
        raise HarnessValidationError(
            "select.generated needs the taxonomy registry (taxonomy/faq.yaml), which does not exist yet (M8)"
        )
    faq = yaml_io.load(faq_path)
    items = {
        str(i["faq_id"]): i
        for i in (faq.get("items") if isinstance(faq, dict) else faq) or []
    }
    waves = set(sel.get("waves") or [])
    prios = set(sel.get("priorities") or [])
    modes = set(sel.get("modes") or [])
    variants = sel.get("variants")
    out = []
    for p in sorted(root.glob("*/*/*.yaml")):
        spec = yaml_io.load(p)
        if not isinstance(spec, dict):
            continue
        item = items.get(str(spec.get("faq_id")))
        if item is None:
            continue
        if waves and item.get("wave") not in waves:
            continue
        if prios and item.get("priority") not in prios:
            continue
        if modes and spec.get("mode") not in modes:
            continue
        if isinstance(variants, list) and spec.get("variant_id") not in variants:
            continue
        out.append(p)
    return out


def resolve_selection(programme: Mapping[str, Any]) -> list[Selected]:
    """The ordered list E (§32B.1); exit 2 on a missing path, an empty selection or a duplicate key or pair."""
    sel = programme["select"] or {}
    paths: list[Path] = []
    for s in sel.get("specs") or []:
        p = Path(s)
        p = p if p.is_absolute() else repo_root() / p
        if not p.exists():
            raise HarnessValidationError(f"programme: spec {s} does not exist")
        paths.append(p)
    if sel.get("generated"):
        paths += _generated_files(sel["generated"])
    seen: dict[Path, None] = {}
    for p in paths:
        seen.setdefault(p.resolve(), None)
    chosen = []
    for p in seen:
        spec = yaml_io.load(p)
        if not isinstance(spec, dict) or "experiment_key" not in spec:
            raise HarnessValidationError(f"programme: {p} is not a spec")
        try:
            rel = str(p.relative_to(repo_root().resolve()))
        except ValueError:
            rel = str(p)
        chosen.append(Selected(p, rel, spec))
    if not chosen:
        raise HarnessValidationError("programme: the selection is empty")
    keys: dict[str, str] = {}
    pairs: dict[tuple[str, str, str], str] = {}
    for c in chosen:
        if c.experiment_key in keys:
            raise HarnessValidationError(
                f"programme: {c.rel} and {keys[c.experiment_key]} share experiment_key {c.experiment_key}"
            )
        keys[c.experiment_key] = c.rel
        pair = (
            str(c.spec.get("faq_id")),
            str(c.spec.get("variant_id")),
            str(c.spec.get("intervention_id")),
        )
        if pair[0] != "none":
            if pair in pairs:
                raise HarnessValidationError(
                    f"programme: {c.rel} and {pairs[pair]} share (faq_id, variant_id, intervention_id) {pair}"
                )
            pairs[pair] = c.rel
    return sorted(chosen, key=lambda c: c.experiment_key)


# ---------------------------------------------------------------------------
# State (§32B.3)
# ---------------------------------------------------------------------------


def slug(key: str) -> str:
    return re.sub(r"[_.]", "-", key)


def file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def panel_file(name: str) -> Path:
    from financial_llm_harness.models.panels import panel_path

    return panel_path(name)


def write_state(pdir: Path, state: Mapping[str, Any]) -> None:
    B.atomic_write_text(pdir / "state.json", B.json_text(state))


def read_state(pdir: Path) -> dict[str, Any]:
    return json.loads((pdir / "state.json").read_text(encoding="utf-8"))


def find_programme_dir(programme_key: str, programme_run_id: str | None) -> Path:
    root = programmes_dir()
    cands = []
    for d in sorted(root.glob(f"*_{slug(programme_key)}_*")) if root.is_dir() else []:
        s = d / "state.json"
        if not s.exists():
            continue
        st = json.loads(s.read_text(encoding="utf-8"))
        if programme_run_id is None or st["programme_run_id"].startswith(
            programme_run_id
        ):
            cands.append((st.get("started_at") or "", d))
    if not cands:
        raise HarnessError(
            f"no programme run of {programme_key}"
            + (f" with ID {programme_run_id}" if programme_run_id else "")
        )
    if programme_run_id is not None and len(cands) > 1:
        raise HarnessValidationError(
            f"programme run ID {programme_run_id} is ambiguous"
        )
    return sorted(cands)[-1][1]


# ---------------------------------------------------------------------------
# Generate, validate, freeze and plan (§32B.3 steps 3–4)
# ---------------------------------------------------------------------------


def _prepare(
    sel: Selected,
    panel: str,
    out: Path,
    code_ref: CodeRef | None,
    clock: Callable[[], datetime],
) -> None:
    from financial_llm_harness.generation.generate import generate
    from financial_llm_harness.generation.validate_bundle import freeze

    res = generate(sel.path, out, panel, code_ref=code_ref, clock=clock)
    freeze(out, code_ref=code_ref, regenerated=res.corpus, clock=clock)


def plan_rows(manifests: Mapping[str, Path]) -> list[dict[str, Any]]:
    from financial_llm_harness.execution.plan import plan_data

    rows = []
    for key in sorted(manifests):
        d = plan_data(manifests[key])
        for m in d["models"]:
            rows.append(
                {
                    "row_id": ids.row_id("plan", key, m["model_key"]),
                    "experiment_key": key,
                    "model_key": m["model_key"],
                    "n_clusters": d["n_clusters"],
                    "trials": m["trials"],
                    "calls": m["calls"],
                    "input_tokens_est": m["input_tokens_est"],
                    "output_tokens_max": m["output_tokens_max"],
                    "cost_usd_low": m["cost_usd_low"],
                    "cost_usd_high": m["cost_usd_high"],
                }
            )
    return rows


PLAN_COLUMNS = (
    "row_id",
    "experiment_key",
    "model_key",
    "n_clusters",
    "trials",
    "calls",
    "input_tokens_est",
    "output_tokens_max",
    "cost_usd_low",
    "cost_usd_high",
)


def _plan_text(
    programme: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    skipped: Mapping[str, str],
) -> str:
    from financial_llm_harness.models.panels import load_panel, split_by_api_key

    calls = sum(int(r["calls"]) for r in rows)
    hi = sum(float(r["cost_usd_high"]) for r in rows)
    b = programme["budget"]
    keyed, no_key = split_by_api_key(load_panel(programme["panel"]).models)
    lines = [
        f"programme plan: {programme['programme_key']} on panel {programme['panel']}",
        f"  models tested: {', '.join(m['model_key'] for m in keyed) or 'none'}",
    ]
    for sk in no_key:
        lines.append(f"  model skipped: {sk['model_key']} ({sk['key_env']} is not set)")
    for key in sorted({r["experiment_key"] for r in rows}):
        er = [r for r in rows if r["experiment_key"] == key]
        lines.append(
            f"  {key}: calls {sum(int(r['calls']) for r in er)}, cost ≤ ${sum(float(r['cost_usd_high']) for r in er):,.2f}"
        )
    for key, msg in sorted(skipped.items()):
        lines.append(f"  {key}: SKIPPED_INVALID — {msg}")
    lines.append(
        f"  total calls {calls} (budget {b['max_calls']}), cost ≤ ${hi:,.2f} (budget ${float(b['max_cost_usd']):,.2f})"
    )
    return "\n".join(lines)


def _check_budget(
    programme: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    calls = sum(int(r["calls"]) for r in rows)
    hi = sum(float(r["cost_usd_high"]) for r in rows)
    b = programme["budget"]
    if calls > int(b["max_calls"]) or hi > float(b["max_cost_usd"]):
        raise BudgetExhausted(
            f"programme plan exceeds the programme budget: calls {calls} > {b['max_calls']} "
            f"or cost upper bound ${hi:,.2f} > ${float(b['max_cost_usd']):,.2f} (§32B.3)"
        )


def cli_plan(path: Path) -> tuple[int, str]:
    """``programme plan``: generate, validate and freeze in a temporary directory, then plan; writes nothing."""
    from financial_llm_harness.generation.generate import utc_now

    programme = load_programme(path)
    sel = resolve_selection(programme)
    with tempfile.TemporaryDirectory(prefix="fh-programme-plan-") as tmp:
        manifests, skipped = {}, {}
        for s in sel:
            out = Path(tmp) / s.experiment_key
            try:
                _prepare(s, programme["panel"], out, None, utc_now)
                manifests[s.experiment_key] = out
            except (HarnessValidationError, StaleProbe) as exc:
                skipped[s.experiment_key] = str(exc)
        rows = plan_rows(manifests)
        text = _plan_text(programme, rows, skipped)
        _check_budget(programme, rows)
    return 0, text


# ---------------------------------------------------------------------------
# Run (§32B.3)
# ---------------------------------------------------------------------------


def _spend(run_dir: Path) -> tuple[int, float]:
    import sqlite3

    from financial_llm_harness.storage.db import db_path

    if not db_path(run_dir).exists():
        return 0, 0.0
    conn = sqlite3.connect(str(db_path(run_dir)))
    try:
        r = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0) FROM response"
        ).fetchone()
        e = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0) FROM eval_call WHERE status = 'complete' OR raw_artifact IS NOT NULL"
        ).fetchone()
        return int(r[0]) + int(e[0]), float(r[1]) + float(e[1])
    finally:
        conn.close()


def _remaining(
    programme: Mapping[str, Any], state: Mapping[str, Any]
) -> tuple[int, float]:
    from financial_llm_harness.execution.runner import resolve_run

    calls, cost = 0, 0.0
    for e in state["experiments"]:
        if e.get("run_id"):
            try:
                c, u = _spend(resolve_run(e["run_id"]))
            except HarnessError:
                continue
            calls += c
            cost += u
    b = programme["budget"]
    return int(b["max_calls"]) - calls, float(b["max_cost_usd"]) - cost


def _cap(
    manifest_budget: Mapping[str, Any], remaining: tuple[int, float]
) -> dict[str, Any]:
    """Budget cap = min(manifest cap, programme budget remaining) (§30, §32B.3)."""
    calls, cost = remaining
    mc = manifest_budget.get("max_calls")
    mu = manifest_budget.get("max_cost_usd")
    return {
        "max_calls": max(0, calls if mc is None else min(int(mc), calls)),
        "max_cost_usd": max(0.0, cost if mu is None else min(float(mu), cost)),
    }


def run_programme(
    path: Path,
    *,
    yes: bool,
    resume: str | None = None,
    programme_run_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
    run_id_for: Callable[[str], str] | None = None,
    code_ref: CodeRef | None = None,
    stop_after: int | None = None,
) -> ProgrammeResult:
    """Library entry point of ``programme run`` (§31A, E-23)."""
    from financial_llm_harness.generation.generate import iso_z, utc_now

    clk = clock or utc_now
    programme = load_programme(path)
    if resume is not None:
        pdir = find_programme_dir(programme["programme_key"], resume)
        state = read_state(pdir)
        if state["programme_sha256"] != file_sha(path):
            raise HarnessValidationError(
                "programme --resume: the programme file has changed"
            )
        if state["panel_sha256"] != file_sha(panel_file(programme["panel"])):
            raise HarnessValidationError(
                "programme --resume: the panel file has changed"
            )
        sel = {s.experiment_key: s for s in resolve_selection(programme)}
    else:
        selection = resolve_selection(programme)
        if not yes:
            code, text = cli_plan(path)
            return ProgrammeResult(code, text + "\n(add --yes to run)")
        prid = programme_run_id or str(uuid.uuid4())
        started = clk()
        pdir = (
            programmes_dir()
            / f"{started.strftime('%Y-%m-%d')}_{slug(programme['programme_key'])}_{prid[:8]}"
        )
        if pdir.exists():
            raise HarnessError(f"programme directory {pdir} already exists")
        pdir.mkdir(parents=True)
        if any(
            s.spec.get("data_classification", "synthetic") != "synthetic"
            for s in selection
        ):
            (pdir / ".gitignore").write_text("*\n", encoding="utf-8")
        shutil.copyfile(path, pdir / "programme.yaml")
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "programme_run_id": prid,
            "programme_key": programme["programme_key"],
            "programme_sha256": file_sha(path),
            "panel_sha256": file_sha(panel_file(programme["panel"])),
            "started_at": iso_z(started),
            "status": "RUNNING",
            "experiments": [
                {
                    "experiment_key": s.experiment_key,
                    "spec_path": s.rel,
                    "manifest_hash": None,
                    "run_id": None,
                    "status": "PENDING",
                    "exit_code": None,
                    "message": None,
                }
                for s in selection
            ],
        }
        write_state(pdir, state)
        sel = {s.experiment_key: s for s in selection}
        # step 3: generate, validate and freeze every experiment against the programme panel
        for e in state["experiments"]:
            out = pdir / "manifests" / e["experiment_key"]
            try:
                _prepare(
                    sel[e["experiment_key"]], programme["panel"], out, code_ref, clk
                )
                e["manifest_hash"] = B.load_manifest(out)["manifest_hash"]
            except (HarnessValidationError, StaleProbe) as exc:
                if out.exists():
                    shutil.rmtree(out)
                e.update(
                    status="SKIPPED_INVALID", exit_code=exc.exit_code, message=str(exc)
                )
                write_state(pdir, state)
                if programme["on_experiment_failure"] == "stop":
                    state["status"] = "STOPPED"
                    write_state(pdir, state)
                    return ProgrammeResult(exc.exit_code, str(exc), pdir, prid)
            write_state(pdir, state)
        # step 4: plan against the programme budget, before any call
        manifests = {
            e["experiment_key"]: pdir / "manifests" / e["experiment_key"]
            for e in state["experiments"]
            if e["status"] != "SKIPPED_INVALID"
        }
        rows = plan_rows(manifests)
        (pdir / "results").mkdir(exist_ok=True)
        from financial_llm_harness.reporting.tables import csv_text

        B.atomic_write_text(pdir / "results" / "plan.csv", csv_text(PLAN_COLUMNS, rows))
        try:
            _check_budget(programme, rows)
        except BudgetExhausted as exc:
            state["status"] = "PAUSED_BUDGET"
            write_state(pdir, state)
            return ProgrammeResult(3, str(exc), pdir, prid)
    return _run_experiments(
        programme, pdir, state, clk, run_id_for, code_ref, stop_after
    )


def _run_experiments(
    programme: Mapping[str, Any],
    pdir: Path,
    state: dict[str, Any],
    clk: Callable[[], datetime],
    run_id_for: Callable[[str], str] | None,
    code_ref: CodeRef | None,
    stop_after: int | None,
) -> ProgrammeResult:
    from financial_llm_harness.execution.runner import (
        resume_experiment,
        run_experiment,
    )

    prid = state["programme_run_id"]
    done_now = 0
    state["status"] = "RUNNING"
    write_state(pdir, state)
    for e in state["experiments"]:
        if e["status"] in ("COMPLETE", "SKIPPED_INVALID", "FAILED"):
            continue
        if stop_after is not None and done_now >= stop_after:
            raise InjectedProgrammeStop()
        bundle = pdir / "manifests" / e["experiment_key"]
        manifest = B.load_manifest(bundle)
        cap = _cap(manifest["budget"], _remaining(programme, state))
        try:
            if e["run_id"] is not None and e["status"] in (
                "RUNNING",
                "INCOMPLETE",
                "PAUSED_BUDGET",
            ):
                res = resume_experiment(
                    e["run_id"],
                    yes=True,
                    clock=clk,
                    code_ref=code_ref,
                    max_cost_usd=cap["max_cost_usd"],
                    max_calls=cap["max_calls"],
                )
            else:
                rid = (
                    run_id_for(e["experiment_key"]) if run_id_for else str(uuid.uuid4())
                )
                e["run_id"] = rid
                e["status"] = "RUNNING"
                write_state(pdir, state)
                res = run_experiment(
                    bundle / B.MANIFEST_FILE,
                    None,
                    yes=True,
                    run_id=rid,
                    clock=clk,
                    code_ref=code_ref,
                    budget_override=cap,
                )
        except PolicyRefusal as exc:
            e.update(status="FAILED", exit_code=5, message=str(exc))
            state["status"] = "STOPPED"
            write_state(pdir, state)
            return ProgrammeResult(5, str(exc), pdir, prid)
        except HarnessError as exc:
            e.update(status="FAILED", exit_code=exc.exit_code, message=str(exc))
            write_state(pdir, state)
            if programme["on_experiment_failure"] == "stop":
                state["status"] = "STOPPED"
                write_state(pdir, state)
                return ProgrammeResult(exc.exit_code, str(exc), pdir, prid)
            continue
        e["exit_code"] = res.exit_code
        e["status"] = res.status or "FAILED"
        e["message"] = None
        write_state(pdir, state)
        done_now += 1
        if res.status == "PAUSED_BUDGET":
            state["status"] = "PAUSED_BUDGET"
            write_state(pdir, state)
            return ProgrammeResult(
                3, f"{e['experiment_key']}: programme budget exhausted", pdir, prid
            )
    valid = [e for e in state["experiments"] if e["status"] != "SKIPPED_INVALID"]
    complete = all(e["status"] == "COMPLETE" for e in valid)
    state["status"] = "COMPLETE" if complete else "INCOMPLETE"
    write_state(pdir, state)
    from financial_llm_harness.reporting.programme import render_programme

    render_programme(pdir)
    code = 0 if complete else 4
    return ProgrammeResult(
        code, f"programme {prid}: {state['status']} → {pdir}", pdir, prid
    )


def cli_status(path: Path, run_id: str | None) -> tuple[int, str]:
    programme = load_programme(path)
    pdir = find_programme_dir(programme["programme_key"], run_id)
    st = read_state(pdir)
    lines = [f"programme {st['programme_run_id']}: {st['status']} ({pdir.name})"]
    for e in st["experiments"]:
        lines.append(
            f"  {e['experiment_key']}: {e['status']}"
            + (f" run {e['run_id'][:8]}" if e.get("run_id") else "")
            + (f" exit {e['exit_code']}" if e.get("exit_code") is not None else "")
        )
    calls, cost = _remaining(programme, st)
    lines.append(f"  budget remaining: calls {calls}, ${cost:,.2f}")
    return 0, "\n".join(lines)


def cli_report(path: Path, run_id: str | None) -> tuple[int, str]:
    from financial_llm_harness.reporting.programme import render_programme

    programme = load_programme(path)
    pdir = find_programme_dir(programme["programme_key"], run_id)
    render_programme(pdir)
    return 0, f"programme report written to {pdir}"
