"""CLI command implementations (spec §32). Each returns ``(exit_code, stdout_text)``; ``cli.py`` prints."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from financial_llm_harness.errors import (
    HarnessValidationError,
    NotImplementedCommand,
)

Result = tuple[int, str]


def cmd_generate(args: argparse.Namespace) -> Result:
    from financial_llm_harness.generation.generate import generate

    res = generate(args.spec, args.output, args.panel)
    return (
        0,
        f"wrote bundle {res.bundle_dir} (design {res.manifest['design_hash'][:12]}, manifest {res.manifest['manifest_hash'][:12]})",
    )


def cmd_validate(args: argparse.Namespace) -> Result:
    from financial_llm_harness.generation.validate_bundle import validate_bundle

    rep = validate_bundle(args.manifest.parent)
    return (
        0,
        f"valid: N_clusters {rep.n_clusters} ≥ N_required {rep.n_required}; subset_n {rep.subset_n} (computed {rep.computed_subset_n})",
    )


def cmd_freeze(args: argparse.Namespace) -> Result:
    from financial_llm_harness.generation.validate_bundle import freeze

    mh = freeze(args.manifest.parent)
    return 0, f"frozen: manifest_hash {mh}"


def cmd_plan(args: argparse.Namespace) -> Result:
    from financial_llm_harness.execution.plan import render_plan

    return 0, render_plan(args.manifest.parent)


def cmd_run(args: argparse.Namespace) -> Result:
    from financial_llm_harness.execution.runner import resume_experiment, run_experiment

    if args.resume:
        res = resume_experiment(
            args.resume,
            yes=args.yes,
            max_cost_usd=args.max_cost_usd,
            max_calls=args.max_calls,
        )
        return (
            res.exit_code,
            f"run {res.run_id}: {res.status}" if res.status else res.plan_text,
        )
    if args.file is None:
        raise HarnessValidationError(
            "run needs a SPEC.yaml or MANIFEST.yaml, or --resume RUN_ID"
        )
    res = run_experiment(args.file, args.panel, yes=args.yes, dry_run=args.dry_run)
    if res.run_id is None:
        return 0, res.plan_text + (
            "\n(dry run: stopped after the policy check)"
            if args.dry_run
            else "\n(add --yes to run)"
        )
    return (
        res.exit_code,
        f"{res.plan_text}\nrun {res.run_id}: {res.status} → {res.run_dir}",
    )


def cmd_status(args: argparse.Namespace) -> Result:
    from financial_llm_harness.execution.runner import resolve_run
    from financial_llm_harness.storage.db import db_path

    run_dir = resolve_run(args.run_id)
    conn = sqlite3.connect(str(db_path(run_dir)))
    try:
        run = conn.execute(
            "SELECT run_id, status, budget_override_json FROM run"
        ).fetchone()
        lines = [f"run {run[0]}: {run[1]}"]
        for mk, st, n in conn.execute(
            "SELECT model_key, status, COUNT(*) FROM trial GROUP BY model_key, status ORDER BY model_key, status"
        ):
            lines.append(f"  {mk} {st}: {n}")
        spend = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(SUM(cost_usd),0) FROM response"
        ).fetchone()
        lines.append(
            f"  spend: calls {spend[0]}, input tokens {spend[1]}, output tokens {spend[2]}, cost ${spend[3]:.2f}"
        )
        manifest_budget = json.loads(json.dumps(_manifest_budget(run_dir)))
        lines.append(f"  caps: {manifest_budget}; override: {run[2]}")
    finally:
        conn.close()
    return 0, "\n".join(lines)


def _manifest_budget(run_dir: Path) -> dict[str, object]:
    from financial_llm_harness.core import bundle as B

    return dict(B.load_manifest(run_dir / "manifest")["budget"])


def _locked_context(run_id: str):  # type: ignore[no-untyped-def]
    from financial_llm_harness.core import bundle as B
    from financial_llm_harness.execution.runner import open_context, resolve_run
    from financial_llm_harness.generation.generate import utc_now
    from financial_llm_harness.generation.validate_bundle import current_code_ref

    run_dir = resolve_run(run_id)
    manifest = B.load_manifest(run_dir / "manifest")
    ref = current_code_ref(manifest, None)
    return run_dir, open_context(run_dir, utc_now, ref)


def cmd_evaluate(args: argparse.Namespace) -> Result:
    from financial_llm_harness.evaluation.deterministic import evaluate_deterministic
    from financial_llm_harness.execution.runner import resolve_run
    from financial_llm_harness.statistics.analysis import analyse_sensitivity
    from financial_llm_harness.storage.archive import WriterLock

    run_dir = resolve_run(args.run_id)
    with WriterLock(run_dir):
        _, ctx = _locked_context(args.run_id)
        try:
            evaluate_deterministic(ctx)
            if any(
                f.get("extractor") in ("semantic_verbalisation", "semantic_inclusion")
                for f in ctx.manifest["feature_spec"]
            ):
                analyse_sensitivity(ctx)
                from financial_llm_harness.evaluation.semantic import evaluate_semantic

                evaluate_semantic(ctx)
        finally:
            ctx.db.close()
    return 0, f"evaluated {run_dir.name}"


def cmd_analyse(args: argparse.Namespace) -> Result:
    from financial_llm_harness.execution.runner import resolve_run
    from financial_llm_harness.statistics.analysis import (
        analyse_attribution,
        analyse_sensitivity,
    )
    from financial_llm_harness.storage.archive import WriterLock

    run_dir = resolve_run(args.run_id)
    with WriterLock(run_dir):
        _, ctx = _locked_context(args.run_id)
        try:
            analyse_sensitivity(ctx)
            analyse_attribution(ctx)
        finally:
            ctx.db.close()
    return 0, f"analysed {run_dir.name}"


def cmd_render(args: argparse.Namespace) -> Result:
    from financial_llm_harness.execution.runner import resolve_run
    from financial_llm_harness.reporting.index import write_index
    from financial_llm_harness.reporting.replay import render_with_db

    run_dir = resolve_run(args.run_id)
    render_with_db(run_dir)
    write_index()
    return 0, f"rendered {run_dir.name}"


def cmd_archive(args: argparse.Namespace) -> Result:
    from financial_llm_harness.execution.runner import resolve_run
    from financial_llm_harness.storage.archive import archive_run

    run_dir = resolve_run(args.run_id)
    sha = archive_run(run_dir)
    return 0, f"archived {run_dir.name}: {sha}"


def cmd_replay(args: argparse.Namespace) -> Result:
    from financial_llm_harness.reporting.replay import reevaluate, reexecute

    if args.reevaluate:
        run_dir = reevaluate(args.run_id, re_judge=args.re_judge)
        from financial_llm_harness.reporting.index import write_index

        write_index()
        return 0, f"replayed {run_dir.name}: outputs re-rendered"
    res = reexecute(args.run_id, yes=args.yes)
    if res.run_id is None:
        return res.exit_code, "re-execution needs --yes"
    return (
        res.exit_code,
        f"replication run {res.run_id}: {res.status} → {res.run_dir}/replication.md",
    )


def cmd_compare(args: argparse.Namespace) -> Result:
    from financial_llm_harness.reporting.compare import compare_runs

    out = compare_runs(args.run_ids, args.name)
    return 0, f"wrote {out}"


def cmd_models(args: argparse.Namespace) -> Result:
    from financial_llm_harness.errors import MissingApiKey
    from financial_llm_harness.models.base import list_models
    from financial_llm_harness.models.panels import has_api_key, key_env_of

    if not has_api_key(args.provider):
        raise MissingApiKey(
            f"models list: {key_env_of(args.provider)} is not set for provider {args.provider!r}"
        )
    return 0, "\n".join(list_models(args.provider, args.region))


def cmd_probe(args: argparse.Namespace) -> Result:
    from financial_llm_harness.models.probe import run_probe, run_probes

    if args.model_key is not None:
        path = run_probe(args.model_key, args.panel)
        return 0, f"probe written to {path}"
    paths, skipped = run_probes(args.panel)
    lines = [f"probe written to {p}" for p in paths]
    lines += [f"skipped {s['model_key']}: {s['key_env']} is not set" for s in skipped]
    return 0, "\n".join(lines) or f"panel {args.panel}: no model to probe"


def cmd_calibrate(args: argparse.Namespace) -> Result:
    from financial_llm_harness.evaluation.calibration import calibrate

    path = calibrate(args.feature, args.labels, args.panel)
    return 0, f"calibration record written to {path}"


def cmd_sandbox(args: argparse.Namespace) -> Result:
    from financial_llm_harness.evaluation.sandbox import build_images

    tags = build_images()
    return 0, "\n".join(tags)


def cmd_programme(args: argparse.Namespace) -> Result:
    import financial_llm_harness.execution.programme as P

    if args.action == "plan":
        return P.cli_plan(args.programme)
    if args.action == "run":
        res = P.run_programme(args.programme, yes=args.yes, resume=args.resume)
        return res.exit_code, res.message
    if args.action == "status":
        return P.cli_status(args.programme, args.resume)
    if args.action == "report":
        return P.cli_report(args.programme, args.resume)
    raise NotImplementedCommand(args.action)


def cmd_discovery(args: argparse.Namespace) -> Result:
    from financial_llm_harness.discovery import import_legacy

    n = import_legacy(args.results_dir, args.experiment_key, args.dataset, args.output)
    return 0, f"wrote {n} hypotheses to {args.output}"


def cmd_taxonomy(args: argparse.Namespace) -> Result:
    import financial_llm_harness.taxonomy.cli as T

    return T.dispatch(args)
