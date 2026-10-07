"""Replay (spec §31) and archive-aware rendering (§0.9, §32A.8).

``replay --reevaluate`` first runs the §4 frozen-input checks (bundle hashes,
executable source, evaluator prompt and parser hashes) and refuses with
FROZEN_INPUT_DRIFT, writing nothing, on any mismatch. It then re-parses the
stored responses, re-runs the deterministic evaluators and the analysis, and
re-renders; the outputs are byte-identical to the original render.
``replay --reexecute`` is a new run from the same manifest, reported as a
replication.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core.code_ref import CodeRef
from financial_llm_harness.errors import FrozenInputDrift, HarnessError
from financial_llm_harness.storage.archive import (
    ARCHIVE_NAME,
    WriterLock,
    extract_archive,
)
from financial_llm_harness.storage.db import db_path


@contextmanager
def ensure_db(run_dir: Path, *, into_run_dir: bool = True) -> Iterator[Path]:
    """Yield the directory holding ``db/`` and ``artifacts/`` (§0.9 Extraction).

    When the run's ``db/`` is absent, the archive is verified against
    ``ARCHIVE.sha256`` and every member validated before anything is written.
    ``render`` and ``replay`` extract into the run directory; ``programme report``
    (``into_run_dir=False``) extracts into a temporary directory that is deleted
    afterwards, leaving the run directory as it was.
    """
    if db_path(run_dir).exists():
        yield run_dir
        return
    if not (run_dir / ARCHIVE_NAME).exists():
        raise HarnessError(f"{run_dir.name}: no db/run.sqlite and no {ARCHIVE_NAME}")
    if into_run_dir:
        with tempfile.TemporaryDirectory(prefix="fh-extract-", dir=run_dir) as tmp:
            # validate and extract fully before moving anything into place
            extract_archive(run_dir, Path(tmp))
            for part in ("db", "artifacts"):
                src = Path(tmp) / part
                if src.exists():
                    if (run_dir / part).exists():
                        shutil.rmtree(run_dir / part)
                    src.rename(run_dir / part)
        yield run_dir
        return
    with tempfile.TemporaryDirectory(prefix="fh-extract-") as tmp:
        extract_archive(run_dir, Path(tmp))
        yield Path(tmp)


def render_with_db(run_dir: Path) -> None:
    """L12 render from the run's database, extracting the archive first when ``db/`` is absent."""
    from financial_llm_harness.reporting.render import render_run

    with ensure_db(run_dir):
        with WriterLock(run_dir):
            render_run(run_dir)


def reevaluate(
    run_id: str,
    *,
    re_judge: bool = False,
    code_ref: CodeRef | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Path:
    """``replay RUN_ID --reevaluate``; returns the run directory."""
    from financial_llm_harness.evaluation.deterministic import evaluate_deterministic
    from financial_llm_harness.execution.runner import open_context, resolve_run
    from financial_llm_harness.generation.generate import utc_now
    from financial_llm_harness.generation.validate_bundle import current_code_ref
    from financial_llm_harness.reporting.render import render_run
    from financial_llm_harness.statistics.analysis import (
        analyse_attribution,
        analyse_sensitivity,
    )

    run_dir = resolve_run(run_id)
    bundle_dir = run_dir / "manifest"
    manifest = B.load_manifest(bundle_dir)
    ref = current_code_ref(manifest, code_ref)
    drift = B.frozen_input_mismatches(bundle_dir, ref, require_lock=True)
    if drift:
        raise FrozenInputDrift(drift)
    semantic = any(
        f.get("extractor") in ("semantic_verbalisation", "semantic_inclusion")
        for f in manifest["feature_spec"]
    )
    with ensure_db(run_dir), WriterLock(run_dir):
        ctx = open_context(run_dir, clock or utc_now, ref)
        try:
            evaluate_deterministic(ctx, execute_code=False)
            analyse_sensitivity(ctx)
            if semantic and re_judge:
                from financial_llm_harness.evaluation.semantic import (
                    evaluate_semantic,
                )

                evaluate_semantic(ctx, re_judge=True)
                analyse_sensitivity(ctx)
            analyse_attribution(ctx)
            ctx.db.commit()
        finally:
            ctx.db.close()
        render_run(run_dir)
    return run_dir


@dataclass(frozen=True)
class ReplicationResult:
    run_id: str | None
    run_dir: Path | None
    status: str | None
    exit_code: int
    report: dict[str, Any] | None


def _primary_clusters(summary_run: Path) -> dict[str, dict[int, float]]:
    """Per model: cluster_ordinal → x_c on the primary contrast, from ``results/clusters.csv``."""
    import csv

    summary = json.loads((summary_run / "results" / "summary.json").read_text())
    d = summary["design"]
    contrast = (
        "B_vs_A"
        if d["estimand"] != "arm_level"
        else f"level_{B.load_manifest(summary_run / 'manifest')['hypothesis']['level_arm']}"
    )
    out: dict[str, dict[int, float]] = {}
    with open(summary_run / "results" / "clusters.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["contrast"] != contrast or row["x_c"] == "":
                continue
            out.setdefault(row["model_key"], {})[int(row["cluster_ordinal"])] = float(
                row["x_c"]
            )
    return out


def replication_report(original: Path, replica: Path) -> dict[str, Any]:
    """θ difference with a cluster-paired CI, and label agreement, per model."""
    from financial_llm_harness.statistics.paired import ci

    so = json.loads((original / "results" / "summary.json").read_text())
    sr = json.loads((replica / "results" / "summary.json").read_text())
    alpha = float(so["design"]["alpha"])
    xo, xr = _primary_clusters(original), _primary_clusters(replica)
    by_r = {m["model_key"]: m for m in sr["models"]}
    models = []
    for m in so["models"]:
        mk = m["model_key"]
        r: Mapping[str, Any] = by_r.get(mk) or {"primary": {}}
        common = sorted(set(xo.get(mk, {})) & set(xr.get(mk, {})))
        diffs = np.array([xr[mk][c] - xo[mk][c] for c in common], dtype=float)
        interval = ci(diffs, alpha) if len(diffs) else None
        po, pr = m["primary"], r["primary"]
        models.append(
            {
                "model_key": mk,
                "theta_original": po.get("theta"),
                "theta_replication": pr.get("theta"),
                "theta_difference": float(diffs.mean()) if len(diffs) else None,
                "ci": list(interval) if interval is not None else [None, None],
                "n_clusters_paired": len(common),
                "zone_agrees": po.get("zone") == pr.get("zone"),
                "sensitivity_label_agrees": po.get("sensitivity_label")
                == pr.get("sensitivity_label"),
                "attribution_label_agrees": po.get("attribution_label")
                == pr.get("attribution_label"),
            }
        )
    return {
        "schema_version": "1.0",
        "original_run_id": so["run"]["run_id"],
        "replication_run_id": sr["run"]["run_id"],
        "manifest_hash": so["run"]["manifest_hash"],
        "alpha": alpha,
        "models": models,
    }


def _replication_md(rep: Mapping[str, Any]) -> str:
    from financial_llm_harness.reporting.markdown import ci, code, table, theta

    rows = [
        [
            code(m["model_key"]),
            theta(m["theta_original"]),
            theta(m["theta_replication"]),
            theta(m["theta_difference"]),
            ci(*m["ci"]),
            str(m["n_clusters_paired"]),
            "yes" if m["zone_agrees"] else "no",
            "yes" if m["sensitivity_label_agrees"] else "no",
            "yes" if m["attribution_label_agrees"] else "no",
        ]
        for m in rep["models"]
    ]
    return "\n".join(
        [
            f"# Replication of run {code(rep['original_run_id'][:8])}",
            "",
            f"Replication run {code(rep['replication_run_id'][:8])} · manifest {code(rep['manifest_hash'][:8])}",
            "",
            "## 1. Primary effect and labels",
            "",
            table(
                [
                    "Model",
                    "θ̂ original",
                    "θ̂ replication",
                    "Difference",
                    "CI (cluster-paired)",
                    "Clusters",
                    "Zone agrees",
                    "Sensitivity agrees",
                    "Attribution agrees",
                ],
                rows,
            ),
            "",
        ]
    )


def reexecute(
    run_id: str,
    *,
    yes: bool,
    new_run_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
    code_ref: CodeRef | None = None,
) -> ReplicationResult:
    """``replay RUN_ID --reexecute``: a new run from the same frozen manifest, reported as a replication."""
    from financial_llm_harness.execution.runner import resolve_run, run_experiment

    original = resolve_run(run_id)
    res = run_experiment(
        original / "manifest" / B.MANIFEST_FILE,
        None,
        yes=yes,
        run_id=new_run_id,
        clock=clock,
        code_ref=code_ref,
    )
    if res.run_dir is None or res.status is None:
        return ReplicationResult(None, None, None, res.exit_code, None)
    rep = replication_report(original, res.run_dir)
    B.atomic_write_text(res.run_dir / "replication.json", B.json_text(rep))
    B.atomic_write_text(res.run_dir / "replication.md", _replication_md(rep))
    return ReplicationResult(res.run_id, res.run_dir, res.status, res.exit_code, rep)
