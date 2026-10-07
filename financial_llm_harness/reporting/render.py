"""L12 render: every output file from ``db/run.sqlite`` and artefacts only (spec §31A, §32A.8).

Output is byte-identical across renders. Timestamps and host details go only to
``logs/build_info.json``.
"""

from __future__ import annotations

import json
import platform
import shutil
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from financial_llm_harness.core import bundle as B
from financial_llm_harness.reporting import (
    dictionary,
    examples,
    extras,
    html,
    markdown,
    summary,
    svg,
    tables,
)
from financial_llm_harness.storage.db import db_path


def _fd(
    source: str, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> html.FigureData:
    return html.FigureData(
        source, tuple(columns), [[tables.cell(r.get(c)) for c in columns] for r in rows]
    )


def figure_data(
    rd: tables.RunData,
    summ: Mapping[str, Any],
    effects: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, Any]],
    attribution: Sequence[Mapping[str, Any]],
    costs: Sequence[Mapping[str, Any]],
) -> dict[str, html.FigureData]:
    """The exact CSV rows behind each figure (§32A.5)."""
    d = summ["design"]
    level = d["estimand"] == "arm_level"
    contrast = f"level_{rd.manifest['hypothesis']['level_arm']}" if level else "B_vs_A"
    prim = [
        e
        for e in effects
        if e["feature_id"] == d["primary_feature"] and e["contrast"] == contrast
    ]
    eff = _fd("results/effects.csv", tables.EFFECTS_COLUMNS, prim)
    arms = {"A", "B"} if not level else {rd.manifest["hypothesis"]["level_arm"]}
    return {
        "01_primary_effect.svg": eff,
        "02_arm_rates.svg": eff,
        "03_controls.svg": _fd(
            "results/controls.csv", tables.CONTROLS_COLUMNS, controls
        ),
        "04_missingness.svg": _fd(
            "results/missingness.csv",
            tables.MISSINGNESS_COLUMNS,
            [m for m in missing if m["arm"] in arms and m["status_class"] == "invalid"],
        ),
        "05_attribution.svg": _fd(
            "results/attribution.csv", tables.ATTRIBUTION_COLUMNS, attribution
        ),
        "06_cost.svg": _fd("results/costs.csv", tables.COSTS_COLUMNS, costs),
    }


def render_run(run_dir: Path, data_dir: Path | None = None) -> None:
    """Write ``README.md``, ``results/``, ``report/`` and ``models/`` of ``run_dir``.

    ``data_dir`` holds ``db/`` and ``artifacts/`` when they are not in the run
    directory (an archived run extracted to a temporary directory, §0.9).
    """
    src = data_dir or run_dir
    conn = sqlite3.connect(str(db_path(src)))
    try:
        rd = tables.load_run_data(run_dir, conn, src)
        results = run_dir / "results"
        report = run_dir / "report"
        results.mkdir(exist_ok=True)
        report.mkdir(exist_ok=True)
        effects = tables.effects_rows(rd)
        labels = tables.labels_rows(rd)
        controls = tables.controls_rows(rd)
        missing = tables.missingness_rows(rd)
        attribution = tables.attribution_rows(rd)
        clusters = tables.clusters_rows(rd)
        trials = tables.trials_rows(rd)
        costs = tables.costs_rows(rd)
        files = {
            "effects.csv": tables.csv_text(tables.EFFECTS_COLUMNS, effects),
            "labels.csv": tables.csv_text(tables.LABELS_COLUMNS, labels),
            "controls.csv": tables.csv_text(tables.CONTROLS_COLUMNS, controls),
            "missingness.csv": tables.csv_text(tables.MISSINGNESS_COLUMNS, missing),
            "attribution.csv": tables.csv_text(tables.ATTRIBUTION_COLUMNS, attribution),
            "clusters.csv": tables.csv_text(tables.CLUSTERS_COLUMNS, clusters),
            "trials.csv": tables.csv_text(tables.TRIALS_COLUMNS, trials),
            "costs.csv": tables.csv_text(tables.COSTS_COLUMNS, costs),
        }
        # consistency.csv and detection.csv for pairwise/ranking and type-D experiments (§32A.3)
        files.update(extras.optional_tables(rd))
        for stale in ("consistency.csv", "detection.csv"):
            if stale not in files and (results / stale).exists():
                (results / stale).unlink()
        for name, text in files.items():
            B.atomic_write_text(results / name, text)
        summ = summary.build_summary(rd)
        B.atomic_write_text(results / "summary.json", B.json_text(summ))
        B.atomic_write_text(
            results / "DATA_DICTIONARY.md", dictionary.render_dictionary(sorted(files))
        )
        md = markdown.render_report(
            rd, summ, effects, controls, missing, attribution, costs, labels
        )
        B.atomic_write_text(report / "report.md", md + "\n")
        figs = svg.render_figures(
            rd, summ, effects, controls, missing, attribution, costs
        )
        fig_dir = report / "figures"
        if fig_dir.exists():
            shutil.rmtree(fig_dir)
        fig_dir.mkdir()
        for name, text in figs.items():
            B.atomic_write_text(fig_dir / name, text)
        B.atomic_write_text(
            report / "report.html",
            html.render_html(
                md,
                figs,
                figure_data(rd, summ, effects, controls, missing, attribution, costs),
            ),
        )
        examples.render_models(rd, clusters)
        B.atomic_write_text(
            run_dir / "README.md", markdown.render_readme(rd, summ) + "\n"
        )
    finally:
        conn.close()
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    B.atomic_write_text(
        logs / "build_info.json",
        json.dumps(
            {
                "rendered_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "host": platform.node(),
                "python": platform.python_version(),
            },
            indent=2,
        )
        + "\n",
    )
