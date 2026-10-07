"""``financial-harness compare RUN_ID… --name NAME`` (spec §32A.7).

Writes ``comparisons/<YYYY-MM-DD>_<name>/`` with ``matrix.csv`` (one row per
model × experiment, each ``labels_row_id`` resolving into that run's
``labels.csv``), ``matrix.svg``, ``comparison.md`` and ``comparison.html``.
Inputs are the runs' rendered ``results/`` files. Runs of one experiment must
share δ, the effect scale and the oracle (exit 2 otherwise). File contents hold
no comparison date, so re-running on the same runs gives byte-identical files;
a second comparison of the same name on the same date exits 2.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import ids
from financial_llm_harness.core.paths import comparisons_dir
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.oracles import glyph, label_class
from financial_llm_harness.reporting import markdown as MD
from financial_llm_harness.reporting import svg as S
from financial_llm_harness.reporting.tables import csv_text

NAME_RE = re.compile(r"^[a-z0-9-]{1,48}$")
MATRIX_COLUMNS = (
    "row_id",
    "model_key",
    "experiment_key",
    "run_id",
    "labels_row_id",
    "zone",
    "sensitivity_label",
    "attribution_label",
    "composite_label",
    "is_bias",
    "is_normative_failure",
)


@dataclass(frozen=True)
class RunView:
    run_dir: Path
    summary: dict[str, Any]
    labels: list[dict[str, str]]

    @property
    def run_id(self) -> str:
        return str(self.summary["run"]["run_id"])

    @property
    def experiment_key(self) -> str:
        return str(self.summary["run"]["experiment_key"])

    @property
    def model_keys(self) -> list[str]:
        return [m["model_key"] for m in self.summary["models"]]

    def design_identity(self) -> dict[str, Any]:
        d = self.summary["design"]
        primary = next(
            (r for r in self.labels if r["feature_id"] == d["primary_feature"]), {}
        )
        return {
            "delta": d["delta"],
            "effect_scale": d["effect_scale"],
            "oracle_type": d["oracle_type"],
            "oracle_confidence": d["oracle_confidence"],
            "oracle_direction": primary.get("oracle_direction"),
        }

    def label_row(self, model_key: str) -> dict[str, str] | None:
        pf = self.summary["design"]["primary_feature"]
        for r in self.labels:
            if r["model_key"] == model_key and r["feature_id"] == pf:
                return r
        return None


def load_view(run_dir: Path) -> RunView:
    results = run_dir / "results"
    if not (results / "summary.json").exists():
        raise HarnessValidationError(
            f"{run_dir.name}: no results/summary.json; render it first"
        )
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    with open(results / "labels.csv", newline="", encoding="utf-8") as fh:
        labels = list(csv.DictReader(fh))
    return RunView(run_dir, summary, labels)


def matrix_rows(
    views: Sequence[RunView],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """(models in first-seen panel order, experiment keys sorted, rows); exit 2 on a mismatch."""
    by_exp: dict[str, list[RunView]] = {}
    for v in views:
        by_exp.setdefault(v.experiment_key, []).append(v)
    for exp, vs in by_exp.items():
        first = vs[0].design_identity()
        for v in vs[1:]:
            if v.design_identity() != first:
                raise HarnessValidationError(
                    f"compare: runs of {exp} differ in δ, effect scale or oracle "
                    f"({first} vs {v.design_identity()}); they are not comparable (§32A.7)"
                )
    models: list[str] = []
    for v in views:
        for mk in v.model_keys:
            if mk not in models:
                models.append(mk)
    experiments = sorted(by_exp)
    rows = []
    for mk in models:
        for exp in experiments:
            holders = [v for v in by_exp[exp] if mk in v.model_keys]
            if len(holders) > 1:
                raise HarnessValidationError(
                    f"compare: model {mk} appears in more than one run of {exp}"
                )
            v = holders[0] if holders else None
            lab = v.label_row(mk) if v is not None else None
            rows.append(
                {
                    "row_id": ids.row_id("matrix", mk, exp, v.run_id if v else ""),
                    "model_key": mk,
                    "experiment_key": exp,
                    "run_id": v.run_id if v else None,
                    "labels_row_id": lab["row_id"] if lab else None,
                    "zone": (lab or {}).get("zone") or None,
                    "sensitivity_label": (lab or {}).get("sensitivity_label") or None,
                    "attribution_label": (lab or {}).get("attribution_label") or None,
                    "composite_label": (lab or {}).get("composite_label") or None,
                    "is_bias": None if lab is None else lab["is_bias"] == "true",
                    "is_normative_failure": None
                    if lab is None
                    else lab["is_normative_failure"] == "true",
                }
            )
    return models, experiments, rows


def _headline(r: Mapping[str, Any]) -> str | None:
    return r.get("composite_label") or r.get("sensitivity_label")


def matrix_svg(
    models: Sequence[str], experiments: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> str:
    """Label matrix: rows = models, columns = experiments; glyph + label; 15% status wash; ink text."""
    cell = {(r["model_key"], r["experiment_key"]): r for r in rows}
    left = S.LEFT
    col_w = (S.WIDTH - left - 16) / max(1, len(experiments))
    c = S.Canvas(len(models), "Label matrix")
    c.text(16, 24, "Label matrix", cls="ink")
    for j, exp in enumerate(experiments):
        c.text(left + j * col_w + 6, 52, exp, cls="ink2", title=f"experiment {exp}")
    for i, mk in enumerate(models):
        y = 60 + i * S.ROW_H
        c.text(16, y + 24, mk, cls="ink2")
        for j, exp in enumerate(experiments):
            r = cell[(mk, exp)]
            lab = _headline(r)
            x = left + j * col_w
            token = S.STATUS_TOKEN[label_class(lab)] if lab else "muted"
            tip = (
                f"{mk} × {exp}: {lab or 'absent'}; run {r['run_id'] or '—'}; labels row_id {r['labels_row_id'] or '—'}; "
                f"matrix row_id {r['row_id']}"
            )
            c.add(
                f'<rect x="{S.r2(x + 2)}" y="{S.r2(y + 4)}" width="{S.r2(col_w - 4)}" height="{S.r2(S.ROW_H - 8)}" '
                f'style="fill:var(--{token});fill-opacity:0.15"><title>{escape(tip)}</title></rect>'
            )
            text = f"{glyph(lab)} {lab}" if lab else "—"
            c.text(x + 8, y + 24, text, cls="ink", title=tip)
    return c.svg()


def comparison_md(
    name: str,
    views: Sequence[RunView],
    models: Sequence[str],
    experiments: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    publishable = all(v.summary["output_policy"]["publishable"] for v in views)
    out = [f"# Comparison {MD.code(name)}", ""]
    out.append(
        f"{len(views)} runs · {len(experiments)} experiments · {len(models)} models · "
        f"{'publishable' if publishable else 'not publishable: includes a non-publishable run'}"
    )
    out += ["", "## 1. Label matrix", "", "![Label matrix](matrix.svg)", ""]
    cell = {(r["model_key"], r["experiment_key"]): r for r in rows}
    table_rows = [
        [MD.code(mk)] + [MD.lab(_headline(cell[(mk, e)])) for e in experiments]
        for mk in models
    ]
    out += [MD.table(["Model", *[MD.code(e) for e in experiments]], table_rows), ""]
    out += ["## 2. Runs", ""]
    run_rows = []
    for v in sorted(views, key=lambda v: (v.experiment_key, v.run_id)):
        run = v.summary["run"]
        d = v.summary["design"]
        run_rows.append(
            [
                MD.code(v.experiment_key),
                MD.code(v.run_id[:8]),
                MD.code(run["manifest_hash"][:8]),
                run["status"],
                MD.num(float(d["delta"]), 2),
                MD.code(d["effect_scale"]),
                MD.code(d["oracle_type"]),
            ]
        )
    out += [
        MD.table(
            ["Experiment", "Run", "Manifest", "Status", "δ", "Effect scale", "Oracle"],
            run_rows,
        ),
        "",
    ]
    out += [
        "## 3. Provenance",
        "",
        "Every cell of `matrix.csv` names the run and the `labels.csv` row it came from; a model absent from an "
        "experiment's runs has an empty label.",
        "",
    ]
    return "\n".join(out)


def compare_runs(
    run_ids: Sequence[str],
    name: str,
    *,
    clock: Callable[[], datetime] | None = None,
) -> Path:
    from financial_llm_harness.execution.runner import resolve_run
    from financial_llm_harness.generation.generate import utc_now
    from financial_llm_harness.reporting import html as H

    if not NAME_RE.match(name):
        raise HarnessValidationError(
            f"compare: name {name!r} must match [a-z0-9-]{{1,48}} (§32A.1)"
        )
    views = [load_view(resolve_run(r)) for r in run_ids]
    if len({v.run_id for v in views}) != len(views):
        raise HarnessValidationError("compare: a run is named twice")
    models, experiments, rows = matrix_rows(views)
    date = (clock or utc_now)().strftime("%Y-%m-%d")
    out = comparisons_dir() / f"{date}_{name}"
    if out.exists():
        raise HarnessValidationError(
            f"compare: {out} exists; a comparison of the same name on the same date is not overwritten"
        )
    out.mkdir(parents=True)
    if not all(v.summary["run"]["data_classification"] == "synthetic" for v in views):
        (out / ".gitignore").write_text("*\n", encoding="utf-8")
    B.atomic_write_text(out / "matrix.csv", csv_text(MATRIX_COLUMNS, rows))
    svg_text = matrix_svg(models, experiments, rows)
    B.atomic_write_text(out / "matrix.svg", svg_text)
    md = comparison_md(name, views, models, experiments, rows)
    B.atomic_write_text(out / "comparison.md", md + "\n")
    data = H.FigureData(
        "matrix.csv",
        MATRIX_COLUMNS,
        [[_cell(r.get(c)) for c in MATRIX_COLUMNS] for r in rows],
    )
    B.atomic_write_text(
        out / "comparison.html",
        H.render_html(md, {"matrix.svg": svg_text}, {"matrix.svg": data}),
    )
    return out


def _cell(v: Any) -> str:
    from financial_llm_harness.reporting.tables import cell

    return cell(v)
