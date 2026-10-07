"""L13: ``runs/INDEX.md`` and ``runs/index.json``, one row per run, newest first (spec §32A.7)."""

from __future__ import annotations

import json
from typing import Any

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core.paths import runs_dir


def index_rows() -> list[dict[str, Any]]:
    root = runs_dir()
    rows = []
    if not root.is_dir():
        return rows
    for d in sorted(root.iterdir()):
        s = d / "results" / "summary.json"
        if not s.exists():
            continue
        summ = json.loads(s.read_text(encoding="utf-8"))
        run = summ["run"]
        publishable = bool(summ.get("output_policy", {}).get("publishable", True))
        rows.append(
            {
                "date": (run.get("started_at") or "")[:10],
                "started_at": run.get("started_at") or "",
                "run_id": run["run_id"],
                "path": d.name,
                "experiment_key": run["experiment_key"],
                "mode": run["mode"],
                "status": run["status"],
                "publishable": publishable,
                "models": [
                    {
                        "model_key": m["model_key"],
                        "headline_label": (
                            m["primary"].get("composite_label")
                            or m["primary"].get("sensitivity_label")
                        )
                        if publishable
                        else None,
                    }
                    for m in summ["models"]
                ],
            }
        )
    rows.sort(key=lambda r: (r["started_at"], r["run_id"]), reverse=True)
    return rows


def write_index() -> None:
    root = runs_dir()
    rows = index_rows()
    if not root.is_dir():
        return
    B.atomic_write_text(
        root / "index.json", B.json_text({"schema_version": "1.0", "runs": rows})
    )
    lines = [
        "# Runs",
        "",
        "| Date | Run | Experiment | Mode | Models | Status | Publishable | Headline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        heads = "; ".join(
            f"`{m['model_key']}` {m['headline_label'] or '—'}" for m in r["models"]
        )
        models = ", ".join(f"`{m['model_key']}`" for m in r["models"])
        lines.append(
            f"| {r['date']} | [`{r['run_id'][:8]}`]({r['path']}/README.md) | `{r['experiment_key']}` | {r['mode']} | {models} | {r['status']} | "
            f"{'true' if r['publishable'] else 'false'} | {heads} |"
        )
    B.atomic_write_text(root / "INDEX.md", "\n".join(lines) + "\n")
