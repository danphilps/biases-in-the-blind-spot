"""``plan`` (spec §30, §31A L4): counts, tokens, cost range and power per contrast."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from financial_llm_harness.core import bundle as B
from financial_llm_harness.execution import budget as BU
from financial_llm_harness.statistics import power

log = logging.getLogger("financial_llm_harness.execution.plan")


def plan_data(bundle_dir: Path) -> dict[str, Any]:
    manifest = B.load_manifest(bundle_dir)
    renderings = B.read_jsonl(bundle_dir / B.RENDERINGS_FILE)
    cases = B.read_jsonl(bundle_dir / B.CASES_FILE)
    report = json.loads((bundle_dir / B.CORPUS_REPORT_FILE).read_text(encoding="utf-8"))
    manifest = dict(manifest)
    manifest["_corpus_report"] = report
    pricing = BU.load_pricing()
    rows = BU.plan_rows(manifest, renderings, pricing)
    ev_rows = []
    for ev in manifest.get("evaluators") or []:
        provider, model = ev["model_key"].split("--")[0], None
        del provider, model
        ev_rows.append(ev["model_key"])
    keys = [
        f"{m['provider']}/{m['model']}" for m in manifest["sampling_spec"]["models"]
    ]
    stale = BU.stale_prices(pricing, keys, datetime.now(UTC).date())
    for k in stale:
        log.warning("pricing.yaml entry %s was checked more than 90 days ago", k)
    n_clusters = int(report["power"]["n_clusters"])
    primary = report["power"]["primary"]
    h = manifest["hypothesis"]
    alpha = float(h["alpha"])
    delta = float(h["delta"])
    contrasts = []
    pv = float(primary["variance"])
    if primary["level"] is not None:
        contrasts.append(
            {
                "contrast": primary["contrast"],
                "n_required": primary["n_required"],
                "variance": pv,
                "source": primary["source"],
                "against": n_clusters,
                "power": power.power_level(n_clusters, pv, delta, alpha),
            }
        )
    else:
        contrasts.append(
            {
                "contrast": primary["contrast"],
                "n_required": primary["n_required"],
                "variance": pv,
                "source": primary["source"],
                "against": n_clusters,
                "power": power.power_superiority(n_clusters, pv, delta, alpha),
            }
        )
    subset_n = int(manifest["controls"].get("subset_n") or 0)
    for c in report["power"]["controls"]:
        contrasts.append(
            {
                "contrast": c["contrast"],
                "n_required": c["n_required"],
                "variance": c["variance"],
                "source": c["source"],
                "against": subset_n,
                "power": None,
            }
        )
    arm_counts: dict[str, int] = {}
    for r in renderings:
        arm_counts[r["arm"]] = arm_counts.get(r["arm"], 0) + 1
    control_trials = sum(
        v for a, v in arm_counts.items() if a in ("A0", "PL", "PCA", "PCB")
    )
    return {
        "experiment_key": manifest["experiment_key"],
        "n_cases": len(cases),
        "n_clusters": n_clusters,
        "arms": manifest["arms"],
        "renderings": len(renderings),
        "renderings_by_arm": arm_counts,
        "control_share": control_trials / len(renderings) if renderings else 0.0,
        "models": [r.__dict__ for r in rows],
        "evaluators": ev_rows,
        "contrasts": contrasts,
        "stale_prices": stale,
        "input_estimate_method": "chars/3 upper bound",
    }


def render_plan(bundle_dir: Path) -> str:
    d = plan_data(bundle_dir)
    lines = [
        f"plan: {d['experiment_key']}",
        f"  cases {d['n_cases']}, clusters {d['n_clusters']}, arms {', '.join(d['arms'])}, renderings {d['renderings']}",
        f"  control arms' share of trials: {d['control_share']:.3f}",
    ]
    for m in d["models"]:
        lines.append(
            f"  {m['model_key']}: trials {m['trials']}, calls {m['calls']}, input tokens ≈ {m['input_tokens_est']} ({d['input_estimate_method']}), "
            f"output tokens ≤ {m['output_tokens_max']}, cost ${m['cost_usd_low']:.2f}–${m['cost_usd_high']:.2f}"
        )
    for c in d["contrasts"]:
        pw = "" if c["power"] is None else f", power at N {c['power']:.3f}"
        lines.append(
            f"  N_required {c['contrast']}: {c['n_required']} (variance {c['variance']}, {c['source']}) against {c['against']}{pw}"
        )
    return "\n".join(lines)
