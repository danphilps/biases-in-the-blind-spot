"""Planning pilots: size a confirmatory experiment from a small discovery run (spec §22, §23.6, §27, CR-7).

Without a pilot every binary contrast is planned at the variance bound σ² = 1.0, so the
A0 and PL equivalence gates alone need 857 clusters at a margin of 0.10. ``pilot make``
writes a discovery-mode copy of a confirmatory spec (same design, disjoint ``disc`` seed
stream, controls on every cluster); ``pilot plan`` reads that run's ``effects.csv`` and
writes the confirmatory spec with ``statistical_spec.planning_ref`` entries and the
``n_cases`` they imply, with the projected calls, tokens and cost.

Each planning variance is the pilot's sd_c², raised to its one-sided 80% upper confidence
limit by default (a small pilot understates σ² about half the time) and floored by a
rule-of-three bound when the pilot saw no discordance. Controls stay at full power: only
the variance that sizes them changes.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy import stats

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.manifest import load_yaml_mapping
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.statistics import power

PILOT_PREFIX = "pilot__"
DEFAULT_PILOT_CASES = 100
UCL_LEVEL = 0.80
N_CASES_HEADROOM = (
    1.1  # §23.6: n_cases ≈ 1.1 × binding N leaves room for cluster merging
)
CONTROL_ARMS = ("A0", "PL", "PCA", "PCB")


# ---------------------------------------------------------------------------
# pilot make
# ---------------------------------------------------------------------------


def make_pilot_spec(
    spec_path: Path, out: Path, n_cases: int = DEFAULT_PILOT_CASES
) -> Path:
    """Write the discovery-mode pilot of a confirmatory spec: same design, ``disc`` stream, n_cases cases."""
    spec = load_yaml_mapping(spec_path)
    if spec["mode"] != "confirmatory":
        raise HarnessValidationError(
            f"{spec_path}: a pilot is made from a confirmatory spec"
        )
    if n_cases < 30:
        raise HarnessValidationError("a pilot needs at least 30 cases (§22 floor)")
    pilot = copy.deepcopy(spec)
    pilot["experiment_key"] = PILOT_PREFIX + spec["experiment_key"]
    pilot["mode"] = "discovery"
    pilot["scenario_spec"]["n_cases"] = int(n_cases)
    pilot["scenario_spec"]["seed_stream"] = (
        "disc"  # cases disjoint from the confirmatory draw (§27)
    )
    pilot["controls"]["subset_n"] = None  # discovery: controls run on every cluster
    pilot["controls"]["pilot"] = True
    pilot["statistical_spec"]["planning_ref"] = []
    out.write_text(yaml_io.dump(pilot), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# pilot plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Projection:
    model_key: str
    calls: int
    calls_before: int
    pilot_calls: int
    input_tokens: float
    output_tokens: float
    cost_usd: float | None
    cost_usd_before: float | None


def planning_variance_from_row(
    row: Mapping[str, str], blocks_per_cluster: int, binary: bool, inflate: bool
) -> tuple[float, str]:
    """(variance, derivation) from one pilot effects.csv row."""
    n = int(row["n_clusters"])
    sd = float(row["sd_c"]) if row["sd_c"] not in ("", None) else 0.0
    v = sd * sd
    how = f"pilot sd_c² = {v:.6g} over {n} clusters"
    if inflate and n > 1:
        factor = (n - 1) / float(stats.chi2.ppf(1 - UCL_LEVEL, n - 1))
        v *= factor
        how += f"; × (n−1)/χ²_{{{1 - UCL_LEVEL:.2f},n−1}} = {factor:.4f} (one-sided {UCL_LEVEL:.0%} upper confidence limit)"
    if binary:
        floor = 3.0 / (n * blocks_per_cluster * blocks_per_cluster)
        if v < floor:
            v = floor
            how += f"; floored at 3/(n·k²) = {floor:.6g}, k = {blocks_per_cluster} blocks per cluster (rule of three)"
    return v, how


def _read_effects(run_dir: Path) -> tuple[list[dict[str, str]], str]:
    path = run_dir / "results" / "effects.csv"
    if not path.is_file():
        raise HarnessValidationError(f"{path}: the pilot run has no effects.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return rows, B.file_sha256(path)


def _token_means(run_dir: Path) -> dict[str, tuple[float, float, int]]:
    """model_key → (mean input tokens, mean output tokens incl. reasoning, responses) over the pilot's responses."""
    from financial_llm_harness.reporting.replay import ensure_db
    from financial_llm_harness.storage.db import db_path

    out: dict[str, tuple[float, float, int]] = {}
    with ensure_db(run_dir, into_run_dir=False) as data_dir:
        conn = sqlite3.connect(str(db_path(data_dir)))
        try:
            for mk, i, o, n in conn.execute(
                "SELECT t.model_key, AVG(COALESCE(r.input_tokens, 0)), "
                "AVG(COALESCE(r.output_tokens, 0) + COALESCE(r.reasoning_tokens, 0)), COUNT(*) "
                "FROM response r JOIN trial t ON t.trial_id = r.trial_id GROUP BY t.model_key"
            ):
                out[str(mk)] = (float(i or 0.0), float(o or 0.0), int(n))
        finally:
            conn.close()
    return out


def projected_calls(
    manifest_like: Mapping[str, Any], n_clusters: int, subset_n: int, arms: list[str]
) -> int:
    """Trials per model: main arms on every cluster, control arms on the subset, P·R blocks each (§18, §22)."""
    k = int(manifest_like["paraphrase_spec"]["P"]) * int(
        manifest_like["sampling_spec"]["R"]
    )
    main = sum(1 for a in arms if a not in CONTROL_ARMS)
    ctrl = sum(1 for a in arms if a in CONTROL_ARMS)
    return k * (main * n_clusters + ctrl * subset_n)


def requirements(spec: Mapping[str, Any], model_key: str | None) -> tuple[int, int]:
    """(primary N_required, computed subset_n) for one model's planning entries (§22, §23.6)."""
    primary = next(f for f in spec["feature_spec"] if f.get("primary") is True)
    st = spec["statistical_spec"]
    req = power.primary_requirement(
        spec["oracle_spec"]["type"],
        spec["hypothesis"],
        st,
        primary,
        spec["oracle_spec"].get("tol"),
        model_key,
    )
    arms = ["A", "B", "A0", "PL", "PCA", "PCB"]
    ctrl = power.control_requirements(
        arms, spec["controls"], spec["hypothesis"], st, primary, model_key
    )
    return req.n_required, power.computed_subset_n(ctrl)


def plan_from_pilot(
    pilot_run_id: str, spec_path: Path, out: Path, *, inflate: bool = True
) -> tuple[Path, list[Projection]]:
    """Write ``out``: the confirmatory spec planned from the pilot run; return the per-model projection."""
    from financial_llm_harness.execution import budget as BU
    from financial_llm_harness.execution.runner import resolve_run

    spec = load_yaml_mapping(spec_path)
    run_dir = resolve_run(pilot_run_id)
    pilot = B.load_manifest(run_dir / "manifest")
    if pilot["mode"] != "discovery":
        raise HarnessValidationError(
            "planning_ref must come from a discovery pilot, never confirmatory data (§23.6)"
        )
    if pilot["experiment_key"] != PILOT_PREFIX + spec["experiment_key"]:
        raise HarnessValidationError(
            f"run {pilot_run_id} is the pilot of {pilot['experiment_key']!r}, not of {spec['experiment_key']!r}"
        )
    for key in (
        "hypothesis",
        "intervention_spec",
        "controls",
        "paraphrase_spec",
        "response_spec",
        "feature_spec",
    ):
        a, b = copy.deepcopy(pilot[key]), copy.deepcopy(spec[key])
        for c in (a, b):  # keys `generate` fills or the pilot alone sets
            if isinstance(c, dict):
                for k in (
                    "subset_n",
                    "pilot",
                    "enforcement_resolved",
                    "enforcement_strata",
                ):
                    c.pop(k, None)
        if a != b:
            raise HarnessValidationError(
                f"the pilot's {key} differs from the spec's: not a pilot of the same design (§22)"
            )
    run_id = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["run_id"]
    rows, sha = _read_effects(run_dir)
    h = spec["hypothesis"]
    pf = h["primary_feature"]
    primary = next(f for f in spec["feature_spec"] if f["feature_id"] == pf)
    binary = primary.get("type") == "binary"
    main_contrast = (
        f"level_{h['level_arm']}" if h["estimand"] == "arm_level" else "B_vs_A"
    )
    wanted = (main_contrast, "A0_vs_A", "PL_vs_A", "PCB_vs_PCA")
    k = int(spec["paraphrase_spec"]["P"]) * int(spec["sampling_spec"]["R"])
    refs: list[dict[str, Any]] = []
    models = [m["model_key"] for m in pilot["sampling_spec"]["models"]]
    for mk in models:
        for contrast in wanted:
            row = next(
                (
                    r
                    for r in rows
                    if r["model_key"] == mk
                    and r["feature_id"] == pf
                    and r["contrast"] == contrast
                ),
                None,
            )
            if row is None:
                raise HarnessValidationError(
                    f"pilot effects.csv has no {pf} {contrast} row for {mk}"
                )
            v, how = planning_variance_from_row(row, k, binary, inflate)
            refs.append(
                {
                    "run_id": run_id,
                    "model_key": mk,
                    "feature_id": pf,
                    "contrast": contrast,
                    "variance": float(f"{v:.6g}"),
                    "source_row_id": row["row_id"],
                    "source_sha256": sha,
                    "derivation": how,
                }
            )
    planned = copy.deepcopy(spec)
    planned["statistical_spec"]["planning_ref"] = refs
    need = [requirements(planned, mk) for mk in models]
    n_prim = max(n for n, _ in need)
    subset = max(s for _, s in need)
    n_cases = math.ceil(N_CASES_HEADROOM * max(n_prim, subset, 30))
    planned["scenario_spec"]["n_cases"] = n_cases
    planned["controls"]["subset_n"] = None
    out.write_text(yaml_io.dump(planned), encoding="utf-8")

    arms = list(pilot["arms"])
    n0, s0 = requirements(spec, None)
    before_clusters = int(spec["scenario_spec"]["n_cases"])
    before_subset = min(s0, before_clusters)
    tokens = _token_means(run_dir)
    pricing = BU.load_pricing()
    proj: list[Projection] = []
    for m in pilot["sampling_spec"]["models"]:
        mk = m["model_key"]
        calls = projected_calls(spec, n_cases, subset, arms)
        calls0 = projected_calls(spec, before_clusters, before_subset, arms)
        i, o, n_pilot = tokens.get(mk, (0.0, 0.0, 0))
        price = BU.price_for(pricing, m["provider"], m["model"])
        per_call = (
            None
            if price is None
            else price.cost(int(round(i)), int(round(o)), batch=bool(m.get("batch")))
        )
        proj.append(
            Projection(
                mk,
                calls,
                calls0,
                n_pilot,
                i,
                o,
                None if per_call is None else per_call * calls,
                None if per_call is None else per_call * calls0,
            )
        )
    return out, proj


def projection_text(spec_out: Path, proj: list[Projection]) -> str:
    lines = [f"wrote {spec_out}"]
    for p in proj:
        cost = (
            ""
            if p.cost_usd is None or p.cost_usd_before is None
            else f", cost ≈ ${p.cost_usd:,.2f} (was ${p.cost_usd_before:,.2f} at the no-pilot bound)"
        )
        lines.append(
            f"  {p.model_key}: calls {p.calls:,} + pilot {p.pilot_calls:,} (was {p.calls_before:,}, "
            f"−{1 - (p.calls + p.pilot_calls) / p.calls_before:.0%}); "
            f"pilot mean tokens/call in {p.input_tokens:.0f}, out {p.output_tokens:.0f}{cost}"
        )
    return "\n".join(lines)
