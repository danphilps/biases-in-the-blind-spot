"""Planning pilots (spec §22, §23.6, §27, CR-7): a small discovery run sizes the confirmatory one."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from scipy import stats

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import yaml_io
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.execution import pilot as PI
from financial_llm_harness.execution.runner import run_experiment
from financial_llm_harness.generation.generate import generate
from financial_llm_harness.generation.validate_bundle import validate_bundle
from tests.harness.helpers import (
    FIXED_CODE_REF,
    S1_M2_SPEC,
    S1_SPEC,
    fixed_clock,
)

PILOT_RUN = "0000d100-0000-4000-8000-000000000000"


def _load(path: Path) -> dict:
    data = yaml_io.load(path)
    assert isinstance(data, dict)
    return data


def test_pilot_spec_is_a_discovery_copy_with_controls(tmp_path: Path) -> None:
    out = PI.make_pilot_spec(S1_SPEC, tmp_path / "pilot.yaml", n_cases=40)
    spec, pilot = _load(S1_SPEC), _load(out)
    assert pilot["experiment_key"] == "pilot__" + spec["experiment_key"]
    assert pilot["mode"] == "discovery"
    assert pilot["scenario_spec"]["seed_stream"] == "disc"
    assert pilot["scenario_spec"]["n_cases"] == 40
    assert pilot["controls"]["pilot"] is True
    for key in (
        "hypothesis",
        "intervention_spec",
        "paraphrase_spec",
        "response_spec",
        "feature_spec",
    ):
        assert pilot[key] == spec[key]
    res = generate(out, tmp_path / "b", "ci_fake", code_ref=FIXED_CODE_REF)
    # controls run, on every cluster, though the no-pilot bound would ask for 857
    assert res.manifest["arms"] == ["A", "B", "A0", "PL", "PCA", "PCB"]
    assert res.manifest["controls"]["subset_n"] == res.corpus.n_clusters
    validate_bundle(tmp_path / "b", code_ref=FIXED_CODE_REF)
    with pytest.raises(HarnessValidationError):
        PI.make_pilot_spec(S1_SPEC, tmp_path / "x.yaml", n_cases=10)


def test_pilot_flag_is_refused_in_confirmatory_mode(tmp_path: Path) -> None:
    spec = _load(S1_SPEC)
    spec["controls"]["pilot"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_io.dump(spec), encoding="utf-8")
    with pytest.raises(HarnessValidationError, match="planning pilot") as exc:
        generate(path, tmp_path / "b", "ci_fake", code_ref=FIXED_CODE_REF)
    assert exc.value.exit_code == 2


def test_planning_variance_inflation_and_floor() -> None:
    row = {"n_clusters": "100", "sd_c": "0.3"}
    v, how = PI.planning_variance_from_row(row, 3, binary=True, inflate=False)
    assert v == pytest.approx(0.09)
    v, how = PI.planning_variance_from_row(row, 3, binary=True, inflate=True)
    assert v == pytest.approx(0.09 * 99 / stats.chi2.ppf(0.2, 99))
    assert v > 0.09 and "upper confidence limit" in how
    # no discordance seen: rule-of-three floor 3/(n·k²), never zero
    v, how = PI.planning_variance_from_row(
        {"n_clusters": "100", "sd_c": "0.0"}, 3, True, True
    )
    assert v == pytest.approx(3 / (100 * 9)) and "rule of three" in how


def test_pilot_plans_a_smaller_valid_confirmatory_spec(
    tmp_path: Path, harness_home: Path
) -> None:
    pilot_spec = PI.make_pilot_spec(S1_SPEC, tmp_path / "pilot.yaml", n_cases=60)
    res = run_experiment(
        pilot_spec,
        "ci_fake",
        yes=True,
        run_id=PILOT_RUN,
        clock=fixed_clock,
        code_ref=FIXED_CODE_REF,
    )
    assert res.status == "COMPLETE" and res.run_dir is not None
    out, proj = PI.plan_from_pilot(PILOT_RUN, S1_SPEC, tmp_path / "planned.yaml")
    planned = _load(out)
    refs = planned["statistical_spec"]["planning_ref"]
    sha = B.file_sha256(res.run_dir / "results" / "effects.csv")
    assert {r["contrast"] for r in refs} == {
        "B_vs_A",
        "A0_vs_A",
        "PL_vs_A",
        "PCB_vs_PCA",
    }
    assert all(r["run_id"] == PILOT_RUN and r["source_sha256"] == sha for r in refs)
    assert all(r["variance"] > 0 for r in refs)
    n_cases = planned["scenario_spec"]["n_cases"]
    assert 30 <= n_cases < _load(S1_SPEC)["scenario_spec"]["n_cases"]
    # the planned design passes every §22 / §23.6 power gate at its own N
    generate(out, tmp_path / "b", "ci_fake", code_ref=FIXED_CODE_REF)
    rep = validate_bundle(tmp_path / "b", code_ref=FIXED_CODE_REF)
    assert rep.n_clusters >= rep.n_required and rep.subset_n >= rep.computed_subset_n
    expected_n = math.ceil(1.1 * max(rep.n_required, rep.computed_subset_n, 30))
    assert n_cases >= rep.n_required and n_cases <= expected_n + 1
    for p in proj:
        assert p.pilot_calls > 0 and p.calls + p.pilot_calls < p.calls_before
    # a pilot of one design never plans another
    with pytest.raises(HarnessValidationError, match="pilot of"):
        PI.plan_from_pilot(PILOT_RUN, S1_M2_SPEC, tmp_path / "other.yaml")
