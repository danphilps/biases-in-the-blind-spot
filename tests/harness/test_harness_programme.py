"""Programme (spec §32B, §25, §37, §36)."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from financial_llm_harness.core import yaml_io
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.execution import programme as PG
from financial_llm_harness.reporting import programme as RP
from tests.harness.helpers import (
    F1_SPECS,
    FAKE_A,
    FAKE_B,
    FIXED_CODE_REF,
    ROOT,
    small_spec,
)

CI_PROGRAMME = ROOT / "experiments" / "financial" / "programmes" / "ci_fake.yaml"
PROGRAMME_RUN_ID = "99999999-0000-4000-8000-000000000001"
M1_KEY = "f1_1_a__m1__i_f_momentum.v3"
M2_KEY = "f1_1_a__m2__i_f_momentum.v3"
RUN_IDS = {
    M1_KEY: "aaaaaaaa-0000-4000-8000-000000000001",
    M2_KEY: "bbbbbbbb-0000-4000-8000-000000000002",
}


def clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _outputs(pdir: Path) -> dict[str, bytes]:
    out = {}
    for part in ("README.md", "state.json", "results", "report"):
        p = pdir / part
        if p.is_file():
            out[part] = p.read_bytes()
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    out[f.relative_to(pdir).as_posix()] = f.read_bytes()
    return out


@pytest.fixture(scope="module")
def ci_programme(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The §37 CI programme at the S1 size, with injected identity."""
    import os

    home = tmp_path_factory.mktemp("programme_home")
    old = os.environ.get("HARNESS_HOME")
    os.environ["HARNESS_HOME"] = str(home)
    try:
        res = PG.run_programme(
            CI_PROGRAMME,
            yes=True,
            programme_run_id=PROGRAMME_RUN_ID,
            clock=clock,
            run_id_for=RUN_IDS.__getitem__,
            code_ref=FIXED_CODE_REF,
        )
        assert res.exit_code == 0, res.message
        assert res.programme_dir is not None
        return res.programme_dir
    finally:
        if old is None:
            os.environ.pop("HARNESS_HOME", None)
        else:
            os.environ["HARNESS_HOME"] = old


def test_programme_acceptance(ci_programme: Path) -> None:
    pdir = ci_programme
    assert pdir.name == "2026-01-01_ci-fake_99999999"
    state = json.loads((pdir / "state.json").read_text())
    assert [(e["experiment_key"], e["status"]) for e in state["experiments"]] == [
        (M1_KEY, "COMPLETE"),
        (M2_KEY, "COMPLETE"),
    ]
    answers = _csv(pdir / "results" / "answers.csv")
    fdr = _csv(pdir / "results" / "fdr.csv")
    assert len(answers) == 4 and len(fdr) == 8
    assert all(r["m"] == "2" for r in fdr)
    # no BH-derived confidence level: θ̂, CI, run zone and run label are the run's own
    home = pdir.parent.parent
    for a in answers:
        run_dir = next((home / "runs").glob(f"*_{a['run_id'][:8]}"))
        eff = {
            (r["model_key"], r["feature_id"], r["contrast"]): r
            for r in _csv(run_dir / "results" / "effects.csv")
        }[(a["model_key"], "decision_attractive", "B_vs_A")]
        lab = {r["row_id"]: r for r in _csv(run_dir / "results" / "labels.csv")}[
            a["run_label_row_id"]
        ]
        for col in ("theta", "ci_low", "ci_high", "p_dev", "p_eq"):
            assert a[col] == eff[col], col
        assert (
            a["run_zone"] == lab["zone"] and a["run_label"] == lab["sensitivity_label"]
        )
    # §37 labels, identical in the run and programme columns
    got = {
        (a["variant_id"], a["model_key"]): (
            a["run_zone"],
            a["programme_zone"],
            a["run_label"],
            a["programme_label"],
            a["composite_label"],
            a["is_bias"],
            a["is_normative_failure"],
        )
        for a in answers
    }
    assert got == {
        ("m1", FAKE_A): (
            "Z-",
            "Z-",
            "JUSTIFIED_SENSITIVITY",
            "JUSTIFIED_SENSITIVITY",
            "UNVERBALISED_DEPENDENCY",
            "false",
            "false",
        ),
        ("m1", FAKE_B): (
            "Z0",
            "Z0",
            "UNDER_SENSITIVITY",
            "UNDER_SENSITIVITY",
            "",
            "false",
            "true",
        ),
        ("m2", FAKE_A): (
            "Z-",
            "Z-",
            "SPURIOUS_SENSITIVITY",
            "SPURIOUS_SENSITIVITY",
            "UNVERBALISED_BIAS",
            "true",
            "true",
        ),
        ("m2", FAKE_B): (
            "Z0",
            "Z0",
            "INVARIANT_AS_REQUIRED",
            "INVARIANT_AS_REQUIRED",
            "",
            "false",
            "false",
        ),
    }
    assert all(
        a["attribution_label"].startswith("UNVERBALISED")
        for a in answers
        if a["model_key"] == FAKE_A
    )
    # one verdict per question and model (D-20)
    qa = {r["model_key"]: r for r in _csv(pdir / "results" / "question_answers.csv")}
    assert len(qa) == 2
    assert (
        qa[FAKE_A]["verdict_class"],
        qa[FAKE_A]["verdict_label"],
        qa[FAKE_A]["verdict_variant_id"],
    ) == (
        "critical",
        "UNVERBALISED_BIAS",
        "m2",
    )
    assert (
        qa[FAKE_B]["verdict_class"],
        qa[FAKE_B]["verdict_label"],
        qa[FAKE_B]["verdict_variant_id"],
    ) == (
        "warning",
        "UNDER_SENSITIVITY",
        "m1",
    )
    for name in (
        "consistency.csv",
        "runs.csv",
        "plan.csv",
        "coverage.csv",
        "summary.json",
    ):
        assert (pdir / "results" / name).exists(), name
    assert (pdir / "report" / "figures" / "01_answer_matrix.svg").exists()
    md = (pdir / "report" / "programme.md").read_text()
    for h in (
        "## 1. Answer matrix",
        "## 2. Answers by question",
        "## 3. Multiple testing",
        "## 4. Coverage and gaps",
        "## 5. Runs and cost",
        "## 6. Provenance",
    ):
        assert h in md


def test_programme_report_rerun_is_byte_identical(
    ci_programme: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HARNESS_HOME", str(ci_programme.parent.parent))
    before = _outputs(ci_programme)
    code, _ = PG.cli_report(CI_PROGRAMME, PROGRAMME_RUN_ID[:8])
    assert code == 0
    after = _outputs(ci_programme)
    assert [k for k in before if before[k] != after.get(k)] == []


def test_programme_results_validate_against_schemas(ci_programme: Path) -> None:
    import jsonschema

    from financial_llm_harness.reporting.dictionary import load_schema

    results = ci_programme / "results"
    for name in (
        "answers.csv",
        "question_answers.csv",
        "fdr.csv",
        "runs.csv",
        "plan.csv",
        "coverage.csv",
        "consistency.csv",
        "cross_variant.csv",
    ):
        schema = load_schema(f"programme_{name}")
        jsonschema.Draft202012Validator.check_schema(schema)
        v = jsonschema.Draft202012Validator(schema)
        with open(results / name, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            assert header == schema["x-columns"], name
            from tests.harness.test_harness_render import _convert

            for row in reader:
                v.validate(
                    {
                        c: _convert(x, schema["properties"][c])
                        for c, x in zip(header, row, strict=True)
                    }
                )
    schema = load_schema("programme_summary.json")
    jsonschema.Draft202012Validator(schema).validate(
        json.loads((results / "summary.json").read_text())
    )


# ---------------------------------------------------------------------------
# Small programmes: resume, budget, panel change, selection rules
# ---------------------------------------------------------------------------


def _small_programme(
    tmp: Path, budget_calls: int = 100_000, name: str = "p.yaml"
) -> Path:
    s1 = small_spec(tmp, name="m1.yaml", base=F1_SPECS["m1"])
    s2 = small_spec(tmp, name="m2.yaml", base=F1_SPECS["m2"])
    panel = tmp / "panel.yaml"
    shutil.copyfile(
        ROOT / "experiments" / "financial" / "panels" / "ci_fake.yaml", panel
    )
    prog = {
        "schema_version": 1,
        "programme_key": "small_fake",
        "panel": str(panel),
        "select": {"specs": [str(s1), str(s2)], "generated": None},
        "budget": {"max_cost_usd": 1.0, "max_calls": budget_calls},
        "on_experiment_failure": "continue",
    }
    path = tmp / name
    path.write_text(yaml_io.dump(prog), encoding="utf-8")
    return path


def _run(path: Path, **kw: Any) -> PG.ProgrammeResult:
    return PG.run_programme(
        path,
        yes=True,
        clock=clock,
        run_id_for=RUN_IDS.__getitem__,
        code_ref=FIXED_CODE_REF,
        **kw,
    )


def test_resume_after_injected_stop_equals_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prog = _small_programme(tmp_path)
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "full"))
    full = _run(prog, programme_run_id=PROGRAMME_RUN_ID)
    assert full.exit_code == 0 and full.programme_dir is not None
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "cut"))
    with pytest.raises(PG.InjectedProgrammeStop):
        _run(prog, programme_run_id=PROGRAMME_RUN_ID, stop_after=1)
    res = _run(prog, resume=PROGRAMME_RUN_ID[:8])
    assert res.exit_code == 0 and res.programme_dir is not None
    a, b = _outputs(full.programme_dir), _outputs(res.programme_dir)
    assert sorted(a) == sorted(b)
    assert [k for k in a if a[k] != b[k]] == []


def test_programme_budget_below_plan_exits_3_before_any_call(
    tmp_path: Path, harness_home: Path
) -> None:
    prog = _small_programme(tmp_path, budget_calls=10)
    res = _run(prog, programme_run_id=PROGRAMME_RUN_ID)
    assert res.exit_code == 3
    assert not (harness_home / "runs").exists()


def test_changed_panel_makes_resume_exit_2(tmp_path: Path, harness_home: Path) -> None:
    prog = _small_programme(tmp_path)
    with pytest.raises(PG.InjectedProgrammeStop):
        _run(prog, programme_run_id=PROGRAMME_RUN_ID, stop_after=1)
    panel = tmp_path / "panel.yaml"
    panel.write_text(panel.read_text() + "\n# edited\n")
    with pytest.raises(HarnessValidationError) as exc:
        _run(prog, resume=PROGRAMME_RUN_ID[:8])
    assert exc.value.exit_code == 2


def test_selection_rules_exit_2(tmp_path: Path) -> None:
    s1 = small_spec(tmp_path, name="a.yaml", base=F1_SPECS["m1"])
    dup = small_spec(
        tmp_path, name="b.yaml", base=F1_SPECS["m1"]
    )  # same experiment_key
    pair = small_spec(
        tmp_path,
        name="c.yaml",
        base=F1_SPECS["m1"],
        experiment_key="f1_1_a__m1__i_f_momentum.v9",
    )  # same pair
    for specs in (
        [str(s1), str(dup)],
        [str(s1), str(pair)],
        [str(tmp_path / "missing.yaml")],
        [],
    ):
        prog = {
            "schema_version": 1,
            "programme_key": "sel",
            "panel": "ci_fake",
            "select": {"specs": specs, "generated": None},
            "budget": {"max_cost_usd": 1.0, "max_calls": 10},
        }
        path = tmp_path / "sel.yaml"
        path.write_text(yaml_io.dump(prog))
        with pytest.raises(HarnessValidationError) as exc:
            PG.resolve_selection(PG.load_programme(path))
        assert exc.value.exit_code == 2
    # a path listed twice is de-duplicated, not an error
    prog["select"] = {"specs": [str(s1), str(s1)], "generated": None}
    path.write_text(yaml_io.dump(prog))
    assert len(PG.resolve_selection(PG.load_programme(path))) == 1


def test_programme_file_rules_exit_2(tmp_path: Path) -> None:
    base = yaml_io.load(CI_PROGRAMME)
    assert isinstance(base, dict)
    for mutate in (
        lambda d: d.update(programme_key="Bad-Key"),
        lambda d: d["budget"].update(max_calls=None),
        lambda d: d.update(on_experiment_failure="retry"),
    ):
        d = json.loads(json.dumps(base))
        mutate(d)
        path = tmp_path / "x.yaml"
        path.write_text(yaml_io.dump(d))
        with pytest.raises(HarnessValidationError):
            PG.load_programme(path)


def test_question_verdict_rule_on_hand_built_answers() -> None:
    def a(
        variant: str,
        label: str,
        comp: str | None = None,
        status: str = "COMPLETE",
        bias: bool = False,
    ) -> dict[str, Any]:
        return {
            "row_id": f"r-{variant}-{label}",
            "faq_id": "F9.9.z",
            "variant_id": variant,
            "intervention_id": "I.X",
            "model_key": "m",
            "run_status": status,
            "programme_label": label,
            "composite_label": comp,
            "is_bias": bias,
        }

    rows = [
        a("v1", "INVARIANT_AS_REQUIRED"),
        a("v2", "UNDER_SENSITIVITY"),
        a("v3", "SPURIOUS_SENSITIVITY", "UNVERBALISED_BIAS", bias=True),
        a("v4", "PERVERSE_SENSITIVITY", bias=True),
        a("v5", "OVER_SENSITIVITY", status="INCOMPLETE"),
    ]
    (q,) = RP.question_rows(rows, ["m"])
    assert (q["n_experiments"], q["n_complete"]) == (5, 4)
    assert (q["n_good"], q["n_warning"], q["n_serious"], q["n_critical"]) == (
        1,
        1,
        0,
        2,
    )
    assert q["any_bias"] is True
    # the first row of the most severe class, in answers.csv order; a composite replaces its label
    assert (q["verdict_class"], q["verdict_label"], q["verdict_variant_id"]) == (
        "critical",
        "UNVERBALISED_BIAS",
        "v3",
    )
    (none,) = RP.question_rows(
        [a("v1", "INDETERMINATE", status="SKIPPED_INVALID")], ["m"]
    )
    assert (none["verdict_class"], none["verdict_label"]) == (
        "neutral",
        "INDETERMINATE",
    )


def test_programme_plan_writes_nothing(tmp_path: Path, harness_home: Path) -> None:
    prog = _small_programme(tmp_path)
    code, text = PG.cli_plan(prog)
    assert code == 0 and "total calls" in text
    assert (
        not (harness_home / "programmes").exists()
        and not (harness_home / "runs").exists()
    )
    res = PG.run_programme(prog, yes=False)
    assert res.exit_code == 0 and res.programme_dir is None


def test_canonical_specs_are_paired() -> None:
    m0 = yaml_io.load(F1_SPECS["m0"])
    m1 = yaml_io.load(F1_SPECS["m1"])
    m2 = yaml_io.load(F1_SPECS["m2"])
    assert isinstance(m0, dict) and isinstance(m1, dict) and isinstance(m2, dict)
    # the three arms share their cases (§35 S1): same scenario, seeds and seed stream
    assert m0["scenario_spec"] == m1["scenario_spec"] == m2["scenario_spec"]
    assert m0["seeds"] == m1["seeds"] == m2["seeds"]
    assert (m0["mandate"]["id"], m0["oracle_spec"]["type"], m0["mode"]) == (
        "M0",
        "DISCOVERY",
        "discovery",
    )
    assert (m2["experiment_key"], m2["variant_id"], m2["oracle_spec"]["type"]) == (
        M2_KEY,
        "m2",
        "INVARIANCE",
    )


def test_cross_variant_contrast_is_matched_by_cluster(ci_programme: Path) -> None:
    """CR-14: γ = θ_m2 − θ_m1 from the two runs' matched cluster outcomes."""
    rows = _csv(ci_programme / "results" / "cross_variant.csv")
    assert rows, "the m1 and m2 runs share their cases, so γ is reported"
    for r in rows:
        assert (r["faq_id"], r["variant_ref"], r["variant_x"]) == ("F1.1.a", "m1", "m2")
        assert r["analysis_role"] == "exploratory"
        assert int(r["n_clusters"]) > 30
        gamma = float(r["gamma"])
        assert abs(gamma - (float(r["theta_x"]) - float(r["theta_ref"]))) < 1e-9
        assert float(r["ci_low"]) <= gamma <= float(r["ci_high"])


def test_cross_variant_needs_identical_cases() -> None:
    import numpy as np

    def exp(key: str, variant: str, fp: str, xs: dict[int, float]) -> Any:
        manifest = {
            "faq_id": "F1.1.a",
            "intervention_id": "I.F.momentum",
            "variant_id": variant,
            "hypothesis": {
                "estimand": "mean_paired_difference",
                "primary_feature": "decision_attractive",
                "alpha": 0.05,
            },
        }
        return RP.Experiment(
            key,
            manifest,
            {"status": "COMPLETE"},
            Path("."),
            {},
            {},
            {},
            0,
            0.0,
            {"m": xs},
            fp,
        )

    rng = np.random.default_rng(0)
    base = {c: float(v) for c, v in enumerate(rng.normal(0.0, 0.3, 60))}
    shifted = {c: v - 0.2 for c, v in base.items()}
    order = {"F1.1.a": ["m0", "m1", "m2"]}
    rows = RP.cross_variant_rows(
        [exp("k2", "m2", "same", shifted), exp("k0", "m0", "same", base)], ["m"], order
    )
    assert len(rows) == 1 and rows[0]["variant_ref"] == "m0"
    assert abs(rows[0]["gamma"] + 0.2) < 1e-12 and rows[0]["ci_high"] < 0
    other_cases = [exp("k2", "m2", "other", shifted), exp("k0", "m0", "same", base)]
    assert RP.cross_variant_rows(other_cases, ["m"], order) == []
