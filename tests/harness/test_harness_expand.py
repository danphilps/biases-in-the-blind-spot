"""``taxonomy expand`` (spec §32B.2, D-15, D-19, M8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from financial_llm_harness import families
from financial_llm_harness.cli import main
from financial_llm_harness.core import yaml_io
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.taxonomy import expand as X
from financial_llm_harness.taxonomy import registry as R

V1_FILES = {
    "F1.1.a/m1/I.F.valuation.yaml",
    "F1.1.a/m1/I.F.leverage.yaml",
    "F1.1.a/m2/I.F.valuation.yaml",
    "F1.1.a/m2/I.F.leverage.yaml",
    *(
        f"F1.1.a/{v}/{c}.yaml"
        for v in ("m1", "m2")
        for c in ("I.ID.name", "I.ID.ceo", "I.USER.stance", "I.PRES.order")
    ),
    "F6.1.b/base/I.PRES.distractor.yaml",
    "Q3.1.a/base/I.HYPE.label.yaml",
    "Q3.1.a/base/I.USER.stance.yaml",
    "M6.4.a/base/I.USER.stance.yaml",
    "K.M.1/base/I.CODE.comment.yaml",
    "K.M.1/base/I.HYPE.label.yaml",
}


@pytest.fixture
def gen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Expand into a temporary generated/ directory."""
    monkeypatch.setattr(R, "generated_dir", lambda: tmp_path)
    monkeypatch.setattr(X, "generated_dir", lambda: tmp_path)
    return tmp_path


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*.yaml")}


def _report(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / X.REPORT_NAME).read_text(encoding="utf-8"))


def test_expand_v1_writes_every_pair_and_reports_none(gen: Path) -> None:
    assert main(["taxonomy", "expand", "--wave", "V1"]) == 0
    assert _files(gen) == V1_FILES
    report = _report(gen)
    assert not [r for r in report if r["wave"] == "V1"]
    # canonical pairs are never generated nor reported (§32B.2 step 1b); K.M.1 has no I.NONE (skip rule)
    keys = {(r["faq_id"], r["intervention_id"]) for r in report}
    assert ("K.M.1", "I.NONE") not in keys
    for rel in V1_FILES:
        spec = yaml_io.load(gen / rel)
        faq, vid, fname = rel.split("/")
        iid = fname[: -len(".yaml")]
        assert spec["experiment_key"] == R.experiment_key(faq, vid, iid, 1)
        assert (spec["faq_id"], spec["variant_id"], spec["intervention_id"]) == (
            faq,
            vid,
            iid,
        )
        assert spec["sampling_spec"]["panel"] == "ci_fake"
        assert spec["hypothesis"]["source"] == "a_priori"
        assert spec["parent_discovery_experiment_id"] is None
    # the registry still validates with the generated files in place (no orphans)
    R.validate_registry()


def test_sibling_signs_modes_and_mandates(gen: Path) -> None:
    assert main(["taxonomy", "expand", "--item", "F1.1.a"]) == 0
    m1v = yaml_io.load(gen / "F1.1.a/m1/I.F.valuation.yaml")
    # item sign − per unit increase of P/E, concept_direction +1 → s = −1 (§32B.2 step 4)
    assert m1v["oracle_spec"]["type"] == "DIRECTIONAL"
    assert m1v["oracle_spec"]["sign"] == -1 and m1v["oracle_spec"]["item_sign"] == -1
    assert m1v["hypothesis"]["direction"] == "negative"
    assert (
        m1v["mode"] == "confirmatory"
        and m1v["hypothesis"]["correction_family"] == "programme"
    )
    assert m1v["mandate"]["id"] == "M1" and m1v["mandate"]["text"]
    assert m1v["intervention_spec"]["node"] == "pe" and m1v["concept"] == "valuation"
    m1l = yaml_io.load(gen / "F1.1.a/m1/I.F.leverage.yaml")
    # B.conv is conventional and its checklist is not signed off: discovery (§T10.6, D-24)
    assert m1l["mode"] == "discovery" and m1l["scenario_spec"]["seed_stream"] == "disc"
    assert m1l["hypothesis"]["correction_family"] == "exploratory"
    m2l = yaml_io.load(gen / "F1.1.a/m2/I.F.leverage.yaml")
    assert m2l["oracle_spec"]["type"] == "DISCOVERY" and m2l["mode"] == "discovery"
    assert m2l["mandate"]["id"] == "M2"
    m2v = yaml_io.load(gen / "F1.1.a/m2/I.F.valuation.yaml")
    assert m2v["oracle_spec"]["mandate_id"] == "M2" and m2v["oracle_spec"]["sign"] == -1


def test_report_covers_the_whole_registry(gen: Path) -> None:
    assert main(["taxonomy", "expand", "--item", "F1.1.a"]) == 0
    report = _report(gen)
    assert report == sorted(
        report, key=lambda r: (r["faq_id"], r["variant_id"], r["intervention_id"])
    )
    v11 = [r for r in report if r["wave"] == "V1.1"]
    assert v11 and {r["reason"] for r in v11} == {"item_not_implemented"}
    assert ("F3.6.a", "base", "I.NONE") in {
        (r["faq_id"], r["variant_id"], r["intervention_id"]) for r in v11
    }  # correctness pair of an X item (D-19)
    # a wave whose items are all specified exits 2 unless --allow-partial
    assert main(["taxonomy", "expand", "--wave", "V1.1"]) == 2
    assert main(["taxonomy", "expand", "--wave", "V1.1", "--allow-partial"]) == 0


def test_missing_content_is_reported_never_invented(
    gen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = families.load_defaults

    def without_valuation(fid: str) -> dict[str, Any]:
        d = real(fid)
        if fid != "fundamental_judgement":
            return d
        import copy

        d = copy.deepcopy(d)
        del d["entries"]["F1.1.a"]["variants"]["m2"]["I.F.valuation"]
        return d

    monkeypatch.setattr(families, "load_defaults", without_valuation)
    monkeypatch.setattr(X, "load_defaults", without_valuation)
    assert main(["taxonomy", "expand", "--item", "F1.1.a"]) == 2
    rows = [r for r in _report(gen) if r["faq_id"] == "F1.1.a"]
    assert rows == [
        {
            "faq_id": "F1.1.a",
            "variant_id": "m2",
            "intervention_id": "I.F.valuation",
            "wave": "V1",
            "reason": "unimplemented_content",
            "missing": [
                "families/fundamental_judgement/defaults.yaml#F1.1.a/I.F.valuation"
            ],
        }
    ]
    assert not (gen / "F1.1.a/m2/I.F.valuation.yaml").exists()
    assert main(["taxonomy", "expand", "--item", "F1.1.a", "--allow-partial"]) == 0


def test_version_immutability(gen: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert main(["taxonomy", "expand", "--item", "Q3.1.a"]) == 0
    path = gen / "Q3.1.a/base/I.USER.stance.yaml"
    first = path.read_bytes()
    assert main(["taxonomy", "expand", "--item", "Q3.1.a"]) == 0
    assert path.read_bytes() == first  # deterministic
    data = yaml_io.load(path)
    data["research_question"] = "edited by hand"
    path.write_text(yaml_io.dump(data), encoding="utf-8")
    with pytest.raises(HarnessValidationError, match="unchanged version"):
        X.run_expand("Q3.1.a", None, force=True)
    # a file from a lower item version is overwritten only with --force
    data["experiment_key"] = data["experiment_key"].replace(".v1", ".v0")
    path.write_text(yaml_io.dump(data), encoding="utf-8")
    with pytest.raises(HarnessValidationError, match="--force"):
        X.run_expand("Q3.1.a", None)
    assert X.run_expand("Q3.1.a", None, force=True) == 0
    assert path.read_bytes() == first


def test_unknown_selection_exits_2(gen: Path) -> None:
    assert main(["taxonomy", "expand", "--item", "Z9.9.z"]) == 2
    assert main(["taxonomy", "expand", "--wave", "V9"]) == 2


def test_programme_selects_generated_files_by_registry_wave(gen: Path) -> None:
    from financial_llm_harness.execution.programme import resolve_selection

    assert main(["taxonomy", "expand", "--wave", "V1"]) == 0
    sel = {
        "select": {
            "specs": ["experiments/financial/specs/momentum_stock_add.yaml"],
            "generated": {
                "waves": ["V1"],
                "priorities": ["P1"],
                "modes": ["confirmatory"],
                "variants": ["m1"],
            },
        }
    }
    chosen = resolve_selection(sel)
    keys = [c.experiment_key for c in chosen]
    assert keys == sorted(keys)
    assert "f1_1_a__m1__i_f_momentum.v1" in keys  # the canonical spec, listed by path
    generated = [
        c for c in chosen if c.spec.get("battery") or c.path.is_relative_to(gen)
    ]
    assert {c.spec["variant_id"] for c in generated} == {"m1"}
    assert all(c.spec["mode"] == "confirmatory" for c in generated)
    # m1 leverage is discovery (unsigned B.conv checklist), so a confirmatory selection omits it
    assert "f1_1_a__m1__i_f_leverage.v1" not in keys
    assert "f1_1_a__m1__i_f_valuation.v1" in keys


def test_a_higher_version_file_is_never_overwritten(gen: Path) -> None:
    assert main(["taxonomy", "expand", "--item", "M6.4.a"]) == 0
    path = gen / "M6.4.a/base/I.USER.stance.yaml"
    data = yaml_io.load(path)
    data["experiment_key"] = data["experiment_key"].replace(".v1", ".v2")
    path.write_text(yaml_io.dump(data), encoding="utf-8")
    for force in (False, True):
        with pytest.raises(HarnessValidationError, match="version 2"):
            X.run_expand("M6.4.a", None, force=force)


def test_wave_selection_reads_the_registry_field(
    gen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = R.load_registry

    def moved() -> R.Registry:
        reg = real()
        item = reg.items["F6.1.b"]
        reg.items["F6.1.b"] = R.Item({**item.data, "wave": "V1.1"})
        return reg

    monkeypatch.setattr(R, "load_registry", moved)
    assert X.run_expand(None, "V1", allow_partial=True) == 0
    assert not (
        gen / "F6.1.b"
    ).exists()  # the item's wave field, not the §T8 table, decides
    assert (gen / "Q3.1.a/base/I.HYPE.label.yaml").exists()


def test_correctness_experiment_and_its_missing_inputs(
    gen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§32B.2 step 1a: with the skip rule lifted, K.M.1's I.NONE pair needs a scorer per check,
    a correctness feature and a full defaults entry; supplying exactly those makes it expand."""
    from financial_llm_harness.generation.corpus import derive_arms

    monkeypatch.setattr(X, "_skip_correctness", lambda item, vid, family: False)
    reg = R.validate_registry()
    item = reg.items["K.M.1"]
    rows = {r.intervention_id: r for r in X.expand_item(reg, item, R.family_listing())}
    none = rows["I.NONE"]
    assert none.spec is None and none.reason == "unimplemented_content"
    assert "families/code_task/family.yaml#scorers/HT.LEAK" in none.missing
    assert "families/code_task/defaults.yaml#K.M.1/correctness_feature" in none.missing
    s5 = yaml_io.load(
        X.repo_root() / "experiments/financial/specs/temporal_norm_code.yaml"
    )
    entry = {k: s5[k] for k in X.COPIED} | {
        "mandate": "CT1",
        "hypothesis": s5["hypothesis"],
        "correctness_feature": "valid_point_in_time_normaliser",
        "correctness_context": {
            "node": "purpose",
            "value": "signal for a walk-forward backtest",
        },
    }
    real_defaults = X.load_defaults
    monkeypatch.setattr(
        X,
        "load_defaults",
        lambda fid: {"entries": {"K.M.1": entry}}
        if fid == "code_task"
        else real_defaults(fid),
    )
    import dataclasses

    real_family = X.load_family
    scorers = {c: {"hidden_test": c} for c in item.correctness_checks}

    def with_scorers(fid: str) -> Any:
        fam = real_family(fid)
        if fid != "code_task":
            return fam
        return dataclasses.replace(fam, raw={**fam.raw, "scorers": scorers})

    monkeypatch.setattr(X, "load_family", with_scorers)
    rows = {r.intervention_id: r for r in X.expand_item(reg, item, R.family_listing())}
    spec = rows["I.NONE"].spec
    assert spec is not None and spec["experiment_key"] == "k_m_1__base__i_none.v1"
    assert derive_arms(spec) == ["A", "A0"]
    assert spec["controls"]["placebo"] is None and spec["controls"]["positive"] is None
    assert spec["oracle_spec"]["type"] == "LEVEL" and spec["oracle_spec"]["tau"] == 0.90
    assert spec["intervention_spec"] is None and spec["concept"] == "correctness"
    assert not any(
        f.get("analysis_role") == "attribution" for f in spec["feature_spec"]
    )
    assert spec["correctness_context"]["node"] == "purpose"
