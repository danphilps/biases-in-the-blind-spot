"""Frozen closure; Hash acyclicity (spec §0.10, §3, §4, §36)."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import ids, yaml_io
from financial_llm_harness.core.code_ref import CodeRef, compute_code_ref, package_dir
from financial_llm_harness.errors import FrozenInputDrift, HarnessValidationError
from financial_llm_harness.execution.runner import run_experiment, trial_seed
from financial_llm_harness.generation.corpus import build_corpus
from financial_llm_harness.generation.generate import generate
from financial_llm_harness.generation.validate_bundle import freeze
from tests.harness.helpers import (
    ci_fake_entries,
    load_spec,
    local_entry,
    small_spec,
    write_panel,
    write_probe,
)

REF = CodeRef("0000000", "0" * 64, None, None, False)
TEMPLATE = (
    package_dir()
    / "families"
    / "fundamental_judgement"
    / "paraphrases"
    / "stock_add_v1.terse.yaml"
)


@pytest.fixture
def restore_files() -> Iterator[list[Path]]:
    """Paths edited by a test; each is restored from its backup (or removed) afterwards."""
    paths: list[Path] = []
    yield paths
    for p in paths:
        backup = p.with_name(p.name + ".bak-test")
        if backup.exists():
            p.write_bytes(backup.read_bytes())
            backup.unlink()
        elif p.exists():
            p.unlink()


def _edit(
    paths: list[Path], path: Path, new_text: str | None = None, append: str = ""
) -> None:
    backup = path.with_name(path.name + ".bak-test")
    if path.exists():
        backup.write_bytes(path.read_bytes())
    paths.append(path)
    if new_text is not None:
        path.write_text(new_text, encoding="utf-8")
    else:
        path.write_text(path.read_text(encoding="utf-8") + append, encoding="utf-8")


def _frozen(tmp_path: Path) -> Path:
    out = tmp_path / "bundle"
    res = generate(small_spec(tmp_path), out, code_ref=REF)
    freeze(out, code_ref=REF, regenerated=res.corpus)
    return out


def test_editing_used_template_is_drift_before_any_call(
    tmp_path: Path, harness_home: Path, restore_files: list[Path]
) -> None:
    out = _frozen(tmp_path)
    _edit(restore_files, TEMPLATE, append="# edited\n")
    with pytest.raises(FrozenInputDrift) as exc:
        run_experiment(out / B.MANIFEST_FILE, None, yes=True, code_ref=REF)
    assert exc.value.exit_code == 2
    assert any("stock_add_v1.terse.yaml" in m for m in exc.value.mismatches)
    assert not (harness_home / "runs").exists()


def test_editing_unused_file_changes_no_hash(
    tmp_path: Path, restore_files: list[Path]
) -> None:
    out = _frozen(tmp_path)
    unused = TEMPLATE.with_name("unused_test_template.yaml")
    _edit(restore_files, unused, new_text="provenance: null\n")
    assert B.frozen_input_mismatches(out, REF) == []


def test_editing_family_scorer_changes_tree_hash(
    tmp_path: Path, restore_files: list[Path]
) -> None:
    before = compute_code_ref(False)
    out = tmp_path / "bundle"
    res = generate(small_spec(tmp_path), out, code_ref=before)
    freeze(out, code_ref=before, regenerated=res.corpus)
    fam_py = package_dir() / "families" / "fundamental_judgement" / "family.py"
    _edit(restore_files, fam_py, append="\n# edited for a drift test\n")
    after = compute_code_ref(False)
    assert after.harness_tree_sha256 != before.harness_tree_sha256
    mism = B.frozen_input_mismatches(out, after)
    assert any("harness_tree_sha256" in m for m in mism)


def test_editing_bank_or_mandate_is_drift(
    tmp_path: Path, restore_files: list[Path]
) -> None:
    out = _frozen(tmp_path)
    bank = (
        package_dir()
        / "families"
        / "fundamental_judgement"
        / "banks"
        / "hq_cities.yaml"
    )
    _edit(restore_files, bank, append="# x\n")
    mism = B.frozen_input_mismatches(out, REF)
    assert any("hq_cities.yaml" in m for m in mism)


def test_clean_tree_rule(tmp_path: Path, harness_home: Path) -> None:
    entry = local_entry()
    write_probe(harness_home, entry)
    panel = write_panel(tmp_path, [entry])
    dirty = CodeRef(None, "0" * 64, None, None, True)
    with pytest.raises(HarnessValidationError, match="clean-tree") as exc:
        generate(small_spec(tmp_path), tmp_path / "b", str(panel), code_ref=dirty)
    assert exc.value.exit_code == 2
    # fake-only and discovery specs freeze with dirty: true and both tree hashes recorded
    res = generate(small_spec(tmp_path), tmp_path / "b2", code_ref=dirty)
    assert res.manifest["code_ref"]["dirty"] is True
    assert res.manifest["code_ref"]["harness_tree_sha256"] == "0" * 64
    disc = small_spec(
        tmp_path,
        name="disc.yaml",
        mode="discovery",
        scenario_spec__seed_stream="disc",
        hypothesis__correction_family="exploratory",
    )
    res2 = generate(disc, tmp_path / "b3", str(panel), code_ref=dirty)
    assert res2.manifest["code_ref"]["dirty"] is True


# ---------------------------------------------------------------------------
# Hash acyclicity (§3, §4)
# ---------------------------------------------------------------------------

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def test_generation_computes_no_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **k: object) -> str:
        raise AssertionError("an ID was derived before freeze")

    for name in (
        "uid",
        "experiment_id",
        "testcase_id",
        "arm_variant_id",
        "rendering_id",
        "cluster_id",
        "block_id",
        "trial_id",
    ):
        monkeypatch.setattr(ids, name, boom)
    import financial_llm_harness.generation.generate as G

    monkeypatch.setattr(G, "make_experiment_id", lambda dh: "deferred")
    res = generate(small_spec(tmp_path), tmp_path / "b", code_ref=REF)
    for fname in (B.CASES_FILE, B.RENDERINGS_FILE, B.ASSETS_FILE):
        text = (tmp_path / "b" / fname).read_text()
        assert res.manifest["design_hash"] not in text
        assert res.manifest["manifest_hash"] not in text
        assert not _UUID.search(text)


def test_regeneration_and_hash_separation(tmp_path: Path, harness_home: Path) -> None:
    spec = small_spec(tmp_path)
    a = generate(spec, tmp_path / "a", code_ref=REF)
    b = generate(spec, tmp_path / "b", code_ref=REF)
    assert a.manifest["design_hash"] == b.manifest["design_hash"]
    assert a.manifest["manifest_hash"] == b.manifest["manifest_hash"]
    # panel, budget and code identity change manifest_hash only
    entries = copy.deepcopy(ci_fake_entries())
    entries[0]["params"]["theta"] = -0.1
    panel = write_panel(tmp_path, entries)
    c = generate(spec, tmp_path / "c", str(panel), code_ref=REF)
    d = generate(
        spec, tmp_path / "d", code_ref=CodeRef("1111111", "1" * 64, None, None, False)
    )
    budget_spec = small_spec(tmp_path, name="budget.yaml", budget__max_calls=123456)
    e = generate(budget_spec, tmp_path / "e", code_ref=REF)
    for other in (c, d, e):
        assert other.manifest["design_hash"] == a.manifest["design_hash"]
        assert other.manifest["experiment_id"] == a.manifest["experiment_id"]
        assert other.manifest["manifest_hash"] != a.manifest["manifest_hash"]
    dh = a.manifest["design_hash"]
    mk = ci_fake_entries()[0]["model_key"]
    assert trial_seed(dh, 3, 1, 0, mk, "B") == trial_seed(
        c.manifest["design_hash"], 3, 1, 0, mk, "B"
    )
    assert ids.testcase_id(dh, 3) == ids.testcase_id(c.manifest["design_hash"], 3)
    # seeds and n_cases change both hashes
    f = generate(
        small_spec(tmp_path, name="seed.yaml", seeds__generation=7),
        tmp_path / "f",
        code_ref=REF,
    )
    g = generate(
        small_spec(tmp_path, name="n.yaml", n_cases=55), tmp_path / "g", code_ref=REF
    )
    for other in (f, g):
        assert other.manifest["design_hash"] != dh
        assert other.manifest["manifest_hash"] != a.manifest["manifest_hash"]


def test_control_subset_depends_only_on_ordinals() -> None:
    spec = load_spec()
    s1 = copy.deepcopy(spec)
    s1["scenario_spec"]["n_cases"] = 50
    s1["statistical_spec"]["planning_ref"] = []
    s1["controls"]["subset_n"] = None
    from tests.harness.helpers import fake_planning_refs

    s1["statistical_spec"]["planning_ref"] = fake_planning_refs()
    c1 = build_corpus(s1)
    s2 = copy.deepcopy(s1)
    s2["sampling_spec"]["panel"] = "other"
    s2["budget"]["max_calls"] = 1
    c2 = build_corpus(s2)
    carried1 = [sorted(c["arms"]) for c in c1.cases]
    carried2 = [sorted(c["arms"]) for c in c2.cases]
    assert carried1 == carried2
    import numpy as np

    from financial_llm_harness.core.ids import uint64

    perm = np.random.default_rng(
        uint64("controls", spec["seeds"]["generation"])
    ).permutation(c1.n_clusters)
    subset = {int(i) for i in perm[: c1.subset_n]}
    for c in c1.cases:
        assert ("A0" in c["arms"]) == (c["cluster_ordinal"] in subset)


def test_bundle_files_are_canonical(tmp_path: Path) -> None:
    out = _frozen(tmp_path)
    lock = (out / B.LOCK_FILE).read_text().strip()
    manifest = yaml_io.load(out / B.MANIFEST_FILE)
    assert lock == manifest["manifest_hash"]
    rows = json.loads((out / B.ASSETS_FILE).read_text())
    assert rows == sorted(rows, key=lambda r: (r["role"], r["logical_name"]))
    roles = {r["role"] for r in rows}
    assert {
        "template",
        "paraphrase",
        "bank",
        "mandate",
        "contract",
        "family_config",
        "oracle",
        "panel",
        "request_profile",
    } <= roles
    first = B.read_jsonl(out / B.RENDERINGS_FILE)[0]
    assert list(first) == [
        "case_idx",
        "arm",
        "paraphrase_idx",
        "system",
        "user",
        "response_contract",
        "prompt_hash",
    ]


# ---------------------------------------------------------------------------
# Resume and replay clauses (M2)
# ---------------------------------------------------------------------------


def test_template_edit_before_resume_stops_before_l7(
    tmp_path: Path,
    harness_home: Path,
    restore_files: list[Path],
) -> None:
    import sqlite3

    from financial_llm_harness.execution.runner import (
        InjectedInterruption,
        resume_experiment,
    )
    from financial_llm_harness.storage.db import db_path
    from tests.harness.helpers import GOLDEN_RUN_ID, fixed_clock

    spec = small_spec(tmp_path)
    with pytest.raises(InjectedInterruption):
        run_experiment(
            spec,
            "ci_fake",
            yes=True,
            run_id=GOLDEN_RUN_ID,
            clock=fixed_clock,
            code_ref=REF,
            interrupt_after=40,
        )
    run_dir = next((harness_home / "runs").glob("*_00000000"))

    def counts() -> tuple[int, int]:
        conn = sqlite3.connect(str(db_path(run_dir)))
        try:
            pending = conn.execute(
                "SELECT COUNT(*) FROM trial WHERE status = 'PENDING'"
            ).fetchone()[0]
            attempts = conn.execute("SELECT COUNT(*) FROM attempt").fetchone()[0]
            return pending, attempts
        finally:
            conn.close()

    before = counts()
    _edit(restore_files, TEMPLATE, append="# edited\n")
    with pytest.raises(FrozenInputDrift) as exc:
        resume_experiment(GOLDEN_RUN_ID, yes=True, clock=fixed_clock, code_ref=REF)
    assert exc.value.exit_code == 2
    assert counts() == before  # no trial executed: stopped before L7
    conn = sqlite3.connect(str(db_path(run_dir)))
    try:
        events = conn.execute(
            "SELECT payload_json FROM event WHERE kind = 'frozen_input_drift'"
        ).fetchall()
    finally:
        conn.close()
    assert len(events) == 1
    assert "stock_add_v1.terse.yaml" in events[0][0]


def test_replay_refuses_after_hashed_source_edit(
    tmp_path: Path,
    harness_home: Path,
    restore_files: list[Path],
) -> None:
    from financial_llm_harness.cli import main
    from tests.harness.helpers import GOLDEN_RUN_ID, output_files, run_small

    run_dir = run_small(tmp_path)
    before = output_files(run_dir)
    _edit(restore_files, TEMPLATE, append="# edited\n")
    from financial_llm_harness.reporting.replay import reevaluate

    with pytest.raises(FrozenInputDrift):
        reevaluate(GOLDEN_RUN_ID, code_ref=REF)
    assert main(["replay", GOLDEN_RUN_ID, "--reevaluate"]) == 2
    assert output_files(run_dir) == before
