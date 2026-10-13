"""code_task: bank draws, datasets, code-fixture assets and the fake response (spec §12, §13, §35 S5, §T6).

A code task's fixture lives in ``fixtures/<task>/``: ``fixture.yaml`` (role
``family_config``: allowed file, checks, expected invariants, dataset nodes,
references), ``starter/`` (role ``template``) and ``tests_hidden/`` (role
``hidden_tests``). Directory assets carry their provenance block in
``PROVENANCE.yaml`` (§8.1). The case's ``data/`` panel is generated from the
case state with the §12 dataset seed when the sandbox runs (§13.1).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.code_ref import package_dir
from financial_llm_harness.core.ids import uint64
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.families import (
    bank_entries,
    family_dir,
    provenance_of,
    rel_to_package,
)
from financial_llm_harness.generation.provenance import Provenance, parse_provenance

FAMILY_ID = "code_task"
DATASET_GENERATORS = ("panel_v1",)
FIRST_DATE = "2000-01-03"
PROVENANCE_FILE = "PROVENANCE.yaml"
N_CUTS = 5
# the code_comment block (§T4.4): empty unless the I.CODE.comment cue adds a comment
BLOCK_NODES: dict[str, list[str]] = {"code_comment": []}


def render_blocks(
    state: Mapping[str, Any], nodes: Mapping[str, Any]
) -> dict[str, list[tuple[str, str | None]]]:
    """The base design shows no comment above the function under test."""
    del state, nodes
    return {"code_comment": []}


def primary_checks(spec: Mapping[str, Any]) -> list[str]:
    """The §T6.1 hidden-test IDs a spec's primary feature conjoins (§32B.2 step 1a skip rule)."""
    primary = spec["hypothesis"]["primary_feature"]
    feat = next(f for f in spec["feature_spec"] if f["feature_id"] == primary)
    if feat.get("extractor") != "code_all":
        return []
    fx = load_fixture(spec["task"])
    return sorted(fx.hidden_test_ids[t] for t in feat["tests"])


def draw_feature_names(rng: np.random.Generator, drawn: Mapping[str, Any]) -> str:
    """``n_features`` distinct bank names, in bank order, comma-separated (§35 S5)."""
    entries = [str(e) for e in bank_entries(FAMILY_ID, "feature_names")]
    k = int(drawn["n_features"])
    if not 1 <= k <= len(entries):
        raise HarnessValidationError(
            f"feature_names: n_features {k} outside 1..{len(entries)} (bank size)"
        )
    picked = sorted(int(i) for i in rng.choice(len(entries), size=k, replace=False))
    return ", ".join(entries[i] for i in picked)


def draw_vendor_label(rng: np.random.Generator, drawn: Mapping[str, Any]) -> str:
    entries = bank_entries(FAMILY_ID, "vendor_labels")
    return str(entries[int(rng.integers(0, len(entries)))])


# ---------------------------------------------------------------------------
# Fixtures (§13.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fixture:
    task: str
    task_id: str
    directory: Path
    sandbox_image: str
    allowed_file: str
    show_public_tests: bool
    checks: dict[str, str]  # check → hidden test function
    hidden_test_ids: dict[str, str]  # check → §T6.1 ID
    leakage_fixture: str
    burn_in_node: str
    min_dates_per_burn_in: int
    generator: str
    dataset_nodes: tuple[str, ...]
    references: dict[str, tuple[str, ...]]  # reference file → checks it must fail

    @property
    def logical(self) -> str:
        return rel_to_package(self.directory)

    @property
    def tests_hidden(self) -> Path:
        return self.directory / "tests_hidden"

    @property
    def starter(self) -> Path:
        return self.directory / "starter"

    def starter_text(self) -> str:
        return (self.starter / self.allowed_file).read_text(encoding="utf-8")


def fixture_dir(task: str) -> Path:
    return family_dir(FAMILY_ID) / "fixtures" / task


def load_fixture(task: str) -> Fixture:
    path = fixture_dir(task) / "fixture.yaml"
    where = rel_to_package(path) if path.exists() else f"fixtures/{task}/fixture.yaml"
    if not path.exists():
        raise HarnessValidationError(f"code task {task!r}: missing {where}")
    data = yaml_io.load(path)
    if not isinstance(data, Mapping):
        raise HarnessValidationError(f"{where}: expected a mapping")
    if data.get("task") != task:
        raise HarnessValidationError(f"{where}: task {data.get('task')!r} != {task!r}")
    allowed = list(data.get("allowed_files") or [])
    if len(allowed) != 1:
        raise HarnessValidationError(
            f"{where}: V1 supports a single allowed file, got {allowed} (§13.1)"
        )
    if data.get("starter") != "starter" or data.get("tests_hidden") != "tests_hidden":
        raise HarnessValidationError(
            f"{where}: starter/ and tests_hidden/ are fixed names"
        )
    checks = {str(k): str(v["test"]) for k, v in (data.get("checks") or {}).items()}
    ids = {str(k): str(v["hidden_test"]) for k, v in (data.get("checks") or {}).items()}
    inv = dict(data.get("expected_invariants") or {})
    ds = dict(data.get("dataset") or {})
    if ds.get("generator") not in DATASET_GENERATORS:
        raise HarnessValidationError(
            f"{where}: unknown dataset generator {ds.get('generator')!r}"
        )
    refs = {
        str(k): tuple(str(c) for c in (v or []))
        for k, v in (data.get("references") or {}).items()
    }
    return Fixture(
        task=task,
        task_id=str(data["task_id"]),
        directory=path.parent,
        sandbox_image=str(data["sandbox_image"]),
        allowed_file=str(allowed[0]),
        show_public_tests=bool(data.get("show_public_tests", False)),
        checks=checks,
        hidden_test_ids=ids,
        leakage_fixture=str(inv["leakage_fixture"]),
        burn_in_node=str(inv["burn_in_node"]),
        min_dates_per_burn_in=int(inv.get("min_dates_per_burn_in", 1)),
        generator=str(ds["generator"]),
        dataset_nodes=tuple(str(n) for n in ds.get("nodes") or ()),
        references=refs,
    )


# ---------------------------------------------------------------------------
# Datasets (§12): generated from the state with the dataset seed
# ---------------------------------------------------------------------------


def dataset_seed(manifest: Mapping[str, Any], case_idx: int) -> int:
    """``uint64("dataset" | seed_stream | seeds.generation | case_idx)`` (§12)."""
    return uint64(
        "dataset",
        manifest["scenario_spec"]["seed_stream"],
        int(manifest["seeds"]["generation"]),
        int(case_idx),
    )


def dataset_dates(state: Mapping[str, Any]) -> np.ndarray:
    """The panel's dates: ``n_dates`` business days from 2000-01-03."""
    n = int(state["n_dates"])
    start = np.datetime64(FIRST_DATE, "D")
    return np.busday_offset(start, np.arange(n), roll="forward")


def dataset_arrays(state: Mapping[str, Any], seed: int) -> dict[str, np.ndarray]:
    """``panel_v1``: a balanced date × asset panel whose columns are not standardised.

    Each (asset, feature) series is ``loc + scale · ε`` with ``loc`` ~ U(−10, 10),
    ``scale`` ~ U(0.5, 5) and ε i.i.d. N(0, 1), so a z-score is non-trivial and
    the hidden tests' affine-invariance and leakage checks are well defined.
    The location and scale ranges are a fixture choice, not a scenario
    distribution: no check's pass condition depends on them.
    """
    rng = np.random.default_rng(seed)
    n_assets = int(state["n_assets"])
    n_features = int(state["n_features"])
    names = [s.strip() for s in str(state["feature_names"]).split(",")]
    if len(names) != n_features:
        raise HarnessValidationError(
            f"feature_names lists {len(names)} names, n_features is {n_features}"
        )
    dates = dataset_dates(state)
    loc = rng.uniform(-10.0, 10.0, size=(n_assets, n_features))
    scale = rng.uniform(0.5, 5.0, size=(n_assets, n_features))
    eps = rng.standard_normal((len(dates), n_assets, n_features))
    return {
        "dates": dates,
        "assets": np.array([f"A{i:03d}" for i in range(1, n_assets + 1)]),
        "columns": np.array(names),
        "values": loc + scale * eps,
    }


def case_data(
    manifest: Mapping[str, Any], case_idx: int, state: Mapping[str, Any], fx: Fixture
) -> dict[str, bytes]:
    """The files of the case's ``data/`` directory (§13.1)."""
    from financial_llm_harness.evaluation.sandbox import npz_bytes

    arrays = dataset_arrays(state, dataset_seed(manifest, case_idx))
    meta = {"burn_in": int(state[fx.burn_in_node])}
    return {
        Path(fx.leakage_fixture).name: npz_bytes(arrays),
        "meta.json": (json.dumps(meta, sort_keys=True) + "\n").encode("utf-8"),
    }


# ---------------------------------------------------------------------------
# Assets and provenance (§0.10, §8.1)
# ---------------------------------------------------------------------------


def code_asset_rows(spec: Mapping[str, Any]) -> list[B.AssetRow]:
    """``template`` starter, ``family_config`` fixture manifest, ``hidden_tests`` and ``sandbox_recipe`` rows."""
    from financial_llm_harness.evaluation.sandbox import recipe_logical

    fx = load_fixture(spec["task"])
    rows = [
        B.make_asset("template", f"{fx.logical}/starter", "package"),
        B.make_asset("family_config", f"{fx.logical}/fixture.yaml", "package"),
        B.make_asset("hidden_tests", f"{fx.logical}/tests_hidden", "package"),
        B.make_asset("sandbox_recipe", recipe_logical(fx.sandbox_image), "package"),
    ]
    if fx.show_public_tests:
        rows.append(B.make_asset("template", f"{fx.logical}/tests_public", "package"))
    return rows


def directory_provenance(directory: Path, logical: str, role: str) -> Provenance:
    path = directory / PROVENANCE_FILE
    if not path.is_file():
        raise HarnessValidationError(f"{logical}: missing {PROVENANCE_FILE} (§8.1)")
    data = yaml_io.load(path)
    if not isinstance(data, Mapping):
        raise HarnessValidationError(f"{logical}/{PROVENANCE_FILE}: expected a mapping")
    return parse_provenance(provenance_of(data), logical, role)


def code_provenances(spec: Mapping[str, Any]) -> list[Provenance]:
    fx = load_fixture(spec["task"])
    manifest_data = yaml_io.load(fx.directory / "fixture.yaml")
    assert isinstance(manifest_data, Mapping)
    out = [
        parse_provenance(
            provenance_of(manifest_data), f"{fx.logical}/fixture.yaml", "oracle"
        ),
        directory_provenance(fx.starter, f"{fx.logical}/starter", "text"),
        directory_provenance(fx.tests_hidden, f"{fx.logical}/tests_hidden", "oracle"),
    ]
    if fx.show_public_tests:
        out.append(
            directory_provenance(
                fx.directory / "tests_public", f"{fx.logical}/tests_public", "text"
            )
        )
    return out


# ---------------------------------------------------------------------------
# Validation (§9, §13.5)
# ---------------------------------------------------------------------------


def cut_points(dates: np.ndarray) -> list[Any]:
    """t_k = u_⌊k·n/6⌋, k = 1..5 (1-based positions), over the sorted unique dates (§13.5)."""
    u = np.unique(dates)
    n = len(u)
    if n < 6:
        raise HarnessValidationError(
            f"leakage fixture has {n} unique dates; §13.5 needs at least 6"
        )
    cuts = [u[(k * n) // 6 - 1] for k in range(1, N_CUTS + 1)]
    for k, t in enumerate(cuts, start=1):
        if not (np.any(u <= t) and np.any(u > t)):
            raise HarnessValidationError(
                f"leakage cut point {k} has no rows on one side (§13.5)"
            )
    return cuts


def _test_functions(tests_hidden: Path) -> set[str]:
    names: set[str] = set()
    for path in sorted(tests_hidden.rglob("test_*.py")):
        names.update(
            re.findall(r"^def (test_\w+)\(", path.read_text(encoding="utf-8"), re.M)
        )
    return names


def validate_code_design(
    manifest: Mapping[str, Any], cases: Iterable[Mapping[str, Any]]
) -> None:
    """Fixture, feature, prompt and leakage-fixture checks of a code experiment (exit 2)."""
    from financial_llm_harness.evaluation.sandbox import RECIPES, recipe_dir

    fx = load_fixture(manifest["task"])
    if fx.sandbox_image not in RECIPES or not recipe_dir(fx.sandbox_image).is_dir():
        raise HarnessValidationError(
            f"code task {fx.task}: unknown sandbox image {fx.sandbox_image!r}"
        )
    if fx.show_public_tests and not (fx.directory / "tests_public").is_dir():
        raise HarnessValidationError(
            f"code task {fx.task}: show_public_tests without tests_public/"
        )
    if not (fx.starter / fx.allowed_file).is_file():
        raise HarnessValidationError(
            f"code task {fx.task}: starter/{fx.allowed_file} is missing"
        )
    defined = _test_functions(fx.tests_hidden)
    for check, test in fx.checks.items():
        if test not in defined:
            raise HarnessValidationError(
                f"code task {fx.task}: check {check} names {test}, which tests_hidden/ does not define"
            )
    for f in manifest["feature_spec"]:
        named: Sequence[str] = ()
        if f.get("extractor") == "code_outcome":
            named = [f["test"]]
        elif f.get("extractor") == "code_all":
            named = list(f["tests"])
        unknown = [t for t in named if t not in fx.checks]
        if unknown:
            raise HarnessValidationError(
                f"feature {f['feature_id']}: checks {unknown} are not in fixture {fx.task}"
            )
    starter = fx.starter_text().strip()
    for logical in manifest["paraphrase_spec"]["template_files"]:
        data = yaml_io.load(package_dir() / logical)
        user = (
            str(data.get("user", "")).replace("{code_comment}", "")
            if isinstance(data, Mapping)
            else ""
        )
        if starter not in user:
            raise HarnessValidationError(
                f"{logical}: the prompt does not show starter/{fx.allowed_file} verbatim (§13.1)"
            )
    for case in cases:
        for arm, state in sorted(case["arms"].items()):
            where = f"case {case['case_idx']} arm {arm}"
            for node in (*fx.dataset_nodes, fx.burn_in_node):
                if node not in state:
                    raise HarnessValidationError(f"{where}: state has no {node!r}")
            dates = dataset_dates(state)
            try:
                cut_points(dates)
            except HarnessValidationError as exc:
                raise HarnessValidationError(f"{where}: {exc}") from exc
            burn_in = int(state[fx.burn_in_node])
            if len(dates) < fx.min_dates_per_burn_in * burn_in:
                raise HarnessValidationError(
                    f"{where}: {len(dates)} dates < {fx.min_dates_per_burn_in} × burn-in {burn_in} (§35 S5)"
                )


# ---------------------------------------------------------------------------
# Fake response (§20.11)
# ---------------------------------------------------------------------------

_PROSE = (
    "Here is the module.",
    "The implementation follows.",
    "Below is a complete implementation.",
)


def reference_sources(task: str) -> tuple[list[str], list[str]]:
    """(known-good, known-bad) reference sources of a fixture, in file order."""
    fx = load_fixture(task)
    good: list[str] = []
    bad: list[str] = []
    for rel, fails in sorted(fx.references.items()):
        text = (fx.directory / rel).read_text(encoding="utf-8")
        (bad if fails else good).append(text)
    return good, bad


_CONTRACT_TASK = {"normalise_code_v1": "temporal_norm_code"}


def fake_response(
    contract_id: str,
    y: int | float,
    mention: bool,
    rng: np.random.Generator,
    concept_factor: str | None = None,
    reasoning_mode: str = "none",
) -> str:
    """A ``code_v1`` response: a known-good reference when ``y`` is 1, else a known-bad one."""
    del mention, concept_factor, reasoning_mode
    task = _CONTRACT_TASK.get(contract_id)
    if task is None:
        raise ValueError(f"fake_response: unknown contract {contract_id!r}")
    good, bad = reference_sources(task)
    pool = good if int(y) == 1 else bad
    code = pool[int(rng.integers(0, len(pool)))]
    prose = _PROSE[int(rng.integers(0, len(_PROSE)))]
    return f"{prose}\n\n```python\n{code}```\n"
