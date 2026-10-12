"""``validate`` and ``freeze`` of a bundle (spec §4, §0.10, §8.1, §20.4, §22, §23.6; §31A L2–L3).

``validate`` re-runs every check on a bundle: manifest structure, the frozen
inputs, deterministic regeneration of the corpus (cases and renderings must be
byte-identical), power, panel, probes, provenance and independence.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.canonical_json import sha256_text
from financial_llm_harness.core.code_ref import CodeRef, compute_code_ref, package_dir
from financial_llm_harness.core.manifest import MANIFEST_ORDER, check_design
from financial_llm_harness.core.paths import repo_root
from financial_llm_harness.errors import (
    FrozenInputDrift,
    HarnessValidationError,
    StaleProbe,
)
from financial_llm_harness.families import provenance_of
from financial_llm_harness.generation.corpus import Corpus, CueHook, build_corpus
from financial_llm_harness.generation.provenance import (
    Provenance,
    author_families,
    live_violations,
    parse_provenance,
)
from financial_llm_harness.models.base import profile_sha256
from financial_llm_harness.models.panels import derive_model_key, load_panel
from financial_llm_harness.statistics import power

PROBE_MAX_AGE_DAYS = 30
OPTIONAL_MANIFEST_KEYS = ("battery", "correctness_context", "provenance")
TEXT_ROLES = ("template", "paraphrase", "bank", "mandate", "contract")
ORACLE_FILES = (
    "family.yaml",
    "fixture.yaml",
    "defaults.yaml",
    "taxonomy/faq.yaml",
    "taxonomy/bases.yaml",
    "taxonomy/interventions.yaml",
)


@dataclass
class ValidationReport:
    manifest: dict[str, Any]
    n_clusters: int
    n_required: int
    subset_n: int
    computed_subset_n: int
    provenances: list[Provenance]


def current_code_ref(manifest: Mapping[str, Any], injected: CodeRef | None) -> CodeRef:
    if injected is not None:
        return injected
    uses_legacy = (manifest.get("code_ref") or {}).get("legacy_tree_sha256") is not None
    return compute_code_ref(uses_legacy)


def manifest_targets_non_fake(manifest: Mapping[str, Any]) -> bool:
    models = manifest["sampling_spec"]["models"]
    evs = manifest.get("evaluators") or []
    return any(m["provider"] != "fake" for m in models) or any(
        not str(e["model_key"]).startswith("fake--") for e in evs
    )


def file_provenances(
    assets: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> list[Provenance]:
    """Provenance of every prompt-visible and oracle-bearing file the bundle uses."""
    seen: dict[str, Provenance] = {}
    for row in assets:
        if row["source"] != "package":
            continue
        logical = row["logical_name"].split("#", 1)[0]
        role: str | None = None
        if row["role"] in TEXT_ROLES:
            role = "text"
        elif (
            row["role"] == "hidden_tests"
            or logical.endswith(ORACLE_FILES)
            or logical.startswith("taxonomy/checklists/")
        ):
            role = "oracle"
        if role is None or logical in seen:
            continue
        path = package_dir() / logical
        if path.is_dir():
            # a directory asset (code starter, hidden tests) carries PROVENANCE.yaml (§8.1)
            path = path / "PROVENANCE.yaml"
            if not path.is_file():
                raise HarnessValidationError(
                    f"{logical}: missing PROVENANCE.yaml (§8.1)"
                )
        if not path.exists() or path.suffix not in (".yaml", ".yml"):
            continue
        data = yaml_io.load(path)
        seen[logical] = parse_provenance(provenance_of(data), logical, role)
    out = list(seen.values())
    if manifest.get("provenance") is not None:
        out.append(
            parse_provenance(
                manifest["provenance"], f"manifest:{manifest['experiment_key']}", "both"
            )
        )
    return out


def regenerate(manifest: Mapping[str, Any], cue_hook: CueHook | None = None) -> Corpus:
    design = dict(manifest)
    if cue_hook is None and manifest.get("battery"):
        from financial_llm_harness.taxonomy.battery import cue_hook_for

        cue_hook = cue_hook_for(manifest)
    return build_corpus(design, cue_hook)


def validate_bundle(
    bundle_dir: Path,
    *,
    code_ref: CodeRef | None = None,
    regenerated: Corpus | None = None,
    clock: Callable[[], datetime] | None = None,
    require_lock: bool = False,
) -> ValidationReport:
    """Every L2 check; raises HarnessValidationError (2), FrozenInputDrift (2) or StaleProbe (6)."""
    manifest = B.load_manifest(bundle_dir)
    where = str(bundle_dir / B.MANIFEST_FILE)
    missing = [
        k
        for k in MANIFEST_ORDER
        if k not in manifest and k not in OPTIONAL_MANIFEST_KEYS
    ]
    if missing:
        raise HarnessValidationError(f"{where}: missing keys {missing}")
    check_design(manifest, where)
    ref = current_code_ref(manifest, code_ref)
    drift = B.frozen_input_mismatches(bundle_dir, ref, require_lock=require_lock)
    if drift:
        raise FrozenInputDrift(drift)
    assets = B.load_assets(bundle_dir)
    # --- panel and probes (§20.4, §20.9) --------------------------------------------
    models = manifest["sampling_spec"]["models"]
    for m in models:
        if derive_model_key(m) != m["model_key"]:
            raise HarnessValidationError(
                f"model_key {m['model_key']!r} differs from its derived key (§20.6)"
            )
    panel_name = manifest["sampling_spec"]["panel_ref"]["name"]
    panel = load_panel(panel_name)
    if B.file_sha256(panel.path) != manifest["sampling_spec"]["panel_ref"]["sha256"]:
        raise FrozenInputDrift([f"panel {panel_name}: file changed since generate"])
    now = (clock or (lambda: datetime.now(UTC)))()
    probed = (
        list(models) + [_evaluator_model(manifest, panel)]
        if manifest.get("evaluators")
        else list(models)
    )
    for m in probed:
        if m is None or m["provider"] == "fake":
            continue
        _check_probe(bundle_dir, m, now)
    # --- deterministic regeneration (§4 regeneration, §5) -------------------------
    corpus = regenerated if regenerated is not None else regenerate(manifest)
    if sha256_text(B.jsonl_text(corpus.cases)) != manifest["corpus"]["cases"]["sha256"]:
        raise HarnessValidationError(
            "regenerated cases.jsonl differs from the bundle (non-deterministic generator or drift)"
        )
    if (
        sha256_text(B.jsonl_text(corpus.renderings))
        != manifest["corpus"]["renderings"]["sha256"]
    ):
        raise HarnessValidationError(
            "regenerated renderings.jsonl differs from the bundle"
        )
    if manifest["mode"] == "confirmatory" and manifest.get(
        "parent_discovery_experiment_id"
    ):
        from financial_llm_harness.discovery import check_parent_overlap

        check_parent_overlap(
            manifest, corpus.cases, corpus.family.scenario_schema.nodes
        )
    if list(corpus.design_arms) != list(manifest["arms"]):
        raise HarnessValidationError(
            f"arms {manifest['arms']} differ from the derived {corpus.design_arms}"
        )
    # --- power (§22, §23.6) ---------------------------------------------------------
    n_clusters = corpus.n_clusters
    req = corpus.primary_requirement
    # a discovery run (e.g. a planning pilot, CR-7) carries no confirmatory label, so its
    # power is reported by `plan`, not enforced; its controls run on every cluster it has
    discovery = manifest["mode"] == "discovery"
    if n_clusters < req.n_required and not discovery:
        raise HarnessValidationError(
            f"power: N_clusters {n_clusters} < N_required {req.n_required} for {req.contrast} "
            f"(planning variance {req.variance.variance}, {req.variance.source}; §23.6)"
        )
    computed = (
        power.computed_subset_n(corpus.control_requirements)
        if corpus.control_requirements
        else 0
    )
    if discovery:
        computed = min(computed, n_clusters)
    subset_n = int(manifest["controls"]["subset_n"] or 0)
    if subset_n < computed:
        raise HarnessValidationError(
            f"controls.subset_n {subset_n} < computed {computed} (§22)"
        )
    if subset_n > n_clusters:
        raise HarnessValidationError(
            f"controls.subset_n {subset_n} > N_clusters {n_clusters} (§22)"
        )
    if not discovery:
        _panel_planning_check(manifest, n_clusters, subset_n)
    # --- provenance and independence (§2.7, §8.1, D-14) -----------------------------
    provs = file_provenances(assets, manifest)
    non_fake = manifest_targets_non_fake(manifest)
    if non_fake:
        bad = live_violations(provs)
        if bad:
            raise HarnessValidationError(
                "live runs need human-authored, approved prompt-visible and oracle-bearing files (§8.1, D-14): "
                + "; ".join(bad[:20])
            )
    ind = manifest["independence"]
    expected_targets = sorted({m["family"] for m in models})
    if ind.get("target_families") != expected_targets:
        raise HarnessValidationError(
            "independence.target_families does not match the panel"
        )
    if sorted(ind.get("text_author_families") or []) != author_families(
        provs, ("text", "both")
    ):
        raise HarnessValidationError(
            "independence.text_author_families does not match the files used"
        )
    if sorted(ind.get("oracle_author_families") or []) != author_families(
        provs, ("oracle", "both")
    ):
        raise HarnessValidationError(
            "independence.oracle_author_families does not match the files used"
        )
    live_targets = {m["family"] for m in models if m["provider"] != "fake"}
    for key in ("text_author_families", "oracle_author_families", "evaluator_families"):
        overlap = live_targets & set(ind.get(key) or [])
        if overlap:
            raise HarnessValidationError(
                f"independence: target families {sorted(overlap)} also in {key} (§2.7)"
            )
    # --- semantic features (§16) -------------------------------------------------------
    for ev in manifest.get("evaluators") or []:
        if manifest["mode"] == "confirmatory":
            cal = ev.get("calibration_ref")
            if not cal or float(cal.get("kappa", 0.0)) < 0.70:
                raise HarnessValidationError(
                    f"semantic feature {ev['feature_id']}: no deployable calibration record (κ ≥ 0.70) for this evaluator (§16)"
                )
    _basis_rule(manifest)
    _fake_planning_refs(manifest)
    if manifest.get("battery"):
        from financial_llm_harness.taxonomy.battery import validate_irrelevance

        validate_irrelevance(manifest)
    if manifest["task_family"] == "code_task":
        from financial_llm_harness.families.code_task.family import validate_code_design

        validate_code_design(manifest, corpus.cases)
    if (
        manifest["mode"] == "confirmatory"
        and non_fake
        and manifest["code_ref"]["dirty"]
    ):
        raise HarnessValidationError(
            "clean-tree rule: confirmatory non-fake manifest from a dirty tree (§0.10)"
        )
    return ValidationReport(
        manifest, n_clusters, req.n_required, subset_n, computed, provs
    )


def _evaluator_model(manifest: Mapping[str, Any], panel: Any) -> dict[str, Any] | None:
    ev = panel.evaluator
    if ev is None:
        return None
    from financial_llm_harness.generation.generate import resolve_model

    if ev["provider"] == "fake":
        return None
    model, _ = resolve_model(ev)
    return model


def _check_probe(bundle_dir: Path, m: Mapping[str, Any], now: datetime) -> None:
    ref = m.get("probe_ref")
    if not ref:
        raise StaleProbe(f"{m['model_key']}: no probe (§20.4)")
    path = bundle_dir / ref["path"]
    if not path.exists():
        raise StaleProbe(f"{m['model_key']}: probe file missing from the bundle")
    probe = json.loads(path.read_text(encoding="utf-8"))
    date = datetime.strptime(str(ref["date"]), "%Y-%m-%d").replace(tzinfo=UTC)
    if (now - date).days > PROBE_MAX_AGE_DAYS:
        raise StaleProbe(
            f"{m['model_key']}: probe of {ref['date']} is older than {PROBE_MAX_AGE_DAYS} days"
        )
    rp = m["request_profile"]
    if (
        probe.get("request_profile_id") != rp["id"]
        or probe.get("request_profile_sha256") != rp["sha256"]
    ):
        raise StaleProbe(
            f"{m['model_key']}: probe ran under another request profile (§20.4)"
        )
    if rp["sha256"] != profile_sha256(m["provider"], rp["id"]):
        raise StaleProbe(f"{m['model_key']}: request profile changed since the probe")


def _panel_planning_check(
    manifest: Mapping[str, Any], n_clusters: int, subset_n: int
) -> None:
    """Non-fake models without a planning_ref entry are planned at the no-pilot bound (§22)."""
    st = manifest["statistical_spec"]
    refs = st.get("planning_ref") or []
    primary = next(f for f in manifest["feature_spec"] if f.get("primary") is True)
    h = manifest["hypothesis"]
    for m in manifest["sampling_spec"]["models"]:
        if m["provider"] == "fake":
            continue
        mine = [r for r in refs if r.get("model_key") == m["model_key"]]
        st_m = dict(st)
        st_m["planning_ref"] = mine
        req = power.primary_requirement(
            manifest["oracle_spec"]["type"],
            h,
            st_m,
            primary,
            manifest["oracle_spec"].get("tol"),
            m["model_key"],
        )
        if req.n_required > n_clusters:
            raise HarnessValidationError(
                f"model {m['model_key']}: contrast {req.contrast} needs {req.n_required} clusters at "
                f"{req.variance.source} variance; N_clusters is {n_clusters} (§22)"
            )
        for c in power.control_requirements(
            manifest["arms"], manifest["controls"], h, st_m, primary, m["model_key"]
        ):
            if c.n_required > subset_n:
                raise HarnessValidationError(
                    f"model {m['model_key']}: control contrast {c.contrast} needs {c.n_required}; subset_n is {subset_n} (§22)"
                )


def _fake_planning_refs(manifest: Mapping[str, Any]) -> None:
    keys = {m["model_key"]: m["provider"] for m in manifest["sampling_spec"]["models"]}
    for r in manifest["statistical_spec"].get("planning_ref") or []:
        if r.get("run_id") is None and keys.get(r.get("model_key")) != "fake":
            raise HarnessValidationError(
                f"planning_ref for {r.get('model_key')}: a fake-design variance may only name a fake model (§22)"
            )


def _basis_rule(manifest: Mapping[str, Any]) -> None:
    """§27: a confirmatory a_priori hypothesis cites only established, or signed-off conventional, bases."""
    if manifest["mode"] != "confirmatory":
        return
    if manifest["hypothesis"]["source"] != "a_priori":
        return
    path = package_dir() / "taxonomy" / "bases.yaml"
    if not path.exists():
        return
    data = yaml_io.load(path)
    table = {b["id"]: b for b in data.get("bases") or []}
    for b in manifest["oracle_spec"].get("basis") or []:
        entry = table.get(b)
        if entry is None:
            raise HarnessValidationError(
                f"oracle basis {b!r} is not in taxonomy/bases.yaml"
            )
        conf = entry.get("confidence")
        if conf == "established":
            continue
        if conf == "conventional":
            from financial_llm_harness.taxonomy.registry import checklist_signed_off

            if checklist_signed_off(entry):
                continue
        raise HarnessValidationError(
            f"confirmatory a_priori hypothesis cites basis {b!r} ({conf}) without a signed-off checklist (§27)"
        )


def freeze(
    bundle_dir: Path,
    *,
    code_ref: CodeRef | None = None,
    regenerated: Corpus | None = None,
    clock: Callable[[], datetime] | None = None,
) -> str:
    """Write ``manifest.lock`` after the §4 freeze conditions; return ``manifest_hash``."""
    lock = B.read_lock(bundle_dir)
    report = validate_bundle(
        bundle_dir,
        code_ref=code_ref,
        regenerated=regenerated,
        clock=clock,
        require_lock=False,
    )
    mh = str(report.manifest["manifest_hash"])
    if lock is not None and lock != mh:
        raise FrozenInputDrift([f"manifest.lock {lock} != manifest_hash {mh}"])
    if lock is None:
        B.atomic_write_text(bundle_dir / B.LOCK_FILE, mh + "\n")
    drift = B.frozen_input_mismatches(
        bundle_dir, current_code_ref(report.manifest, code_ref), require_lock=True
    )
    if drift:
        raise FrozenInputDrift(drift)
    return mh


def repository_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        return str(path)
