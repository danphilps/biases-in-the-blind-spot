"""``generate``: spec → frozen-bundle files without the lock (spec §4, §0.10, §31A L1).

Resolves the panel, probes, evaluators and enforcement; builds the pre-hash
corpus; hashes every consumed asset and the source tree; then computes
``design_hash``, ``experiment_id`` and ``manifest_hash``, in that order.
"""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.canonical_json import sha256_canonical
from financial_llm_harness.core.code_ref import CodeRef, compute_code_ref
from financial_llm_harness.core.hashes import compute_design_hash, compute_manifest_hash
from financial_llm_harness.core.ids import experiment_id as make_experiment_id
from financial_llm_harness.core.manifest import (
    SCHEMA_VERSION,
    check_design,
    check_spec_keys,
    load_yaml_mapping,
    order_manifest,
)
from financial_llm_harness.core.paths import probes_dir, repo_root
from financial_llm_harness.errors import HarnessValidationError, StaleProbe
from financial_llm_harness.families import (
    family_dir,
    load_defaults,
    load_mandate,
    provenance_of,
    rel_to_package,
)
from financial_llm_harness.generation.corpus import Corpus, CueHook, build_corpus
from financial_llm_harness.generation.provenance import (
    Provenance,
    author_families,
    parse_provenance,
)
from financial_llm_harness.models.base import (
    ModelCapabilities,
    load_providers,
    profile_sha256,
    request_profile,
    resolve_profile_id,
)
from financial_llm_harness.models.fake import fake_capabilities
from financial_llm_harness.models.panels import Panel, load_panel

GENERATOR_VERSION = "structured_world@1"
SEMANTIC_EXTRACTORS = ("semantic_verbalisation", "semantic_inclusion")


@dataclass
class GenerateResult:
    bundle_dir: Path
    manifest: dict[str, Any]
    assets: list[dict[str, Any]]
    corpus: Corpus
    provenances: list[Provenance]


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Panel, probes and evaluators (§4, §20.4, §20.9)
# ---------------------------------------------------------------------------


def latest_probe(model_key: str) -> Path | None:
    d = probes_dir() / model_key
    if not d.is_dir():
        return None
    files = sorted(p for p in d.glob("*.json") if p.is_file())
    return files[-1] if files else None


def resolve_model(
    entry: Mapping[str, Any], require_probe: bool = True
) -> tuple[dict[str, Any], Path | None]:
    """A manifest ``models`` entry with its frozen capabilities, and the probe path."""
    provider = entry["provider"]
    profile_id = resolve_profile_id(
        provider, entry["transport"], entry.get("request_profile")
    )
    profile = request_profile(provider, profile_id)
    ref = {
        "id": profile_id,
        "profile_version": int(profile["profile_version"]),
        "sha256": profile_sha256(provider, profile_id),
    }
    probe_path: Path | None = None
    if provider == "fake":
        caps = fake_capabilities(entry)
        probe_ref = None
        revision = caps.revision
    else:
        probe_path = latest_probe(entry["model_key"])
        if probe_path is None:
            if require_probe:
                raise StaleProbe(
                    f"no probe file for {entry['model_key']} under {probes_dir()} (§20.4)"
                )
            raise StaleProbe(f"no probe for {entry['model_key']}")
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        caps = ModelCapabilities.from_dict(probe["capabilities"])
        probe_ref = {
            "path": f"probes/{entry['model_key']}.json",
            "sha256": B.file_sha256(probe_path),
            "date": probe_path.stem,
            "request_profile_id": probe.get("request_profile_id"),
            "request_profile_sha256": probe.get("request_profile_sha256"),
        }
        revision = caps.revision
    model = {
        "model_key": entry["model_key"],
        "provider": provider,
        "family": entry["family"],
        "model": entry["model"],
        "region": entry.get("region"),
        "transport": entry["transport"],
        "thinking": copy.deepcopy(entry["thinking"]),
        "revision": revision,
        "request_profile": ref,
        "upstream_routing": copy.deepcopy(entry.get("upstream_routing")),
        "params": copy.deepcopy(entry.get("params") or {}),
        "batch": bool(entry.get("batch")),
        "probe_ref": probe_ref,
        "capabilities": caps.to_dict(),
    }
    return model, probe_path


def resolve_evaluators(
    design: Mapping[str, Any], panel: Panel
) -> tuple[list[dict[str, Any]], list[B.AssetRow], Path | None, dict[str, Any] | None]:
    """Manifest ``evaluators`` for every semantic feature (§16)."""
    semantic = [
        f for f in design["feature_spec"] if f.get("extractor") in SEMANTIC_EXTRACTORS
    ]
    if not semantic:
        return [], [], None, None
    ev = panel.evaluator
    if ev is None:
        raise HarnessValidationError(
            "a semantic feature needs the panel's evaluator entry (§16, §20.9)"
        )
    import financial_llm_harness.evaluation.semantic as sem

    model, probe_path = resolve_model(ev)
    out = []
    rows = [
        B.make_asset("evaluator_prompt", sem.PROMPT_OBJECT, "package"),
        B.make_asset("evaluator_parser", sem.PARSER_OBJECT, "package"),
    ]
    for f in semantic:
        version = sem.evaluator_version(model, len(semantic_calls(model)))
        cal = sem.find_calibration(f["feature_id"], version, ev["provider"])
        out.append(
            {
                "feature_id": f["feature_id"],
                "model_key": ev["model_key"],
                "provider": ev["provider"],
                "model": ev["model"],
                "family": ev["family"],
                "prompt_hash": sem.prompt_hash(),
                "parser_version": sem.PARSER_VERSION,
                "evaluator_version": version,
                "calibration_ref": cal,
            }
        )
    return out, rows, probe_path, model


def semantic_calls(model: Mapping[str, Any]) -> list[int]:
    """Call numbers per evaluation: one at temperature 0 if accepted, otherwise three (§16)."""
    caps = ModelCapabilities.from_dict(model["capabilities"])
    return [0] if caps.param_status("temperature") != "rejected" else [0, 1, 2]


# ---------------------------------------------------------------------------
# Assets and provenance (§0.10, §8.1)
# ---------------------------------------------------------------------------


def _registry_file(rel: str) -> Path:
    from financial_llm_harness.core.code_ref import package_dir

    return package_dir() / rel


def _registry_has(rel: str, list_key: str, ident: str) -> bool:
    path = _registry_file(rel)
    if not path.exists():
        return False
    data = yaml_io.load(path)
    return any(
        isinstance(i, Mapping) and i.get("id") == ident
        for i in (data.get(list_key) or [])
    )


def collect_assets(
    design: Mapping[str, Any],
    corpus: Corpus,
    panel: Panel,
    models: list[dict[str, Any]],
    extra: list[B.AssetRow],
) -> tuple[list[B.AssetRow], list[Provenance]]:
    fam = corpus.family
    fid = fam.family_id
    rows: list[B.AssetRow] = list(extra)
    provs: list[Provenance] = []
    for logical in corpus.template_logicals:
        rows.append(B.make_asset("template", logical, "package"))
    for logical in corpus.paraphrase_logicals:
        rows.append(B.make_asset("paraphrase", logical, "package"))
    for logical in corpus.bank_files:
        rows.append(B.make_asset("bank", logical, "package"))
    mandate_logical = f"families/{fid}/mandates/{design['mandate']['id']}.yaml"
    rows.append(B.make_asset("mandate", mandate_logical, "package"))
    for logical in corpus.clause_logicals:
        rows.append(B.make_asset("mandate", logical, "package"))
    rows.append(B.make_asset("contract", corpus.contract_logical, "package"))
    rows.append(
        B.make_asset("family_config", f"families/{fid}/family.yaml#core", "package")
    )
    if design["intervention_id"] in fam.interventions:
        rows.append(
            B.make_asset(
                "family_config",
                f"families/{fid}/family.yaml#interventions/{design['intervention_id']}",
                "package",
            )
        )
    mandate_id = design["oracle_spec"].get("mandate_id")
    if mandate_id in fam.oracles:
        rows.append(
            B.make_asset(
                "oracle", f"families/{fid}/family.yaml#oracles/{mandate_id}", "package"
            )
        )
    defaults = load_defaults(fid)
    entry = (defaults.get("entries") or {}).get(design["faq_id"])
    if entry is not None:
        rows.append(
            B.make_asset(
                "family_config",
                f"families/{fid}/defaults.yaml#{design['faq_id']}",
                "package",
            )
        )
        if design["variant_id"] in (entry.get("variants") or {}):
            rows.append(
                B.make_asset(
                    "family_config",
                    f"families/{fid}/defaults.yaml#{design['faq_id']}/{design['variant_id']}",
                    "package",
                )
            )
    if design["faq_id"] != "none" and _registry_has(
        "taxonomy/faq.yaml", "items", design["faq_id"]
    ):
        rows.append(
            B.make_asset("oracle", f"taxonomy/faq.yaml#{design['faq_id']}", "package")
        )
    bases = list(design["oracle_spec"].get("basis") or [])
    pos = (design["controls"] or {}).get("positive") or {}
    bases += list((pos.get("oracle") or {}).get("basis") or [])
    for b in sorted(set(bases)):
        if _registry_has("taxonomy/bases.yaml", "bases", b):
            rows.append(B.make_asset("oracle", f"taxonomy/bases.yaml#{b}", "package"))
    battery = design.get("battery")
    if battery and _registry_has(
        "taxonomy/interventions.yaml", "interventions", design["intervention_id"]
    ):
        rows.append(
            B.make_asset(
                "family_config",
                f"taxonomy/interventions.yaml#{design['intervention_id']}",
                "package",
            )
        )
    panel_logical = (
        panel.path.resolve().relative_to(repo_root().resolve()).as_posix()
        if _is_under(panel.path, repo_root())
        else str(panel.path)
    )
    rows.append(B.make_asset("panel", panel_logical, "repository"))
    for m in models:
        rows.append(
            B.make_asset(
                "request_profile",
                f"models/providers.yaml#{m['request_profile']['id']}",
                "package",
            )
        )
    # provenance blocks (§8.1, §2.7)
    for logical, data in corpus.file_data.items():
        provs.append(parse_provenance(provenance_of(data), logical, "text"))
    for logical in corpus.bank_files:
        data = yaml_io.load(_registry_file(logical))
        provs.append(parse_provenance(provenance_of(data), logical, "text"))
    provs.append(
        parse_provenance(
            provenance_of(load_mandate(fid, design["mandate"]["id"])),
            mandate_logical,
            "text",
        )
    )
    provs.append(
        parse_provenance(
            fam.raw.get("provenance"), f"families/{fid}/family.yaml", "oracle"
        )
    )
    if entry is not None:
        provs.append(
            parse_provenance(
                defaults.get("provenance"), f"families/{fid}/defaults.yaml", "oracle"
            )
        )
    for rel in ("taxonomy/faq.yaml", "taxonomy/bases.yaml"):
        if _registry_file(rel).exists() and any(
            r.logical_name.startswith(rel + "#") for r in rows
        ):
            provs.append(
                parse_provenance(
                    provenance_of(yaml_io.load(_registry_file(rel))), rel, "oracle"
                )
            )
    if battery and any(
        r.logical_name.startswith("taxonomy/interventions.yaml#") for r in rows
    ):
        provs.append(
            parse_provenance(
                provenance_of(
                    yaml_io.load(_registry_file("taxonomy/interventions.yaml"))
                ),
                "taxonomy/interventions.yaml",
                "oracle",
            )
        )
    for checklist in _checklists_for(bases):
        provs.append(
            parse_provenance(
                provenance_of(yaml_io.load(checklist)),
                rel_to_package(checklist),
                "oracle",
            )
        )
        rows.append(B.make_asset("oracle", rel_to_package(checklist), "package"))
    return B.sorted_assets(rows), provs


def _checklists_for(bases: list[str]) -> list[Path]:
    out = []
    path = _registry_file("taxonomy/bases.yaml")
    if not path.exists():
        return out
    data = yaml_io.load(path)
    for b in data.get("bases") or []:
        if b.get("id") in bases and b.get("checklist"):
            p = _registry_file(str(b["checklist"]))
            if p.exists():
                out.append(p)
    return sorted(set(out))


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_enforcement(enforcement: str, models: list[dict[str, Any]]) -> str:
    """E-10: auto → provider_schema iff every panel model's frozen probe reports schema enforced."""
    if enforcement != "auto":
        return enforcement
    ok = all(bool(m["capabilities"].get("supports_schema")) for m in models)
    return "provider_schema" if ok else "prompt_only"


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def generate(
    spec_path: Path,
    out_dir: Path,
    panel_name: str | None = None,
    *,
    code_ref: CodeRef | None = None,
    clock: Callable[[], datetime] | None = None,
    cue_hook: CueHook | None = None,
    spec_data: Mapping[str, Any] | None = None,
) -> GenerateResult:
    """Write the §0.10 bundle (without the lock) into ``out_dir``."""
    spec = dict(spec_data) if spec_data is not None else load_yaml_mapping(spec_path)
    where = str(spec_path)
    check_spec_keys(spec, where)
    check_design(spec, where)
    family_dir(spec["task_family"])
    panel = load_panel(panel_name or spec["sampling_spec"]["panel"])
    load_providers()
    models: list[dict[str, Any]] = []
    probe_paths: dict[str, Path] = {}
    for entry in panel.models:
        model, pp = resolve_model(entry)
        models.append(model)
        if pp is not None:
            probe_paths[entry["model_key"]] = pp
    evaluators, ev_rows, ev_probe, ev_model = resolve_evaluators(spec, panel)
    if ev_probe is not None and ev_model is not None:
        probe_paths[ev_model["model_key"]] = ev_probe
    if cue_hook is None and spec.get("battery"):
        from financial_llm_harness.taxonomy.battery import cue_hook_for

        cue_hook = cue_hook_for(spec)
    corpus = build_corpus(spec, cue_hook)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_rows: list[B.AssetRow] = []
    if probe_paths:
        (out_dir / B.PROBES_DIR).mkdir(exist_ok=True)
    for key, pp in sorted(probe_paths.items()):
        dest = out_dir / B.PROBES_DIR / f"{key}.json"
        shutil.copyfile(pp, dest)
        probe_rows.append(
            B.make_asset("probe", f"probes/{key}.json", "generated", out_dir)
        )
    extra_rows = probe_rows + ev_rows + _code_rows(spec, corpus)
    assets, provs = collect_assets(
        spec, corpus, panel, models + ([ev_model] if ev_model else []), extra_rows
    )
    if spec["task_family"] == "code_task":
        from financial_llm_harness.families.code_task.family import code_provenances

        provs += code_provenances(spec)
    if spec.get("provenance") is not None:
        provs.append(parse_provenance(spec["provenance"], where, "both"))
    target_families = sorted({m["family"] for m in models})
    independence = {
        "target_families": target_families,
        "text_author_families": author_families(provs, ("text", "both")),
        "oracle_author_families": author_families(provs, ("oracle", "both")),
        "evaluator_families": sorted({e["family"] for e in evaluators}),
    }
    manifest: dict[str, Any] = {k: copy.deepcopy(v) for k, v in spec.items()}
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["arms"] = list(corpus.design_arms)
    manifest["controls"] = dict(manifest["controls"])
    manifest["controls"]["subset_n"] = corpus.subset_n
    rs = dict(manifest["response_spec"])
    rs["enforcement_resolved"] = resolve_enforcement(
        rs["enforcement"], models + ([ev_model] if ev_model else [])
    )
    manifest["response_spec"] = rs
    ss = dict(manifest["sampling_spec"])
    ss.pop("panel", None)
    ss["panel_ref"] = {"name": panel.name, "sha256": B.file_sha256(panel.path)}
    ss["models"] = models
    manifest["sampling_spec"] = ss
    manifest["evaluators"] = evaluators
    manifest["independence"] = independence
    manifest["versions"] = {
        "generator": GENERATOR_VERSION,
        "templates": spec["prompt_spec"]["template_version"],
        "oracles": spec["oracle_spec"]["version"],
    }
    cases_text = B.jsonl_text(corpus.cases)
    rend_text = B.jsonl_text(corpus.renderings)
    assets_dicts = [a.to_dict() for a in assets]
    assets_text = B.json_text(assets_dicts)
    B.atomic_write_text(out_dir / B.CASES_FILE, cases_text)
    B.atomic_write_text(out_dir / B.RENDERINGS_FILE, rend_text)
    B.atomic_write_text(out_dir / B.ASSETS_FILE, assets_text)
    B.atomic_write_text(out_dir / B.CORPUS_REPORT_FILE, B.json_text(corpus.report))
    manifest["corpus"] = {
        "cases": {
            "path": B.CASES_FILE,
            "sha256": B.file_sha256(out_dir / B.CASES_FILE),
        },
        "renderings": {
            "path": B.RENDERINGS_FILE,
            "sha256": B.file_sha256(out_dir / B.RENDERINGS_FILE),
        },
        "assets": {
            "path": B.ASSETS_FILE,
            "sha256": B.file_sha256(out_dir / B.ASSETS_FILE),
        },
    }
    uses_legacy = bool(evaluators)
    ref = code_ref if code_ref is not None else compute_code_ref(uses_legacy)
    if not uses_legacy and code_ref is None:
        ref = CodeRef(
            ref.harness_git_sha, ref.harness_tree_sha256, None, None, ref.dirty
        )
    manifest["code_ref"] = ref.to_dict()
    non_fake = any(m["provider"] != "fake" for m in models) or (
        ev_model is not None and ev_model["provider"] != "fake"
    )
    if spec["mode"] == "confirmatory" and non_fake and ref.dirty:
        raise HarnessValidationError(
            "clean-tree rule: a confirmatory manifest with a non-fake target or evaluator cannot be "
            "frozen from a dirty or unidentifiable source tree (§0.10)"
        )
    now = (clock or utc_now)()
    manifest["design_hash"] = compute_design_hash(manifest, assets_dicts)
    manifest["experiment_id"] = make_experiment_id(manifest["design_hash"])
    manifest["created_at"] = iso_z(now)
    manifest["manifest_hash"] = "pending"
    manifest = order_manifest(manifest)
    manifest["manifest_hash"] = compute_manifest_hash(manifest)
    B.atomic_write_text(out_dir / B.MANIFEST_FILE, yaml_io.dump(manifest))
    return GenerateResult(out_dir, manifest, assets_dicts, corpus, provs)


def _code_rows(spec: Mapping[str, Any], corpus: Corpus) -> list[B.AssetRow]:
    """hidden_tests, sandbox_recipe and code-fixture assets for code tasks (§0.10, §13.4)."""
    if spec["task_family"] != "code_task":
        return []
    from financial_llm_harness.families.code_task.family import code_asset_rows

    return code_asset_rows(spec)


def manifest_hash_of(manifest: Mapping[str, Any]) -> str:
    return sha256_canonical(
        {
            k: v
            for k, v in manifest.items()
            if k not in ("manifest_hash", "experiment_id", "created_at")
        }
    )
