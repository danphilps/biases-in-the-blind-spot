"""``taxonomy expand``: registry items → spec files (spec §32B.2, D-15, D-19, E-31).

Every pair of every registry item is evaluated: a pair whose inputs all exist
is compiled to ``experiments/financial/generated/<faq_id>/<variant_id>/<iid>.yaml``
(written only for the selected items); every other pair becomes one row of
``expand_report.json`` with the sorted paths of its missing inputs. Nothing is
fabricated: a missing record, text, scorer or bank stays missing (§32B.2 step 7,
D-23).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.manifest import MANIFEST_ORDER
from financial_llm_harness.core.paths import repo_root
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.families import (
    TaskFamily,
    family_dir,
    load_defaults,
    load_family,
    load_mandate,
)
from financial_llm_harness.taxonomy import battery as BT
from financial_llm_harness.taxonomy.registry import (
    Item,
    Registry,
    canonical_pairs,
    experiment_key,
    family_listing,
    generated_dir,
    items_of_wave,
    load_yaml,
    validate_registry,
)

COPIED = (
    "task",
    "research_question",
    "data_classification",
    "scenario_spec",
    "controls",
    "prompt_spec",
    "paraphrase_spec",
    "response_spec",
    "feature_spec",
    "sampling_spec",
    "statistical_spec",
    "budget",
    "seeds",
)
HYPOTHESIS_ORDER = (
    "source",
    "primary_feature",
    "estimand",
    "level_arm",
    "effect_scale",
    "delta",
    "direction",
    "alpha",
    "power_target",
    "equivalence_margin",
    "correction_family",
    "missing_data",
)
DEFAULT_TAU = 0.90
DEFAULT_CORRECTNESS_DELTA = 0.10
REPORT_NAME = "expand_report.json"


@dataclass
class PairResult:
    faq_id: str
    variant_id: str
    intervention_id: str
    wave: str
    spec: dict[str, Any] | None = None
    reason: str | None = None
    missing: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return (
            generated_dir()
            / self.faq_id
            / self.variant_id
            / f"{self.intervention_id}.yaml"
        )

    def row(self) -> dict[str, Any]:
        return {
            "faq_id": self.faq_id,
            "variant_id": self.variant_id,
            "intervention_id": self.intervention_id,
            "wave": self.wave,
            "reason": self.reason,
            "missing": sorted(set(self.missing)),
        }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _family_of(item: Item, listing: Mapping[str, list[str]]) -> TaskFamily | None:
    fams = listing.get(item.id) or []
    return load_family(fams[0]) if len(fams) == 1 else None


def _entry(family: TaskFamily, faq_id: str) -> dict[str, Any] | None:
    entry = (load_defaults(family.family_id).get("entries") or {}).get(faq_id)
    return dict(entry) if isinstance(entry, Mapping) else None


def _variant_view(entry: Mapping[str, Any], variant_id: str) -> dict[str, Any]:
    """The entry with its variant-level record laid over it (§32B.2 step 6)."""
    view = {k: copy.deepcopy(v) for k, v in entry.items() if k != "variants"}
    over = ((entry.get("variants") or {}).get(variant_id)) or {}
    for k, v in over.items():
        view[k] = copy.deepcopy(v)
    return view


def _ref(family: TaskFamily, faq_id: str, *parts: str) -> str:
    return "/".join([f"families/{family.family_id}/defaults.yaml#{faq_id}", *parts])


def _mode(item: Item, reg: Registry, oracle_type: str, bases: list[str]) -> str:
    from financial_llm_harness.taxonomy.expert_panel import (
        checklist_signed_off,
        first_pass_contested,
    )

    if first_pass_contested(item.id):
        return "discovery"  # the panel's blind first pass left the basis contested (§T10.6)

    """§32B.2 step 5."""
    if item.data.get("priority") == "P3" or oracle_type == "DISCOVERY":
        return "discovery"
    if item.data.get("priority") not in ("P1", "P2"):
        return "discovery"
    for b in bases:
        conf = (reg.bases.get(b) or {}).get("confidence")
        if conf == "established":
            continue
        if conf == "conventional" and checklist_signed_off(b, item.id):
            continue
        return "discovery"
    return "confirmatory"


def _direction(otype: str, sign: int | None) -> str:
    """hypothesis.direction from the oracle (§4): the sign for DIRECTIONAL, none for
    INVARIANCE and LEVEL, two_sided otherwise (DISCOVERY, MAGNITUDE)."""
    if otype == "DIRECTIONAL" and sign is not None:
        return "positive" if int(sign) > 0 else "negative"
    if otype in ("INVARIANCE", "LEVEL"):
        return "none"
    return "two_sided"


def _hypothesis(base: Mapping[str, Any], direction: str, mode: str) -> dict[str, Any]:
    h = dict(base)
    h["source"] = "a_priori"
    h["direction"] = direction
    h["correction_family"] = "programme" if mode == "confirmatory" else "exploratory"
    return {k: h.get(k) for k in HYPOTHESIS_ORDER}


def _mandate_block(family: TaskFamily, mandate_id: str) -> dict[str, Any]:
    m = load_mandate(family.family_id, mandate_id)
    return {"id": mandate_id, "version": int(m["version"]), "text": str(m["text"])}


def _ordered(spec: Mapping[str, Any]) -> dict[str, Any]:
    keys = [k for k in MANIFEST_ORDER if k in spec]
    extra = [k for k in spec if k not in keys]
    if extra:
        raise HarnessValidationError(f"expand: unexpected spec keys {extra}")
    return {k: spec[k] for k in keys}


def _referenced_files(family: TaskFamily, view: Mapping[str, Any]) -> list[str]:
    """Template, paraphrase and contract files a spec needs that do not exist."""
    out: list[str] = []
    root = family_dir(family.family_id)
    for tid in (view.get("prompt_spec") or {}).get("template_family_ids") or []:
        if not (root / "templates" / f"{tid}.yaml").exists():
            out.append(f"families/{family.family_id}/templates/{tid}.yaml")
    for rel in (view.get("paraphrase_spec") or {}).get("template_files") or []:
        if not (family_dir(family.family_id).parent.parent / rel).exists():
            out.append(str(rel))
    cid = (view.get("response_spec") or {}).get("contract_id")
    if cid and not (root / "contracts" / f"{cid}.yaml").exists():
        out.append(f"families/{family.family_id}/contracts/{cid}.yaml")
    return out


def _common(
    item: Item, family: TaskFamily, variant_id: str, iid: str, mode: str
) -> dict[str, Any]:
    return {
        "experiment_key": experiment_key(item.id, variant_id, iid, item.version),
        "variant_id": variant_id,
        "mode": mode,
        "parent_discovery_experiment_id": None,
        "faq_id": item.id,
        "intervention_id": iid,
        "domain": item.data["domain"],
        "task_family": family.family_id,
    }


# ---------------------------------------------------------------------------
# One pair
# ---------------------------------------------------------------------------


def _relevant_entry(item: Item, iid: str) -> dict[str, Any]:
    for r in item.relevant:
        if r["id"] == iid:
            return dict(r)
    raise KeyError(iid)


def expand_relevant(
    reg: Registry,
    item: Item,
    family: TaskFamily,
    entry: Mapping[str, Any],
    variant: Mapping[str, Any],
    iid: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    vid = str(variant["variant_id"])
    view = _variant_view(entry, vid)
    missing: list[str] = []
    idef = family.interventions.get(iid)
    if idef is None:
        missing.append(f"families/{family.family_id}/family.yaml#interventions/{iid}")
    record = view.get(iid)
    if not isinstance(record, Mapping):
        missing.append(_ref(family, item.id, iid))
        record = {}
    for key in ("concept", "intervention_spec", "oracle_spec"):
        if record and record.get(key) is None:
            missing.append(_ref(family, item.id, iid, key))
    for key in COPIED:
        if view.get(key) is None:
            missing.append(_ref(family, item.id, key))
    if not isinstance(view.get("hypothesis"), Mapping):
        missing.append(_ref(family, item.id, "hypothesis"))
    mandate_id = variant.get("mandate") or view.get("mandate")
    if not mandate_id:
        missing.append(_ref(family, item.id, "mandate"))
    elif not (
        family_dir(family.family_id) / "mandates" / f"{mandate_id}.yaml"
    ).exists():
        missing.append(f"families/{family.family_id}/mandates/{mandate_id}.yaml")
    rel = _relevant_entry(item, iid)
    oracle = dict(record.get("oracle_spec") or {})
    if (
        rel.get("oracle_type") == "per_mandate"
        and oracle
        and oracle.get("mandate_id") != mandate_id
    ):
        missing.append(_ref(family, item.id, "variants", vid, iid, "oracle_spec"))
    if not oracle.get("type"):
        if record:
            missing.append(_ref(family, item.id, iid, "oracle_spec", "type"))
    missing += _referenced_files(family, view)
    elicitation = variant.get("elicitation")
    if elicitation and elicitation != (item.data.get("elicitation") or [None])[0]:
        over = (entry.get("variants") or {}).get(vid) or {}
        for key in ("response_spec", "feature_spec", "prompt_spec"):
            if key not in over:
                missing.append(_ref(family, item.id, "variants", vid, key))
    if missing:
        return None, missing
    assert idef is not None
    otype = str(oracle["type"])
    if (
        rel.get("oracle_type") not in (None, "per_mandate")
        and otype != rel["oracle_type"]
    ):
        if not (rel["oracle_type"] == "DIRECTIONAL" and otype == "LEVEL"):  # D-21
            raise HarnessValidationError(
                f"{_ref(family, item.id, iid, 'oracle_spec')}: type {otype} disagrees with faq.yaml {rel['oracle_type']} (§32B.2 step 4)"
            )
    item_sign = oracle.get("item_sign")
    if item_sign is None and rel.get("item_sign") not in (None, "per_mandate"):
        item_sign = 1 if str(rel["item_sign"]) == "+" else -1
    sign = None
    if otype == "DIRECTIONAL":
        if item_sign is None:
            return None, [_ref(family, item.id, iid, "oracle_spec", "item_sign")]
        sign = int(item_sign) * int(idef.concept_direction)
        oracle["sign"] = sign
        oracle["item_sign"] = int(item_sign)
    mode = _mode(item, reg, otype, list(oracle.get("basis") or []))
    spec = _common(item, family, vid, iid, mode)
    spec.update({k: copy.deepcopy(view[k]) for k in COPIED})
    spec["concept"] = record["concept"]
    spec["mandate"] = _mandate_block(family, str(mandate_id))
    spec["hypothesis"] = _hypothesis(view["hypothesis"], _direction(otype, sign), mode)
    spec["intervention_spec"] = copy.deepcopy(dict(record["intervention_spec"]))
    spec["oracle_spec"] = oracle
    spec["sampling_spec"] = {**spec["sampling_spec"], "panel": "ci_fake"}
    if mode == "discovery":
        spec["scenario_spec"] = {**spec["scenario_spec"], "seed_stream": "disc"}
    return _ordered(spec), []


def expand_correctness(
    reg: Registry,
    item: Item,
    family: TaskFamily,
    entry: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """§32B.2 step 1a [D-19]: the I.NONE correctness experiment of one variant."""
    vid = str(variant["variant_id"])
    view = _variant_view(entry, vid)
    missing: list[str] = []
    scorers = family.raw.get("scorers") or {}
    for check in item.correctness_checks:
        if check not in scorers:
            missing.append(f"families/{family.family_id}/family.yaml#scorers/{check}")
    feature = view.get("correctness_feature")
    if not feature:
        missing.append(_ref(family, item.id, "correctness_feature"))
    for key in COPIED:
        if view.get(key) is None:
            missing.append(_ref(family, item.id, key))
    mandate_id = variant.get("mandate") or view.get("mandate")
    if not mandate_id:
        missing.append(_ref(family, item.id, "mandate"))
    missing += _referenced_files(family, view)
    if missing:
        return None, missing
    tau = float(view.get("tau", DEFAULT_TAU))
    delta = float(view.get("correctness_delta", DEFAULT_CORRECTNESS_DELTA))
    bases = list(item.data.get("bases") or [])
    mode = _mode(item, reg, "LEVEL", bases)
    spec = _common(item, family, vid, "I.NONE", mode)
    spec.update({k: copy.deepcopy(view[k]) for k in COPIED})
    spec["concept"] = "correctness"
    spec["mandate"] = _mandate_block(family, str(mandate_id))
    hyp = dict(view.get("hypothesis") or {})
    hyp.update(
        primary_feature=str(feature),
        estimand="arm_level",
        level_arm="A",
        delta=delta,
    )
    spec["hypothesis"] = _hypothesis(hyp, "none", mode)
    spec["intervention_spec"] = None
    controls = dict(spec["controls"])
    spec["controls"] = {
        "null_arm": True,
        "placebo": None,
        "positive": None,
        "control_alpha": controls.get("control_alpha", 0.05),
        "null_margin": controls.get("null_margin", delta),
        "placebo_margin": None,
        "subset_n": None,
    }
    spec["oracle_spec"] = {
        "type": "LEVEL",
        "sign": None,
        "item_sign": None,
        "rule": f"Every correctness check of {item.id} passes: {', '.join(item.correctness_checks)}.",
        "mandate_id": str(mandate_id),
        "basis": bases,
        "provenance": f"faq.yaml#{item.id} correctness_checks (§32B.2 step 1a, D-19)",
        "version": 1,
        "confidence": "established" if mode == "confirmatory" else "conventional",
        "nodes": [],
        "tau": tau,
        "theta_star": None,
        "tol": None,
        "secondary": [],
    }
    if view.get("correctness_context"):
        spec["correctness_context"] = copy.deepcopy(view["correctness_context"])
    spec["feature_spec"] = [
        f for f in spec["feature_spec"] if f.get("analysis_role") != "attribution"
    ]
    spec["sampling_spec"] = {**spec["sampling_spec"], "panel": "ci_fake"}
    if mode == "discovery":
        spec["scenario_spec"] = {**spec["scenario_spec"], "seed_stream": "disc"}
    return _ordered(spec), []


def base_design(
    reg: Registry,
    item: Item,
    family: TaskFamily,
    entry: Mapping[str, Any] | None,
    variant: Mapping[str, Any],
) -> tuple[BT.BaseDesign | None, list[str]]:
    """§T4.4 'Which base': the canonical spec of the variant's first relevant intervention, else defaults."""
    vid = str(variant["variant_id"])
    first = item.relevant[0]["id"] if item.relevant else None
    for rel in (item.data.get("implementation") or {}).get("specs") or []:
        data = load_yaml(repo_root() / rel)
        if (
            isinstance(data, Mapping)
            and data.get("variant_id") == vid
            and data.get("intervention_id") == first
        ):
            spec = {k: v for k, v in data.items() if k != "provenance"}
            return BT.BaseDesign(spec, rel, first), []
    if entry is None:
        return None, ["base_design"]
    if first is not None:
        spec, missing = expand_relevant(reg, item, family, entry, variant, first)
    else:
        spec, missing = expand_correctness(reg, item, family, entry, variant)
    if spec is None:
        return None, ["base_design", *missing]
    return BT.BaseDesign(spec, _ref(family, item.id), first), []


def expand_battery(
    reg: Registry,
    item: Item,
    family: TaskFamily,
    entry: Mapping[str, Any] | None,
    variant: Mapping[str, Any],
    cue_id: str,
    cues: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    vid = str(variant["variant_id"])
    cue = cues.get(cue_id)
    if cue is None:
        return None, [f"taxonomy/interventions.yaml#{cue_id}"]
    base, missing = base_design(reg, item, family, entry, variant)
    if base is None:
        return None, missing
    mandate = load_mandate(family.family_id, base.spec["mandate"]["id"])
    design, missing = BT.battery_design(base, cue, family, item.interaction, mandate)
    if design is None:
        return None, missing
    otype = design["oracle_spec"]["type"]
    mode = _mode(item, reg, otype, ["B.irrelevant"])
    spec = {k: v for k, v in design.items() if k not in ("experiment_key", "mode")}
    spec.update(_common(item, family, vid, cue_id, mode))
    spec["hypothesis"] = _hypothesis(
        design["hypothesis"], _direction(otype, None), mode
    )
    spec["sampling_spec"] = {**spec["sampling_spec"], "panel": "ci_fake"}
    spec["parent_discovery_experiment_id"] = None
    if mode == "discovery":
        spec["scenario_spec"] = {**spec["scenario_spec"], "seed_stream": "disc"}
    return _ordered(spec), []


def expand_item(
    reg: Registry, item: Item, listing: Mapping[str, list[str]]
) -> list[PairResult]:
    claimed = {(v, i) for f, v, i in canonical_pairs(reg) if f == item.id}
    out: list[PairResult] = []
    family = _family_of(item, listing)
    entry = _entry(family, item.id) if family is not None else None
    status = item.data.get("status")
    cues = BT.load_cues()
    for variant in item.variants:
        vid = str(variant["variant_id"])
        pairs = [(vid, i) for i in item.intervention_ids()]
        if item.correctness_checks and not _skip_correctness(item, vid, family):
            pairs.append((vid, "I.NONE"))
        for _, iid in pairs:
            if (vid, iid) in claimed:
                continue
            r = PairResult(item.id, vid, iid, item.wave)
            if status == "specified" or family is None or entry is None:
                r.reason = "item_not_implemented"
                r.missing = [
                    f"families/*/family.yaml#faq_ids/{item.id}",
                    f"families/*/defaults.yaml#{item.id}",
                ]
                out.append(r)
                continue
            if iid == "I.NONE":
                spec, missing = expand_correctness(reg, item, family, entry, variant)
            elif iid in cues:  # a §T4.2 ID, in whichever column it is listed (step 1c)
                spec, missing = expand_battery(
                    reg, item, family, entry, variant, iid, cues
                )
            else:
                spec, missing = expand_relevant(reg, item, family, entry, variant, iid)
            if spec is None:
                r.reason = "unimplemented_content"
                r.missing = missing
            else:
                r.spec = spec
            out.append(r)
    return out


def _skip_correctness(item: Item, vid: str, family: TaskFamily | None) -> bool:
    """Skip rule (§32B.2 step 1a): the variant's canonical spec is LEVEL on the correctness feature.

    The family says which correctness checks a spec's primary conjoins
    (``primary_checks``); K.M.1's S5 primary conjoins its four hidden tests.
    """
    checks_of = getattr(family.module, "primary_checks", None) if family else None
    if checks_of is None:
        return False
    for rel in (item.data.get("implementation") or {}).get("specs") or []:
        data = load_yaml(repo_root() / rel)
        if not isinstance(data, Mapping) or data.get("variant_id") != vid:
            continue
        if data["oracle_spec"]["type"] != "LEVEL":
            continue
        if sorted(checks_of(data)) == sorted(item.correctness_checks):
            return True
    return False


# ---------------------------------------------------------------------------
# Writing (§32B.2 step 8)
# ---------------------------------------------------------------------------


def _version_of(key: Any) -> int | None:
    if not isinstance(key, str) or ".v" not in key:
        return None
    try:
        return int(key.rsplit(".v", 1)[1])
    except ValueError:
        return None


def plan_writes(
    results: list[PairResult], item_versions: Mapping[str, int], force: bool
) -> tuple[list[PairResult], list[str]]:
    writes: list[PairResult] = []
    refused: list[str] = []
    for r in results:
        assert r.spec is not None
        path = r.path
        if not path.exists():
            writes.append(r)
            continue
        try:
            old = yaml_io.load(path)
        except Exception:  # noqa: BLE001 - an unparseable file is refused below
            old = None
        if old == r.spec:
            if path.read_bytes() != yaml_io.dump(r.spec).encode("utf-8"):
                writes.append(r)
            continue
        old_v = (
            _version_of(old.get("experiment_key")) if isinstance(old, Mapping) else None
        )
        new_v = item_versions[r.faq_id]
        if old_v is None or old_v > new_v:
            refused.append(f"{path}: existing file has version {old_v} (§32B.2 step 8)")
        elif old_v == new_v:
            refused.append(
                f"{path}: different content under unchanged version v{new_v}; bump the item version"
            )
        elif not force:
            refused.append(f"{path}: lower version v{old_v}; rerun with --force")
        else:
            writes.append(r)
    return writes, refused


def run_expand(
    item_id: str | None,
    wave: str | None,
    *,
    allow_partial: bool = False,
    force: bool = False,
) -> int:
    reg = validate_registry()
    if item_id is not None:
        if item_id not in reg.items:
            raise HarnessValidationError(f"expand: unknown item {item_id!r}")
        selected = {item_id}
    elif wave is not None:
        selected = {i.id for i in items_of_wave(reg, wave)}
    else:
        raise HarnessValidationError("expand: give --item or --wave")
    listing = family_listing()
    results: list[PairResult] = []
    for item in reg.items.values():
        results += expand_item(reg, item, listing)
    rows = sorted(
        (r.row() for r in results if r.spec is None),
        key=lambda x: (x["faq_id"], x["variant_id"], x["intervention_id"]),
    )
    out_dir = generated_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (out_dir / REPORT_NAME).write_text(report, encoding="utf-8")
    ready = [r for r in results if r.spec is not None and r.faq_id in selected]
    writes, refused = plan_writes(
        ready, {i.id: i.version for i in reg.items.values()}, force
    )
    if refused:
        raise HarnessValidationError(
            "expand: refused to write:\n  " + "\n  ".join(refused)
        )
    for r in writes:
        r.path.parent.mkdir(parents=True, exist_ok=True)
        r.path.write_text(yaml_io.dump(r.spec), encoding="utf-8")
    blocking = [r for r in rows if r["faq_id"] in selected]
    print(
        f"expand: {len(ready)} spec(s) ready, {len(writes)} written; "
        f"{len(rows)} report row(s), {len(blocking)} for the selection"
    )
    if blocking and not allow_partial:
        return 2
    return 0
