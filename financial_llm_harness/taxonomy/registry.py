"""The Part II registry: faq.yaml, interventions.yaml, bases.yaml (spec §T1, §T4, §T7, §34).

``load_registry`` reads the three files; ``validate_registry`` applies every
``taxonomy validate`` rule of §T1, §34 and §T8 and raises HarnessValidationError
(exit 2) on the first set of violations, listing them all.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.code_ref import package_dir
from financial_llm_harness.core.paths import repo_root
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.taxonomy.expert_panel import (
    checklist_path,
    checklist_signed_off,
)

__all__ = ["checklist_signed_off"]

WAVES = ("V1", "V1.1", "V2", "V3")
STATUSES = ("specified", "implemented", "validated")
PRIORITIES = ("P1", "P2", "P3")
DOMAINS = ("fundamental", "quant_models", "ml", "code")
CONTEXTS = ("C1", "C2", "C3", "C4", "C5", "C5k", "C6", "C7", "C8", "C9")
ORACLE_TYPES = (
    "DIRECTIONAL",
    "INVARIANCE",
    "MAGNITUDE",
    "LEVEL",
    "DISCOVERY",
    "per_mandate",
)
OPERATIONS = (
    "ADD",
    "REMOVE",
    "REVERSE",
    "NUMERIC_SHIFT",
    "CATEGORY_CHANGE",
    "FRAMING_CHANGE",
    "CONSTRAINT_ADD",
    "CONSTRAINT_REMOVE",
    "POSITION_CHANGE",
)
# §T2.1 codes; §T5 also writes composites (X+J) and qualifiers (J (ranking))
INTERACTION = re.compile(r"^[JEXSDMPC](\+[JEXSDMPC])*( \([a-z ]+\))?$")
# §T6.1 hidden-test catalogue: a code item's correctness checks are these IDs
HIDDEN_TESTS = (
    "HT.LEAK",
    "HT.ALIGN",
    "HT.PERM",
    "HT.MONO",
    "HT.REF",
    "HT.PSD",
    "HT.ANTI",
    "HT.TRANS",
    "HT.SURV",
    "HT.PURGE",
    "HT.ANNUAL",
    "HT.COMPOUND",
    "HT.NAN",
    "HT.NONTRIV",
    "HT.SHAPE",
    "HT.FINITE",
)
VARIANT_ID = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
VARIANT_KEYS = {"variant_id", "mandate", "elicitation", "description"}
MAX_KEY = 60
# §34: the context objects each shared family serves (C9, the preamble, is universal)
FAMILY_CONTEXTS = {
    "fundamental_judgement": {"C1", "C2"},
    "fundamental_extraction": {"C1", "C3"},
    "corporate_action": {"C1", "C3"},
    "call_analysis": {"C4"},
    "broker_synthesis": {"C1", "C5", "C5k"},
    "quant_method_choice": {"C6"},
    "quant_dgp": {"C6", "C7"},
    "ml_design": {"C6", "C7"},
    "code_task": {"C7", "C8"},
}
ITEM_KEYS = (
    "id",
    "version",
    "wave",
    "domain",
    "category",
    "subcategory",
    "asker",
    "question",
    "interaction",
    "elicitation",
    "variants",
    "context",
    "contract",
    "observables",
    "assessments",
    "interventions",
    "local_interventions",
    "correctness_checks",
    "hypothesised_failure",
    "priority",
    "weight",
    "implementation",
    "status",
)
OPTIONAL_ITEM_KEYS = ("relevant_note", "bases", "basis_note")


@lru_cache(maxsize=1024)
def _load_cached(path_str: str, mtime_ns: int) -> Any:
    return yaml_io.load(Path(path_str))


def load_yaml(path: Path) -> Any:
    """A YAML file, cached by modification time (expand reads each spec many times)."""
    return copy.deepcopy(_load_cached(str(path), path.stat().st_mtime_ns))


def taxonomy_dir() -> Path:
    return package_dir() / "taxonomy"


def slug(text: str) -> str:
    """§32B.2 step 2: lower-case, every character outside [a-z0-9] → ``_``."""
    return re.sub(r"[^a-z0-9]", "_", text.lower())


def experiment_key(
    faq_id: str, variant_id: str, intervention_id: str, version: int
) -> str:
    return (
        f"{slug(faq_id)}__{slug(variant_id)}__{slug(intervention_id)}.v{int(version)}"
    )


@dataclass(frozen=True)
class Item:
    data: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def version(self) -> int:
        return int(self.data["version"])

    @property
    def wave(self) -> str:
        return str(self.data["wave"])

    @property
    def variants(self) -> list[dict[str, Any]]:
        """The declared variants in order, or the single implicit ``base`` (§T1)."""
        declared = list(self.data.get("variants") or [])
        return declared if declared else [{"variant_id": "base"}]

    @property
    def interaction(self) -> str:
        """The primary §T2.1 code: the first of a composite such as X+J."""
        return str(self.data.get("interaction"))[0]

    @property
    def relevant(self) -> list[dict[str, Any]]:
        return list((self.data.get("interventions") or {}).get("relevant") or [])

    @property
    def irrelevant(self) -> list[str]:
        return list((self.data.get("interventions") or {}).get("irrelevant") or [])

    @property
    def correctness_checks(self) -> list[str]:
        return list(self.data.get("correctness_checks") or [])

    def intervention_ids(self) -> list[str]:
        """§32B.2 step 1: relevant in listed order, then irrelevant in listed order."""
        return [r["id"] for r in self.relevant] + self.irrelevant

    def pairs(self) -> list[tuple[str, str]]:
        """(variant_id, intervention_id) pairs, with I.NONE where correctness checks exist."""
        out = []
        for v in self.variants:
            for iid in self.intervention_ids():
                out.append((v["variant_id"], iid))
            if self.correctness_checks:
                out.append((v["variant_id"], "I.NONE"))
        return out


@dataclass(frozen=True)
class Registry:
    items: dict[str, Item]
    interventions: dict[str, dict[str, Any]]
    bases: dict[str, dict[str, Any]]
    provenance: dict[str, Any] = field(default_factory=dict)


def _top(path: Path, list_key: str) -> tuple[dict[str, Any], list[Any]]:
    if not path.exists():
        raise HarnessValidationError(f"registry file {path.name} not found")
    data = yaml_io.load(path)
    if (
        not isinstance(data, Mapping)
        or set(data) != {"provenance", list_key}
        or not isinstance(data[list_key], list)
    ):
        raise HarnessValidationError(
            f"{path.name}: top level must be {{provenance, {list_key}: [...]}} (§T1)"
        )
    return dict(data["provenance"] or {}), list(data[list_key])


def load_registry(directory: Path | None = None) -> Registry:
    d = directory or taxonomy_dir()
    prov, items = _top(d / "faq.yaml", "items")
    _, ints = _top(d / "interventions.yaml", "interventions")
    _, bases = _top(d / "bases.yaml", "bases")
    out: dict[str, Item] = {}
    for raw in items:
        if not isinstance(raw, Mapping) or "id" not in raw:
            raise HarnessValidationError("faq.yaml: every item is a mapping with an id")
        if raw["id"] in out:
            raise HarnessValidationError(f"faq.yaml: item {raw['id']} listed twice")
        out[str(raw["id"])] = Item(dict(raw))
    return Registry(
        out,
        {str(i["id"]): dict(i) for i in ints},
        {str(b["id"]): dict(b) for b in bases},
        prov,
    )


@lru_cache(maxsize=1)
def cached_registry() -> Registry | None:
    if not (taxonomy_dir() / "faq.yaml").exists():
        return None
    return load_registry()


# ---------------------------------------------------------------------------
# taxonomy validate (§T1, §34, §T8)
# ---------------------------------------------------------------------------


def family_listing() -> dict[str, list[str]]:
    """item → the families whose family.yaml lists it in ``faq_ids``."""
    from financial_llm_harness.families import FAMILY_IDS, family_dir, load_family

    out: dict[str, list[str]] = {}
    for fid in FAMILY_IDS:
        if not (family_dir(fid) / "family.yaml").exists():
            continue
        for item in load_family(fid).faq_ids:
            out.setdefault(item, []).append(fid)
    return out


def _check_item_schema(item: Item, errors: list[str]) -> None:
    d = item.data
    where = f"faq.yaml#{item.id}"
    missing = [k for k in ITEM_KEYS if k not in d]
    extra = [k for k in d if k not in ITEM_KEYS and k not in OPTIONAL_ITEM_KEYS]
    if missing:
        errors.append(f"{where}: missing keys {missing}")
    if extra:
        errors.append(f"{where}: unknown keys {extra}")
    if d.get("wave") not in WAVES:
        errors.append(f"{where}: wave {d.get('wave')!r} not in {list(WAVES)} (§T1)")
    if d.get("status") not in STATUSES:
        errors.append(f"{where}: status {d.get('status')!r} invalid")
    if d.get("priority") not in PRIORITIES:
        errors.append(f"{where}: priority {d.get('priority')!r} invalid")
    if d.get("domain") not in DOMAINS:
        errors.append(f"{where}: domain {d.get('domain')!r} invalid")
    if d.get("domain") == "code":
        bad = [c for c in d.get("correctness_checks") or [] if c not in HIDDEN_TESTS]
        if bad:
            errors.append(
                f"{where}: correctness checks {bad} are not §T6.1 hidden tests"
            )
    if not INTERACTION.match(str(d.get("interaction"))):
        errors.append(
            f"{where}: interaction {d.get('interaction')!r} is not a §T2.1 code"
        )
    seen: set[str] = set()
    for v in d.get("variants") or []:
        if not isinstance(v, Mapping) or set(v) - VARIANT_KEYS or "variant_id" not in v:
            errors.append(
                f"{where}: variant {v!r} has keys outside {sorted(VARIANT_KEYS)} (§T1)"
            )
            continue
        vid = str(v["variant_id"])
        if not VARIANT_ID.match(vid):
            errors.append(f"{where}: malformed variant_id {vid!r} (§T1)")
        if vid in seen:
            errors.append(f"{where}: variant_id {vid!r} repeated (§T1)")
        seen.add(vid)
        el = v.get("elicitation")
        if el is not None and el not in (d.get("elicitation") or []):
            errors.append(
                f"{where}: variant {vid} elicitation {el!r} is not an item elicitation"
            )


def _check_references(reg: Registry, item: Item, errors: list[str]) -> None:
    where = f"faq.yaml#{item.id}"
    for c in item.data.get("context") or []:
        if c not in CONTEXTS:
            errors.append(f"{where}: context {c!r} unknown (§T3)")
    local = {li["id"]: li for li in item.data.get("local_interventions") or []}
    for lid, li in local.items():
        if not str(lid).startswith(f"I.ITEM.{item.id}."):
            errors.append(
                f"{where}: local intervention {lid} must be I.ITEM.{item.id}.<slug> (§T1)"
            )
        if li.get("op") not in OPERATIONS:
            errors.append(
                f"{where}: local intervention {lid} op {li.get('op')!r} not a §10 operation"
            )
    for r in item.relevant:
        iid = r["id"]
        if iid not in reg.interventions and iid not in local:
            errors.append(
                f"{where}: relevant {iid} resolves to neither §T4 nor a local intervention"
            )
        if r.get("oracle_type") is not None and r["oracle_type"] not in ORACLE_TYPES:
            errors.append(f"{where}: oracle type {r['oracle_type']!r} invalid")
    for iid in item.irrelevant:
        e = reg.interventions.get(iid)
        if e is None or e.get("kind") != "irrelevant":
            errors.append(f"{where}: irrelevant {iid} is not a §T4.2 cue")
    for b in _item_bases(item):
        if b not in reg.bases:
            errors.append(f"{where}: basis {b} unknown (§T7)")


def _item_bases(item: Item) -> list[str]:
    out = list(item.data.get("bases") or [])
    for r in item.relevant:
        if r.get("basis"):
            out.append(str(r["basis"]))
    return sorted(set(out))


def is_testable(reg: Registry, item: Item) -> bool:
    """§T1: an irrelevant intervention (an INVARIANCE relevant counts) and a relevant or a check."""
    invariant = any(r.get("oracle_type") == "INVARIANCE" for r in item.relevant)
    others = [r for r in item.relevant if r.get("oracle_type") != "INVARIANCE"]
    return (bool(item.irrelevant) or invariant) and (
        bool(others) or bool(item.correctness_checks) or invariant
    )


def validate_registry(reg: Registry | None = None) -> Registry:
    """Every ``taxonomy validate`` rule; raises exit 2 listing all violations."""
    reg = reg or load_registry()
    errors: list[str] = []
    listing = family_listing()
    for item in reg.items.values():
        _check_item_schema(item, errors)
        _check_references(reg, item, errors)
        if not is_testable(reg, item):
            errors.append(f"faq.yaml#{item.id}: not testable (§T1)")
        for b in _item_bases(item):
            if (reg.bases.get(b) or {}).get("confidence") == "conventional":
                path = checklist_path(b, item.id)
                if not path.exists():
                    errors.append(
                        f"faq.yaml#{item.id}: conventional basis {b} has no checklist {path.relative_to(taxonomy_dir().parent)} (§T7)"
                    )
        fams = listing.get(item.id, [])
        if len(fams) > 1:
            errors.append(f"faq.yaml#{item.id}: listed by two families {fams} (§34)")
        if len(fams) == 1:
            allowed = FAMILY_CONTEXTS.get(fams[0], set()) | {"C9"}
            bad = [c for c in item.data.get("context") or [] if c not in allowed]
            if bad:
                errors.append(
                    f"faq.yaml#{item.id}: contexts {bad} outside family {fams[0]} (§34)"
                )
        if (
            item.wave == "V1"
            and (item.data.get("implementation") or {}).get("kind") != "canonical_spec"
        ):
            errors.append(f"faq.yaml#{item.id}: a V1 item must be canonical_spec (§T8)")
        for vid, iid in item.pairs():
            key = experiment_key(item.id, vid, iid, item.version)
            if len(key) > MAX_KEY:
                errors.append(
                    f"faq.yaml#{item.id}: key {key} over {MAX_KEY} characters"
                )
        _check_status(item, fams, errors)
        if fams:
            _check_defaults_oracles(item, fams[0], errors)
    _check_family_contexts(errors)
    _check_canonical_specs(reg, errors)
    _check_generated(reg, errors)
    if errors:
        raise HarnessValidationError("taxonomy validate:\n  " + "\n  ".join(errors))
    return reg


def _check_family_contexts(errors: list[str]) -> None:
    """Each family's declared ``contexts`` is its §34 row."""
    from financial_llm_harness.families import FAMILY_IDS, family_dir, load_family

    for fid in FAMILY_IDS:
        if not (family_dir(fid) / "family.yaml").exists():
            continue
        declared = set(load_family(fid).raw.get("contexts") or [])
        if declared != FAMILY_CONTEXTS[fid]:
            errors.append(
                f"families/{fid}/family.yaml: contexts {sorted(declared)} != §34 row {sorted(FAMILY_CONTEXTS[fid])}"
            )


def _check_status(item: Item, fams: list[str], errors: list[str]) -> None:
    from financial_llm_harness.families import load_defaults

    implemented = bool(fams) and item.id in (
        load_defaults(fams[0]).get("entries") or {}
    )
    status = item.data.get("status")
    if status in ("implemented", "validated") and not implemented:
        errors.append(
            f"faq.yaml#{item.id}: status {status} needs a family listing it and a defaults.yaml entry (§32B.2 step 0)"
        )
    if status == "specified" and implemented:
        errors.append(
            f"faq.yaml#{item.id}: implemented by {fams[0]} but declared specified"
        )


def _check_defaults_oracles(item: Item, family_id: str, errors: list[str]) -> None:
    """A defaults.yaml oracle record's type agrees with faq.yaml (§32B.2 step 4).

    ``per_mandate`` takes its type from the record, which must name the
    variant's mandate; LEVEL may stand for DIRECTIONAL where D-21 applies.
    """
    from financial_llm_harness.families import load_defaults

    entry = (load_defaults(family_id).get("entries") or {}).get(item.id)
    if not isinstance(entry, Mapping):
        return
    mandates = {
        str(v["variant_id"]): v.get("mandate") or entry.get("mandate")
        for v in item.variants
    }
    for r in item.relevant:
        iid, declared = r["id"], r.get("oracle_type")
        records = []
        if isinstance(entry.get(iid), Mapping):
            records.append(("", entry[iid]))
        for vid, over in (entry.get("variants") or {}).items():
            if isinstance(over, Mapping) and isinstance(over.get(iid), Mapping):
                records.append((vid, over[iid]))
        for vid, rec in records:
            otype = (rec.get("oracle_spec") or {}).get("type")
            where = f"families/{family_id}/defaults.yaml#{item.id}/{vid + '/' if vid else ''}{iid}"
            if declared == "per_mandate":
                named = (rec.get("oracle_spec") or {}).get("mandate_id")
                if vid and named != mandates.get(vid):
                    errors.append(
                        f"{where}: per_mandate record names {named}, variant mandate is {mandates.get(vid)}"
                    )
                continue
            if (
                otype is not None
                and otype != declared
                and not (declared == "DIRECTIONAL" and otype == "LEVEL")
            ):
                errors.append(
                    f"{where}: oracle type {otype} disagrees with faq.yaml {declared} (§32B.2 step 4)"
                )


def spec_files() -> list[Path]:
    return sorted((repo_root() / "experiments" / "financial" / "specs").glob("*.yaml"))


def _check_canonical_specs(reg: Registry, errors: list[str]) -> None:
    claimed: dict[tuple[str, str, str], str] = {}
    listed: set[str] = set()
    for item in reg.items.values():
        impl = item.data.get("implementation") or {}
        kind = impl.get("kind", "generated")
        specs = list(impl.get("specs") or [])
        if kind not in ("generated", "canonical_spec"):
            errors.append(f"faq.yaml#{item.id}: implementation.kind {kind!r} invalid")
        if kind == "generated" and specs:
            errors.append(f"faq.yaml#{item.id}: a generated item lists no specs")
        pairs = set(item.pairs())
        for rel in specs:
            listed.add(rel)
            path = repo_root() / rel
            if not path.exists():
                errors.append(f"faq.yaml#{item.id}: canonical spec {rel} missing (§T1)")
                continue
            data = load_yaml(path)
            if not isinstance(data, Mapping):
                errors.append(f"{rel}: not a mapping")
                continue
            if data.get("faq_id") != item.id:
                errors.append(f"{rel}: faq_id {data.get('faq_id')!r} != {item.id}")
                continue
            pair = (str(data.get("variant_id")), str(data.get("intervention_id")))
            if pair not in pairs:
                errors.append(
                    f"{rel}: ({pair[0]}, {pair[1]}) is no pair of {item.id} (§T1)"
                )
                continue
            key = experiment_key(item.id, pair[0], pair[1], item.version)
            if data.get("experiment_key") != key:
                errors.append(
                    f"{rel}: experiment_key {data.get('experiment_key')!r} != {key} (§32B.2 step 2)"
                )
            full = (item.id, *pair)
            if full in claimed:
                errors.append(
                    f"{rel}: pair {full} already claimed by {claimed[full]} (§T1)"
                )
            claimed[full] = rel
    for path in spec_files():
        data = load_yaml(path)
        if not isinstance(data, Mapping) or data.get("faq_id") not in reg.items:
            continue
        item = reg.items[str(data["faq_id"])]
        pair = (str(data.get("variant_id")), str(data.get("intervention_id")))
        rel = path.relative_to(repo_root()).as_posix()
        if pair in set(item.pairs()) and rel not in listed:
            errors.append(
                f"{rel}: claims a pair of {item.id} but is not in its implementation.specs (§T1)"
            )


def generated_dir() -> Path:
    return repo_root() / "experiments" / "financial" / "generated"


def canonical_pairs(reg: Registry) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for item in reg.items.values():
        for rel in (item.data.get("implementation") or {}).get("specs") or []:
            path = repo_root() / rel
            if path.exists():
                data = load_yaml(path)
                if isinstance(data, Mapping):
                    out.add(
                        (
                            item.id,
                            str(data.get("variant_id")),
                            str(data.get("intervention_id")),
                        )
                    )
    return out


def _check_generated(reg: Registry, errors: list[str]) -> None:
    root = generated_dir()
    if not root.exists():
        return
    claimed = canonical_pairs(reg)
    for path in sorted(root.rglob("*.yaml")):
        rel = path.relative_to(root).parts
        if len(rel) != 3:
            continue
        faq_id, vid, fname = rel
        iid = fname[: -len(".yaml")]
        item = reg.items.get(faq_id)
        if (
            item is None
            or (vid, iid) not in set(item.pairs())
            or (faq_id, vid, iid) in claimed
        ):
            errors.append(f"generated/{'/'.join(rel)}: orphan generated file (§T1)")


def items_of_wave(reg: Registry, wave: str) -> list[Item]:
    if wave not in WAVES:
        raise HarnessValidationError(f"unknown wave {wave!r}")
    return [i for i in reg.items.values() if i.wave == wave]


def short_form(iid: str) -> str:
    return iid


def references(items: Iterable[Item]) -> list[str]:
    return sorted(
        {r["id"] for i in items for r in i.relevant}
        | {c for i in items for c in i.irrelevant}
    )
