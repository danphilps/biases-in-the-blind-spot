"""Expert-panel sign-off (spec §T7 checklists, §T10.1, §T10.6, D-10, D-13).

``taxonomy/expert_panel.yaml`` holds the seats, the named members with their
roles and conflict declarations, and the sign-off log. A checklist is signed
off when it has at least one item, every item has a rationale, and at least
two panel members from at least two different roles mapped to the checklist's
domain have approved it, none of them conflicted on it. Without named members
(D-13) nothing is signed off, so every conventional basis keeps its item in
discovery mode (§32B.2 step 5).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.code_ref import package_dir
from financial_llm_harness.errors import HarnessValidationError

ROLES = (
    "quant_analyst",
    "quant_pm",
    "fundamental_analyst",
    "fundamental_pm",
    "curriculum_author",
    "pension_cio",
    "ml_architect",
    "ml_developer",
)
# §T10.1 panel responsibilities: which roles sign which domain's checklists
CHECKLIST_ROLES = {
    "fundamental": {"fundamental_analyst", "fundamental_pm", "curriculum_author"},
    "quant_models": {"quant_analyst", "quant_pm", "curriculum_author"},
    "ml": {"quant_pm", "ml_architect", "ml_developer"},
    "code": {"quant_analyst", "ml_developer"},
}
MIN_APPROVALS = 2
MIN_ROLES = 2


def panel_path() -> Path:
    return package_dir() / "taxonomy" / "expert_panel.yaml"


def checklist_dir() -> Path:
    return package_dir() / "taxonomy" / "checklists"


def load_panel() -> dict[str, Any]:
    path = panel_path()
    if not path.exists():
        return {"members": []}
    data = yaml_io.load(path)
    if not isinstance(data, Mapping):
        raise HarnessValidationError(f"{path}: not a mapping")
    for m in data.get("members") or []:
        if m.get("role") not in ROLES:
            raise HarnessValidationError(
                f"{path}: member {m.get('id')!r} has role {m.get('role')!r} outside §T10.1"
            )
    return dict(data)


def checklist_path(basis_id: str, faq_id: str | None = None) -> Path:
    """``checklists/<basis>.yaml``; B.conv is per item: ``checklists/B.conv/<faq_id>.yaml``."""
    if basis_id == "B.conv":
        return checklist_dir() / "B.conv" / f"{faq_id}.yaml"
    return checklist_dir() / f"{basis_id}.yaml"


def is_signed_off(checklist: Mapping[str, Any], panel: Mapping[str, Any]) -> bool:
    """The §T10.6 rule on one checklist."""
    items = list(checklist.get("items") or [])
    if not items or any(not str(i.get("rationale") or "").strip() for i in items):
        return False
    domains = checklist.get("domain")
    domains = [domains] if isinstance(domains, str) else list(domains or [])
    roles_for = set().union(*(CHECKLIST_ROLES.get(str(d), set()) for d in domains))
    subject = str(checklist.get("faq_id") or checklist.get("basis") or "")
    members = {str(m["id"]): m for m in panel.get("members") or []}
    approvers: dict[str, str] = {}
    for s in checklist.get("signoffs") or []:
        m = members.get(str(s.get("member")))
        if m is None or m.get("role") not in roles_for:
            continue
        if subject in (m.get("conflicts") or []):
            continue
        approvers[str(m["id"])] = str(m["role"])
    return len(approvers) >= MIN_APPROVALS and len(set(approvers.values())) >= MIN_ROLES


def checklist_signed_off(
    basis: Mapping[str, Any] | str, faq_id: str | None = None
) -> bool:
    """Whether a conventional basis (a bases.yaml entry or its ID) has a signed-off checklist."""
    basis_id = basis if isinstance(basis, str) else str(basis.get("id"))
    path = checklist_path(basis_id, faq_id)
    if not path.exists():
        return False
    data = yaml_io.load(path)
    if not isinstance(data, Mapping):
        raise HarnessValidationError(f"{path}: not a mapping")
    return is_signed_off(data, load_panel())


FIRST_PASS_AGREEMENT = 2.0 / 3.0


def first_pass_contested(faq_id: str, panel: Mapping[str, Any] | None = None) -> bool:
    """§T10.6 blind first pass: fewer than 2/3 of reviewers on one option makes the basis contested."""
    records = [
        r
        for r in (panel or load_panel()).get("first_pass") or []
        if r.get("item") == faq_id
    ]
    if not records:
        return False
    counts: dict[str, int] = {}
    for r in records:
        counts[str(r.get("option"))] = counts.get(str(r.get("option")), 0) + 1
    return max(counts.values()) / len(records) < FIRST_PASS_AGREEMENT
