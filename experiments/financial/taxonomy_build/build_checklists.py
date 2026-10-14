"""Writes empty checklist forms (spec §T7, §T10.6) for every conventional basis and every B.conv item.

A form holds no checklist item and no sign-off: the content and the approvals are
the expert panel's (D-10, D-13). Until the panel fills and signs a form, the oracles
resting on it run in discovery mode (§32B.2 step 5). Existing files are never
overwritten. Run from the repo root.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from financial_llm_harness.core import yaml_io
from financial_llm_harness.taxonomy.expert_panel import checklist_path
from financial_llm_harness.taxonomy.registry import _item_bases, load_registry

PROV = {
    "drafted_by": {"kind": "llm", "family": "anthropic", "id": "claude-code-agent"},
    "approved_by": [],
    "approved_on": None,
}
NOTE = (
    "Empty form: the expert panel writes the checklist items (each with a rationale) and "
    "signs it off (§T7, §T10.6). Until then every oracle resting on it runs in discovery mode."
)


def write(path: Path, doc: dict) -> None:
    if path.exists():
        print("kept", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_io.dump(doc), encoding="utf-8")
    print("wrote", path)


def main() -> None:
    reg = load_registry()
    conventional = {
        b for b, e in reg.bases.items() if e.get("confidence") == "conventional"
    }
    users: dict[str, set[str]] = defaultdict(set)
    for item in reg.items.values():
        for b in _item_bases(item):
            if b in conventional:
                users[b].add(str(item.data["domain"]))
                if b == "B.conv":
                    write(
                        checklist_path("B.conv", item.id),
                        {
                            "provenance": PROV,
                            "basis": "B.conv",
                            "faq_id": item.id,
                            "domain": [str(item.data["domain"])],
                            "note": NOTE,
                            "items": [],
                            "signoffs": [],
                            "disagreements": [],
                        },
                    )
    for b in sorted(conventional - {"B.conv"}):
        write(
            checklist_path(b),
            {
                "provenance": PROV,
                "basis": b,
                "faq_id": None,
                "domain": sorted(users.get(b, set())),
                "note": NOTE,
                "items": [],
                "signoffs": [],
                "disagreements": [],
            },
        )


main()
