"""Writes taxonomy/references.bib from the §T7 basis table (spec §T7 'References').

Only what §T7 states is transcribed: authors, year, venue, volume and issue. Titles
and DOIs are not in the spec and are never filled in from memory: each entry says
so, and the user adds them (a `@pytest.mark.live` test resolves every DOI present).
Construction, derivation and expert bases (B.identity, B.planted, B.mandate,
B.irrelevant, B.conv) have no reference. Run from the repo root.
"""

from __future__ import annotations

import re
from pathlib import Path

from financial_llm_harness.core import yaml_io

ROOT = Path(".")
NO_REFERENCE = {"B.identity", "B.planted", "B.mandate", "B.irrelevant", "B.conv"}
_CITE = re.compile(r"^(?P<authors>.+?) \((?P<year>\d{4})\)(?:, (?P<venue>.+))?$")


def entry(bid: str, text: str) -> str:
    parts = [p.strip() for p in text.split(";")]
    first = parts[0]
    year = re.sub(r"\D", "", bid)[-4:]
    for p in parts[
        1:
    ]:  # a row with two works (B.ff1993, B.ff2015): the part with the ID's year
        if p.startswith(f"({year})"):
            first = first.split(" (")[0] + " " + p
    m = _CITE.match(first)
    fields: list[tuple[str, str]] = []
    if m:
        fields.append(("author", m.group("authors").replace(" & ", " and ")))
        fields.append(("year", m.group("year")))
        if m.group("venue"):
            fields.append(("howpublished", m.group("venue")))
    fields.append(
        ("note", f"Spec §T7: {text}. Title and DOI to be added from the source.")
    )
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
    return f"@misc{{{bid},\n{body}\n}}\n"


def main() -> None:
    data = yaml_io.load(ROOT / "financial_llm_harness/taxonomy/bases.yaml")
    out = [
        "% References of the §T7 oracle bases, transcribed from the spec (authors, year, venue).\n"
        "% Titles and DOIs are added by people from the sources; none is written from memory.\n"
    ]
    for b in data["bases"]:
        if b["id"] in NO_REFERENCE:
            continue
        out.append(entry(b["id"], b["basis"]))
    path = ROOT / "financial_llm_harness/taxonomy/references.bib"
    path.write_text("\n".join(out), encoding="utf-8")
    print("wrote", path, len(out) - 1, "entries")


main()
