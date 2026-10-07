"""Reporting, except the golden snapshot (spec §32A, §36)."""

from __future__ import annotations

import csv
import html as html_lib
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from financial_llm_harness.reporting.dictionary import load_schema, schema_files
from financial_llm_harness.reporting.render import render_run
from financial_llm_harness.reporting.svg import ALLOWED_HEX
from tests.harness.helpers import assert_same_files, output_files

FIGURES = (
    "01_primary_effect.svg",
    "02_arm_rates.svg",
    "03_controls.svg",
    "04_missingness.svg",
    "05_attribution.svg",
    "06_cost.svg",
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _convert(value: str, prop: dict[str, Any]) -> Any:
    """CSV cell → JSON value per the column schema: empty is null (§32A.2)."""
    if value == "":
        return None
    if "enum" in prop:
        return value
    types = prop.get("type")
    types = types if isinstance(types, list) else [types]
    if "integer" in types:
        return int(value)
    if "number" in types:
        return float(value)
    if "boolean" in types:
        assert value in ("true", "false"), value
        return value == "true"
    return value


def test_summary_and_every_csv_validate(s1_run: Path) -> None:
    results = s1_run / "results"
    summary = json.loads((results / "summary.json").read_text())
    schema = load_schema("summary.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(summary)
    csvs = sorted(p.name for p in results.glob("*.csv"))
    assert {
        "effects.csv",
        "labels.csv",
        "controls.csv",
        "missingness.csv",
        "attribution.csv",
        "clusters.csv",
        "trials.csv",
        "costs.csv",
    } <= set(csvs)
    for name in csvs:
        assert name in schema_files(), f"{name} has no schema"
        schema = load_schema(name)
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        )
        raw = (results / name).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r\n" not in raw
        with open(results / name, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            assert header == schema["x-columns"], name
            assert header[0] == "row_id"
            n = 0
            for row in reader:
                obj = {
                    c: _convert(v, schema["properties"][c])
                    for c, v in zip(header, row, strict=True)
                }
                validator.validate(obj)
                n += 1
                if n >= 5000:  # trials.csv and clusters.csv: a prefix is enough per run
                    break


def test_data_dictionary_covers_every_file_and_column(s1_run: Path) -> None:
    text = (s1_run / "results" / "DATA_DICTIONARY.md").read_text()
    for p in (s1_run / "results").glob("*.csv"):
        assert f"`{p.name}`" in text
        for col in load_schema(p.name)["x-columns"]:
            assert f"| `{col}` |" in text
    assert "`summary.json`" in text and "`models[].primary.theta`" in text


# ---------------------------------------------------------------------------
# Numbers traceability (§32A.8)
# ---------------------------------------------------------------------------

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUM = re.compile(
    r"(?<![A-Za-z_\d.])([<]?)([−+]?)(\$?)(\d[\d,]*(?:\.\d+)?(?:e[−+-]?\d+)?)(%|K|M)?(?![\d])"
)
NOTATION = ("CI(1−2α)", "1−2α", "1−α", "CI_{1−α}")


def _result_values(run_dir: Path) -> list[float]:
    vals: list[float] = []
    results = run_dir / "results"

    def walk(o: Any) -> None:
        if isinstance(o, bool):
            return
        if isinstance(o, int | float):
            vals.append(float(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(json.loads((results / "summary.json").read_text()))
    for p in results.glob("*.csv"):
        if p.name in ("trials.csv", "clusters.csv"):
            continue  # per-trial and per-cluster values never appear in the report
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                for v in row.values():
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
    return vals


class Tracer:
    """Checks that a presented number is a result value at the presented precision."""

    def __init__(self, values: list[float]) -> None:
        self.exact = set(values)
        self.by_dp: dict[tuple[str, int], set[float]] = {}
        self.values = values

    def _set(self, kind: str, dp: int) -> set[float]:
        key = (kind, dp)
        if key not in self.by_dp:
            scale = {"plain": 1.0, "%": 100.0, "K": 1e-3, "M": 1e-6}[kind]
            self.by_dp[key] = {round(v * scale, dp) + 0.0 for v in self.values}
        return self.by_dp[key]

    def ok(self, sign: str, body: str, suffix: str | None) -> bool:
        text = body.replace(",", "").replace("−", "-")
        x = float(text) * (-1.0 if sign == "−" else 1.0)
        if "e" in text.lower():
            return x in self.exact
        dp = len(text.split(".")[1]) if "." in text else 0
        if dp > 6:
            return x in self.exact
        kind = suffix or "plain"
        return round(x, dp) + 0.0 in self._set(kind, dp)


def _strip_md(text: str) -> str:
    text = re.sub(r"`[^`]*`", " ", text)  # code spans: identifiers, hashes, commands
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # figure links
    text = re.sub(
        r"^(#+)\s+\d+\.\s", r"\1 ", text, flags=re.MULTILINE
    )  # section numbers
    for n in NOTATION:
        text = text.replace(n, " ")
    return text


def _check_numbers(text: str, tracer: Tracer, dates: set[str], where: str) -> int:
    for d in _DATE.findall(text):
        assert d in dates, f"{where}: date {d} not in results"
    text = _DATE.sub(" ", text)
    n = 0
    bad = []
    for m in _NUM.finditer(text):
        lt, sign, _dollar, body, suffix = m.groups()
        if lt == "<":
            continue  # p-value floor "<0.001"
        n += 1
        if not tracer.ok(sign, body, suffix):
            bad.append(m.group(0))
    assert bad == [], f"{where}: numbers absent from results/: {bad[:20]}"
    return n


def _html_text(doc: str) -> str:
    doc = re.sub(r"<style>.*?</style>", " ", doc, flags=re.S)
    doc = re.sub(r"<script>.*?</script>", " ", doc, flags=re.S)
    doc = re.sub(r"<svg.*?</svg>", " ", doc, flags=re.S)  # figures carry axis ticks
    doc = re.sub(r"<nav.*?</nav>", " ", doc, flags=re.S)
    doc = re.sub(r"<details><summary>Data</summary>.*?</details>", " ", doc, flags=re.S)
    doc = re.sub(r"<code>.*?</code>", " ", doc, flags=re.S)
    doc = re.sub(r"(<h2[^>]*>)\d+\.\s", r"\1", doc)
    doc = re.sub(r"<[^>]+>", " ", doc)
    text = html_lib.unescape(doc)
    for n in NOTATION:
        text = text.replace(n, " ")
    return text


def _data_blocks(doc: str) -> Iterator[tuple[str, list[list[str]]]]:
    for m in re.finditer(
        r"<details><summary>Data</summary><p><code>([^<]+)</code></p>(.*?)</details>",
        doc,
        flags=re.S,
    ):
        rows = []
        for tr in re.findall(r"<tr>(.*?)</tr>", m.group(2), flags=re.S):
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.S)
            rows.append([html_lib.unescape(c) for c in cells])
        yield m.group(1), rows


def test_every_presented_number_traces_to_results(s1_run: Path) -> None:
    tracer = Tracer(_result_values(s1_run))
    summary = json.loads((s1_run / "results" / "summary.json").read_text())
    dates = {(summary["run"][k] or "")[:10] for k in ("started_at", "completed_at")} - {
        ""
    }
    md = (s1_run / "report" / "report.md").read_text()
    assert _check_numbers(_strip_md(md), tracer, dates, "report.md") > 50
    readme = (s1_run / "README.md").read_text()
    _check_numbers(_strip_md(readme), tracer, dates, "README.md")
    doc = (s1_run / "report" / "report.html").read_text()
    assert _check_numbers(_html_text(doc), tracer, dates, "report.html") > 50
    # every figure's data block holds exact rows of the stated results file
    blocks = list(_data_blocks(doc))
    assert len(blocks) == len(FIGURES)
    for source, rows in blocks:
        with open(s1_run / source, newline="") as fh:
            file_rows = list(csv.reader(fh))
        assert rows[0] == file_rows[0]
        body = {tuple(r) for r in file_rows[1:]}
        for r in rows[1:]:
            assert tuple(r) in body, (source, r)


def test_tracer_rejects_an_invented_number() -> None:
    tracer = Tracer([0.123, 0.5, 1234.0, 0.025])
    assert tracer.ok("", "0.12", None) and tracer.ok("+", "0.50", None)
    assert tracer.ok("", "2.5", "%") and tracer.ok("", "1.2", "K")
    assert not tracer.ok("", "0.13", None)
    assert not tracer.ok("−", "0.12", None)
    text = _strip_md("## 3. Primary effect\n- θ̂ −0.12, CI(1−2α) [0.13, 0.5]")
    with pytest.raises(AssertionError):
        _check_numbers(text, tracer, set(), "x")


# ---------------------------------------------------------------------------
# Determinism and static checks
# ---------------------------------------------------------------------------


def test_render_twice_is_byte_identical(s1_run: Path) -> None:
    render_run(s1_run)
    first = output_files(s1_run)
    render_run(s1_run)
    assert_same_files(output_files(s1_run), first)
    assert (s1_run / "logs" / "build_info.json").exists()


_MARKS = {"circle", "rect", "path", "line", "polygon", "polyline", "ellipse"}


def _svg_marks_without_title(svg: str) -> list[str]:
    root = ET.fromstring(svg)
    missing = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag not in _MARKS:
            continue
        if tag == "line" and el.get("class") == "grid":
            continue  # hairline gridlines are axis furniture, not data marks
        if not any(child.tag.split("}")[-1] == "title" for child in el):
            missing.append(ET.tostring(el, encoding="unicode")[:120])
    return missing


def test_static_checks(s1_run: Path) -> None:
    report = s1_run / "report"
    doc = (report / "report.html").read_text()
    # self-contained: no external request of any kind
    assert not re.search(r"https?:|//[a-z]", doc, flags=re.I)
    assert not re.search(r"\b(src|href)=\"(?!#)", doc)
    assert "@import" not in doc and "url(" not in doc
    assert len(doc.encode()) <= 5 * 1024 * 1024
    assert ":root{" in doc and "prefers-color-scheme: dark" in doc
    assert "body{" in doc and "background:var(--surface)" in doc
    assert "tabular-nums" in doc and "@page{size:A4" in doc
    script = re.findall(r"<script>(.*?)</script>", doc, flags=re.S)
    assert all(len(s.encode()) <= 5 * 1024 for s in script)
    # colours only from the §32A.6 palette, in the HTML and every standalone figure
    texts = {"report.html": doc}
    for name in FIGURES:
        texts[name] = (report / "figures" / name).read_text()
    for name, text in texts.items():
        hexes = {
            h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b", text)
        }
        assert hexes <= ALLOWED_HEX, (name, hexes - ALLOWED_HEX)
    # every mark has a <title>; each figure is a standalone themed SVG of width 960
    assert sorted(p.name for p in (report / "figures").iterdir()) == list(FIGURES)
    for name in FIGURES:
        svg = texts[name]
        assert 'viewBox="0 0 960 ' in svg and "prefers-color-scheme: dark" in svg
        assert _svg_marks_without_title(svg) == [], name
    for inline in re.findall(r"<svg.*?</svg>", doc, flags=re.S):
        wrapped = inline.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
        assert _svg_marks_without_title(wrapped) == []
    # every figure is followed by its data block
    for name in FIGURES:
        assert f"figures/{name}" in (report / "report.md").read_text()
    assert doc.count("<details><summary>Data</summary>") == len(FIGURES)


def test_report_layout_and_wording(s1_run: Path) -> None:
    md = (s1_run / "report" / "report.md").read_text()
    lines = md.splitlines()
    assert sum(1 for line in lines if line.startswith("# ")) == 1
    h2 = [line for line in lines if line.startswith("## ")]
    assert h2 == [
        "## 1. Headline",
        "## 2. Design",
        "## 3. Primary effect",
        "## 4. Arm-level rates",
        "## 5. Controls",
        "## 6. Missingness",
        "## 7. Attribution",
        "## 8. Secondary and exploratory",
        "## 9. Diagnostics",
        "## 10. Provenance",
    ]
    assert "<" not in re.sub(r"`[^`]*`", "", md).replace("<0.001", "")  # no raw HTML
    assert lines[2].startswith("Run `00000000` · 2026-01-01 · confirmatory · COMPLETE")
    for path in (
        s1_run / "report" / "report.md",
        s1_run / "README.md",
        s1_run / "report" / "report.html",
    ):
        text = path.read_text()
        prose = re.sub(r"`[^`]*`|<code>.*?</code>|<svg.*?</svg>", " ", text, flags=re.S)
        prose = re.sub(
            r"<details><summary>Data</summary>.*?</details>", " ", prose, flags=re.S
        )
        # "bias" appears only inside a label that §24 assigned (§32A.4)
        assert not re.search(r"bias", prose, flags=re.I), path.name


def test_models_outputs(s1_run: Path) -> None:
    from tests.harness.helpers import FAKE_A, FAKE_B

    for mk in (FAKE_A, FAKE_B):
        d = s1_run / "models" / mk
        lines = (d / "responses.jsonl").read_text().splitlines()
        first = json.loads(lines[0])
        assert list(first)[:8] == [
            "trial_id",
            "case_idx",
            "cluster_id",
            "arm",
            "paraphrase_idx",
            "repetition",
            "status",
            "finish_reason",
        ]
        for key in ("decision", "confidence", "material_factors", "reasoning_excerpt"):
            assert key in first
        assert all(
            len(json.loads(x)["reasoning_excerpt"] or "") <= 600 for x in lines[:2000]
        )
        keys = [
            (o["case_idx"], o["arm"], o["paraphrase_idx"], o["repetition"])
            for o in map(json.loads, lines[:500])
        ]
        arm_order = {a: i for i, a in enumerate(["A", "B", "A0", "PL", "PCA", "PCB"])}
        assert keys == sorted(keys, key=lambda k: (k[0], arm_order[k[1]], k[2], k[3]))
        ex = (d / "examples.md").read_text()
        assert ex.startswith(f"# Examples: `{mk}`")
        assert "## 1. Responsive pairs" in ex and "## 3. Parse failures" in ex
        assert "Mandate:" not in ex  # full prompts are never inlined
    ex_a = (s1_run / "models" / FAKE_A / "examples.md").read_text()
    assert ex_a.count("### 1.") == 10  # fake-a is responsive: ten pairs selected


def test_index_lists_the_run(s1_run: Path) -> None:
    home = s1_run.parent.parent
    index = json.loads((home / "runs" / "index.json").read_text())
    rows = index["runs"] if isinstance(index, dict) else index
    assert any(r["run_id"] == "00000000-0000-4000-8000-000000000001" for r in rows)
    assert "00000000" in (home / "runs" / "INDEX.md").read_text()
