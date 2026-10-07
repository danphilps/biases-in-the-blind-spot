"""``report/report.html``: one self-contained file (spec §32A.5).

Converts the fixed ``report.md`` (a small CommonMark + GFM subset written by
``markdown.py``) to HTML with inline CSS and inline SVG. No external request,
no timestamp. Every figure is followed by ``<details><summary>Data</summary>``
holding the exact CSV rows behind it. Colours are the §32A.6 palette tokens on
``:root``, with dark overrides under ``prefers-color-scheme: dark``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

from financial_llm_harness.reporting.svg import PALETTE_DARK, PALETTE_LIGHT

LONG_TABLE = 20
STYLE_CSS = Path(__file__).resolve().parent / "style.css"

# Optional script (≤ 5 KB): expands every <details> for printing. No content needs it.
SCRIPT = (
    "window.addEventListener('beforeprint',function(){"
    "document.querySelectorAll('details').forEach(function(d){d.setAttribute('open','')})});"
)


@dataclass(frozen=True)
class FigureData:
    """The CSV rows behind one figure: header and raw cell strings."""

    source: str
    columns: Sequence[str]
    rows: Sequence[Sequence[str]]


def _tokens(p: Mapping[str, str]) -> str:
    return ";".join(f"--{k}:{v}" for k, v in p.items())


def stylesheet() -> str:
    """Palette tokens on ``:root`` (light, then dark overrides) and ``style.css``, minified to one line."""
    rules = "".join(
        line.strip()
        for line in STYLE_CSS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("/*")
    )
    return (
        f":root{{{_tokens(PALETTE_LIGHT)};color-scheme:light dark}}"
        f"@media (prefers-color-scheme: dark){{:root{{{_tokens(PALETTE_DARK)}}}}}"
        + rules
    )


_CODE = re.compile(r"`([^`]*)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_IMG = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")


def inline(text: str) -> str:
    """Escape and convert code spans and bold; code spans are opaque."""
    out: list[str] = []
    pos = 0
    for m in _CODE.finditer(text):
        out.append(
            _BOLD.sub(
                r"<strong>\1</strong>", escape(text[pos : m.start()], quote=False)
            )
        )
        out.append(f"<code>{escape(m.group(1), quote=False)}</code>")
        pos = m.end()
    out.append(_BOLD.sub(r"<strong>\1</strong>", escape(text[pos:], quote=False)))
    return "".join(out)


def _split_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    cells: list[str] = []
    cur: list[str] = []
    i = 0
    in_code = False
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body) and body[i + 1] == "|":
            cur.append("|")
            i += 2
            continue
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells


_NUMERIC = re.compile(r"^[−\-+]?[\d.,$%]+[KM]?$")


def html_table(
    header: Sequence[str], rows: Sequence[Sequence[str]], *, raw: bool = False
) -> str:
    """A table in its own scroll container; ``raw`` cells are escaped verbatim (CSV values)."""
    fmt = (lambda s: escape(s, quote=False)) if raw else inline
    # Right-align a column only when every non-empty cell in it is numeric.
    numeric = [
        any(j < len(r) and r[j] for r in rows)
        and all(
            j >= len(r) or not r[j] or r[j] == "—" or _NUMERIC.match(r[j]) for r in rows
        )
        for j in range(len(header))
    ]
    th = "".join(
        f'<th class="n">{fmt(h)}</th>' if numeric[j] else f"<th>{fmt(h)}</th>"
        for j, h in enumerate(header)
    )
    body = []
    for r in rows:
        tds = "".join(
            f'<td class="n">{fmt(c)}</td>'
            if j < len(numeric) and numeric[j]
            else f"<td>{fmt(c)}</td>"
            for j, c in enumerate(r)
        )
        body.append(f"<tr>{tds}</tr>")
    table = f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    if not raw and len(rows) > LONG_TABLE:
        return f"<details><summary>Table</summary>{table}</details>"
    return table


def _slug(heading: str) -> str:
    m = re.match(r"^(\d+)\.", heading)
    if m:
        return f"s{m.group(1)}"
    return "s-" + re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def _figure(
    src: str, alt: str, figs: Mapping[str, str], data: Mapping[str, FigureData]
) -> str:
    name = src.rsplit("/", 1)[-1]
    svg = figs.get(name)
    if svg is None:
        return f"<p>{escape(alt)}: figure not rendered.</p>"
    # Inline SVG needs no namespace attribute; dropping it keeps the file free of URLs.
    inline_svg = svg.strip().replace(' xmlns="http://www.w3.org/2000/svg"', "", 1)
    out = [f'<figure aria-label="{escape(alt)}">{inline_svg}</figure>']
    fd = data.get(name)
    if fd is not None:
        out.append(
            f"<details><summary>Data</summary><p><code>{escape(fd.source)}</code></p>"
            f"{html_table(fd.columns, fd.rows, raw=True)}</details>"
        )
    return "".join(out)


def render_html(
    md: str, figs: Mapping[str, str], data: Mapping[str, FigureData]
) -> str:
    """Self-contained HTML of ``report.md`` with inline figures and their data blocks."""
    lines = md.split("\n")
    body: list[str] = []
    toc: list[tuple[str, str]] = []
    title = ""
    i = 0
    para: list[str] = []

    def flush_para() -> None:
        if para:
            body.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            flush_para()
            i += 1
            continue
        if s.startswith("### "):
            flush_para()
            body.append(f"<h3>{inline(s[4:])}</h3>")
        elif s.startswith("## "):
            flush_para()
            h = s[3:]
            sid = _slug(h)
            toc.append((sid, h))
            body.append(f'<h2 id="{sid}">{inline(h)}</h2>')
        elif s.startswith("# "):
            flush_para()
            title = s[2:]
            body.append(f"<h1>{inline(title)}</h1>")
        elif _IMG.match(s):
            flush_para()
            m = _IMG.match(s)
            assert m is not None
            body.append(_figure(m.group(2), m.group(1), figs, data))
        elif s.startswith("|"):
            flush_para()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            header = _split_row(block[0])
            rows = [_split_row(b) for b in block[2:]]
            body.append(html_table(header, rows))
            continue
        elif s.startswith("- "):
            flush_para()
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(f"<li>{inline(lines[i].strip()[2:])}</li>")
                i += 1
            body.append(f"<ul>{''.join(items)}</ul>")
            continue
        else:
            para.append(s)
        i += 1
    flush_para()
    plain_title = _CODE.sub(r"\1", title) or "Report"
    nav = "".join(f'<li><a href="#{sid}">{inline(h)}</a></li>' for sid, h in toc)
    return (
        "<!doctype html>\n"
        '<html lang="en-GB">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(plain_title)}</title>\n"
        f"<style>{stylesheet()}</style>\n</head>\n<body>\n"
        f'<nav class="toc" aria-label="Contents"><ol>{nav}</ol></nav>\n'
        f"<main>\n{chr(10).join(body)}\n</main>\n"
        f"<script>{SCRIPT}</script>\n</body>\n</html>\n"
    )
