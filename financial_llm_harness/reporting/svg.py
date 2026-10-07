"""Hand-written SVG figures (spec §32A.6).

Primitives: axis, hairline grid, point, interval, bar, text. Deterministic:
fixed element order, coordinates rounded to 0.01, no random IDs, no
timestamps. viewBox width 960; height = 40 px per row + 80 px of chrome. An
embedded ``<style>`` carries the light and dark tokens. Every mark has a
``<title>`` holding its exact values and ``row_id``. Colour encodes only arm
or status class, never model identity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any
from xml.sax.saxutils import escape

from financial_llm_harness.oracles import glyph, label_class

WIDTH = 960
ROW_H = 40
CHROME = 80
LEFT = 300
RIGHT_TEXT = 340

PALETTE_LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
    "band": "#f0efec",
    "armA": "#2a78d6",
    "armB": "#eb6834",
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    "c1": "#2a78d6",
    "c2": "#eb6834",
    "c3": "#1baf7a",
    "c4": "#eda100",
    "c5": "#e87ba4",
    "c6": "#008300",
}
PALETTE_DARK = {
    "surface": "#1a1a19",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "baseline": "#383835",
    "band": "#383835",
    "armA": "#3987e5",
    "armB": "#d95926",
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    "c1": "#3987e5",
    "c2": "#d95926",
    "c3": "#199e70",
    "c4": "#c98500",
    "c5": "#d55181",
    "c6": "#008300",
}
ALLOWED_HEX = {v.lower() for v in (*PALETTE_LIGHT.values(), *PALETTE_DARK.values())}
STATUS_TOKEN = {
    "good": "good",
    "warning": "warning",
    "serious": "serious",
    "critical": "critical",
    "neutral": "muted",
}
MINUS = "−"


def r2(x: float) -> str:
    v = round(float(x), 2)
    if v == 0:
        v = 0.0
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def num(x: float | None, dp: int = 2) -> str:
    if x is None:
        return "—"
    return f"{float(x):+.{dp}f}".replace("-", MINUS)


def style_block() -> str:
    light = ";".join(f"--{k}:{v}" for k, v in PALETTE_LIGHT.items())
    dark = ";".join(f"--{k}:{v}" for k, v in PALETTE_DARK.items())
    return (
        "<style>"
        f"svg{{{light}}}"
        f"@media (prefers-color-scheme: dark){{svg{{{dark}}}}}"
        ".bg{fill:var(--surface)}.ink{fill:var(--ink)}.ink2{fill:var(--ink2)}.muted{fill:var(--muted)}"
        ".grid{stroke:var(--grid);stroke-width:1}.base{stroke:var(--baseline);stroke-width:1}"
        ".band{fill:var(--band)}text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;font-size:13px}"
        ".axis{fill:var(--muted);font-size:11px}.rt{font-size:12px}"
        "</style>"
    )


class Canvas:
    def __init__(self, n_rows: int, title: str) -> None:
        self.height = max(1, n_rows) * ROW_H + CHROME
        self.parts: list[str] = []
        self.title = title

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(
        self,
        x: float,
        y: float,
        s: str,
        cls: str = "ink",
        anchor: str = "start",
        title: str | None = None,
    ) -> None:
        t = f"<title>{escape(title)}</title>" if title else ""
        self.add(
            f'<text x="{r2(x)}" y="{r2(y)}" class="{cls}" text-anchor="{anchor}">{t}{escape(s)}</text>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str,
        width: float,
        title: str | None = None,
        cls: str | None = None,
    ) -> None:
        t = f"<title>{escape(title)}</title>" if title else ""
        c = f' class="{cls}"' if cls else ""
        style = (
            f' style="stroke:var(--{stroke});stroke-width:{r2(width)}"'
            if not cls
            else ""
        )
        self.add(
            f'<line x1="{r2(x1)}" y1="{r2(y1)}" x2="{r2(x2)}" y2="{r2(y2)}"{c}{style}>{t}</line>'
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        title: str | None = None,
        rx: float = 0,
        cls: str | None = None,
    ) -> None:
        t = f"<title>{escape(title)}</title>" if title else ""
        c = f' class="{cls}"' if cls else f' style="fill:var(--{fill})"'
        rr = f' rx="{r2(rx)}"' if rx else ""
        self.add(
            f'<rect x="{r2(x)}" y="{r2(y)}" width="{r2(max(w, 0))}" height="{r2(h)}"{rr}{c}>{t}</rect>'
        )

    def bar(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        title: str,
        rounded_end: bool = True,
    ) -> None:
        """Horizontal bar: square baseline end; 4 px rounded data end when ``rounded_end``."""
        w = max(w, 0.0)
        r = min(4.0, w / 2, h / 2) if rounded_end else 0.0
        if r <= 0:
            self.rect(x, y, w, h, fill, title=title)
            return
        d = (
            f"M{r2(x)} {r2(y)}H{r2(x + w - r)}A{r2(r)} {r2(r)} 0 0 1 {r2(x + w)} {r2(y + r)}"
            f"V{r2(y + h - r)}A{r2(r)} {r2(r)} 0 0 1 {r2(x + w - r)} {r2(y + h)}H{r2(x)}Z"
        )
        self.add(
            f'<path d="{d}" style="fill:var(--{fill})"><title>{escape(title)}</title></path>'
        )

    def point(
        self, x: float, y: float, fill: str, title: str, hollow: bool = False
    ) -> None:
        if hollow:
            self.add(
                f'<circle cx="{r2(x)}" cy="{r2(y)}" r="5" style="fill:var(--surface);stroke:var(--{fill});stroke-width:2">'
                f"<title>{escape(title)}</title></circle>"
            )
        else:
            self.add(
                f'<circle cx="{r2(x)}" cy="{r2(y)}" r="5" style="fill:var(--{fill});stroke:var(--surface);stroke-width:2">'
                f"<title>{escape(title)}</title></circle>"
            )

    def svg(self) -> str:
        body = "".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {self.height}" role="img" '
            f'aria-label="{escape(self.title)}">{style_block()}'
            f'<rect x="0" y="0" width="{WIDTH}" height="{self.height}" class="bg"><title>{escape(self.title)}</title></rect>'
            f"{body}</svg>\n"
        )


class Scale:
    def __init__(self, lo: float, hi: float, x0: float, x1: float) -> None:
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo, hi = -1.0, 1.0
        self.lo, self.hi, self.x0, self.x1 = lo, hi, x0, x1

    def __call__(self, v: float) -> float:
        v = min(max(v, self.lo), self.hi)
        return self.x0 + (v - self.lo) / (self.hi - self.lo) * (self.x1 - self.x0)

    def ticks(self, n: int = 5) -> list[float]:
        span = self.hi - self.lo
        step = 10 ** math.floor(math.log10(span / n)) if span > 0 else 1.0
        for m in (1, 2, 5, 10):
            if span / (step * m) <= n:
                step *= m
                break
        start = math.ceil(self.lo / step) * step
        out = []
        v = start
        while v <= self.hi + 1e-12:
            out.append(round(v, 10))
            v += step
        return out


def _axis(c: Canvas, sc: Scale, y_top: float, y_bottom: float, fmt_dp: int = 2) -> None:
    for t in sc.ticks():
        x = sc(t)
        c.line(x, y_top, x, y_bottom, "grid", 1, cls="grid")
        c.text(
            x,
            y_bottom + 16,
            num(t, fmt_dp) if t != 0 else "0",
            cls="axis",
            anchor="middle",
        )


def _range(
    values: Sequence[float | None], pad: float = 0.1, include: Sequence[float] = ()
) -> tuple[float, float]:
    vals = [
        float(v) for v in values if v is not None and math.isfinite(float(v))
    ] + list(include)
    if not vals:
        return -1.0, 1.0
    lo, hi = min(vals), max(vals)
    span = hi - lo or 0.2
    return lo - pad * span, hi + pad * span


def forest(
    rows: Sequence[Mapping[str, Any]],
    delta: float | None,
    title: str,
    level_tau: float | None = None,
) -> str:
    """01: one row per model; band [−δ, δ]; 2 px CI_{1−α}; 6 px CI_{1−2α}; marker by label status."""
    c = Canvas(len(rows), title)
    plot_x0, plot_x1 = LEFT, WIDTH - RIGHT_TEXT
    vals: list[float | None] = []
    for r in rows:
        vals += [r.get("ci_low"), r.get("ci_high"), r.get("theta")]
    inc = (
        [0.0]
        + ([-delta, delta] if delta and level_tau is None else [])
        + ([level_tau] if level_tau is not None else [])
    )
    sc = Scale(*_range(vals, include=inc), plot_x0, plot_x1)
    top, bottom = 40, 40 + len(rows) * ROW_H
    c.text(16, 24, title, cls="ink")
    if delta and level_tau is None:
        c.rect(
            sc(-delta),
            top,
            sc(delta) - sc(-delta),
            bottom - top,
            "band",
            title=f"equivalence band ±{delta}",
            cls="band",
        )
    _axis(c, sc, top, bottom)
    ref = level_tau if level_tau is not None else 0.0
    c.line(
        sc(ref),
        top,
        sc(ref),
        bottom,
        "baseline",
        1,
        title=f"reference {ref}",
        cls="base",
    )
    for i, r in enumerate(rows):
        y = top + i * ROW_H + ROW_H / 2
        c.text(16, y + 4, str(r["label_text"]), cls="ink2")
        tip = f"{r['model_key']}: θ̂ {r.get('theta')} CI [{r.get('ci_low')}, {r.get('ci_high')}] CI(1−2α) [{r.get('ci2a_low')}, {r.get('ci2a_high')}] row_id {r['row_id']}"
        if r.get("ci_low") is not None:
            c.line(sc(r["ci_low"]), y, sc(r["ci_high"]), y, "ink2", 2, title=tip)
        if r.get("ci2a_low") is not None:
            c.line(sc(r["ci2a_low"]), y, sc(r["ci2a_high"]), y, "ink2", 6, title=tip)
        if r.get("theta") is not None:
            cls = label_class(r.get("label"))
            c.point(sc(r["theta"]), y, STATUS_TOKEN[cls], tip)
        right = f"{num(r.get('theta'))} [{num(r.get('ci_low'))}, {num(r.get('ci_high'))}] {glyph(r.get('label'))} {r.get('label') or '—'}"
        c.text(WIDTH - RIGHT_TEXT + 12, y + 4, right, cls="ink rt", title=tip)
    return c.svg()


def dumbbell(rows: Sequence[Mapping[str, Any]], title: str) -> str:
    """02: arm means on [0, 1]; dot A (series 1), dot B (series 2), muted connector, legend."""
    c = Canvas(len(rows), title)
    sc = Scale(0.0, 1.0, LEFT, WIDTH - 60)
    top, bottom = 40, 40 + len(rows) * ROW_H
    c.text(16, 24, title, cls="ink")
    c.point(LEFT, 20, "armA", "legend: arm A")
    c.text(LEFT + 10, 24, "arm A", cls="ink2")
    c.point(LEFT + 80, 20, "armB", "legend: arm B")
    c.text(LEFT + 90, 24, "arm B", cls="ink2")
    _axis(c, sc, top, bottom)
    for i, r in enumerate(rows):
        y = top + i * ROW_H + ROW_H / 2
        c.text(16, y + 4, str(r["label_text"]), cls="ink2")
        a, b = r.get("mean_a"), r.get("mean_b")
        tip = f"{r['model_key']}: mean A {a}, mean B {b}, row_id {r['row_id']}"
        if a is not None and b is not None:
            c.line(sc(a), y, sc(b), y, "muted", 2, title=tip)
        if a is not None:
            c.point(sc(a), y, "armA", f"arm A {a} ({r['row_id']})")
        if b is not None:
            c.point(sc(b), y, "armB", f"arm B {b} ({r['row_id']})")
    return c.svg()


def small_multiples(
    panels: Sequence[tuple[str, Sequence[Mapping[str, Any]], float | None]],
    models: Sequence[str],
    title: str,
) -> str:
    """03: null, placebo, positive panels with shared rows; each panel its own x-scale."""
    c = Canvas(len(models), title)
    top, bottom = 40, 40 + len(models) * ROW_H
    c.text(16, 24, title, cls="ink")
    width = (WIDTH - LEFT - 20) / max(1, len(panels))
    for i, mk in enumerate(models):
        c.text(16, top + i * ROW_H + ROW_H / 2 + 4, mk, cls="ink2")
    for j, (name, rows, margin) in enumerate(panels):
        x0 = LEFT + j * width + 10
        x1 = x0 + width - 20
        vals: list[float | None] = []
        for r in rows:
            vals += [r.get("ci_low"), r.get("ci_high")]
        inc = [0.0] + ([-margin, margin] if margin else [])
        sc = Scale(*_range(vals, include=inc), x0, x1)
        c.text(x0, top - 6, name, cls="ink")
        if margin and name != "positive":
            c.rect(
                sc(-margin),
                top,
                sc(margin) - sc(-margin),
                bottom - top,
                "band",
                title=f"{name}: expected region ±{margin}",
                cls="band",
            )
        elif margin:
            c.rect(
                x0,
                top,
                sc(-margin) - x0,
                bottom - top,
                "band",
                title=f"{name}: expected region beyond −{margin}",
                cls="band",
            )
        _axis(c, sc, top, bottom)
        c.line(sc(0), top, sc(0), bottom, "baseline", 1, title="zero", cls="base")
        by_model = {r["model_key"]: r for r in rows}
        for i, mk in enumerate(models):
            r = by_model.get(mk)
            if r is None:
                continue
            y = top + i * ROW_H + ROW_H / 2
            tip = f"{name} {mk}: θ̂ {r.get('theta')} CI [{r.get('ci_low')}, {r.get('ci_high')}] passed {r.get('passed')} row_id {r['row_id']}"
            if r.get("ci_low") is not None:
                c.line(sc(r["ci_low"]), y, sc(r["ci_high"]), y, "ink2", 2, title=tip)
            if r.get("ci2a_low") is not None:
                c.line(
                    sc(r["ci2a_low"]), y, sc(r["ci2a_high"]), y, "ink2", 6, title=tip
                )
            if r.get("theta") is not None:
                c.point(
                    sc(r["theta"]), y, "good" if r.get("passed") else "serious", tip
                )
    return c.svg()


def stacked(
    rows: Sequence[Mapping[str, Any]], threshold: float | None, title: str
) -> str:
    """04: two bars per model (A, B); segments = invalid-status categories; hairline at max_invalid_rate."""
    c = Canvas(len(rows), title)
    top, bottom = 40, 40 + len(rows) * ROW_H
    cats: list[str] = []
    for r in rows:
        for s, _v, _rid in r["segments"]:
            if s not in cats:
                cats.append(s)
    tokens = ["c1", "c2", "c3", "c4", "c5", "c6"]
    color = {s: tokens[i % len(tokens)] for i, s in enumerate(cats)}
    hi = (
        max(
            [sum(v for _s, v, _ in r["segments"]) for r in rows]
            + [threshold or 0.0, 0.05]
        )
        * 1.15
    )
    sc = Scale(0.0, hi, LEFT, WIDTH - 60)
    # A legend only for two or more series; a single category is named in the title.
    c.text(16, 24, f"{title}: {cats[0]}" if len(cats) == 1 else title, cls="ink")
    if len(cats) >= 2:
        for k, s in enumerate(cats):
            c.rect(LEFT + 150 * k, 12, 12, 12, color[s], title=f"legend: {s}")
            c.text(LEFT + 150 * k + 16, 23, s, cls="ink2")
    _axis(c, sc, top, bottom, fmt_dp=2)
    if threshold is not None:
        c.line(
            sc(threshold),
            top,
            sc(threshold),
            bottom,
            "ink2",
            1,
            title=f"max_invalid_rate {threshold}",
        )
    for i, r in enumerate(rows):
        y = top + i * ROW_H + 8
        c.text(16, y + 16, str(r["label_text"]), cls="ink2")
        x = sc(0)
        segs = r["segments"]
        for k, (s, v, rid) in enumerate(segs):
            w = sc(v) - sc(0)
            c.bar(
                x,
                y,
                w,
                20,
                color[s],
                title=f"{r['label_text']} {s}: rate {v} row_id {rid}",
                rounded_end=k == len(segs) - 1,
            )
            x += w + (2 if w > 0 else 0)
    return c.svg()


def attribution_plot(rows: Sequence[Mapping[str, Any]], tau: float, title: str) -> str:
    """05: π_R with CI (filled marker), π_0 (hollow marker), vertical line at τ_v."""
    c = Canvas(len(rows), title)
    sc = Scale(0.0, 1.0, LEFT, WIDTH - 60)
    top, bottom = 40, 40 + len(rows) * ROW_H
    c.text(16, 24, title, cls="ink")
    _axis(c, sc, top, bottom)
    c.line(sc(tau), top, sc(tau), bottom, "ink2", 1, title=f"τ_v {tau}")
    for i, r in enumerate(rows):
        y = top + i * ROW_H + ROW_H / 2
        c.text(16, y + 4, str(r["label_text"]), cls="ink2")
        tip = f"{r['label_text']}: π_R {r.get('pi_r')} CI [{r.get('ci_low')}, {r.get('ci_high')}] π_0 {r.get('pi_0')} row_id {r['row_id']}"
        if r.get("ci_low") is not None:
            c.line(sc(r["ci_low"]), y, sc(r["ci_high"]), y, "ink2", 2, title=tip)
        if r.get("pi_r") is not None:
            c.point(sc(r["pi_r"]), y, "armA", tip)
        if r.get("pi_0") is not None:
            c.point(
                sc(r["pi_0"]),
                y,
                "armA",
                f"π_0 {r['pi_0']} row_id {r['row_id']}",
                hollow=True,
            )
    return c.svg()


def bars(rows: Sequence[Mapping[str, Any]], title: str) -> str:
    """06: single-series horizontal bars of cost_usd, value at the tip, tokens in <title>."""
    c = Canvas(len(rows), title)
    hi = max([float(r["value"]) for r in rows] + [1.0])
    sc = Scale(0.0, hi * 1.2, LEFT, WIDTH - 120)
    top, bottom = 40, 40 + len(rows) * ROW_H
    c.text(16, 24, title, cls="ink")
    for i, r in enumerate(rows):
        y = top + i * ROW_H + 8
        c.text(16, y + 16, str(r["label_text"]), cls="ink2")
        w = sc(float(r["value"])) - sc(0)
        c.bar(sc(0), y, max(w, 1), 20, "armA", title=str(r["title"]))
        c.text(
            sc(0) + max(w, 1) + 8,
            y + 15,
            str(r["value_text"]),
            cls="ink",
            title=str(r["title"]),
        )
    c.line(sc(0), top, sc(0), bottom, "baseline", 1, title="zero", cls="base")
    return c.svg()


def render_figures(
    rd: Any,
    summary: Mapping[str, Any],
    effects: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    missing: Sequence[Mapping[str, Any]],
    attribution: Sequence[Mapping[str, Any]],
    costs: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """The six figures of §32A.6, keyed by file name."""
    d = summary["design"]
    pf = d["primary_feature"]
    level = d["estimand"] == "arm_level"
    contrast = f"level_{rd.manifest['hypothesis']['level_arm']}" if level else "B_vs_A"
    labels = {
        m["model_key"]: m["primary"]["sensitivity_label"] for m in summary["models"]
    }
    eff = {(e["model_key"], e["feature_id"], e["contrast"]): e for e in effects}
    prim_rows = []
    for mk in rd.model_keys:
        e = eff.get((mk, pf, contrast)) or {}
        prim_rows.append(
            {
                **e,
                "model_key": mk,
                "label_text": mk,
                "label": labels.get(mk),
                "row_id": e.get("row_id", ""),
            }
        )
    tau = rd.manifest["oracle_spec"].get("tau") if level else None
    figs = {
        "01_primary_effect.svg": forest(
            prim_rows, None if level else float(d["delta"]), "Primary effect", tau
        )
    }
    figs["02_arm_rates.svg"] = dumbbell(
        [
            {
                **r,
                "mean_a": r.get("mean_a") if not level else r.get("theta"),
                "mean_b": r.get("mean_b"),
            }
            for r in prim_rows
        ],
        "Arm-level rates",
    )
    panels = []
    for name, margin in (
        ("null", d.get("null_margin")),
        ("placebo", d.get("placebo_margin")),
        ("positive", d["delta"]),
    ):
        panels.append(
            (
                name,
                [c for c in controls if c["control"] == name],
                float(margin) if margin is not None else None,
            )
        )
    figs["03_controls.svg"] = small_multiples(panels, rd.model_keys, "Controls")
    arms = ["A", "B"] if not level else [rd.manifest["hypothesis"]["level_arm"]]
    srows = []
    for mk in rd.model_keys:
        for arm in arms:
            segs = [
                (m["status"], float(m["rate"]), m["row_id"])
                for m in missing
                if m["model_key"] == mk
                and m["arm"] == arm
                and m["status_class"] == "invalid"
            ]
            srows.append({"label_text": f"{mk} [{arm}]", "segments": segs})
    thr = float(rd.manifest["hypothesis"]["missing_data"]["max_invalid_rate"])
    figs["04_missingness.svg"] = stacked(srows, thr, "Invalid responses by status")
    arows = [
        {**a, "label_text": f"{a['model_key']} [{a['channel']}]"} for a in attribution
    ]
    figs["05_attribution.svg"] = attribution_plot(arows, 0.30, "Attribution")
    crows = [
        {
            "label_text": f"{c['model_key']} ({c['role']})",
            "value": float(c["cost_usd"]),
            "value_text": f"${float(c['cost_usd']):,.2f}",
            "title": f"{c['model_key']} {c['role']}: cost {c['cost_usd']} USD, calls {c['calls']}, input tokens {c['input_tokens']}, output tokens {c['output_tokens']}, row_id {c['row_id']}",
        }
        for c in costs
    ]
    figs["06_cost.svg"] = bars(crows, "Cost by model and role")
    return figs
