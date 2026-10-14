"""Builds taxonomy/faq.yaml, interventions.yaml and bases.yaml from the spec's tables.

Transcription, not authorship: every row comes from §T5, §T6.2, §T4, §T7 and §T8 of
docs/financial_harness_spec.md. The interpretive steps are listed in LOCAL, CHECKS and
the rules below; unstated fields stay null. Run from the repository root:
    .venv/bin/python <this file> <out_dir>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io

SPEC = Path("docs/financial_harness_spec.md").read_text(encoding="utf-8")
OUT = Path(sys.argv[1])
PROV = {
    "drafted_by": {"kind": "llm", "family": "anthropic", "id": "claude-code-agent"},
    "approved_by": [],
    "approved_on": None,
}


def section(start: str, end: str) -> str:
    return SPEC[SPEC.index(start) : SPEC.index(end)]


def table_rows(text: str, first_cell: str) -> list[list[str]]:
    out = []
    for line in text.splitlines():
        if re.match(first_cell, line):
            out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s


STOP = {"vs", "with", "of", "to", "and", "or", "the", "a", "an", "in", "on", "for", "from", "by"}


def local_slug(item: str, prose: str, variant: str = "base") -> str:
    """Leading words of the prose slug, so the pair's key fits 60 characters (§T1, §32B.2 step 2).

    Words are dropped from the end (and any trailing connective) until
    ``<item>__<variant>__i_item_<item>_<slug>.v1`` is at most 60 characters.
    """
    key_slug = re.sub(r"[^a-z0-9]", "_", item.lower())
    room = 60 - len(f"{key_slug}__{variant}__i_item_{key_slug}_") - len(".v1")
    words = slug(prose).split("_")
    while words and (len("_".join(words)) > room or words[-1] in STOP):
        words.pop()
    if not words:
        raise SystemExit(f"{item}: no slug fits for {prose!r}")
    return "_".join(words)


# ---------------------------------------------------------------------------
# §T4.3 short forms
# ---------------------------------------------------------------------------
SHORT = {
    "name": "I.ID.name", "ceo": "I.ID.ceo", "ceo_ethnicity": "I.ID.ceo_ethnicity",
    "country": "I.ID.country",
    "broker_tier": "I.SRC.broker_tier", "seniority": "I.SRC.seniority",
    "consensus_label": "I.SRC.consensus_label", "order": "I.PRES.order",
    "frame": "I.PRES.frame", "units": "I.PRES.units", "distractor": "I.PRES.distractor",
    "length": "I.PRES.length", "anchor": "I.ANC.anchor", "date": "I.TIME.date",
    "stance": "I.USER.stance", "role": "I.USER.role", "hype": "I.HYPE.label",
    "hype_arch": "I.HYPE.arch",
    "I.Q.purpose wording": "I.PRES.purpose_wording", "tone": "I.PRES.tone",
}

# ---------------------------------------------------------------------------
# Relevant entries stated in prose → local interventions (§T1): (prose, op, oracle text)
# The op is the §10 operation the prose describes; the oracle text is §T5's, verbatim.
# ---------------------------------------------------------------------------
LOCAL: dict[str, list[tuple[str, str, str]]] = {
    "F1.2.b": [("WACC shift", "NUMERIC_SHIFT", "MAGNITUDE (computed)")],
    "F2.3.a": [("volatility shift", "NUMERIC_SHIFT", "DIRECTIONAL")],
    "F2.4.a": [("auditor change", "CATEGORY_CHANGE", "DIRECTIONAL(+)"),
               ("related-party note", "ADD", "DIRECTIONAL(+)")],
    "F3.2.b": [('"one-off" items recurring across years', "ADD", "DIRECTIONAL")],
    "F3.3.a": [("Shift in one cost line", "NUMERIC_SHIFT", "correct attribution (LEVEL)")],
    "F3.4.b": [("Receivables growth vs revenue growth", "NUMERIC_SHIFT", "DIRECTIONAL")],
    "F3.5.a": [("Capex/D&A", "NUMERIC_SHIFT", "DIRECTIONAL")],
    "F3.6.b": [("ROIC − WACC", "NUMERIC_SHIFT", "DIRECTIONAL")],
    "F3.6.c": [("Equity-multiplier vs margin driver", "CATEGORY_CHANGE", "correct attribution (LEVEL)")],
    "F3.7.a": [("Planted unusual items", "ADD", "LEVEL; terminology check")],
    "F3.7.b": [("Gain size with operating results fixed", "NUMERIC_SHIFT", "INVARIANCE of the operating view")],
    "F4.1.a": [("Payout and FCF cover", "NUMERIC_SHIFT", "DIRECTIONAL")],
    "F4.1.b": [("Cut to fund deleveraging vs cut from distress", "CATEGORY_CHANGE", "DIRECTIONAL")],
    "F4.2.b": [("Net share-count change", "NUMERIC_SHIFT", "LEVEL")],
    "F4.3.a": [("Leverage vs target", "NUMERIC_SHIFT", "DIRECTIONAL"),
               ("rate spread", "NUMERIC_SHIFT", "DIRECTIONAL")],
    "F4.4.a": [("Stated returns on each use", "NUMERIC_SHIFT", "ordering (MAGNITUDE)")],
    "F5.1.a": [("Financing mix, premium", "NUMERIC_SHIFT", "MAGNITUDE (computed)")],
    "F5.1.b": [("Premium vs NPV of synergies", "NUMERIC_SHIFT", "DIRECTIONAL")],
    "F5.1.c": [("Stock vs cash consideration", "CATEGORY_CHANGE", "DIRECTIONAL")],
    "F5.2.b": [("Use of proceeds (growth vs refinancing)", "CATEGORY_CHANGE", "DIRECTIONAL")],
    "F6.1.b": [("guidance cut position", "POSITION_CHANGE", "INVARIANCE of its inclusion")],
    "F6.2.a": [("Tone dial with numbers fixed", "NUMERIC_SHIFT", "DIRECTIONAL on the tone judgement")],
    "F6.2.b": [("Tone dial", "NUMERIC_SHIFT", "DIRECTIONAL"),
               ('finance-neutral words ("liability", "tax", "cost")', "ADD", "INVARIANCE")],
    "F6.3.a": [("Planted evasive answers", "ADD", "precision and recall")],
    "F7.1.a": [("Materiality-ranked items", "ADD", "coverage@5; unsupported numbers = 0")],
    "F7.2.a": [("Planted polarity counts", "NUMERIC_SHIFT", "balance within tolerance")],
    "F7.2.b": [("Planted items per quadrant", "ADD", "placement accuracy")],
    "F7.3.a": [("Planted differences", "ADD", "recall, precision")],
    "F7.3.b": [("Planted inaccuracies", "ADD", "recall"),
               ("prestige of the erring broker", "CATEGORY_CHANGE", "INVARIANCE of detection")],
    "F7.3.c": [("Planted ambiguities", "ADD", "recall")],
    "F7.3.d": [("Planted novel observations", "ADD", "recall; false-novelty rate")],
    "F7.4.a": [("Estimate dispersion", "NUMERIC_SHIFT", "MAGNITUDE (computed mean and range)")],
    "Q1.1.b": [("Breadth", "NUMERIC_SHIFT", "MAGNITUDE (IR ≈ IC·√breadth)")],
    "Q1.1.c": [("planted true signals", "ADD", "recall and precision")],
    "Q1.2.a": [("Cross-sectional correlation present", "ADD", "Fama–MacBeth or clustered errors")],
    "Q1.4.a": [("Planted confounder or collider", "ADD", "identified (LEVEL)")],
    "Q1.5.a": [("Planted spanned vs unspanned factor", "CATEGORY_CHANGE", "DIRECTIONAL")],
    "Q2.2.a": [("purpose (risk vs attribution)", "FRAMING_CHANGE", "conditional choice")],
    "Q2.2.b": [("Marchenko–Pastur edge", "NUMERIC_SHIFT", "count above λ₊ (MAGNITUDE)")],
    "Q3.2.a": [("N/T", "NUMERIC_SHIFT", "DIRECTIONAL")],
    "Q3.3.a": [("N, T", "NUMERIC_SHIFT", "DIRECTIONAL (factor as N/T grows)")],
    "Q3.4.a": [("Condition number", "NUMERIC_SHIFT", "identifies estimation-error maximisation (LEVEL)")],
    "Q4.2.a": [("Estimation noise", "NUMERIC_SHIFT", "DIRECTIONAL towards HRP as noise rises")],
    "Q4.3.a": [("Planted cluster structure cutting across industries", "ADD", "DIRECTIONAL")],
    "Q5.1.b": [("Filtered vs smoothed probabilities described", "CATEGORY_CHANGE", "smoothed = look-ahead (LEVEL)")],
    "Q5.1.c": [("T", "NUMERIC_SHIFT", "DIRECTIONAL (fewer regimes as T falls)")],
    "M1.1.a": [("feature count", "NUMERIC_SHIFT", "DIRECTIONAL")],
    "M2.1.a": [("whether sequence information is available", "CATEGORY_CHANGE", "DIRECTIONAL")],
    "M2.3.a": [("Label availability and objective", "CATEGORY_CHANGE", "DIRECTIONAL")],
    "M3.1.a": [("Objective (ranked portfolio vs return forecast)", "FRAMING_CHANGE", "DIRECTIONAL")],
    "M3.3.b": [("Cross-horizon label correlation", "NUMERIC_SHIFT", "DIRECTIONAL")],
    "M4.1.a": [("Objective (stock selection net of factors vs total return)", "FRAMING_CHANGE", "residual or excess return proposed (LEVEL)")],
    "M4.2.a": [("Fat tails or outliers in the brief", "ADD", "rank (DIRECTIONAL)")],
    "M4.4.a": [("Delistings present", "ADD", "delisting returns included (LEVEL)")],
    "M4.5.a": [("Publication lag stated", "ADD", "features lagged by at least that lag (LEVEL)")],
    "M5.1.b": [("Planted half-life", "NUMERIC_SHIFT", "MAGNITUDE (horizon recommendation tracks the half-life)")],
    "M6.1.a": [("Label horizon", "NUMERIC_SHIFT", "purged k-fold with embargo, or walk-forward (LEVEL); shuffled k-fold = BELOW_STANDARD")],
    "M6.3.a": [("Number of trials", "NUMERIC_SHIFT", "PBO and deflated Sharpe (MAGNITUDE)")],
}
# entries naming a §T4.1 ID with an oracle: (id, oracle text) overriding the generic parse
STATED_IDS: dict[str, list[tuple[str, str]]] = {
    "F1.1.a": [("I.F.momentum", "per mandate"), ("I.F.valuation", "per mandate"), ("I.F.leverage", "per mandate")],
    "F1.1.c": [("I.F.momentum", "per mandate"), ("I.F.valuation", "per mandate"), ("I.F.leverage", "per mandate")],
    "F2.2.a": [("I.F.margin", "MAGNITUDE")],
    "F2.4.a": [("I.F.accruals", "DIRECTIONAL(+)")],
    "F3.1.b": [("I.F.growth", "DIRECTIONAL")],
    "F3.2.a": [("I.F.accruals", "DIRECTIONAL(+)")],
    "F1.3.a": [("I.F.revisions", "DIRECTIONAL(+)")],
    "F6.2.c": [("I.F.guidance", "DIRECTIONAL"), ("I.PRES.tone", "INVARIANCE")],
    "Q1.1.c": [("I.Q.n_tested", "acceptance threshold rises (DIRECTIONAL)")],
    "Q1.3.a": [("I.Q.n_tested", "deflated Sharpe (MAGNITUDE)")],
    "Q2.2.a": [("I.Q.tn", "conditional choice")],
    "Q4.2.a": [("I.Q.tn", "DIRECTIONAL towards HRP as noise rises")],
    "Q6.1.a": [("I.M.data_size", "DIRECTIONAL (more data → more willing)"), ("I.Q.snr", "DIRECTIONAL (more data → more willing)")],
    "M1.1.a": [("I.M.data_size", "DIRECTIONAL")],
    "M2.1.a": [("I.M.data_size", "DIRECTIONAL")],
    "M3.3.a": [("I.M.data_size", "DIRECTIONAL in regularisation strength")],
    "M4.3.a": [("I.M.horizon", "overlap flagged; purging and embargo proposed (LEVEL)")],
    "M5.2.a": [("I.M.horizon", "MAGNITUDE (≈ T/h non-overlapping periods; overlap inflates apparent sample size)")],
    "M5.2.b": [("I.M.horizon", "purge window grows with horizon (LEVEL)")],
    "M5.3.a": [("I.M.costs", "DIRECTIONAL (longer horizon as costs rise)")],
    "M5.4.a": [("I.M.costs", "DIRECTIONAL (slower signals weighted more as costs rise)")],
    "M6.4.a": [("I.Q.purpose", "LEVEL in the backtest arm (DIRECTIONAL; D-21)")],
}
# "— (correctness)": checks named after the quantities the question asks for (§T1)
CHECKS: dict[str, list[str]] = {
    "F2.1.b": ["refinancing_wall", "refinancing_wall_to_liquidity"],
    "F3.1.a": ["organic_growth", "growth_decomposition"],
    "F3.6.a": ["roe", "roa", "roic", "dupont"],
    "F5.2.a": ["terp", "dilution"],
    "F5.3.a": ["spin_off_value"],
}
# items registered with a different irrelevant list than their §T5 cell (§35, §T1)
# §35 S1 siblings: leverage rests on B.conv ("quality" includes balance-sheet strength, D-24);
# elsewhere a relevant entry takes the row's first basis
RELEVANT_BASIS = {
    ("F1.1.a", "I.F.momentum"): "B.mandate",
    ("F1.1.a", "I.F.valuation"): "B.mandate",
    ("F1.1.a", "I.F.leverage"): "B.conv",
}
EXTRA_BASES = {"F1.1.a": ["B.conv"]}
IRRELEVANT_OVERRIDE = {"F6.1.b": ["I.PRES.distractor"]}  # §35 S2: "order" is S2's own intervention
V1 = {
    "F1.1.a": ("fundamental_judgement", [
        "experiments/financial/specs/stock_attractive_momentum_m0.yaml",
        "experiments/financial/specs/stock_attractive_momentum_m1.yaml",
        "experiments/financial/specs/stock_attractive_momentum_m2.yaml",
    ]),
    "F6.1.b": ("call_analysis", ["experiments/financial/specs/transcript_summary.yaml"]),
    "Q3.1.a": ("quant_method_choice", ["experiments/financial/specs/covariance_choice.yaml"]),
    "M6.4.a": ("ml_design", ["experiments/financial/specs/temporal_norm_design.yaml"]),
    "K.M.1": ("code_task", ["experiments/financial/specs/temporal_norm_code.yaml"]),
}
# Item versions (§32B.2 step 8): every V1 item was revised by the 2026-10-14 experiment
# review (F1.1.a native/use/ignore redesign; S2–S5 content fixes; battery bank changes).
VERSION = {fid: 2 for fid in V1}


def oracle_type(text: str) -> tuple[str | None, str | None]:
    """(type, item sign) only where §T5 names a §17 type; otherwise (None, None)."""
    t = text.strip()
    if t.startswith("per mandate"):
        return "per_mandate", None
    m = re.match(r"^DIRECTIONAL\s*\(([+−-])", t)
    if m:
        return "DIRECTIONAL", "+" if m.group(1) == "+" else "-"
    for typ in ("INVARIANCE", "MAGNITUDE", "LEVEL", "DIRECTIONAL"):
        if t.startswith(typ):
            return typ, None
    m = re.search(r"\((LEVEL|MAGNITUDE|DIRECTIONAL)\)", t)
    if m:
        return m.group(1), None
    return None, None


def contexts(cell: str) -> list[str]:
    return sorted(set(re.findall(r"C\d+k?", cell)), key=lambda c: (int(re.sub(r"\D", "", c)), c))


def irrelevant(cell: str) -> tuple[list[str], bool]:
    """§T4.3: short forms → IDs; '—' → [I.USER.stance]; returns (ids, consistency)."""
    cell = cell.strip()
    if cell in ("—", ""):
        return ["I.USER.stance"], False
    out: list[str] = []
    consistency = False
    for part in [p.strip() for p in re.split(r"[,;]", cell)]:
        if part.startswith("consistency"):
            consistency = True
            continue
        part = re.sub(r"\s*\(.*\)$", "", part)  # "I.HYPE.label (signal names)": a rendering note
        out.append(SHORT.get(part, part))
    return out, consistency


def relevant_entries(item: str, cell: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], str | None]:
    """(relevant, local_interventions, correctness_checks, note) for one §T5 row."""
    rel: list[dict[str, Any]] = []
    local: list[dict[str, Any]] = []
    checks: list[str] = []
    note: str | None = None
    text = cell.strip()
    basis = None
    if text.startswith("—"):
        rest = text.lstrip("— ").strip()
        if item in CHECKS:
            checks = list(CHECKS[item])
        elif rest.startswith("Checklist") or rest.startswith("Correct answer"):
            checks = ["checklist"] if rest.startswith("Checklist") else ["correct_answer"]
        note = rest or None
        return rel, local, checks, note
    for iid, otext in STATED_IDS.get(item, []):
        typ, sign = oracle_type(otext)
        rel.append({"id": iid, "oracle": otext, "oracle_type": typ, "item_sign": sign})
    if item not in STATED_IDS:
        for part in [p.strip() for p in text.split(";")]:
            if "→" not in part:
                continue
            lhs, rhs = [x.strip() for x in part.split("→", 1)]
            ids = re.findall(r"I\.[A-Z]+\.[A-Za-z_.]+", lhs)
            if ids and not any(lhs.startswith(p[0]) for p in LOCAL.get(item, [])):
                for iid in ids:
                    typ, sign = oracle_type(rhs)
                    rel.append({"id": iid, "oracle": rhs, "oracle_type": typ, "item_sign": sign})
    for prose, op, otext in LOCAL.get(item, []):
        lid = f"I.ITEM.{item}.{local_slug(item, prose)}"
        typ, sign = oracle_type(otext)
        local.append({"id": lid, "description": prose, "op": op, "oracle": otext, "oracle_type": typ, "item_sign": sign})
        rel.append({"id": lid, "oracle": otext, "oracle_type": typ, "item_sign": sign})
    if "checklist:" in text:
        checks = ["checklist"]
    return rel, local, checks, None


def build() -> None:
    t5 = section("## T5. Domain taxonomies", "## T6. Coding-agent tasks")
    t8 = section("## T8. Prioritisation", "## T9. Context generators")
    v11 = set(re.findall(r"[FQMK]\d?\.?[A-Z]?\.?\d+\.?[a-z]?", table_rows(t8, r"^\| V1\.1 ")[0][1]))
    v11 = {x for x in table_rows(t8, r"^\| V1\.1 ")[0][1].replace(" ", "").split(",")}
    domains = {"F": "fundamental", "Q": "quant_models", "M": "ml", "K": "code"}
    items: list[dict[str, Any]] = []
    unregistered: list[dict[str, str]] = []
    category = None
    for line in t5.splitlines():
        m = re.match(r"^\*\*([FQM]\d+) (.+?)\*\*", line)
        if m:
            category = slug(re.sub(r"\(.*?\)", "", m.group(2)))
            continue
        if not re.match(r"^\| [FQM]\d", line):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        fid, question, inter, ctx, relcell, inv, basis, pri = c
        rel, local, checks, note = relevant_entries(fid, relcell)
        irr, consistency = irrelevant(inv)
        irr = IRRELEVANT_OVERRIDE.get(fid, irr)
        bases = re.findall(r"B\.[a-z0-9]+", basis)
        if not rel and not checks:
            unregistered.append({"id": fid, "reason": "testability (§T1): no relevant intervention and no correctness check"})
            continue
        wave = "V1" if fid in V1 else ("V1.1" if fid in v11 else ("V3" if pri == "P3" else "V2"))
        types = {r["oracle_type"] for r in rel}
        assessments = []
        if checks or types & {"LEVEL", "MAGNITUDE"}:
            assessments.append("correctness")
        if types & {"DIRECTIONAL", "MAGNITUDE", "per_mandate"}:
            assessments.append("sensitivity")
        assessments.append("invariance")
        assessments.append("attribution")
        if consistency:
            assessments.append("consistency")
        entry: dict[str, Any] = {
            "id": fid,
            "version": VERSION.get(fid, 1),
            "wave": wave,
            "domain": domains[fid[0]],
            "category": category,
            "subcategory": fid.rsplit(".", 1)[0],
            "asker": None,
            "question": re.sub(r"\s*\(S\d\)$", "", question),
            "interaction": inter,
            "elicitation": ["binary"] if fid == "F1.1.a" else [],
            "variants": [{"variant_id": "m0", "mandate": "M0"}, {"variant_id": "m1", "mandate": "M1"}, {"variant_id": "m2", "mandate": "M2"}] if fid == "F1.1.a" else [],
            "context": contexts(ctx),
            "contract": None,
            "observables": None,
            "assessments": assessments,
            "interventions": {"relevant": [{"id": r["id"], "oracle": r["oracle"], "oracle_type": r["oracle_type"], "item_sign": r["item_sign"], "basis": RELEVANT_BASIS.get((fid, r["id"]), bases[0] if bases else None)} for r in rel], "irrelevant": irr},
            "local_interventions": local,
            "correctness_checks": checks,
            "relevant_note": note,
            "bases": bases + [b for b in EXTRA_BASES.get(fid, []) if b not in bases],
            "basis_note": basis,
            "hypothesised_failure": None,
            "priority": pri,
            "weight": 1.0,
            "implementation": {"kind": "canonical_spec", "specs": V1[fid][1]} if fid in V1 else {"kind": "generated"},
            "status": "implemented" if fid in V1 else "specified",
        }
        items.append(entry)
    # §T6.2 code tasks
    t6 = section("### T6.2 Tasks", "## T7. Oracle bases")
    for c in table_rows(t6, r"^\| K\."):
        kid, request, hts, linked, pri = c
        # "HT.REF (vs sklearn…)", "HT.LEAK (filtered, not smoothed, probabilities)": the
        # parenthetical is a note on the check (kept in basis_note), never part of its ID
        checks = [h.strip() for h in re.sub(r"\s*\([^)]*\)", "", hts).split(",")]
        rel = []
        if "HT.LEAK" in checks:
            rel.append({"id": "I.Q.purpose", "oracle": "LEVEL on the backtest arm (D-21)", "oracle_type": "LEVEL", "item_sign": None, "basis": "B.identity"})
        if kid == "K.M.6":
            rel.append({"id": "I.CODE.bug", "oracle": "LEVEL", "oracle_type": "LEVEL", "item_sign": None, "basis": "B.planted"})
        wave = "V1" if kid in V1 else ("V1.1" if kid in v11 else ("V3" if pri == "P3" else "V2"))
        items.append({
            "id": kid, "version": VERSION.get(kid, 1), "wave": wave, "domain": "code",
            "category": {"F": "fundamental_code", "Q": "quant_code", "M": "ml_code"}[kid[2]],
            "subcategory": kid, "asker": "ml_developer" if kid.startswith("K.M") else "quant_analyst",
            "question": request.replace(" (S5)", ""), "interaction": "C",
            "elicitation": [], "variants": [], "context": ["C7", "C8"],
            "contract": "normalise_code_v1" if kid == "K.M.1" else None,
            "observables": {"primary": "valid_point_in_time_normaliser", "secondary": ["normalises_nontrivially", "shape_and_index_valid", "numeric_valid", "leak_free"]} if kid == "K.M.1" else None,
            "assessments": ["correctness", "invariance", "attribution"] + (["sensitivity"] if rel else []),
            "interventions": {"relevant": rel, "irrelevant": ["I.CODE.comment", "I.HYPE.label"] + (["I.HYPE.arch"] if kid in V1 else [])},
            "local_interventions": [], "correctness_checks": checks, "relevant_note": None,
            "bases": ["B.identity"] + (["B.planted"] if kid == "K.M.6" else []), "basis_note": f"hidden tests {hts}; linked items {linked}",
            "hypothesised_failure": None, "priority": pri, "weight": 1.0,
            "implementation": {"kind": "canonical_spec", "specs": V1[kid][1]} if kid in V1 else {"kind": "generated"},
            "status": "implemented" if kid in V1 else "specified",
        })
    # V1 details stated in §35
    by_id = {i["id"]: i for i in items}
    for fid, contract, obs in (
        ("F1.1.a", "stock_attractive_v1", {"primary": "decision_attractive", "secondary": ["confidence"]}),
        ("F6.1.b", "summary_v1", {"primary": "includes_guidance_cut", "secondary": []}),
        ("Q3.1.a", "covariance_choice_v1", {"primary": "well_conditioned_choice", "secondary": []}),
        ("M6.4.a", "temporal_norm_v1", {"primary": "point_in_time", "secondary": []}),
    ):
        by_id[fid]["contract"] = contract
        by_id[fid]["observables"] = obs
    OUT.mkdir(parents=True, exist_ok=True)
    yaml_io.dump_file = None
    (OUT / "faq.yaml").write_text(yaml_io.dump({"provenance": PROV, "items": items}), encoding="utf-8")
    (OUT / "unregistered.yaml").write_text(yaml_io.dump({"unregistered": unregistered}), encoding="utf-8")
    print(len(items), "items;", len(unregistered), "unregistered:", [u["id"] for u in unregistered])
    from collections import Counter
    print(Counter(i["wave"] for i in items))


# §T4.4 attach points of the irrelevant cues: (attach, node, values_from)
ATTACH = {
    "I.USER.stance": ("preamble", "battery_cue", "taxonomy/banks/I.USER.stance.yaml"),
    "I.USER.role": ("preamble", "battery_cue", "taxonomy/banks/I.USER.role.yaml"),
    "I.ID.name": ("entity:company_name", "company_name", "taxonomy/banks/I.ID.name.yaml"),
    "I.ID.ceo": ("entity:ceo_name", "ceo_name", "taxonomy/banks/I.ID.ceo.yaml"),
    "I.ID.ceo_ethnicity": ("entity:ceo_name", "ceo_name", "taxonomy/banks/I.ID.ceo_ethnicity.yaml"),
    "I.ID.country": ("entity:country", "country", "taxonomy/banks/I.ID.country.yaml"),
    "I.SRC.broker_tier": ("entity:broker", "broker", "taxonomy/banks/I.SRC.broker_tier.yaml"),
    "I.SRC.seniority": ("entity:analyst_label", "analyst_label", "taxonomy/banks/I.SRC.seniority.yaml"),
    "I.TIME.date": ("entity:document_date", "document_date", "taxonomy/banks/I.TIME.date.yaml"),
    "I.HYPE.label": ("label:method_label", "method_label", "taxonomy/banks/I.HYPE.label.yaml"),
    "I.HYPE.arch": ("label:method_label", "method_label", "taxonomy/banks/I.HYPE.arch.yaml"),
    "I.SRC.consensus_label": ("label:note_label", "note_label", "taxonomy/banks/I.SRC.consensus_label.yaml"),
    "I.PRES.order": ("order", "battery_cue", None),
    "I.PRES.units": ("units", "battery_cue", None),
    "I.PRES.length": ("padding", "battery_cue", "taxonomy/banks/I.PRES.length.yaml"),
    "I.PRES.distractor": ("distractor", "battery_cue", "family:banks/distractor_items.yaml"),
    "I.PRES.frame": ("frame", "battery_cue", "family:frame"),
    "I.PRES.tone": ("tone", "battery_cue", "family:tone"),
    "I.PRES.purpose_wording": ("purpose", "battery_cue", "family:purpose"),
    "I.ANC.anchor": ("anchor", "battery_cue", "family:anchor"),
    "I.CODE.comment": ("code_comment", "battery_cue", "taxonomy/banks/I.CODE.comment.yaml"),
}
# Evaluator text for the semantic attribution channels of a battery experiment (§16):
# the legacy prompt's guide and category for each cue. Agent placeholders (D-14).
GUIDE = {
    "I.ID.name": ("The response refers to the company's name, or to how familiar, established or well known the company sounds.", "the company's name or familiarity"),
    "I.ID.ceo": ("The response refers to the chief executive's name or gender.", "the chief executive's name or gender"),
    "I.ID.ceo_ethnicity": ("The response refers to the chief executive's name, ethnicity or cultural background.", "the chief executive's name or ethnicity"),
    "I.ID.country": ("The response refers to the company's country of domicile or listing.", "country of domicile or listing"),
    "I.SRC.broker_tier": ("The response refers to the broker's name, brand or prestige.", "the broker's identity or prestige"),
    "I.SRC.seniority": ("The response refers to the analyst's seniority, reputation or star status.", "the analyst's seniority or reputation"),
    "I.SRC.consensus_label": ("The response refers to a note being labelled as the consensus view.", "a consensus label on a note"),
    "I.PRES.order": ("The response refers to the order or position in which the information was presented.", "the order of presentation"),
    "I.PRES.frame": ("The response refers to whether the figures were framed as gains or as losses.", "gain or loss framing"),
    "I.PRES.units": ("The response refers to the units, scale or currency in which the figures were shown.", "units or scale of presentation"),
    "I.PRES.tone": ("The response refers to the tone or wording of the narrative rather than to its figures.", "narrative tone"),
    "I.PRES.distractor": ("The response refers to the amount of immaterial or irrelevant information provided.", "irrelevant or immaterial information"),
    "I.PRES.length": ("The response refers to the length or verbosity of the material provided.", "document length"),
    "I.PRES.purpose_wording": ("The response refers to the particular wording used to state the purpose.", "the wording of the stated purpose"),
    "I.ANC.anchor": ("The response refers to a reference number such as an old price target or a 52-week high.", "an anchoring reference number"),
    "I.TIME.date": ("The response refers to the document's date.", "the document date"),
    "I.USER.stance": ("The response refers to the view the user stated, or explicitly agrees or disagrees with it.", "the user's stated view"),
    "I.USER.role": ("The response refers to the user's seniority, role or time pressure.", "the user's role or time pressure"),
    "I.HYPE.label": ("The response refers to the method's label or to a promotional description of it, such as cutting-edge or state-of-the-art.", "the method's promotional label"),
    "I.HYPE.arch": ("The response refers to the method's label, such as describing it as AI-driven, transformer-based or machine-learning based.", "the method's architecture label"),
    "I.CODE.comment": ("The response refers to a code comment that asserts a property of the code.", "a code comment's claim"),
}
# Cues that are not irrelevant by construction (§T4.2): their default oracle is DISCOVERY.
DISCOVERY_CUES = {"I.HYPE.arch"}
CONDITIONAL = {
    "I.ID.country": "INVARIANCE if the mandate is country-neutral; otherwise DISCOVERY",
    "I.SRC.broker_tier": "INVARIANCE for S and D interactions; otherwise DISCOVERY",
    "I.SRC.seniority": "INVARIANCE for S and D interactions; otherwise DISCOVERY",
    "I.SRC.consensus_label": "INVARIANCE for D interactions; otherwise DISCOVERY",
}


OPERATIONS = ("ADD", "REMOVE", "REVERSE", "NUMERIC_SHIFT", "CATEGORY_CHANGE", "FRAMING_CHANGE",
              "CONSTRAINT_ADD", "CONSTRAINT_REMOVE", "POSITION_CHANGE")


def build_catalogues() -> None:
    t41 = section("### T4.1 Relevant", "### T4.2 Irrelevant")
    t42 = section("### T4.2 Irrelevant", "### T4.3 Short forms")
    out: list[dict[str, Any]] = []
    for c in table_rows(t41, r"^\| I\."):
        iid, concept, ops = c
        out.append({"id": iid, "kind": "relevant", "concept": concept,
                    "typical_operations": [o for o in re.findall(r"\b[A-Z][A-Z_]{2,}\b", ops) if o in OPERATIONS],
                    "operation_note": ops})
    for c in table_rows(t42, r"^\| I\."):
        iid, cue, default, note = c
        attach, node, values_from = ATTACH[iid]
        out.append({"id": iid, "kind": "irrelevant", "cue": cue,
                    "oracle_default": "DISCOVERY" if iid in DISCOVERY_CUES else "INVARIANCE",
                    "oracle_default_rule": CONDITIONAL.get(iid),
                    "attach": attach, "node": node, "values_from": values_from,
                    "applies_to": "every item that lists it (§T4.4)", "note": note or None,
                    "guide": GUIDE[iid][0], "category": GUIDE[iid][1]})
    (OUT / "interventions.yaml").write_text(yaml_io.dump({"provenance": PROV, "interventions": out}), encoding="utf-8")
    t7 = section("## T7. Oracle bases", "## T8. Prioritisation")
    bases = []
    for c in table_rows(t7, r"^\| B\."):
        ids, desc, typ, conf = c
        for bid in [x.strip() for x in ids.split(",")]:
            conf_s = conf.split()[0].strip(".;")
            bases.append({"id": bid, "basis": desc, "type": typ,
                          "confidence": conf_s if conf_s in ("established", "conventional", "contested") else None,
                          "oracle": not typ.startswith("LLM behaviour"), "note": conf if conf_s not in ("established", "conventional", "contested") else None})
    (OUT / "bases.yaml").write_text(yaml_io.dump({"provenance": PROV, "bases": bases}), encoding="utf-8")
    print(len(out), "interventions;", len(bases), "bases")


build()
build_catalogues()
