"""Per-model files: ``models/<model_key>/responses.jsonl`` and ``examples.md`` (spec §32A.7).

Both are functions of the database, the artefacts and the frozen bundle. The
example selection is seeded from ``design_hash`` and the model key, so it is
identical across renders, resumes and replays. Full prompts are never inlined.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from financial_llm_harness.core import bundle as B
from financial_llm_harness.core import ids
from financial_llm_harness.evaluation.parse import Contract, parse_response
from financial_llm_harness.execution.runner import manifest_contract
from financial_llm_harness.families import load_family
from financial_llm_harness.reporting.tables import RunData
from financial_llm_harness.storage.artifacts import ArtifactStore

REASONING_EXCERPT = 600
TRACE_EXCERPT = 300
N_RESPONSIVE = 10
N_CONTRADICTORY = 5
N_PARSE_FAILURES = 5

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|!~])")


def _excerpt(text: str | None, n: int) -> str | None:
    if text is None:
        return None
    return text[:n]


def md_escape(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", text.replace("\r", " ").replace("\n", " "))


def bold_mentions(text: str, lexicon: Sequence[str]) -> str:
    """Escape ``text`` for Markdown and bold every case-insensitive lexicon match."""
    terms = sorted({t for t in lexicon if t}, key=lambda t: (-len(t), t))
    if not terms:
        return md_escape(text)
    pat = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
    out: list[str] = []
    pos = 0
    for m in pat.finditer(text):
        out.append(md_escape(text[pos : m.start()]))
        out.append(f"**{md_escape(m.group(0))}**")
        pos = m.end()
    out.append(md_escape(text[pos:]))
    return "".join(out)


def _fenced(text: str) -> str:
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _compact(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, list):
        parts = []
        for item in v:
            if isinstance(item, Mapping):
                parts.append(":".join(str(item[k]) for k in item))
            else:
                parts.append(str(item))
        return "; ".join(parts) or "—"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _trial_rows(rd: RunData, model_key: str) -> list[dict[str, Any]]:
    q = rd.conn.execute(
        "SELECT t.trial_id, t.case_idx, t.arm, t.paraphrase_idx, t.repetition, t.status, rd.prompt_hash, "
        "r.finish_reason, r.text_artifact, r.trace_artifact, r.input_tokens, r.output_tokens, r.reasoning_tokens, "
        "r.latency_ms, r.cost_usd FROM trial t JOIN rendering rd ON rd.rendering_id = t.rendering_id "
        "LEFT JOIN response r ON r.attempt_id = t.used_attempt_id WHERE t.model_key = ?",
        (model_key,),
    ).fetchall()
    arm_order = {a: i for i, a in enumerate(rd.manifest["arms"])}
    rows = [dict(r) for r in q]
    rows.sort(
        key=lambda r: (
            r["case_idx"],
            arm_order.get(r["arm"], 99),
            r["paraphrase_idx"],
            r["repetition"],
        )
    )
    return rows


def _response_line(
    rd: RunData,
    t: Mapping[str, Any],
    parsed: Mapping[str, Any] | None,
    contract: Contract,
    mode: str,
    store: ArtifactStore,
) -> dict[str, Any]:
    dh = rd.manifest["design_hash"]
    line: dict[str, Any] = {
        "trial_id": t["trial_id"],
        "case_idx": t["case_idx"],
        "cluster_id": ids.cluster_id(dh, rd.cases[t["case_idx"]]["cluster_ordinal"]),
        "arm": t["arm"],
        "paraphrase_idx": t["paraphrase_idx"],
        "repetition": t["repetition"],
        "status": t["status"],
        "finish_reason": t["finish_reason"],
    }
    fields = list(contract.decision_fields)
    for extra in ("confidence", "material_factors"):
        if extra not in fields:
            fields.append(extra)
    for f in fields:
        line[f] = (parsed or {}).get(f)
    ft = contract.free_text_key(mode)
    free = (parsed or {}).get(ft) if ft else None
    line["reasoning_excerpt"] = _excerpt(
        free if isinstance(free, str) else None, REASONING_EXCERPT
    )
    trace = store.get_text(t["trace_artifact"]) if t["trace_artifact"] else None
    line["trace_excerpt"] = _excerpt(trace, TRACE_EXCERPT)
    line["input_tokens"] = t["input_tokens"]
    line["output_tokens"] = t["output_tokens"]
    line["reasoning_tokens"] = t["reasoning_tokens"]
    line["latency_ms"] = t["latency_ms"]
    line["cost_usd"] = t["cost_usd"]
    line["prompt_sha256"] = t["prompt_hash"]
    line["response_sha256"] = t["text_artifact"]
    line["trace_sha256"] = t["trace_artifact"]
    return line


def _diff_table(case: Mapping[str, Any], arm_a: str, arm_b: str) -> str:
    diff = case.get("diff") or []
    sa, sb = case["arms"][arm_a], case["arms"][arm_b]
    rows = [f"| Field | {arm_a} | {arm_b} |", "|---|---|---|"]
    for node in diff:
        rows.append(
            f"| `{node}` | `{_compact(sa.get(node))}` | `{_compact(sb.get(node))}` |"
        )
    return "\n".join(rows)


def _seeded_pick(n: int, k: int, *seed_parts: object) -> list[int]:
    if n <= k:
        return list(range(n))
    rng = np.random.default_rng(ids.uint64(*seed_parts))
    return sorted(int(i) for i in rng.choice(n, size=k, replace=False))


def _arm_block(
    lines: Mapping[str, Mapping[str, Any]],
    arms: Sequence[str],
    show_fields: Sequence[str],
    lexicon: Sequence[str],
) -> list[str]:
    out = [
        "| | " + " | ".join(arms) + " |",
        "|---|" + "|".join("---" for _ in arms) + "|",
    ]
    out.append(
        "| status | " + " | ".join(f"`{lines[a]['status']}`" for a in arms) + " |"
    )
    for f in show_fields:
        out.append(
            f"| {f} | "
            + " | ".join(f"`{_compact(lines[a].get(f))}`" for a in arms)
            + " |"
        )
    out.append("")
    for a in arms:
        ex = lines[a].get("reasoning_excerpt")
        if ex:
            out.append(f"Arm {a} reasoning excerpt:")
            out.append("")
            out.append("> " + bold_mentions(ex, lexicon))
            out.append("")
    out.append(
        "Full text: "
        + "; ".join(
            f"{a} prompt `{lines[a]['prompt_sha256']}`, response `{lines[a]['response_sha256']}`"
            for a in arms
        )
    )
    out.append("")
    return out


def render_models(rd: RunData, clusters: Sequence[Mapping[str, Any]]) -> None:
    """Write ``models/<model_key>/responses.jsonl`` and ``examples.md`` for every panel model."""
    manifest = rd.manifest
    contract = manifest_contract(manifest)
    mode = manifest["response_spec"]["reasoning_mode"]
    abstain = bool(manifest["response_spec"].get("abstain_allowed", False))
    store = ArtifactStore(rd.data_dir / "artifacts")
    family = load_family(manifest["task_family"])
    idef = family.interventions.get(manifest.get("intervention_id") or "")
    lexicon = list(idef.concept_lexicon) if idef is not None else []
    pf = rd.primary_feature()["feature_id"]
    feature_ids = {f["feature_id"] for f in manifest["feature_spec"]}
    show_fields = [
        f
        for f in contract.decision_fields
        if contract.fields.get(f, {}).get("type") != "object_list"
    ]
    dh = manifest["design_hash"]
    models_dir = rd.run_dir / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    for mk in rd.model_keys:
        out_dir = models_dir / mk
        out_dir.mkdir(parents=True)
        trials = _trial_rows(rd, mk)
        lines: list[dict[str, Any]] = []
        by_block: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(
            dict
        )
        raw_text: dict[str, str | None] = {}
        for t in trials:
            text = store.get_text(t["text_artifact"]) if t["text_artifact"] else None
            raw_text[t["trial_id"]] = text
            parsed = None
            if text is not None and contract.kind == "json":
                res = parse_response(contract, mode, abstain, text)
                parsed = res.data if res.status == "SUCCESS" else None
            line = _response_line(rd, t, parsed, contract, mode, store)
            lines.append(line)
            by_block[(t["case_idx"], t["paraphrase_idx"], t["repetition"])][
                t["arm"]
            ] = line
        B.atomic_write_text(out_dir / "responses.jsonl", B.jsonl_text(lines))

        prim = rd.primary_row(mk)
        theta = (prim or {}).get("results", {}).get("theta")
        e = 0 if not theta else (1 if theta > 0 else -1)
        values = {
            r[0]: json.loads(r[1]) if r[1] is not None else None
            for r in rd.conn.execute(
                "SELECT o.trial_id, o.value_json FROM observation o JOIN trial t ON t.trial_id = o.trial_id "
                "WHERE t.model_key = ? AND o.feature_id = ? AND o.status = 'OK'",
                (mk, pf),
            ).fetchall()
        }
        responsive = {
            c["cluster_ordinal"]
            for c in clusters
            if c["model_key"] == mk
            and c["contrast"] == "B_vs_A"
            and c.get("responsive") is True
        }
        pairs: list[tuple[int, int, int]] = []
        if e != 0 and "A" in manifest["arms"] and "B" in manifest["arms"]:
            for key in sorted(by_block):
                blk = by_block[key]
                if "A" not in blk or "B" not in blk:
                    continue
                if rd.cases[key[0]]["cluster_ordinal"] not in responsive:
                    continue
                va, vb = (
                    values.get(blk["A"]["trial_id"]),
                    values.get(blk["B"]["trial_id"]),
                )
                if (
                    isinstance(va, int | float)
                    and isinstance(vb, int | float)
                    and e * (float(vb) - float(va)) > 0
                ):
                    pairs.append(key)
        contra: list[dict[str, Any]] = []
        if "contradictory_reasoning" in feature_ids:
            flagged = {
                r[0]
                for r in rd.conn.execute(
                    "SELECT o.trial_id, o.value_json FROM observation o JOIN trial t ON t.trial_id = o.trial_id "
                    "WHERE t.model_key = ? AND o.feature_id = 'contradictory_reasoning' AND o.status = 'OK'",
                    (mk,),
                ).fetchall()
                if r[1] is not None and json.loads(r[1]) in (1, True)
            }
            contra = [ln for ln in lines if ln["trial_id"] in flagged]
        failures = [ln for ln in lines if ln["status"] == "PARSE_FAILURE"]

        md: list[str] = [f"# Examples: `{mk}`", ""]
        md.append(
            f"Seeded selection (seed key: design `{dh[:8]}`, model `{mk}`): up to {N_RESPONSIVE} responsive pairs, "
            f"{N_CONTRADICTORY} contradictory-reasoning cases and {N_PARSE_FAILURES} parse failures. "
            "Concept mentions are in bold (deterministic lexicon). Full prompts are not inlined; artefact hashes identify them."
        )
        md.append("")
        md += ["## 1. Responsive pairs", ""]
        pick = _seeded_pick(len(pairs), N_RESPONSIVE, "examples", dh, mk, "responsive")
        if not pick:
            md += ["None.", ""]
        for j, idx in enumerate(pick, 1):
            key = pairs[idx]
            case = rd.cases[key[0]]
            blk = by_block[key]
            md.append(
                f"### 1.{j} Case {key[0]}, paraphrase {key[1]}, repetition {key[2]}"
            )
            md.append("")
            md.append(
                f"Case `{ids.testcase_id(dh, key[0])}` · cluster `{blk['A']['cluster_id']}`"
            )
            md.append("")
            md += [_diff_table(case, "A", "B"), ""]
            md += _arm_block(blk, ["A", "B"], show_fields, lexicon)
        md += ["## 2. Contradictory reasoning", ""]
        pick = _seeded_pick(
            len(contra), N_CONTRADICTORY, "examples", dh, mk, "contradictory"
        )
        if not pick:
            md += ["None.", ""]
        for j, idx in enumerate(pick, 1):
            ln = contra[idx]
            case = rd.cases[ln["case_idx"]]
            md.append(
                f"### 2.{j} Case {ln['case_idx']}, arm {ln['arm']}, paraphrase {ln['paraphrase_idx']}"
            )
            md.append("")
            md.append(
                f"Case `{ids.testcase_id(dh, ln['case_idx'])}` · cluster `{ln['cluster_id']}`"
            )
            md.append("")
            if "A" in case["arms"] and "B" in case["arms"]:
                md += [_diff_table(case, "A", "B"), ""]
            md += _arm_block({ln["arm"]: ln}, [ln["arm"]], show_fields, lexicon)
        md += ["## 3. Parse failures", ""]
        pick = _seeded_pick(
            len(failures), N_PARSE_FAILURES, "examples", dh, mk, "parse_failure"
        )
        if not pick:
            md += ["None.", ""]
        for j, idx in enumerate(pick, 1):
            ln = failures[idx]
            md.append(
                f"### 3.{j} Case {ln['case_idx']}, arm {ln['arm']}, paraphrase {ln['paraphrase_idx']}"
            )
            md.append("")
            md.append(
                f"Case `{ids.testcase_id(dh, ln['case_idx'])}` · cluster `{ln['cluster_id']}`"
            )
            md.append("")
            text = raw_text.get(ln["trial_id"]) or ""
            md += ["Response excerpt:", "", _fenced(text[:REASONING_EXCERPT]), ""]
            md.append(
                f"Full text: prompt `{ln['prompt_sha256']}`, response `{ln['response_sha256']}`"
            )
            md.append("")
        B.atomic_write_text(out_dir / "examples.md", "\n".join(md).rstrip("\n") + "\n")
