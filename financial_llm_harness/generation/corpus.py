"""The pre-hash corpus: cases, clusters, control subset and exact renderings (spec §5, §6.2, §9, §22).

Deterministic given the design, its seeds, the executable source and the
consumed assets. No step here reads or computes a hash-derived ID (§3, §4).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as sstats

from financial_llm_harness.core.canonical_json import sha256_canonical
from financial_llm_harness.core.ids import uint64
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.evaluation.parse import Contract, contract_from_dict
from financial_llm_harness.families import (
    BankTracker,
    TaskFamily,
    load_clause,
    load_contract,
    load_family,
    load_yaml_file,
    rel_to_package,
)
from financial_llm_harness.generation import render as R
from financial_llm_harness.generation.intervention import (
    InterventionSpec,
    apply_intervention,
    check_closure_declaration,
    declared_diff,
    diff_nodes,
    parse_intervention_spec,
    placebo_state,
    set_node,
)
from financial_llm_harness.generation.scenario import (
    Dist,
    World,
    attempt_rng,
    build_world,
    draw_state,
    violated_constraints,
)
from financial_llm_harness.generation.validate_case import (
    CaseRejected,
    check_answerability,
    check_measurability,
    check_neutrality,
    check_round_trip,
    check_state_isolation,
    check_text_isolation,
)
from financial_llm_harness.generation.validate_corpus import (
    Clustering,
    chaining_guard,
    check_direction_balance,
    check_template_concentration,
    cluster_cases,
    realised_correlation,
    realised_marginals,
    structured_edges,
    template_concentration,
    text_edges,
)
from financial_llm_harness.statistics import power

ARM_ORDER = ("A", "B", "A0", "PL", "PCA", "PCB")
CONTROL_ARMS = ("A0", "PL", "PCA", "PCB")
MAX_REJECTION_SHARE = 0.30


def derive_arms(design: Mapping[str, Any]) -> list[str]:
    """§3: [A, B] plus, in confirmatory mode, A0, PL, PCA, PCB; correctness: [A] + A0."""
    correctness = design["intervention_id"] == "I.NONE"
    arms = ["A"] if correctness else ["A", "B"]
    if design["mode"] != "confirmatory":
        return arms
    controls = design["controls"]
    if controls.get("null_arm"):
        arms.append("A0")
    if not correctness:
        if controls.get("placebo"):
            arms.append("PL")
        if controls.get("positive"):
            arms += ["PCA", "PCB"]
        if arms != list(ARM_ORDER):
            raise HarnessValidationError(
                "a confirmatory experiment needs the null, placebo and positive controls (§2.11, §22)"
            )
    elif "A0" not in arms:
        raise HarnessValidationError(
            "a confirmatory correctness experiment needs the null arm (§22)"
        )
    return arms


@dataclass
class Templates:
    base: R.BaseTemplate
    base_logical: str
    paraphrases: list[R.Paraphrase]
    paraphrase_logicals: list[str]
    data: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_templates(design: Mapping[str, Any], family: TaskFamily) -> Templates:
    ps = design["prompt_spec"]
    fam_ids = list(ps["template_family_ids"])
    if len(fam_ids) != 1:
        raise HarnessValidationError(
            "V1 renders one template family per experiment (prompt_spec.template_family_ids)"
        )
    base_rel = f"templates/{fam_ids[0]}.yaml"
    base_data = load_yaml_file(family.family_id, base_rel)
    base_logical = f"families/{family.family_id}/{base_rel}"
    base = R.parse_base_template(base_data, base_logical)
    paras: list[R.Paraphrase] = []
    logicals: list[str] = []
    data = {base_logical: base_data}
    pspec = design["paraphrase_spec"]
    for i, rel in enumerate(pspec["template_files"]):
        prefix = f"families/{family.family_id}/"
        if not str(rel).startswith(prefix):
            raise HarnessValidationError(f"paraphrase file {rel!r} is outside {prefix}")
        pdata = load_yaml_file(family.family_id, str(rel)[len(prefix) :])
        p = R.parse_paraphrase(pdata, str(rel))
        if p.template_family_id not in fam_ids:
            raise HarnessValidationError(
                f"{rel}: template family {p.template_family_id!r} not in prompt_spec"
            )
        if p.paraphrase_family != pspec["families"][i]:
            raise HarnessValidationError(
                f"{rel}: paraphrase family {p.paraphrase_family!r} != paraphrase_spec.families[{i}]"
            )
        R.check_literal_numerals(base, p)
        paras.append(p)
        logicals.append(str(rel))
        data[str(rel)] = pdata
    if len(paras) != int(pspec["P"]):
        raise HarnessValidationError(
            "paraphrase_spec.P must equal the number of template files"
        )
    return Templates(base, base_logical, paras, logicals, data)


def load_design_contract(
    design: Mapping[str, Any], family: TaskFamily
) -> tuple[Contract, str, dict[str, Any]]:
    rs = design["response_spec"]
    cid = rs["contract_id"]
    if cid not in family.response_contracts:
        raise HarnessValidationError(
            f"contract {cid!r} is not a contract of family {family.family_id!r}"
        )
    data = load_contract(family.family_id, cid)
    contract = contract_from_dict(data, f"contracts/{cid}.yaml")
    if contract.version != int(rs["contract_version"]):
        raise HarnessValidationError(
            f"contract {cid} version {contract.version} != response_spec.contract_version"
        )
    return contract, f"families/{family.family_id}/contracts/{cid}.yaml", data


@dataclass
class ArmRender:
    """Per-arm rendering inputs (mandate text and cue overrides)."""

    mandate: str
    preamble: str = ""
    display_overrides: dict[str, str] = field(default_factory=dict)
    precision_overrides: dict[str, int] = field(default_factory=dict)
    field_order: list[str] | None = None
    blocks: dict[str, list[tuple[str, str | None]]] = field(default_factory=dict)
    note: str = ""


CueHook = Callable[
    [int, str, dict[str, Any], np.random.Generator, list[str]], ArmRender | None
]


@dataclass
class Corpus:
    design_arms: list[str]
    cases: list[dict[str, Any]]
    renderings: list[dict[str, Any]]
    report: dict[str, Any]
    n_clusters: int
    subset_n: int
    primary_requirement: power.PrimaryRequirement
    control_requirements: list[power.ControlRequirement]
    bank_files: list[str]
    template_logicals: list[str]
    paraphrase_logicals: list[str]
    contract_logical: str
    clause_logicals: list[str]
    file_data: dict[str, dict[str, Any]]
    world: World
    family: TaskFamily


def _declared_moments(d: Dist) -> dict[str, Any]:
    out: dict[str, Any] = {"dist": d.text}
    try:
        if d.kind == "normal":
            out.update(mean=d.p("mu"), sd=d.p("sigma"))
        elif d.kind == "lognormal":
            mu, s = d.p("mu"), d.p("sigma")
            out.update(
                mean=math.exp(mu + s * s / 2),
                sd=math.sqrt((math.exp(s * s) - 1) * math.exp(2 * mu + s * s)),
            )
        elif d.kind == "uniform":
            lo, hi = d.p("low"), d.p("high")
            out.update(mean=(lo + hi) / 2, sd=(hi - lo) / math.sqrt(12))
        elif d.kind == "truncnormal":
            mu, s = d.p("mu"), d.p("sigma")
            a, b = (d.p("low") - mu) / s, (d.p("high") - mu) / s
            tn = sstats.truncnorm(a, b, loc=mu, scale=s)
            out.update(mean=float(tn.mean()), sd=float(tn.std()))
        elif d.kind == "beta":
            a, b, lo, hi = d.p("a"), d.p("b"), d.p("low"), d.p("high")
            out.update(
                mean=lo + (hi - lo) * a / (a + b),
                sd=(hi - lo) * math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))),
            )
        elif d.kind == "categorical":
            out.update(
                shares={str(v): p for v, p in zip(d.values, d.probs, strict=True)}
            )
        elif d.kind == "fixed":
            out.update(value=d.values[0])
    except (ValueError, ZeroDivisionError):
        pass
    return out


def _feature(design: Mapping[str, Any], fid: str) -> dict[str, Any]:
    for f in design["feature_spec"]:
        if f["feature_id"] == fid:
            return dict(f)
    raise HarnessValidationError(f"feature {fid!r} not in feature_spec")


def build_corpus(design: Mapping[str, Any], cue_hook: CueHook | None = None) -> Corpus:
    """Generate, validate and cluster the corpus; select controls; render every arm."""
    family = load_family(design["task_family"])
    fns = _fns_getter(family)
    with BankTracker() as tracker:
        corpus = _build(design, family, fns, tracker, cue_hook)
    corpus.bank_files = sorted(tracker.used)
    return corpus


def _fns_getter(family: TaskFamily) -> Callable[[str], Callable[..., Any]]:
    module = family.module

    def get(name: str) -> Callable[..., Any]:
        fn = getattr(module, name, None)
        if fn is None or not callable(fn):
            raise HarnessValidationError(
                f"family {family.family_id!r} has no function {name!r}"
            )
        return fn

    return get


def _build(
    design: Mapping[str, Any],
    family: TaskFamily,
    fns: Callable[[str], Callable[..., Any]],
    tracker: BankTracker,
    cue_hook: CueHook | None,
) -> Corpus:
    sc = design["scenario_spec"]
    world = build_world(family.scenario_schema, sc, fns)
    arms = derive_arms(design)
    ispec: InterventionSpec | None = None
    if design["intervention_spec"] is not None:
        ispec = parse_intervention_spec(design["intervention_spec"])
        check_closure_declaration(world, ispec)
    templates = load_templates(design, family)
    contract, contract_logical, contract_data = load_design_contract(design, family)
    rs = design["response_spec"]
    mode = rs["reasoning_mode"]
    abstain = bool(rs["abstain_allowed"])
    contract_text = contract.instruction_text(mode, abstain)
    response_contract = contract.response_contract(mode, abstain)
    file_data: dict[str, dict[str, Any]] = dict(templates.data)
    file_data[contract_logical] = contract_data
    mandate_text = design["mandate"]["text"]
    controls = design["controls"]
    clause_logicals: list[str] = []
    pc_mandate = mandate_text
    pos = controls.get("positive") if "PCA" in arms else None
    if pos is not None and pos.get("mandate_clause"):
        clause = load_clause(family.family_id, pos["mandate_clause"])
        logical = (
            f"families/{family.family_id}/mandates/clauses/{pos['mandate_clause']}.yaml"
        )
        clause_logicals.append(logical)
        file_data[logical] = clause
        pc_mandate = f"{mandate_text}; {clause['text']}"
    pcb_note = ""
    if pos is not None and pos.get("pcb_clause"):
        # PCB alone carries this sentence (§35 S3–S5: "PCB the same brief plus the sentence …")
        clause = load_clause(family.family_id, pos["pcb_clause"])
        logical = (
            f"families/{family.family_id}/mandates/clauses/{pos['pcb_clause']}.yaml"
        )
        clause_logicals.append(logical)
        file_data[logical] = clause
        pcb_note = f"{clause['text']}\n\n"
    # a family may render a block of items from the state (§35 S2): its render_blocks
    # gives each arm's blocks and BLOCK_NODES the state nodes each block shows
    blocks_fn = getattr(family.module, "render_blocks", None)
    block_nodes = getattr(family.module, "BLOCK_NODES", None)
    rendered_set: set[str] = set()
    for p in templates.paraphrases:
        rendered_set |= R.rendered_nodes(templates.base, p, block_nodes)
    primary = _feature(design, design["hypothesis"]["primary_feature"])
    check_measurability(primary)
    oracle_nodes = list(design["oracle_spec"].get("nodes") or [])
    if (
        ispec is not None
        and ispec.node not in oracle_nodes
        and design["oracle_spec"]["type"] not in ("DISCOVERY", "LEVEL")
    ):
        # a LEVEL rule references the level arm's state (oracle_spec.nodes), not the intervened node
        oracle_nodes.append(ispec.node)
    check_answerability(rendered_set, [n for n in oracle_nodes if n in world.nodes])
    leak_terms = list(design["prompt_spec"].get("leak_terms") or [])
    seeds = design["seeds"]
    n_cases = int(sc["n_cases"])
    policy = design["prompt_spec"]["field_order_policy"]
    accepted: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    k = 0
    while len(accepted) < n_cases:
        if k >= 2 * n_cases:
            raise HarnessValidationError(
                f"generate: {len(accepted)} of {n_cases} cases accepted after {k} attempts "
                f"(acceptance rate {len(accepted) / max(k, 1):.3f}); rejections {dict(rejections)} (§6.2)"
            )
        rng = attempt_rng(sc["seed_stream"], int(seeds["generation"]), k)
        k += 1
        case_idx = len(accepted)
        try:
            row = _attempt(
                design,
                world,
                ispec,
                arms,
                rng,
                case_idx,
                templates,
                contract_text,
                mandate_text,
                pc_mandate,
                leak_terms,
                policy,
                int(seeds["paraphrase"]),
                cue_hook,
                pcb_note,
                blocks_fn,
            )
        except CaseRejected as exc:
            rejections[exc.reason] += 1
            continue
        accepted.append(row)
    rejected = k - len(accepted)
    if rejected / k > MAX_REJECTION_SHARE:
        raise HarnessValidationError(
            f"generate: {rejected} of {k} attempts rejected (> 30%): the generator is defective (§9.1); "
            f"rejections {dict(rejections)}"
        )
    # --- corpus level (§9.2) ---------------------------------------------------
    base_states = [r["sampled"] for r in accepted]
    displayed = [r.pop("_displayed") for r in accepted]
    slot_texts = [r.pop("_slot_text") for r in accepted]
    s_edges = structured_edges(base_states, world.nodes, rendered_set, displayed)
    t_edges = text_edges(slot_texts)
    clustering: Clustering = cluster_cases(len(accepted), s_edges, t_edges)
    chaining_guard(clustering, n_cases)
    varying = _check_non_triviality(design, world, accepted, rendered_set, ispec)
    concentration = template_concentration(
        [[p.template_family_id for p in templates.paraphrases] for _ in accepted]
    )
    check_template_concentration(concentration)
    strata = _stratum_coverage(sc, accepted)
    direction_share = None
    if ispec is not None and ispec.counterbalance:
        by_cluster = [accepted[m[0]]["counterbalanced"] for m in clustering.members]
        direction_share = check_direction_balance(by_cluster)
    # --- power and the control subset (§22, §23.6) -------------------------------
    st = design["statistical_spec"]
    otype = design["oracle_spec"]["type"]
    tol = design["oracle_spec"].get("tol")
    primary_req = power.primary_requirement(
        otype, design["hypothesis"], st, primary, tol
    )
    ctrl_reqs = power.control_requirements(
        arms, controls, design["hypothesis"], st, primary
    )
    computed = (
        power.computed_subset_n(ctrl_reqs)
        if any(a in arms for a in CONTROL_ARMS)
        else 0
    )
    spec_subset = controls.get("subset_n")
    if spec_subset is not None and int(spec_subset) < computed:
        raise HarnessValidationError(
            f"controls.subset_n {spec_subset} is below the computed {computed} (§22, D-22)"
        )
    subset_n = computed if spec_subset is None else int(spec_subset)
    n_clusters = clustering.n_clusters
    if subset_n > n_clusters:
        raise HarnessValidationError(
            f"controls.subset_n {subset_n} exceeds N_clusters {n_clusters}; the subset is never truncated (§22)"
        )
    perm = np.random.default_rng(
        uint64("controls", int(seeds["generation"]))
    ).permutation(n_clusters)
    subset = {int(i) for i in perm[:subset_n]}
    # --- cases.jsonl and renderings.jsonl -----------------------------------------
    cases: list[dict[str, Any]] = []
    renderings: list[dict[str, Any]] = []
    for case in accepted:
        idx = case["case_idx"]
        ordinal = clustering.ordinal_of_case[idx]
        carried = [a for a in arms if a not in CONTROL_ARMS or ordinal in subset]
        arm_states = {a: case["arms"][a] for a in carried}
        cases.append(
            {
                "case_idx": idx,
                "cluster_ordinal": ordinal,
                "source": "synthetic",
                "split": sc["seed_stream"],
                "counterbalanced": case["counterbalanced"],
                "arms": arm_states,
                "state_hashes": {a: sha256_canonical(s) for a, s in arm_states.items()},
                "diff": case["diff"],
                "truth": case["truth"],
                "cue": case.get("cue"),
                "field_order": case["field_order"],
                "sampled": case["sampled"],
            }
        )
        for a in carried:
            for p_idx, msg in enumerate(case["messages"][a]):
                frozen = {
                    "system": msg[0],
                    "user": msg[1],
                    "response_contract": response_contract,
                }
                renderings.append(
                    {
                        "case_idx": idx,
                        "arm": a,
                        "paraphrase_idx": p_idx,
                        "system": msg[0],
                        "user": msg[1],
                        "response_contract": response_contract,
                        "prompt_hash": sha256_canonical(frozen),
                    }
                )
    marg = realised_marginals(base_states, world.nodes)
    declared = {name: _declared_moments(d) for name, d in world.dists.items()}
    corr = sc.get("correlation") or {}
    report = {
        "schema_version": "1.0",
        "n_cases": n_cases,
        "attempts": k,
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / k,
        "rejections": dict(sorted(rejections.items())),
        "marginals": {
            n: {"realised": marg[n], "declared": declared.get(n)} for n in sorted(marg)
        },
        "correlation": {
            "nodes": list(corr.get("nodes") or []),
            "declared": corr.get("matrix"),
            "realised": realised_correlation(
                base_states, list(corr.get("nodes") or [])
            ),
        },
        "near_duplicates": {
            "structured_edges": clustering.n_structured_edges,
            "text_edges": clustering.n_text_edges,
            "n_clusters": n_clusters,
            "largest_cluster": clustering.largest,
            "chaining_limit": max(10, math.ceil(0.02 * n_cases)),
        },
        "template_concentration": concentration,
        "stratum_coverage": strata,
        "direction_balance": direction_share,
        "non_triviality": {"varying_relevant_fields": varying},
        "arms": arms,
        "control_subset": {
            "subset_n": subset_n,
            "computed": computed,
            "n_clusters_with_controls": len(subset),
        },
        "power": {
            "primary": {
                "contrast": primary_req.contrast,
                "n_required": primary_req.n_required,
                "superiority": primary_req.superiority,
                "equivalence": primary_req.equivalence,
                "level": primary_req.level,
                "variance": primary_req.variance.variance,
                "source": primary_req.variance.source,
            },
            "controls": [
                {
                    "contrast": c.contrast,
                    "n_required": c.n_required,
                    "variance": c.variance.variance,
                    "source": c.variance.source,
                }
                for c in ctrl_reqs
            ],
            "n_clusters": n_clusters,
        },
    }
    return Corpus(
        design_arms=arms,
        cases=cases,
        renderings=renderings,
        report=report,
        n_clusters=n_clusters,
        subset_n=subset_n,
        primary_requirement=primary_req,
        control_requirements=ctrl_reqs,
        bank_files=[],
        template_logicals=[templates.base_logical],
        paraphrase_logicals=templates.paraphrase_logicals,
        contract_logical=contract_logical,
        clause_logicals=clause_logicals,
        file_data=file_data,
        world=world,
        family=family,
    )


def _attempt(
    design: Mapping[str, Any],
    world: World,
    ispec: InterventionSpec | None,
    arms: list[str],
    rng: np.random.Generator,
    case_idx: int,
    templates: Templates,
    contract_text: str,
    mandate_text: str,
    pc_mandate: str,
    leak_terms: Sequence[str],
    policy: str,
    paraphrase_seed: int,
    cue_hook: CueHook | None,
    pcb_note: str = "",
    blocks_fn: Callable[..., dict[str, list[tuple[str, str | None]]]] | None = None,
) -> dict[str, Any]:
    base = draw_state(world, rng)
    bad = violated_constraints(world, base)
    if bad:
        raise CaseRejected("constraint", f"sampled state: {bad}")
    controls = design["controls"]
    if ispec is not None:
        sa, sb, flipped = apply_intervention(world, ispec, base, case_idx)
        states: dict[str, dict[str, Any]] = {"A": sa, "B": sb}
    else:
        sa, flipped = dict(base), False
        states = {"A": sa}
    if "A0" in arms:
        states["A0"] = dict(sa)
    if "PL" in arms:
        pl = controls["placebo"]
        states["PL"] = placebo_state(world, sa, pl["node"], pl["values"], rng)
    if "PCA" in arms:
        pos = controls["positive"]
        states["PCA"] = set_node(
            world, sa, pos["node"], pos.get("via"), pos["baseline"]
        )
        states["PCB"] = set_node(
            world, sa, pos["node"], pos.get("via"), pos["counterfactual"]
        )
    for arm, s in states.items():
        bad = violated_constraints(world, s)
        if bad:
            raise CaseRejected("constraint", f"arm {arm}: {bad}")
    diff: list[str] = []
    if ispec is not None:
        d = diff_nodes(world, states["A"], states["B"])
        check_state_isolation(d, declared_diff(world, ispec), "B vs A")
        diff = sorted(d)
    order = R.field_order(templates.base, policy, paraphrase_seed, case_idx)
    arm_ctx: dict[str, ArmRender] = {}
    for arm in states:
        arm_ctx[arm] = ArmRender(
            mandate=pc_mandate if arm in ("PCA", "PCB") else mandate_text,
            field_order=list(order),
            note=pcb_note if arm == "PCB" else "",
            blocks=blocks_fn(states[arm], world.nodes) if blocks_fn else {},
        )
    cue_info: dict[str, Any] | None = None
    if cue_hook is not None:
        for arm in list(states):
            override = cue_hook(case_idx, arm, states[arm], rng, list(order))
            if override is not None:
                arm_ctx[arm] = override
        cue_info = getattr(cue_hook, "case_info", lambda _i: None)(case_idx)
    messages: dict[str, list[tuple[str, str]]] = {}
    rendered: dict[str, list[R.Rendered]] = {}
    for arm, s in states.items():
        ac = arm_ctx[arm]
        rendered[arm] = []
        messages[arm] = []
        for para in templates.paraphrases:
            ctx = R.RenderContext(
                state=s,
                nodes=world.nodes,
                mandate=ac.mandate,
                contract=contract_text,
                preamble=ac.preamble,
                field_order=ac.field_order,
                display_overrides=ac.display_overrides,
                precision_overrides=ac.precision_overrides,
                blocks=ac.blocks,
                note=ac.note,
            )
            r = R.render(templates.base, para, ctx)
            check_round_trip(
                r, world.nodes, s, ac.display_overrides, ac.precision_overrides
            )
            check_neutrality(r, leak_terms)
            rendered[arm].append(r)
            messages[arm].append((r.system, r.user))
    for p_idx in range(len(templates.paraphrases)):
        if "B" in rendered:
            check_block_isolation(arm_ctx["A"], arm_ctx["B"], set(diff), "B vs A")
            check_text_isolation(
                rendered["A"][p_idx],
                rendered["B"][p_idx],
                set(diff) | _cue_nodes(arm_ctx),
                _cue_kinds(arm_ctx),
            )
        if "A0" in rendered and messages["A0"][p_idx] != messages["A"][p_idx]:
            raise CaseRejected("isolation", "A0 renders differently from A")
        if "PL" in rendered:
            dpl = diff_nodes(world, states["A"], states["PL"])
            check_block_isolation(arm_ctx["A"], arm_ctx["PL"], dpl, "PL vs A")
            check_text_isolation(
                rendered["A"][p_idx],
                rendered["PL"][p_idx],
                dpl,
                _block_kind(arm_ctx["A"], arm_ctx["PL"]),
            )
        if "PCA" in rendered:
            dpa = diff_nodes(world, states["A"], states["PCA"])
            check_block_isolation(arm_ctx["A"], arm_ctx["PCA"], dpa, "PCA vs A")
            check_text_isolation(
                rendered["A"][p_idx],
                rendered["PCA"][p_idx],
                dpa,
                ("mandate", *_block_kind(arm_ctx["A"], arm_ctx["PCA"])),
            )
            dpc = diff_nodes(world, states["PCA"], states["PCB"])
            check_block_isolation(arm_ctx["PCA"], arm_ctx["PCB"], dpc, "PCB vs PCA")
            check_text_isolation(
                rendered["PCA"][p_idx],
                rendered["PCB"][p_idx],
                dpc,
                ("note", *_block_kind(arm_ctx["PCA"], arm_ctx["PCB"])),
            )
    a0 = rendered["A"][0]
    displayed = {s.node or s.slot: a0.user[s.start : s.end] for s in a0.state_spans()}
    slot_text = " ".join(
        a0.user[s.start : s.end] for s in a0.spans if s.kind in ("state", "block")
    )
    truth = None
    oracle = design["oracle_spec"]
    if oracle.get("truth_fn"):
        fn = world.fns(str(oracle["truth_fn"]))
        truth = {"d_star": float(fn(states["A"], states.get("B", states["A"])))}
    return {
        "case_idx": case_idx,
        "arms": states,
        "counterbalanced": bool(flipped),
        "diff": diff,
        "truth": truth,
        "cue": cue_info,
        "field_order": list(order),
        "sampled": base,
        "messages": messages,
        "_displayed": displayed,
        "_slot_text": slot_text,
    }


def check_block_isolation(
    x: ArmRender, y: ArmRender, diff: set[str], label: str
) -> None:
    """Two arms' blocks may differ only by order or by removed items, and only with a state diff.

    Per block, the multisets of items are equal (a reorder: POSITION_CHANGE or a
    placebo swap) or one contains the other (a removal: REMOVE). No item may be
    added or reworded, so the block carries no other change (§9.1, §35 S2).
    """
    for name in sorted(set(x.blocks) | set(y.blocks)):
        a, b = x.blocks.get(name, []), y.blocks.get(name, [])
        if a == b:
            continue
        if not diff:
            raise CaseRejected(
                "isolation", f"{label}: block {name!r} differs with no state diff"
            )
        # items compare by content: a separator newline moves with an item's position
        ca = Counter((t.strip(), n) for t, n in a)
        cb = Counter((t.strip(), n) for t, n in b)
        subset = all(cb[k] <= ca[k] for k in cb) or all(ca[k] <= cb[k] for k in ca)
        if not subset:
            raise CaseRejected(
                "isolation", f"{label}: block {name!r} items are added or reworded"
            )


def _block_kind(x: ArmRender, y: ArmRender) -> tuple[str, ...]:
    return ("block",) if x.blocks != y.blocks else ()


def _cue_nodes(arm_ctx: Mapping[str, ArmRender]) -> set[str]:
    out: set[str] = set()
    for a in ("A", "B"):
        if a in arm_ctx:
            out |= set(arm_ctx[a].display_overrides)
    return out


def _cue_kinds(arm_ctx: Mapping[str, ArmRender]) -> set[str]:
    kinds: set[str] = set()
    a, b = arm_ctx.get("A"), arm_ctx.get("B")
    if a and b:
        if a.preamble != b.preamble:
            kinds.add("preamble")
        if a.field_order != b.field_order:
            kinds.add("state")
        if a.blocks != b.blocks:
            kinds.add("block")
    return kinds


def _check_non_triviality(
    design: Mapping[str, Any],
    world: World,
    accepted: Sequence[Mapping[str, Any]],
    rendered_set: set[str],
    ispec: InterventionSpec | None,
) -> list[str]:
    """At least two other decision-relevant rendered fields vary across cases (§9.1)."""
    diff = set(accepted[0]["diff"]) if accepted else set()
    varying = []
    for name, nd in world.nodes.items():
        if not nd.relevant or name not in rendered_set or name in diff:
            continue
        vals = {str(c["arms"]["A"][name]) for c in accepted}
        if len(vals) > 1:
            varying.append(name)
    if len(varying) < 2:
        raise HarnessValidationError(
            f"non-triviality: only {varying} vary among the decision-relevant rendered fields (§9.1)"
        )
    oracle = design["oracle_spec"]
    if oracle["type"] != "LEVEL" and oracle.get("truth_fn"):
        truths = {
            round(float(c["truth"]["d_star"]), 12) for c in accepted if c.get("truth")
        }
        if len(truths) <= 1 and ispec is not None and ispec.form == "set":
            raise HarnessValidationError(
                "non-triviality: the per-case truth is a function of the manipulated field alone (§9.1)"
            )
    return sorted(varying)


def _stratum_coverage(
    sc: Mapping[str, Any], accepted: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    strata = sc.get("strata") or {}
    out: dict[str, Any] = {}
    for node, n_min in strata.items():
        counts = Counter(str(c["sampled"][node]) for c in accepted)
        short = {k: v for k, v in counts.items() if v < int(n_min)}
        if short:
            raise HarnessValidationError(
                f"stratum coverage: {node} levels {short} below n_min {n_min} (§9.2)"
            )
        out[node] = dict(sorted(counts.items()))
    return out


def corpus_logicals(corpus: Corpus) -> list[str]:
    return [rel_to_package(corpus.family.directory)]
