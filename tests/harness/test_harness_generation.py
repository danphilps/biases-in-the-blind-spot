"""Unit: generation; Property; Corpus structure (spec §6, §9, §22, §36)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.families import BankTracker, NodeDef, load_family
from financial_llm_harness.generation import render as R
from financial_llm_harness.generation.corpus import (
    _check_non_triviality,
    _fns_getter,
    build_corpus,
    load_templates,
)
from financial_llm_harness.generation.generate import generate
from financial_llm_harness.generation.intervention import (
    apply_intervention,
    declared_diff,
    diff_nodes,
    parse_intervention_spec,
)
from financial_llm_harness.generation.scenario import (
    attempt_rng,
    build_world,
    draw_state,
    parse_constraint,
    parse_dist,
    violated_constraints,
)
from financial_llm_harness.generation.validate_case import (
    CaseRejected,
    check_round_trip,
)
from financial_llm_harness.generation.validate_corpus import (
    chaining_guard,
    check_direction_balance,
    cluster_cases,
    structured_edges,
    text_edges,
)
from tests.harness.helpers import load_spec, small_spec


def _world() -> Any:
    spec = load_spec()
    fam = load_family("fundamental_judgement")
    return build_world(
        fam.scenario_schema, spec["scenario_spec"], _fns_getter(fam)
    ), spec


# ---------------------------------------------------------------------------
# Grammars (§6.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "normal(mu=0, sigma=1)",
        "normal(0.5, 2)",
        "lognormal(mu=3.5, sigma=0.6)",
        "uniform(low=0, high=1)",
        "truncnormal(mu=0.08, sigma=0.3, low=-0.55, high=1.4)",
        "beta(a=2, b=5, low=0, high=1)",
        "categorical(values=[a, b, c])",
        "categorical(values=[Industrials, Health Care], probs=[0.25, 0.75])",
        "categorical(values=['x y', \"z\"], probs=[0.5, 0.5])",
        "fixed(3)",
        "fixed(value=abc)",
        "fn(sample_price_12m_ago)",
    ],
)
def test_distribution_grammar_accepts(text: str) -> None:
    d = parse_dist(text)
    assert d.kind == text.split("(")[0]


@pytest.mark.parametrize(
    "text",
    [
        "gamma(a=1, b=2)",
        "normal(mu=__import__('os'), sigma=1)",
        "normal(mu=1)",
        "normal(mu=1, sigma=1, extra=2)",
        "normal(mu=1, sigma=0)",
        "uniform(low=2, high=1)",
        "beta(a=-1, b=1, low=0, high=1)",
        "categorical(values=[])",
        "categorical(values=[a, b], probs=[0.3, 0.3])",
        "categorical(values=[a, a])",
        "normal(mu=0, sigma=1) + 1",
        "fn(os.system)",
        "fn()",
        "lambda: 1",
        "normal(mu=0, mu=1)",
    ],
)
def test_distribution_grammar_rejects(text: str) -> None:
    with pytest.raises(HarnessValidationError) as exc:
        parse_dist(text)
    assert exc.value.exit_code == 2


@pytest.mark.parametrize(
    "text",
    [
        "0 < pe < 60",
        "net_debt_ebitda > -3",
        "x <= 1e3",
        "60 > pe >= 0",
        "-0.6 < m < 1.5",
    ],
)
def test_constraint_grammar_accepts(text: str) -> None:
    parse_constraint(text)


@pytest.mark.parametrize(
    "text",
    [
        "pe < 60 and pe > 0",
        "pe",
        "pe = 3",
        "a < b",
        "0 < pe < 60 < 70",
        "abs(pe) < 3",
        "pe < x + 1",
    ],
)
def test_constraint_grammar_rejects(text: str) -> None:
    with pytest.raises(HarnessValidationError):
        parse_constraint(text)


def test_constraint_semantics() -> None:
    c = parse_constraint("0 < pe < 60")
    assert c.holds(10) and not c.holds(0) and not c.holds(60)
    c2 = parse_constraint("net_debt_ebitda > -3")
    assert c2.holds(-2.9) and not c2.holds(-3)


# ---------------------------------------------------------------------------
# DAG, closure, constraints (§6.2–§6.5)
# ---------------------------------------------------------------------------


def test_dag_recompute_and_closure_diff() -> None:
    world, spec = _world()
    ispec = parse_intervention_spec(spec["intervention_spec"])
    with BankTracker():
        drawn = draw_state(world, attempt_rng("conf", 1, 0))
    base: dict[str, Any] = dict(drawn)
    assert base["momentum_12m"] == pytest.approx(
        base["price_now"] / base["price_12m_ago"] - 1
    )
    assert base["pe"] == pytest.approx(base["price_now"] / base["eps_ttm"])
    ebitda = base["revenue"] * base["ebit_margin"]
    assert base["ev_ebitda"] == pytest.approx(
        (base["market_cap"] + base["net_debt_ebitda"] * ebitda) / ebitda
    )
    a, b, flipped = apply_intervention(world, ispec, drawn, 0)
    assert not flipped
    assert (
        diff_nodes(world, a, b)
        == {"price_12m_ago", "momentum_12m"}
        == declared_diff(world, ispec)
    )
    assert a["momentum_12m"] == pytest.approx(0.25, abs=1e-3)
    assert b["momentum_12m"] == pytest.approx(-0.25, abs=1e-3)
    assert a["pe"] == b["pe"] == base["pe"]


def test_constraints_reported() -> None:
    world, _ = _world()
    with BankTracker():
        s = draw_state(world, attempt_rng("conf", 1, 3))
    s = dict(s)
    s["pe"] = 75.0
    assert "0 < pe < 60" in violated_constraints(world, s)


def test_set_with_counterbalance_exits_2() -> None:
    spec = load_spec()["intervention_spec"]
    bad = dict(spec, counterbalance=True)
    with pytest.raises(HarnessValidationError) as exc:
        parse_intervention_spec(bad)
    assert exc.value.exit_code == 2


def test_closure_declaration_must_match() -> None:
    from financial_llm_harness.generation.intervention import check_closure_declaration

    world, spec = _world()
    bad = copy.deepcopy(spec["intervention_spec"])
    bad["closure"]["recompute"] = []
    with pytest.raises(HarnessValidationError):
        check_closure_declaration(world, parse_intervention_spec(bad))


def test_subset_n_computed_and_bounds(tmp_path: Path) -> None:
    spec = load_spec()
    corpus = build_corpus(spec)
    assert corpus.subset_n == 857
    assert corpus.n_clusters >= 857
    # a smaller spec value exits 2
    small = copy.deepcopy(spec)
    small["controls"]["subset_n"] = 100
    with pytest.raises(HarnessValidationError) as exc:
        build_corpus(small)
    assert exc.value.exit_code == 2
    # computed subset above N_clusters exits 2 at generate
    tiny = copy.deepcopy(spec)
    tiny["scenario_spec"]["n_cases"] = 60
    with pytest.raises(HarnessValidationError, match="exceeds N_clusters"):
        build_corpus(tiny)


def test_corpus_report_marginals(tmp_path: Path) -> None:
    res = generate(small_spec(tmp_path), tmp_path / "bundle")
    rep = json.loads((tmp_path / "bundle" / "corpus_report.json").read_text())
    assert rep["marginals"]["price_now"]["realised"]["mean"] > 0
    assert rep["marginals"]["price_now"]["declared"]["dist"].startswith("lognormal")
    assert rep["marginals"]["sector"]["realised"]["shares"]
    assert rep["correlation"]["realised"] is not None
    assert len(rep["correlation"]["realised"]) == len(rep["correlation"]["nodes"])
    assert rep["acceptance_rate"] >= 0.7
    assert res.corpus.report["near_duplicates"]["n_clusters"] == res.corpus.n_clusters


# ---------------------------------------------------------------------------
# Near-duplicate clustering (§9.2)
# ---------------------------------------------------------------------------


def _nodes(numeric: list[str], categorical: list[str]) -> dict[str, NodeDef]:
    out = {}
    for n in numeric:
        out[n] = NodeDef(
            n, "primitive", "numeric", None, 2, "number", None, (), (), True
        )
    for n in categorical:
        out[n] = NodeDef(
            n, "primitive", "categorical", None, 0, "text", None, (), (), True
        )
    return out


def test_clustering_fixture_structured_and_text_edges() -> None:
    nodes = _nodes(["x", "y", "z", "w", "const"], ["sector"])
    rng = np.random.default_rng(0)
    states = [
        {
            "x": float(v[0]),
            "y": float(v[1]),
            "z": float(v[2]),
            "w": float(v[3]),
            "const": 1.0,
            "sector": "S",
        }
        for v in rng.normal(size=(20, 4))
    ]
    # A=0, B=1, C=2: A–B structured (identical numerics), B–C text-only
    states[1] = dict(states[0])
    states[2] = dict(states[2], sector="T")
    rendered = set(nodes)
    displayed = [{k: str(v) for k, v in s.items()} for s in states]
    s_edges = structured_edges(states, nodes, rendered, displayed)
    assert (0, 1) in s_edges
    assert all(2 not in e for e in s_edges)
    texts = [f"case {i} " + " ".join(f"w{i}_{j}" for j in range(10)) for i in range(20)]
    texts[2] = texts[1]
    t_edges = text_edges(texts)
    assert (1, 2) in t_edges
    cl = cluster_cases(20, s_edges, t_edges)
    assert cl.members[0] == [0, 1, 2]
    assert cl.ordinal_of_case[:4] == [0, 0, 0, 1]
    assert all(m[0] < n[0] for m, n in zip(cl.members, cl.members[1:], strict=False))


def test_zero_variance_field_omitted_from_k() -> None:
    nodes = _nodes(["x", "y", "z", "const"], [])
    states = [
        {"x": float(i), "y": float(i * 2 % 7), "z": float(i % 5), "const": 3.0}
        for i in range(30)
    ]
    states[1] = dict(states[0])
    displayed = [{k: str(v) for k, v in s.items()} for s in states]
    edges = structured_edges(states, nodes, set(nodes), displayed)
    assert (0, 1) in edges  # k = 3 numeric fields (const omitted), distance rule


def test_one_categorical_difference_is_never_a_near_duplicate() -> None:
    nodes = _nodes(["x", "y", "z"], ["sector"])
    states = [
        {"x": 1.0 * i, "y": 2.0 * i, "z": 0.5 * i, "sector": "A"} for i in range(10)
    ]
    states[1] = dict(states[0], sector="B")
    displayed = [{k: str(v) for k, v in s.items()} for s in states]
    assert (0, 1) not in structured_edges(states, nodes, set(nodes), displayed)


def test_fewer_than_three_numeric_fields_need_identical_display() -> None:
    nodes = _nodes(["x", "y"], [])
    states = [{"x": 1.0, "y": 2.0}, {"x": 1.0001, "y": 2.0}, {"x": 1.0, "y": 2.0}]
    displayed = [
        {"x": "1.00", "y": "2.00"},
        {"x": "1.00", "y": "2.00"},
        {"x": "1.00", "y": "2.00"},
    ]
    displayed[1] = {"x": "1.01", "y": "2.00"}
    edges = structured_edges(states, nodes, set(nodes), displayed)
    assert edges == [(0, 2)]


def test_chaining_guard_exits_2_on_dense_field() -> None:
    nodes = _nodes(["x"], [])
    rng = np.random.default_rng(1)
    vals = rng.integers(0, 20, size=500)
    states = [{"x": float(v)} for v in vals]
    displayed = [{"x": str(int(v))} for v in vals]
    cl = cluster_cases(500, structured_edges(states, nodes, set(nodes), displayed), [])
    with pytest.raises(HarnessValidationError, match="chaining guard") as exc:
        chaining_guard(cl, 500)
    assert exc.value.exit_code == 2


def test_s1_generator_passes_chaining_guard() -> None:
    corpus = build_corpus(load_spec())
    assert (
        corpus.report["near_duplicates"]["largest_cluster"]
        <= corpus.report["near_duplicates"]["chaining_limit"]
    )


def test_non_triviality_level_exempt_and_magnitude_fails() -> None:
    world, spec = _world()
    with BankTracker():
        states = [draw_state(world, attempt_rng("conf", 3, k)) for k in range(30)]
    rendered = set(world.nodes)
    accepted = [{"arms": {"A": s}, "diff": [], "truth": None} for s in states]
    level = copy.deepcopy(spec)
    level["oracle_spec"] = dict(spec["oracle_spec"], type="LEVEL", truth_fn=None)
    assert len(_check_non_triviality(level, world, accepted, rendered, None)) >= 2
    mag = copy.deepcopy(spec)
    mag["oracle_spec"] = dict(
        spec["oracle_spec"], type="MAGNITUDE", truth_fn="anything"
    )
    accepted_const = [
        {"arms": {"A": s}, "diff": [], "truth": {"d_star": 0.5}} for s in states
    ]
    ispec = parse_intervention_spec(spec["intervention_spec"])
    with pytest.raises(
        HarnessValidationError, match="function of the manipulated field"
    ):
        _check_non_triviality(mag, world, accepted_const, rendered, ispec)


def test_direction_balance() -> None:
    assert check_direction_balance([True, False] * 20) == 0.5
    with pytest.raises(HarnessValidationError):
        check_direction_balance([True] * 7 + [False] * 3)


# ---------------------------------------------------------------------------
# Property tests (hypothesis)
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(k=st.integers(min_value=0, max_value=10_000))
def test_property_arms_constraints_closure_round_trip(k: int) -> None:
    from financial_llm_harness.generation.corpus import (
        _attempt,
        derive_arms,
        load_design_contract,
    )

    world, spec = _world()
    fam = load_family("fundamental_judgement")
    ispec = parse_intervention_spec(spec["intervention_spec"])
    templates = load_templates(spec, fam)
    contract, _, _ = load_design_contract(spec, fam)
    text = contract.instruction_text("reasoning_before", False)
    arms = derive_arms(spec)
    with BankTracker():
        try:
            row = _attempt(
                spec,
                world,
                ispec,
                arms,
                attempt_rng("conf", 99, k),
                0,
                templates,
                text,
                spec["mandate"]["text"],
                spec["mandate"]["text"] + "; clause",
                spec["prompt_spec"]["leak_terms"],
                "randomised_across_cases",
                5,
                None,
            )
        except CaseRejected:
            return
    for arm, s in row["arms"].items():
        assert violated_constraints(world, s) == [], arm
    assert diff_nodes(world, row["arms"]["A"], row["arms"]["B"]) == declared_diff(
        world, ispec
    )
    for s in row["arms"].values():
        ctx = R.RenderContext(s, world.nodes, "m", text, field_order=row["field_order"])
        r = R.render(templates.base, templates.paraphrases[0], ctx)
        check_round_trip(r, world.nodes, s)


@settings(max_examples=25, deadline=None)
@given(n=st.integers(min_value=20, max_value=400))
def test_property_counterbalance_share(n: int) -> None:
    world, spec = _world()
    shift = {
        "op": "NUMERIC_SHIFT",
        "node": "net_debt_ebitda",
        "via": None,
        "form": "shift",
        "baseline": None,
        "counterfactual": None,
        "delta": 0.5,
        "relative_delta": None,
        "closure": {"recompute": ["ev_ebitda"], "held_fixed": [], "compensate": []},
        "counterbalance": True,
    }
    ispec = parse_intervention_spec(shift)
    flips = []
    with BankTracker():
        base = draw_state(world, attempt_rng("conf", 2, 0))
    for case_idx in range(n):
        a, b, flipped = apply_intervention(world, ispec, base, case_idx)
        assert float(b["net_debt_ebitda"]) > float(
            a["net_debt_ebitda"]
        )  # B always holds the larger value
        flips.append(flipped)
    share = sum(flips) / len(flips)
    assert 0.45 <= share <= 0.55
