"""Shared helpers for the harness tests (imported as ``tests.harness.helpers``)."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.code_ref import CodeRef

ROOT = Path(__file__).resolve().parents[2]
S1_SPEC = ROOT / "experiments" / "financial" / "specs" / "momentum_stock_add.yaml"
S1_M2_SPEC = ROOT / "experiments" / "financial" / "specs" / "momentum_stock_add_m2.yaml"
GOLDEN_RUN_ID = "00000000-0000-4000-8000-000000000001"
FAKE_A = "fake--a--b89ac294"
FAKE_B = "fake--b--fc598cd9"
FIXED_CODE_REF = CodeRef("0000000", "0" * 64, None, None, False)


def fixed_clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def load_spec(path: Path = S1_SPEC) -> dict[str, Any]:
    data = yaml_io.load(path)
    assert isinstance(data, dict)
    return data


def fake_planning_refs(variance: float = 0.05) -> list[dict[str, Any]]:
    """Fake-design planning variances (§22): run_id null, analytic derivation stated, fake models only."""
    refs = []
    for mk in (FAKE_A, FAKE_B):
        for contrast in ("B_vs_A", "A0_vs_A", "PL_vs_A", "PCB_vs_PCA"):
            refs.append(
                {
                    "run_id": None,
                    "model_key": mk,
                    "feature_id": "decision_add",
                    "contrast": contrast,
                    "variance": variance,
                    "source_row_id": None,
                    "source_sha256": None,
                    "derivation": "test-only bound for the fake mechanism",
                }
            )
    return refs


def small_spec(
    tmp: Path,
    n_cases: int = 50,
    name: str = "small.yaml",
    base: Path = S1_SPEC,
    **overrides: Any,
) -> Path:
    """A small S1-shaped spec planned with fake-design variances, for fast lifecycle tests."""
    spec = copy.deepcopy(load_spec(base))
    spec["scenario_spec"]["n_cases"] = n_cases
    spec["statistical_spec"]["planning_ref"] = fake_planning_refs()
    spec["statistical_spec"]["bootstrap_B"] = 500
    for key, value in overrides.items():
        cur: Any = spec
        parts = key.split("__")
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = value
    path = tmp / name
    path.write_text(yaml_io.dump(spec), encoding="utf-8")
    return path


THINK_DEFAULT = {"mode": "provider_default", "effort": None, "budget_tokens": None}


def local_entry(
    model: str = "local-model", family: str = "local_family", **kw: Any
) -> dict[str, Any]:
    """A non-fake (self-hosted) panel entry with its derived key."""
    from financial_llm_harness.models.panels import derive_model_key

    e: dict[str, Any] = {
        "provider": "local",
        "family": family,
        "model": model,
        "region": None,
        "transport": "native",
        "thinking": dict(THINK_DEFAULT),
        "request_profile": None,
        "batch": False,
        "upstream_routing": None,
        "params": {},
    }
    e.update(kw)
    e = {"model_key": derive_model_key(e), **e}
    return e


def write_probe(
    home: Path, entry: dict[str, Any], date: str = "2026-01-01", **caps: Any
) -> Path:
    """A probe file in the §20.4 layout, for tests that need a non-fake model."""
    import json

    from financial_llm_harness.models.base import profile_sha256, resolve_profile_id

    pid = resolve_profile_id(
        entry["provider"], entry["transport"], entry.get("request_profile")
    )
    capabilities = {
        "provider": entry["provider"],
        "model": entry["model"],
        "region": entry.get("region"),
        "transport": entry["transport"],
        "serving": "self_hosted",
        "revision": None,
        "request_profile_id": pid,
        "request_profile_sha256": profile_sha256(entry["provider"], pid),
        "context_tokens": 128000,
        "max_output_tokens": 8192,
        "temperature_range": [0.0, 2.0],
        "seed_parameter": "accepted",
        "seed_reproducibility_rate": 1.0,
        "supports_system": True,
        "supports_schema": True,
        "thinking_effective": "not_controllable",
        "trace_visibility": "none",
        "stream_required": False,
        "upstream_routing": None,
        "probe_serving_fingerprint": None,
        "param_support": [
            ["max_output_tokens", "accepted"],
            ["seed", "accepted"],
            ["temperature", "accepted"],
        ],
    }
    capabilities.update(caps)
    d = home / "probes" / entry["model_key"]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{date}.json"
    payload = {
        "schema_version": 1,
        "model_key": entry["model_key"],
        "date": date,
        "request_profile_id": pid,
        "request_profile_sha256": profile_sha256(entry["provider"], pid),
        "capabilities": capabilities,
        "checks": {},
        "calls": 0,
        "cost_usd": 0.0,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


_CI_FAKE_EVALUATOR = object()


def ci_fake_evaluator() -> dict[str, Any]:
    data = yaml_io.load(ROOT / "experiments" / "financial" / "panels" / "ci_fake.yaml")
    return dict(data["evaluator"])


def write_panel(
    tmp: Path,
    models: list[dict[str, Any]],
    evaluator: Any = _CI_FAKE_EVALUATOR,
    name: str = "panel.yaml",
) -> Path:
    """A panel file; the evaluator defaults to ci_fake's fake evaluator, which the S1
    specs' semantic reasoning channel needs (§16); pass None for a panel without one."""
    if evaluator is _CI_FAKE_EVALUATOR:
        evaluator = ci_fake_evaluator()
    path = tmp / name
    path.write_text(
        yaml_io.dump({"panel_version": 1, "models": models, "evaluator": evaluator}),
        encoding="utf-8",
    )
    return path


def ci_fake_entries() -> list[dict[str, Any]]:
    data = yaml_io.load(ROOT / "experiments" / "financial" / "panels" / "ci_fake.yaml")
    return list(data["models"])


OUTPUT_PARTS = ("README.md", "results", "report", "models")


def output_files(run_dir: Path) -> dict[str, bytes]:
    """Every file that must be byte-identical across renders, resumes and replays (§32A.8)."""
    out: dict[str, bytes] = {}
    for part in OUTPUT_PARTS:
        p = run_dir / part
        if p.is_file():
            out[part] = p.read_bytes()
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    out[f.relative_to(run_dir).as_posix()] = f.read_bytes()
    return out


def assert_same_files(a: dict[str, bytes], b: dict[str, bytes]) -> None:
    assert sorted(a) == sorted(b)
    differing = [k for k in a if a[k] != b[k]]
    assert differing == [], f"files differ: {differing}"


def run_small(
    tmp: Path,
    run_id: str = GOLDEN_RUN_ID,
    *,
    inject: bool = True,
    name: str = "small.yaml",
    **overrides: Any,
) -> Path:
    """A COMPLETE small fake run in the current HARNESS_HOME; returns its run directory."""
    from financial_llm_harness.execution.runner import run_experiment

    spec = small_spec(tmp, name=name, **overrides)
    if inject:
        res = run_experiment(
            spec,
            "ci_fake",
            yes=True,
            run_id=run_id,
            clock=fixed_clock,
            code_ref=FIXED_CODE_REF,
        )
    else:
        res = run_experiment(spec, "ci_fake", yes=True, run_id=run_id)
    assert res.status == "COMPLETE" and res.run_dir is not None
    return res.run_dir


# ---------------------------------------------------------------------------
# Cassettes (§20.10, E-5): replayed through httpx.MockTransport, no network
# ---------------------------------------------------------------------------

CASSETTES = Path(__file__).resolve().parent / "cassettes"


class Cassette:
    """Replays ``tests/harness/cassettes/<dir>/<case>.json`` entries in order.

    A request matches on method, URL path and canonical JSON body; an unmatched
    request is recorded and answered with HTTP 599 so the test fails visibly.
    """

    def __init__(self, directory: str, *cases: str) -> None:
        self.entries: list[dict[str, Any]] = []
        for case in cases:  # no case: an empty cassette (see from_entries)
            self.entries += json.loads(
                (self.dir(directory) / f"{case}.json").read_text()
            )
        self.used = [False] * len(self.entries)
        self.requests: list[Any] = []
        self.response_bytes: list[bytes] = []
        self.unmatched: list[str] = []

    @staticmethod
    def dir(directory: str) -> Path:
        return CASSETTES / directory

    @classmethod
    def from_entries(cls, entries: list[dict[str, Any]]) -> Cassette:
        """A cassette built in a test from hand-written entries in the same format."""
        c = cls("__none__")
        c.entries = list(entries)
        c.used = [False] * len(c.entries)
        return c

    def handler(self, request: Any) -> Any:
        from urllib.parse import urlparse

        import httpx

        from financial_llm_harness.core.canonical_json import canonical

        body = json.loads(request.content) if request.content else None
        for i, e in enumerate(self.entries):
            if self.used[i]:
                continue
            r = e["request"]
            if (
                r["method"] == request.method
                and urlparse(r["url"]).path == request.url.path
                and canonical(r["json"]) == canonical(body)
            ):
                self.used[i] = True
                self.requests.append(request)
                resp = e["response"]
                data = (
                    resp["body"].encode()
                    if isinstance(resp["body"], str)
                    else json.dumps(resp["body"]).encode()
                )
                self.response_bytes.append(data)
                return httpx.Response(
                    resp["status"], headers=resp["headers"], content=data
                )
        self.unmatched.append(f"{request.method} {request.url.path} {body}")
        return httpx.Response(599, content=b"unmatched cassette request")

    def client(self) -> Any:
        import httpx

        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def assert_exhausted(self) -> None:
        assert self.unmatched == [], f"unmatched requests: {self.unmatched}"
        assert all(self.used), "cassette entries not requested"


def run_slice(spec: Path, home: Path, run_id: str = GOLDEN_RUN_ID) -> Path:
    """A canonical slice spec at its §35 size on ci_fake, with the E-23 injected identity."""
    import os

    from financial_llm_harness.execution.runner import run_experiment

    old = os.environ.get("HARNESS_HOME")
    os.environ["HARNESS_HOME"] = str(home)
    try:
        res = run_experiment(
            spec,
            "ci_fake",
            yes=True,
            run_id=run_id,
            clock=fixed_clock,
            code_ref=FIXED_CODE_REF,
        )
    finally:
        if old is None:
            os.environ.pop("HARNESS_HOME", None)
        else:
            os.environ["HARNESS_HOME"] = old
    assert res.status == "COMPLETE" and res.run_dir is not None, res.status
    return res.run_dir


def read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def assert_level_slice(
    run_dir: Path, feature: str, tau: float, n_cases: int, subset_n: int
) -> None:
    """§36 Slices for S3–S5: arm_level primary on B with a LEVEL label and an empty zone,
    B − A a DISCOVERY secondary, and the three controls passing (§35, D-21)."""
    manifest = yaml_io.load(run_dir / "manifest" / "manifest.yaml")
    assert isinstance(manifest, dict)
    h = manifest["hypothesis"]
    assert (h["estimand"], h["level_arm"], h["primary_feature"]) == (
        "arm_level",
        "B",
        feature,
    )
    assert manifest["oracle_spec"]["type"] == "LEVEL"
    assert float(manifest["oracle_spec"]["tau"]) == tau
    assert manifest["statistical_spec"]["planning_ref"] == []
    assert manifest["scenario_spec"]["n_cases"] == n_cases
    assert manifest["controls"]["subset_n"] == subset_n
    labels = read_csv(run_dir / "results" / "labels.csv")
    for row in labels:
        assert row["feature_id"] == feature
        assert row["oracle_type"] == "LEVEL"
        assert row["zone"] == ""
        assert row["sensitivity_label"] in ("MEETS_STANDARD", "BELOW_STANDARD")
    effects = read_csv(run_dir / "results" / "effects.csv")
    prim = [e for e in effects if e["analysis_role"] == "primary"]
    assert {e["contrast"] for e in prim} == {"level_B"}
    assert all(e["estimand"] == "arm_level" for e in prim)
    sec = [
        e
        for e in effects
        if e["analysis_role"] == "secondary" and e["feature_id"] == feature
    ]
    assert {e["contrast"] for e in sec} == {"B_vs_A"} and len(sec) == len(prim)
    assert all(e["correction_family"] == "within_experiment" for e in sec)
    controls = read_csv(run_dir / "results" / "controls.csv")
    assert {c["control"] for c in controls} == {"null", "placebo", "positive"}
    assert all(c["passed"] == "true" for c in controls)
    pos = [c for c in controls if c["control"] == "positive"]
    assert all(c["expected"] == "JUSTIFIED_SENSITIVITY" for c in pos)
    assert all(float(c["theta"]) > 0 for c in pos)  # DIRECTIONAL(+) (§35)


def pcb_note_only_in_pcb(run_dir: Path, sentence: str) -> None:
    from financial_llm_harness.core import bundle as B

    rows = B.read_jsonl(run_dir / "manifest" / B.RENDERINGS_FILE)
    with_note = {r["arm"] for r in rows if sentence in r["user"]}
    assert with_note == {"PCB"}
    assert all(sentence in r["user"] for r in rows if r["arm"] == "PCB")
