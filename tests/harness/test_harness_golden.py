"""Golden; Golden and identity; Reporting: golden snapshot (spec §32A.8, §36, E-23).

``HARNESS_UPDATE_GOLDEN=1 uv run pytest tests/harness/test_harness_golden.py``
rewrites the snapshot; the rewrite is reviewed like any code change.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest

from tests.harness.helpers import (
    FAKE_A,
    FAKE_B,
    FIXED_CODE_REF,
    GOLDEN_RUN_ID,
    output_files,
    run_small,
)

GOLDEN = Path(__file__).resolve().parent / "golden"
# The fixed observations, θ and labels of the canonical fixture travel with the report snapshot.
GOLDEN_RESULTS = ("summary.json", "labels.csv", "effects.csv", "controls.csv")


def _snapshot(run_dir: Path) -> dict[str, bytes]:
    out = {
        p.relative_to(run_dir).as_posix(): p.read_bytes()
        for p in sorted((run_dir / "report").rglob("*"))
        if p.is_file()
    }
    for name in GOLDEN_RESULTS:
        out[f"results/{name}"] = (run_dir / "results" / name).read_bytes()
    return out


def test_golden_report_snapshot(s1_run: Path) -> None:
    current = _snapshot(s1_run)
    if os.environ.get("HARNESS_UPDATE_GOLDEN") == "1":
        if GOLDEN.exists():
            shutil.rmtree(GOLDEN)
        for rel, data in current.items():
            dest = GOLDEN / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    stored = {
        p.relative_to(GOLDEN).as_posix(): p.read_bytes()
        for p in sorted(GOLDEN.rglob("*"))
        if p.is_file()
    }
    assert sorted(stored) == sorted(current)
    differing = [k for k in stored if stored[k] != current[k]]
    assert differing == [], (
        f"golden snapshot differs: {differing}; review, then rewrite with HARNESS_UPDATE_GOLDEN=1"
    )


def test_golden_fixture_gives_fixed_theta_and_labels(s1_run: Path) -> None:
    summary = json.loads((GOLDEN / "results" / "summary.json").read_text())
    assert summary["run"]["run_id"] == GOLDEN_RUN_ID
    assert summary["run"]["started_at"] == "2026-01-01T00:00:00Z"
    assert summary["run"]["code_ref"] == FIXED_CODE_REF.to_dict()
    models = {m["model_key"]: m["primary"] for m in summary["models"]}
    assert models[FAKE_A]["sensitivity_label"] == "JUSTIFIED_SENSITIVITY"
    assert models[FAKE_A]["zone"] == "Z-"
    assert models[FAKE_A]["composite_label"] == "UNVERBALISED_DEPENDENCY"
    assert models[FAKE_B]["sensitivity_label"] == "UNDER_SENSITIVITY"
    assert models[FAKE_B]["is_normative_failure"] is True
    # the live run reproduces the stored observations' θ exactly
    live = json.loads((s1_run / "results" / "summary.json").read_text())
    assert [m["primary"]["theta"] for m in live["models"]] == [
        m["primary"]["theta"] for m in summary["models"]
    ]


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _identity(run_dir: Path) -> dict[str, str]:
    s = json.loads((run_dir / "results" / "summary.json").read_text())
    cr = s["run"]["code_ref"]
    return {
        "run_id": s["run"]["run_id"],
        "manifest_hash": s["run"]["manifest_hash"],
        "design_hash": s["run"]["design_hash"],
        "git": cr["harness_git_sha"] or "",
        "tree": cr["harness_tree_sha256"],
        "dirty": "true" if cr["dirty"] else "false",
    }


_CODE_FIELD = re.compile(
    r"(Harness git SHA|Harness tree|Dirty|harness)(\W*(?:<[^>]+>\W*)*)"
    r"([0-9a-f]{7,64}(?:-dirty)?|True|False|None|unidentified)"
)
_CODE_JSON = re.compile(
    r'"(harness_git_sha|harness_tree_sha256|legacy_git_sha|legacy_tree_sha256|dirty)": [^,\n}]+'
)


def _normalise(text: str, ident: dict[str, str]) -> str:
    """Replace run ID, manifest hash, code identity and dates by placeholders, in context only."""
    text = text.replace(ident["manifest_hash"], "MH").replace(ident["run_id"], "RUN_ID")
    for short, tag in (
        (ident["run_id"][:8], "RUN8"),
        (ident["manifest_hash"][:8], "MH8"),
    ):
        text = re.sub(rf"(`|<code>){short}(`|</code>)", rf"\g<1>{tag}\g<2>", text)
    text = _CODE_FIELD.sub(r"\1\2CODE", text)
    text = _CODE_JSON.sub(r'"\1": CODE', text)
    text = _TS.sub("TS", text)
    return _DATE.sub("DATE", text)


def test_run_without_injection_differs_only_in_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "golden_home"))
    injected = run_small(tmp_path)
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "plain_home"))
    plain = run_small(
        tmp_path, run_id="2f7c1e9a-5b3d-4c8e-9a1f-0123456789ab", inject=False
    )
    ia, ib = _identity(injected), _identity(plain)
    assert ia["design_hash"] == ib["design_hash"]
    assert ia["manifest_hash"] != ib["manifest_hash"]
    assert ia["tree"] == "0" * 64 and ib["tree"] != "0" * 64
    fa, fb = output_files(injected), output_files(plain)
    assert sorted(fa) == sorted(fb)
    # IDs derived from manifest_hash (trial, analysis and row IDs) correspond position by position
    mapping: dict[str, str] = {}
    for rel in fa:
        ua = _UUID.findall(fa[rel].decode())
        ub = _UUID.findall(fb[rel].decode())
        assert len(ua) == len(ub), rel
        for a, b in zip(ua, ub, strict=True):
            if a != b and a != ia["run_id"]:
                assert mapping.setdefault(a, b) == b, (rel, a)
    assert mapping, "manifest-derived IDs should differ"
    differing = []
    for rel in fa:
        ta = fa[rel].decode()
        if mapping:
            ta = _UUID.sub(lambda m: mapping.get(m.group(0), m.group(0)), ta)
        na = _normalise(ta, ia)
        nb = _normalise(fb[rel].decode(), ib)
        if na != nb:
            differing.append(rel)
    assert differing == [], f"outputs differ beyond identity: {differing}"
