"""Unit: hash and ID stability; Adapters: model keys (spec §3, §4, §0.10, §20.6)."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from financial_llm_harness.core import code_ref as code_ref_mod
from financial_llm_harness.core import ids, yaml_io
from financial_llm_harness.core.canonical_json import (
    canonical,
    sha256_canonical,
    tree_sha256,
)
from financial_llm_harness.core.hashes import (
    DESIGN_KEYS,
    compute_design_hash,
    compute_manifest_hash,
    design_projection,
)
from financial_llm_harness.errors import HarnessValidationError
from financial_llm_harness.models.panels import derive_model_key, load_panel

ROOT = Path(__file__).resolve().parents[2]


def test_canonical_json_form() -> None:
    assert (
        canonical({"b": 1, "a": [1.5, "é", None, True]})
        == '{"a":[1.5,"é",null,true],"b":1}'.encode()
    )
    assert canonical(0.1) == b"0.1"
    with pytest.raises(HarnessValidationError):
        canonical(float("nan"))
    with pytest.raises(HarnessValidationError):
        canonical({"x": float("inf")})


def test_uint31_uint64() -> None:
    digest = hashlib.sha256(b"abc|1|2").digest()
    assert ids.uint31("abc", 1, 2) == int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    assert ids.uint64("abc", 1, 2) == int.from_bytes(digest[:8], "big")
    assert 0 <= ids.uint31("x") < 2**31


def test_uuid5_keys() -> None:
    ns = uuid.uuid5(uuid.NAMESPACE_URL, "financial-llm-harness")
    assert ids.HARNESS_NS == ns
    assert ids.experiment_id("d") == str(uuid.uuid5(ns, "experiment/d"))
    tc = ids.testcase_id("d", 3)
    assert tc == str(uuid.uuid5(ns, "d/case/3"))
    av = ids.arm_variant_id(tc, "B")
    assert av == str(uuid.uuid5(ns, f"{tc}/B"))
    assert ids.rendering_id(av, 2) == str(uuid.uuid5(ns, f"{av}/2"))
    assert ids.cluster_id("d", 7) == str(uuid.uuid5(ns, "d/cluster/7"))
    assert ids.block_id("m", 1, 2, 0, "k") == str(uuid.uuid5(ns, "m/1/2/0/k"))
    t = ids.trial_id("m", 1, "A", 2, 0, "k")
    assert t == str(uuid.uuid5(ns, "m/1/A/2/0/k"))
    a = ids.attempt_id(t, 1)
    assert a == str(uuid.uuid5(ns, f"{t}/1"))
    o = ids.observation_id(t, "f", "v1")
    assert o == str(uuid.uuid5(ns, f"{t}/f/v1"))
    assert ids.eval_call_id(o, 0, 1) == str(uuid.uuid5(ns, f"{o}/0/1"))
    assert ids.analysis_id("m", "k", "f", "B_vs_A", "sensitivity", 1) == str(
        uuid.uuid5(ns, "m/k/f/B_vs_A/sensitivity/1")
    )


def _manifest() -> dict:
    m: dict[str, Any] = {k: {"v": k} for k in DESIGN_KEYS}
    m["response_spec"] = {
        "contract_id": "c",
        "enforcement": "auto",
        "enforcement_resolved": "prompt_only",
    }
    m["sampling_spec"] = {
        "R": 1,
        "panel_ref": {"name": "p"},
        "models": [{"model_key": "x"}],
    }
    m["corpus"] = {
        "cases": {"path": "cases.jsonl", "sha256": "c1"},
        "renderings": {"path": "renderings.jsonl", "sha256": "r1"},
        "assets": {"path": "assets.json", "sha256": "a1"},
    }
    m["data_classification"] = "synthetic"
    m["evaluators"] = []
    m["independence"] = {}
    m["budget"] = {"max_calls": 10}
    m["code_ref"] = {"harness_tree_sha256": "0" * 64}
    m["design_hash"] = "dh"
    m["experiment_id"] = "eid"
    m["created_at"] = "2026-01-01T00:00:00Z"
    m["manifest_hash"] = "mh"
    return m


ASSETS = [
    {"role": "template", "logical_name": "t.yaml", "sha256": "s1", "source": "package"},
    {"role": "panel", "logical_name": "p.yaml", "sha256": "s2", "source": "repository"},
    {
        "role": "request_profile",
        "logical_name": "models/providers.yaml#fake/in_process",
        "sha256": "s3",
        "source": "package",
    },
]


def test_design_projection_exact_keys() -> None:
    proj = design_projection(_manifest(), ASSETS)
    expected = set(DESIGN_KEYS) | {"corpus", "scientific_assets"}
    assert set(proj) == expected
    assert "enforcement_resolved" not in proj["response_spec"]
    assert (
        "panel_ref" not in proj["sampling_spec"]
        and "models" not in proj["sampling_spec"]
    )
    assert proj["corpus"] == {"cases": {"sha256": "c1"}, "renderings": {"sha256": "r1"}}
    assert proj["scientific_assets"] == [
        {"role": "template", "logical_name": "t.yaml", "sha256": "s1"}
    ]


def test_design_hash_ignores_envelope_and_manifest_hash_covers_it() -> None:
    m = _manifest()
    dh = compute_design_hash(m, ASSETS)
    mh = compute_manifest_hash(m)
    for key, value in (
        ("budget", {"max_calls": 11}),
        ("code_ref", {"harness_tree_sha256": "1" * 64}),
        ("evaluators", [{"feature_id": "x"}]),
        ("data_classification", "licensed"),
    ):
        m2 = _manifest()
        m2[key] = value
        assert compute_design_hash(m2, ASSETS) == dh
        assert compute_manifest_hash(m2) != mh
    m2 = _manifest()
    m2["response_spec"]["enforcement_resolved"] = "provider_schema"
    assert compute_design_hash(m2, ASSETS) == dh
    assert compute_manifest_hash(m2) != mh
    m2 = _manifest()
    m2["sampling_spec"]["models"] = [{"model_key": "y"}]
    assert compute_design_hash(m2, ASSETS) == dh
    # non-scientific asset rows never enter design_hash
    assets2 = [dict(a) for a in ASSETS]
    assets2[1]["sha256"] = "changed"
    assert compute_design_hash(m, assets2) == dh
    assets2[0]["sha256"] = "changed"
    assert compute_design_hash(m, assets2) != dh
    m2 = _manifest()
    m2["seeds"] = {"v": "other"}
    assert compute_design_hash(m2, ASSETS) != dh
    # derived values never enter their own hash
    m2 = _manifest()
    m2["manifest_hash"] = "zz"
    m2["experiment_id"] = "zz"
    m2["created_at"] = "zz"
    assert compute_manifest_hash(m2) == mh
    m2["design_hash"] = "other"
    assert compute_manifest_hash(m2) != mh
    assert compute_design_hash(m2, ASSETS) == dh


def test_tree_hash_rule(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 2\n")
    expected = sha256_canonical(
        [
            ["a.py", hashlib.sha256(b"x = 1\n").hexdigest()],
            ["sub/b.py", hashlib.sha256(b"y = 2\n").hexdigest()],
        ]
    )
    assert tree_sha256(tmp_path) == expected
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.pyc").write_bytes(b"\0")
    assert tree_sha256(tmp_path) == expected


def test_harness_tree_sha256_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "pkg"
    (pkg / "reporting").mkdir(parents=True)
    (pkg / "families" / "code_task" / "fixtures" / "t").mkdir(parents=True)
    (pkg / "mod.py").write_text("a = 1\n")
    (pkg / "reporting" / "style.css").write_text("body{}\n")
    (pkg / "families" / "code_task" / "fixtures" / "t" / "bad.py").write_text("x\n")
    (pkg / "data.yaml").write_text("k: v\n")
    monkeypatch.setattr(
        code_ref_mod, "package_dir", lambda name="financial_llm_harness": pkg
    )
    h0 = code_ref_mod.harness_tree_sha256()
    (pkg / "data.yaml").write_text("k: w\n")  # outside the tree rule
    (pkg / "families" / "code_task" / "fixtures" / "t" / "bad.py").write_text("y\n")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\1")
    assert code_ref_mod.harness_tree_sha256() == h0
    (pkg / "mod.py").write_text("a = 2\n")
    h1 = code_ref_mod.harness_tree_sha256()
    assert h1 != h0
    (pkg / "reporting" / "style.css").write_text("body{color:red}\n")
    assert code_ref_mod.harness_tree_sha256() != h1


def test_code_ref_outside_git_is_unidentifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "m.py").write_text("")
    monkeypatch.setattr(
        code_ref_mod, "package_dir", lambda name="financial_llm_harness": pkg
    )
    ref = code_ref_mod.compute_code_ref(uses_legacy=False)
    assert ref.harness_git_sha is None
    assert ref.dirty is True
    assert ref.legacy_tree_sha256 is None and ref.legacy_git_sha is None


# ---------------------------------------------------------------------------
# Model keys (§20.6)
# ---------------------------------------------------------------------------

THINK = {"mode": "provider_default", "effort": None, "budget_tokens": None}


def _entry(**kw: object) -> dict:
    e = {
        "provider": "deepseek",
        "model": "deepseek-reasoner",
        "region": None,
        "transport": "native",
        "thinking": dict(THINK),
        "upstream_routing": None,
    }
    e.update(kw)
    return e


def test_model_key_derivation() -> None:
    e = _entry()
    key = derive_model_key(e)
    ident = {
        k: e[k]
        for k in (
            "provider",
            "model",
            "region",
            "transport",
            "thinking",
            "upstream_routing",
        )
    }
    suffix = hashlib.sha256(
        json.dumps(ident, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    assert key == f"deepseek--deepseek-reasoner--{suffix}"
    qwen = derive_model_key(
        _entry(
            provider="qwen",
            model="Qwen/Max:1",
            region="cn-hongkong",
            thinking={"mode": "on", "effort": None, "budget_tokens": None},
        )
    )
    assert qwen.startswith("qwen--qwen_max_1--cn-hongkong--think-on--")
    g = derive_model_key(
        _entry(
            provider="google",
            model="m",
            thinking={"mode": "on", "effort": "low", "budget_tokens": None},
        )
    )
    assert g.startswith("google--m--think-low--")
    off = derive_model_key(
        _entry(thinking={"mode": "off", "effort": None, "budget_tokens": None})
    )
    assert "--think-off--" in off
    b = derive_model_key(
        _entry(thinking={"mode": "on", "effort": None, "budget_tokens": 2048})
    )
    assert "--think-b2048--" in b


def test_model_key_limits_and_collisions() -> None:
    long_key = derive_model_key(_entry(model="x" * 200))
    prefix, _, suffix = long_key.rpartition("--")
    assert len(long_key) <= 72 and len(prefix) <= 62 and len(suffix) == 8
    assert long_key.startswith("deepseek--xxxx")
    a = derive_model_key(_entry(transport="native"))
    b = derive_model_key(_entry(transport="openai_compat"))
    assert a != b
    c = derive_model_key(
        _entry(
            provider="openrouter",
            upstream_routing={"provider": "x", "quantization": None},
        )
    )
    d = derive_model_key(
        _entry(
            provider="openrouter",
            upstream_routing={"provider": "y", "quantization": None},
        )
    )
    assert c != d
    for k in (a, b, c, d):
        assert "---" not in k


def test_ci_fake_holds_derived_keys() -> None:
    data = yaml_io.load(ROOT / "experiments" / "financial" / "panels" / "ci_fake.yaml")
    for entry in [*data["models"], data["evaluator"]]:
        assert entry["model_key"] == derive_model_key(entry)
    panel = load_panel("ci_fake")
    assert [m["model_key"].rsplit("--", 1)[0] for m in panel.models] == [
        "fake--a",
        "fake--b",
    ]


def test_panel_key_mismatch_exits_2(tmp_path: Path) -> None:
    data = yaml_io.load(ROOT / "experiments" / "financial" / "panels" / "ci_fake.yaml")
    data["models"][0]["model_key"] = "fake--a--00000000"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_io.dump(data))
    with pytest.raises(HarnessValidationError) as exc:
        load_panel(str(path))
    assert exc.value.exit_code == 2
