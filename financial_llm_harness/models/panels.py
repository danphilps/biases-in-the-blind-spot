"""Model panels and model keys (spec §20.6, §20.9).

A panel is the set of target models an experiment or programme runs against.
Model keys are ``<prefix>--<8 hex>``, at most 72 characters, derived from the
canonical identity of the panel entry. Only entries whose provider has a
populated API key are run; ``generate`` records the others as skipped.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from financial_llm_harness.core import yaml_io
from financial_llm_harness.core.canonical_json import sha256_canonical
from financial_llm_harness.core.paths import panels_dir
from financial_llm_harness.core.schema import (
    as_bool,
    as_list,
    as_mapping,
    as_opt_int,
    as_opt_str,
    as_str,
    check_keys,
    fail,
)
from financial_llm_harness.errors import HarnessValidationError, MissingApiKey
from financial_llm_harness.models.base import (
    PROTOCOL_FOR_ADAPTER,
    load_providers,
    provider_entry,
    request_profile,
    resolve_profile_id,
)

ENTRY_KEYS = (
    "model_key",
    "provider",
    "family",
    "model",
    "region",
    "transport",
    "thinking",
    "request_profile",
    "batch",
    "upstream_routing",
    "params",
)
MAX_KEY = 72
MAX_PREFIX = 62
_BAD_CHARS = re.compile(r"[^a-z0-9._-]")
_DASH_RUNS = re.compile(r"-+")


def canonical_identity(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "provider": entry["provider"],
        "model": entry["model"],
        "region": entry.get("region"),
        "transport": entry.get("transport"),
        "thinking": entry.get("thinking"),
        "upstream_routing": entry.get("upstream_routing"),
    }


def _part(text: str) -> str:
    s = _BAD_CHARS.sub("_", str(text).lower())
    s = _DASH_RUNS.sub("-", s)
    return s.strip("-")


def thinking_setting(thinking: Mapping[str, Any] | None) -> str | None:
    """The ``--think-<setting>`` part, or None for ``provider_default``."""
    if thinking is None:
        return None
    mode = thinking.get("mode")
    if mode == "provider_default":
        return None
    if mode == "off":
        return "off"
    if thinking.get("effort") is not None:
        return str(thinking["effort"])
    if thinking.get("budget_tokens") is not None:
        return f"b{thinking['budget_tokens']}"
    return "on"


def derive_model_key(entry: Mapping[str, Any]) -> str:
    """§20.6: ``<provider>--<model>[--<region>][--think-<setting>]--<8 hex>``."""
    suffix = sha256_canonical(canonical_identity(entry))[:8]
    parts = [_part(entry["provider"]), _part(entry["model"])]
    if entry.get("region") is not None:
        parts.append(_part(entry["region"]))
    setting = thinking_setting(entry.get("thinking"))
    if setting is not None:
        parts.append("think-" + _part(setting))
    prefix = "--".join(parts)[:MAX_PREFIX].rstrip("-")
    return f"{prefix}--{suffix}"


# ---------------------------------------------------------------------------
# API-key gating: only providers with a populated key are called
# ---------------------------------------------------------------------------


def key_env_of(provider: str) -> str | None:
    """The provider's ``key_env`` in providers.yaml; None for providers without a key (fake, local)."""
    return provider_entry(provider).get("key_env")


def has_api_key(provider: str) -> bool:
    """True when the provider needs no key, or its key variable is set and not blank."""
    env = key_env_of(provider)
    return env is None or os.environ.get(env, "").strip() != ""


def split_by_api_key(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(entries whose provider has a key, one skip record per entry without one), panel order kept."""
    keyed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for e in entries:
        if has_api_key(e["provider"]):
            keyed.append(e)
        else:
            skipped.append(
                {
                    "model_key": e["model_key"],
                    "provider": e["provider"],
                    "model": e["model"],
                    "key_env": key_env_of(e["provider"]),
                    "reason": "no_api_key",
                }
            )
    return keyed, skipped


def require_api_keys(entries: list[Mapping[str, Any]], what: str) -> None:
    """Raise MissingApiKey naming every entry whose provider key is unset (a frozen model is never dropped)."""
    missing = [
        f"{e['model_key']} ({key_env_of(e['provider'])})"
        for e in entries
        if not has_api_key(e["provider"])
    ]
    if missing:
        raise MissingApiKey(
            f"{what}: no API key for {', '.join(missing)}; set the variable in the environment or .env"
        )


@dataclass(frozen=True)
class Panel:
    name: str
    path: Path
    data: dict[str, Any]

    @property
    def models(self) -> list[dict[str, Any]]:
        return list(self.data.get("models") or [])

    @property
    def evaluator(self) -> dict[str, Any] | None:
        return self.data.get("evaluator")

    def entry(self, model_key: str) -> dict[str, Any]:
        for m in self.models:
            if m["model_key"] == model_key:
                return m
        ev = self.evaluator
        if ev is not None and ev["model_key"] == model_key:
            return ev
        raise HarnessValidationError(
            f"model key {model_key!r} is not in panel {self.name!r}"
        )


def panel_path(name: str) -> Path:
    if "/" in name or name.endswith(".yaml"):
        return Path(name)
    return panels_dir() / f"{name}.yaml"


def load_panel(name: str) -> Panel:
    path = panel_path(name)
    if not path.exists():
        raise HarnessValidationError(f"panel {name!r} not found at {path}")
    data = yaml_io.load(path)
    in_dir = path.resolve().parent == panels_dir().resolve()
    ident = path.stem if in_dir else str(path)
    panel = Panel(ident, path, as_mapping(data, str(path)))  # type: ignore[arg-type]
    validate_panel(panel)
    return panel


def _validate_entry(
    entry: Any, where: str, providers: Mapping[str, Any]
) -> dict[str, Any]:
    e = dict(as_mapping(entry, where))
    check_keys(e, where, required=ENTRY_KEYS)
    key = as_str(e["model_key"], f"{where}.model_key")
    provider = as_str(e["provider"], f"{where}.provider")
    if provider not in providers:
        raise fail(where, f"provider {provider!r} is not in providers.yaml")
    family = as_str(e["family"], f"{where}.family")
    if family.strip() == "":
        raise fail(where, "family is empty")
    as_str(e["model"], f"{where}.model")
    as_opt_str(e["region"], f"{where}.region")
    transport = as_str(e["transport"], f"{where}.transport")
    if transport not in ("native", "openai_compat"):
        raise fail(where, f"transport {transport!r} must be native or openai_compat")
    thinking = as_mapping(e["thinking"], f"{where}.thinking")
    check_keys(
        thinking, f"{where}.thinking", required=("mode", "effort", "budget_tokens")
    )
    if thinking["mode"] not in ("off", "on", "provider_default"):
        raise fail(where, f"thinking.mode {thinking['mode']!r} is invalid")
    if thinking["effort"] not in (None, "low", "medium", "high"):
        raise fail(where, f"thinking.effort {thinking['effort']!r} is invalid")
    as_opt_int(thinking["budget_tokens"], f"{where}.thinking.budget_tokens")
    prov = providers[provider]
    if transport != "native" and transport not in (prov.get("alt_transports") or {}):
        raise fail(where, f"provider {provider!r} has no transport {transport!r}")
    profile_id = resolve_profile_id(
        provider, transport, as_opt_str(e["request_profile"], where)
    )
    profile = request_profile(provider, profile_id)
    adapter = prov["adapter"]
    expected_protocols = PROTOCOL_FOR_ADAPTER[adapter]
    if transport == "openai_compat":
        expected_protocols = ("chat_completions",)
    if profile["protocol"] not in expected_protocols:
        raise fail(
            where,
            f"profile {profile_id!r} protocol {profile['protocol']!r} does not match "
            f"adapter {adapter!r} and transport {transport!r}",
        )
    ur = e["upstream_routing"]
    if provider == "openrouter":
        if ur is None or not isinstance(ur, Mapping) or not ur.get("provider"):
            raise fail(where, "an openrouter entry needs upstream_routing.provider")
        if profile.get("routing_fields") is None:
            raise fail(where, f"openrouter profile {profile_id!r} lacks routing_fields")
        if not (profile.get("fingerprint_paths") or {}).get("upstream_provider"):
            raise fail(
                where,
                f"openrouter profile {profile_id!r} lacks fingerprint_paths.upstream_provider",
            )
    elif ur is not None:
        raise fail(where, "upstream_routing is allowed only for openrouter")
    batch = as_bool(e["batch"], f"{where}.batch")
    if batch and prov.get("batch", "none") == "none":
        raise fail(where, f"batch: true but provider {provider!r} has batch: none")
    as_mapping(e["params"], f"{where}.params")
    derived = derive_model_key(e)
    if key != derived:
        raise fail(where, f"model_key {key!r} differs from the derived key {derived!r}")
    return e


def validate_panel(panel: Panel) -> None:
    """Every §20.9 rejection rule (exit 2)."""
    where = str(panel.path)
    check_keys(
        panel.data,
        where,
        required=("panel_version", "models"),
        optional=("evaluator", "provenance"),
    )
    providers = load_providers()["providers"]
    models = as_list(panel.data["models"], f"{where}.models")
    if not models:
        raise fail(where, "panel has no models")
    seen: set[str] = set()
    families: set[str] = set()
    for i, m in enumerate(models):
        e = _validate_entry(m, f"{where}.models[{i}]", providers)
        if e["model_key"] in seen:
            raise fail(where, f"model_key {e['model_key']!r} repeats")
        seen.add(e["model_key"])
        families.add(e["family"])
    ev = panel.data.get("evaluator")
    if ev is not None:
        e = _validate_entry(ev, f"{where}.evaluator", providers)
        if e["family"] in families:
            raise fail(
                where,
                f"evaluator family {e['family']!r} equals a target model's family",
            )
        if e["model_key"] in seen:
            raise fail(where, "evaluator model_key repeats a target's")
