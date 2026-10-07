"""Deterministic identifiers and integer seeds (spec §3, §29).

Every ID is ``uuid5(HARNESS_NS, key)`` with a ``/``-joined key. Only Block,
Trial and the IDs built from a trial ID depend on ``manifest_hash``. None of
these functions may be called before freeze (hash acyclicity, §4).
"""

from __future__ import annotations

import hashlib
import uuid

HARNESS_NS = uuid.uuid5(uuid.NAMESPACE_URL, "financial-llm-harness")


def uid(key: str) -> str:
    """UUIDv5 of a canonical key under the harness namespace."""
    return str(uuid.uuid5(HARNESS_NS, key))


def _join(args: tuple[object, ...]) -> bytes:
    return "|".join(str(a) for a in args).encode("utf-8")


def uint31(*args: object) -> int:
    """First four sha256 bytes of the ``|``-joined args, big-endian, masked to 31 bits."""
    digest = hashlib.sha256(_join(args)).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def uint64(*args: object) -> int:
    """First eight sha256 bytes of the ``|``-joined args, big-endian."""
    digest = hashlib.sha256(_join(args)).digest()
    return int.from_bytes(digest[:8], "big")


def experiment_id(design_hash: str) -> str:
    return uid(f"experiment/{design_hash}")


def testcase_id(design_hash: str, case_idx: int) -> str:
    return uid(f"{design_hash}/case/{case_idx}")


def arm_variant_id(testcase_id_: str, arm: str) -> str:
    return uid(f"{testcase_id_}/{arm}")


def rendering_id(arm_variant_id_: str, paraphrase_idx: int) -> str:
    return uid(f"{arm_variant_id_}/{paraphrase_idx}")


def cluster_id(design_hash: str, cluster_ordinal: int) -> str:
    return uid(f"{design_hash}/cluster/{cluster_ordinal}")


def block_id(
    manifest_hash: str,
    case_idx: int,
    paraphrase_idx: int,
    repetition: int,
    model_key: str,
) -> str:
    return uid(f"{manifest_hash}/{case_idx}/{paraphrase_idx}/{repetition}/{model_key}")


def trial_id(
    manifest_hash: str,
    case_idx: int,
    arm: str,
    paraphrase_idx: int,
    repetition: int,
    model_key: str,
) -> str:
    return uid(
        f"{manifest_hash}/{case_idx}/{arm}/{paraphrase_idx}/{repetition}/{model_key}"
    )


def attempt_id(trial_id_: str, attempt_no: int) -> str:
    return uid(f"{trial_id_}/{attempt_no}")


def observation_id(trial_id_: str, feature_id: str, evaluator_version: str) -> str:
    return uid(f"{trial_id_}/{feature_id}/{evaluator_version}")


def eval_call_id(observation_id_: str, call_no: int, attempt_no: int) -> str:
    return uid(f"{observation_id_}/{call_no}/{attempt_no}")


def analysis_id(
    manifest_hash: str,
    model_key: str,
    feature_id: str,
    contrast: str,
    analysis_kind: str,
    analysis_version: int,
) -> str:
    return uid(
        f"{manifest_hash}/{model_key}/{feature_id}/{contrast}/{analysis_kind}/{analysis_version}"
    )


def row_id(*natural_key: object) -> str:
    """UUIDv5 of a results-file row's natural key (§32A.2)."""
    return uid("row/" + "/".join(str(k) for k in natural_key))
