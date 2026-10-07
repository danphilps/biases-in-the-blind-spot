"""Code sandbox (spec §13.1, §13.4, §28, E-22, E-29).

The backend is the Docker or Podman CLI, called with an argv list and
``shell=False``; the sandbox fails closed. For each execution the runner builds
two fresh host directories: ``case_ro/`` (``starter/`` holding only the allowed
file, replaced by the response's code, and ``data/``) and ``tests_hidden/`` (a
copy of the case's hidden-test tree). Both are mounted read-only; ``/work`` and
``/tmp`` are size-capped tmpfs mounts. The hidden-test tree hash is checked
against ``assets.json`` before the container starts and again after it exits.

Status mapping (§28), from the container's per-test outcomes:
- SUCCESS: pytest exit code 0 and every hidden test PASSED;
- TEST_FAILURE: any other run with exit code 0 or 1, or a timeout (the
  model's code ran too long; ``timeout: true`` is recorded);
- COMPILE_FAILURE: every hidden test ERRORs and pytest reports a
  ``SyntaxError``, ``IndentationError`` or ``TabError`` (the solution does not
  compile under the image's Python);
- RUNTIME_FAILURE: any hidden test ERRORs (the solution module cannot be
  imported), or the exit code is neither 0 nor 1 (e.g. 137 after the memory
  limit);
- SANDBOX_FAILURE (infrastructure, missing): no usable backend, an image that
  cannot be built, a hidden-test hash mismatch, the backend's own error (exit
  125 with no output) or a failed copy into ``/work``.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np

from financial_llm_harness.core.canonical_json import tree_sha256
from financial_llm_harness.core.code_ref import package_dir
from financial_llm_harness.errors import HarnessError
from financial_llm_harness.evaluation.parse import MAX_CODE_BYTES

RECIPES = ("base", "torch")
BACKENDS = ("docker", "podman")
TIMEOUT_S = 60.0
OUTPUT_CAP = 1_048_576
TMPFS_BYTES = 67_108_864
BUILD_TIMEOUT_S = 1800.0
SCRIPT = (
    "cp -R /case_ro/starter/. /case_ro/data /work && cd /work && "
    "exec python -m pytest -q -rA -p no:cacheprovider /tests_hidden"
)
_SUMMARY_HEADER = "short test summary info"
_SUMMARY_LINE = re.compile(r"^(PASSED|FAILED|ERROR|XPASS|XFAIL|SKIPPED) (\S+)")
_COMPILE_ERROR = re.compile(r"^E\s+(SyntaxError|IndentationError|TabError):", re.M)
_RANK = {"PASSED": 0, "ABSENT": 1, "FAILED": 2, "ERROR": 3}


class SandboxError(HarnessError):
    """An infrastructure failure of the sandbox (status SANDBOX_FAILURE)."""


# ---------------------------------------------------------------------------
# Images (§13.4, E-22)
# ---------------------------------------------------------------------------


def recipe_dir(name: str) -> Path:
    if name not in RECIPES:
        raise SandboxError(f"unknown sandbox image {name!r}")
    return package_dir() / "families" / "code_task" / "sandbox" / name


def recipe_logical(name: str) -> str:
    return f"families/code_task/sandbox/{name}"


def recipe_sha256(name: str) -> str:
    """The §0.10 tree hash of the recipe directory (its ``sandbox_recipe`` row)."""
    return tree_sha256(recipe_dir(name))


def image_tag(name: str) -> str:
    return f"financial-harness-sandbox-{name}:{recipe_sha256(name)[:12]}"


def find_backend() -> str | None:
    """The first Docker or Podman CLI on ``PATH`` whose engine answers, else None."""
    for cli in BACKENDS:
        path = shutil.which(cli)
        if path is None:
            continue
        try:
            r = subprocess.run(
                [path, "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=30,
                shell=False,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            return path
    return None


def image_id(backend: str, tag: str) -> str | None:
    try:
        r = subprocess.run(
            [backend, "image", "inspect", "--format", "{{.Id}}", tag],
            capture_output=True,
            text=True,
            timeout=60,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = r.stdout.strip()
    return (out or None) if r.returncode == 0 else None


def build_image(name: str, backend: str) -> str:
    """Build one recipe and tag it ``financial-harness-sandbox-<name>:<recipe_sha256[:12]>``."""
    recipe = recipe_dir(name)
    for required in ("Dockerfile", "requirements.in", "requirements.txt"):
        if not (recipe / required).is_file():
            raise SandboxError(
                f"sandbox image {name}: {recipe_logical(name)}/{required} is missing"
            )
    tag = image_tag(name)
    try:
        r = subprocess.run(
            [backend, "build", "--tag", tag, str(recipe)],
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_S,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SandboxError(f"sandbox image {name}: build timed out") from exc
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:])
        raise SandboxError(f"sandbox image {name}: build failed\n{tail}")
    return tag


def ensure_image(name: str, backend: str) -> str:
    """The local image ID for a recipe, building the image when it is missing."""
    tag = image_tag(name)
    ident = image_id(backend, tag)
    if ident is None:
        build_image(name, backend)
        ident = image_id(backend, tag)
    if ident is None:
        raise SandboxError(f"sandbox image {name}: no image ID after the build")
    return ident


def build_images(names: Sequence[str] = RECIPES) -> list[str]:
    """``financial-harness sandbox build``: build every recipe; one line per image."""
    backend = find_backend()
    if backend is None:
        raise SandboxError("no usable Docker or Podman backend (§13.4, D-6)")
    lines: list[str] = []
    errors: list[str] = []
    for name in names:
        try:
            build_image(name, backend)
            lines.append(f"{image_tag(name)} {image_id(backend, image_tag(name))}")
        except SandboxError as exc:
            errors.append(str(exc))
    if errors:
        raise SandboxError("\n".join(lines + errors))
    return lines


# ---------------------------------------------------------------------------
# Datasets (§12)
# ---------------------------------------------------------------------------


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """A deterministic ``.npz``: fixed member order, timestamps and compression."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(
                member, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, member.getvalue(), compresslevel=6)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# One execution (§13.1, §13.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeCase:
    """Everything one sandbox execution needs."""

    code: str
    allowed_file: str
    data: Mapping[str, bytes]
    tests_hidden: Path
    tests_hidden_sha256: str
    image: str
    tests: Sequence[str]


@dataclass(frozen=True)
class Proc:
    exit_code: int | None
    stdout: bytes  # the first OUTPUT_CAP bytes: what is stored (§13.4)
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool
    stdout_tail: bytes = (
        b""  # the last OUTPUT_CAP bytes when truncated: parsed, never stored
    )

    def parse_text(self) -> str:
        """The text outcomes are read from, so truncation never changes a status (§13.4)."""
        text = self.stdout + (
            b"\n" + self.stdout_tail if self.stdout_truncated else b""
        )
        return text.decode("utf-8", "replace")


@dataclass(frozen=True)
class SandboxResult:
    status: str
    tests: dict[str, str] = field(default_factory=dict)
    timeout: bool = False
    exit_code: int | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    image_id: str | None = None
    failure: str | None = None
    hash_check: str | None = None

    @property
    def ran(self) -> bool:
        return self.status not in ("SANDBOX_FAILURE", "PARSE_FAILURE")


Spawn = Callable[[list[str], float], Proc]


def sandbox_argv(
    backend: str, name: str, case_ro: Path, tests_hidden: Path, image: str
) -> list[str]:
    """The §13.4 argv: one element per token; the script is one constant element."""
    return [
        backend,
        "run",
        "--rm",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--mount",
        f"type=tmpfs,destination=/work,tmpfs-size={TMPFS_BYTES},tmpfs-mode=1777",
        "--mount",
        f"type=tmpfs,destination=/tmp,tmpfs-size={TMPFS_BYTES},tmpfs-mode=1777",
        "--mount",
        f"type=bind,source={case_ro},destination=/case_ro,readonly",
        "--mount",
        f"type=bind,source={tests_hidden},destination=/tests_hidden,readonly",
        "--memory=512m",
        "--memory-swap=512m",
        "--cpus=1",
        "--pids-limit=64",
        "--ulimit",
        "nofile=256",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "HOME=/work",
        "-e",
        "PYTHONPATH=/work",
        image,
        "sh",
        "-c",
        SCRIPT,
    ]


def _drain(
    stream: IO[bytes], sink: list[bytes], tail: bytearray, flags: list[bool]
) -> None:
    """Keep the first ``OUTPUT_CAP`` bytes (stored) and the last ``OUTPUT_CAP`` (parsed only)."""
    kept = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        if kept < OUTPUT_CAP:
            part = chunk[: OUTPUT_CAP - kept]
            sink.append(part)
            kept += len(part)
            if len(part) < len(chunk):
                flags[0] = True
        else:
            flags[0] = True
        tail += chunk
        if len(tail) > OUTPUT_CAP:
            del tail[: len(tail) - OUTPUT_CAP]


def spawn(argv: list[str], timeout: float) -> Proc:
    """Run the backend CLI; stdout and stderr are read through a 1 MiB cap each."""
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    assert proc.stdout is not None and proc.stderr is not None
    out: list[bytes] = []
    err: list[bytes] = []
    out_tail = bytearray()
    err_tail = bytearray()
    out_flag = [False]
    err_flag = [False]
    readers = [
        threading.Thread(
            target=_drain, args=(proc.stdout, out, out_tail, out_flag), daemon=True
        ),
        threading.Thread(
            target=_drain, args=(proc.stderr, err, err_tail, err_flag), daemon=True
        ),
    ]
    for t in readers:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    if timed_out:
        _stop_container(argv)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    for t in readers:
        t.join(timeout=30)
    return Proc(
        None if timed_out else proc.returncode,
        b"".join(out),
        b"".join(err),
        out_flag[0],
        err_flag[0],
        timed_out,
        bytes(out_tail) if out_flag[0] else b"",
    )


def _stop_container(argv: Sequence[str]) -> None:
    """On timeout: ``kill`` then ``rm -f`` the named container (§13.4)."""
    backend = argv[0]
    name = argv[argv.index("--name") + 1]
    for cmd in ([backend, "kill", name], [backend, "rm", "-f", name]):
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
                shell=False,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


def parse_outcomes(stdout: str, tests: Sequence[str]) -> dict[str, str]:
    """Per-test outcomes from the ``-rA`` short test summary (§13.4).

    Only lines after the last summary header count; a test absent from it is
    ``ABSENT`` (failed). Where a test appears twice (e.g. a teardown error), the
    worse outcome wins, so printed text can never turn a failure into a pass.
    """
    out = dict.fromkeys(tests, "ABSENT")
    head = stdout.rfind(_SUMMARY_HEADER)
    if head < 0:
        return out
    for line in stdout[head:].splitlines()[1:]:
        m = _SUMMARY_LINE.match(line.strip())
        if m is None:
            continue
        name = m.group(2).rsplit("::", 1)[-1]
        if name not in out:
            continue
        outcome = m.group(1) if m.group(1) in _RANK else "FAILED"
        prev = out[name]
        if prev == "ABSENT" or _RANK[outcome] > _RANK[prev]:
            out[name] = outcome
    return out


def classify(proc: Proc, tests: Mapping[str, str]) -> tuple[str, str | None]:
    """(status, SANDBOX_FAILURE reason or None) for one finished execution."""
    if proc.timed_out:
        return "TEST_FAILURE", None
    stdout = proc.parse_text()
    stderr = proc.stderr.decode("utf-8", "replace")
    if proc.exit_code == 125 and not stdout.strip():
        return "SANDBOX_FAILURE", "backend_error"
    if _SUMMARY_HEADER not in stdout and stderr.lstrip().startswith("cp:"):
        return "SANDBOX_FAILURE", "setup_failed"
    outcomes = list(tests.values())
    if proc.exit_code == 0 and outcomes and all(o == "PASSED" for o in outcomes):
        return "SUCCESS", None
    if (
        outcomes
        and all(o == "ERROR" for o in outcomes)
        and _COMPILE_ERROR.search(stdout)
    ):
        return "COMPILE_FAILURE", None
    if any(o == "ERROR" for o in outcomes) or proc.exit_code not in (0, 1):
        return "RUNTIME_FAILURE", None
    return "TEST_FAILURE", None


def _make_readable(root: Path) -> None:
    """The container runs as 65534:65534, so the mounted trees must be world-readable."""
    root.chmod(0o755)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


def run_case(
    case: CodeCase,
    attempt_id: str,
    *,
    backend: str | None,
    image: str | None = None,
    runner: Spawn | None = None,
    timeout: float = TIMEOUT_S,
) -> SandboxResult:
    """Execute one response's code against the hidden tests (§13.4).

    ``image`` is the local image ID, resolved by the caller with
    ``ensure_image``; ``runner`` defaults to ``spawn`` (injectable for tests).
    """
    if len(case.code.encode("utf-8")) > MAX_CODE_BYTES:
        return SandboxResult("PARSE_FAILURE", failure="code block over 262,144 bytes")
    if backend is None:
        return SandboxResult("SANDBOX_FAILURE", failure="no_backend")
    if image is None:
        try:
            image = ensure_image(case.image, backend)
        except SandboxError as exc:
            return SandboxResult("SANDBOX_FAILURE", failure=f"image_build: {exc}")
    with tempfile.TemporaryDirectory(prefix="harness-sandbox-") as tmp:
        root = Path(tmp)
        case_ro = root / "case_ro"
        (case_ro / "starter").mkdir(parents=True)
        (case_ro / "data").mkdir()
        (case_ro / "starter" / case.allowed_file).write_text(
            case.code, encoding="utf-8"
        )
        for name, payload in sorted(case.data.items()):
            (case_ro / "data" / name).write_bytes(payload)
        hidden = root / "tests_hidden"
        shutil.copytree(
            case.tests_hidden, hidden, ignore=shutil.ignore_patterns("__pycache__")
        )
        _make_readable(case_ro)
        _make_readable(hidden)
        if tree_sha256(hidden) != case.tests_hidden_sha256:
            return SandboxResult(
                "SANDBOX_FAILURE",
                image_id=image,
                failure="hidden_tests_hash",
                hash_check="before",
            )
        name = f"harness-{attempt_id[:12]}"
        argv = sandbox_argv(backend, name, case_ro, hidden, image)
        try:
            proc = (runner or spawn)(argv, timeout)
        except OSError as exc:
            return SandboxResult(
                "SANDBOX_FAILURE", image_id=image, failure=f"backend_error: {exc}"
            )
        if tree_sha256(hidden) != case.tests_hidden_sha256:
            return SandboxResult(
                "SANDBOX_FAILURE",
                exit_code=proc.exit_code,
                stdout=proc.stdout,
                stderr=proc.stderr,
                stdout_truncated=proc.stdout_truncated,
                stderr_truncated=proc.stderr_truncated,
                image_id=image,
                failure="hidden_tests_hash",
                hash_check="after",
            )
    tests = parse_outcomes(proc.parse_text(), case.tests)
    status, reason = classify(proc, tests)
    return SandboxResult(
        status,
        tests=tests,
        timeout=proc.timed_out,
        exit_code=proc.exit_code,
        stdout=proc.stdout,
        stderr=proc.stderr,
        stdout_truncated=proc.stdout_truncated,
        stderr_truncated=proc.stderr_truncated,
        image_id=image,
        failure=reason,
    )


def default_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def result_payload(result: SandboxResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "tests": dict(result.tests),
        "timeout": result.timeout,
        "exit_code": result.exit_code,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
        "image_id": result.image_id,
        "failure": result.failure,
    }


# ---------------------------------------------------------------------------
# L8 for code tasks (§28, §31A)
# ---------------------------------------------------------------------------


def executed_trials(db: Any) -> set[str]:
    rows = db.execute(
        "SELECT payload_json FROM event WHERE kind = 'code_outcomes'"
    ).fetchall()
    return {json.loads(r[0])["trial_id"] for r in rows}


def execute_code_trials(ctx: Any, workers: int | None = None) -> None:
    """Run each parsed, not yet executed code response in the sandbox (§13.4).

    A trial is pending when its selected response parsed (status SUCCESS) and no
    ``code_outcomes`` event exists for it, or when an earlier execution ended
    SANDBOX_FAILURE. Results are written in trial order, so the database does
    not depend on the worker count. Re-evaluation (replay) never calls this:
    it reads the stored ``code_outcomes`` events.
    """
    from financial_llm_harness.core import bundle as B
    from financial_llm_harness.evaluation.code_static import static_check
    from financial_llm_harness.evaluation.parse import parse_code_response
    from financial_llm_harness.families.code_task.family import case_data, load_fixture

    manifest = ctx.manifest
    fx = load_fixture(manifest["task"])
    done = executed_trials(ctx.db)
    rows = [
        dict(r)
        for r in ctx.db.execute(
            "SELECT t.trial_id, t.case_idx, t.arm, t.used_attempt_id, r.text_artifact "
            "FROM trial t JOIN response r ON r.attempt_id = t.used_attempt_id "
            "WHERE t.status IN ('SUCCESS', 'SANDBOX_FAILURE') ORDER BY t.case_idx, t.trial_id"
        ).fetchall()
    ]
    pending = [r for r in rows if r["trial_id"] not in done]
    ctx.db.commit()
    if not pending:
        return
    hidden_rows = [
        a for a in B.load_assets(ctx.bundle_dir) if a["role"] == "hidden_tests"
    ]
    if len(hidden_rows) != 1:
        raise SandboxError("assets.json needs exactly one hidden_tests row (§0.10)")
    expected = hidden_rows[0]["sha256"]
    backend = find_backend()
    image: str | None = None
    failure: str | None = None
    if backend is None:
        failure = "no_backend"
    else:
        try:
            image = ensure_image(fx.sandbox_image, backend)
        except SandboxError as exc:
            failure = f"image_build: {exc}"
    n_workers = workers or default_workers()
    by_case: dict[int, list[dict[str, Any]]] = {}
    for r in pending:
        by_case.setdefault(int(r["case_idx"]), []).append(r)
    for case_idx in sorted(by_case):
        jobs: list[
            tuple[dict[str, Any], CodeCase | None, dict[str, Any], str | None]
        ] = []
        data_sha: dict[tuple[Any, ...], tuple[str, dict[str, bytes]]] = {}
        for r in by_case[case_idx]:
            text = ctx.store.get_text(r["text_artifact"])
            parsed = parse_code_response(text)
            if parsed.status != "SUCCESS" or parsed.code is None:
                continue
            state = ctx.cases[case_idx]["arms"][r["arm"]]
            key = tuple(state[n] for n in (*fx.dataset_nodes, fx.burn_in_node))
            if key not in data_sha:
                data = case_data(manifest, case_idx, state, fx)
                data_sha[key] = (
                    ctx.store.put(data[Path(fx.leakage_fixture).name]),
                    data,
                )
            sha, data = data_sha[key]
            case = CodeCase(
                code=parsed.code,
                allowed_file=fx.allowed_file,
                data=data,
                tests_hidden=fx.tests_hidden,
                tests_hidden_sha256=expected,
                image=fx.sandbox_image,
                tests=list(fx.checks.values()),
            )
            jobs.append((r, case, static_check(parsed.code).to_dict(), sha))
        results: list[SandboxResult]
        if failure is not None:
            results = [SandboxResult("SANDBOX_FAILURE", failure=failure) for _ in jobs]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(
                    pool.map(
                        lambda job: run_case(
                            job[1],
                            job[0]["used_attempt_id"],
                            backend=backend,
                            image=image,
                        ),
                        jobs,
                    )
                )
        for (r, _case, static, sha), res in zip(jobs, results, strict=True):
            _record(ctx, fx, r, res, static, sha)
        ctx.db.commit()


def _record(
    ctx: Any,
    fx: Any,
    row: Mapping[str, Any],
    res: SandboxResult,
    static: Mapping[str, Any],
    dataset_sha: str | None,
) -> None:
    trial_id = row["trial_id"]
    if res.hash_check is not None:
        ctx.db.event(
            "hidden_tests_hash_mismatch",
            {"trial_id": trial_id, "check": res.hash_check},
        )
    if res.status == "SANDBOX_FAILURE":
        ctx.db.event(
            "sandbox_failure",
            {"trial_id": trial_id, "failure": res.failure, "image_id": res.image_id},
        )
        ctx.db.execute(
            "UPDATE trial SET status = 'SANDBOX_FAILURE', sandbox_image_id = ? WHERE trial_id = ?",
            (res.image_id, trial_id),
        )
        return
    outcomes = {
        check: res.tests.get(test) == "PASSED" for check, test in fx.checks.items()
    }
    payload = result_payload(res) | {
        "trial_id": trial_id,
        "attempt_id": row["used_attempt_id"],
        "outcomes": outcomes,
        "stdout_artifact": ctx.store.put(res.stdout),
        "stderr_artifact": ctx.store.put(res.stderr),
        "dataset_artifact": dataset_sha,
        "static": dict(static),
    }
    ctx.db.event("code_outcomes", payload)
    ctx.db.execute(
        "UPDATE trial SET status = ?, sandbox_image_id = ? WHERE trial_id = ?",
        (res.status, res.image_id, trial_id),
    )
