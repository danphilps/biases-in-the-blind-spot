"""Run archives: writer lock, WAL checkpoint, deterministic tar, safe extraction (spec §0.9).

``archive`` takes the exclusive ``db/LOCK``, commits, checkpoints the WAL with
``TRUNCATE``, verifies ``integrity_check`` and writes a byte-deterministic
``run_archive.tar.gz`` of ``db/run.sqlite`` and ``artifacts/``. Extraction
validates every member before anything is written.
"""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import io
import os
import re
import shutil
import sqlite3
import tarfile
from pathlib import Path
from types import TracebackType

from financial_llm_harness.errors import HarnessError, LockHeld

ARCHIVE_NAME = "run_archive.tar.gz"
ARCHIVE_SHA_NAME = "ARCHIVE.sha256"
MAX_MEMBER_BYTES = 1 << 30  # 1 GiB
MAX_TOTAL_BYTES = 64 << 30  # 64 GiB
MAX_MEMBERS = 5_000_000
_TMP_RE = re.compile(r"\.tmp\d*$")


class ArchiveError(HarnessError):
    """Archive creation or extraction refused (exit 1)."""


class WriterLock:
    """Exclusive, non-blocking ``fcntl.flock`` on ``<run_dir>/db/LOCK``."""

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "db" / "LOCK"
        self._fh: io.BufferedRandom | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")  # noqa: SIM115 - held for the lock's lifetime
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise LockHeld(f"another process holds {self.path}") from exc
        self._fh = fh  # type: ignore[assignment]

    def release(self) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None

    def __enter__(self) -> WriterLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def checkpoint_and_verify(db_file: Path) -> None:
    """Steps 2–5 of §0.9: commit, WAL checkpoint (TRUNCATE), integrity check."""
    conn = sqlite3.connect(str(db_file), isolation_level=None)
    try:
        if conn.in_transaction:
            conn.execute("COMMIT")
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if row is None or row[0] != 0:
            raise ArchiveError(f"WAL checkpoint busy for {db_file}")
    finally:
        conn.close()
    conn = sqlite3.connect(str(db_file), isolation_level=None)
    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
    finally:
        conn.close()
    if [tuple(r) for r in rows] != [("ok",)]:
        raise ArchiveError(f"integrity_check failed for {db_file}: {rows}")
    wal = db_file.with_name(db_file.name + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise ArchiveError(f"{wal} is not empty after checkpoint")


def _archive_members(run_dir: Path) -> list[tuple[str, Path]]:
    members: dict[str, Path] = {}
    db_file = run_dir / "db" / "run.sqlite"
    members["db"] = run_dir / "db"
    members["db/run.sqlite"] = db_file
    art = run_dir / "artifacts"
    if art.exists():
        members["artifacts"] = art
        for path in art.rglob("*"):
            if path.is_symlink():
                continue
            rel = path.relative_to(run_dir).as_posix()
            if path.is_file() and _TMP_RE.search(path.name):
                continue
            if path.is_dir() or path.is_file():
                members[rel] = path
    return sorted(members.items())


def deterministic_tar_gz(members: list[tuple[str, Path]]) -> bytes:
    """Regular files and directories, sorted, mtime 0, uid/gid 0, fixed modes, gzip mtime 0."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name, path in members:
            info = tarfile.TarInfo(name)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tar.addfile(info)
            else:
                data = path.read_bytes()
                info.type = tarfile.REGTYPE
                info.mode = 0o644
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    out = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def archive_run(run_dir: Path) -> str:
    """Write ``run_archive.tar.gz`` and ``ARCHIVE.sha256``; return the archive sha256."""
    db_file = run_dir / "db" / "run.sqlite"
    if not db_file.exists():
        raise ArchiveError(f"no database at {db_file}")
    with WriterLock(run_dir):
        checkpoint_and_verify(db_file)
        data = deterministic_tar_gz(_archive_members(run_dir))
    sha = hashlib.sha256(data).hexdigest()
    _atomic_write(run_dir / ARCHIVE_NAME, data)
    _atomic_write(run_dir / ARCHIVE_SHA_NAME, f"{sha}  {ARCHIVE_NAME}\n".encode())
    return sha


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _validate_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    seen: set[str] = set()
    total = 0
    members = tar.getmembers()
    if len(members) > MAX_MEMBERS:
        raise ArchiveError("archive has too many members")
    for m in members:
        name = m.name
        if name.startswith("/") or os.path.isabs(name):
            raise ArchiveError(f"absolute member {name!r}")
        parts = Path(name).parts
        if ".." in parts or name.split("/").count("..") > 0:
            raise ArchiveError(f"member with '..': {name!r}")
        if not parts or parts[0] not in ("db", "artifacts"):
            raise ArchiveError(f"member outside db/ and artifacts/: {name!r}")
        if not (m.isreg() or m.isdir()):
            raise ArchiveError(f"member {name!r} is not a regular file or directory")
        norm = "/".join(parts)
        if norm in seen:
            raise ArchiveError(f"duplicate member {name!r}")
        seen.add(norm)
        if m.isreg():
            if m.size > MAX_MEMBER_BYTES:
                raise ArchiveError(f"member {name!r} exceeds the size limit")
            total += m.size
            if total > MAX_TOTAL_BYTES:
                raise ArchiveError("archive exceeds the total size limit")
    return members


def verify_archive_hash(run_dir: Path) -> Path:
    archive = run_dir / ARCHIVE_NAME
    sha_file = run_dir / ARCHIVE_SHA_NAME
    if not archive.exists() or not sha_file.exists():
        raise ArchiveError(f"no archive in {run_dir}")
    expected = sha_file.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise ArchiveError(f"archive hash mismatch: {actual} != {expected}")
    return archive


def extract_archive(run_dir: Path, target_dir: Path) -> None:
    """Verify, validate every member, then extract ``db/`` and ``artifacts/`` into ``target_dir``."""
    archive = verify_archive_hash(run_dir)
    try:
        tar = tarfile.open(archive, mode="r:gz")
    except (tarfile.TarError, OSError) as exc:
        raise ArchiveError(f"corrupted archive: {exc}") from exc
    with tar:
        try:
            members = _validate_members(tar)
        except tarfile.TarError as exc:
            raise ArchiveError(f"corrupted archive: {exc}") from exc
        for m in members:
            dest = target_dir.joinpath(*Path(m.name).parts)
            if m.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            src = tar.extractfile(m)
            if src is None:
                raise ArchiveError(f"cannot read member {m.name!r}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                shutil.copyfileobj(src, fh)
