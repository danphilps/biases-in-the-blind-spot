"""Archive (spec §0.9): checkpoint, integrity, deterministic tar and safe extraction."""

from __future__ import annotations

import hashlib
import io
import sqlite3
import tarfile
from pathlib import Path

import pytest

from financial_llm_harness.errors import LockHeld
from financial_llm_harness.storage import archive as arch
from financial_llm_harness.storage.artifacts import ArtifactStore
from financial_llm_harness.storage.db import connect, db_path


def _make_run(run_dir: Path) -> sqlite3.Connection:
    conn = connect(db_path(run_dir))
    conn.execute(
        "INSERT INTO run(run_id, manifest_hash, status, resume_count) VALUES ('r', 'm', 'RUNNING', 0)"
    )
    store = ArtifactStore(run_dir / "artifacts")
    store.put(b"hello")
    store.put(b"world")
    return conn


def test_archive_twice_is_byte_identical(tmp_path: Path) -> None:
    run = tmp_path / "run"
    conn = _make_run(run)
    conn.close()
    sha1 = arch.archive_run(run)
    b1 = (run / arch.ARCHIVE_NAME).read_bytes()
    sha2 = arch.archive_run(run)
    assert sha1 == sha2 == hashlib.sha256(b1).hexdigest()
    assert (run / arch.ARCHIVE_NAME).read_bytes() == b1
    assert (run / arch.ARCHIVE_SHA_NAME).read_text().split()[0] == sha1
    with tarfile.open(run / arch.ARCHIVE_NAME, "r:gz") as tar:
        names = tar.getnames()
        for m in tar.getmembers():
            assert m.mtime == 0 and m.uid == 0 and m.gid == 0
            assert m.uname == "" and m.gname == ""
            assert m.mode == (0o755 if m.isdir() else 0o644)
    assert names == sorted(names)
    assert "db/LOCK" not in names
    assert not any(n.endswith(("-wal", "-shm")) for n in names)


def test_wal_transactions_are_checkpointed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    conn = _make_run(run)
    conn.execute("INSERT INTO event(ts, kind, payload_json) VALUES ('t', 'late', '{}')")
    wal = db_path(run).with_name("run.sqlite-wal")
    assert wal.exists() and wal.stat().st_size > 0
    arch.archive_run(run)  # the writer connection stays open but idle
    out = tmp_path / "out"
    arch.extract_archive(run, out)
    assert not (out / "db" / "run.sqlite-wal").exists()
    c2 = sqlite3.connect(out / "db" / "run.sqlite")
    assert c2.execute("SELECT kind FROM event").fetchall() == [("late",)]
    assert c2.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    c2.close()
    conn.close()


def test_archive_refuses_while_lock_held(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _make_run(run).close()
    with arch.WriterLock(run):
        with pytest.raises(LockHeld) as exc:
            arch.archive_run(run)
    assert exc.value.exit_code == 1


def test_corrupted_archive_exits_1(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _make_run(run).close()
    arch.archive_run(run)
    data = bytearray((run / arch.ARCHIVE_NAME).read_bytes())
    data[len(data) // 2] ^= 0xFF
    (run / arch.ARCHIVE_NAME).write_bytes(bytes(data))
    with pytest.raises(arch.ArchiveError) as exc:
        arch.extract_archive(run, tmp_path / "out")
    assert exc.value.exit_code == 1
    # a missing archive also exits 1
    (run / arch.ARCHIVE_NAME).unlink()
    with pytest.raises(arch.ArchiveError):
        arch.extract_archive(run, tmp_path / "out")


def _write_tar(
    run: Path, build: list[tarfile.TarInfo | tuple[tarfile.TarInfo, bytes]]
) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        for item in build:
            if isinstance(item, tuple):
                info, data = item
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.addfile(item)
    run.mkdir(parents=True, exist_ok=True)
    data = raw.getvalue()
    (run / arch.ARCHIVE_NAME).write_bytes(data)
    (run / arch.ARCHIVE_SHA_NAME).write_text(
        f"{hashlib.sha256(data).hexdigest()}  {arch.ARCHIVE_NAME}\n"
    )


def _file(name: str) -> tuple[tarfile.TarInfo, bytes]:
    return (tarfile.TarInfo(name), b"x")


def _link(name: str, target: str, kind: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = target
    return info


@pytest.mark.parametrize(
    "members",
    [
        [_file("/etc/passwd")],
        [_file("db/../../escape")],
        [_link("db/link", "/etc/passwd", tarfile.SYMTYPE)],
        [_file("db/a"), _link("db/hard", "db/a", tarfile.LNKTYPE)],
        [_link("artifacts/inner", "db/a", tarfile.SYMTYPE)],
        [_file("db/a"), _file("db/a")],
        [_file("logs/x")],
    ],
    ids=[
        "absolute",
        "dotdot",
        "symlink-escape",
        "hardlink",
        "symlink-inside",
        "duplicate",
        "outside",
    ],
)
def test_malicious_members_rejected(tmp_path: Path, members: list) -> None:
    run = tmp_path / "run"
    _write_tar(run, members)
    out = tmp_path / "out"
    with pytest.raises(arch.ArchiveError):
        arch.extract_archive(run, out)
    assert not out.exists()


def test_oversized_member_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(arch, "MAX_MEMBER_BYTES", 4)
    run = tmp_path / "run"
    _write_tar(run, [(tarfile.TarInfo("db/run.sqlite"), b"0123456789")])
    out = tmp_path / "out"
    with pytest.raises(arch.ArchiveError):
        arch.extract_archive(run, out)
    assert not out.exists()
