"""Content-addressed artefact store (spec §29).

Artefacts live at ``<run_dir>/artifacts/sha256/<2>/<62>`` and are written
tmp → fsync → rename; a DB row referencing one is inserted only after it exists.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


class ArtifactStore:
    """Content-addressed store rooted at a run's ``artifacts/`` directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, sha: str) -> Path:
        return self.root / "sha256" / sha[:2] / sha[2:]

    def put(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        path = self.path_for(sha)
        if path.exists():
            return sha
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return sha

    def put_text(self, text: str) -> str:
        return self.put(text.encode("utf-8"))

    def get(self, sha: str) -> bytes:
        return self.path_for(sha).read_bytes()

    def get_text(self, sha: str) -> str:
        return self.get(sha).decode("utf-8")

    def exists(self, sha: str) -> bool:
        return self.path_for(sha).exists()
