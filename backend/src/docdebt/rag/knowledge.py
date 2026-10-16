"""The per-repository knowledge base, stored next to the service state.

One row per chunk:

* ``method``: signature and Javadoc of a documented method *without*
  structural debt (a debt-free Javadoc is a safe example of the project's
  style);
* ``test``: a test method, with the names of the methods it calls;
* ``readme``: one README section;
* ``validated``: a Javadoc a reviewer accepted from a DocDebt proposal.

``method``, ``test`` and ``readme`` chunks are rebuilt by a full scan of the
default branch (``docdebt.service.health``); ``validated`` chunks are kept
across scans and grow with every accepted proposal.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from docdebt.core.auditor import Auditor
from docdebt.core.java_parser import invoked_method_names, parse_java_source
from docdebt.core.models import MethodUnit
from docdebt.detection.context import README_NAMES, chunk_readme
from docdebt.rag.embeddings import Embedder, EmbeddingError

KINDS = ("method", "test", "readme", "validated")
_SCANNED_KINDS = ("method", "test", "readme")
_MAX_CHUNK_CHARS = 1200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    owner       TEXT NOT NULL,
    repo        TEXT NOT NULL,
    kind        TEXT NOT NULL,     -- method | test | readme | validated
    path        TEXT NOT NULL,
    name        TEXT NOT NULL,     -- method name ("" for README sections)
    label       TEXT NOT NULL,     -- what the UI shows, e.g. "Temperature.toCelsius"
    text        TEXT NOT NULL,
    extra       TEXT NOT NULL DEFAULT '',  -- tests: names of the methods they call
    vector      TEXT,              -- JSON list of floats, or NULL (BM25 only)
    embedder    TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_repo ON rag_chunks (owner, repo, kind);
CREATE TABLE IF NOT EXISTS rag_index (
    owner       TEXT NOT NULL,
    repo        TEXT NOT NULL,
    embedder    TEXT,
    indexed_at  REAL NOT NULL,
    commit_sha  TEXT,
    PRIMARY KEY (owner, repo)
);
"""


@dataclass(frozen=True)
class Chunk:
    id: int
    kind: str
    path: str
    name: str
    label: str
    text: str
    extra: str
    vector: list[float] | None
    embedder: str | None


def _is_test_path(path: str) -> bool:
    p = path.replace("\\", "/")
    return "/src/test/" in f"/{p}" or p.endswith("Test.java") or p.endswith("Tests.java")


def _method_text(unit: MethodUnit) -> str:
    return f"{unit.class_name}.{unit.code.split('{', 1)[0].strip()}\n{unit.javadoc or ''}"[:_MAX_CHUNK_CHARS]


class KnowledgeBase:
    """Read and write the ``rag_chunks`` table (one short connection per call)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- writing ---------------------------------------------------------------

    @staticmethod
    def _vectors(texts: list[str], embedder: Embedder | None) -> tuple[list[list[float] | None], str | None]:
        if embedder is None or not texts:
            return [None] * len(texts), None
        try:
            vectors: list[list[float] | None] = []
            for start in range(0, len(texts), 64):
                vectors.extend(embedder.embed(texts[start : start + 64]))
            return vectors, embedder.name
        except EmbeddingError:
            return [None] * len(texts), None

    def _insert(self, owner: str, repo: str, rows: list[dict], embedder: Embedder | None) -> str | None:
        vectors, used = self._vectors([r["text"] for r in rows], embedder)
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO rag_chunks (owner, repo, kind, path, name, label, text, extra, vector, embedder, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        owner,
                        repo,
                        r["kind"],
                        r["path"],
                        r["name"],
                        r["label"],
                        r["text"],
                        r.get("extra", ""),
                        json.dumps(v) if v is not None else None,
                        used if v is not None else None,
                        now,
                    )
                    for r, v in zip(rows, vectors, strict=True)
                ],
            )
        return used

    def rebuild(
        self,
        owner: str,
        repo: str,
        files: dict[str, str],
        *,
        readme: str | None = None,
        embedder: Embedder | None = None,
        commit_sha: str | None = None,
        auditor: Auditor | None = None,
    ) -> dict:
        """Replace the scanned chunks of ``owner/repo`` (validated ones are kept)."""
        auditor = auditor or Auditor()
        rows: list[dict] = []
        for path, source in files.items():
            if not path.endswith(".java"):
                continue
            units = parse_java_source(source, path)
            if _is_test_path(path):
                for u in units:
                    if u.has_body:
                        rows.append(
                            {
                                "kind": "test",
                                "path": path,
                                "name": u.name,
                                "label": f"{u.class_name}.{u.name}",
                                "text": u.code[:_MAX_CHUNK_CHARS],
                                "extra": " ".join(sorted(invoked_method_names(u.code))),
                            }
                        )
                continue
            for u in units:
                if not u.javadoc or auditor.audit(u).has_structural_debt:
                    continue
                rows.append(
                    {
                        "kind": "method",
                        "path": path,
                        "name": u.name,
                        "label": f"{u.class_name}.{u.name}",
                        "text": _method_text(u),
                    }
                )
        for i, section in enumerate(chunk_readme(readme or "")):
            rows.append(
                {"kind": "readme", "path": "README", "name": "", "label": f"README §{i + 1}", "text": section}
            )

        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM rag_chunks WHERE owner = ? AND repo = ? AND kind IN ({','.join('?' * 3)})",
                (owner, repo, *_SCANNED_KINDS),
            )
        used = self._insert(owner, repo, rows, embedder) if rows else None
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO rag_index (owner, repo, embedder, indexed_at, commit_sha) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(owner, repo) DO UPDATE SET embedder = excluded.embedder, "
                "indexed_at = excluded.indexed_at, commit_sha = excluded.commit_sha",
                (owner, repo, used, time.time(), commit_sha),
            )
        return self.stats(owner, repo)

    def add_validated(
        self, owner: str, repo: str, unit: MethodUnit, javadoc: str, embedder: Embedder | None = None
    ) -> None:
        """Remember a Javadoc a reviewer accepted, as an example for later repairs."""
        text = f"{unit.class_name}.{unit.code.split('{', 1)[0].strip()}\n{javadoc}"[:_MAX_CHUNK_CHARS]
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM rag_chunks WHERE owner = ? AND repo = ? AND kind = 'validated' AND path = ? "
                "AND name = ?",
                (owner, repo, unit.file_path, unit.name),
            )
        self._insert(
            owner,
            repo,
            [
                {
                    "kind": "validated",
                    "path": unit.file_path,
                    "name": unit.name,
                    "label": f"{unit.class_name}.{unit.name}",
                    "text": text,
                }
            ],
            embedder,
        )

    # -- reading ---------------------------------------------------------------

    def chunks(self, owner: str, repo: str, kinds: Iterable[str] = KINDS) -> list[Chunk]:
        kinds = list(kinds)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM rag_chunks WHERE owner = ? AND repo = ? "
                f"AND kind IN ({','.join('?' * len(kinds))})",
                (owner, repo, *kinds),
            ).fetchall()
        return [
            Chunk(
                id=r["id"],
                kind=r["kind"],
                path=r["path"],
                name=r["name"],
                label=r["label"],
                text=r["text"],
                extra=r["extra"],
                vector=json.loads(r["vector"]) if r["vector"] else None,
                embedder=r["embedder"],
            )
            for r in rows
        ]

    def stats(self, owner: str, repo: str) -> dict:
        with self._connect() as conn:
            counts = {
                row["kind"]: row["n"]
                for row in conn.execute(
                    "SELECT kind, COUNT(*) AS n FROM rag_chunks WHERE owner = ? AND repo = ? GROUP BY kind",
                    (owner, repo),
                )
            }
            index = conn.execute("SELECT * FROM rag_index WHERE owner = ? AND repo = ?", (owner, repo)).fetchone()
        return {
            "counts": {kind: counts.get(kind, 0) for kind in KINDS},
            "total": sum(counts.values()),
            "embedder": (index["embedder"] if index else None) or "bm25",
            "indexed_at": index["indexed_at"] if index else None,
            "commit_sha": index["commit_sha"] if index else None,
        }


def find_readme(paths: Iterable[str]) -> str | None:
    """The repository's top-level README path, if any."""
    top = {p for p in paths if "/" not in p}
    return next((name for name in README_NAMES if name in top), None)
