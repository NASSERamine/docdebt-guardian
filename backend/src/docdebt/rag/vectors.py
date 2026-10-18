"""Vector index of the knowledge base: Chroma, or an exact SQLite fallback.

The chunks themselves (text, kind, location) live in SQLite
(:mod:`docdebt.rag.knowledge`); this index only maps a chunk id to its
vector and answers "the ``n`` nearest chunks of this repository".

* :class:`ChromaIndex`: a persistent, embedded Chroma collection (cosine
  space), stored next to the service state. No server to run.
* :class:`SqliteVectorIndex`: vectors in the ``rag_chunks.vector`` column and
  an exact cosine search in Python; used when ``chromadb`` is not installed
  and in the tests.

Both return results ordered by distance, then by chunk id, so the same
query on the same index always gives the same order.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from docdebt.rag.embeddings import cosine

logger = logging.getLogger("docdebt.rag")


class VectorIndex(Protocol):
    name: str

    def upsert(self, owner: str, repo: str, items: Sequence[tuple[int, list[float], str]]) -> None: ...

    def delete(self, owner: str, repo: str, chunk_ids: Sequence[int] | None = None) -> None: ...

    def query(self, owner: str, repo: str, vector: list[float], n: int) -> list[tuple[int, float]]: ...


def _repo_key(owner: str, repo: str) -> str:
    return f"{owner}/{repo}"


class ChromaIndex:
    """Embedded Chroma collection ``docdebt_chunks`` under ``directory``."""

    name = "chroma"

    def __init__(self, directory: str | Path) -> None:
        import chromadb  # optional dependency: the "rag" extra
        from chromadb.config import Settings as ChromaSettings

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.directory), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            "docdebt_chunks", metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, owner: str, repo: str, items: Sequence[tuple[int, list[float], str]]) -> None:
        if not items:
            return
        key = _repo_key(owner, repo)
        for start in range(0, len(items), 500):
            batch = items[start : start + 500]
            self._collection.upsert(
                ids=[str(chunk_id) for chunk_id, _, _ in batch],
                embeddings=[vector for _, vector, _ in batch],
                metadatas=[{"repo": key, "kind": kind} for _, _, kind in batch],
            )

    def delete(self, owner: str, repo: str, chunk_ids: Sequence[int] | None = None) -> None:
        if chunk_ids is None:
            self._collection.delete(where={"repo": _repo_key(owner, repo)})
        elif chunk_ids:
            self._collection.delete(ids=[str(i) for i in chunk_ids])

    def query(self, owner: str, repo: str, vector: list[float], n: int) -> list[tuple[int, float]]:
        key = _repo_key(owner, repo)
        count = self._collection.count()
        if count == 0 or n <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[vector], n_results=min(n, count), where={"repo": key}, include=["distances"]
        )
        pairs = [(int(i), float(d)) for i, d in zip(result["ids"][0], result["distances"][0], strict=True)]
        return sorted(pairs, key=lambda p: (round(p[1], 9), p[0]))


class SqliteVectorIndex:
    """Exact cosine search over the vectors stored in ``rag_chunks.vector``."""

    name = "sqlite"

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert(self, owner: str, repo: str, items: Sequence[tuple[int, list[float], str]]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "UPDATE rag_chunks SET vector = ? WHERE id = ?",
                [(json.dumps(vector), chunk_id) for chunk_id, vector, _ in items],
            )
        conn.close()

    def delete(self, owner: str, repo: str, chunk_ids: Sequence[int] | None = None) -> None:
        return  # the vectors go with their rows

    def query(self, owner: str, repo: str, vector: list[float], n: int) -> list[tuple[int, float]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, vector FROM rag_chunks WHERE owner = ? AND repo = ? AND vector IS NOT NULL",
                (owner, repo),
            ).fetchall()
        conn.close()
        pairs = [(int(r["id"]), 1.0 - cosine(vector, json.loads(r["vector"]))) for r in rows]
        return sorted(pairs, key=lambda p: (round(p[1], 9), p[0]))[:n]


def default_index(state_db_path: str | Path) -> VectorIndex:
    """Chroma next to the state database when installed, else the SQLite fallback."""
    try:
        return ChromaIndex(Path(state_db_path).parent / "chroma")
    except ImportError:
        logger.warning("chromadb is not installed (pip install '.[rag]'): exact SQLite vector search is used")
    except Exception:  # noqa: BLE001 - a broken store must not stop the service
        logger.exception("could not open the Chroma index: exact SQLite vector search is used")
    return SqliteVectorIndex(state_db_path)
