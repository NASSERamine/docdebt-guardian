"""Build the retrieved context for one method, and remember what was used.

Always on, for detection and repair alike, in two parts:

1. **Siblings** (the best context of the research): up to three documented
   methods of the same file, the closest to the target first. Read from the
   file under review, no index needed.
2. **Hybrid search** in the repository's knowledge base (debt-free
   documented methods, tests, README sections, accepted fixes): the BM25
   ranking and the vector ranking (``all-MiniLM-L6-v2``, Chroma) are merged
   by Reciprocal Rank Fusion, ``score = sum(1 / (60 + rank))``. Fusion uses
   ranks, never raw scores, so there is no similarity threshold to tune.
   The top four are kept.

Determinism: fixed sizes (3 siblings, top 4, 700 characters per excerpt,
3,600 in total), and ties broken by line distance (siblings) or chunk id
(search). The same index and the same method always give the same context.

Leak guard, as in the research: no excerpt about a method with the target's
own name (the method itself, its overloads, same-name methods elsewhere)
ever enters the context, and commit messages are never used.

Every excerpt is returned with its source, rank and fused score, so the
proposal page shows exactly what the model was given.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from docdebt.core.java_parser import parse_java_source
from docdebt.core.models import MethodUnit
from docdebt.rag.embeddings import Embedder, EmbeddingError, bm25_scores_tokens, tokenize
from docdebt.rag.knowledge import Chunk, KnowledgeBase

logger = logging.getLogger("docdebt.rag")

SOURCES = ("siblings", "similar", "tests", "readme", "validated")
RRF_K = 60
TOP_K = 4
MAX_SIBLINGS = 3
_DENSE_CANDIDATES = 50
_TRIVIAL = re.compile(r"^(get|set|is)[A-Z_]")
_ITEM_CHARS = 700
_TOTAL_CHARS = 3600

_KIND_SOURCE = {"method": "similar", "test": "tests", "readme": "readme", "validated": "validated"}
_TITLES = {
    "siblings": "Documented method in the same file",
    "similar": "Similar documented method in the project",
    "validated": "Javadoc a reviewer accepted in this project",
    "tests": "Test of the project",
    "readme": "README excerpt",
}


@dataclass
class ContextItem:
    source: str
    label: str
    path: str
    score: float | None
    text: str
    rank: int = 0
    lexical_rank: int | None = None
    dense_rank: int | None = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "label": self.label,
            "path": self.path,
            "score": None if self.score is None else round(self.score, 5),
            "rank": self.rank,
            "lexical_rank": self.lexical_rank,
            "dense_rank": self.dense_rank,
            "text": self.text,
        }


@dataclass
class RetrievedContext:
    items: list[ContextItem] = field(default_factory=list)

    @property
    def text(self) -> str:
        parts, total = [], 0
        for item in self.items:
            block = f"--- {_TITLES[item.source]}: {item.label} ({item.path}) ---\n{item.text}"
            if total + len(block) > _TOTAL_CHARS:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)


def _trivial(unit: MethodUnit) -> bool:
    """A one-statement getter or setter: its Javadoc says nothing useful."""
    body = unit.code.split("{", 1)[1] if "{" in unit.code else ""
    return bool(_TRIVIAL.match(unit.name)) and body.count(";") <= 1


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion of several rankings of chunk ids (best first).

    Returns ``(id, score)`` pairs, best first; equal scores are ordered by id.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda p: (-round(p[1], 12), p[0]))


def siblings_of(unit: MethodUnit, file_source: str | None, limit: int = MAX_SIBLINGS) -> list[ContextItem]:
    """The documented methods of the same file closest to ``unit`` (leak guard applied)."""
    if not file_source:
        return []
    methods = parse_java_source(file_source, unit.file_path)
    candidates = [m for m in methods if m.name != unit.name and m.javadoc and m.has_body and not _trivial(m)]
    candidates.sort(key=lambda m: (abs(m.start_line - unit.start_line), m.start_line))
    return [
        ContextItem(
            "siblings",
            f"{m.class_name}.{m.name}",
            m.file_path,
            None,
            f"{m.javadoc}\n{m.code}"[:_ITEM_CHARS],
            rank=i + 1,
        )
        for i, m in enumerate(candidates[:limit])
    ]


class RagProvider:
    """Context for the methods of one repository during one review.

    Used both as the Router's ``context_provider`` (called with a unit,
    returns text) and by the repair node (``build`` returns the excerpts).
    """

    def __init__(
        self,
        kb: KnowledgeBase | None,
        owner: str,
        repo: str,
        *,
        embedder: Embedder | None = None,
        top_k: int = TOP_K,
        max_siblings: int = MAX_SIBLINGS,
    ) -> None:
        self.kb, self.owner, self.repo = kb, owner, repo
        self.embedder = embedder
        self.top_k = top_k
        self.max_siblings = max_siblings
        self.files: dict[str, str] = {}
        self._chunks: list[Chunk] | None = None
        self._tokens: dict[int, list[str]] = {}
        self._cache: dict[str, RetrievedContext] = {}

    def set_files(self, files: dict[str, str]) -> None:
        """The full text of the files under review (fetched from the PR head)."""
        self.files = dict(files)
        self._cache.clear()

    # -- hybrid search ---------------------------------------------------------

    def _kb_chunks(self) -> list[Chunk]:
        if self._chunks is None:
            self._chunks = self.kb.chunks(self.owner, self.repo) if self.kb is not None else []
            self._chunks.sort(key=lambda c: c.id)
            self._tokens = {c.id: tokenize(c.text) for c in self._chunks}
        return self._chunks

    @staticmethod
    def query_text(unit: MethodUnit) -> str:
        return f"{unit.class_name} {unit.name}\n{unit.code[:600]}"

    def _lexical(self, query: str, candidates: list[Chunk]) -> list[int]:
        scores = bm25_scores_tokens(tokenize(query), [self._tokens[c.id] for c in candidates])
        ranked = sorted(
            ((s, c.id) for s, c in zip(scores, candidates, strict=True) if s > 0), key=lambda p: (-p[0], p[1])
        )
        return [chunk_id for _, chunk_id in ranked]

    def _dense(self, query: str, allowed: set[int]) -> list[int]:
        if self.embedder is None or self.kb is None or not allowed:
            return []
        try:
            vector = self.embedder.embed([query])[0]
            hits = self.kb.index.query(self.owner, self.repo, vector, _DENSE_CANDIDATES + 10)
        except EmbeddingError as exc:
            logger.warning("vector search skipped (%s): BM25 only for this method", exc)
            return []
        except Exception:  # noqa: BLE001 - a broken index must not stop a review
            logger.exception("vector search failed: BM25 only for this method")
            return []
        return [chunk_id for chunk_id, _ in hits if chunk_id in allowed][:_DENSE_CANDIDATES]

    def search(self, unit: MethodUnit, exclude: set[tuple[str, str]] | None = None) -> list[ContextItem]:
        """The top ``top_k`` chunks of the knowledge base for ``unit``, by RRF."""
        exclude = exclude or set()
        candidates = [
            c
            for c in self._kb_chunks()
            if c.name != unit.name  # leak guard: the method itself, its overloads, same-name methods
            and (c.path, c.label) not in exclude
        ]
        if not candidates:
            return []
        query = self.query_text(unit)
        lexical = self._lexical(query, candidates)
        dense = self._dense(query, {c.id for c in candidates})
        by_id = {c.id: c for c in candidates}
        lex_rank = {cid: i + 1 for i, cid in enumerate(lexical)}
        dense_rank = {cid: i + 1 for i, cid in enumerate(dense)}
        items = []
        for position, (chunk_id, score) in enumerate(rrf_fuse([lexical, dense])[: self.top_k], start=1):
            c = by_id[chunk_id]
            items.append(
                ContextItem(
                    _KIND_SOURCE.get(c.kind, "similar"),
                    c.label,
                    c.path,
                    score,
                    c.text[:_ITEM_CHARS],
                    rank=position,
                    lexical_rank=lex_rank.get(chunk_id),
                    dense_rank=dense_rank.get(chunk_id),
                )
            )
        return items

    # -- public ------------------------------------------------------------------

    def build(self, unit: MethodUnit) -> RetrievedContext:
        if unit.id in self._cache:
            return self._cache[unit.id]
        siblings = siblings_of(unit, self.files.get(unit.file_path), self.max_siblings)
        found = self.search(unit, exclude={(i.path, i.label) for i in siblings})
        context = RetrievedContext(siblings + found)
        self._cache[unit.id] = context
        return context

    def __call__(self, unit: MethodUnit) -> str:
        """Context text only: the Router's ``context_provider`` interface."""
        return self.build(unit).text
