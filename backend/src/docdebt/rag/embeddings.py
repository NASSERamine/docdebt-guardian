"""Vectorisation of the knowledge base, and the BM25 lexical scorer.

* ``MiniLMEmbedder``: ``all-MiniLM-L6-v2`` (sentence-transformers), the
  embedding model of the research, run inside the service process on the
  CPU: free, private (code never leaves the machine), deterministic.
* BM25 (Okapi) with Java identifiers split apart, the lexical half of the
  hybrid search; it needs no model at all, so retrieval still works when the
  embedding model cannot be loaded.

A failing embedder (package missing, model not downloaded) is never fatal:
retrieval continues with BM25 alone and the status banner says so.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

logger = logging.getLogger("docdebt.rag")

_WORD = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _stem(word: str) -> str:
    """A light English stemmer: plural "s" only (amounts -> amount), never on short words."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase words, with camelCase and snake_case identifiers split apart."""
    return [_stem(w.lower()) for w in _WORD.findall(text or "") if len(w) > 1]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def bm25_scores(query: str, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 score of every document for ``query``."""
    return bm25_scores_tokens(tokenize(query), [tokenize(d) for d in documents], k1, b)


def bm25_scores_tokens(
    query: Sequence[str], docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """BM25 on already tokenized documents (the retriever tokenizes the base once)."""
    if not docs:
        return []
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    terms = set(query)
    scores = []
    for d in docs:
        tf = Counter(d)
        score = 0.0
        for t in terms:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(score)
    return scores


class EmbeddingError(RuntimeError):
    """The embedding model could not answer."""


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


_MODELS: dict[str, object] = {}
_LOCK = threading.Lock()


class MiniLMEmbedder:
    """``all-MiniLM-L6-v2`` on the CPU, normalised vectors (cosine = dot product).

    The model is loaded once per process, on first use; it is downloaded
    from Hugging Face the first time (about 90 MB), then read from the local
    cache. ``DOCDEBT_EMBEDDING_MODEL`` can point to a local copy.
    """

    def __init__(self, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model = model

    @property
    def name(self) -> str:
        return f"minilm:{self.model.rsplit('/', 1)[-1]}"

    def _load(self):
        with _LOCK:
            if self.model not in _MODELS:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:  # the "rag" extra is not installed
                    raise EmbeddingError("sentence-transformers is not installed (pip install '.[rag]')") from exc
                try:
                    _MODELS[self.model] = SentenceTransformer(self.model, device="cpu")
                except Exception as exc:  # noqa: BLE001 - download or load failure
                    raise EmbeddingError(f"cannot load {self.model}: {exc}") from exc
            return _MODELS[self.model]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        try:
            vectors = model.encode(list(texts), normalize_embeddings=True, batch_size=32)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"embedding failed: {exc}") from exc
        return [[float(x) for x in v] for v in vectors]


_status: dict[str, str | None] = {"name": None, "error": None}
_RETRY_AFTER_S = 600.0
_failed_at: dict[str, float] = {}


def choose_embedder(mode: str = "minilm") -> Embedder | None:
    """The embedding model, or ``None`` for BM25 only (``mode="bm25"``, or a
    model that cannot be loaded). The outcome is kept for the status banner."""
    if (mode or "minilm").lower() == "bm25":
        _status.update(name=None, error="disabled (DOCDEBT_RAG_EMBEDDINGS=bm25)")
        return None
    import os
    import time

    embedder = MiniLMEmbedder(os.environ.get("DOCDEBT_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL)
    failed = _failed_at.get(embedder.model)
    if failed is not None and time.monotonic() - failed < _RETRY_AFTER_S:
        return None  # failed less than ten minutes ago: do not slow every review down
    try:
        embedder.embed(["ping"])
    except EmbeddingError as exc:
        _failed_at[embedder.model] = time.monotonic()
        _status.update(name=None, error=str(exc))
        logger.warning("embeddings unavailable (%s): retrieval uses BM25 only", exc)
        return None
    _failed_at.pop(embedder.model, None)
    _status.update(name=embedder.name, error=None)
    return embedder


def embedder_status() -> dict:
    """The last outcome of :func:`choose_embedder` (``name`` is ``None`` when unavailable)."""
    return dict(_status)
