"""Embedding engines for the knowledge base, and the BM25 fallback.

Three engines, chosen by ``choose_embedder``:

* ``ollama``: a local embedding model (``nomic-embed-text`` by default)
  served by the same Ollama server as the chat model: free and private;
* ``openai``: ``text-embedding-3-small``, only for repositories that allow
  sending code to the cloud;
* ``bm25``: no vectors at all, lexical scoring at query time. Always
  available, so retrieval never depends on an embedding server.

An engine that fails (server down, model not pulled) is never fatal: the
caller falls back to BM25.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

logger = logging.getLogger("docdebt.rag")

_WORD = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")


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
    docs = [tokenize(d) for d in documents]
    if not docs:
        return []
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    terms = set(tokenize(query))
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
    """The embedding engine could not answer."""


@dataclass
class OllamaEmbedder:
    base_url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    timeout_s: float = 60.0

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = httpx.post(
                f"{self.base_url.rstrip('/')}/api/embed",
                json={"model": self.model, "input": list(texts)},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            return [list(map(float, v)) for v in response.json()["embeddings"]]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise EmbeddingError(f"Ollama embeddings failed: {exc}") from exc


@dataclass
class OpenAIEmbedder:
    api_key: str
    model: str = "text-embedding-3-small"
    timeout_s: float = 60.0

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": list(texts)},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            return [list(map(float, row["embedding"])) for row in response.json()["data"]]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise EmbeddingError(f"OpenAI embeddings failed: {exc}") from exc


Embedder = OllamaEmbedder | OpenAIEmbedder


_PROBE_TTL_S = 600.0
_probe_cache: dict[tuple[str, str], tuple[float, bool]] = {}


def choose_embedder(mode: str = "auto", *, allow_cloud: bool = False, probe: bool = True) -> Embedder | None:
    """The embedding engine to use, or ``None`` for BM25 only.

    ``mode`` is ``auto`` (Ollama if it answers, else BM25), ``ollama``,
    ``openai`` (only when ``allow_cloud``; otherwise BM25) or ``bm25``.
    Whether Ollama answers is remembered for ten minutes, so a stopped
    Ollama does not slow down every review.
    """
    mode = (mode or "auto").lower()
    if mode == "bm25":
        return None
    if mode == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if allow_cloud and key:
            return OpenAIEmbedder(api_key=key)
        return None
    embedder = OllamaEmbedder(
        base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        model=os.environ.get("DOCDEBT_RAG_OLLAMA_MODEL", "nomic-embed-text"),
        timeout_s=60.0,
    )
    if not probe:
        return embedder
    key = (embedder.base_url, embedder.model)
    cached = _probe_cache.get(key)
    if cached is not None and time.monotonic() - cached[0] < _PROBE_TTL_S:
        return embedder if cached[1] else None
    try:
        OllamaEmbedder(embedder.base_url, embedder.model, timeout_s=5.0).embed(["ping"])
        _probe_cache[key] = (time.monotonic(), True)
        return embedder
    except EmbeddingError as exc:
        _probe_cache[key] = (time.monotonic(), False)
        logger.info("Ollama embeddings unavailable (%s): using BM25 for retrieval", exc)
        return None
