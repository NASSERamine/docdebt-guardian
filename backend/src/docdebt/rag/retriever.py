"""Build the retrieved context for one method, and remember what was used.

Sources (each can be switched off):

* ``siblings``: the two documented methods of the same file closest to the
  target (the source that improved meaning the most in the research);
* ``callees``: documented methods of the same file the target calls;
* ``similar``: the most similar debt-free documented methods of the whole
  repository (knowledge base);
* ``validated``: the closest Javadoc a reviewer accepted before;
* ``tests``: a test method that calls the target;
* ``readme``: the README section closest to the method.

Leak guard, as in the research: no excerpt about a method with the target's
own name (the method itself, its overloads, same-name methods elsewhere)
ever enters the context, and commit messages are never used.

Every excerpt is returned with its source, location and score, so the
dashboard can show exactly what the model was given.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from docdebt.core.java_parser import invoked_method_names, parse_java_source
from docdebt.core.models import MethodUnit
from docdebt.rag.embeddings import Embedder, EmbeddingError, bm25_scores, cosine
from docdebt.rag.knowledge import Chunk, KnowledgeBase

SOURCES = ("siblings", "callees", "similar", "validated", "tests", "readme")
DEFAULT_SOURCES = ("siblings", "callees", "similar", "validated")
_TRIVIAL = re.compile(r"^(get|set|is)[A-Z_]")
_ITEM_CHARS = 700
_TOTAL_CHARS = 3600

_TITLES = {
    "siblings": "Documented method in the same file",
    "callees": "Method called by this one",
    "similar": "Similar documented method in the project",
    "validated": "Javadoc a reviewer accepted in this project",
    "tests": "Test that calls this method",
    "readme": "README excerpt",
}


@dataclass
class ContextItem:
    source: str
    label: str
    path: str
    score: float | None
    text: str

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "label": self.label,
            "path": self.path,
            "score": None if self.score is None else round(self.score, 3),
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
        sources: tuple[str, ...] = DEFAULT_SOURCES,
        top_k: int = 2,
        embedder: Embedder | None = None,
    ) -> None:
        unknown = set(sources) - set(SOURCES)
        if unknown:
            raise ValueError(f"Unknown context sources {sorted(unknown)}. Valid: {SOURCES}.")
        self.kb, self.owner, self.repo = kb, owner, repo
        self.sources = tuple(sources)
        self.top_k = max(1, top_k)
        self.embedder = embedder
        self.files: dict[str, str] = {}
        self._chunks: list[Chunk] | None = None
        self._cache: dict[str, RetrievedContext] = {}

    def set_files(self, files: dict[str, str]) -> None:
        """The full text of the files under review (fetched from the PR head)."""
        self.files = dict(files)
        self._cache.clear()

    # -- sources -------------------------------------------------------------

    def _file_methods(self, unit: MethodUnit) -> list[MethodUnit]:
        source = self.files.get(unit.file_path)
        return parse_java_source(source, unit.file_path) if source else []

    def _siblings(self, unit: MethodUnit, methods: list[MethodUnit]) -> list[ContextItem]:
        candidates = [m for m in methods if m.name != unit.name and m.javadoc and m.has_body and not _trivial(m)]
        candidates.sort(key=lambda m: abs(m.start_line - unit.start_line))
        return [
            ContextItem(
                "siblings", f"{m.class_name}.{m.name}", m.file_path, None, f"{m.javadoc}\n{m.code}"[:_ITEM_CHARS]
            )
            for m in candidates[:2]
        ]

    def _callees(self, unit: MethodUnit, methods: list[MethodUnit]) -> list[ContextItem]:
        called = invoked_method_names(unit.code) - {unit.name}  # leak guard
        return [
            ContextItem(
                "callees", f"{m.class_name}.{m.name}", m.file_path, None, f"{m.javadoc}\n{m.code}"[:_ITEM_CHARS]
            )
            for m in methods
            if m.name in called and m.javadoc
        ][:2]

    def _kb_chunks(self) -> list[Chunk]:
        if self._chunks is None:
            self._chunks = self.kb.chunks(self.owner, self.repo) if self.kb is not None else []
        return self._chunks

    def _rank(self, query: str, chunks: list[Chunk]) -> list[tuple[float, Chunk]]:
        if not chunks:
            return []
        if self.embedder is not None and all(c.vector and c.embedder == self.embedder.name for c in chunks):
            try:
                q = self.embedder.embed([query])[0]
                scored = [(cosine(q, c.vector or []), c) for c in chunks]
                return sorted(scored, key=lambda p: p[0], reverse=True)
            except EmbeddingError:
                pass
        scores = bm25_scores(query, [c.text for c in chunks])
        return sorted(zip(scores, chunks, strict=True), key=lambda p: p[0], reverse=True)

    def _from_kb(self, unit: MethodUnit, kind: str, source: str, k: int) -> list[ContextItem]:
        chunks = [c for c in self._kb_chunks() if c.kind == kind and c.name != unit.name]  # leak guard
        if kind == "test":
            chunks = [c for c in chunks if unit.name in c.extra.split()]
        query = f"{unit.class_name} {unit.name} {unit.code[:600]}"
        ranked = [(s, c) for s, c in self._rank(query, chunks) if s > 0]
        return [ContextItem(source, c.label, c.path, s, c.text[:_ITEM_CHARS]) for s, c in ranked[:k]]

    # -- public --------------------------------------------------------------

    def build(self, unit: MethodUnit) -> RetrievedContext:
        if unit.id in self._cache:
            return self._cache[unit.id]
        methods = self._file_methods(unit) if {"siblings", "callees"} & set(self.sources) else []
        items: list[ContextItem] = []
        # Called methods first: what the method relies on explains it best.
        if "callees" in self.sources:
            items += self._callees(unit, methods)
        if "siblings" in self.sources:
            seen = {(i.path, i.label) for i in items}
            items += [i for i in self._siblings(unit, methods) if (i.path, i.label) not in seen]
        if "validated" in self.sources:
            items += self._from_kb(unit, "validated", "validated", 1)
        if "similar" in self.sources:
            seen = {(i.path, i.label) for i in items}
            items += [
                i
                for i in self._from_kb(unit, "method", "similar", self.top_k + 2)
                if (i.path, i.label) not in seen
            ][: self.top_k]
        if "tests" in self.sources:
            items += self._from_kb(unit, "test", "tests", 1)
        if "readme" in self.sources:
            items += self._from_kb(unit, "readme", "readme", 1)
        context = RetrievedContext(items)
        self._cache[unit.id] = context
        return context

    def __call__(self, unit: MethodUnit) -> str:
        """Context text only: the Router's ``context_provider`` interface."""
        return self.build(unit).text
