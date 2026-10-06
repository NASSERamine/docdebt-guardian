"""Optional project context for the Judge.

Port of the research retrieval (``src/detection/context_retrieval.py``)
adapted to production: the repository is the local checkout of the pull
request head, so nothing is fetched from GitHub, and commit messages are
never used (they leak the label: "fix outdated javadoc").

Sources, each usable alone or combined:

* ``siblings``: the first two other methods of the same file, in source
  order, that have a Javadoc; shown with Javadoc and source.
* ``callees``: methods of the same file whose name the target invokes,
  shown with their Javadoc if any. **Fixed research leak:** the target's
  own name is excluded, so neither the method itself (recursion) nor its
  overloads can put a Javadoc of the target's name into its own context.
* ``tests``: up to two methods of ``<Class>Test.java`` (Maven/Gradle
  ``src/main`` -> ``src/test`` mirror) that invoke the target name; source only.
* ``readme``: the README split at headings, each section cut to 500
  characters, the two sections most similar to the method (cosine of
  all-MiniLM-L6-v2 embeddings); needs the ``context`` extra.

Research aliases are accepted: ``siblings_called`` = siblings + callees,
``siblings_tests`` = siblings + tests.

Context is disabled by default: in the research its effect on detection
accuracy was small and depended on the model and the pattern.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from docdebt.core.java_parser import invoked_method_names, parse_java_source
from docdebt.core.models import MethodUnit

VALID_SOURCES = ("siblings", "callees", "tests", "readme")
_ALIASES = {
    "siblings_called": ("siblings", "callees"),
    "siblings_tests": ("siblings", "tests"),
    "siblings+callees": ("siblings", "callees"),
    "siblings+tests": ("siblings", "tests"),
}
README_NAMES = ("README.md", "Readme.md", "readme.md", "README.rst", "README.txt")


def expand_sources(sources: list[str]) -> tuple[str, ...]:
    """Resolve aliases and validate source names, keeping order and removing duplicates."""
    out: list[str] = []
    for name in sources:
        for resolved in _ALIASES.get(name, (name,)):
            if resolved not in VALID_SOURCES:
                raise ValueError(f"Unknown context source '{resolved}'. Valid: {VALID_SOURCES}.")
            if resolved not in out:
                out.append(resolved)
    return tuple(out)


def chunk_readme(text: str, chunk_size: int = 500) -> list[str]:
    """Split a README at headings (``\\n#``) and cut each section to ``chunk_size``."""
    chunks = []
    for i, section in enumerate(text.split("\n#")):
        section = (("#" + section) if i > 0 else section).strip()
        if section:
            chunks.append(section[:chunk_size])
    return chunks


class ContextRetriever:
    """Build the context text for a method from a local repository checkout."""

    def __init__(
        self,
        repo_root: str | Path,
        sources: list[str],
        *,
        embedder: object | None = None,
        readme_top_k: int = 2,
        readme_chunk_chars: int = 500,
    ) -> None:
        self.root = Path(repo_root).resolve()
        self.sources = expand_sources(sources)
        self._embedder = embedder
        self.readme_top_k = readme_top_k
        self.readme_chunk_chars = readme_chunk_chars
        self._methods_of = lru_cache(maxsize=256)(self._parse_file)

    # -- helpers -----------------------------------------------------------

    def _parse_file(self, path: str) -> tuple[MethodUnit, ...]:
        file_path = Path(path)
        if not file_path.is_file():
            return ()
        source = file_path.read_text(encoding="utf-8", errors="replace")
        return tuple(parse_java_source(source, str(file_path)))

    def _abs(self, unit: MethodUnit) -> Path:
        path = Path(unit.file_path)
        return path if path.is_absolute() else self.root / path

    def _embed_model(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    # -- sources -----------------------------------------------------------

    def _siblings(self, unit: MethodUnit, methods: tuple[MethodUnit, ...]) -> list[str]:
        siblings = [m for m in methods if m.name != unit.name and m.has_javadoc][:2]
        return [f"--- Sibling method in the same file ---\n{m.javadoc}\n{m.code}" for m in siblings]

    def _callees(self, unit: MethodUnit, methods: tuple[MethodUnit, ...]) -> list[str]:
        called = invoked_method_names(unit.code) - {unit.name}  # leak fix
        parts = []
        for m in methods:
            if m.name in called:
                doc = f"{m.javadoc}\n" if m.javadoc else ""
                parts.append(f"--- Called method (same file) ---\n{doc}{m.code}")
        return parts

    def _tests(self, unit: MethodUnit) -> list[str]:
        source_path = self._abs(unit).as_posix()
        if "/src/main/" not in source_path:
            return []
        test_path = Path(source_path.replace("/src/main/", "/src/test/"))
        test_path = test_path.with_name(test_path.stem + "Test.java")
        matching = [
            m for m in self._methods_of(str(test_path)) if m.has_body and unit.name in invoked_method_names(m.code)
        ][:2]
        return [f"--- Test exercising this method ---\n{m.code}" for m in matching]

    def _readme(self, unit: MethodUnit) -> list[str]:
        readme = next((self.root / n for n in README_NAMES if (self.root / n).is_file()), None)
        if readme is None:
            return []
        chunks = chunk_readme(readme.read_text(encoding="utf-8", errors="replace"), self.readme_chunk_chars)
        if not chunks:
            return []
        import numpy as np

        model = self._embed_model()
        chunk_embs = np.asarray(model.encode(chunks, show_progress_bar=False))
        query = f"{unit.name} {unit.code[:200]}"
        query_emb = np.asarray(model.encode([query], show_progress_bar=False))[0]
        sims = chunk_embs @ query_emb / (np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(query_emb) + 1e-12)
        top = np.argsort(sims)[::-1][: self.readme_top_k]
        return [f"--- README excerpt (similarity={sims[i]:.2f}) ---\n{chunks[i]}" for i in top]

    # -- public ------------------------------------------------------------

    def retrieve(self, unit: MethodUnit) -> str:
        """Context text for ``unit``; empty string when nothing is found."""
        methods = self._methods_of(str(self._abs(unit)))
        parts: list[str] = []
        if "siblings" in self.sources:
            parts += self._siblings(unit, methods)
        if "callees" in self.sources:
            parts += self._callees(unit, methods)
        if "tests" in self.sources:
            parts += self._tests(unit)
        if "readme" in self.sources:
            parts += self._readme(unit)
        return "\n\n".join(parts)

    __call__ = retrieve
