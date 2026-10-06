"""Extractor: turn Java files, directories or a diff into ``MethodUnit`` objects.

This is the first node of the pipeline. It decides *which* methods are
analysed; everything downstream works one method at a time.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from docdebt.core.diff import parse_unified_diff
from docdebt.core.java_parser import parse_java_source
from docdebt.core.models import MethodUnit, Visibility

_ALL_VISIBILITIES = frozenset(Visibility)


@dataclass
class ExtractorConfig:
    """Which methods to keep.

    Attributes:
        visibilities: keep only methods with one of these visibilities.
        require_javadoc: skip methods without a Javadoc. Documentation
            debt is about documentation that exists but is wrong, so this
            is ``True`` by default.
        skip_syntax_errors: skip methods whose source does not parse cleanly.
        exclude: glob patterns (matched against the path relative to the
            root) of files to ignore, e.g. ``"**/generated/**"``.
    """

    visibilities: frozenset[Visibility] = _ALL_VISIBILITIES
    require_javadoc: bool = True
    skip_syntax_errors: bool = True
    exclude: tuple[str, ...] = field(default_factory=tuple)


class Extractor:
    """Select the methods to analyse from sources, files, directories or diffs."""

    def __init__(self, config: ExtractorConfig | None = None) -> None:
        self.config = config or ExtractorConfig()

    # -- filtering ---------------------------------------------------------

    def _keep(self, unit: MethodUnit) -> bool:
        cfg = self.config
        if cfg.require_javadoc and not unit.has_javadoc:
            return False
        if cfg.skip_syntax_errors and not unit.syntax_ok:
            return False
        return unit.visibility in cfg.visibilities

    def is_excluded(self, relative_path: str) -> bool:
        path = relative_path.replace("\\", "/")
        return any(
            fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch("/" + path, pattern)
            for pattern in self.config.exclude
        )

    # -- sources -----------------------------------------------------------

    def from_source(self, source: str, file_path: str = "<memory>") -> list[MethodUnit]:
        """Methods of one Java compilation unit, after filtering."""
        return [u for u in parse_java_source(source, file_path) if self._keep(u)]

    def from_file(self, path: str | Path, root: str | Path | None = None) -> list[MethodUnit]:
        """Methods of one ``.java`` file; ``file_path`` is made relative to ``root`` if given."""
        path = Path(path).resolve()
        display = str(path.relative_to(Path(root).resolve())) if root else str(path)
        source = path.read_text(encoding="utf-8", errors="replace")
        return self.from_source(source, display.replace("\\", "/"))

    def iter_java_files(self, root: str | Path) -> Iterator[Path]:
        root = Path(root).resolve()
        for path in sorted(root.rglob("*.java")):
            relative = str(path.relative_to(root)).replace("\\", "/")
            if not self.is_excluded(relative):
                yield path

    def from_directory(self, root: str | Path) -> list[MethodUnit]:
        """Methods of every ``.java`` file under ``root`` (excluded paths skipped)."""
        units: list[MethodUnit] = []
        for path in self.iter_java_files(root):
            units.extend(self.from_file(path, root))
        return units

    def from_paths(self, paths: Iterable[str | Path]) -> list[MethodUnit]:
        """Methods of a mix of files and directories.

        Raises:
            FileNotFoundError: a path does not exist.
            ValueError: a file was given that is not a ``.java`` file. Explicit
                paths are never skipped silently: an empty result would look
                like "no debt found".
        """
        units: list[MethodUnit] = []
        for raw in paths:
            path = Path(raw)
            if not path.exists():
                raise FileNotFoundError(f"Path not found: {path}")
            if path.is_dir():
                units.extend(self.from_directory(path))
            elif path.suffix == ".java":
                units.extend(self.from_file(path))
            else:
                raise ValueError(f"Not a .java file or a directory: {path}")
        return units

    def from_diff(self, diff_text: str, root: str | Path) -> list[MethodUnit]:
        """Methods touched by a diff, read from the checked-out new version under ``root``.

        A method counts as touched when a changed line falls anywhere
        between the first line of its Javadoc and its closing brace.
        """
        root = Path(root).resolve()
        units: list[MethodUnit] = []
        for rel_path, change in parse_unified_diff(diff_text).items():
            if not rel_path.endswith(".java") or self.is_excluded(rel_path):
                continue
            file_path = root / rel_path
            if not file_path.is_file():
                continue
            source = file_path.read_text(encoding="utf-8", errors="replace")
            for unit in self.from_source(source, rel_path):
                if change.touches(unit.span_start, unit.end_line):
                    units.append(unit)
        return units
