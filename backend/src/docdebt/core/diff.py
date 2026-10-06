"""Unified diff reading.

Only the *new* side of the diff matters: the pipeline analyses the code
as it will be after the pull request is merged. For every changed file we
collect the new-file line numbers that were added or modified, plus the
line where a deletion happened, since removing lines inside a method also
changes that method.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass
class FileChange:
    """Changed lines of one file, on the new side of the diff."""

    path: str
    changed_lines: set[int] = field(default_factory=set)
    is_new_file: bool = False

    def touches(self, start: int, end: int) -> bool:
        """True if any changed line falls within ``[start, end]`` (1-based, inclusive)."""
        return any(start <= line <= end for line in self.changed_lines)


def _clean_path(raw: str) -> str | None:
    raw = raw.strip().split("\t", 1)[0]
    if raw == "/dev/null":
        return None
    if raw.startswith(("a/", "b/")):
        raw = raw[2:]
    return raw


def parse_unified_diff(diff_text: str) -> dict[str, FileChange]:
    """Map each file path (new side) to its changed line numbers.

    Deleted files are ignored: there is nothing left to document.
    Works with ``git diff`` output and with the diff GitHub serves for a
    pull request.
    """
    changes: dict[str, FileChange] = {}
    current: FileChange | None = None
    old_path: str | None = None
    new_line = 0
    in_hunk = False

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            current, old_path, in_hunk = None, None, False
            continue
        if line.startswith("--- "):
            old_path = _clean_path(line[4:])
            in_hunk = False
            continue
        if line.startswith("+++ "):
            path = _clean_path(line[4:])
            in_hunk = False
            if path is None:  # file deleted
                current = None
                continue
            current = changes.setdefault(path, FileChange(path=path))
            current.is_new_file = old_path is None
            continue

        match = _HUNK.match(line)
        if match:
            new_line = int(match.group(1))
            in_hunk = current is not None
            continue
        if not in_hunk or current is None:
            continue

        if line.startswith("+"):
            current.changed_lines.add(new_line)
            new_line += 1
        elif line.startswith("-"):
            current.changed_lines.add(max(new_line, 1))
        elif line.startswith("\\"):  # "\ No newline at end of file"
            continue
        else:  # context line
            new_line += 1

    return changes
