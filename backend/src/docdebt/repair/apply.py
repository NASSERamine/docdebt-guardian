"""Show a repair as a diff and write it into a source file."""

from __future__ import annotations

from collections.abc import Sequence
from difflib import unified_diff
from pathlib import Path

from docdebt.core.models import MethodUnit
from docdebt.repair.repairer import RepairResult, RepairStatus


def javadoc_diff(unit: MethodUnit, result: RepairResult) -> str:
    """Unified diff of the Javadoc before and after a verified repair."""
    if result.repaired is None:
        return ""
    return "\n".join(
        unified_diff(
            result.original.splitlines(),
            result.repaired.splitlines(),
            f"{unit.file_path} (current)",
            f"{unit.file_path} (repaired)",
            lineterm="",
            n=2,
        )
    )


def apply_repairs_to_text(
    text: str, repairs: Sequence[tuple[MethodUnit, RepairResult]]
) -> tuple[str, int, list[str]]:
    """Replace the Javadoc of each verified repair directly in ``text``.

    A repair is skipped, with a message, if the lines at its recorded
    position no longer match the Javadoc that was analysed (the source
    changed since). This is the shared core behind both
    :func:`apply_repairs_to_file` (a local checkout) and the GitHub App's
    single-file commit-on-accept flow (``docdebt.service.review_flow``),
    which fetches and writes the same text through the GitHub Contents API
    instead of the filesystem.

    Returns:
        The new text (unchanged from the input if nothing was written), the
        number of Javadocs written, and the skip messages.
    """
    crlf = "\r\n" in text
    lines = text.split("\n")
    written, skipped = 0, []
    ordered = sorted(
        (r for r in repairs if r[1].status == RepairStatus.VERIFIED and r[0].javadoc_start_line),
        key=lambda pair: pair[0].javadoc_start_line or 0,
        reverse=True,
    )
    for unit, result in ordered:
        start, end = (unit.javadoc_start_line or 1) - 1, unit.javadoc_end_line or 0
        current = "\n".join(lines[start:end])
        indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
        if current != indent + result.original:
            skipped.append(f"{unit.file_path}:{unit.start_line} {unit.name}: file changed, not written")
            continue
        new_lines = result.repaired.split("\n") if result.repaired else []
        new_lines[0] = indent + new_lines[0]
        if crlf:
            new_lines = [ln if ln.endswith("\r") else ln + "\r" for ln in new_lines]
        lines[start:end] = new_lines
        written += 1
    return "\n".join(lines), written, skipped


def apply_repairs_to_file(
    path: str | Path, repairs: Sequence[tuple[MethodUnit, RepairResult]]
) -> tuple[int, list[str]]:
    """Replace the Javadoc of each verified repair in the file at ``path``.

    A repair is skipped, with a message, if the lines in the file no longer
    match the Javadoc that was analysed (the file changed in the meantime).

    Returns:
        The number of Javadocs written and the skip messages.
    """
    path = Path(path)
    new_text, written, skipped = apply_repairs_to_text(path.read_text(encoding="utf-8"), repairs)
    if written:
        path.write_text(new_text, encoding="utf-8")
    return written, skipped
