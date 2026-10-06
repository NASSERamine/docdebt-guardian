"""Javadoc text parsing.

``parse_javadoc`` is ported unchanged from the research code
(``src/evaluation/structural.py`` of *Rewriting the Narrative*) so that
structural results stay identical to the ones reported in the paper.
Changing its regular expressions would silently change the Auditor's
verdicts on the benchmark; any change must be re-validated there first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TAG_START = re.compile(r"@(param|return|throws|exception)\b")
_PARAM = re.compile(r"@param\s+(\w+)\s*(.*?)(?=@\w+|\Z)", re.DOTALL)
_RETURN = re.compile(r"@return\s*(.*?)(?=@\w+|\Z)", re.DOTALL)
_THROWS = re.compile(r"@(?:throws|exception)\s+(\S+)\s*(.*?)(?=@\w+|\Z)", re.DOTALL)


@dataclass
class ParsedJavadoc:
    """A Javadoc split into its checkable fields.

    ``params`` maps each documented parameter name to its description.
    When a name is documented twice, the last description wins (same as
    the research implementation); ``param_order`` keeps every occurrence
    so duplicates can still be reported.
    """

    description: str
    params: dict[str, str] = field(default_factory=dict)
    return_desc: str | None = None
    throws: dict[str, str] = field(default_factory=dict)
    param_order: list[str] = field(default_factory=list)


def simple_exception_name(exception_name: str) -> str:
    """``'java.io.IOException'`` -> ``'IOException'``.

    Matching exceptions by simple name tolerates both the fully qualified
    and the bare convention without treating the difference as a defect.
    """
    return exception_name.rsplit(".", 1)[-1]


def strip_comment_markers(doc_text: str) -> str:
    """Remove ``/**``, ``*/`` and the leading ``*`` of every line."""
    text = re.sub(r"^\s*/\*\*|\*/\s*$", "", doc_text.strip())
    lines = [re.sub(r"^\s*\*\s?", "", line) for line in text.splitlines()]
    return "\n".join(lines).strip()


def parse_javadoc(doc_text: str | None) -> ParsedJavadoc:
    """Split a raw Javadoc block into description and tags.

    Accepts the block with or without its ``/** ... */`` markers. Returns
    an empty ``ParsedJavadoc`` for ``None`` or blank input.
    """
    if not isinstance(doc_text, str) or not doc_text.strip():
        return ParsedJavadoc(description="")

    text = strip_comment_markers(doc_text)

    tag_start = _TAG_START.search(text)
    if tag_start:
        description = text[: tag_start.start()].strip()
        tag_block = text[tag_start.start() :]
    else:
        description = text.strip()
        tag_block = ""

    param_matches = list(_PARAM.finditer(tag_block))
    params = {m.group(1): m.group(2).strip() for m in param_matches}
    return_match = _RETURN.search(tag_block)
    return_desc = return_match.group(1).strip() if return_match else None
    throws = {simple_exception_name(m.group(1)): m.group(2).strip() for m in _THROWS.finditer(tag_block)}
    return ParsedJavadoc(
        description=description,
        params=params,
        return_desc=return_desc,
        throws=throws,
        param_order=[m.group(1) for m in param_matches],
    )


def normalize_for_matching(text: str) -> str:
    """Collapse whitespace so quotes can be matched regardless of layout."""
    return re.sub(r"\s+", " ", text).strip()


def javadoc_plain_text(doc_text: str | None) -> str:
    """Javadoc without comment markers and with whitespace collapsed."""
    if not doc_text:
        return ""
    return normalize_for_matching(strip_comment_markers(doc_text))
