"""Parsing of raw model outputs.

``parse_binary_verdict`` is the research implementation, unchanged:
whole-word match, last occurrence wins, so a chain-of-thought answer that
mentions both words while reasoning is judged on its final statement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEBT_WORD = "INCOMPLETE"
CLEAN_WORD = "COMPLETE"


def parse_binary_verdict(
    raw_text: str, positive_word: str = DEBT_WORD, negative_word: str = CLEAN_WORD
) -> bool | None:
    """``True`` for debt, ``False`` for clean, ``None`` when neither word appears.

    Callers must handle ``None`` explicitly: it is an abstention, never a
    default verdict.
    """
    if not isinstance(raw_text, str):
        return None
    text = raw_text.strip().upper()
    pattern = rf"\b{re.escape(positive_word)}\b|\b{re.escape(negative_word)}\b"
    tokens = re.findall(pattern, text)
    if not tokens:
        return None
    return tokens[-1] == positive_word


@dataclass
class ClaimCitation:
    """One contradicted claim as reported by the model (not yet validated)."""

    claim: str
    code: str
    reason: str = ""
    field: str = ""


_FIELD_KEYS = ("CLAIM", "CODE", "REASON", "FIELD")
_LINE = re.compile(r"^\s*[*\-]*\s*(CLAIM|CODE|REASON|FIELD)\s*:\s*(.*)$", re.IGNORECASE)


def _strip_quotes(text: str) -> str:
    text = text.strip()
    for pair in ('""', "''", "``"):
        if len(text) >= 2 and text[0] == pair[0] and text[-1] == pair[1]:
            text = text[1:-1].strip()
    return text


def parse_claim_citations(raw_text: str) -> list[ClaimCitation] | None:
    """Parse the line-based output of the citation prompt.

    A line-based format is used instead of JSON on purpose: in the research
    runs, Javadoc braces (``{@link ...}``, ``{@code ...}``) regularly broke
    JSON produced by local models.

    Returns ``[]`` when the model answered ``NONE``, ``None`` when the output
    follows neither format (treated as a failed explanation).
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    text = raw_text.strip().strip("`").strip()

    claims: list[ClaimCitation] = []
    current: dict[str, str] = {}
    last_key: str | None = None

    def flush() -> None:
        if current.get("CLAIM") and current.get("CODE"):
            claims.append(
                ClaimCitation(
                    claim=_strip_quotes(current["CLAIM"]),
                    code=_strip_quotes(current["CODE"]),
                    reason=current.get("REASON", "").strip(),
                    field=current.get("FIELD", "").strip(),
                )
            )

    for line in text.splitlines():
        match = _LINE.match(line)
        if match:
            key = match.group(1).upper()
            if key == "CLAIM" and current:
                flush()
                current = {}
            current[key] = match.group(2)
            last_key = key
        elif line.strip() == "---":
            flush()
            current, last_key = {}, None
        elif last_key in ("CLAIM", "CODE") and line.strip():
            current[last_key] += " " + line.strip()  # quote continued on the next line
    flush()

    if claims:
        return claims
    if re.fullmatch(r"\W*NONE\W*", text.upper()):
        return []
    return None
