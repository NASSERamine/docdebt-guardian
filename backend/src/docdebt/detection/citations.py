"""Validation of the quotes a model gives to justify a semantic alert.

A semantic alert is only shown when its two quotes really exist: the
documentation excerpt in the Javadoc and the code excerpt in the method.
This is the guard against hallucinated explanations, in the same spirit
as rejecting phantom parameters in generated Javadoc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docdebt.core.javadoc import javadoc_plain_text, normalize_for_matching
from docdebt.llm.parsing import ClaimCitation

MIN_QUOTE_CHARS = 3


@dataclass
class ValidatedCitation:
    citation: ClaimCitation
    doc_found: bool
    code_found: bool

    @property
    def valid(self) -> bool:
        return self.doc_found and self.code_found


def _normalize_quote(text: str) -> str:
    text = re.sub(r"^\s*/\*\*|\*/\s*$", "", text.strip())
    text = re.sub(r"(^|\n)\s*\*\s?", r"\1", text)
    return normalize_for_matching(text)


def quote_in_javadoc(quote: str, javadoc: str) -> bool:
    """True if ``quote`` appears in the Javadoc, ignoring layout and ``*`` markers."""
    needle = _normalize_quote(quote)
    if len(needle.replace(" ", "")) < MIN_QUOTE_CHARS:
        return False
    return needle in javadoc_plain_text(javadoc)


_STRING_CONCAT = re.compile(r'"((?:[^"\\]|\\.)*)"\s*\+\s*"((?:[^"\\]|\\.)*)"')


def flatten_string_concat(code: str) -> str:
    """Collapse adjacent Java string-literal concatenations: ``"a" + "b"`` -> ``"ab"``.

    A model quoting a multi-line concatenated string (common for long messages)
    naturally reproduces it as one continuous string, which no longer matches
    the source text word for word.
    """
    previous = None
    while previous != code:
        previous = code
        code = _STRING_CONCAT.sub(r'"\1\2"', code)
    return code


def _strip_whitespace(text: str) -> str:
    """All whitespace removed: Java code has no semantic whitespace, so this is
    a safe fallback layer for a needle/haystack match that only cares about a
    contiguous sequence of tokens, not spacing around punctuation."""
    return re.sub(r"\s+", "", text)


def quote_in_code(quote: str, code: str) -> bool:
    """True if ``quote`` appears in the method source, ignoring whitespace layout
    and Java string-literal concatenation (``"a" + "b"``)."""
    needle = normalize_for_matching(quote.strip().strip("`"))
    if len(needle.replace(" ", "")) < MIN_QUOTE_CHARS:
        return False
    haystack = normalize_for_matching(flatten_string_concat(code))
    if needle in haystack or needle.rstrip(";{ ") in haystack:
        return True
    # Fallback: compare with every whitespace character removed, so spacing
    # around punctuation (a model writes "foo( \"x\" )", the source has "foo(\"x\")")
    # no longer breaks an otherwise-exact match.
    return _strip_whitespace(needle) in _strip_whitespace(haystack)


def validate_citations(citations: list[ClaimCitation], javadoc: str, code: str) -> list[ValidatedCitation]:
    return [
        ValidatedCitation(
            citation=c,
            doc_found=quote_in_javadoc(c.claim, javadoc),
            code_found=quote_in_code(c.code, code),
        )
        for c in citations
    ]
