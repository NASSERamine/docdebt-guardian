"""Verifier: check a merged repair before it is ever shown to a developer.

Every check works on the *merged* Javadoc, so a repair that fails any of
them is never proposed:

* structure: the Auditor finds no error left (no missing or phantom
  ``@param``, no missing ``@return``, no undocumented declared exception);
* completeness: every field the plan needed from the model was provided,
  with a non-empty text;
* citation: a replaced field no longer contains the claim the Judge cited;
* preservation: every field outside the plan is identical to the original,
  compared with the research ``parse_javadoc`` and block by block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docdebt.core.auditor import AuditorConfig, audit_javadoc
from docdebt.core.javadoc import javadoc_plain_text, normalize_for_matching, parse_javadoc
from docdebt.core.javadoc_edit import (
    DESCRIPTION,
    OTHER,
    PARAM,
    THROWS,
    DocLayout,
    EditPlan,
    FieldRef,
    MergeResult,
    field_text,
    normalize_text,
    parse_layout,
)
from docdebt.core.models import Finding, FindingKind, MethodUnit
from docdebt.repair.plan import ref_for_finding

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "in",
        "on",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "and",
        "or",
        "if",
        "it",
        "its",
        "with",
        "as",
        "by",
        "will",
        "not",
        "no",
        "which",
        "when",
        "at",
        "from",
        "than",
        "then",
        "also",
        "can",
        "may",
        "must",
        "should",
        "would",
        "could",
        "has",
        "have",
        "had",
        "but",
        "so",
        "such",
        "into",
        "out",
        "up",
        "down",
        "over",
        "under",
        "within",
        "without",
    }
)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
MIN_REWRITE_SCOPE_WORDS = 3
REWRITE_SCOPE_COVERAGE = 0.5


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)} - _STOPWORDS


def rewrite_scope_issue(original_field: str, quoted_claim: str | None, replacement_field: str) -> Issue | None:
    """Catch a field ``replace`` that rewrites far more than the disputed claim.

    A repair should change only the part of a field that is actually wrong.
    This compares the words of the original field, minus the words of the
    disputed claim (the only part the model was asked to fix), against the
    model's replacement: if most of that untouched content is gone, the model
    rewrote the whole field rather than correcting the claim.

    This is a word-overlap heuristic, not a meaning check: a field that is
    reworded but keeps its original content passes; the case it catches is a
    field replaced by unrelated prose. Skipped when there is too little
    remaining content to judge (the claim already covers most of the field,
    as for a short field like "Returns the maximum of the two values.").
    """
    remainder = _content_words(original_field) - _content_words(quoted_claim or "")
    if len(remainder) < MIN_REWRITE_SCOPE_WORDS:
        return None
    replacement_lower = replacement_field.lower()
    covered = sum(1 for w in remainder if re.search(rf"\b{re.escape(w)}\b", replacement_lower))
    coverage = covered / len(remainder)
    if coverage >= REWRITE_SCOPE_COVERAGE:
        return None
    return Issue(
        "rewrite_beyond_claim",
        None,
        "Keep the rest of the field as close as possible to the original wording; "
        "change only the part that is actually wrong.",
    )


@dataclass(frozen=True)
class Issue:
    """A reason a merged repair was rejected. ``message`` is sent back to the model on retry."""

    code: str
    label: str | None
    message: str


def _missing_message(ref: FieldRef) -> str:
    if ref.kind == PARAM:
        return f"Document the parameter '{ref.name}' with a @param tag."
    if ref.kind == THROWS:
        return f"Document the declared exception '{ref.name}' with a @throws tag."
    if ref.kind == DESCRIPTION:
        return "Include a corrected description."
    return f"Include a {ref.label} tag."


def check_preservation(original: str, merged: str, plan: EditPlan) -> list[Issue]:
    """Issues for every field outside the plan that differs between the two texts."""
    issues: list[Issue] = []
    old, new = parse_layout(original), parse_layout(merged)

    def index(layout: DocLayout) -> dict[tuple[str, str | None], list[str]]:
        table: dict[tuple[str, str | None], list[str]] = {}
        for block in layout.blocks:
            table.setdefault((block.kind, block.name), []).append(normalize_text(" ".join(block.content)))
        return table

    old_index, new_index = index(old), index(new)
    touched = {(r.kind, r.name) for r in plan.touched}
    for key, texts in old_index.items():
        if key in touched:
            continue
        if new_index.get(key) != texts:
            label = FieldRef(key[0], key[1]).label if key[0] != OTHER else f"@{key[1]}"
            issues.append(Issue("preservation", label, f"The field {label} must stay unchanged."))
    for key in new_index:
        if key not in old_index and key not in touched:
            issues.append(Issue("unexpected_field", None, f"Unexpected new field {key}."))

    a, b = parse_javadoc(original), parse_javadoc(merged)
    # The description is checked block by block above: parse_javadoc's "description" also
    # swallows any tag written before the first @param, which moves when a tag is added.
    checks: list[tuple[str, str | None, str, str | None]] = []
    checks += [(PARAM, n, a.params[n], b.params.get(n)) for n in a.params]
    checks += [(THROWS, n, a.throws[n], b.throws.get(n)) for n in a.throws]
    for kind, name, before, after in checks:
        if (kind, name) in touched:
            continue
        if normalize_text(before) != normalize_text(after or ""):
            label = FieldRef(kind, name).label
            issues.append(Issue("preservation", label, f"The field {label} must stay unchanged."))
    if (a.return_desc is None) != (b.return_desc is None) and ("return", None) not in touched:
        issues.append(Issue("preservation", "@return", "The @return tag must stay unchanged."))
    return issues


class Verifier:
    def __init__(self, auditor_config: AuditorConfig | None = None) -> None:
        self.auditor_config = auditor_config or AuditorConfig()

    def verify(
        self,
        unit: MethodUnit,
        original: str,
        plan: EditPlan,
        findings: list[Finding],
        merge: MergeResult,
    ) -> list[Issue]:
        issues: list[Issue] = []
        reported: set[str] = set()

        for ref in merge.missing_in_model:
            issues.append(Issue("missing_in_output", ref.label, _missing_message(ref)))
            reported.add(ref.label)

        layout = parse_layout(merge.text, unit.indent)
        by_ref = {b.ref: b for b in layout.blocks if b.ref is not None}
        for change in merge.changes:
            if change.action in ("add", "replace"):
                block = by_ref.get(change.ref)
                if block is None or not field_text(block):
                    issues.append(Issue("empty_field", change.ref.label, f"{change.ref.label} must not be empty."))
                    reported.add(change.ref.label)

        for finding in audit_javadoc(merge.text, unit.signature, unit.id, config=self.auditor_config):
            if finding.is_debt and (finding.field or "") not in reported:
                issues.append(Issue(f"structural:{finding.rule}", finding.field, finding.message))

        replaced = {c.ref for c in merge.changes if c.action == "replace"}
        for change in merge.changes:
            if change.action == "replace" and change.before and change.after:
                claim = next((f.quoted_doc for f in findings if ref_for_finding(unit, f) == change.ref), None)
                scope_issue = rewrite_scope_issue(change.before, claim, change.after)
                if scope_issue:
                    issues.append(Issue(scope_issue.code, change.ref.label, scope_issue.message))

        plain = javadoc_plain_text(merge.text)
        for finding in findings:
            if finding.kind != FindingKind.SEMANTIC or not finding.quoted_doc:
                continue
            ref = ref_for_finding(unit, finding)
            claim = normalize_for_matching(finding.quoted_doc)
            if ref in replaced and claim and claim in plain:
                issues.append(
                    Issue(
                        "claim_still_present",
                        ref.label,
                        f'The statement "{finding.quoted_doc}" is contradicted by the code '
                        f'("{finding.quoted_code}") and must be corrected.',
                    )
                )

        issues.extend(check_preservation(original, merge.text, plan))
        return issues
