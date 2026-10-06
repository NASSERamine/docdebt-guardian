"""Turn findings into an edit plan: which fields may change."""

from __future__ import annotations

import re

from docdebt.core.javadoc import simple_exception_name
from docdebt.core.javadoc_edit import (
    DESCRIPTION,
    PARAM,
    RETURN,
    THROWS,
    EditPlan,
    FieldRef,
    normalize_text,
    parse_layout,
)
from docdebt.core.models import Finding, FindingKind, MethodUnit


def field_ref_from_text(text: str | None, unit: MethodUnit) -> FieldRef:
    """Field named by a model's ``FIELD:`` line ("@param x", "@return", ...).

    Anything unrecognised, or a ``@param`` naming no real parameter, falls
    back to the description.
    """
    value = (text or "").strip().lower().lstrip("@").strip()
    if value.startswith("param"):
        m = re.match(r"param\s+(\w+)", (text or "").strip().lstrip("@"), re.IGNORECASE)
        if m and m.group(1) in unit.signature.param_names:
            return FieldRef(PARAM, m.group(1))
    elif value.startswith("return"):
        return FieldRef(RETURN)
    elif value.startswith(("throws", "exception")):
        m = re.match(r"(?:throws|exception)\s+(\S+)", (text or "").strip().lstrip("@"), re.IGNORECASE)
        if m:
            return FieldRef(THROWS, simple_exception_name(m.group(1)))
    return FieldRef(DESCRIPTION)


def ref_for_finding(unit: MethodUnit, finding: Finding) -> FieldRef:
    """The field a semantic finding is about.

    The ``FIELD:`` label written by a model is unreliable: a 7B model quoted a
    sentence of the description and labelled it ``@return``. The quoted claim,
    however, was validated word for word against the Javadoc, so the field is
    located from it. The model's label only breaks ties (when several fields
    contain the quote) and is the fallback when no field contains it.
    """
    labelled = field_ref_from_text(finding.field, unit)
    if not finding.quoted_doc:
        return labelled
    quote = normalize_text(finding.quoted_doc)
    holders = [
        block.ref
        for block in parse_layout(unit.javadoc or "").blocks
        if block.ref is not None and quote and quote in normalize_text(" ".join(block.content))
    ]
    if not holders or labelled in holders:
        return labelled
    return holders[0]


def _structural_ref(finding: Finding) -> FieldRef | None:
    parts = (finding.field or "").split()
    name = parts[1] if len(parts) > 1 else None
    if finding.rule in ("missing-param", "unknown-param") and name:
        return FieldRef(PARAM, name)
    if finding.rule == "missing-return":
        return FieldRef(RETURN)
    if finding.rule == "missing-throws" and name:
        return FieldRef(THROWS, name)
    return None


def plan_from_findings(unit: MethodUnit, findings: list[Finding]) -> EditPlan:
    """Edit plan for the debt findings (warnings are ignored).

    * ``missing-*`` structural rules add a field;
    * ``unknown-param`` removes the documented parameter that does not exist;
    * a semantic finding replaces the field it disputes.
    """
    plan = EditPlan()
    for finding in findings:
        if not finding.is_debt:
            continue
        if finding.kind == FindingKind.STRUCTURAL:
            ref = _structural_ref(finding)
            if ref is None:
                continue
            (plan.remove if finding.rule == "unknown-param" else plan.add).add(ref)
        else:
            plan.replace.add(ref_for_finding(unit, finding))
    return plan.normalize()
