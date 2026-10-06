"""Auditor: deterministic structural check of a Javadoc against its method signature.

ERROR rules reproduce exactly the research ``verify_structural`` check
(a Javadoc is structurally resolved when no parameter is missing, no
documented parameter is unknown, ``@return`` is present when required and,
with ``require_throws``, every declared exception is documented). WARNING
rules add developer-facing hygiene checks that never count as debt, so
they cannot change benchmark results.

Rule ids (stable, used in reports, suppressions and tests):

=====================  ========  ==============================================
rule                   severity  meaning
=====================  ========  ==============================================
missing-param          error     a real parameter has no ``@param`` tag
unknown-param          error     ``@param`` names no real parameter (phantom)
missing-return         error     non-void, non-constructor method, no ``@return``
missing-throws         error     declared exception has no ``@throws`` tag
duplicate-param        warning   the same parameter is documented twice
empty-param            warning   ``@param name`` with no description
empty-return           warning   ``@return`` with no description
return-on-void         warning   ``@return`` on a void method or constructor
=====================  ========  ==============================================
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from docdebt.core.javadoc import parse_javadoc, simple_exception_name
from docdebt.core.models import Finding, FindingKind, MethodSignature, MethodUnit, Severity


@dataclass
class AuditorConfig:
    """Auditor switches.

    Attributes:
        require_throws: report undocumented declared exceptions as errors
            (matches the research structural check).
        hygiene_warnings: emit the WARNING rules.
    """

    require_throws: bool = True
    hygiene_warnings: bool = True


@dataclass
class AuditReport:
    """All structural findings for one method."""

    unit: MethodUnit
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def has_structural_debt(self) -> bool:
        return bool(self.errors)


def _tag_line(javadoc: str, pattern: str, base_line: int | None) -> int | None:
    """Absolute line of the first Javadoc line matching ``pattern``, if known."""
    if base_line is None:
        return None
    regex = re.compile(pattern)
    for offset, line in enumerate(javadoc.splitlines()):
        if regex.search(line):
            return base_line + offset
    return base_line


def audit_javadoc(
    javadoc: str | None,
    signature: MethodSignature,
    method_id: str,
    *,
    base_line: int | None = None,
    config: AuditorConfig | None = None,
) -> list[Finding]:
    """Structural findings for a Javadoc text and a signature.

    Works on any Javadoc text, not only the one found in the source, so the
    repair Verifier can reuse it on a proposed fix.

    Args:
        javadoc: the raw ``/** ... */`` block (``None`` is treated as empty).
        signature: the method signature the Javadoc must match.
        method_id: id copied into every finding.
        base_line: absolute line of the Javadoc's first line, used to place
            findings; ``None`` leaves line numbers empty.
        config: rule switches.
    """
    cfg = config or AuditorConfig()
    doc = javadoc or ""
    parsed = parse_javadoc(doc)
    findings: list[Finding] = []

    def add(rule: str, severity: Severity, message: str, fld: str, line: int | None) -> None:
        findings.append(
            Finding(
                kind=FindingKind.STRUCTURAL,
                severity=severity,
                rule=rule,
                message=message,
                method_id=method_id,
                field=fld,
                line=line,
            )
        )

    real_params = signature.param_names
    doc_params = set(parsed.params)

    for name in real_params:
        if name not in doc_params:
            add(
                "missing-param",
                Severity.ERROR,
                f"Parameter '{name}' has no @param tag.",
                f"@param {name}",
                base_line,
            )
    for name in sorted(doc_params - set(real_params)):
        add(
            "unknown-param",
            Severity.ERROR,
            f"@param '{name}' does not match any parameter of {signature.name}().",
            f"@param {name}",
            _tag_line(doc, rf"@param\s+{re.escape(name)}\b", base_line),
        )

    if not signature.is_void and parsed.return_desc is None:
        add(
            "missing-return",
            Severity.ERROR,
            f"{signature.name}() returns {signature.return_type} but has no @return tag.",
            "@return",
            base_line,
        )

    if cfg.require_throws:
        documented = set(parsed.throws)
        for exc in signature.throws:
            simple = simple_exception_name(exc)
            if simple not in documented:
                add(
                    "missing-throws",
                    Severity.ERROR,
                    f"Declared exception '{simple}' has no @throws tag.",
                    f"@throws {simple}",
                    base_line,
                )

    if cfg.hygiene_warnings:
        for name, count in Counter(parsed.param_order).items():
            if count > 1:
                add(
                    "duplicate-param",
                    Severity.WARNING,
                    f"Parameter '{name}' is documented {count} times.",
                    f"@param {name}",
                    _tag_line(doc, rf"@param\s+{re.escape(name)}\b", base_line),
                )
        for name, desc in parsed.params.items():
            if name in real_params and not desc:
                add(
                    "empty-param",
                    Severity.WARNING,
                    f"@param '{name}' has no description.",
                    f"@param {name}",
                    _tag_line(doc, rf"@param\s+{re.escape(name)}\b", base_line),
                )
        if parsed.return_desc is not None:
            if signature.is_void:
                add(
                    "return-on-void",
                    Severity.WARNING,
                    f"@return is documented but {signature.name}() returns nothing.",
                    "@return",
                    _tag_line(doc, r"@return\b", base_line),
                )
            elif not parsed.return_desc:
                add(
                    "empty-return",
                    Severity.WARNING,
                    "@return has no description.",
                    "@return",
                    _tag_line(doc, r"@return\b", base_line),
                )

    return findings


class Auditor:
    """Runs ``audit_javadoc`` on extracted methods."""

    def __init__(self, config: AuditorConfig | None = None) -> None:
        self.config = config or AuditorConfig()

    def audit(self, unit: MethodUnit) -> AuditReport:
        findings = audit_javadoc(
            unit.javadoc,
            unit.signature,
            unit.id,
            base_line=unit.javadoc_start_line,
            config=self.config,
        )
        return AuditReport(unit=unit, findings=findings)

    def audit_all(self, units: list[MethodUnit]) -> list[AuditReport]:
        return [self.audit(u) for u in units]
