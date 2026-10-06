"""Shared data model for the whole pipeline.

These dataclasses are deliberately plain (no ORM, no pydantic) so the
deterministic core stays dependency-free and trivially testable.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    """How serious a finding is.

    Only ERROR findings count as documentation debt. WARNING findings are
    reported to the developer but never make a method "debt" on their own.
    """

    ERROR = "error"
    WARNING = "warning"


class FindingKind(StrEnum):
    """Which part of the pipeline produced a finding."""

    STRUCTURAL = "structural"  # deterministic Auditor
    SEMANTIC = "semantic"  # LLM Judge, backed by a validated citation


class Visibility(StrEnum):
    PUBLIC = "public"
    PROTECTED = "protected"
    PACKAGE = "package"
    PRIVATE = "private"


@dataclass(frozen=True)
class Parameter:
    """One formal parameter of a method, as declared in the source."""

    name: str
    type: str


@dataclass(frozen=True)
class MethodSignature:
    """The part of a method that its Javadoc contract must match.

    ``return_type`` is ``None`` for constructors. ``throws`` keeps the
    exception names exactly as written (simple or fully qualified).
    """

    name: str
    parameters: tuple[Parameter, ...]
    return_type: str | None
    throws: tuple[str, ...] = ()
    is_constructor: bool = False

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def param_types(self) -> list[str]:
        return [p.type for p in self.parameters]

    @property
    def is_void(self) -> bool:
        """True for ``void`` methods and for constructors (no @return expected)."""
        return self.is_constructor or self.return_type == "void"


@dataclass(frozen=True)
class MethodUnit:
    """A single Java method (or constructor) together with its Javadoc.

    Line numbers are 1-based and inclusive. ``code`` starts at the first
    annotation or modifier and ends at the closing brace (or semicolon for
    bodiless methods); it never includes the Javadoc.
    """

    file_path: str
    class_name: str
    signature: MethodSignature
    code: str
    javadoc: str | None
    start_line: int
    end_line: int
    javadoc_start_line: int | None = None
    visibility: Visibility = Visibility.PACKAGE
    has_body: bool = True
    syntax_ok: bool = True
    indent: str = ""

    @property
    def name(self) -> str:
        return self.signature.name

    @property
    def javadoc_end_line(self) -> int | None:
        """Last line of the Javadoc (1-based, inclusive), when the method has one."""
        if self.javadoc_start_line is None or not self.javadoc:
            return None
        return self.javadoc_start_line + self.javadoc.count("\n")

    @property
    def id(self) -> str:
        """Stable identifier: file, enclosing class, name and parameter types.

        Parameter types are included so overloads get distinct ids.
        """
        types = ",".join(self.signature.param_types)
        return f"{self.file_path}::{self.class_name}.{self.name}({types})"

    @property
    def has_javadoc(self) -> bool:
        return bool(self.javadoc and self.javadoc.strip())

    @property
    def span_start(self) -> int:
        """First line owned by this method, Javadoc included."""
        return self.javadoc_start_line or self.start_line


@dataclass
class Finding:
    """One problem found in a method's Javadoc.

    Structural findings carry a ``rule`` id such as ``missing-param``.
    Semantic findings carry the exact documentation excerpt they dispute
    (``quoted_doc``) and the code excerpt that contradicts it
    (``quoted_code``); both are validated against the real text before a
    semantic finding is ever emitted.
    """

    kind: FindingKind
    severity: Severity
    rule: str
    message: str
    method_id: str
    field: str | None = None
    line: int | None = None
    quoted_doc: str | None = None
    quoted_code: str | None = None
    confidence: float | None = None
    model: str | None = None
    prompt_version: str | None = None
    extra: dict[str, object] = dataclasses.field(default_factory=dict)

    @property
    def is_debt(self) -> bool:
        return self.severity == Severity.ERROR

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "kind": self.kind.value,
            "severity": self.severity.value,
            "rule": self.rule,
            "message": self.message,
            "method_id": self.method_id,
            "field": self.field,
            "line": self.line,
            "quoted_doc": self.quoted_doc,
            "quoted_code": self.quoted_code,
            "confidence": self.confidence,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }
        if self.extra:
            data["extra"] = self.extra
        return data
