"""JSON-safe (de)serialization of core objects, for crossing a checkpoint boundary.

LangGraph's checkpointer stores state as msgpack and warns that pickling an
arbitrary dataclass across a checkpoint will be rejected in a future version
(confirmed against langgraph 1.2). Every object that needs to survive an
``interrupt`` — a :class:`MethodUnit`, a :class:`Finding` — is therefore
turned into a plain ``dict`` of strings, numbers and lists before it enters
graph state, and rebuilt on the other side. This module is the only place
that knows both directions of that conversion.
"""

from __future__ import annotations

from docdebt.core.models import (
    Finding,
    FindingKind,
    MethodSignature,
    MethodUnit,
    Parameter,
    Severity,
    Visibility,
)


def unit_to_dict(unit: MethodUnit) -> dict:
    """A :class:`MethodUnit` as a JSON-safe dict. Inverse of :func:`unit_from_dict`."""
    sig = unit.signature
    return {
        "file_path": unit.file_path,
        "class_name": unit.class_name,
        "signature": {
            "name": sig.name,
            "parameters": [{"name": p.name, "type": p.type} for p in sig.parameters],
            "return_type": sig.return_type,
            "throws": list(sig.throws),
            "is_constructor": sig.is_constructor,
        },
        "code": unit.code,
        "javadoc": unit.javadoc,
        "start_line": unit.start_line,
        "end_line": unit.end_line,
        "javadoc_start_line": unit.javadoc_start_line,
        "visibility": unit.visibility.value,
        "has_body": unit.has_body,
        "syntax_ok": unit.syntax_ok,
        "indent": unit.indent,
    }


def unit_from_dict(data: dict) -> MethodUnit:
    """Inverse of :func:`unit_to_dict`."""
    sig_data = data["signature"]
    signature = MethodSignature(
        name=sig_data["name"],
        parameters=tuple(Parameter(p["name"], p["type"]) for p in sig_data["parameters"]),
        return_type=sig_data["return_type"],
        throws=tuple(sig_data["throws"]),
        is_constructor=sig_data["is_constructor"],
    )
    return MethodUnit(
        file_path=data["file_path"],
        class_name=data["class_name"],
        signature=signature,
        code=data["code"],
        javadoc=data["javadoc"],
        start_line=data["start_line"],
        end_line=data["end_line"],
        javadoc_start_line=data["javadoc_start_line"],
        visibility=Visibility(data["visibility"]),
        has_body=data["has_body"],
        syntax_ok=data["syntax_ok"],
        indent=data["indent"],
    )


def finding_from_dict(data: dict) -> Finding:
    """Inverse of :meth:`Finding.to_dict`."""
    return Finding(
        kind=FindingKind(data["kind"]),
        severity=Severity(data["severity"]),
        rule=data["rule"],
        message=data["message"],
        method_id=data["method_id"],
        field=data.get("field"),
        line=data.get("line"),
        quoted_doc=data.get("quoted_doc"),
        quoted_code=data.get("quoted_code"),
        confidence=data.get("confidence"),
        model=data.get("model"),
        prompt_version=data.get("prompt_version"),
        extra=data.get("extra") or {},
    )
