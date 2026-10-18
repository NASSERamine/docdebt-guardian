"""
src/evaluation/structural.py

Deterministic, LLM-free structural completeness check: does a Javadoc
document exactly the @param / @return / @throws tags that the method's
real signature declares -- no missing tag, no tag referring to
something that does not exist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import javalang


# Signature extraction

@dataclass
class MethodSignature:
    param_names: list[str]
    is_void: bool
    is_constructor: bool
    throws: list[str] = field(default_factory=list)


def extract_signature_from_method_code(method_code: str) -> MethodSignature | None:
    """Parse a single method's source (with or without a surrounding
    class wrapper) and return its real signature. Returns None if the
    code cannot be parsed at all.

    Tries the wrapped form first (an isolated method/constructor body
    is the overwhelmingly common input shape), then falls back to
    parsing the code as-is only if wrapping fails. See module docstring
    for why a text-search heuristic is not used to decide this upfront.
    """
    if not method_code or not str(method_code).strip():
        return None

    code = str(method_code).strip()

    tree = None
    for candidate in (f"class _Wrapper {{\n{code}\n}}", code):
        try:
            tree = javalang.parse.parse(candidate)
            break
        except Exception:
            continue
    if tree is None:
        return None

    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        return MethodSignature(
            param_names=[p.name for p in node.parameters],
            is_void=node.return_type is None,
            is_constructor=False,
            throws=list(node.throws) if node.throws else [],
        )
    for _, node in tree.filter(javalang.tree.ConstructorDeclaration):
        return MethodSignature(
            param_names=[p.name for p in node.parameters],
            is_void=True,
            is_constructor=True,
            throws=list(node.throws) if node.throws else [],
        )
    return None


def simple_exception_name(exception_name: str) -> str:
    """'java.io.IOException' -> 'IOException'. Matching by simple name
    tolerates the fully-qualified vs bare conventions seen across real
    codebases without treating the difference itself as a defect."""
    return exception_name.rsplit(".", 1)[-1]


# Javadoc parsing

@dataclass
class ParsedJavadoc:
    description: str
    params: dict[str, str] = field(default_factory=dict)
    return_desc: str | None = None
    throws: dict[str, str] = field(default_factory=dict)


def parse_javadoc(doc_text: str) -> ParsedJavadoc:
    if not isinstance(doc_text, str) or not doc_text.strip():
        return ParsedJavadoc(description="")

    text = re.sub(r"^\s*/\*\*|\*/\s*$", "", doc_text.strip())
    lines = [re.sub(r"^\s*\*\s?", "", line) for line in text.splitlines()]
    text = "\n".join(lines).strip()

    tag_start = re.search(r"@(param|return|throws|exception)\b", text)
    if tag_start:
        description = text[:tag_start.start()].strip()
        tag_block = text[tag_start.start():]
    else:
        description = text.strip()
        tag_block = ""

    params = {
        m.group(1): m.group(2).strip()
        for m in re.finditer(r"@param\s+(\w+)\s*(.*?)(?=@\w+|\Z)", tag_block, re.DOTALL)
    }
    return_match = re.search(r"@return\s*(.*?)(?=@\w+|\Z)", tag_block, re.DOTALL)
    return_desc = return_match.group(1).strip() if return_match else None
    throws = {
        simple_exception_name(m.group(1)): m.group(2).strip()
        for m in re.finditer(r"@(?:throws|exception)\s+(\S+)\s*(.*?)(?=@\w+|\Z)", tag_block, re.DOTALL)
    }

    return ParsedJavadoc(description=description, params=params, return_desc=return_desc, throws=throws)


# Structural verification

@dataclass
class StructuralVerdict:
    resolved: bool
    missing_params: list[str] = field(default_factory=list)
    phantom_params: list[str] = field(default_factory=list)
    missing_return: bool = False
    missing_throws: list[str] = field(default_factory=list)
    phantom_throws: list[str] = field(default_factory=list)


def verify_structural(generated_doc: str, signature: MethodSignature) -> StructuralVerdict:
    parsed = parse_javadoc(generated_doc)

    real_params = set(signature.param_names)
    doc_params = set(parsed.params.keys())
    missing_params = sorted(real_params - doc_params)
    phantom_params = sorted(doc_params - real_params)

    missing_return = (
        (not signature.is_void) and (not signature.is_constructor) and parsed.return_desc is None
    )

    real_throws = {simple_exception_name(t) for t in signature.throws}
    doc_throws = set(parsed.throws.keys())
    missing_throws = sorted(real_throws - doc_throws)
    phantom_throws = sorted(doc_throws - real_throws)

    resolved = not (missing_params or phantom_params or missing_return or missing_throws)

    return StructuralVerdict(
        resolved=resolved,
        missing_params=missing_params,
        phantom_params=phantom_params,
        missing_return=missing_return,
        missing_throws=missing_throws,
        phantom_throws=phantom_throws,
    )


def evaluate_structural_column(df, generated_col: str, code_col: str = "method_code") -> dict:
    """Evaluate structural completeness for one (model, strategy) result
    column. `df` must have `code_col` (Java source) and `generated_col`
    (generated Javadoc). Returns the per-row detail table plus the
    aggregate resolution rate and average gap counts."""
    import pandas as pd

    signature_cache: dict[str, MethodSignature | None] = {}
    rows = []
    for _, row in df.iterrows():
        code = row[code_col]
        if code not in signature_cache:
            signature_cache[code] = extract_signature_from_method_code(code)
        signature = signature_cache[code]

        if signature is None:
            rows.append({"resolved": None, "n_missing": None, "n_phantom": None, "status": "sig_parse_failed"})
            continue

        generated = row[generated_col]
        if pd.isna(generated) or not str(generated).strip():
            rows.append({
                "resolved": False, "n_missing": len(signature.param_names),
                "n_phantom": 0, "status": "empty_generation",
            })
            continue

        verdict = verify_structural(generated, signature)
        rows.append({
            "resolved": verdict.resolved,
            "n_missing": len(verdict.missing_params) + len(verdict.missing_throws) + int(verdict.missing_return),
            "n_phantom": len(verdict.phantom_params) + len(verdict.phantom_throws),
            "status": "ok",
        })

    result = pd.DataFrame(rows)
    valid = result[result["status"] == "ok"]
    return {
        "n_total": len(result),
        "n_excluded": len(result) - len(valid),
        "structural_resolved_pct": round(valid["resolved"].mean(), 6) if len(valid) else float("nan"),
        "avg_missing": round(valid["n_missing"].mean(), 6) if len(valid) else float("nan"),
        "avg_phantom": round(valid["n_phantom"].mean(), 6) if len(valid) else float("nan"),
        "detail": result,
    }