"""Research ``src/mitigation/semantic_triage.py`` (hybrid Replacement, layer 2),
kept as tested; only the offline evaluation (pandas) is left out.

Layer 2 of the hybrid Replacement mechanism: deterministic, zero-token
semantic triage. Inputs are ONLY the stale Javadoc (javadoc_before, as
parsed by javadoc_spans), the layer-1 plan, and the method code. Each
existing field kept (or transferred) by layer 1 is checked against a
fixed set of rules; a rule that fires produces a flag with a
human-readable evidence string, later shown to the LLM (layer 3).

Rules are fixed a priori -- no threshold is tuned on the evaluation data:

  S0 empty_text          field has no text at all
  S1 unknown_identifier  a code-like identifier cited in the field
                         ({@code x}, or a camelCase / snake_case word)
                         appears nowhere in the method code
  S2 unknown_literal     a number (other than 0/1) or a double-quoted
                         string in the field appears in no code literal
  S3a return_null_claim  @return mentions null, but no `return null`
  S3b return_bool_claim  @return mentions true/false, return type is not boolean
  S3c container_mismatch field names a container kind (list, map, set,
                         array, stream, optional, iterator) that
                         contradicts a container-typed return/param type
  S3d param_null_claim   @param says it may be null, but the code rejects
                         null for it (requireNonNull / checkNotNull /
                         `if (p == null) throw`)
  S3e throws_unverified  documented exception neither declared nor thrown
                         in the body (weak: callees may throw it)

The description is always judged by the LLM regardless (policy decided
in the design discussion); its flags only add evidence.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

import javalang

SIGNALS = ("empty_text", "unknown_identifier", "unknown_literal", "return_null_claim",
           "return_bool_claim", "container_mismatch", "param_null_claim", "throws_unverified")

_JAVA_WORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue",
    "default", "do", "double", "else", "enum", "extends", "final", "finally", "float", "for", "goto", "if",
    "implements", "import", "instanceof", "int", "interface", "long", "native", "new", "package", "private",
    "protected", "public", "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this",
    "throw", "throws", "transient", "try", "void", "volatile", "while", "true", "false", "null", "var",
}
_NULL_GUARDS = {"requireNonNull", "checkNotNull", "notNull"}
_CONTAINER_FAMILIES = {
    "list": ("list",),
    "map": ("map",),
    "set": ("set",),
    "array": ("[]",),
    "stream": ("stream",),
    "optional": ("optional",),
    "iterator": ("iterator", "iterable"),
}
_INLINE_RE = re.compile(r"\{@(\w+)\s*([^{}]*)\}")
_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")
_CAMEL_RE = re.compile(r"\b[a-z][a-z0-9]*[A-Z]\w*\b|\b[a-z][a-z0-9]*_[a-z0-9_]+\b")
_NUMBER_RE = re.compile(r"(?<![\w.#-])(-?\d+(?:\.\d+)?)(?![\w]|\.\d)")
_QUOTED_RE = re.compile(r'"([^"\n]{1,40})"')
_MAY_BE_NULL_RE = re.compile(r"\b(?:may|can|could|might)\s+be\s+(?:\{@code\s+)?null\b|\bnullable\b|\bor\s+(?:\{@code\s+)?null\b", re.I)


# ----------------------------------------------------------------------
# Code facts for semantic checks
# ----------------------------------------------------------------------
@dataclass
class SemanticFacts:
    identifiers: set
    numbers: set                  # absolute numeric values of code literals
    strings: set                  # contents of string literals
    returns_null: bool
    return_type: str | None       # e.g. "boolean", "List<String>" rendered as "list", "int[]"
    param_types: dict = field(default_factory=dict)
    null_rejected_params: set = field(default_factory=set)


def _render_type(t, varargs=False):
    if t is None:
        return None
    name = t.name if hasattr(t, "name") else str(t)
    sub = t
    while getattr(sub, "sub_type", None) is not None:      # java.util.List -> List
        sub = sub.sub_type
        name = sub.name
    dims = len(getattr(t, "dimensions", None) or []) + (1 if varargs else 0)
    return name + "[]" * dims


_TK = javalang.tokenizer
# NOTE: javalang's DecimalInteger does NOT subclass Integer, so list the classes explicitly
_INT_TOKENS = (_TK.DecimalInteger, _TK.HexInteger, _TK.OctalInteger, _TK.BinaryInteger)
_FLOAT_TOKENS = (_TK.DecimalFloatingPoint, _TK.HexFloatingPoint)


def _parse_number(tok):
    v = tok.value.replace("_", "")
    try:
        if isinstance(tok, _TK.OctalInteger):
            return abs(int(v.rstrip("lL"), 8))
        if isinstance(tok, _INT_TOKENS):
            return abs(int(v.rstrip("lL"), 0) if not isinstance(tok, _TK.DecimalInteger) else int(v.rstrip("lL")))
        if isinstance(tok, _TK.HexFloatingPoint):
            return abs(float.fromhex(v.rstrip("fFdD")))
        return abs(float(v.rstrip("fFdD")))
    except ValueError:
        return None


def extract_semantic_facts(method_code):
    if not isinstance(method_code, str) or not method_code.strip():
        return None
    code = method_code.strip()
    identifiers, numbers, strings = set(), set(), set()
    try:
        for tok in javalang.tokenizer.tokenize(code):
            if isinstance(tok, javalang.tokenizer.Identifier):
                identifiers.add(tok.value)
            elif isinstance(tok, _INT_TOKENS + _FLOAT_TOKENS):
                n = _parse_number(tok)
                if n is not None:
                    numbers.add(n)
            elif isinstance(tok, javalang.tokenizer.String):
                strings.add(tok.value[1:-1])
    except Exception:
        identifiers = set(re.findall(r"[A-Za-z_]\w*", code))
        numbers = {abs(float(x)) for x in re.findall(r"\b\d+(?:\.\d+)?\b", code)}
        strings = set(re.findall(r'"([^"\\]*)"', code))

    node, tree = None, None
    for candidate in (f"class _Wrapper {{\n{code}\n}}", code):
        try:
            tree = javalang.parse.parse(candidate)
            break
        except Exception:
            continue
    if tree is not None:
        for _, n in tree.filter(javalang.tree.MethodDeclaration):
            node = n
            break
        if node is None:
            for _, n in tree.filter(javalang.tree.ConstructorDeclaration):
                node = n
                break
    if node is None:
        return SemanticFacts(identifiers, numbers, strings, False, None)

    returns_null = any(isinstance(r.expression, javalang.tree.Literal) and r.expression.value == "null"
                       for _, r in node.filter(javalang.tree.ReturnStatement))
    param_names = {p.name for p in node.parameters}
    param_types = {p.name: _render_type(p.type, getattr(p, "varargs", False)) for p in node.parameters}

    rejected = set()
    for _, inv in node.filter(javalang.tree.MethodInvocation):
        if inv.member in _NULL_GUARDS and inv.arguments:
            a = inv.arguments[0]
            if isinstance(a, javalang.tree.MemberReference) and a.member in param_names:
                rejected.add(a.member)
    for _, st in node.filter(javalang.tree.IfStatement):
        c = st.condition
        if isinstance(c, javalang.tree.BinaryOperation) and c.operator == "==":
            ops = (c.operandl, c.operandr)
            ref = next((o for o in ops if isinstance(o, javalang.tree.MemberReference) and o.member in param_names and not o.qualifier), None)
            is_null = any(isinstance(o, javalang.tree.Literal) and o.value == "null" for o in ops)
            throws = st.then_statement is not None and any(True for _ in st.then_statement.filter(javalang.tree.ThrowStatement))
            if ref is not None and is_null and throws:
                rejected.add(ref.member)

    return_type = None
    if isinstance(node, javalang.tree.MethodDeclaration):
        return_type = "void" if node.return_type is None else _render_type(node.return_type)
    return SemanticFacts(identifiers, numbers, strings, returns_null, return_type, param_types, rejected)


# ----------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------
def _plain_text(text):
    t = _URL_RE.sub(" ", text or "")
    t = _INLINE_RE.sub(" ", t)
    return _HTML_RE.sub(" ", t)


def _cited_identifiers(text):
    """Identifiers the field explicitly presents as code: lowerCamel /
    snake_case tokens in {@code ...}, and camelCase / snake_case words in
    plain prose. UpperCamel names (types, possibly external) and
    {@link ...} targets (possibly sibling members) are not verifiable
    from the method alone and are skipped."""
    names = set()
    for tag, content in _INLINE_RE.findall(text or ""):
        if tag.lower() in ("code", "literal"):
            for tok in re.findall(r"[A-Za-z_]\w*", content):
                if tok[0].islower() and tok not in _JAVA_WORDS and len(tok) > 1:
                    names.add(tok)
    names |= set(_CAMEL_RE.findall(_plain_text(text)))
    return names


def _container_kinds(text):
    words = set(re.findall(r"[a-z]+", _plain_text(text).lower()))
    return {k for k in _CONTAINER_FAMILIES if k in words or k + "s" in words}


def _type_family(type_str):
    if not type_str:
        return set()
    t = type_str.lower()
    return {k for k, marks in _CONTAINER_FAMILIES.items() if any(m in t for m in marks)}


def triage_fields(parsed, actions, sem):
    """Flags for every field layer 1 keeps (or transfers). Returns a list of
    {key, kind, signal, evidence}. `sem` is a SemanticFacts."""
    by_key = {f.key: f for f in (parsed.fields if parsed is not None else [])}
    targets = []
    for a in actions:
        if a["action"] == "keep" and a["key"] in by_key and by_key[a["key"]].kind != "other":
            f = by_key[a["key"]]
            targets.append((a["key"], f.kind, f.name, f.text, a.get("reason")))
        elif a["action"] == "transfer":
            targets.append((a["key"], "param", a["key"].split(" ", 1)[1], a["text"], "rename"))

    flags = []
    def flag(key, kind, signal, evidence):
        flags.append({"key": key, "kind": kind, "signal": signal, "evidence": evidence})

    for key, kind, name, text, reason in targets:
        if not (text or "").strip():
            flag(key, kind, "empty_text", "the field has no text")
            continue
        unknown = sorted(n for n in _cited_identifiers(text) if n not in sem.identifiers)
        if unknown:
            flag(key, kind, "unknown_identifier", "mentions " + ", ".join(f"`{u}`" for u in unknown) + ", absent from the code")

        plain = _plain_text(text)
        bad_nums = []
        for s in _NUMBER_RE.findall(plain):
            v = abs(float(s))
            if v in (0.0, 1.0):
                continue
            if v not in sem.numbers:
                bad_nums.append(s)
        bad_strs = [s for s in _QUOTED_RE.findall(text) if s not in sem.strings]
        if bad_nums or bad_strs:
            items = bad_nums + [f'"{s}"' for s in bad_strs]
            flag(key, kind, "unknown_literal", "mentions " + ", ".join(items) + ", which appear in no code literal")

        lower = text.lower()
        if kind == "return":
            if re.search(r"\bnull\b", lower) and not sem.returns_null:
                flag(key, kind, "return_null_claim", "mentions null but the method has no `return null`")
            if re.search(r"\b(true|false)\b", lower) and sem.return_type and sem.return_type.lower() not in ("boolean", "boolean[]") \
                    and sem.return_type != "Boolean":
                flag(key, kind, "return_bool_claim", f"mentions true/false but the return type is `{sem.return_type}`")
            fam, kinds = _type_family(sem.return_type), _container_kinds(text)
            if fam and kinds and not (kinds & fam):
                flag(key, kind, "container_mismatch", f"describes a {'/'.join(sorted(kinds))} but the return type is `{sem.return_type}`")
        if kind == "param":
            ptype = sem.param_types.get(name)
            fam, kinds = _type_family(ptype), _container_kinds(text)
            if fam and kinds and not (kinds & fam):
                flag(key, kind, "container_mismatch", f"describes a {'/'.join(sorted(kinds))} but `{name}` is `{ptype}`")
            if name in sem.null_rejected_params and _MAY_BE_NULL_RE.search(text):
                flag(key, kind, "param_null_claim", f"says `{name}` may be null but the code rejects null")
        if kind == "throws" and reason == "throws_unverified":
            flag(key, kind, "throws_unverified", f"`{name}` is neither declared nor thrown directly in the body")
    return flags
