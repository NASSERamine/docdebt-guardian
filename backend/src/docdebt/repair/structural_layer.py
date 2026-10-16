"""Research ``src/mitigation/structural_layer.py`` (hybrid Replacement, layer 1),
kept as tested; only the offline evaluation (pandas) is left out.

Layer 1 of the hybrid Replacement mechanism: deterministic, zero-token
structural planning. Compares the parsed stale Javadoc (javadoc_spans)
with facts extracted from the method's AST (javalang) and decides, per
field, one of:

    delete    - phantom @param / @param <T>, duplicate @param, @return on a
                void method or constructor
    transfer  - phantom @param paired with a missing parameter (rename):
                its text is moved verbatim to the new name
    write     - missing @param / @param <T> / @return / declared @throws:
                content must be produced by the LLM (layer 3)
    keep      - everything else (later triaged by layer 2)

Nothing here calls a model. build_patch() turns a plan (+ texts written
by the LLM later) into ONE apply_span_patch call on the original doc, so
untouched lines stay byte-identical.

"""
from __future__ import annotations
import difflib
from dataclasses import dataclass, field

import javalang

from docdebt.repair.spans import apply_span_patch

RENAME_RULES = ("none", "unique", "position", "name_sim", "name_or_position")


# ----------------------------------------------------------------------
# AST facts
# ----------------------------------------------------------------------
@dataclass
class CodeFacts:
    param_names: list
    type_params: list
    is_void: bool
    is_constructor: bool
    declared_throws: list          # simple names, from the throws clause
    thrown_in_body: list = field(default_factory=list)   # simple names of `throw new X(...)`


def _simple(name):
    return name.split(".")[-1] if isinstance(name, str) else name


def extract_code_facts(method_code):
    """Same wrapping strategy as evaluation.structural.extract_signature_from_method_code,
    plus type parameters and exceptions thrown in the body."""
    if not isinstance(method_code, str) or not method_code.strip():
        return None
    code = method_code.strip()
    tree = None
    for candidate in (f"class _Wrapper {{\n{code}\n}}", code):
        try:
            tree = javalang.parse.parse(candidate)
            break
        except Exception:
            continue
    if tree is None:
        return None
    node, is_ctor = None, False
    for _, n in tree.filter(javalang.tree.MethodDeclaration):
        node = n
        break
    if node is None:
        for _, n in tree.filter(javalang.tree.ConstructorDeclaration):
            node, is_ctor = n, True
            break
    if node is None:
        return None
    thrown = []
    for _, t in node.filter(javalang.tree.ThrowStatement):
        expr = t.expression
        if isinstance(expr, javalang.tree.ClassCreator) and expr.type is not None:
            thrown.append(_simple(expr.type.name))
    return CodeFacts(
        param_names=[p.name for p in node.parameters],
        type_params=[f"<{tp.name}>" for tp in (getattr(node, "type_parameters", None) or [])],
        is_void=is_ctor or node.return_type is None,
        is_constructor=is_ctor,
        declared_throws=[_simple(t) for t in (node.throws or [])],
        thrown_in_body=sorted(set(thrown)),
    )


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------
def _name_sim(a, b):
    a, b = a.lower(), b.lower()
    if a in b or b in a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def pair_renames(phantoms, missing, doc_order, sig_order, rule, min_name_sim=0.5):
    """phantoms/missing: names. Returns {missing_name: phantom_name}.
    Rules are fixed in advance (see RENAME_RULES) and compared once on
    the ground truth -- no per-item tuning."""
    if rule == "none" or not phantoms or not missing:
        return {}
    if rule == "unique":
        return {missing[0]: phantoms[0]} if len(phantoms) == 1 and len(missing) == 1 else {}
    cands = []
    for p in phantoms:
        for m in missing:
            same_pos = doc_order.index(p) == sig_order.index(m)
            ns = _name_sim(p, m)
            if rule == "position":
                score = 1.0 if same_pos else 0.0
            elif rule == "name_sim":
                score = ns if ns >= min_name_sim else 0.0
            else:  # name_or_position
                score = ns + (0.5 if same_pos else 0.0)
                score = score if (ns >= min_name_sim or same_pos) else 0.0
            if score > 0:
                cands.append((score, p, m))
    pairs, used_p, used_m = {}, set(), set()
    for score, p, m in sorted(cands, reverse=True):
        if p not in used_p and m not in used_m:
            pairs[m] = p
            used_p.add(p)
            used_m.add(m)
    if rule == "position" and not pairs and len(phantoms) == 1 and len(missing) == 1:
        pairs = {missing[0]: phantoms[0]}
    return pairs


def plan_structural_fixes(parsed, facts, rename_rule="name_sim"):
    """Returns a list of action dicts: {key, action, reason, source_key, text}.
    parsed may be None (no existing doc): everything required becomes 'write'."""
    actions = []
    fields = parsed.fields if parsed is not None else []
    real = list(facts.param_names)

    # @param
    doc_params = [f for f in fields if f.kind == "param"]
    seen, unique_doc = set(), []
    for f in doc_params:
        if f.name in seen:
            actions.append({"key": f.key, "action": "delete", "reason": "duplicate_param"})
        else:
            seen.add(f.name)
            unique_doc.append(f)
    doc_names = [f.name for f in unique_doc]
    phantoms = [n for n in doc_names if n not in real]
    missing = [n for n in real if n not in doc_names]
    pairs = pair_renames(phantoms, missing, doc_names, real, rename_rule)
    paired_phantoms = set(pairs.values())
    for f in unique_doc:
        if f.name in real:
            actions.append({"key": f.key, "action": "keep", "reason": "param_ok"})
        elif f.name not in paired_phantoms:
            actions.append({"key": f.key, "action": "delete", "reason": "phantom_param"})
    for m in missing:
        if m in pairs:
            src = next(f for f in unique_doc if f.name == pairs[m])
            actions.append({"key": f"@param {m}", "action": "transfer", "reason": "rename",
                            "source_key": src.key, "text": src.text})
        else:
            actions.append({"key": f"@param {m}", "action": "write", "reason": "missing_param"})

    # @param <T>
    doc_tps = [f for f in fields if f.kind == "type_param"]
    for f in doc_tps:
        ok = f.name in facts.type_params
        actions.append({"key": f.key, "action": "keep" if ok else "delete",
                        "reason": "type_param_ok" if ok else "phantom_type_param"})
    for tp in facts.type_params:
        if tp not in {f.name for f in doc_tps}:
            actions.append({"key": f"@param {tp}", "action": "write", "reason": "missing_type_param"})

    # @return
    doc_ret = [f for f in fields if f.kind == "return"]
    if facts.is_void:
        for f in doc_ret:
            actions.append({"key": f.key, "action": "delete", "reason": "return_on_void"})
    elif doc_ret:
        for f in doc_ret:
            actions.append({"key": f.key, "action": "keep", "reason": "return_ok"})
    else:
        actions.append({"key": "@return", "action": "write", "reason": "missing_return"})

    # @throws: declared-but-undocumented -> write; documented-but-undeclared -> keep
    # (unchecked exceptions may legitimately come from callees; left to layer 2)
    doc_thr = [f for f in fields if f.kind == "throws"]
    doc_thr_names = {_simple(f.name) for f in doc_thr}
    for f in doc_thr:
        declared = _simple(f.name) in facts.declared_throws
        in_body = _simple(f.name) in facts.thrown_in_body
        actions.append({"key": f.key, "action": "keep",
                        "reason": "throws_declared" if declared else ("throws_in_body" if in_body else "throws_unverified")})
    for t in facts.declared_throws:
        if t not in doc_thr_names:
            actions.append({"key": f"@throws {t}", "action": "write", "reason": "missing_declared_throws"})

    # description / other tags
    for f in fields:
        if f.kind == "description":
            actions.append({"key": f.key, "action": "keep", "reason": "description"})
        elif f.kind == "other":
            actions.append({"key": f.key, "action": "keep", "reason": "other_tag"})
    if not any(f.kind == "description" for f in fields):
        actions.append({"key": "description", "action": "write", "reason": "missing_description"})

    for a in actions:
        a.setdefault("source_key", None)
        a.setdefault("text", None)
    return actions


def build_patch(parsed, facts, actions, written=None, replaced=None):
    """Single apply_span_patch call on the ORIGINAL doc. Renames are applied
    in place (same position and layout as the phantom tag).
    written:  {key: text} produced by the LLM for 'write' actions (missing ones are skipped)
    replaced: {key: text} LLM corrections of existing (or transferred) fields (layer 3)."""
    written, replaced = dict(written or {}), dict(replaced or {})
    existing = {f.key for f in parsed.fields}
    delete = {a["key"] for a in actions if a["action"] == "delete"}
    rename = {a["source_key"]: a["key"].split(" ", 1)[1] for a in actions if a["action"] == "transfer"}
    target_to_source = {a["key"]: a["source_key"] for a in actions if a["action"] == "transfer"}
    replace, insert = {}, []
    for src in (replaced, written):
        for key, text in src.items():
            if key in target_to_source:
                replace[target_to_source[key]] = text
            elif key in existing and key not in delete:
                replace[key] = text
            else:
                insert.append((key, text))
    return apply_span_patch(parsed, replace=replace, delete=delete, insert=insert, rename=rename,
                            param_order=list(facts.type_params) + list(facts.param_names))
