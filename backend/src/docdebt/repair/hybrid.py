"""Research ``src/mitigation/hybrid_replacement.py`` (hybrid Replacement, layers 3 and 4).

Kept as tested. Changes for the service: imports, an optional ``context``
section in the prompt (retrieved project context, shown to the model as
reference only), and the offline re-application helper left out.

Hybrid Replacement, layers 3 (targeted LLM prompt) and 4 (deterministic
post-validation), plus final assembly. Layers 1-2 live in
structural_layer.py and semantic_triage.py. This module never calls a
model: the notebook orchestration does (batching, retries, tokens).

Inference inputs: method_code + javadoc_before ONLY.

Per item:
  prepare_hybrid_item()  layer 1 plan + layer 2 flags -> WRITE / JUDGE tasks
  build_hybrid_prompt()  one prompt for a batch of items (braces encoded)
  validate_item_patch()  layer 4 on the model's answer for one item
  assemble_hybrid_doc()  one span patch on the ORIGINAL doc + structural check

Variants (ablation):
  judge="all"      every existing field is judged (default)
  judge="flagged"  description + fields flagged by layer 2
"""
from __future__ import annotations
import json
import re

from docdebt.repair.brace_tokens import decode_braces, encode_braces, looks_truncated_at_brace
from docdebt.repair.semantic_triage import _cited_identifiers, extract_semantic_facts, triage_fields
from docdebt.repair.spans import parse_javadoc_spans
from docdebt.repair.structural_layer import _simple, build_patch, extract_code_facts, plan_structural_fixes

JUDGE_VARIANTS = ("all", "flagged")
_KEEP_WORDS = {"keep", "unchanged", "\"keep\"", "keep."}
_JUDGED_KINDS = ("description", "type_param", "param", "return", "throws")
_EMPTY_DOC = "/**\n */"


# ----------------------------------------------------------------------
# Item preparation (layers 1 + 2)
# ----------------------------------------------------------------------
def prepare_hybrid_item(item_id, method_code, stale_doc, judge="all", rename_rule="name_sim"):
    """Returns a dict with everything needed downstream, or None if the
    method code cannot be parsed."""
    if judge not in JUDGE_VARIANTS:
        raise ValueError(f"judge must be one of {JUDGE_VARIANTS}")
    facts = extract_code_facts(method_code)
    if facts is None:
        return None
    sem = extract_semantic_facts(method_code)
    doc = stale_doc if isinstance(stale_doc, str) and stale_doc.strip() else _EMPTY_DOC
    parsed = parse_javadoc_spans(doc)
    actions = plan_structural_fixes(parsed, facts, rename_rule=rename_rule)
    flags = triage_fields(parsed, actions, sem)

    hints = {}
    for f in flags:
        if f["signal"] != "empty_text":
            hints.setdefault(f["key"], []).append(f["evidence"])
    empty = {f["key"] for f in flags if f["signal"] == "empty_text"}

    by_key = {f.key: f for f in parsed.fields}
    write, judge_tasks, context = [], [], []
    for a in actions:
        key, act = a["key"], a["action"]
        if act == "write":
            write.append({"key": key, "reason": a["reason"], "existing": False})
        elif act in ("keep", "transfer"):
            kind = by_key[key].kind if act == "keep" else "param"
            if kind not in _JUDGED_KINDS:
                continue
            text = by_key[key].text if act == "keep" else a["text"]
            if key in empty:
                write.append({"key": key, "reason": "empty_text", "existing": act == "keep"})
                continue
            task = {"key": key, "text": text, "hints": hints.get(key, []), "transferred": act == "transfer"}
            if judge == "all" or key == "description" or key in hints:
                judge_tasks.append(task)
            else:
                context.append(task)
    return {
        "item_id": item_id, "method_code": method_code, "stale_doc": doc, "parsed": parsed,
        "facts": facts, "sem": sem, "actions": actions, "flags": flags,
        "write": write, "judge": judge_tasks, "context": context, "judge_variant": judge,
    }


# ----------------------------------------------------------------------
# Layer 3: prompt
# ----------------------------------------------------------------------
def _item_block(it):
    lines = [f'--- ITEM "{it["item_id"]}" ---', "METHOD CODE:", it["method_code"].strip(), ""]
    if it["write"]:
        lines.append("FIELDS TO WRITE (missing or empty -- write accurate text from the code):")
        lines += [f"  {json.dumps(w['key'])}" for w in it["write"]]
        lines.append("")
    if it["judge"]:
        lines.append('FIELDS TO JUDGE (answer "keep" unless the code contradicts the text or it is clearly incomplete):')
        for j in it["judge"]:
            lines.append(f"  {json.dumps(j['key'])}: {json.dumps(encode_braces(j['text']))}")
            for h in j["hints"]:
                lines.append(f"      possible issue (verify, may be a false alarm): {encode_braces(h)}")
        lines.append("")
    if it["context"]:
        lines.append("OTHER FIELDS (context only, do not return them):")
        lines += [f"  {json.dumps(c['key'])}: {json.dumps(encode_braces(c['text']))}" for c in it["context"]]
        lines.append("")
    return "\n".join(lines)


def _context_block(context):
    if not context or not str(context).strip():
        return ""
    return (
        "PROJECT CONTEXT (from the same repository; reference only -- use it to match the project's\n"
        "vocabulary and conventions, never copy it, never return it):\n"
        f"{encode_braces(str(context).strip())}\n--- END OF CONTEXT ---\n\n"
    )


def build_hybrid_prompt(items, context=None):
    blocks = "\n".join(_item_block(it) for it in items)
    return f'''You are repairing outdated Java documentation, field by field, against the actual code,
for {len(items)} independent item(s). Judge each item separately.

{blocks}
{_context_block(context)}Rules:
- For every field under FIELDS TO WRITE, provide accurate text describing it, based on the code.
- For every field under FIELDS TO JUDGE, answer exactly "keep" if the text is still correct.
  Only if the code contradicts it, or it is clearly incomplete, provide the corrected text.
  Never rephrase text that is already correct.
- Field text only: no "@param"/"@return" prefixes, no comment markers.
- In field texts, \u00a7LINK\u00a7 and \u00a7CODE\u00a7 stand for Javadoc inline tags and \u00a7END\u00a7 for their
  closing brace: keep them exactly as they are in any text you return.

Respond with ONLY a single JSON object keyed by item id, each value mapping every WRITE and
JUDGE field key of that item to its text or "keep":
{{"<item_id>": {{"<field key>": "<text or keep>", ...}}, ...}}
Every item id listed above must appear exactly once as a top-level key.'''


# ----------------------------------------------------------------------
# Layer 4: post-validation
# ----------------------------------------------------------------------
_TAG_LEAK_RE = re.compile(r"(^|\s)@(param|return|throws|exception)\b|/\*\*|\*/")


_TOKEN_VARIANT_RE = {
    # a model may re-emit our section-sign tokens with other symbols (e.g. U+2061/U+2069):
    # accept 1-3 NON-ASCII symbol characters on each side of the token word
    # (ASCII punctuation such as a following comma is never consumed)
    "{@link": re.compile(r"[^\x00-\x7f\w\s]{1,3}LINK[^\x00-\x7f\w\s]{1,3}"),
    "{@code": re.compile(r"[^\x00-\x7f\w\s]{1,3}CODE[^\x00-\x7f\w\s]{1,3}"),
    "}": re.compile(r"[^\x00-\x7f\w\s]{1,3}END[^\x00-\x7f\w\s]{1,3}"),
}


def _decode_tokens_robust(text):
    text = decode_braces(text)
    for repl, rx in _TOKEN_VARIANT_RE.items():
        text = rx.sub(repl, text)
    return text


def _has_token_residue(text, original):
    """Leftovers of our brace tokens that could not be decoded: a section
    sign, invisible format characters (Unicode category Cf), or a bare
    LINK/CODE word that the original field did not contain."""
    import unicodedata
    if "\u00a7" in text or any(unicodedata.category(ch) == "Cf" for ch in text):
        return True
    orig = original or ""
    return any(re.search(rf"\b{w}\b", text) and not re.search(rf"\b{w}\b", orig) for w in ("LINK", "CODE"))


def _sanitize(key, text):
    """Deterministic clean-up of harmless formatting slips before checking."""
    text = "\n".join(re.sub(r"^\s*\*\s?", "", ln).rstrip() for ln in str(text).split("\n")).strip()
    if key != "description":
        head = key.split("#")[0]
        if key.startswith("@throws"):
            m = re.match(rf"^@(?:throws|exception)\s+\S*{re.escape(key.split(' ', 1)[1])}\s*", text)
            if m:
                text = text[m.end():]
        elif text.startswith(head):
            text = text[len(head):]
        text = text.strip()
    return text


def _format_ok(text):
    return bool(text.strip()) and not _TAG_LEAK_RE.search(text) and not looks_truncated_at_brace(text)


def _new_identifiers(text, original, sem):
    """Identifiers cited by the new text that appear neither in the code nor
    ANYWHERE in the original field (including {@link ...} targets, which a
    model may legitimately unwrap into plain words)."""
    orig = set(re.findall(r"[A-Za-z_]\w*", original or ""))
    return sorted(n for n in _cited_identifiers(text) if n not in sem.identifiers and n not in orig)


_INLINE_WRAP_RE = re.compile(r"\{@\w+\s*([^{}]*)\}")


def _content_form(text):
    """Markup-insensitive form: inline tags unwrapped ({@code x} -> x,
    {@link A#b(C) label} -> A#b(C) label), HTML tags dropped, then case,
    punctuation and whitespace ignored. Two texts with the same content
    form differ only in presentation."""
    t = text or ""
    prev = None
    while prev != t:
        prev, t = t, _INLINE_WRAP_RE.sub(r" \1 ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^\w]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def _is_deletion_only(new_text, original):
    """True if the new text adds no word that the original did not already
    contain (multiset inclusion on the markup-insensitive content form):
    the model only removed, reordered or reformatted text. Such an edit
    brings no information from the code, so the original is kept."""
    from collections import Counter
    new_words = Counter(_content_form(new_text).split())
    old_words = Counter(_content_form(original).split())
    return not (new_words - old_words)


def _key_forms(key):
    """Equivalent spellings of a field key a model may return:
    whitespace/case variants, the tag without '@' ('param x', 'return'),
    the bare name ('x' for '@param x', 'IOException' for '@throws IOException'),
    '@exception'/'@returns' synonyms, and for exceptions the simple name
    without package/outer class or trailing punctuation
    ('@throws SmackException.NotLoggedInException' -> 'notloggedinexception')."""
    k = re.sub(r"\s+", " ", str(key).strip().strip('"').strip()).lower()
    forms = {k}
    if k in ("description", "desc", "@description", "summary"):
        return forms | {"description", "desc", "@description", "summary"}
    if k in ("@return", "return", "@returns", "returns"):
        return forms | {"@return", "return", "@returns", "returns"}
    m = re.match(r"^@?param[ _:](\S+)$", k)
    if m:
        name = m.group(1)
        return forms | {f"@param {name}", f"param {name}", f"param_{name}", f"param:{name}", name}
    m = re.match(r"^@?(?:throws|exception)[ _:](\S+)$", k)
    if m:
        full = m.group(1).rstrip(".,;:")
        out = set(forms)
        for name in {full, full.split(".")[-1]}:
            out |= {f"@throws {name}", f"@exception {name}", f"throws {name}", f"exception {name}",
                    f"throws_{name}", f"exception_{name}", name}
        return out
    return forms


def _align_patch_keys(expected, patch):
    """Map each expected key to the model's value: exact key first, then an
    equivalent spelling (see _key_forms) if it identifies exactly one
    expected key. Returns (aligned {expected_key: value}, unmatched model keys)."""
    aligned, used = {}, set()
    for k in expected:
        if k in patch:
            aligned[k], _ = patch[k], used.add(k)
    form_owner = {}
    for k in expected:
        if k not in aligned:
            for f in _key_forms(k):
                form_owner.setdefault(f, set()).add(k)
    for pk, v in patch.items():
        if pk in used:
            continue
        owners = set()
        for f in _key_forms(pk):
            owners |= form_owner.get(f, set())
        owners = {o for o in owners if o not in aligned}
        if len(owners) == 1:
            aligned[owners.pop()] = v
            used.add(pk)
    return aligned, [pk for pk in patch if pk not in used]


def validate_item_patch(it, patch):
    """patch: the model's dict for this item (brace tokens NOT yet decoded).
    Returns (written {key: text}, replaced {key: text}, log [..], failed_writes [keys])."""
    patch = patch if isinstance(patch, dict) else {}
    written, replaced, log, failed = {}, {}, [], []
    expected = [w["key"] for w in it["write"]] + [j["key"] for j in it["judge"]]
    patch, unmatched = _align_patch_keys(expected, patch)
    for k in unmatched:
        log.append({"key": k, "task": "extra", "outcome": "ignored_extra_key"})

    for w in it["write"]:
        key, raw = w["key"], patch.get(w["key"])
        if not isinstance(raw, str) or raw.strip().lower() in _KEEP_WORDS:
            failed.append(key)
            log.append({"key": key, "task": "write", "outcome": "write_failed"})
            continue
        text = _sanitize(key, _decode_tokens_robust(raw))
        if not _format_ok(text) or _has_token_residue(text, None):
            failed.append(key)
            log.append({"key": key, "task": "write", "outcome": "write_failed"})
            continue
        new_ids = _new_identifiers(text, None, it["sem"])
        written[key] = text
        log.append({"key": key, "task": "write", "outcome": "write_flagged" if new_ids else "accepted",
                    "detail": ", ".join(new_ids) or None})

    for j in it["judge"]:
        key, raw = j["key"], patch.get(j["key"])
        if raw is None:
            log.append({"key": key, "task": "judge", "outcome": "kept_no_verdict"})
            continue
        if not isinstance(raw, str) or raw.strip().lower() in _KEEP_WORDS:
            log.append({"key": key, "task": "judge", "outcome": "kept"})
            continue
        text = _sanitize(key, _decode_tokens_robust(raw))
        if not _format_ok(text) or _has_token_residue(text, j["text"]):
            outcome, detail = "reverted_format", None
        elif (new_ids := _new_identifiers(text, j["text"], it["sem"])):
            outcome, detail = "reverted_hallucination", ", ".join(new_ids)
        elif _content_form(text) == _content_form(j["text"]):
            outcome, detail = "reverted_cosmetic", None     # presentation-only change (incl. markup)
        elif _is_deletion_only(text, j["text"]):
            outcome, detail = "reverted_deletion_only", None  # only removed/reordered text, nothing added
        else:
            outcome, detail = "accepted", None
            replaced[key] = text
        log.append({"key": key, "task": "judge", "outcome": outcome, "detail": detail})
    return written, replaced, log, failed


# ----------------------------------------------------------------------
# Assembly + structural check
# ----------------------------------------------------------------------
def assemble_hybrid_doc(it, written, replaced):
    """One apply_span_patch on the ORIGINAL stale doc (renames in place)."""
    return build_patch(it["parsed"], it["facts"], it["actions"], written=written, replaced=replaced)


def check_structure(doc, facts):
    p = parse_javadoc_spans(doc)
    fields = p.fields if p is not None else []
    doc_params = [f.name for f in fields if f.kind == "param"]
    doc_throws = {_simple(f.name) for f in fields if f.kind == "throws"}
    has_return = any(f.kind == "return" and f.text.strip() for f in fields)
    empty_params = [f.name for f in fields if f.kind == "param" and not f.text.strip()]
    problems = {
        "missing_params": sorted(set(facts.param_names) - set(doc_params)),
        "phantom_params": sorted(set(doc_params) - set(facts.param_names)),
        "duplicate_params": sorted({n for n in doc_params if doc_params.count(n) > 1}),
        "empty_params": sorted(empty_params),
        "missing_return": (not facts.is_void) and not has_return,
        "return_on_void": facts.is_void and any(f.kind == "return" for f in fields),
        "missing_declared_throws": sorted(set(facts.declared_throws) - doc_throws),
    }
    ok = not any(problems.values())
    return ok, problems


def retry_item_for_writes(it, failed_keys):
    """A reduced copy of the item asking ONLY for the failed WRITE fields
    (no judging) -- used for a cheap solo retry."""
    reduced = dict(it)
    reduced["write"] = [w for w in it["write"] if w["key"] in set(failed_keys)]
    reduced["judge"], reduced["context"] = [], []
    return reduced
