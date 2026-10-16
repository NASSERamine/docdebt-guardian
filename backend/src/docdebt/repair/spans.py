"""Span-preserving Javadoc parser and patcher (research ``src/mitigation/javadoc_spans.py``,
copied verbatim apart from this header).

Span-preserving Javadoc parser and patcher for the hybrid Replacement
mechanism. Unlike parse_javadoc/reconstruct_javadoc (which re-render the
whole block and silently drop @see/@since/@deprecated/@author/...), this
module records, for every field, the exact line range it occupies in the
ORIGINAL text. A patch rewrites only the lines of the fields it touches:
every other line (other tags, <p>, wrapping, alignment, indentation) is
emitted byte-identical.

Parsing follows the Javadoc spec: a block tag starts only at the
beginning of a comment line (after the leading '*'), so inline tags such
as {@link X} / {@code X} can never be mistaken for field boundaries.
Lines inside <pre>...</pre> are never treated as tag starts (protects
code examples containing annotations such as @Override).
"""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field as dc_field

_OPENER_RE = re.compile(r"^(\s*/\*\*+\s?)(.*)$", re.DOTALL)
_STAR_RE = re.compile(r"^(\s*\*(?!/)\s?)(.*)$", re.DOTALL)
_LEADING_WS_RE = re.compile(r"^(\s*)(.*)$", re.DOTALL)
_CLOSER_RE = re.compile(r"(\s*\*+/\s*)$")
_TAG_START_RE = re.compile(r"^\s*@(\w+)")
_PARAM_HEAD_RE = re.compile(r"^(\s*@param\s+(<?\w+>?))")
_THROWS_HEAD_RE = re.compile(r"^(\s*@(?:throws|exception)\s+(\S+))")
_RETURN_HEAD_RE = re.compile(r"^(\s*@return\b)")
_OTHER_HEAD_RE = re.compile(r"^(\s*@\w+)")
_PRE_OPEN_RE = re.compile(r"<pre\b[^>]*>", re.IGNORECASE)     # not "<prefix>"
_PRE_CLOSE_RE = re.compile(r"</pre\s*>", re.IGNORECASE)


@dataclass
class DocLine:
    raw: str        # original line text, without its newline
    prefix: str     # leading whitespace + '/**' or '*' marker + one optional space
    content: str    # text after the prefix (alignment spaces included)
    suffix: str     # closing '*/' (with surrounding whitespace) if on this line


@dataclass
class JavadocField:
    key: str                 # "description", "@param x", "@param <T>", "@return", "@throws E", "@see", "@since#2", ...
    kind: str                # description | param | type_param | return | throws | other
    name: str | None         # parameter / exception name, or tag name for "other"
    text: str                # field text, comment markers stripped, lines joined with "\n"
    first_line: int
    last_line: int
    head: str = ""           # tag head as written, e.g. "@param scene" (empty for description)


@dataclass
class ParsedJavadocSpans:
    original: str
    lines: list
    newlines: list           # newline sequence after each line ("" for the last one if none)
    fields: list = dc_field(default_factory=list)
    cont_prefix: str = " * "

    def get(self, key):
        return next((f for f in self.fields if f.key == key), None)

    def by_kind(self, kind):
        return [f for f in self.fields if f.kind == kind]


def _split_line(raw, is_first):
    m = _OPENER_RE.match(raw) if is_first else None
    if m is None:
        m = _STAR_RE.match(raw) or _LEADING_WS_RE.match(raw)
    prefix, content = m.group(1), m.group(2)
    suffix = ""
    closer = _CLOSER_RE.search(content)
    if closer:
        suffix = closer.group(1)
        content = content[:closer.start()]
    return DocLine(raw=raw, prefix=prefix, content=content, suffix=suffix)


def _detect_cont_prefix(lines):
    candidates = [ln.prefix for ln in lines[1:] if "*" in ln.prefix and ln.content.strip()]
    if candidates:
        return Counter(candidates).most_common(1)[0][0]
    indent = re.match(r"^(\s*)", lines[0].raw).group(1) if lines else ""
    return indent + " * "


def parse_javadoc_spans(doc_text):
    """Parse a raw Javadoc comment into fields with line spans.
    Returns None for non-string / empty input."""
    if not isinstance(doc_text, str) or not doc_text.strip():
        return None
    pieces = doc_text.splitlines(keepends=True)
    raws = [p.rstrip("\r\n") for p in pieces]
    newlines = [p[len(r):] for p, r in zip(pieces, raws)]
    lines = [_split_line(r, i == 0) for i, r in enumerate(raws)]
    for ln in lines:
        assert ln.prefix + ln.content + ln.suffix == ln.raw, f"line split not lossless: {ln.raw!r}"

    # locate tag starts (never inside <pre> blocks)
    tag_starts, in_pre = [], False
    for i, ln in enumerate(lines):
        if not in_pre and _TAG_START_RE.match(ln.content):
            tag_starts.append(i)
        opens, closes = len(_PRE_OPEN_RE.findall(ln.content)), len(_PRE_CLOSE_RE.findall(ln.content))
        if opens > closes:
            in_pre = True
        elif closes > opens:
            in_pre = False

    parsed = ParsedJavadocSpans(original=doc_text, lines=lines, newlines=newlines,
                                cont_prefix=_detect_cont_prefix(lines))

    def last_nonblank(a, b):
        while b >= a and not lines[b].content.strip():
            b -= 1
        return b

    # description
    desc_end = (tag_starts[0] - 1) if tag_starts else len(lines) - 1
    a = 0
    while a <= desc_end and not lines[a].content.strip():
        a += 1
    if a <= desc_end:
        b = last_nonblank(a, desc_end)
        text = "\n".join(lines[i].content.strip() for i in range(a, b + 1)).strip()
        parsed.fields.append(JavadocField("description", "description", None, text, a, b))

    # block tags
    seen = Counter()
    for n, start in enumerate(tag_starts):
        end = last_nonblank(start, (tag_starts[n + 1] - 1) if n + 1 < len(tag_starts) else len(lines) - 1)
        first_content = lines[start].content
        kind, name, head = "other", None, None
        for regex, k in ((_PARAM_HEAD_RE, "param"), (_THROWS_HEAD_RE, "throws")):
            m = regex.match(first_content)
            if m:
                kind, name, head = k, m.group(2), m.group(1)
                break
        if head is None:
            m = _RETURN_HEAD_RE.match(first_content)
            if m:
                kind, head = "return", m.group(1)
            else:
                head = _OTHER_HEAD_RE.match(first_content).group(1)
                name = head.strip()[1:]
        if kind == "param" and name.startswith("<"):
            kind = "type_param"
        body = [first_content[len(head):].strip()] + [lines[i].content.strip() for i in range(start + 1, end + 1)]
        text = "\n".join(body).strip()
        if kind in ("param", "type_param"):
            base_key = f"@param {name}"
        elif kind == "throws":
            base_key = f"@throws {name}"
        elif kind == "return":
            base_key = "@return"
        else:
            base_key = f"@{name}"
        seen[base_key] += 1
        key = base_key if seen[base_key] == 1 else f"{base_key}#{seen[base_key]}"
        parsed.fields.append(JavadocField(key, kind, name, text, start, end, head.strip()))
    return parsed


# ----------------------------------------------------------------------
# Patching
# ----------------------------------------------------------------------
_KIND_RANK = {"description": 0, "type_param": 1, "param": 2, "return": 3, "throws": 4, "other": 5}


def _closer_indent(parsed):
    cp = parsed.cont_prefix
    return cp[:cp.index("*")] if "*" in cp else cp


def _render_replacement(parsed, f, new_text):
    """New raw lines (without newlines) for field f, keeping its own
    prefixes, tag head spacing and continuation alignment."""
    lines = parsed.lines
    first, last = lines[f.first_line], lines[f.last_line]
    new_lines = (new_text or "").split("\n")
    cont_line = next((lines[j] for j in range(f.first_line + 1, f.last_line + 1) if lines[j].content.strip()), None)
    if cont_line is not None:
        cont_prefix = cont_line.prefix
        cont_indent = re.match(r"^(\s*)", cont_line.content).group(1) if f.kind != "description" else ""
    else:
        cont_prefix, cont_indent = parsed.cont_prefix, ""

    out = []
    if f.kind == "description":
        out.append(first.prefix + new_lines[0])
        out += [cont_prefix + ln for ln in new_lines[1:]]
    else:
        head_pos = first.content.index(f.head)
        head_part = first.content[:head_pos + len(f.head)]
        rest = first.content[head_pos + len(f.head):]
        # an empty single-line tag ('@throws X' with no text) gets its new text on the same line
        text_on_first_line = bool(rest.strip()) or cont_line is None
        if text_on_first_line:
            sep = re.match(r"^(\s*)", rest).group(1) or " "
            out.append(first.prefix + head_part + sep + new_lines[0])
            out += [cont_prefix + cont_indent + ln for ln in new_lines[1:]]
        else:
            out.append(first.prefix + head_part)
            out += [cont_prefix + cont_indent + ln for ln in new_lines]
    out = [o.rstrip() if not o.strip().startswith("/**") else o for o in out]
    if last.suffix:
        out[-1] = out[-1] + last.suffix
    return out


def _render_new_field(parsed, key, text):
    cp = parsed.cont_prefix
    new_lines = (text or "").split("\n")
    out = [(cp + key + (" " + new_lines[0] if new_lines[0] else "")).rstrip()]
    out += [(cp + ln).rstrip() for ln in new_lines[1:]]
    return out


def _insertion_anchor(parsed, key, kind, param_order):
    """Line index AFTER which the new field is inserted, and whether a
    blank separator line is needed before it."""
    def rank(f):
        r = _KIND_RANK[f.kind]
        if f.kind == "param" and param_order and f.name in param_order:
            return (r, param_order.index(f.name))
        return (r, -1 if f.kind == "param" else 0)

    name = key.split(" ", 1)[1] if " " in key else None
    new_rank = (_KIND_RANK[kind], param_order.index(name) if (kind == "param" and param_order and name in param_order) else 10**6)
    tags = [f for f in parsed.fields if f.kind != "description"]
    before = [f for f in tags if rank(f) <= new_rank]
    if before:
        return before[-1].last_line, False
    if tags:
        return tags[0].first_line - 1, False
    desc = parsed.get("description")
    if desc is not None:
        return desc.last_line, True
    return 0, False


def apply_span_patch(parsed, replace=None, delete=None, insert=None, param_order=None, rename=None):
    """Apply a field-level patch while keeping every untouched line byte-identical.

    replace:     {field_key: new_text}  for existing fields
    delete:      iterable of field_keys to remove
    insert:      list of (field_key, text) for new fields, e.g. ("@param x", "...")
                 kind is inferred from the key; placement follows Javadoc
                 convention (params in signature order, then @return, then @throws)
    param_order: real parameter names in signature order (for @param placement)
    rename:      {existing "@param old" key: "newName"} -- renames the tag IN PLACE,
                 keeping its position and layout (its text may also be in replace)
    """
    replace, delete, insert = dict(replace or {}), set(delete or ()), list(insert or [])
    rename = dict(rename or {})
    known = {f.key for f in parsed.fields}
    unknown = (set(replace) | delete | set(rename)) - known
    if set(rename) & delete:
        raise ValueError("A field cannot be both renamed and deleted")
    if unknown:
        raise KeyError(f"Unknown field keys: {sorted(unknown)}")
    if set(replace) & delete:
        raise ValueError("A field cannot be both replaced and deleted")
    if not replace and not delete and not insert and not rename:
        return parsed.original

    lines = parsed.lines
    n = len(lines)
    by_first = {f.first_line: f for f in parsed.fields}
    nl = next((x for x in parsed.newlines if x), "\n")

    # inserted blocks, keyed by the line index they follow
    def insert_rank(item):
        k = item[0]
        if k == "description":
            return (-1, 0)
        kind = ("type_param" if k.startswith("@param <") else "param") if k.startswith("@param") else \
               "return" if k.startswith("@return") else \
               "throws" if (k.startswith("@throws") or k.startswith("@exception")) else "other"
        pos = param_order.index(k.split(" ", 1)[1]) if (kind == "param" and param_order and k.split(" ", 1)[1] in param_order) else 10**6
        return (_KIND_RANK[kind], pos)

    after = {}
    for key, text in sorted(insert, key=insert_rank):
        if key == "description":
            if parsed.get("description") is not None:
                raise ValueError("description already exists: use replace")
            has_tags = any(f.kind != "description" for f in parsed.fields)
            block = [(parsed.cont_prefix + ln).rstrip() for ln in (text or "").split("\n")]
            block += [parsed.cont_prefix.rstrip()] if has_tags else []
            after.setdefault(0, [])[:0] = block      # right after the opening /** line
            continue
        if key.startswith("@param"):
            kind = "type_param" if key.split(" ", 1)[1].startswith("<") else "param"
        elif key.startswith("@return"):
            kind = "return"
        elif key.startswith("@throws") or key.startswith("@exception"):
            kind = "throws"
        else:
            kind = "other"
        anchor, need_sep = _insertion_anchor(parsed, key, kind, param_order)
        need_sep = need_sep and anchor not in after   # one separator per insertion point
        block = ([parsed.cont_prefix.rstrip()] if need_sep else []) + _render_new_field(parsed, key, text)
        after.setdefault(anchor, []).extend(block)

    def is_blank(raw):
        return not _split_line(raw, False).content.strip()

    def emit_after(line_idx):
        block = after.pop(line_idx, None)
        if not block:
            return
        closer = _CLOSER_RE.search(out[-1]) if out else None
        if closer and out[-1].strip() != "*/" and out[-1].strip():
            # the closing */ shares the anchor line: split it off
            out[-1] = out[-1][:closer.start()].rstrip()
            out.extend(block)
            out.append(_closer_indent(parsed) + "*/")
        elif out and out[-1].strip() == "*/":
            closer_line = out.pop()
            out.extend(block)
            out.append(closer_line)
        else:
            out.extend(block)

    out = []          # raw output lines
    i = 0
    while i < n:
        f = by_first.get(i)
        if f is not None and (f.key in replace or f.key in rename):
            if f.key in replace:
                rendered = _render_replacement(parsed, f, replace[f.key])
            else:
                rendered = [lines[j].raw for j in range(f.first_line, f.last_line + 1)]
            if f.key in rename:
                rendered[0] = re.sub(rf"(@param\s+){re.escape(f.name)}(?![\w>])",
                                     lambda m: m.group(1) + rename[f.key], rendered[0], count=1)
            out.extend(rendered)
            emit_after(f.last_line)
            i = f.last_line + 1
            continue
        if f is not None and f.key in delete:
            a, b = f.first_line, f.last_line
            keep = []
            if "/**" in lines[a].prefix:
                keep.append(lines[a].prefix.rstrip())
            nxt = b + 1
            skip_next_blank = False
            prev_blank = bool(out) and is_blank(out[-1])
            if lines[b].suffix:
                if prev_blank:
                    out.pop()
                keep.append(_closer_indent(parsed) + lines[b].suffix.strip())
            elif prev_blank and nxt < n and not lines[nxt].content.strip():
                if lines[nxt].suffix:      # blank line, then the closer line
                    out.pop()
                else:                      # blank, [deleted], blank -> keep one
                    skip_next_blank = True
            out.extend(keep)
            emit_after(b)
            i = b + 2 if skip_next_blank else b + 1
            continue
        out.append(lines[i].raw)
        emit_after(i)
        i += 1

    for leftover in after.values():  # anchor not reached (should not happen)
        out.extend(leftover)

    trailing = parsed.newlines[-1] if parsed.newlines else ""
    return nl.join(out) + trailing


def normalize_field_text(text):
    """Whitespace-insensitive form used to decide whether a field changed."""
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_field_text_loose(text):
    """Case/punctuation/whitespace-insensitive form, to separate cosmetic edits."""
    t = re.sub(r"[^\w{}@#]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()