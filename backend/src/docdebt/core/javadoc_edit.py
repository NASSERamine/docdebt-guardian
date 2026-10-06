"""Field-level editing of a Javadoc, leaving everything else untouched.

A Javadoc is split into *blocks*: the description, then one block per tag
(``@param``, ``@return``, ``@throws``, or any other tag such as ``@see``).
An ``EditPlan`` names the fields that may change. ``apply_edits`` builds a
new Javadoc where:

* fields named in the plan take their text from a model-written Javadoc;
* every other block is copied from the original.

In the usual layout (``/**`` alone on its line, one ``*`` per line, ``*/``
alone on its line), untouched blocks keep their original source lines
byte for byte, so preservation of already-correct documentation holds by
construction and does not depend on what the model wrote.

Unusual layouts (single-line comments, tags in the middle of a line) are
re-rendered in the standard multi-line layout: the text of every untouched
field is preserved, but its line breaks and indentation may change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from docdebt.core.javadoc import simple_exception_name
from docdebt.core.models import MethodSignature

_BLOCK_TAG = re.compile(r"^@(\w+)")
_PARAM = re.compile(r"^@param\s+(\w+)")
_RETURN = re.compile(r"^@return\b")
_THROWS = re.compile(r"^@(?:throws|exception)\s+(\S+)")
_STAR_LINE = re.compile(r"^(\s*)\*(?!/)")
_MID_LINE_TAG = re.compile(
    r"(?<=\S)[ \t]+(?=@(?:param|return|throws|exception|see|since|author|version|deprecated|serial\w*)\b)"
)

DESCRIPTION = "description"
PARAM = "param"
RETURN = "return"
THROWS = "throws"
OTHER = "other"
SLOT = "_slot"  # placeholder left where a block was removed, reused by its replacement


@dataclass(frozen=True)
class FieldRef:
    """One editable field: the description, a ``@param``, ``@return`` or ``@throws``."""

    kind: str
    name: str | None = None

    @property
    def label(self) -> str:
        if self.kind == DESCRIPTION:
            return "description"
        if self.kind == RETURN:
            return "@return"
        return f"@{self.kind} {self.name}"


@dataclass
class DocBlock:
    """The description or one tag of a Javadoc.

    ``content`` holds the text lines without comment markers; ``raw`` holds
    the original source lines (markers included) and is ``None`` for blocks
    that were built or re-rendered.
    """

    kind: str
    name: str | None
    content: list[str]
    raw: list[str] | None = None

    @property
    def ref(self) -> FieldRef | None:
        return None if self.kind in (OTHER, SLOT) else FieldRef(self.kind, self.name)


@dataclass
class DocLayout:
    """A parsed Javadoc: its blocks plus what is needed to render it back."""

    blocks: list[DocBlock]
    open_line: str
    close_line: str
    indent: str
    mode: str  # "raw" (source lines preserved) or "normalized"
    trailing_newline: bool = False


@dataclass
class EditPlan:
    """Fields that may change. Anything not listed here must stay identical."""

    add: set[FieldRef] = field(default_factory=set)
    remove: set[FieldRef] = field(default_factory=set)
    replace: set[FieldRef] = field(default_factory=set)

    @property
    def touched(self) -> set[FieldRef]:
        return self.add | self.remove | self.replace

    @property
    def wanted_from_model(self) -> set[FieldRef]:
        return self.add | self.replace

    @property
    def is_empty(self) -> bool:
        return not self.touched

    def normalize(self) -> EditPlan:
        """Resolve overlaps: a removal wins, and an addition wins over a replacement."""
        self.replace -= self.remove
        self.replace -= self.add
        self.add -= self.remove
        return self


@dataclass
class FieldChange:
    ref: FieldRef
    action: str  # "add", "remove" or "replace"
    before: str | None
    after: str | None


@dataclass
class MergeResult:
    text: str
    changes: list[FieldChange] = field(default_factory=list)
    missing_in_model: list[FieldRef] = field(default_factory=list)
    unchanged: list[FieldRef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _classify(first_line: str) -> tuple[str, str | None]:
    text = first_line.lstrip()
    m = _PARAM.match(text)
    if m:
        return PARAM, m.group(1)
    if _RETURN.match(text):
        return RETURN, None
    m = _THROWS.match(text)
    if m:
        return THROWS, simple_exception_name(m.group(1))
    m = _BLOCK_TAG.match(text)
    return OTHER, (m.group(1) if m else None)


def _split_blocks(contents: list[str], raws: list[str] | None) -> list[DocBlock]:
    blocks: list[DocBlock] = []
    current = DocBlock(DESCRIPTION, None, [], [] if raws is not None else None)
    for i, content in enumerate(contents):
        if _BLOCK_TAG.match(content.lstrip()):
            if current.content:
                blocks.append(current)
            kind, name = _classify(content)
            current = DocBlock(kind, name, [], [] if raws is not None else None)
        current.content.append(content)
        if raws is not None and current.raw is not None:
            current.raw.append(raws[i])
    if current.content:
        blocks.append(current)
    return blocks


def _normalized_contents(text: str) -> list[str]:
    body = re.sub(r"\*/$", "", re.sub(r"^/\*\*", "", text.strip()))
    out: list[str] = []
    for i, line in enumerate(body.split("\n")):
        if i == 0:
            content = line.strip()
        else:
            m = re.match(r"^\s*\*\s?", line)
            content = (line[m.end() :] if m else line.strip()).rstrip()
        out.extend(_MID_LINE_TAG.split(content) if content else [""])
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def parse_layout(doc_text: str, base_indent: str = "") -> DocLayout:
    """Split a raw Javadoc into blocks.

    ``base_indent`` is the indentation used when a comment has to be
    re-rendered (single-line comments, tags in mid-line).
    """
    trailing_newline = doc_text.endswith("\n")
    lines = doc_text.rstrip("\n").split("\n")
    if (
        len(lines) >= 2
        and lines[0].strip() == "/**"
        and lines[-1].strip() == "*/"
        and all(_STAR_LINE.match(line) for line in lines[1:-1])
    ):
        middle = lines[1:-1]
        contents = [re.sub(r"^\s*\*\s?", "", line) for line in middle]
        if not any(_MID_LINE_TAG.search(c) for c in contents):
            star = _STAR_LINE.match(middle[0]) if middle else None
            indent = star.group(1) if star else lines[-1][: len(lines[-1]) - len(lines[-1].lstrip())]
            return DocLayout(
                blocks=_split_blocks(contents, middle),
                open_line=lines[0],
                close_line=lines[-1],
                indent=indent,
                mode="raw",
                trailing_newline=trailing_newline,
            )
    indent = base_indent + " "
    return DocLayout(
        blocks=_split_blocks(_normalized_contents(doc_text), None),
        open_line="/**",
        close_line=indent + "*/",
        indent=indent,
        mode="normalized",
        trailing_newline=trailing_newline,
    )


# ---------------------------------------------------------------------------
# Field text helpers
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def field_text(block: DocBlock) -> str:
    """The prose of a block, without its tag and name, whitespace collapsed."""
    text = " ".join(line.strip() for line in block.content if line.strip())
    if block.kind == PARAM:
        text = re.sub(r"^@param\s+\w+\s*", "", text)
    elif block.kind == RETURN:
        text = re.sub(r"^@return\s*", "", text)
    elif block.kind == THROWS:
        text = re.sub(r"^@(?:throws|exception)\s+\S+\s*", "", text)
    return normalize_text(text)


def _render_content(content: list[str], indent: str) -> list[str]:
    return [f"{indent}* {c}" if c.strip() else f"{indent}*" for c in content]


def render_layout(layout: DocLayout) -> str:
    lines = [layout.open_line]
    for block in layout.blocks:
        lines.extend(block.raw if block.raw is not None else _render_content(block.content, layout.indent))
    lines.append(layout.close_line)
    return "\n".join(lines) + ("\n" if layout.trailing_newline else "")


def _find(blocks: list[DocBlock], ref: FieldRef) -> int | None:
    for i, block in enumerate(blocks):
        if block.ref == ref:
            return i
    return None


def _trailing_blank(content: list[str]) -> list[str]:
    tail: list[str] = []
    for line in reversed(content):
        if line.strip():
            break
        tail.append(line)
    return tail


def _strip_trailing_blank(content: list[str]) -> list[str]:
    end = len(content)
    while end and not content[end - 1].strip():
        end -= 1
    return content[:end]


def _rank(kind: str, name: str | None, signature: MethodSignature) -> tuple[int, int]:
    if kind == DESCRIPTION:
        return (0, 0)
    if kind == PARAM:
        names = signature.param_names
        return (1, names.index(name) if name in names else len(names) + 1000)
    if kind == RETURN:
        return (2, 0)
    if kind == THROWS:
        simple = [simple_exception_name(t) for t in signature.throws]
        return (3, simple.index(name) if name in simple else len(simple) + 1000)
    return (4, 0)


# ---------------------------------------------------------------------------
# Applying a plan
# ---------------------------------------------------------------------------


def _ordered(refs: set[FieldRef], signature: MethodSignature) -> list[FieldRef]:
    return sorted(refs, key=lambda r: (*_rank(r.kind, r.name, signature), r.label))


def apply_edits(
    original: str,
    model_doc: str,
    plan: EditPlan,
    signature: MethodSignature,
    *,
    base_indent: str = "",
) -> MergeResult:
    """Build the repaired Javadoc.

    Args:
        original: the current Javadoc.
        model_doc: a complete Javadoc written by the model; only the fields
            named in ``plan`` are read from it.
        plan: fields that may be added, removed or replaced.
        signature: the method signature, used to order new tags.
        base_indent: indentation for re-rendered comments.

    Returns:
        The merged text, the applied changes, the fields the model failed to
        provide, and the requested replacements where the model kept the
        original text.
    """
    layout = parse_layout(original, base_indent)
    model = parse_layout(model_doc, base_indent)
    model_blocks = {b.ref: b for b in model.blocks if b.ref is not None}
    blocks = layout.blocks
    result = MergeResult(text=original)

    def replace_block(index: int, ref: FieldRef, source: DocBlock) -> None:
        old = blocks[index]
        before, after = field_text(old), field_text(source)
        if normalize_text(before) == normalize_text(after):
            result.unchanged.append(ref)
            return
        content = _strip_trailing_blank(source.content) + _trailing_blank(old.content)
        blocks[index] = DocBlock(old.kind, old.name, content, None)
        result.changes.append(FieldChange(ref, "replace", before, after))

    def insert_block(ref: FieldRef, source: DocBlock) -> None:
        # A renamed tag takes the place of the tag it replaces, so the diff stays minimal.
        for i, block in enumerate(blocks):
            if block.kind == SLOT and block.name == ref.kind:
                blocks[i] = DocBlock(source.kind, source.name, _strip_trailing_blank(source.content), None)
                result.changes.append(FieldChange(ref, "add", None, field_text(source)))
                return
        rank = _rank(ref.kind, ref.name, signature)
        position = 0
        for i, block in enumerate(blocks):
            if block.kind not in (OTHER, SLOT) and _rank(block.kind, block.name, signature) <= rank:
                position = i + 1
        has_tags = any(b.kind not in (DESCRIPTION, SLOT) for b in blocks)
        if ref.kind != DESCRIPTION and not has_tags and blocks and blocks[0].kind == DESCRIPTION:
            head = blocks[0]
            if head.content and head.content[-1].strip():
                head.content.append("")
                if head.raw is not None:
                    head.raw.append(f"{layout.indent}*")
        content = _strip_trailing_blank(source.content)
        if ref.kind == DESCRIPTION and blocks:
            content = [*content, ""]
        blocks.insert(position, DocBlock(source.kind, source.name, content, None))
        result.changes.append(FieldChange(ref, "add", None, field_text(source)))

    for ref in _ordered(plan.remove, signature):
        index = _find(blocks, ref)
        if index is not None:
            removed = blocks[index]
            blocks[index] = DocBlock(SLOT, removed.kind, [], None)
            result.changes.append(FieldChange(ref, "remove", field_text(removed), None))

    for ref in _ordered(plan.replace | plan.add, signature):
        source = model_blocks.get(ref)
        if source is None:
            result.missing_in_model.append(ref)
            continue
        index = _find(blocks, ref)
        if index is None:
            insert_block(ref, source)
        else:
            replace_block(index, ref, source)

    layout.blocks = [b for b in blocks if b.kind != SLOT]
    if result.changes:
        result.text = render_layout(layout)
    return result
