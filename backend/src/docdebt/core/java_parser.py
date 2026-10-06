"""Java source parsing with tree-sitter.

tree-sitter replaces the ``javalang`` parser used in the research code for
three reasons: it tolerates syntax errors (a pull request can contain code
that does not compile yet), it supports recent Java syntax (records,
switch expressions, text blocks), and it reports exact byte offsets, so
the method source and its Javadoc are cut out without the brace-counting
heuristic of the research ``extract_node_source``, which swallowed the
next method for bodiless (abstract or interface) methods.

``extract_signature_from_method_code`` keeps the research function name
and contract, so benchmark code written against the research module keeps
working. ``scripts/parity_check.py`` compares both implementations on the
benchmark.
"""

from __future__ import annotations

import dataclasses
import re
from functools import lru_cache

import tree_sitter_java
from tree_sitter import Language, Node, Parser

from docdebt.core.models import MethodSignature, MethodUnit, Parameter, Visibility

_METHOD_NODES = {"method_declaration", "constructor_declaration"}
_TYPE_NODES = {
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
    "annotation_type_declaration",
}
_INTERFACE_BODIES = {"interface_body", "annotation_type_body"}
_WRAPPER_CLASS = "_DocDebtWrapper"


# The Language object is kept at module level on purpose: the parser must
# never outlive the grammar it points to.
_JAVA_LANGUAGE = Language(tree_sitter_java.language())


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(_JAVA_LANGUAGE)


class _Source:
    """The encoded source kept alive next to its tree.

    Text is always sliced from these bytes by offset instead of reading
    ``Node.text``: with some py-tree-sitter releases, ``Node.text`` reads
    a buffer that is freed once the temporary ``bytes`` passed to
    ``Parser.parse`` goes out of scope, which crashes the interpreter.
    """

    def __init__(self, source: str) -> None:
        self.data = source.encode("utf-8")
        self.tree = _parser().parse(self.data)

    def indent_of(self, node: Node) -> str:
        """Whitespace between the start of the node's line and the node itself."""
        line_start = self.data.rfind(b"\n", 0, node.start_byte) + 1
        prefix = self.data[line_start : node.start_byte].decode("utf-8", errors="replace")
        return prefix if not prefix.strip() else ""

    def text(self, node: Node | None) -> str:
        if node is None:
            return ""
        return self.data[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _child_of_type(node: Node, type_name: str) -> Node | None:
    for child in node.children:
        if child.type == type_name:
            return child
    return None


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


def _parameter(src: _Source, node: Node) -> Parameter | None:
    """Build a Parameter from a formal_parameter or spread_parameter node.

    Receiver parameters (``Foo this``) are not real parameters for Javadoc
    purposes and return ``None``.
    """
    if node.type == "formal_parameter":
        name = src.text(node.child_by_field_name("name"))
        type_text = _squash(src.text(node.child_by_field_name("type")))
        dims = node.child_by_field_name("dimensions")
        if dims is not None:  # C-style array: int values[]
            type_text += _squash(src.text(dims))
        return Parameter(name=name, type=type_text) if name else None

    if node.type == "spread_parameter":  # varargs: String... args
        declarator = _child_of_type(node, "variable_declarator")
        name = src.text(declarator.child_by_field_name("name")) if declarator else ""
        type_node = next(
            (c for c in node.named_children if c.type not in {"modifiers", "variable_declarator"}),
            None,
        )
        type_text = _squash(src.text(type_node)) + "..."
        return Parameter(name=name, type=type_text) if name else None

    return None


def _signature(src: _Source, node: Node) -> MethodSignature:
    is_constructor = node.type == "constructor_declaration"
    params_node = node.child_by_field_name("parameters")
    parameters: list[Parameter] = []
    if params_node is not None:
        for child in params_node.named_children:
            param = _parameter(src, child)
            if param is not None:
                parameters.append(param)

    throws_node = _child_of_type(node, "throws")
    throws = tuple(_squash(src.text(c)) for c in throws_node.named_children) if throws_node else ()

    return_type = None if is_constructor else _squash(src.text(node.child_by_field_name("type")))
    return MethodSignature(
        name=src.text(node.child_by_field_name("name")),
        parameters=tuple(parameters),
        return_type=return_type,
        throws=throws,
        is_constructor=is_constructor,
    )


# ---------------------------------------------------------------------------
# Context around a method node
# ---------------------------------------------------------------------------


def _javadoc_node(src: _Source, node: Node) -> Node | None:
    """The ``/** ... */`` comment immediately preceding the declaration."""
    previous = node.prev_named_sibling
    if previous is not None and previous.type == "block_comment" and src.text(previous).startswith("/**"):
        return previous
    return None


def _enclosing_class(src: _Source, node: Node) -> str:
    names: list[str] = []
    parent = node.parent
    while parent is not None:
        if parent.type in _TYPE_NODES:
            names.append(src.text(parent.child_by_field_name("name")))
        elif parent.type == "object_creation_expression" and _child_of_type(parent, "class_body"):
            names.append("<anonymous>")
        parent = parent.parent
    return ".".join(reversed(names))


def _visibility(node: Node) -> Visibility:
    modifiers = _child_of_type(node, "modifiers")
    tokens = {c.type for c in modifiers.children} if modifiers else set()
    if "private" in tokens:
        return Visibility.PRIVATE
    if "public" in tokens:
        return Visibility.PUBLIC
    if "protected" in tokens:
        return Visibility.PROTECTED
    if node.parent is not None and node.parent.type in _INTERFACE_BODIES:
        return Visibility.PUBLIC  # interface members are implicitly public
    return Visibility.PACKAGE


def _iter_method_nodes(root: Node):
    """Depth-first, source-ordered walk over every method and constructor."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in _METHOD_NODES:
            yield node
        stack.extend(reversed(node.children))


def _method_unit(src: _Source, node: Node, file_path: str) -> MethodUnit:
    doc_node = _javadoc_node(src, node)
    return MethodUnit(
        file_path=file_path,
        class_name=_enclosing_class(src, node),
        signature=_signature(src, node),
        code=src.text(node),
        javadoc=src.text(doc_node) if doc_node else None,
        start_line=node.start_point.row + 1,
        end_line=node.end_point.row + 1,
        javadoc_start_line=(doc_node.start_point.row + 1) if doc_node else None,
        visibility=_visibility(node),
        has_body=node.child_by_field_name("body") is not None,
        syntax_ok=not node.has_error,
        indent=src.indent_of(node),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_java_source(source: str, file_path: str = "<memory>") -> list[MethodUnit]:
    """Every method and constructor of a Java compilation unit, in source order.

    Syntax errors do not stop parsing: affected methods are still returned,
    with ``syntax_ok=False`` so callers can decide to skip them.
    """
    src = _Source(source)
    return [_method_unit(src, node, file_path) for node in _iter_method_nodes(src.tree.root_node)]


def parse_method_snippet(code: str, file_path: str = "<snippet>") -> MethodUnit | None:
    """Parse the source of a single method, optionally preceded by its Javadoc.

    The snippet is first parsed as-is (tree-sitter accepts a bare method
    declaration); if that fails, it is wrapped in a dummy class, mirroring
    the research implementation. Returns the first method found, or
    ``None`` when nothing parses.
    """
    if not code or not code.strip():
        return None
    code = code.strip("\n")

    bare = parse_java_source(code, file_path)
    if bare and bare[0].syntax_ok:
        return bare[0]

    wrapped = parse_java_source(f"class {_WRAPPER_CLASS} {{\n{code}\n}}", file_path)
    if wrapped:
        unit = wrapped[0]
        class_name = unit.class_name.removeprefix(_WRAPPER_CLASS).lstrip(".")
        return dataclasses.replace(
            unit,
            class_name=class_name,
            start_line=unit.start_line - 1,
            end_line=unit.end_line - 1,
            javadoc_start_line=(unit.javadoc_start_line - 1) if unit.javadoc_start_line else None,
        )
    return bare[0] if bare else None


def extract_signature_from_method_code(method_code: str) -> MethodSignature | None:
    """Signature of a single method's source, or ``None`` if unparsable.

    Same name and contract as the research function, so benchmark code can
    switch implementations without other changes. Code with syntax errors
    returns ``None``, as in the research version.
    """
    unit = parse_method_snippet(method_code)
    if unit is None or not unit.syntax_ok:
        return None
    return unit.signature


def invoked_method_names(code: str) -> set[str]:
    """Names of all methods invoked inside a piece of Java code.

    Name-only (no type resolution), exactly like the research retrieval.
    """
    src = _Source(code)
    names: set[str] = set()
    stack = [src.tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "method_invocation":
            name = src.text(node.child_by_field_name("name"))
            if name:
                names.add(name)
        stack.extend(node.children)
    return names
