"""
php_parser.py
-------------
Tree-sitter based extraction for PHP, run alongside parser.py's ast-based
Python extraction so gitnexus-py can index multi-language repos into one
shared graph. This is the multi-language proof point mentioned in the
README: Python via `ast`, PHP via `tree-sitter` + `tree-sitter-php`, both
feeding the same Module/Class/Function/CONTAINS/IMPORTS/CALLS/INHERITS
schema.

Design mirrors parser.py:
    - node ids use the same _mod_id/_cls_id/_func_id/_ext_id scheme, so
      PHP and Python nodes live in the same id space (they never collide,
      since rel_path always differs by extension)
    - two-stage resolution: stage 1 registers Class/Function/Method
      nodes + CONTAINS + IMPORTS; stage 2 resolves INHERITS (extends);
      stage 3 resolves CALLS (bare calls, $this->method(),
      self::/parent::/ClassName::method())
    - same "best-effort, name-based" philosophy as the Python side. PHP
      namespaces/PSR-4 autoloading aren't modeled at all: a `use
      Foo\\Bar\\Baz;` is recorded as an External dependency by name, and
      `extends`/static calls that don't resolve within the same file fall
      back to a repo-wide search by simple class name (only used if the
      name is unambiguous - i.e. exactly one class with that name exists
      in the repo).

tree-sitter and tree-sitter-php are an optional dependency: only imported
when the caller actually finds a .php file to parse (see parser.py).
"""

from __future__ import annotations

import os

import tree_sitter as ts
import tree_sitter_php as tsphp

from .parser import Graph, Node, _mod_id, _cls_id, _func_id, _ext_id, _qualname, _container_id

_PHP_LANGUAGE = ts.Language(tsphp.language_php())

_DEF_TYPES = ("function_definition", "method_declaration")
_REQUIRE_TYPES = (
    "require_expression", "require_once_expression",
    "include_expression", "include_once_expression",
)


def _text(node) -> str:
    return node.text.decode("utf8") if node is not None else ""


def _docblock_before(node) -> str:
    """PHPDoc comment (`/** ... */`) immediately preceding a class/function
    declaration, if any - tree-sitter surfaces it as a `comment` sibling
    node right before the declaration (see the parse-tree shape this was
    verified against). This is PHP's equivalent of Python's
    ast.get_docstring (see parser.py) - without it, every PHP Function/
    Class node's docstring was silently empty, even when the source has
    real developer-written documentation, which starved both TF-IDF
    (retrieval.py) and semantic search (db.vector_search) of the one
    thing they actually need: text describing what the code does.

    Only recognizes `/** ... */` blocks (the PHPDoc convention), not
    `//` or `/* */` comments generally, mirroring how ast.get_docstring
    only picks up an actual docstring literal, not arbitrary comments.
    """
    sib = getattr(node, "prev_sibling", None)
    if sib is None or sib.type != "comment":
        return ""
    raw = _text(sib).strip()
    if not raw.startswith("/**"):
        return ""
    raw = raw.removeprefix("/**").removesuffix("*/")
    lines = [line.strip().removeprefix("*").strip() for line in raw.splitlines()]
    return "\n".join(line for line in lines if line).strip()[:300]


def _string_literal_text(node) -> str | None:
    """Pull the literal text out of a PHP `string`/`encapsed_string` node
    (best-effort - doesn't attempt to evaluate interpolated strings)."""
    if node is None:
        return None
    for child in node.children:
        if child.type == "string_content":
            return _text(child)
    return None


def _parse_file(root: str, rel_path: str) -> ts.Tree | None:
    full_path = os.path.join(root, rel_path)
    try:
        with open(full_path, "rb") as fh:
            source = fh.read()
    except OSError as e:
        print(f"  [skip] {rel_path}: {e}")
        return None
    parser = ts.Parser(_PHP_LANGUAGE)
    return parser.parse(source)


def _resolve_require_target(rel_path: str, literal: str, php_file_set: set[str]) -> str | None:
    """Resolve a require/include literal path relative to the including
    file's directory (no include_path search, no extension inference)."""
    base_dir = os.path.dirname(rel_path)
    candidate = os.path.normpath(os.path.join(base_dir, literal)).replace(os.sep, "/")
    for f in php_file_set:
        if f.replace(os.sep, "/") == candidate:
            return f
    return None


def parse_php_repo(root: str, php_files: list[str], graph: Graph) -> None:
    """Parse every file in `php_files` and merge Class/Function/CONTAINS/
    IMPORTS/CALLS/INHERITS into the shared `graph` (the caller has already
    added Module nodes for these files).
    """
    php_file_set = set(php_files)
    trees: dict[str, ts.Tree] = {}
    class_bases: dict[str, str] = {}            # class_id -> raw `extends` name (unresolved)
    class_by_simple_name: dict[str, list[str]] = {}  # simple class name -> [class_id, ...]

    # ---- stage 1: Class/Function/Method nodes, CONTAINS, IMPORTS --------
    for rel_path in php_files:
        tree = _parse_file(root, rel_path)
        if tree is None:
            continue
        trees[rel_path] = tree
        mod_id = _mod_id(rel_path)

        def visit(node, scope_stack: list[str]):
            if node.type == "class_declaration":
                name = _text(node.child_by_field_name("name"))
                qual = _qualname(scope_stack, name)
                node_id = _cls_id(rel_path, qual)
                graph.add_node(Node(id=node_id, kind="Class", name=name, file=rel_path,
                                     docstring=_docblock_before(node)))
                graph.add_edge(_container_id(graph, rel_path, scope_stack), node_id, "CONTAINS")
                class_by_simple_name.setdefault(name, []).append(node_id)

                base_clause = next((c for c in node.children if c.type == "base_clause"), None)
                if base_clause is not None:
                    # the base class name has no field name in this
                    # grammar version - pick it out by child node type
                    base_name_node = next((c for c in base_clause.children if c.type == "name"), None)
                    if base_name_node is not None:
                        class_bases[node_id] = _text(base_name_node)

                body = node.child_by_field_name("body")
                if body is not None:
                    for child in body.children:
                        visit(child, scope_stack + [name])
                return

            if node.type in _DEF_TYPES:
                name = _text(node.child_by_field_name("name"))
                qual = _qualname(scope_stack, name)
                node_id = _func_id(rel_path, qual)
                graph.add_node(Node(id=node_id, kind="Function", name=name, file=rel_path,
                                     docstring=_docblock_before(node)))
                graph.add_edge(_container_id(graph, rel_path, scope_stack), node_id, "CONTAINS")
                body = node.child_by_field_name("body")
                if body is not None:
                    visit(body, scope_stack + [name])
                return

            if node.type in _REQUIRE_TYPES:
                str_node = next((c for c in node.children if c.type in ("string", "encapsed_string")), None)
                literal = _string_literal_text(str_node)
                if literal:
                    target = _resolve_require_target(rel_path, literal, php_file_set)
                    if target:
                        graph.add_edge(mod_id, _mod_id(target), "IMPORTS")
                    else:
                        ext_id = _ext_id(literal)
                        graph.add_node(Node(id=ext_id, kind="External", name=literal, file=""))
                        graph.add_edge(mod_id, ext_id, "IMPORTS")

            elif node.type == "namespace_use_clause":
                # Full PSR-4 resolution would need an autoload map we
                # don't have; record it as an external dependency by name
                # instead of guessing a file.
                qualified = _text(node)
                ext_id = _ext_id(qualified)
                graph.add_node(Node(id=ext_id, kind="External", name=qualified, file=""))
                graph.add_edge(mod_id, ext_id, "IMPORTS")

            for child in node.children:
                visit(child, scope_stack)

        visit(tree.root_node, [])

    # ---- stage 2: INHERITS - same-file by name, else unambiguous repo-wide ----
    bases_of: dict[str, list[str]] = {}
    for class_id, base_name in class_bases.items():
        _, rel_path, _qual = class_id.split("::", 2)
        same_file_id = _cls_id(rel_path, base_name)
        target = same_file_id if same_file_id in graph.nodes else None
        if target is None:
            candidates = class_by_simple_name.get(base_name, [])
            target = candidates[0] if len(candidates) == 1 else None
        if target:
            graph.add_edge(class_id, target, "INHERITS")
            bases_of.setdefault(class_id, []).append(target)

    # ---- stage 3: CALLS ---------------------------------------------------
    def find_method_in_class(class_id: str, method_name: str) -> str | None:
        _, rel_path, qual = class_id.split("::", 2)
        fid = _func_id(rel_path, f"{qual}.{method_name}")
        return fid if fid in graph.nodes else None

    def find_method_via_ancestors(class_id: str, method_name: str) -> str | None:
        seen = {class_id}
        queue = list(bases_of.get(class_id, []))
        while queue:
            base_id = queue.pop(0)
            if base_id in seen:
                continue
            seen.add(base_id)
            found = find_method_in_class(base_id, method_name)
            if found:
                return found
            queue.extend(bases_of.get(base_id, []))
        return None

    def same_file_function(rel_path: str, simple_name: str) -> str | None:
        for nid, n in graph.nodes.items():
            if n.kind == "Function" and n.file == rel_path and n.name == simple_name:
                return nid
        return None

    def repo_wide_function(simple_name: str) -> str | None:
        matches = [nid for nid, n in graph.nodes.items() if n.kind == "Function" and n.name == simple_name]
        return matches[0] if len(matches) == 1 else None

    def resolve_class_ref(rel_path: str, name: str) -> str | None:
        same_file_id = _cls_id(rel_path, name)
        if same_file_id in graph.nodes:
            return same_file_id
        candidates = class_by_simple_name.get(name, [])
        return candidates[0] if len(candidates) == 1 else None

    for rel_path, tree in trees.items():

        def visit(node, scope_stack: list[str], class_stack: list[str]):
            container_id = _container_id(graph, rel_path, scope_stack)
            in_function = container_id.startswith("function::")

            if node.type == "function_call_expression" and in_function:
                callee = _text(node.child_by_field_name("function"))
                if callee:
                    target = same_file_function(rel_path, callee) or repo_wide_function(callee)
                    if target and target != container_id:
                        graph.add_edge(container_id, target, "CALLS")

            elif node.type == "member_call_expression" and in_function and class_stack:
                obj = node.child_by_field_name("object")
                method = _text(node.child_by_field_name("name"))
                if method and _text(obj) == "$this":
                    class_id = _cls_id(rel_path, ".".join(class_stack))
                    target = find_method_in_class(class_id, method) or find_method_via_ancestors(class_id, method)
                    if target and target != container_id:
                        graph.add_edge(container_id, target, "CALLS")

            elif node.type == "scoped_call_expression" and in_function and class_stack:
                scope_text = _text(node.child_by_field_name("scope"))
                method = _text(node.child_by_field_name("name"))
                class_id = _cls_id(rel_path, ".".join(class_stack))
                target = None
                if scope_text == "parent":
                    target = find_method_via_ancestors(class_id, method)
                elif scope_text in ("self", "static"):
                    target = find_method_in_class(class_id, method) or find_method_via_ancestors(class_id, method)
                elif scope_text:
                    target_class = resolve_class_ref(rel_path, scope_text)
                    if target_class:
                        target = find_method_in_class(target_class, method) or find_method_via_ancestors(target_class, method)
                if target and target != container_id:
                    graph.add_edge(container_id, target, "CALLS")

            if node.type == "class_declaration":
                name = _text(node.child_by_field_name("name"))
                body = node.child_by_field_name("body")
                if body is not None:
                    for child in body.children:
                        visit(child, scope_stack + [name], class_stack + [name])
                return

            if node.type in _DEF_TYPES:
                name = _text(node.child_by_field_name("name"))
                body = node.child_by_field_name("body")
                if body is not None:
                    visit(body, scope_stack + [name], class_stack)
                return

            for child in node.children:
                visit(child, scope_stack, class_stack)

        visit(tree.root_node, [], [])
