"""
view_extractor.py
------------------
Third extractor in the same deterministic, no-LLM idiom-mining style as
rbac_extractor.py / route_extractor.py: recovers "which view does this page
render" facts.

View templates are already indexed as ordinary Module nodes (they're just
.php files under */Views/, discovered the same as any other PHP file) - the
only thing missing is the *edge*: CodeIgniter's `view('name', $data)` /
`view('App\\Modules\\Icab\\Views\\...', $data)` calls are bare calls to a
global helper function, so today's CALLS resolution in php_parser.py (which
only handles self/cls/same-file/import-table calls) never links them to the
target Module.

Walks every already-discovered PHP file for `view(...)` call expressions
with a string-literal first argument, and resolves that string to a real
indexed Module file:
    - namespaced form ('App\\Modules\\Icab\\Views\\Configuration\\list') ->
      convert to a path the same way route_extractor/rbac_extractor do
      (App\\ -> app/, backslashes -> path separators, .php appended) and
      match directly against known Module files.
    - bare form ('dashboard') -> try the calling file's own module's Views
      dir first, then fall back to an unambiguous repo-wide basename match
      - the same "best-effort, name-based, only when unambiguous"
      resolution philosophy php_parser.py already documents and uses for
      INHERITS/CALLS.
Unresolved calls are still recorded (raw string, resolved=False) rather
than dropped - see db.py's Render table docstring for why.
"""

from __future__ import annotations

import re

import tree_sitter as ts

from .php_parser import _parse_file, _text, _string_literal_text

_DEF_TYPES = ("function_definition", "method_declaration")


def _walk(node):
    yield node
    for c in node.children:
        yield from _walk(c)


def _call_function_name(node: ts.Node) -> str | None:
    """Bare function-call name, e.g. `view(...)` - a plain
    function_call_expression, not a ->method() call."""
    if node.type != "function_call_expression":
        return None
    fn_node = node.child_by_field_name("function")
    return _text(fn_node) if fn_node is not None else None


def _call_args(node: ts.Node) -> list[ts.Node]:
    args_node = node.child_by_field_name("arguments")
    if args_node is None:
        return []
    out = []
    for arg in args_node.children:
        if arg.type == "argument" and arg.children:
            out.append(arg.children[0])
    return out


def _string_literal(node: ts.Node | None) -> str | None:
    if node is None or node.type not in ("string", "encapsed_string"):
        return None
    return _string_literal_text(node)


def _enclosing_kind(node: ts.Node) -> str:
    """"Function" if this call is textually inside a function/method body,
    else "Module" (top-level template code) - same fallback shape
    rbac_extractor.py's _enclosing_target uses, simplified since Render
    facts are keyed by file, not by the specific enclosing function id."""
    p = node.parent
    while p is not None:
        if p.type in _DEF_TYPES:
            return "Function"
        p = p.parent
    return "Module"


def _views_dir_for(rel_path: str) -> str | None:
    """The Views directory a bare view name would be resolved from from
    within `rel_path` - CI4 convention: a Controller/Module's own
    `Controllers/` sibling `Views/` dir, or a View resolving another
    partial relative to its own directory."""
    norm = rel_path.replace("\\", "/")
    if "/Controllers" in norm:
        base = norm.rsplit("/Controllers", 1)[0]
        return f"{base}/Views"
    return "/".join(norm.split("/")[:-1]) or None


def _resolve_view(rel_path: str, view_arg: str, known_files: dict[str, str]) -> str | None:
    if "\\" in view_arg or "/" in view_arg:
        path = view_arg.replace("\\", "/").lstrip("/")
        if path.lower().startswith("app/"):
            path = "app/" + path[len("app/"):]
        candidate = path if path.lower().endswith(".php") else path + ".php"
        return known_files.get(candidate)

    views_dir = _views_dir_for(rel_path)
    if views_dir:
        candidate = f"{views_dir}/{view_arg}.php"
        hit = known_files.get(candidate)
        if hit:
            return hit

    target_suffix = "/" + view_arg + ".php"
    matches = [orig for norm, orig in known_files.items()
               if norm.endswith(target_suffix) or norm == view_arg + ".php"]
    return matches[0] if len(matches) == 1 else None


def extract(root: str, php_files: list[str]) -> list[dict]:
    """Run the extraction over every already-discovered PHP file (every one
    of which becomes a Module node - see php_parser.py - so `php_files`
    doubles as the "known files a view name might resolve to" set, no
    separate DB query needed)."""
    known_files = {f.replace("\\", "/"): f for f in php_files}
    rows: list[dict] = []

    for rel_path in php_files:
        tree = _parse_file(root, rel_path)
        if tree is None:
            continue
        for node in _walk(tree.root_node):
            if _call_function_name(node) != "view":
                continue
            args = _call_args(node)
            if not args:
                continue
            view_arg = _string_literal(args[0])
            if not view_arg:
                continue
            target_file = _resolve_view(rel_path, view_arg, known_files)
            rows.append({
                "source_file": rel_path, "source_kind": _enclosing_kind(node),
                "lineno": node.start_point[0] + 1, "view_arg": view_arg,
                "target_file": target_file, "resolved": target_file is not None,
            })
    return rows
