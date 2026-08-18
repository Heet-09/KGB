"""
route_extractor.py
-------------------
Second extractor in the same "mine a specific idiom deterministically, no
LLM anywhere" style as rbac_extractor.py - this one recovers a URL -> page
mapping, which nothing else in the graph knows.

Investigated before writing this (see the plan): naively parsing
`Config/Routes.php` files for literal `$routes->get(...)` calls covers only
a fraction of this repo's routes. Most of them are generated at *runtime*
by two loops that scan a module's `Controllers/` folder and build a route
per file found - fully static in the sense that they only depend on which
Controller files exist (which parser.discover_php_files already knows), so
they're reproduced here in Python rather than actually run as PHP:

    pass A (explicit): tree-sitter walk over every Routes.php for literal
        `$routes->get/post/put/delete/add(pattern, target)` calls (method
        name matched regardless of the receiver variable - `$routes` at the
        top level, `$route` inside a group() closure, doesn't matter) and
        `group(prefix, [options], closure)` calls, whose prefix/namespace
        get attached (via ancestor walk, not sequential statement replay)
        to every route call nested inside. Calls whose pattern/target
        aren't string literals (i.e. built from variables/concatenation -
        the generator loops below) are silently skipped here; they're
        Pass B's job instead.

    pass B (convention): mirrors the two generator loops found in
        app/Config/Routes.php and app/Modules/Ilabo/Config/Routes.php -
        for every Controller file already discovered under
        `Modules/<X>/Controllers/` (X != Ilabo) and separately under
        `Modules/Ilabo/Controllers/Bo/`, synthesizes the two routes each
        loop iteration would produce (`{module}/{controller}` -> `::index`,
        `{module}/{controller}/(:any)` -> `::$1`).

Both passes emit the same row shape: {pattern, http_method, target, kind,
source_file, lineno}. `target` is kept as the raw string the source
encodes, `$1`/`$2` placeholders included - see db.resolve_url for why
substitution happens at question time instead of here.
"""

from __future__ import annotations

import re

import tree_sitter as ts

from .php_parser import _parse_file, _text, _string_literal_text

_ROUTE_VERBS = {"get", "post", "put", "delete", "patch", "add", "match"}


def _walk(node):
    yield node
    for c in node.children:
        yield from _walk(c)


def _call_method_name(node: ts.Node) -> str | None:
    """Method name of a `$obj->method(...)` call, regardless of the
    receiver variable's name - `$routes->group(...)` and the `$route`
    passed into its closure are both just "group" here."""
    if node.type != "member_call_expression":
        return None
    name_node = node.child_by_field_name("name")
    return _text(name_node) if name_node is not None else None


def _call_args(node: ts.Node) -> list[ts.Node]:
    """The actual expression node inside each `argument` wrapper - not the
    wrapper itself, mirroring how rbac_extractor.py unwraps arguments."""
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


def _array_string_value(node: ts.Node | None, key: str) -> str | None:
    """Pull `'key' => 'value'` out of a PHP `array('key' => 'value', ...)`
    literal - used for a group()'s `array('namespace' => '...')` option.
    tree-sitter-php exposes no field names for array_element_initializer's
    key/value (same quirk rbac_extractor.py notes for if_statement/
    base_clause), so this uses the same positional-fallback approach:
    first child is the key expression, last child is the value."""
    if node is None or node.type != "array_creation_expression":
        return None
    for elem in node.children:
        if elem.type != "array_element_initializer" or len(elem.children) < 2:
            continue
        k = _string_literal(elem.children[0])
        if k == key:
            return _string_literal(elem.children[-1])
    return None


def _ancestor_route_scope(node: ts.Node) -> tuple[str, str | None]:
    """Walk up from a route-verb call collecting every enclosing group()'s
    prefix (outermost first, joined with '/') and the closest ancestor
    group's own 'namespace' option (if any) - being a descendant of a
    group() call's argument list is exactly what "nested inside that
    group's closure" means here, no sequential-statement tracking needed."""
    groups: list[tuple[str, str | None]] = []
    p = node.parent
    while p is not None:
        if _call_method_name(p) == "group":
            args = _call_args(p)
            prefix = _string_literal(args[0]) if args else None
            namespace = _array_string_value(args[1], "namespace") if len(args) > 1 else None
            groups.append((prefix or "", namespace))
        p = p.parent
    groups.reverse()  # outermost first
    combined_prefix = "/".join(seg.strip("/") for seg, _ in groups if seg.strip("/"))
    closest_namespace = next((ns for _, ns in reversed(groups) if ns), None)
    return combined_prefix, closest_namespace


def _default_namespace(root_node: ts.Node) -> str | None:
    """First `setDefaultNamespace('...')` call in the file - every observed
    Routes.php in this repo calls it exactly once near the top, so "first
    one found, applies file-wide" is a safe simplification (no need to
    replay statement order)."""
    for node in _walk(root_node):
        if _call_method_name(node) == "setDefaultNamespace":
            args = _call_args(node)
            if args:
                return _string_literal(args[0])
    return None


def _qualify_target(target: str, namespace: str | None) -> str:
    """A bare target like 'Dashboard::index' is relative to the file's (or
    enclosing group's) namespace; '\\Fully\\Qualified\\Class::method' is
    already absolute."""
    if target.startswith("\\") or namespace is None:
        return target
    class_part, sep, method_part = target.partition("::")
    if "\\" in class_part:
        return target  # already namespaced, just missing the leading backslash
    return f"{namespace.rstrip(chr(92))}\\{class_part}" + (f"{sep}{method_part}" if sep else "")


def _extract_explicit(root: str, rel_path: str) -> list[dict]:
    tree = _parse_file(root, rel_path)
    if tree is None:
        return []
    default_ns = _default_namespace(tree.root_node)

    rows = []
    for node in _walk(tree.root_node):
        method = _call_method_name(node)
        if method not in _ROUTE_VERBS:
            continue
        args = _call_args(node)
        if len(args) < 2:
            continue
        own_pattern = _string_literal(args[0])
        target = _string_literal(args[1])
        if own_pattern is None or target is None:
            continue  # built from a variable/concatenation - Pass B's territory, not this one

        prefix, group_ns = _ancestor_route_scope(node)
        full_pattern = "/".join(seg for seg in (prefix, own_pattern.strip("/")) if seg)
        namespace = group_ns or default_ns
        rows.append({
            "pattern": full_pattern, "http_method": method.upper(),
            "target": _qualify_target(target, namespace), "kind": "explicit",
            "source_file": rel_path, "lineno": node.start_point[0] + 1,
        })
    return rows


# The two runtime route generators actually present in this repo (see
# module docstring) - kept as data, not a generic "any CI4 auto-route"
# engine, matching rbac_extractor.py's "only patterns actually observed"
# philosophy.
_MODULE_CONTROLLER_RE = re.compile(r"^app[/\\]Modules[/\\]([^/\\]+)[/\\]Controllers[/\\]([^/\\]+)\.php$", re.IGNORECASE)
_ILABO_BO_CONTROLLER_RE = re.compile(r"^app[/\\]Modules[/\\]Ilabo[/\\]Controllers[/\\]Bo[/\\]([^/\\]+)\.php$", re.IGNORECASE)


def _lcfirst(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


def _extract_convention(php_files: list[str]) -> list[dict]:
    rows = []
    for rel_path in php_files:
        norm = rel_path.replace("\\", "/")

        m = _MODULE_CONTROLLER_RE.match(norm)
        if m and m.group(1).lower() != "ilabo":
            module, controller = m.group(1), m.group(2)
            seg = _lcfirst(controller)
            module_lc = module.lower()
            target_ns = f"App\\Modules\\{module}\\Controllers\\{controller}"
            for pattern, target in (
                (f"{module_lc}/{seg}", f"{target_ns}::index"),
                (f"{module_lc}/{seg}/(:any)", f"{target_ns}::$1"),
            ):
                rows.append({"pattern": pattern, "http_method": "GET", "target": target,
                             "kind": "convention", "source_file": "app/Config/Routes.php", "lineno": 0})
            continue

        m = _ILABO_BO_CONTROLLER_RE.match(norm)
        if m:
            controller = m.group(1)
            seg = _lcfirst(controller)
            target_ns = f"App\\Modules\\Ilabo\\Controllers\\Bo\\{controller}"
            for pattern, target in (
                (f"ilabo/bo/{seg}", f"{target_ns}::index"),
                (f"ilabo/bo/{seg}/(:any)", f"{target_ns}::$1"),
            ):
                rows.append({"pattern": pattern, "http_method": "GET", "target": target,
                             "kind": "convention", "source_file": "app/Modules/Ilabo/Config/Routes.php", "lineno": 0})
    return rows


def extract(root: str, php_files: list[str]) -> list[dict]:
    """Run both passes. `php_files` is the already-discovered list from
    parser.discover_php_files - Pass B reads directly from it (no re-glob
    of disk), Pass A parses just the Routes.php files within it."""
    routes_files = [f for f in php_files if f.replace("\\", "/").rstrip("/").endswith("Config/Routes.php")]

    rows: list[dict] = []
    for rel_path in routes_files:
        rows.extend(_extract_explicit(root, rel_path))
    rows.extend(_extract_convention(php_files))
    return rows
