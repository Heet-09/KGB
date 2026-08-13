"""
parser.py
---------
Walks a Python codebase and extracts a code graph using the built-in `ast`
module. No LLM calls happen here - this is pure static analysis, so it is
fast and deterministic (this is the same approach GitNexus uses, just
Python-only instead of tree-sitter multi-language).

Extracted entities (nodes):
    - Module   (a .py file)
    - Class
    - Function (includes methods)

Extracted relationships (edges):
    - CONTAINS   Module -> Class / Function
    - CONTAINS   Class  -> Function (methods)
    - IMPORTS    Module -> Module
    - CALLS      Function -> Function
    - INHERITS   Class -> Class

Two-pass design
----------------
Pass 1 (_DefVisitor) walks every file and registers all Module/Class/Function
nodes, CONTAINS edges, IMPORTS edges, and each file's import-alias table
(local name -> where it came from). Pass 2 (_RefVisitor) then re-walks every
file to resolve CALLS and INHERITS, now that the *entire* repo's nodes and
import tables are known. This buys two things over a single combined pass:
  - forward references within a file resolve (a function calling another
    function defined later in the same file no longer gets skipped), and
  - calls/bases that go through an import (e.g. `from .db import open_db`,
    `import parser; parser.parse_repo()`) can be resolved across files,
    not just within the same file.
Resolution is still name-based best-effort, not real type inference: calls
through arbitrary object instances (e.g. `some_obj.method()` where some_obj
isn't a known module alias) still fall back to same-file simple-name
matching, same as before.
"""

from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import dataclass, field


@dataclass
class Node:
    id: str            # unique id, e.g. "module::path/to/file.py"
    kind: str           # "Module" | "Class" | "Function"
    name: str
    file: str
    lineno: int = 0
    end_lineno: int = 0
    docstring: str = ""


@dataclass
class Edge:
    src: str
    dst: str
    kind: str            # "CONTAINS" | "IMPORTS" | "CALLS" | "INHERITS"


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, src: str, dst: str, kind: str) -> None:
        # Only keep edges where both endpoints exist (or dst is an external
        # module we still want to record for IMPORTS)
        self.edges.append(Edge(src=src, dst=dst, kind=kind))


@dataclass
class ImportBinding:
    """Records what a locally-visible import name refers to, so pass 2 can
    resolve calls/bases that go through an import across files.

    kind="module": `local_name` refers to an entire imported module
        (e.g. `import parser` or `import gitnexus_py.parser as p`).
    kind="symbol": `local_name` refers to one symbol pulled out of a module
        (e.g. `from .db import open_db` or `from .db import open_db as o`).
    """
    kind: str                # "module" | "symbol"
    target_module: str       # rel_path of the resolved target module
    orig_name: str | None = None  # only set for kind="symbol"


def _mod_id(rel_path: str) -> str:
    return f"module::{rel_path}"


def _cls_id(rel_path: str, qualname: str) -> str:
    return f"class::{rel_path}::{qualname}"


def _func_id(rel_path: str, qualname: str) -> str:
    return f"function::{rel_path}::{qualname}"


def _ext_id(dotted_name: str) -> str:
    return f"external::{dotted_name}"


def _qualname(scope_stack: list[str], name: str) -> str:
    return ".".join(scope_stack + [name]) if scope_stack else name


def _container_id(graph: Graph, rel_path: str, scope_stack: list[str]) -> str:
    """Id of the nearest enclosing class/function, or the module."""
    if not scope_stack:
        return _mod_id(rel_path)
    qual = ".".join(scope_stack)
    # try function id first, then class id
    fid = _func_id(rel_path, qual)
    if fid in graph.nodes:
        return fid
    cid = _cls_id(rel_path, qual)
    if cid in graph.nodes:
        return cid
    return _mod_id(rel_path)


def _resolve_module_name(rel_path: str, level: int, module: str | None) -> str:
    """Resolve an ImportFrom's module string to a dotted module name,
    handling relative imports (level > 0) against the importing file's own
    package path.
    """
    if level and level > 0:
        pkg_parts = rel_path.replace(os.sep, "/").split("/")[:-1]  # drop filename
        up = max(level - 1, 0)
        if up:
            pkg_parts = pkg_parts[:-up] if up <= len(pkg_parts) else []
        if module:
            pkg_parts = pkg_parts + module.split(".")
        return ".".join(pkg_parts)
    return module or ""


class _DefVisitor(ast.NodeVisitor):
    """Pass 1: registers Module/Class/Function nodes, CONTAINS/IMPORTS
    edges, and this file's import-alias table for pass 2 to use."""

    def __init__(self, graph: Graph, rel_path: str, module_index: dict[str, str]):
        self.graph = graph
        self.rel_path = rel_path
        self.module_index = module_index  # dotted module name -> rel_path
        self.mod_node_id = _mod_id(rel_path)
        self._scope_stack: list[str] = []  # stack of qualnames (class/function)
        self.import_bindings: dict[str, ImportBinding] = {}

    def _current_container_id(self) -> str:
        return _container_id(self.graph, self.rel_path, self._scope_stack)

    # ---- visitors --------------------------------------------------------
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            target = self.module_index.get(alias.name)
            if target:
                self.graph.add_edge(self.mod_node_id, _mod_id(target), "IMPORTS")
                # Only bind a local name when it's unambiguous: an explicit
                # alias, or a non-dotted `import x` (a bare `import a.b.c`
                # without `as` binds the top-level package `a`, not `a.b.c`,
                # so we skip binding it to avoid resolving calls wrongly).
                if alias.asname:
                    self.import_bindings[alias.asname] = ImportBinding("module", target)
                elif "." not in alias.name:
                    self.import_bindings[alias.name] = ImportBinding("module", target)
            else:
                # stdlib / third-party - not a file in this repo, but still
                # record the dependency instead of silently dropping it.
                self._add_external_import(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module_name = _resolve_module_name(self.rel_path, node.level or 0, node.module)
        target = self.module_index.get(module_name) if module_name else None
        if target:
            self.graph.add_edge(self.mod_node_id, _mod_id(target), "IMPORTS")
            for alias in node.names:
                local_name = alias.asname or alias.name
                self.import_bindings[local_name] = ImportBinding("symbol", target, alias.name)
        elif module_name and not node.level:
            # only for absolute imports - a relative import that failed to
            # resolve is a broken/edge-case reference, not "external"
            self._add_external_import(module_name)
        self.generic_visit(node)

    def _add_external_import(self, dotted_name: str) -> None:
        ext_id = _ext_id(dotted_name)
        self.graph.add_node(Node(id=ext_id, kind="External", name=dotted_name, file=""))
        self.graph.add_edge(self.mod_node_id, ext_id, "IMPORTS")

    def visit_ClassDef(self, node: ast.ClassDef):
        qual = _qualname(self._scope_stack, node.name)
        node_id = _cls_id(self.rel_path, qual)
        self.graph.add_node(Node(
            id=node_id, kind="Class", name=node.name, file=self.rel_path,
            lineno=node.lineno, end_lineno=getattr(node, "end_lineno", node.lineno),
            docstring=(ast.get_docstring(node) or "")[:300],
        ))
        self.graph.add_edge(self._current_container_id(), node_id, "CONTAINS")

        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        qual = _qualname(self._scope_stack, node.name)
        node_id = _func_id(self.rel_path, qual)
        self.graph.add_node(Node(
            id=node_id, kind="Function", name=node.name, file=self.rel_path,
            lineno=node.lineno, end_lineno=getattr(node, "end_lineno", node.lineno),
            docstring=(ast.get_docstring(node) or "")[:300],
        ))
        self.graph.add_edge(self._current_container_id(), node_id, "CONTAINS")

        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore


class _InheritsVisitor(ast.NodeVisitor):
    """Pass 2a: resolves INHERITS edges only (same-file by name, cross-file
    through this file's import bindings). Run over *every* file before any
    CALLS resolution starts, so that by the time we resolve self.method()
    calls we have the complete, whole-repo inheritance graph to walk -
    including ancestors that live in files processed later.
    """

    def __init__(self, graph: Graph, rel_path: str, import_bindings: dict[str, ImportBinding]):
        self.graph = graph
        self.rel_path = rel_path
        self.import_bindings = import_bindings
        self._scope_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        qual = _qualname(self._scope_stack, node.name)
        node_id = _cls_id(self.rel_path, qual)

        for base in node.bases:
            if not isinstance(base, ast.Name):
                continue  # e.g. `class Foo(some.Mod.Base)` - out of scope
            base_id = _cls_id(self.rel_path, base.id)
            if base_id in self.graph.nodes:
                self.graph.add_edge(node_id, base_id, "INHERITS")
                continue
            binding = self.import_bindings.get(base.id)
            if binding and binding.kind == "symbol":
                cross_id = _cls_id(binding.target_module, binding.orig_name)
                if cross_id in self.graph.nodes:
                    self.graph.add_edge(node_id, cross_id, "INHERITS")

        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore


def _is_super_call(expr: ast.expr) -> bool:
    """True for the value of `super().method(...)` - i.e. expr is the
    `super()` call itself."""
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "super"
    )


class _CallVisitor(ast.NodeVisitor):
    """Pass 2b: resolves CALLS edges, now that INHERITS is fully known
    (see _InheritsVisitor). Resolution order per call:
      1. self.method()/cls.method()/super().method() - class-aware: the
         enclosing class's own methods first (skipped for super()), then
         its ancestors via INHERITS, breadth-first (best-effort MRO, not
         real C3 linearization, but right for single/simple inheritance).
      2. same-file match by simple name (covers other attribute calls too,
         same as before - not class-aware, just name-based)
      3. cross-file match through this file's import bindings (bare name
         imported directly, or `module_alias.func()` / class base through
         an imported module)
    """

    def __init__(self, graph: Graph, rel_path: str, import_bindings: dict[str, ImportBinding],
                 bases_of: dict[str, list[str]]):
        self.graph = graph
        self.rel_path = rel_path
        self.import_bindings = import_bindings
        self.bases_of = bases_of
        self._scope_stack: list[str] = []
        self._class_stack: list[str] = []  # enclosing class names only, for self/cls/super

    def _current_container_id(self) -> str:
        return _container_id(self.graph, self.rel_path, self._scope_stack)

    def _same_file_function(self, simple_name: str) -> str | None:
        for nid, n in self.graph.nodes.items():
            if n.kind == "Function" and n.file == self.rel_path and n.name == simple_name:
                return nid
        return None

    def _find_method_in_class(self, class_id: str, method_name: str) -> str | None:
        # class_id looks like "class::<rel_path>::<qualname>"
        _, rel_path, qualname = class_id.split("::", 2)
        fid = _func_id(rel_path, f"{qualname}.{method_name}")
        return fid if fid in self.graph.nodes else None

    def _find_method_via_ancestors(self, class_id: str, method_name: str) -> str | None:
        """Breadth-first walk up INHERITS from class_id's direct bases."""
        seen = {class_id}
        queue = list(self.bases_of.get(class_id, []))
        while queue:
            base_id = queue.pop(0)
            if base_id in seen:
                continue
            seen.add(base_id)
            found = self._find_method_in_class(base_id, method_name)
            if found:
                return found
            queue.extend(self.bases_of.get(base_id, []))
        return None

    # ---- visitors --------------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef):
        self._scope_stack.append(node.name)
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore

    def visit_Call(self, node: ast.Call):
        callee_name = None
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee_name = node.func.attr

        if callee_name:
            container = self._current_container_id()
            if container.startswith("function::"):
                target = self._resolve_call_target(node.func, callee_name)
                if target and target != container:
                    self.graph.add_edge(container, target, "CALLS")
        self.generic_visit(node)

    def _resolve_call_target(self, func_expr: ast.expr, callee_name: str) -> str | None:
        # 1. self.method() / cls.method() / super().method() - class-aware,
        #    walking the enclosing class's own methods then its ancestors.
        if isinstance(func_expr, ast.Attribute) and self._class_stack:
            class_id = _cls_id(self.rel_path, ".".join(self._class_stack))
            is_self_or_cls = isinstance(func_expr.value, ast.Name) and func_expr.value.id in ("self", "cls")
            if is_self_or_cls:
                found = self._find_method_in_class(class_id, callee_name)
                if found:
                    return found
                found = self._find_method_via_ancestors(class_id, callee_name)
                if found:
                    return found
            elif _is_super_call(func_expr.value):
                found = self._find_method_via_ancestors(class_id, callee_name)
                if found:
                    return found

        # 2. same-file, best-effort match by simple name (covers plain
        #    calls and other attribute calls alike - not class-aware)
        target = self._same_file_function(callee_name)
        if target:
            return target

        # 3. cross-file: a bare name imported directly as a function,
        #    e.g. `from .db import open_db` then `open_db(...)`
        if isinstance(func_expr, ast.Name):
            binding = self.import_bindings.get(callee_name)
            if binding and binding.kind == "symbol":
                fid = _func_id(binding.target_module, binding.orig_name)
                if fid in self.graph.nodes:
                    return fid

        # 4. cross-file: `module_alias.func()` where module_alias is a
        #    whole imported module, e.g. `import parser` then
        #    `parser.parse_repo(...)`
        elif isinstance(func_expr, ast.Attribute) and isinstance(func_expr.value, ast.Name):
            binding = self.import_bindings.get(func_expr.value.id)
            if binding and binding.kind == "module":
                fid = _func_id(binding.target_module, callee_name)
                if fid in self.graph.nodes:
                    return fid

        return None


def _build_module_index(py_files: list[str], root: str) -> dict[str, str]:
    """Maps dotted module names (e.g. 'pkg.sub.mod') -> relative file path,
    so `import pkg.sub.mod` can be resolved to an actual file in the repo.
    """
    index: dict[str, str] = {}
    for rel_path in py_files:
        no_ext = rel_path[:-3] if rel_path.endswith(".py") else rel_path
        dotted = no_ext.replace(os.sep, ".").replace("/", ".")
        if dotted.endswith(".__init__"):
            dotted = dotted[: -len(".__init__")]
        index[dotted] = rel_path
    return index


_DEFAULT_EXCLUDE_DIRS = {
    ".git", "__pycache__", "venv", ".venv", "env", ".env", "virtualenv",
    "node_modules", "build", "dist", ".mypy_cache", ".pytest_cache",
}


def _is_venv_dir(parent_dir: str, name: str) -> bool:
    """True if `name` under `parent_dir` looks like a virtualenv root -
    checked via the `pyvenv.cfg` marker file every venv has, so this
    catches env dirs regardless of what they're named (not just the
    common venv/.venv/env spellings in _DEFAULT_EXCLUDE_DIRS).
    """
    return os.path.isfile(os.path.join(parent_dir, name, "pyvenv.cfg"))


def _walk_source_files(root: str, extension: str, exclude_dirs: set[str] | None) -> list[str]:
    exclude_dirs = exclude_dirs or _DEFAULT_EXCLUDE_DIRS
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in exclude_dirs and not d.startswith(".") and not _is_venv_dir(dirpath, d)
        ]
        for f in filenames:
            if f.endswith(extension):
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, root)
                files.append(rel)
    return sorted(files)


def discover_python_files(root: str, exclude_dirs: set[str] | None = None) -> list[str]:
    return _walk_source_files(root, ".py", exclude_dirs)


def compute_file_hashes(root: str, py_files: list[str]) -> dict[str, str]:
    """SHA-1 of each file's raw bytes, used by `index --incremental` to
    detect which files actually changed since the last run.
    """
    hashes: dict[str, str] = {}
    for rel_path in py_files:
        full_path = os.path.join(root, rel_path)
        try:
            with open(full_path, "rb") as fh:
                hashes[rel_path] = hashlib.sha1(fh.read()).hexdigest()
        except OSError:
            continue
    return hashes


def discover_php_files(root: str, exclude_dirs: set[str] | None = None) -> list[str]:
    """Same as discover_python_files, for .php files. Kept dependency-free
    (no tree-sitter import) so `index` can always check whether a repo has
    any PHP before deciding whether PHP support needs to be installed.
    """
    exclude_dirs = exclude_dirs or (_DEFAULT_EXCLUDE_DIRS | {"vendor"})
    return _walk_source_files(root, ".php", exclude_dirs)


def parse_repo(root: str) -> Graph:
    """Parse every .py (and, if present, .php) file under `root` and
    return a populated Graph. PHP support is a second, independent
    extraction pass (see php_parser.py) feeding the same graph - it's a
    soft dependency: tree-sitter/tree-sitter-php are only imported if a
    .php file is actually found.
    """
    graph = Graph()
    py_files = discover_python_files(root)
    module_index = _build_module_index(py_files, root)

    # Register module nodes up front so IMPORTS can target them right away.
    # Docstrings get filled in below, once each file's actually parsed.
    for rel_path in py_files:
        graph.add_node(Node(id=_mod_id(rel_path), kind="Module", name=rel_path, file=rel_path))

    trees: dict[str, ast.Module] = {}
    import_bindings_by_file: dict[str, dict[str, ImportBinding]] = {}

    # Pass 1: register Class/Function nodes, CONTAINS, IMPORTS, and each
    # file's import-alias table. Nothing calls-related is resolved yet, so
    # this pass doesn't care what order files (or definitions) come in.
    for rel_path in py_files:
        full_path = os.path.join(root, rel_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                source = fh.read()
            tree = ast.parse(source, filename=rel_path)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"  [skip] {rel_path}: {e}")
            continue

        graph.nodes[_mod_id(rel_path)].docstring = (ast.get_docstring(tree) or "")[:300]
        trees[rel_path] = tree
        visitor = _DefVisitor(graph, rel_path, module_index)
        visitor.visit(tree)
        import_bindings_by_file[rel_path] = visitor.import_bindings

    # Pass 2a: resolve INHERITS across the whole repo first, so pass 2b can
    # walk a *complete* ancestor chain when resolving self.method() calls -
    # including ancestors defined in files we haven't visited yet below.
    for rel_path, tree in trees.items():
        visitor = _InheritsVisitor(graph, rel_path, import_bindings_by_file[rel_path])
        visitor.visit(tree)

    bases_of: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "INHERITS":
            bases_of.setdefault(edge.src, []).append(edge.dst)

    # Pass 2b: resolve CALLS now that the whole repo's nodes, import
    # tables, and inheritance graph are known - this is what lets a call
    # reference something defined later in the same file, reached through
    # an import, or reached through self/cls/super() and inheritance.
    for rel_path, tree in trees.items():
        visitor = _CallVisitor(graph, rel_path, import_bindings_by_file[rel_path], bases_of)
        visitor.visit(tree)

    # PHP pass, if the repo has any .php files. php_parser is imported
    # lazily (and only here) so tree-sitter/tree-sitter-php stay optional
    # for Python-only repos - importing it is what pulls them in.
    php_files = discover_php_files(root)
    if php_files:
        try:
            from .php_parser import parse_php_repo
        except ImportError as e:
            print(f"  [skip] found {len(php_files)} .php file(s) but PHP support isn't "
                  f"installed ({e}); run: pip install tree-sitter tree-sitter-php")
        else:
            for rel_path in php_files:
                graph.add_node(Node(id=_mod_id(rel_path), kind="Module", name=rel_path, file=rel_path))
            parse_php_repo(root, php_files, graph)

    return graph
