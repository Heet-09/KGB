"""
rbac_extractor.py
------------------
First concrete extractor in the Concept/RULE framework (see PLAN in the
project chat history): mines PHP source for the "role catalog + role-gated
conditional" idiom deterministically, with tree-sitter pattern matching -
no LLM anywhere in this file. This is meant to reproduce, by machine, the
same table that was hand-built by grepping ACI's IcabPermissionChecker.php
and friends.

Two passes, mirroring php_parser.py's own two-stage style:

    pass 1 (constant_catalog): find `define('NAME', 'value')` calls whose
        constant name looks role-shaped (see _CONCEPT_NAME_PATTERNS) and
        record them as Concept rows.

    pass 2 (branch_gate): find `if (...)` conditions that mention a
        role-like variable (see _ROLE_VAR_RE) and reference one of the
        constants caught in pass 1 (via in_array(), ==, !=, whether or not
        wrapped in strtolower()/strtoupper()). The RULE's effect is read
        off the consequence block's own `$flagVar = TRUE/FALSE;` (the
        idiom this codebase actually uses to encode allow/deny), not from
        the presence/absence of `!` on the condition - the latter doesn't
        reliably predict which way access goes.

Both passes are intentionally narrow (only patterns actually observed in
ACI). Extending coverage means widening _CONCEPT_NAME_PATTERNS/_ROLE_VAR_RE
or adding new passes - not rewriting this one to "understand" more code.
"""

from __future__ import annotations

import re

import tree_sitter as ts

from .parser import _func_id, _mod_id, _qualname
from .php_parser import _parse_file, _string_literal_text, _text

# Registry of (Concept.kind, name_pattern). "Role" is the first idiom this
# proves out; new concept kinds (Status, FeatureFlag, ...) plug in here
# without touching the walking logic below.
#
# Requires BOTH a role-type noun AND a code/role suffix in the same
# constant name - a bare "ROLE" match is too broad and catches
# non-identity constants like CONST_ROLE_TABLE (a DB table name),
# CONST_ROLE_CD (a session-field name, not any specific role's code), or
# ROLE_WISE_REDIRECT_LINK (a config key) - none of which name an actual
# role. ACI's real role-identity constants all pair a role noun with
# _CODE/_ROLE_CD/_ROLE_KEY, e.g. ICAB_ADMIN_CODE, SOCLEAN_ADMIN_ROLE_CD.
_ROLE_NOUNS = (r"ADMIN|MANAGER|READER|CHEMIST|LABORANTIN|PLANNING|SECTOR|AGENCY"
               r"|REGIONAL|NATIONAL|SERVICE_TARIF|USR")
_ROLE_SUFFIX = r"_CODE|_ROLE_CD|_ROLE_KEY"
_CONCEPT_NAME_PATTERNS = [
    ("Role", re.compile(rf"(?:{_ROLE_NOUNS}).*(?:{_ROLE_SUFFIX})|(?:{_ROLE_SUFFIX}).*(?:{_ROLE_NOUNS})", re.I)),
]

# Only conditions that mention a variable whose name contains "role" count
# as an access-control check - keeps false positives down at the cost of
# missing sites that use an unrelated variable name (e.g. ACI's `$rolCd`,
# missing the "e"). Coverage gaps like that are meant to be visible/fixable
# here, not silently guessed around.
_ROLE_VAR_RE = re.compile(r"role", re.I)
_FLAG_VAR_RE = re.compile(r"flag|redirect|deny|block", re.I)


def _classify(const_name: str) -> str | None:
    for kind, pattern in _CONCEPT_NAME_PATTERNS:
        if pattern.search(const_name):
            return kind
    return None


def _infer_app(rel_path: str) -> str:
    """'Modules/Icab/...' -> 'Icab'; else the top-level folder - same
    "nearest thing to a domain boundary a plain file layout gives you"
    heuristic web.py's _top_folder uses for the codebase summary."""
    parts = rel_path.replace("\\", "/").split("/")
    if "Modules" in parts:
        i = parts.index("Modules")
        if i + 1 < len(parts):
            return parts[i + 1]
    return parts[0] if parts else "global"


def _concept_id(app: str, code: str) -> str:
    return f"concept::{app}::{code}"


def _enclosing_target(node: ts.Node, rel_path: str) -> tuple[str, str]:
    """Nearest enclosing function/method id (+ its graph kind), or the
    module if `node` isn't inside one - reuses php_parser's own
    scope/qualname scheme so RULE edges land on the exact Function nodes
    CONTAINS/CALLS already indexed."""
    p = node.parent
    while p is not None:
        if p.type in ("function_definition", "method_declaration"):
            fn_name = _text(p.child_by_field_name("name"))
            cls = p.parent
            while cls is not None and cls.type != "class_declaration":
                cls = cls.parent
            qual = _qualname([_text(cls.child_by_field_name("name"))], fn_name) if cls is not None else fn_name
            return _func_id(rel_path, qual), "Function"
        p = p.parent
    return _mod_id(rel_path), "Module"


def _mentions_role_var(node: ts.Node) -> bool:
    for n in _walk(node):
        if n.type == "variable_name":
            # `name` isn't exposed as a named field on variable_name in this
            # tree-sitter-php grammar version (`$roleCd` -> variable_name
            # [$, name]) - pick the identifier child by type instead.
            name_node = next((c for c in n.children if c.type == "name"), None)
            if name_node is not None and _ROLE_VAR_RE.search(_text(name_node)):
                return True
    return False


def _walk(node: ts.Node):
    yield node
    for c in node.children:
        yield from _walk(c)


def _literal_codes_in(node: ts.Node) -> list[str]:
    """Every bare constant name and string literal referenced anywhere
    under `node` - candidates for "this is one of the catalogued role
    codes", filtered against the known catalog by the caller."""
    codes = []
    for n in _walk(node):
        if n.type == "name":
            codes.append(_text(n))
        elif n.type in ("string", "encapsed_string"):
            lit = _string_literal_text(n)
            if lit:
                codes.append(lit)
    return codes


def _flag_effect(consequence: ts.Node | None) -> str:
    """Read the branch's actual effect off a top-level `$flagVar = TRUE;`
    (or FALSE/1/0) assignment in its body, e.g. `$redirectFlag = TRUE;` in
    IcabPermissionChecker.php means "this branch denies access" - the
    opposite of what a naive "negated condition = deny" guess would say
    about half these branches. Falls back to "checked" (polarity unknown)
    rather than guessing when no such assignment is found."""
    if consequence is None:
        return "checked"
    for stmt in consequence.children:
        if stmt.type != "expression_statement" or not stmt.children:
            continue
        expr = stmt.children[0]
        if expr.type != "assignment_expression":
            continue
        left = expr.child_by_field_name("left")
        right = expr.child_by_field_name("right")
        if left is None or right is None or not _FLAG_VAR_RE.search(_text(left)):
            continue
        val = _text(right).strip().lower()
        if val in ("true", "1"):
            return "deny"
        if val in ("false", "0"):
            return "allow"
    return "checked"


def extract(root: str, php_files: list[str]) -> tuple[list[dict], list[dict]]:
    """Run both passes over `php_files` (relative to `root`). Returns
    (concepts, rules) shaped for db.load_concepts / db.load_rules."""
    concepts: dict[str, dict] = {}      # concept_id -> row, dedup'd
    code_index: dict[str, str] = {}     # CODE.upper() -> concept_id (define() is process-global in
                                         # PHP, not namespaced by folder - a permission check in
                                         # app/Filters/ can reference a constant defined under
                                         # Modules/Icab/, so lookup can't be scoped by inferred `app`)
    trees: dict[str, ts.Tree] = {}

    # ---- pass 1: constant_catalog --------------------------------------
    for rel_path in php_files:
        tree = _parse_file(root, rel_path)
        if tree is None:
            continue
        trees[rel_path] = tree
        app = _infer_app(rel_path)

        for node in _walk(tree.root_node):
            if node.type != "function_call_expression":
                continue
            if _text(node.child_by_field_name("function")) != "define":
                continue
            args_node = node.child_by_field_name("arguments")
            if args_node is None:
                continue
            args = [c for c in args_node.children if c.type == "argument"]
            if len(args) < 2 or not args[0].children or not args[1].children:
                continue
            name_lit = _string_literal_text(args[0].children[0])
            val_lit = _string_literal_text(args[1].children[0])
            if not name_lit:
                continue
            kind = _classify(name_lit)
            if not kind:
                continue
            code = val_lit or name_lit
            cid = _concept_id(app, code)
            if code.upper() not in code_index:
                concepts[cid] = {
                    "id": cid, "kind": kind, "code": code, "name": name_lit,
                    "app": app, "source_file": rel_path, "lineno": node.start_point[0] + 1,
                }
                # branch_gate sites reference a role by either its short
                # code value (e.g. 'AMI', ACI's IcabPermissionChecker.php's
                # 'red'/'che' literals) or the constant's own name (e.g.
                # `== ICAB_AGENCY_MANAGER_CODE`) - index both to the same
                # Concept so either spelling resolves.
                code_index[code.upper()] = cid
                code_index[name_lit.upper()] = cid

    # ---- pass 2: branch_gate --------------------------------------------
    rules: list[dict] = []
    seen: set[tuple[str, str, str, int]] = set()
    for rel_path, tree in trees.items():
        for node in _walk(tree.root_node):
            if node.type != "if_statement":
                continue
            # child_by_field_name("condition"/"consequence") is unreliable
            # for if_statement on this tree-sitter-php build (returns None
            # even though the field exists in the grammar) - fall back to
            # positional lookup among direct children, which is stable.
            cond = next((c for c in node.children if c.type == "parenthesized_expression"), None)
            if cond is None or not _mentions_role_var(cond):
                continue
            matched_ids = {
                code_index[raw.upper()]
                for raw in _literal_codes_in(cond)
                if raw.upper() in code_index
            }
            if not matched_ids:
                continue
            target_id, target_kind = _enclosing_target(node, rel_path)
            consequence = next((c for c in node.children if c.type == "compound_statement"), None)
            effect = _flag_effect(consequence)
            condition_text = _text(cond).strip()[:160]
            lineno = node.start_point[0] + 1
            for cid in matched_ids:
                key = (cid, target_id, effect, lineno)
                if key in seen:
                    continue
                seen.add(key)
                rules.append({
                    "concept_id": cid, "target_id": target_id, "target_kind": target_kind,
                    "effect": effect, "condition": condition_text, "lineno": lineno,
                })

    return list(concepts.values()), rules
