"""
db.py
-----
Sets up an embedded Kuzu graph database (zero-server, single-file-on-disk,
same idea as GitNexus's client-side KuzuDB) and loads a Graph (from
parser.py) into it.

Schema:
    Node tables : Module, Class, Function, External   (each keyed by string `id`)
    Rel tables  : CONTAINS, IMPORTS, CALLS, INHERITS  (generic ANY->ANY)

    External nodes represent stdlib/third-party modules that are imported
    but not part of the scanned repo (so there's no file to parse for
    them) - kept so IMPORTS edges to them aren't silently dropped.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import kuzu

from .embeddings import EMBED_DIM, embed_texts, is_available as embeddings_available
from .parser import Graph

NODE_KINDS = ("Module", "Class", "Function", "External")
REL_KINDS = ("CONTAINS", "IMPORTS", "CALLS", "INHERITS")

# Concept/RULE: a generic layer for "business meaning encoded in code" that
# isn't captured by the CONTAINS/CALLS/INHERITS structural graph above -
# role codes, status codes, feature flags, etc. `kind` on Concept
# distinguishes which idiom produced it (e.g. "Role"); RULE.effect records
# whether the concept allows or denies access to the target Function/Module
# it's attached to. See rbac_extractor.py for the first (Role) extractor.
_CONCEPT_NODE_DDL = """
CREATE NODE TABLE Concept (
    id STRING PRIMARY KEY,
    kind STRING,
    code STRING,
    name STRING,
    app STRING,
    source_file STRING,
    lineno INT64
)
"""

_RULE_REL_DDL = """
CREATE REL TABLE RULE (
    FROM Concept TO Function,
    FROM Concept TO Module,
    effect STRING,
    condition STRING,
    lineno INT64
)
"""

# Route: one row per URL pattern -> target, from route_extractor.py. `target`
# is kept as the raw string the source encodes (may contain $1/$2
# placeholders for CI4's wildcard-substitution routes) - substitution only
# happens at question time, against a real URL the user actually asked
# about (see db.resolve_url), never guessed at extraction time.
_ROUTE_NODE_DDL = """
CREATE NODE TABLE Route (
    id STRING PRIMARY KEY,
    pattern STRING,
    http_method STRING,
    target STRING,
    kind STRING,
    source_file STRING,
    lineno INT64
)
"""

# Render: one row per `view(...)` call found by view_extractor.py. Modeled
# as its own node (like Route) rather than an edge, specifically so an
# *unresolved* view reference (the string didn't match a known file) can
# still be recorded and surfaced honestly instead of silently dropped -
# there's no real graph node to point an edge at in that case.
_RENDER_NODE_DDL = """
CREATE NODE TABLE Render (
    id STRING PRIMARY KEY,
    source_file STRING,
    source_kind STRING,
    lineno INT64,
    view_arg STRING,
    target_file STRING,
    resolved BOOLEAN
)
"""


def _node_table_ddl(kind: str) -> str:
    # `embedding` backs the semantic-search vector index (see
    # ensure_vector_index / vector_search) - a fixed-size FLOAT array is
    # required for Kuzu's HNSW index (a variable-length LIST won't work).
    return f"""
    CREATE NODE TABLE {kind} (
        id STRING PRIMARY KEY,
        name STRING,
        file STRING,
        lineno INT64,
        end_lineno INT64,
        docstring STRING,
        embedding FLOAT[{EMBED_DIM}]
    )
    """


def open_db(db_path: str, fresh: bool = False) -> tuple[kuzu.Database, kuzu.Connection]:
    """Open (or create) the Kuzu database at db_path.

    fresh=True wipes any existing database first, so re-indexing a repo
    doesn't leave stale nodes/edges behind. Newer Kuzu versions store an
    embedded DB as a single file (plus a .wal sidecar) rather than a
    directory, so this removes whichever shape is actually on disk instead
    of assuming a directory (shutil.rmtree on a file raises WinError 267
    on Windows: "the directory name is invalid").
    """
    path = Path(db_path)
    if fresh and path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    wal = Path(str(db_path) + ".wal")
    if fresh and wal.exists():
        wal.unlink()

    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)
    _ensure_schema(conn)
    return db, conn


def _ensure_schema(conn: kuzu.Connection) -> None:
    existing = {row[0] for row in conn.execute("CALL show_tables() RETURN name;").get_as_df().values.tolist()} \
        if _has_any_tables(conn) else set()

    for kind in NODE_KINDS:
        if kind not in existing:
            conn.execute(_node_table_ddl(kind))
        else:
            _ensure_embedding_column(conn, kind)

    # Kuzu rel tables need FROM/TO pairs declared. We create one rel table
    # per (relationship kind) that can connect any of the three node kinds
    # to any other, using Kuzu's multi-pair rel table syntax.
    for rel in REL_KINDS:
        if rel not in existing:
            pairs = ", ".join(f"FROM {a} TO {b}" for a in NODE_KINDS for b in NODE_KINDS)
            conn.execute(f"CREATE REL TABLE {rel} ({pairs})")

    if "Concept" not in existing:
        conn.execute(_CONCEPT_NODE_DDL)
    if "RULE" not in existing:
        conn.execute(_RULE_REL_DDL)
    if "Route" not in existing:
        conn.execute(_ROUTE_NODE_DDL)
    if "Render" not in existing:
        conn.execute(_RENDER_NODE_DDL)


def _ensure_embedding_column(conn: kuzu.Connection, kind: str) -> None:
    """Schema migration for databases indexed before semantic search was
    added: ALTER in the `embedding` column if it's missing. Existing rows
    get NULL for it until the next `index` run recomputes embeddings and
    ensure_vector_index() (re)builds the HNSW index."""
    cols = {row["name"] for _, row in conn.execute(f"CALL TABLE_INFO('{kind}') RETURN *;").get_as_df().iterrows()}
    if "embedding" not in cols:
        conn.execute(f"ALTER TABLE {kind} ADD embedding FLOAT[{EMBED_DIM}]")


def ensure_vector_index(conn: kuzu.Connection) -> None:
    """Create Kuzu's native HNSW vector index on each node kind's
    `embedding` column, if it doesn't already exist - lets vector_search()
    do fast approximate nearest-neighbor lookups instead of a full scan.
    Safe to call every time `index` runs (idempotent - "already exists"
    errors are swallowed)."""
    conn.execute("INSTALL vector; LOAD EXTENSION vector;")
    for kind in NODE_KINDS:
        try:
            conn.execute(f"CALL CREATE_VECTOR_INDEX('{kind}', '{kind.lower()}_vec_idx', 'embedding')")
        except RuntimeError as exc:
            if "already exists" not in str(exc).lower():
                raise


def vector_search(conn: kuzu.Connection, query_text: str, top_k: int = 15) -> list[dict]:
    """Semantic search over Module/Class/Function nodes via Kuzu's HNSW
    vector index - finds nodes whose docstring/name is *conceptually*
    close to query_text even with no shared vocabulary, unlike
    retrieval.rank_context's TF-IDF (exact-word) matching. Returns []
    (not an error) if fastembed isn't installed or the index hasn't been
    built yet (e.g. an older DB that hasn't been re-indexed since this
    was added) - callers should treat this as "no semantic hits", falling
    back to TF-IDF alone."""
    if not embeddings_available():
        return []
    from .embeddings import embed_query

    try:
        qvec = embed_query(query_text)
    except Exception:
        return []

    results = []
    for kind in ("Module", "Class", "Function"):
        try:
            df = conn.execute(
                f"CALL QUERY_VECTOR_INDEX('{kind}', '{kind.lower()}_vec_idx', $qvec, $k) "
                f"RETURN node.name AS name, node.file AS file, node.docstring AS docstring, distance",
                {"qvec": qvec, "k": top_k},
            ).get_as_df()
        except RuntimeError:
            continue  # index not built for this kind yet
        for _, row in df.iterrows():
            results.append({
                "kind": kind, "name": str(row["name"]), "file": str(row["file"] or ""),
                "docstring": str(row["docstring"] or ""), "distance": float(row["distance"]),
            })

    results.sort(key=lambda r: r["distance"])
    return results[:top_k]


def _has_any_tables(conn: kuzu.Connection) -> bool:
    try:
        df = conn.execute("CALL show_tables() RETURN name;").get_as_df()
        return len(df) > 0
    except Exception:
        return False


def delete_files(conn: kuzu.Connection, rel_paths: list[str]) -> None:
    """Remove all nodes (and their incident edges, via DETACH DELETE)
    belonging to the given file paths. Used by `index --incremental` to
    clear out stale data for changed/removed files before re-inserting
    fresh nodes/edges for them, without wiping the whole database.
    """
    for kind in NODE_KINDS:
        for rel_path in rel_paths:
            conn.execute(f"MATCH (n:{kind} {{file: $file}}) DETACH DELETE n", {"file": rel_path})


def export_graph_json(conn: kuzu.Connection) -> dict:
    """Pull the whole graph back out of Kuzu as plain dicts, e.g. for
    `cli.visualize` or `cli.obsidian` to render into their own output
    formats."""
    nodes = []
    for kind in NODE_KINDS:
        df = conn.execute(f"MATCH (n:{kind}) RETURN n.id, n.name, n.file, n.docstring").get_as_df()
        for _, row in df.iterrows():
            nodes.append({
                "id": str(row.iloc[0]), "kind": kind,
                "name": str(row.iloc[1]), "file": str(row.iloc[2] or ""),
                "docstring": str(row.iloc[3] or ""),
            })

    edges = []
    for rel in REL_KINDS:
        df = conn.execute(f"MATCH (a)-[:{rel}]->(b) RETURN a.id, b.id").get_as_df()
        for _, row in df.iterrows():
            edges.append({"src": str(row.iloc[0]), "dst": str(row.iloc[1]), "kind": rel})

    return {"nodes": nodes, "edges": edges}


def list_function_names(conn: kuzu.Connection) -> list[str]:
    """All distinct Function names in the graph - used to spot which
    function(s) a natural-language question is actually asking about
    (see retrieval.find_mentioned_names)."""
    df = conn.execute("MATCH (f:Function) RETURN DISTINCT f.name AS name").get_as_df()
    return [str(v) for v in df["name"].tolist()]


def get_callers(conn: kuzu.Connection, name: str) -> list[dict]:
    """Functions that directly call `name` - same query used by the
    `callers` CLI command and the /api/callers dashboard route."""
    df = conn.execute(
        """
        MATCH (caller:Function)-[:CALLS]->(callee:Function)
        WHERE callee.name = $name
        RETURN caller.name AS caller, caller.file AS file, caller.lineno AS line
        """,
        {"name": name},
    ).get_as_df()
    return df.to_dict(orient="records")


def get_impact(conn: kuzu.Connection, name: str, max_hops: int = 3) -> list[dict]:
    """Transitive blast radius (up to `max_hops` hops) of changing `name` -
    same query used by the `impact` CLI command and the /api/impact
    dashboard route.

    Kuzu (like most Cypher engines) doesn't support parameterizing a
    variable-length path's hop count (`*1..$n`) - it has to be a literal
    in the query text. max_hops is clamped to a sane range and cast to
    int before being formatted in, so this stays injection-safe even
    though it isn't passed as a bound parameter like `name` is.
    """
    max_hops = max(1, min(int(max_hops), 10))
    df = conn.execute(
        f"""
        MATCH (callee:Function)<-[:CALLS*1..{max_hops}]-(caller:Function)
        WHERE callee.name = $name
        RETURN DISTINCT caller.name AS affected_function, caller.file AS file
        """,
        {"name": name},
    ).get_as_df()
    return df.to_dict(orient="records")


def load_graph(conn: kuzu.Connection, graph: Graph) -> None:
    """Bulk-insert all nodes and edges from an in-memory Graph into Kuzu."""
    nodes = list(graph.nodes.values())

    # Embed all nodes' "name + docstring" in one batched model call (much
    # faster than one call per node) so vector_search has something to
    # query. Skipped gracefully if fastembed isn't installed - embedding
    # stays NULL and semantic search just returns no hits for these nodes.
    embeddings: list[list[float] | None] = [None] * len(nodes)
    if embeddings_available() and nodes:
        texts = [f"{n.name} {n.docstring}" for n in nodes]
        embeddings = embed_texts(texts)

    # Insert nodes, grouped by kind (Kuzu tables are per-kind)
    for node, embedding in zip(nodes, embeddings):
        conn.execute(
            f"""
            MERGE (n:{node.kind} {{id: $id}})
            SET n.name = $name, n.file = $file, n.lineno = $lineno,
                n.end_lineno = $end_lineno, n.docstring = $docstring,
                n.embedding = $embedding
            """,
            {
                "id": node.id, "name": node.name, "file": node.file,
                "lineno": node.lineno, "end_lineno": node.end_lineno,
                "docstring": node.docstring, "embedding": embedding,
            },
        )

    # Insert edges. Need each endpoint's kind to target the right rel pair.
    for edge in graph.edges:
        src_node = graph.nodes.get(edge.src)
        dst_node = graph.nodes.get(edge.dst)
        if not src_node or not dst_node:
            continue  # skip edges pointing to unresolved/external nodes
        conn.execute(
            f"""
            MATCH (a:{src_node.kind} {{id: $src}}), (b:{dst_node.kind} {{id: $dst}})
            MERGE (a)-[:{edge.kind}]->(b)
            """,
            {"src": edge.src, "dst": edge.dst},
        )


def load_concepts(conn: kuzu.Connection, concepts: list[dict]) -> None:
    """Bulk-insert Concept nodes (see rbac_extractor.extract). Each dict:
    {id, kind, code, name, app, source_file, lineno}."""
    for c in concepts:
        conn.execute(
            """
            MERGE (n:Concept {id: $id})
            SET n.kind = $kind, n.code = $code, n.name = $name, n.app = $app,
                n.source_file = $source_file, n.lineno = $lineno
            """,
            c,
        )


def load_rules(conn: kuzu.Connection, rules: list[dict]) -> None:
    """Bulk-insert RULE edges from a Concept to the Function/Module it
    gates. Each dict: {concept_id, target_id, target_kind, effect,
    condition, lineno} - target_kind must be "Function" or "Module"."""
    for r in rules:
        conn.execute(
            f"""
            MATCH (c:Concept {{id: $concept_id}}), (t:{r['target_kind']} {{id: $target_id}})
            MERGE (c)-[e:RULE]->(t)
            SET e.effect = $effect, e.condition = $condition, e.lineno = $lineno
            """,
            {
                "concept_id": r["concept_id"], "target_id": r["target_id"],
                "effect": r["effect"], "condition": r["condition"], "lineno": r["lineno"],
            },
        )


def list_concepts(conn: kuzu.Connection, kind: str | None = None) -> list[dict]:
    """All extracted Concepts (optionally filtered to one `kind`, e.g.
    "Role") - the deterministic answer to "what types of X are there"."""
    if kind:
        df = conn.execute(
            "MATCH (c:Concept) WHERE c.kind = $kind "
            "RETURN c.code AS code, c.name AS name, c.app AS app, c.source_file AS source_file, c.lineno AS lineno "
            "ORDER BY c.app, c.code",
            {"kind": kind},
        ).get_as_df()
    else:
        df = conn.execute(
            "MATCH (c:Concept) RETURN c.kind AS kind, c.code AS code, c.name AS name, "
            "c.app AS app, c.source_file AS source_file, c.lineno AS lineno "
            "ORDER BY c.kind, c.app, c.code"
        ).get_as_df()
    return df.to_dict(orient="records")


def get_concept_rules(conn: kuzu.Connection, code: str) -> list[dict]:
    """Every RULE attached to the Concept with this code - the deterministic
    answer to "what can <role> access/do"."""
    df = conn.execute(
        """
        MATCH (c:Concept {code: $code})-[r:RULE]->(t)
        RETURN t.name AS target, t.file AS file, r.effect AS effect, r.condition AS condition, r.lineno AS lineno
        ORDER BY r.effect, target
        """,
        {"code": code},
    ).get_as_df()
    return df.to_dict(orient="records")


def get_rules_for_file(conn: kuzu.Connection, file: str) -> list[dict]:
    """Every RULE gating a Function/Module whose `.file` matches `file` -
    the deterministic, page-scoped answer to "what rules gate this page",
    instead of dumping every RULE in the whole repo (see
    web.graph_facts_for_question)."""
    df = conn.execute(
        """
        MATCH (c:Concept)-[r:RULE]->(t)
        WHERE t.file = $file
        RETURN c.code AS role, t.name AS target, r.effect AS effect, r.condition AS condition, r.lineno AS lineno
        ORDER BY r.lineno
        """,
        {"file": file},
    ).get_as_df()
    return df.to_dict(orient="records")


def list_module_files(conn: kuzu.Connection) -> list[str]:
    """Every distinct Module `.file` path in the graph - used to spot which
    file/page a natural-language question is naming (see
    retrieval.find_mentioned_file)."""
    df = conn.execute("MATCH (m:Module) RETURN DISTINCT m.file AS file").get_as_df()
    return [str(v) for v in df["file"].tolist() if v]


def get_inherits(conn: kuzu.Connection, class_name: str) -> list[dict]:
    """Ancestor classes `class_name` extends, transitively - the
    deterministic answer to "what does X inherit / extend"."""
    df = conn.execute(
        """
        MATCH (c:Class {name: $name})-[:INHERITS*1..5]->(ancestor:Class)
        RETURN DISTINCT ancestor.name AS ancestor, ancestor.file AS file
        """,
        {"name": class_name},
    ).get_as_df()
    return df.to_dict(orient="records")


def get_subclasses(conn: kuzu.Connection, class_name: str) -> list[dict]:
    """Classes that (transitively) extend `class_name` - the reverse of
    get_inherits(), the deterministic answer to "what extends/subclasses
    X"."""
    df = conn.execute(
        """
        MATCH (child:Class)-[:INHERITS*1..5]->(ancestor:Class {name: $name})
        RETURN DISTINCT child.name AS child, child.file AS file
        """,
        {"name": class_name},
    ).get_as_df()
    return df.to_dict(orient="records")


def load_routes(conn: kuzu.Connection, routes: list[dict]) -> None:
    """Bulk-insert Route nodes (see route_extractor.extract). Each dict:
    {pattern, http_method, target, kind, source_file, lineno}. `target` is
    kept as the raw string the source encodes - resolving it (including
    substituting any $1/$2 wildcard placeholders) happens later, at
    question time, in resolve_url()."""
    for r in routes:
        rid = f"route::{r['kind']}::{r['source_file']}::{r['lineno']}::{r['http_method']}::{r['pattern']}"
        conn.execute(
            """
            MERGE (n:Route {id: $id})
            SET n.pattern = $pattern, n.http_method = $http_method, n.target = $target,
                n.kind = $kind, n.source_file = $source_file, n.lineno = $lineno
            """,
            {
                "id": rid, "pattern": r["pattern"], "http_method": r["http_method"],
                "target": r["target"], "kind": r["kind"], "source_file": r["source_file"],
                "lineno": r["lineno"],
            },
        )


def list_routes(conn: kuzu.Connection) -> list[dict]:
    """Every extracted Route - the deterministic answer to "what
    routes/pages exist" (mirrors `roles`'s list_concepts). Includes both
    explicitly-declared routes and the convention-generated ones (see
    route_extractor.py)."""
    df = conn.execute(
        "MATCH (r:Route) RETURN r.pattern AS pattern, r.http_method AS http_method, "
        "r.target AS target, r.kind AS kind, r.source_file AS source_file, r.lineno AS lineno "
        "ORDER BY r.pattern"
    ).get_as_df()
    return df.to_dict(orient="records")


_ROUTE_PLACEHOLDER_RE = re.compile(r"\(:any\)|\(:num\)")


def _route_pattern_to_regex(pattern: str) -> re.Pattern:
    """Turn a CI4-style route pattern (literal segments + `(:any)`/`(:num)`
    wildcards) into an anchored regex with one capture group per wildcard,
    in source order - so matching a real URL against it both confirms the
    route and captures the values `$1`, `$2`, ... in resolve_url() refer
    to."""
    parts = []
    for chunk in re.split(r"(\(:any\)|\(:num\))", pattern):
        if chunk == "(:any)":
            parts.append("(.*)")
        elif chunk == "(:num)":
            parts.append(r"(\d+)")
        else:
            parts.append(re.escape(chunk))
    return re.compile("^" + "".join(parts) + "$", re.IGNORECASE)


def _normalize_url_path(url: str) -> str:
    """Strip scheme/host/query/fragment and leading/trailing slashes from a
    URL or bare path, so `https://app.example.com/icab/configuration?x=1`
    and `/icab/configuration` and `icab/configuration` all normalize to the
    same comparable string."""
    url = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^/]+", "", url.strip())  # scheme://host
    url = url.split("?", 1)[0].split("#", 1)[0]
    return url.strip("/")


def _target_class_to_file(target_class: str, known_files: dict[str, str]) -> str | None:
    """Resolve a `Namespace\\Class` string (no `::method`) to a real
    indexed Module file, via CI4's namespace<->path convention (`App\\` ->
    `app/`, backslashes -> path separators, `.php` appended) - the same
    convention rbac_extractor.py's _infer_app relies on elsewhere in this
    codebase. `known_files` maps a forward-slash-normalized file path to
    the original (possibly backslash-separated) path stored in the graph."""
    path = target_class.lstrip("\\").replace("\\", "/")
    if path.lower().startswith("app/"):
        path = "app/" + path[len("app/"):]
    return known_files.get(path + ".php")


def resolve_url(conn: kuzu.Connection, url: str) -> dict:
    """Resolve a real URL/path to a Controller file + method, by matching it
    against every extracted Route (explicit and convention-generated
    alike) and, for a match, substituting any `$1`/`$2` placeholders in the
    target with the values captured from *this* URL's own wildcard
    segments - never a guessed version number or argument.

    Returns {resolved: True, file, class, method, matched_pattern} on a
    clean match, or {resolved: False, matched_pattern, target} when a
    route pattern matched but the (possibly-substituted) target class
    isn't a file actually in the graph - e.g. an API version that isn't
    indexed - or {resolved: False} when no route pattern matches at all.
    """
    normalized = _normalize_url_path(url)
    routes = list_routes(conn)
    known_files = {f.replace("\\", "/"): f for f in list_module_files(conn)}

    for route in routes:
        pattern = str(route["pattern"]).strip("/")
        regex = _route_pattern_to_regex(pattern)
        m = regex.match(normalized)
        if not m:
            continue
        target = str(route["target"])
        for i, group in enumerate(m.groups(), start=1):
            target = target.replace(f"${i}", group)

        class_ref, _, method = target.rpartition("::")
        method = method or "index"
        if not class_ref:
            return {"resolved": False, "matched_pattern": route["pattern"], "target": target}

        file = _target_class_to_file(class_ref, known_files)
        if file is None:
            return {"resolved": False, "matched_pattern": route["pattern"], "target": target}

        return {
            "resolved": True, "file": file, "class": class_ref.rsplit("\\", 1)[-1],
            "method": method, "matched_pattern": route["pattern"],
        }

    return {"resolved": False}


def get_role_page_matrix(conn: kuzu.Connection) -> list[dict]:
    """Every RULE edge, grouped implicitly by (role, page) - the same data
    get_rules_for_file/list_concepts already expose, aggregated across
    every page at once instead of one page at a time. Deterministic answer
    to "give me the role x page permission matrix"."""
    df = conn.execute(
        """
        MATCH (c:Concept {kind: 'Role'})-[r:RULE]->(t)
        RETURN c.code AS role, t.file AS file, r.effect AS effect
        ORDER BY role, file
        """
    ).get_as_df()
    return df.to_dict(orient="records")


def load_renders(conn: kuzu.Connection, renders: list[dict]) -> None:
    """Bulk-insert Render nodes (see view_extractor.extract). Each dict:
    {source_file, source_kind, lineno, view_arg, target_file, resolved}.
    Unresolved rows (target_file=None) are still inserted - see the Render
    table's own docstring for why this is a node, not an edge."""
    for r in renders:
        rid = f"render::{r['source_file']}::{r['lineno']}::{r['view_arg']}"[:300]
        conn.execute(
            """
            MERGE (n:Render {id: $id})
            SET n.source_file = $source_file, n.source_kind = $source_kind, n.lineno = $lineno,
                n.view_arg = $view_arg, n.target_file = $target_file, n.resolved = $resolved
            """,
            {
                "id": rid, "source_file": r["source_file"], "source_kind": r["source_kind"],
                "lineno": r["lineno"], "view_arg": r["view_arg"],
                "target_file": r.get("target_file") or "", "resolved": bool(r.get("resolved")),
            },
        )


def get_views_rendered(conn: kuzu.Connection, file: str) -> list[dict]:
    """Every view-render call found in `file` - the deterministic (where
    resolved) or honest-best-effort (where not) answer to "which view(s)
    does this page render"."""
    df = conn.execute(
        "MATCH (r:Render {source_file: $file}) RETURN r.view_arg AS view_arg, "
        "r.target_file AS target_file, r.resolved AS resolved, r.lineno AS lineno ORDER BY r.lineno",
        {"file": file},
    ).get_as_df()
    return df.to_dict(orient="records")


def get_pages_using_view(conn: kuzu.Connection, view_file: str) -> list[dict]:
    """Reverse of get_views_rendered - every page that renders `view_file`
    (only resolved rows, since an unresolved row by definition doesn't
    name a real target_file)."""
    df = conn.execute(
        "MATCH (r:Render {target_file: $file, resolved: true}) "
        "RETURN r.source_file AS source_file, r.lineno AS lineno ORDER BY r.source_file",
        {"file": view_file},
    ).get_as_df()
    return df.to_dict(orient="records")
