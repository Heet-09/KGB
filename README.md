# gitnexus-py

A local, zero-server code knowledge graph for **Python** codebases — a
lightweight Python-native take on [GitNexus](https://github.com/DuyTranSipher/sipher-git-nexus).

Instead of a browser + WASM KuzuDB, this runs entirely on your machine
using Python's built-in `ast` module for parsing and embedded
[Kuzu](https://kuzudb.com/) (an on-disk graph DB, no server to run) for
storage/querying.

## What it does

1. **Parses** every `.py` file in a repo with the `ast` module, and every
   `.php` file with `tree-sitter` + `tree-sitter-php` (optional - only
   needed if the repo actually has `.php` files) — no LLM calls, so it's
   fast and 100% deterministic.
2. **Builds a graph**: `Module`, `Class`, `Function` nodes; `CONTAINS`,
   `IMPORTS`, `CALLS`, `INHERITS` edges.
3. **Stores it in Kuzu**, an embedded graph database that supports Cypher
   queries, so you can ask real graph questions ("what calls this
   function, transitively, up to 3 hops?") instead of just grepping text.
4. **(Optional) Graph RAG**: the `ask` command pulls relevant graph
   context and sends it to Groq's LLM API to answer natural-language
   questions about the codebase.
5. **Visualizes the graph** three ways:
   - `serve` launches a local web **dashboard** (stdlib-only HTTP server,
     no new dependencies) with tabs for everything the CLI can do: Stats,
     Explore (callers/impact search by function name), Graph (the same
     interactive force-directed view, fetched live), Ask, and a raw
     Cypher console. This is the friendliest way to use the tool.
   - `visualize` exports a single self-contained HTML file (force-directed
     layout, drag/pan/zoom) you open straight in a browser - no server,
     no CDN dependency.
   - `obsidian` exports one Markdown note per node, wikilinked along
     CONTAINS/IMPORTS/CALLS/INHERITS - open the output folder as an
     [Obsidian](https://obsidian.md) vault and use its built-in Graph View
     (filters, tag-colored groups, local graph per note, search) for free.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Index a repo (builds/overwrites the graph DB)
python -m gitnexus_py.cli index /path/to/repo --db ./graph.db

# 1b. Or incrementally - skips the DB wipe and only touches files whose
#     content changed since the last --incremental run
python -m gitnexus_py.cli index /path/to/repo --db ./graph.db --incremental

# 2. See what got indexed
python -m gitnexus_py.cli stats --db ./graph.db

# 3. Who calls a given function?
python -m gitnexus_py.cli callers my_function --db ./graph.db

# 4. Before refactoring: what's the blast radius of changing this function?
python -m gitnexus_py.cli impact my_function --db ./graph.db

# 5. Run raw Cypher queries directly
python -m gitnexus_py.cli cypher "MATCH (f:Function) RETURN f.name LIMIT 10" --db ./graph.db

# 5b. Open the full web dashboard - Stats / Explore / Graph / Ask / Cypher,
#     all in one browser tab (opens automatically at http://127.0.0.1:8765)
python -m gitnexus_py.cli serve --db ./graph.db

# 6. Or just export the graph view - writes one self-contained HTML file,
#    open it in a browser (force-directed layout, drag/pan/zoom, no server)
python -m gitnexus_py.cli visualize --db ./graph.db --out ./graph.html

# 6b. ...or export it as an Obsidian vault (one note per node) and use
#     Obsidian's own Graph View instead
python -m gitnexus_py.cli obsidian --db ./graph.db --out ./obsidian_vault

# 7. Ask a natural-language question (needs GROQ_API_KEY env var)
export GROQ_API_KEY=your_key_here
python -m gitnexus_py.cli ask "what does the parser module do?" --db ./graph.db
```

## Known limitations (MVP — be aware before relying on this)

- **Two languages, not GitNexus's 12+.** Python via `ast`, plus a second,
  independent extraction pass for PHP via `tree-sitter` + `tree-sitter-php`
  (`php_parser.py`) - both feed the same Module/Class/Function graph.
  Adding a third language means writing another extraction module in this
  style; there's no shared "any tree-sitter grammar" abstraction (yet).
  PHP resolution is coarser than Python's: namespaces/PSR-4 autoloading
  aren't modeled - `use Foo\Bar;` becomes an `External` node by name
  rather than a resolved file, and `extends`/`ClassName::method()` that
  don't resolve within the same file fall back to a repo-wide search by
  simple class name (only when that name is unambiguous).
- **Call resolution is still name-based best-effort, not type inference.**
  `CALLS` edges are resolved in this order: (1) `self.method()` /
  `cls.method()` / `super().method()` walk the enclosing class's own
  methods then its ancestors via `INHERITS` (which itself resolves
  cross-file through imports), so overrides and cross-file base classes
  work; (2) same-file match by simple name; (3) cross-file through the
  calling file's own import table (`from .db import open_db` /
  `import parser; parser.parse_repo()`). What's still unresolved: calls
  through an arbitrary object instance whose type isn't known statically
  (e.g. `some_obj.method()` where `some_obj` isn't `self`/`cls` falls back
  to same-file name matching, not real dispatch), and dynamic dispatch
  (`getattr`, decorators that rewrite the callable).
- **Standard library / third-party imports get a lightweight `External`
  node** (not a real Module - no file to parse), so `IMPORTS` edges to
  them are kept instead of silently dropped, but nothing about their
  internals (classes/functions/calls) is known.
- **Incremental re-indexing is hash-based, not truly per-file.**
  `index --incremental` hashes each file's content against a
  `.manifest.json` sidecar next to `--db`, skips the run entirely if
  nothing changed, and only deletes+rewrites Kuzu data for files whose
  hash changed - but the whole repo is still re-parsed in memory every
  run, since cross-file `CALLS`/`INHERITS` resolution needs the full
  picture. So it saves the DB wipe and most Kuzu writes, not the parse
  itself; that's the gap vs. GitNexus's "<2s incremental update" claim.
- **Semantic search (optional).** `ask`/`serve` now do hybrid retrieval:
  TF-IDF (`retrieval.py`, exact vocabulary overlap) plus real semantic
  search via a local `fastembed` embedding model + Kuzu's native HNSW
  vector index (`db.vector_search`) - so a question like "formula
  quantity" can match a docstring about "qty of formula line items" with
  zero shared words. Requires `pip install fastembed` (~150-250MB, ONNX
  Runtime-based, not PyTorch) and re-running `index` so embeddings get
  computed and the vector index gets built - without it, retrieval
  silently falls back to TF-IDF alone. **Existing databases indexed
  before this feature need a re-`index` to get embeddings** - opening an
  old DB auto-migrates the schema (adds the column) but doesn't backfill
  vectors for already-indexed nodes.

## Project layout

```
gitnexus_py/
  parser.py     # ast-based Python extraction -> in-memory Graph (nodes/edges)
  php_parser.py # tree-sitter based PHP extraction, feeds the same Graph
  db.py         # Kuzu schema + bulk loader + incremental delete_files
  retrieval.py  # local TF-IDF ranking for the `ask` command's context
  viz.py        # self-contained HTML force-directed graph export
  web.py        # `serve` dashboard: stdlib HTTP server + JSON API + single-page UI
  obsidian.py   # exports the graph as an Obsidian vault (one note per node)
  cli.py        # typer CLI: index / stats / cypher / callers / impact / ask / visualize / obsidian / serve
```

## Why this matters for KGB (INDEC)

This is functionally the same shape as the Knowledge Graph Builder
concept: parse code → build a graph → answer change-impact and
documentation questions off it, with an eye toward measuring token
savings vs. dumping raw code into an LLM context. The `impact` command
here is a direct, working version of "change impact analysis," and the
`ask` command's context-window size (compare `len(context)` vs. raw file
size) is a real, measurable stand-in for the token-saving baseline that
was flagged as missing from the KGB plan.
# KGB