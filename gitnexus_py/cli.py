"""
cli.py
------
Command-line interface for gitnexus-py.

Usage:
    python -m gitnexus_py.cli index /path/to/repo --db ./graph.db
    python -m gitnexus_py.cli stats --db ./graph.db
    python -m gitnexus_py.cli cypher "MATCH (f:Function) RETURN f.name LIMIT 10" --db ./graph.db
    python -m gitnexus_py.cli callers my_function --db ./graph.db
    python -m gitnexus_py.cli ask "what does the parser module do?" --db ./graph.db
"""

from __future__ import annotations

import json
import os

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .db import (open_db, load_graph, delete_files, export_graph_json, get_callers, get_impact,
                 load_concepts, load_rules, list_concepts, get_concept_rules, ensure_vector_index)
from .embeddings import is_available as embeddings_available
from .parser import parse_repo, discover_python_files, discover_php_files, compute_file_hashes
from .viz import render_html
from .obsidian import export_vault
from .web import run_server, _ask
from . import rbac_extractor

app = typer.Typer(add_completion=False, help="GitNexus-py: local code knowledge graph")
console = Console()

DEFAULT_DB = "./gitnexus.db"


def _manifest_path(db_path: str) -> str:
    """Sidecar file (next to, not inside, the Kuzu db dir) recording each
    file's content hash as of the last successful index - what
    `--incremental` diffs against."""
    return db_path.rstrip("/\\") + ".manifest.json"


@app.command()
def index(
    repo_path: str = typer.Argument(..., help="Path to the repo to index"),
    db: str = typer.Option(DEFAULT_DB, "--db", help="Path to store the Kuzu graph DB"),
    fresh: bool = typer.Option(True, "--fresh/--no-fresh", help="Wipe existing DB before indexing"),
    incremental: bool = typer.Option(
        False, "--incremental",
        help="Skip re-parsing/writing files whose content hash hasn't changed since the last "
             "--incremental run (tracked in a .manifest.json next to --db). Implies --no-fresh.",
    ),
):
    """Parse a Python repo and build/update the knowledge graph.

    Note: even with --incremental, the whole repo is still re-parsed in
    memory each run (cross-file CALLS/INHERITS resolution needs the full
    picture) - what's actually skipped is the expensive part: the DB wipe,
    and re-writing nodes/edges for files whose content didn't change.
    """
    repo_path = os.path.abspath(repo_path)
    manifest_file = _manifest_path(db)

    if incremental and fresh:
        console.print("[yellow]--incremental implies --no-fresh; skipping the DB wipe.[/yellow]")
        fresh = False

    to_touch: set[str] | None = None
    new_hashes: dict[str, str] = {}
    if incremental:
        source_files = discover_python_files(repo_path) + discover_php_files(repo_path)
        new_hashes = compute_file_hashes(repo_path, source_files)
        old_hashes: dict[str, str] = {}
        if os.path.exists(manifest_file):
            with open(manifest_file, "r", encoding="utf-8") as fh:
                old_hashes = json.load(fh)
        changed = {f for f, h in new_hashes.items() if old_hashes.get(f) != h}
        removed = set(old_hashes) - set(new_hashes)
        to_touch = changed | removed
        if old_hashes and not to_touch:
            console.print("[green]Nothing changed since the last --incremental index. Skipping.[/green]")
            return
        console.print(
            f"[bold]Incremental[/bold]: {len(changed)} changed, {len(removed)} removed, "
            f"{len(source_files) - len(changed)} unchanged"
        )

    console.print(f"[bold]Parsing[/bold] {repo_path} ...")
    graph = parse_repo(repo_path)
    console.print(f"  found {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    console.print(f"[bold]Loading into Kuzu[/bold] {db} ...")
    _, conn = open_db(db, fresh=fresh)
    if to_touch:
        delete_files(conn, sorted(to_touch))

    if embeddings_available():
        console.print("[bold]Embedding[/bold] node names/docstrings for semantic search ...")
    else:
        console.print("[dim]fastembed not installed - skipping semantic search embeddings "
                       "(Ask falls back to TF-IDF only). pip install fastembed to enable it.[/dim]")
    load_graph(conn, graph)
    ensure_vector_index(conn)

    php_files = discover_php_files(repo_path)
    if php_files:
        console.print(f"[bold]Extracting RBAC facts[/bold] from {len(php_files)} PHP file(s) ...")
        concepts, rules = rbac_extractor.extract(repo_path, php_files)
        load_concepts(conn, concepts)
        load_rules(conn, rules)
        console.print(f"  found {len(concepts)} concept(s), {len(rules)} rule(s)")

    if incremental:
        with open(manifest_file, "w", encoding="utf-8") as fh:
            json.dump(new_hashes, fh)

    console.print("[green]Done.[/green]")


@app.command()
def stats(db: str = typer.Option(DEFAULT_DB, "--db")):
    """Show node/edge counts per type."""
    _, conn = open_db(db, fresh=False)
    table = Table(title="Graph stats")
    table.add_column("Kind")
    table.add_column("Count", justify="right")
    for kind in ("Module", "Class", "Function", "External"):
        n = conn.execute(f"MATCH (n:{kind}) RETURN count(n)").get_as_df().iloc[0, 0]
        table.add_row(kind, str(n))
    for kind in ("CONTAINS", "IMPORTS", "CALLS", "INHERITS"):
        n = conn.execute(f"MATCH ()-[r:{kind}]->() RETURN count(r)").get_as_df().iloc[0, 0]
        table.add_row(f"[dim]{kind}[/dim]", str(n))
    console.print(table)


@app.command()
def visualize(
    db: str = typer.Option(DEFAULT_DB, "--db"),
    out: str = typer.Option("./graph.html", "--out", help="Path to write the HTML visualization"),
    limit: int = typer.Option(500, "--limit", help="Max nodes to include (keeps large graphs readable)"),
):
    """Export the graph to a single self-contained HTML file - a
    force-directed view you open directly in a browser. No CDN scripts, no
    server process; same zero-server idea as the rest of the tool."""
    _, conn = open_db(db, fresh=False)
    data = export_graph_json(conn)

    if len(data["nodes"]) > limit:
        keep_ids = {n["id"] for n in data["nodes"][:limit]}
        data["nodes"] = [n for n in data["nodes"] if n["id"] in keep_ids]
        data["edges"] = [e for e in data["edges"] if e["src"] in keep_ids and e["dst"] in keep_ids]
        console.print(f"[yellow]Graph exceeds --limit ({limit}) nodes - showing the first {limit}.[/yellow]")

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render_html(data))
    console.print(
        f"[green]Wrote[/green] {out} ({len(data['nodes'])} nodes, {len(data['edges'])} edges) - open it in a browser."
    )


@app.command()
def obsidian(
    db: str = typer.Option(DEFAULT_DB, "--db"),
    out: str = typer.Option("./obsidian_vault", "--out", help="Folder to write as an Obsidian vault"),
):
    """Export the graph as an Obsidian vault - one note per node, linked
    via Obsidian wikilinks along CONTAINS/IMPORTS/CALLS/INHERITS edges. Open
    the output folder in Obsidian (File -> Open folder as vault) and use
    its built-in Graph View to explore the codebase visually."""
    _, conn = open_db(db, fresh=False)
    data = export_graph_json(conn)
    export_vault(data, out)
    console.print(
        f"[green]Wrote[/green] {len(data['nodes'])} notes to {out} - "
        f"open that folder in Obsidian (File -> Open folder as vault) and check the Graph View."
    )


@app.command()
def serve(
    db: str = typer.Option(DEFAULT_DB, "--db"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't auto-open a browser tab"),
):
    """Launch a local web dashboard for the whole tool - Stats, Explore
    (callers/impact search), Graph (interactive force-directed view), Ask
    (Graph RAG), and a raw Cypher console - all in one browser tab backed
    by a stdlib-only HTTP server (no new dependencies, nothing to deploy).
    The dashboard's repo dropdown can also index new repos on the fly
    (folder upload) and switch between every repo you've indexed so far."""
    run_server(db, host=host, port=port, open_browser=not no_browser)


@app.command()
def cypher(query: str = typer.Argument(..., help="Raw Cypher query"),
           db: str = typer.Option(DEFAULT_DB, "--db")):
    """Run a raw Cypher query against the graph."""
    _, conn = open_db(db, fresh=False)
    df = conn.execute(query).get_as_df()
    console.print(df.to_string(index=False))


@app.command()
def callers(function_name: str = typer.Argument(..., help="Function name to find callers of"),
            db: str = typer.Option(DEFAULT_DB, "--db")):
    """List all functions that call a given function name."""
    _, conn = open_db(db, fresh=False)
    rows = get_callers(conn, function_name)
    if not rows:
        console.print(f"No callers found for [bold]{function_name}[/bold]")
    else:
        console.print(pd.DataFrame(rows).to_string(index=False))


@app.command()
def impact(function_name: str = typer.Argument(..., help="Function name to check blast radius of"),
           db: str = typer.Option(DEFAULT_DB, "--db"),
           hops: int = typer.Option(3, "--hops", help="How many CALLS hops back to trace (1-10)")):
    """Show the transitive blast radius (callers of callers) of a function - useful before refactors."""
    _, conn = open_db(db, fresh=False)
    rows = get_impact(conn, function_name, max_hops=hops)
    if not rows:
        console.print(f"No dependents found for [bold]{function_name}[/bold] (safe to change in isolation)")
    else:
        console.print(f"[bold]{len(rows)} function(s)[/bold] transitively depend on {function_name}:")
        console.print(pd.DataFrame(rows).to_string(index=False))


@app.command()
def roles(
    code: str = typer.Argument(None, help="Role code/constant to show rules for (omit to list all roles)"),
    db: str = typer.Option(DEFAULT_DB, "--db"),
):
    """Deterministic, no-LLM answer to "what roles/permissions are there":
    lists every Role Concept extracted by rbac_extractor.py, or (with a
    code argument) every RULE gating that one role - built entirely from
    graph facts, never generated/guessed text."""
    _, conn = open_db(db, fresh=False)
    if code:
        rows = get_concept_rules(conn, code)
        if not rows:
            console.print(f"No rules found for role [bold]{code}[/bold] (or it doesn't exist - see `roles` with no argument).")
            return
        table = Table(title=f"Rules gating role {code}")
        for col in ("effect", "target", "file", "lineno", "condition"):
            table.add_column(col)
        for r in rows:
            table.add_row(str(r["effect"]), str(r["target"]), str(r["file"]), str(r["lineno"]), str(r["condition"])[:70])
        console.print(table)
    else:
        rows = list_concepts(conn, kind="Role")
        if not rows:
            console.print("No roles extracted - run `index` on a repo with RBAC-shaped PHP code first.")
            return
        table = Table(title="Roles")
        for col in ("app", "code", "name", "source_file", "lineno"):
            table.add_column(col)
        for r in rows:
            table.add_row(str(r["app"]), str(r["code"]), str(r["name"]), str(r["source_file"]), str(r["lineno"]))
        console.print(table)


@app.command()
def ask(question: str = typer.Argument(..., help="Natural language question about the codebase"),
        db: str = typer.Option(DEFAULT_DB, "--db"),
        model: str = typer.Option("llama-3.3-70b-versatile", "--model")):
    """Ask a natural-language question; pulls relevant graph context and
    sends it to Groq's LLM API (set GROQ_API_KEY env var to use this).
    This mirrors GitNexus's Graph RAG agent, just via a REST call instead
    of an in-browser agent.

    If the graph context isn't enough to answer confidently, or the
    question is ambiguous, it'll ask a clarifying question back instead
    of guessing - answer it at the prompt to continue, or Ctrl+C to stop.
    """
    if not os.environ.get("GROQ_API_KEY"):
        console.print("[red]GROQ_API_KEY not set.[/red] export GROQ_API_KEY=... and retry.")
        raise typer.Exit(1)

    _, conn = open_db(db, fresh=False)

    history: list[dict] = []
    current = question
    while True:
        result = _ask(conn, current, model, history)
        if "error" in result:
            console.print(f"[red]{result['error']}[/red]")
            raise typer.Exit(1)
        if result["type"] == "clarify":
            console.print(f"[cyan]🤔  {result['content']}[/cyan]")
            history.append({"question": current, "answer": result["content"]})
            current = typer.prompt("Your answer")
            continue
        console.print(result["content"])
        return


if __name__ == "__main__":
    app()
