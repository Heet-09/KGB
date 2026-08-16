"""
docgen.py
---------
Generates module- and chapter-wise Markdown documentation by walking the
already-indexed Kuzu graph - no user question involved, unlike web._ask.
For each module (top-level file-path segment), clustering.cluster_module
groups its Class/Function nodes into "chapters" (tightly cross-file-
connected feature clusters), and each chapter gets one LLM call (reusing
web._call_groq) that turns its node/relationship listing into a Markdown
chapter. Chapters are assembled into docs/<module>/<chapter-slug>.md plus
a generated mkdocs.yml, renderable via `mkdocs build`/`mkdocs serve`.

Incremental regeneration (--incremental): a docs_manifest.json sidecar in
the output dir records each chapter's structural content hash and member
node-id set as of the last run. Re-clustering is always redone in full
(cheap, no LLM) so the manifest is always compared against the *current*
graph; only chapters that are new, changed, or whose membership changed
get a fresh `_call_groq` call. This is only sound because clustering is
deterministic and locally-stable under edits - see clustering.py's
module docstring.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import kuzu

from . import clustering
from .clustering import Chapter
from .db import get_module_prefixes, get_nodes_by_file_prefix, get_cluster_edges, get_edges_among
from .web import _call_groq, _summarize

MANIFEST_FILENAME = "docs_manifest.json"


def _cluster_all_modules(conn: kuzu.Connection, max_chapter_nodes: int) -> list[Chapter]:
    """Re-derive the full, current chapter set from the graph - cheap
    (no LLM calls), so it's safe to run every time, including on every
    --incremental invocation, to know what *should* exist before diffing
    against the manifest."""
    chapters: list[Chapter] = []
    for prefix in get_module_prefixes(conn):
        nodes = get_nodes_by_file_prefix(conn, prefix)
        if not nodes:
            continue
        node_ids = [n["id"] for n in nodes]
        calls = get_cluster_edges(conn, node_ids, "CALLS", cross_file_only=True)
        inherits = get_cluster_edges(conn, node_ids, "INHERITS", cross_file_only=False)
        chapters.extend(clustering.cluster_module(nodes, calls + inherits, max_chapter_nodes))
    return chapters


def _node_signature(node: dict) -> str:
    return f"{node['id']}|{node['file']}|{node['lineno']}|{node['end_lineno']}|{node['docstring']}"


def _hash_chapter(nodes_by_id: dict[str, dict], chapter: Chapter) -> str:
    """Structural hash a chapter is stale against: id + file + line range +
    docstring for every member node, sorted for order-independence.
    Deliberately excludes embeddings (irrelevant to doc text) to keep this
    cheap and stable."""
    sigs = sorted(_node_signature(nodes_by_id[nid]) for nid in chapter.node_ids if nid in nodes_by_id)
    return hashlib.sha256("\n".join(sigs).encode("utf-8")).hexdigest()


def build_chapter_prompt(conn: kuzu.Connection, chapter: Chapter, nodes_by_id: dict[str, dict]) -> tuple[str, str]:
    members = [nodes_by_id[nid] for nid in chapter.node_ids if nid in nodes_by_id]
    lines = [f"Module: {chapter.module}    Chapter: {chapter.chapter_id.split('/', 1)[-1]}", "", "Functions/Classes:"]
    for n in members:
        loc = f"{n['file']}:{n['lineno']}-{n['end_lineno']}"
        doc = (n["docstring"] or "").strip().splitlines()[0] if n["docstring"] else ""
        lines.append(f"- ({n['kind']}) {n['name']} [{loc}]" + (f": {doc[:160]}" if doc else ""))

    edges = get_edges_among(conn, chapter.node_ids)
    if edges:
        lines.append("")
        lines.append("Relationships:")
        for e in edges:
            verb = {"CALLS": "calls", "IMPORTS": "imports", "INHERITS": "inherits from"}[e["kind"]]
            lines.append(f"- {e['src_name']} {verb} {e['dst_name']}")

    user_prompt = "\n".join(lines)
    system_prompt = (
        "You are a senior engineer writing one chapter of a codebase's reference documentation. "
        "You're given a cluster of functions/classes from one module that are closely related "
        "(they call each other or inherit from each other), their locations, docstrings, and their "
        "relationships. Write a Markdown chapter explaining what this code does: "
        "start with '## Overview' (2-4 sentences on the chapter's purpose/responsibility), then "
        "'## Key components' (one subsection per function/class - what it does, not just its "
        "signature), then '## How they interact' (the call/inheritance flow, in prose). "
        "Infer purpose from names/docstrings/relationships - don't just restate the listing. "
        "Output valid Markdown only, no preamble, no code fences around the whole thing."
    )
    return system_prompt, user_prompt


def generate_chapter_markdown(conn: kuzu.Connection, chapter: Chapter, nodes_by_id: dict[str, dict],
                               model: str | None) -> str:
    system_prompt, user_prompt = build_chapter_prompt(conn, chapter, nodes_by_id)
    return _call_groq(system_prompt, user_prompt, model)


def _load_manifest(output_dir: str) -> dict:
    path = Path(output_dir) / MANIFEST_FILENAME
    if not path.exists():
        return {"chapters": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"chapters": {}}


def _save_manifest(output_dir: str, manifest: dict) -> None:
    path = Path(output_dir) / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_chapter(output_dir: str, chapter: Chapter, markdown: str) -> str:
    """Writes docs/<module>/<slug>.md and returns the path, relative to
    output_dir, for use in mkdocs.yml's nav."""
    rel_path = f"{chapter.chapter_id}.md"
    full_path = Path(output_dir) / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(markdown, encoding="utf-8")
    return rel_path.replace("\\", "/")


def build_mkdocs_nav(chapters_by_module: dict[str, list[Chapter]]) -> list:
    nav: list = [{"Home": "index.md"}]
    for module in sorted(chapters_by_module):
        entries = []
        for chapter in sorted(chapters_by_module[module], key=lambda c: c.chapter_id):
            title = chapter.chapter_id.split("/", 1)[-1].replace("-", " ").title()
            entries.append({title: f"{chapter.chapter_id}.md"})
        nav.append({module: entries})
    return nav


def _yaml_dump_nav(nav: list, indent: int = 0) -> str:
    """Tiny hand-rolled nav-only YAML writer - avoids adding PyYAML as a
    dependency just for this. Only needs to handle the shapes build_mkdocs_nav
    produces: a list of single-key dicts, values either a string or a
    nested list of the same shape."""
    lines = []
    pad = "  " * indent
    for item in nav:
        (key, value), = item.items()
        if isinstance(value, str):
            lines.append(f'{pad}- {key}: {value}')
        else:
            lines.append(f'{pad}- {key}:')
            lines.append(_yaml_dump_nav(value, indent + 1))
    return "\n".join(lines)


def write_mkdocs_yml(output_dir: str, site_name: str, nav: list) -> None:
    body = f"site_name: {site_name}\ndocs_dir: .\nnav:\n{_yaml_dump_nav(nav, indent=1)}\n"
    (Path(output_dir) / "mkdocs.yml").write_text(body, encoding="utf-8")


def generate_docs(conn: kuzu.Connection, output_dir: str, incremental: bool = False,
                   model: str | None = None, max_chapter_nodes: int = clustering.MAX_CHAPTER_NODES,
                   site_name: str = "Codebase Docs") -> dict:
    """Top-level orchestrator - the single entry point cli.py's `docs
    generate` command calls. Returns a small summary dict for the CLI to
    print (chapters written/reused/deleted)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    old_manifest = _load_manifest(output_dir) if incremental else {"chapters": {}}

    chapters = _cluster_all_modules(conn, max_chapter_nodes)
    all_node_ids = sorted({nid for c in chapters for nid in c.node_ids})
    nodes_by_id: dict[str, dict] = {}
    for prefix in get_module_prefixes(conn):
        for n in get_nodes_by_file_prefix(conn, prefix):
            nodes_by_id[n["id"]] = n

    written, reused = 0, 0
    new_chapters: dict[str, dict] = {}
    chapters_by_module: dict[str, list[Chapter]] = {}
    for chapter in chapters:
        chapters_by_module.setdefault(chapter.module, []).append(chapter)
        content_hash = _hash_chapter(nodes_by_id, chapter)
        old = old_manifest.get("chapters", {}).get(chapter.chapter_id)
        node_id_set_unchanged = old is not None and sorted(old.get("node_ids", [])) == sorted(chapter.node_ids)
        stale = not incremental or old is None or old.get("content_hash") != content_hash or not node_id_set_unchanged

        if stale:
            markdown = generate_chapter_markdown(conn, chapter, nodes_by_id, model)
            output_path = write_chapter(output_dir, chapter, markdown)
            written += 1
        else:
            output_path = old["output_path"]
            reused += 1

        new_chapters[chapter.chapter_id] = {
            "module": chapter.module, "files": chapter.files, "node_ids": chapter.node_ids,
            "content_hash": content_hash, "output_path": output_path,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    deleted = 0
    for old_id, old_entry in old_manifest.get("chapters", {}).items():
        if old_id not in new_chapters:
            stale_path = Path(output_dir) / old_entry["output_path"]
            if stale_path.exists():
                stale_path.unlink()
            deleted += 1

    index_path = Path(output_dir) / "index.md"
    index_path.write_text(f"# {site_name}\n\n" + _summarize(conn, model), encoding="utf-8")

    write_mkdocs_yml(output_dir, site_name, build_mkdocs_nav(chapters_by_module))
    _save_manifest(output_dir, {"chapters": new_chapters})

    return {"modules": len(chapters_by_module), "chapters": len(chapters),
            "written": written, "reused": reused, "deleted": deleted}
