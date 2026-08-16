"""
clustering.py
--------------
Segments a module's Class/Function nodes into "chapters" for
docgen.generate_docs - groups of tightly-connected code that read as one
logical feature, rather than one chapter per class or per file.

Deliberately NOT graph-clustering-library-based (no Louvain/networkx):
plain deterministic Union-Find over cross-file CALLS/INHERITS edges within
a module. Two properties this buys that a modularity-optimization
algorithm doesn't:

  1. Determinism - no random seed, no iteration-order dependence. The same
     graph always produces the same components.
  2. Locality of change - adding/removing a node or edge only ever affects
     the component(s) touching it; an edit somewhere in the module can't
     silently reshuffle an unrelated cluster elsewhere. Louvain-style
     modularity optimization recomputes gains globally each pass and can
     shift boundaries anywhere in the graph from one local edit.

Property 2 is what makes docgen.py's incremental regeneration sound: a
chapter's membership only ever changes because of a real, local content
change, so "chapter unaffected by this edit" is a safe assumption to build
change detection on.

This module is pure Python over plain node/edge dicts - no DB or LLM
coupling, so it's cheap to unit-test and to re-run per `docs generate`
invocation without touching the network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_CHAPTER_NODES = 40
MISC_CHAPTER_NAME = "misc"


@dataclass
class Chapter:
    chapter_id: str          # "<module>/<slug>", stable across re-clustering (see _anchor_slug)
    module: str
    node_ids: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)   # sorted, deduped


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug or "chapter"


def _anchor_slug(member_files: list[str]) -> str:
    """Derive a chapter's slug from its "anchor" file - the file most of
    its member nodes actually live in (mode, ties broken alphabetically),
    not just any file it happens to touch. A single file's functions can
    end up split across multiple connected components (they call
    different, unrelated other files), so picking "alphabetically first
    file in the touched set" would collide whenever two such components
    both touch that file; picking the dominant file per component avoids
    that in the common case. `member_files` is the (non-deduped) file of
    every member node, so counting occurrences gives the true mode."""
    counts: dict[str, int] = {}
    for f in member_files:
        counts[f] = counts.get(f, 0) + 1
    anchor = max(sorted(counts), key=lambda f: counts[f])
    basename = anchor.replace("\\", "/").rsplit("/", 1)[-1]
    basename = basename.rsplit(".", 1)[0]  # strip extension
    return _slugify(basename)


def _dedupe_chapter_ids(chapters: list["Chapter"]) -> None:
    """Safety net for the rare case where two components still land on the
    same anchor slug (e.g. genuinely tied dominant files) - appends a
    deterministic numeric suffix ordered by each chapter's sorted node_ids,
    so chapter_ids stay unique and reproducible across runs."""
    by_id: dict[str, list[Chapter]] = {}
    for c in chapters:
        by_id.setdefault(c.chapter_id, []).append(c)
    for chapter_id, group in by_id.items():
        if len(group) <= 1:
            continue
        group.sort(key=lambda c: tuple(sorted(c.node_ids)))
        for i, chapter in enumerate(group, start=1):
            chapter.chapter_id = f"{chapter_id}-{i}"


class _UnionFind:
    """Minimal dict-based union-find - no external dependency needed for
    something this small."""

    def __init__(self, items: list[str]):
        self._parent = {item: item for item in items}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic tie-break: lower id becomes the root, so the
            # resulting grouping doesn't depend on edge insertion order
            if ra > rb:
                ra, rb = rb, ra
            self._parent[rb] = ra


def cluster_module(nodes: list[dict], cross_file_edges: list[tuple[str, str]],
                    max_chapter_nodes: int = MAX_CHAPTER_NODES) -> list[Chapter]:
    """nodes: list of {id, name, file, ...} (Class/Function nodes for one
    module, from db.get_nodes_by_file_prefix). cross_file_edges: list of
    (src_id, dst_id) pairs from db.get_cluster_edges (CALLS + INHERITS,
    already restricted to this module's node set).

    Returns one Chapter per connected component, except:
      - all singleton nodes (no in-module cross-file edges) are merged
        into one shared "misc" chapter per module, rather than one chapter
        each - avoids fragmenting a module with many small utility
        files/classes into dozens of one-node chapters.
      - components larger than max_chapter_nodes are deterministically
        split by file (sorted lexicographically, greedily bin-packed) -
        never by traversal/discovery order, which would be edge-order
        dependent and so not stable under edits.
    """
    if not nodes:
        return []

    module = _module_of(nodes)
    by_id = {n["id"]: n for n in nodes}
    uf = _UnionFind(list(by_id.keys()))
    for src, dst in cross_file_edges:
        if src in by_id and dst in by_id:
            uf.union(src, dst)

    components: dict[str, list[str]] = {}
    for node_id in by_id:
        root = uf.find(node_id)
        components.setdefault(root, []).append(node_id)

    misc_ids: list[str] = []
    real_components: list[list[str]] = []
    for member_ids in components.values():
        if len(member_ids) == 1:
            misc_ids.extend(member_ids)
        else:
            real_components.append(member_ids)

    chapters: list[Chapter] = []
    for member_ids in real_components:
        chapters.extend(_split_oversized(module, member_ids, by_id, max_chapter_nodes))

    if misc_ids:
        chapters.append(Chapter(
            chapter_id=f"{module}/{MISC_CHAPTER_NAME}",
            module=module,
            node_ids=sorted(misc_ids),
            files=sorted({by_id[i]["file"] for i in misc_ids}),
        ))

    _dedupe_chapter_ids(chapters)
    # deterministic overall ordering, independent of dict/set iteration order
    chapters.sort(key=lambda c: c.chapter_id)
    return chapters


def _module_of(nodes: list[dict]) -> str:
    file = nodes[0]["file"].replace("\\", "/").strip("/")
    return file.split("/", 1)[0] if "/" in file else "(root)"


def _split_oversized(module: str, member_ids: list[str], by_id: dict[str, dict],
                      max_chapter_nodes: int) -> list[Chapter]:
    files = sorted({by_id[i]["file"] for i in member_ids})
    if len(member_ids) <= max_chapter_nodes:
        member_files = [by_id[i]["file"] for i in member_ids]
        return [Chapter(
            chapter_id=f"{module}/{_anchor_slug(member_files)}",
            module=module, node_ids=sorted(member_ids), files=files,
        )]

    # Deterministic split: walk files in sorted order, greedily bin-pack
    # their nodes into sub-chapters capped at max_chapter_nodes. A single
    # file's own node count is never split across bins (keeps a file's
    # nodes together in one chapter).
    ids_by_file: dict[str, list[str]] = {}
    for node_id in member_ids:
        ids_by_file.setdefault(by_id[node_id]["file"], []).append(node_id)

    bins: list[list[str]] = []
    bin_files: list[list[str]] = []
    for file in files:
        file_ids = sorted(ids_by_file[file])
        if bins and len(bins[-1]) + len(file_ids) <= max_chapter_nodes:
            bins[-1].extend(file_ids)
            bin_files[-1].append(file)
        else:
            bins.append(list(file_ids))
            bin_files.append([file])

    result = []
    for bin_ids, bin_file_list in zip(bins, bin_files):
        bin_member_files = [by_id[i]["file"] for i in bin_ids]
        result.append(Chapter(
            chapter_id=f"{module}/{_anchor_slug(bin_member_files)}",
            module=module, node_ids=sorted(bin_ids), files=sorted(bin_file_list),
        ))
    return result
