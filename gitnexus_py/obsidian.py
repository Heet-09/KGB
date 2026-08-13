"""
obsidian.py
-----------
Exports the graph as an Obsidian vault: one Markdown note per node
(Module/Class/Function/External), linked to related notes via
Obsidian wikilinks along CONTAINS/IMPORTS/CALLS/INHERITS edges.

Opening the output folder as an Obsidian vault (File -> Open folder as
vault) gives you Obsidian's built-in Graph View as a free way to explore
the code graph visually - no custom JS to maintain, and it comes with
Obsidian's own filtering, tag-based groups/colors, local graph per note,
and search, none of which viz.py's canvas view has.
"""

from __future__ import annotations

import os
import re

_INVALID_FS_CHARS = re.compile(r'[\\/:*?"<>|]')

_EDGE_SECTIONS = [
    # (edge kind, heading, which adjacency map to read it from)
    ("CONTAINS", "Contains", "out"),
    ("CONTAINS", "Contained By", "in"),
    ("IMPORTS", "Imports", "out"),
    ("IMPORTS", "Imported By", "in"),
    ("CALLS", "Calls", "out"),
    ("CALLS", "Called By", "in"),
    ("INHERITS", "Inherits From", "out"),
    ("INHERITS", "Inherited By", "in"),
]


def _sanitize(text: str) -> str:
    text = _INVALID_FS_CHARS.sub("_", text).strip()
    return text[:180] or "untitled"


def _note_title(node: dict) -> str:
    """A short, filesystem-safe, human-readable label for a node - used as
    the note's filename, which is what Obsidian shows in Graph View and
    what an Obsidian wikilink resolves against.

    Node ids look like "<kind>::<rel_path>[::<qualname>]" and are already
    globally unique, so slugging everything after the first "::" plus the
    kind is enough to keep titles unique without needing the full id.
    """
    node_id = node["id"]
    qual = node_id.split("::", 1)[1] if "::" in node_id else node_id
    return _sanitize(f"{qual} ({node['kind']})")


def export_vault(data: dict, out_dir: str) -> None:
    """Write `data` (see db.export_graph_json) as an Obsidian vault under
    out_dir - one .md note per node, wikilinked along every edge.
    """
    os.makedirs(out_dir, exist_ok=True)

    # node id -> unique note title (no .md extension)
    titles: dict[str, str] = {}
    used: set[str] = set()
    for node in data["nodes"]:
        base = _note_title(node)
        title, n = base, 2
        while title in used:
            title = f"{base} {n}"
            n += 1
        used.add(title)
        titles[node["id"]] = title

    outgoing: dict[str, dict[str, list[str]]] = {}
    incoming: dict[str, dict[str, list[str]]] = {}
    for edge in data["edges"]:
        outgoing.setdefault(edge["src"], {}).setdefault(edge["kind"], []).append(edge["dst"])
        incoming.setdefault(edge["dst"], {}).setdefault(edge["kind"], []).append(edge["src"])

    for node in data["nodes"]:
        title = titles[node["id"]]
        # normalize to forward slashes: avoids a literal backslash landing
        # in a double-quoted YAML frontmatter string (an invalid escape
        # sequence there, e.g. "pkg\cli.py" -> "\c" isn't a real escape)
        # and reads the same on every OS.
        file_display = node["file"].replace("\\", "/")
        lines = [
            "---",
            f'kind: {node["kind"]}',
            f'file: "{file_display}"',
            f'tags: [gitnexus, {node["kind"]}]',
            "---",
            "",
            f"# {node['name']}",
            "",
            f"*{node['kind']}* in `{file_display}`" if file_display else f"*{node['kind']}*",
        ]
        docstring = (node.get("docstring") or "").strip()
        if docstring:
            lines += ["", docstring]

        for edge_kind, heading, direction in _EDGE_SECTIONS:
            adjacency = outgoing if direction == "out" else incoming
            target_ids = adjacency.get(node["id"], {}).get(edge_kind, [])
            target_titles = sorted({titles[t] for t in target_ids if t in titles})
            if not target_titles:
                continue
            lines += ["", f"## {heading}"]
            lines += [f"- [[{t}]]" for t in target_titles]

        path = os.path.join(out_dir, f"{title}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
