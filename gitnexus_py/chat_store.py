"""
chat_store.py
-------------
Persistence for the "Ask the graph" chat panel (see web.py's /api/chats*
routes and the Ask tab's JS). Separate SQLite database from the per-repo
Kuzu graph DBs in db.py - chat turns are plain relational rows (one chat
has many turns, in order), which is exactly what SQLite is for, and
keeping it out of Kuzu means chat history survives independently of
re-indexing/switching a repo's graph.

One DB file for *all* repos (`.gitnexus_chats.db`, next to
`.gitnexus_repos.json` - see db._registry_path), with each chat tagged by
the repo db_path it belongs to, so the "New chat" / chat list in the UI
only ever shows chats for the currently active repo.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    repo_db_path TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chats_repo ON chats(repo_db_path, updated_at);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(id),
    seq INTEGER NOT NULL,
    question TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '',
    image_data_url TEXT,
    context_question TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, seq);
"""

# messages.context_question was added after the original schema shipped -
# CREATE TABLE IF NOT EXISTS above won't retrofit it onto an existing
# .gitnexus_chats.db, so add it by hand on old databases (idempotent: SQLite
# has no "ADD COLUMN IF NOT EXISTS", hence the try/except).
_MIGRATIONS = [
    "ALTER TABLE messages ADD COLUMN context_question TEXT NOT NULL DEFAULT ''",
]

_TITLE_MAX = 60


def _db_path() -> Path:
    return Path.cwd() / ".gitnexus_chats.db"


_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    """Lazily open (and schema-init) the one chat-history connection. The
    dashboard handler runs single-threaded (see web.py's make_handler
    docstring) so, like ServerState's Kuzu connection, there's no need for
    a connection pool - check_same_thread=False just guards against the
    server's browser-opening timer thread touching it during startup."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
        for stmt in _MIGRATIONS:
            try:
                _conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # already applied (column/table exists) - fine
        _conn.commit()
    return _conn


def _make_title(question: str) -> str:
    q = (question or "").strip().replace("\n", " ")
    if not q:
        return "(image)"
    return q if len(q) <= _TITLE_MAX else q[: _TITLE_MAX - 1].rstrip() + "…"


def create_chat(repo_db_path: str, title: str = "") -> dict:
    """Start a new, empty chat for a repo - the "New chat" button's API
    call. Title is filled in from the first message later (see
    add_message) if not given up front."""
    conn = get_conn()
    chat_id = uuid.uuid4().hex
    now = time.time()
    conn.execute(
        "INSERT INTO chats (id, repo_db_path, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (chat_id, repo_db_path, title, now, now),
    )
    conn.commit()
    return {"id": chat_id, "repo_db_path": repo_db_path, "title": title, "created_at": now, "updated_at": now}


def get_chat(chat_id: str) -> dict | None:
    """One chat's own row (id, repo_db_path, title, timestamps) - used to
    check whether a chat already has a title before spending an LLM call
    generating one (see web.py's /api/chat-messages)."""
    conn = get_conn()
    row = conn.execute(
        "SELECT id, repo_db_path, title, created_at, updated_at FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()
    return dict(row) if row else None


def list_chats(repo_db_path: str) -> list[dict]:
    """Every chat for a repo, most recently active first - what the chat
    history dropdown/list in the Ask tab populates itself from."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM chats "
        "WHERE repo_db_path = ? ORDER BY updated_at DESC",
        (repo_db_path,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_messages(chat_id: str) -> list[dict]:
    """All turns of one chat, oldest first - reconstructs the askHistory
    array the JS keeps client-side (see web.py's /api/ask), so reopening a
    past chat looks exactly like it did live."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT seq, question, answer, type, detail, image_data_url, context_question FROM messages "
        "WHERE chat_id = ? ORDER BY seq ASC",
        (chat_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def add_message(chat_id: str, question: str, answer: str, type_: str = "",
                 detail: str = "", image_data_url: str | None = None, title: str | None = None,
                 context_question: str = "") -> dict:
    """Append one question/answer turn to a chat, bump its updated_at (so
    it sorts to the top of list_chats), and - the first time only - set
    its title. `title`, when given, is an LLM-generated summary of this
    turn (see web.py's _generate_chat_title, called for a chat's first
    message); an existing (already-titled) chat always keeps its title,
    and a falsy `title` (LLM unavailable/failed) falls back to a plain
    truncation of the question, same idea as most chat apps naming a
    thread after its opening message when nothing smarter is available.

    `context_question` is what actually gets replayed to the LLM as this
    turn's "question" on later turns (see _call_groq's history handling in
    web.py) - for an image turn this is the Gemini-extracted description
    (see web.py's _ask_with_image), not the often-empty/placeholder
    `question` text shown in the UI, so a follow-up still has the image's
    content in context after the chat is reloaded from disk. Falls back to
    `question` when not given (plain text turns)."""
    conn = get_conn()
    row = conn.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown chat_id: {chat_id!r}")

    seq_row = conn.execute("SELECT COALESCE(MAX(seq), -1) + 1 AS n FROM messages WHERE chat_id = ?",
                            (chat_id,)).fetchone()
    seq = seq_row["n"]
    now = time.time()
    conn.execute(
        "INSERT INTO messages (chat_id, seq, question, answer, type, detail, image_data_url, "
        "context_question, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (chat_id, seq, question, answer, type_, detail, image_data_url,
         context_question or question, now),
    )
    final_title = row["title"] or title or _make_title(question)
    conn.execute("UPDATE chats SET title = ?, updated_at = ? WHERE id = ?", (final_title, now, chat_id))
    conn.commit()
    return {"seq": seq, "title": final_title, "updated_at": now}


def delete_chat(chat_id: str) -> None:
    """Remove a chat and all its turns - used by the chat list's delete
    control."""
    conn = get_conn()
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()


def delete_message(chat_id: str, seq: int) -> None:
    """Remove one turn from a chat (the per-message delete control) without
    touching the rest - `seq` values are left with a gap rather than
    renumbered, since seq only needs to be a stable ordering key, not a
    dense 0..n-1 range (get_messages/add_message never assume density,
    add_message's own COALESCE(MAX(seq),-1)+1 just needs the *next* number
    to stay ahead of it)."""
    conn = get_conn()
    conn.execute("DELETE FROM messages WHERE chat_id = ? AND seq = ?", (chat_id, seq))
    conn.commit()
