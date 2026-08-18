"""
web.py
------
A small local dashboard for gitnexus-py: one `serve` command spins up a
stdlib-only HTTP server (no new dependencies) that exposes the graph DB
through a JSON API and a single-page HTML/CSS/JS front-end with tabs for
everything the CLI can already do - Stats, Explore (callers/impact search),
Graph (the same force-directed canvas viz.py exports, fetched live instead
of baked into a static file), Ask (Graph RAG via Groq), and Cypher (a raw
query console).

Kept intentionally dependency-free (http.server + json from the stdlib)
to stay close to the rest of the project's zero-server, no-CDN ethos -
"zero-server" here means "you don't install or configure a server", not
"there's no process listening on a port".
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socketserver
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import kuzu

# One log file per working directory (next to .gitnexus_repos.json), plus
# console output - so a failed /api/ask (a bad model name, an HTTP error
# from Groq, GROQ_API_KEY missing, ...) leaves a real diagnostic trail
# instead of just the generic "HTTP Error 404: Not Found" the browser
# shows (see _call_groq - that generic text is exactly str(HTTPError),
# which drops the response body Groq actually explains the failure in).
logger = logging.getLogger("gitnexus_py")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _file_handler = logging.FileHandler(Path.cwd() / "gitnexus_py.log", encoding="utf-8")
    _file_handler.setFormatter(_fmt)
    logger.addHandler(_file_handler)
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(_fmt)
    logger.addHandler(_console_handler)

from . import chat_store
from .db import (
    export_graph_json, get_callers, get_impact, list_function_names,
    load_graph, open_db, ensure_vector_index, vector_search, NODE_KINDS, REL_KINDS,
    list_concepts, get_concept_rules, get_rules_for_file, list_module_files, get_inherits,
    get_subclasses, resolve_url, get_role_page_matrix, get_views_rendered,
)
from .parser import parse_repo
from .retrieval import (
    rank_context, find_mentioned_names, find_hop_count, find_mentioned_file, classify_intents,
    extract_url_or_path,
)

_NODE_COLORS = {
    "Module": "#4c6ef5",
    "Class": "#f76707",
    "Function": "#2f9e44",
    "External": "#868e96",
}

_EDGE_COLORS = {
    "CONTAINS": "#adb5bd",
    "IMPORTS": "#4c6ef5",
    "CALLS": "#2f9e44",
    "INHERITS": "#f76707",
}

_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>gitnexus-py dashboard</title>
<style>
  :root {
    --bg: #0f1115; --panel: #1a1d24; --panel-2: #20242d; --text: #e8e8e8;
    --muted: #9aa0aa; --border: #2a2e37; --accent: #4c6ef5; --accent-2: #2f9e44;
    --danger: #e03131; --code-bg: #14161b;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f7f7f8; --panel: #ffffff; --panel-2: #f0f1f3; --text: #1a1a1a;
      --muted: #666; --border: #e0e0e0; --code-bg: #f0f1f3;
    }
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font: 14px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; height: 100%; }
  #app { display: flex; flex-direction: column; height: 100vh; }

  header { display: flex; align-items: center; gap: 16px; padding: 10px 18px;
    border-bottom: 1px solid var(--border); background: var(--panel); flex-shrink: 0; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  header .sub { color: var(--muted); font-size: 12px; }
  nav { display: flex; gap: 4px; margin-left: auto; }
  nav button { background: none; border: 1px solid transparent; color: var(--muted);
    padding: 7px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; }
  nav button:hover { background: var(--panel-2); color: var(--text); }
  nav button.active { background: var(--accent); color: #fff; }

  main { flex: 1; overflow: auto; padding: 20px; }
  .view { display: none; max-width: 1100px; margin: 0 auto; }
  .view.active { display: block; }
  .view.wide { max-width: none; }

  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 18px; margin-bottom: 16px; }
  .card h2 { font-size: 13px; margin: 0 0 12px; text-transform: uppercase; letter-spacing: .04em;
    color: var(--muted); }

  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
  .stat-tile { background: var(--panel-2); border-radius: 8px; padding: 14px; text-align: center; }
  .stat-tile .n { font-size: 26px; font-weight: 700; }
  .stat-tile .k { color: var(--muted); font-size: 12px; margin-top: 2px; }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }
  tr:hover td { background: var(--panel-2); }
  .empty { color: var(--muted); padding: 10px 0; font-size: 13px; }

  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  input[type=text], textarea, select {
    background: var(--code-bg); border: 1px solid var(--border); color: var(--text);
    border-radius: 6px; padding: 8px 10px; font-size: 13px; font-family: inherit;
  }
  input[type=text] { flex: 1; min-width: 180px; }
  textarea { width: 100%; min-height: 90px; font-family: ui-monospace, Consolas, monospace;
    resize: vertical; }
  button.primary { background: var(--accent); color: #fff; border: none; border-radius: 6px;
    padding: 8px 16px; font-size: 13px; font-weight: 600; cursor: pointer; }
  button.primary:hover { filter: brightness(1.08); }
  button.primary:disabled { opacity: .5; cursor: default; }

  .badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px;
    font-weight: 600; color: #fff; }
  .muted { color: var(--muted); }
  code.inline { background: var(--code-bg); padding: 1px 5px; border-radius: 4px;
    font-family: ui-monospace, Consolas, monospace; }
  .error { color: var(--danger); font-size: 13px; margin-top: 8px; white-space: pre-wrap; }
  .answer { white-space: pre-wrap; line-height: 1.6; margin-top: 12px; }
  .hint { color: var(--muted); font-size: 12px; margin-top: 6px; }

  #askResult { display: flex; flex-direction: column; gap: 14px; margin-top: 14px;
    min-height: 80px; overflow-y: auto; padding-right: 4px; }
  #askResult:empty { display: none; }
  .chat-turn { display: flex; flex-direction: column; gap: 6px; max-width: 80%; }
  .chat-turn.turn-q { align-self: flex-end; align-items: flex-end; }
  .chat-turn.turn-a { align-self: flex-start; align-items: flex-start; }
  .chat-q { position: relative; background: var(--accent); color: #fff; border-radius: 12px 12px 2px 12px;
    padding: 9px 14px; white-space: pre-wrap; line-height: 1.5; }
  .chat-q-img { max-width: 220px; max-height: 160px; border-radius: 8px; display: block; margin-bottom: 6px; }
  .msg-del { position: absolute; top: -8px; left: -8px; width: 18px; height: 18px; border-radius: 50%;
    background: var(--panel); border: 1px solid var(--border); color: var(--muted); font-size: 12px;
    line-height: 16px; text-align: center; cursor: pointer; visibility: hidden; }
  .chat-turn.turn-q:hover .msg-del { visibility: visible; }
  .msg-del:hover { color: var(--danger); border-color: var(--danger); }
  #askImagePreview { display: flex; align-items: center; gap: 10px; background: var(--panel-2);
    border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
  #askImagePreview img { max-height: 60px; border-radius: 6px; }
  #askImagePreview .remove { cursor: pointer; color: var(--muted); font-size: 13px; margin-left: auto; }
  #askImagePreview .remove:hover { color: var(--text); }
  #askAttachBtn { font-size: 16px; padding: 8px 10px; }
  #askAttachBtn.attached { border-color: var(--accent); }
  .chat-a { background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 12px 12px 12px 2px; padding: 9px 14px; white-space: pre-wrap; line-height: 1.6; }
  .chat-a.clarify { border-color: var(--accent); }
  .chat-label { font-size: 10px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
    color: var(--muted); padding: 0 2px; }
  .chat-a.clarify + .chat-label, .turn-a .chat-label.clarify-label { color: var(--accent); }
  .chat-clear { background: none; border: 1px solid var(--border); color: var(--muted);
    border-radius: 6px; padding: 4px 10px; font-size: 12px; cursor: pointer; }
  .chat-clear:hover { background: var(--panel-2); color: var(--text); }
  .chat-typing { display: inline-flex; gap: 4px; align-self: flex-start; background: var(--panel-2);
    border: 1px solid var(--border); border-radius: 12px 12px 12px 2px; padding: 11px 14px; }
  .chat-typing span { width: 6px; height: 6px; border-radius: 50%; background: var(--muted);
    animation: chat-blink 1.2s infinite ease-in-out; }
  .chat-typing span:nth-child(2) { animation-delay: .2s; }
  .chat-typing span:nth-child(3) { animation-delay: .4s; }
  @keyframes chat-blink { 0%, 80%, 100% { opacity: .25; } 40% { opacity: 1; } }
  .chat-empty { color: var(--muted); font-size: 13px; padding: 10px 2px; }
  .chat-detail { width: 100%; }
  .chat-detail summary { cursor: pointer; font-size: 12px; font-weight: 600; color: var(--muted);
    list-style: none; padding: 2px 2px; user-select: none; }
  .chat-detail summary::-webkit-details-marker { display: none; }
  .chat-detail summary::before { content: "Details \25B8"; }
  .chat-detail[open] summary::before { content: "Details \25BE"; }
  .chat-detail summary:hover { color: var(--text); }
  .chat-detail-body { margin-top: 4px; padding: 8px 12px; background: var(--code-bg);
    border: 1px solid var(--border); border-radius: 8px; font-size: 12.5px; line-height: 1.6;
    color: var(--muted); white-space: pre-wrap; }

  /* ---- ChatGPT-style sidebar for the Ask tab ---- */
  .ask-layout { display: flex; gap: 16px; align-items: flex-start; height: calc(100vh - 100px); }
  .chat-sidebar { width: 260px; flex-shrink: 0; height: 100%; display: flex; flex-direction: column;
    gap: 8px; background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px; }
  .chat-sidebar-new { display: flex; align-items: center; justify-content: center; gap: 6px;
    background: none; border: 1px solid var(--border); color: var(--text); border-radius: 8px;
    padding: 10px 12px; font-size: 13px; font-weight: 500; cursor: pointer; flex-shrink: 0; }
  .chat-sidebar-new span { font-size: 15px; line-height: 1; }
  .chat-sidebar-new:hover { background: var(--panel-2); border-color: var(--accent); }
  .chat-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 2px; }
  .chat-list-empty { color: var(--muted); font-size: 12.5px; padding: 10px 8px; }
  .chat-list-item { display: flex; align-items: center; gap: 6px; padding: 8px 10px;
    border-radius: 8px; cursor: pointer; font-size: 13px; color: var(--text); }
  .chat-list-item:hover { background: var(--panel-2); }
  .chat-list-item.active { background: var(--panel-2); box-shadow: inset 0 0 0 1px var(--accent); }
  .chat-list-item .title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .chat-list-item .del { flex-shrink: 0; color: var(--muted); font-size: 13px; line-height: 1;
    padding: 2px 4px; border-radius: 4px; visibility: hidden; }
  .chat-list-item:hover .del { visibility: visible; }
  .chat-list-item .del:hover { color: var(--danger); background: var(--panel); }
  .ask-main { flex: 1; min-width: 0; height: 100%; display: flex; flex-direction: column;
    overflow: hidden; margin-bottom: 0; }
  .ask-main #askResult { flex: 1; }

  #graphWrap { position: relative; height: calc(100vh - 130px); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden; background: var(--panel); }
  #canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  #canvas.dragging { cursor: grabbing; }
  #graphPanel { position: absolute; top: 12px; left: 12px; background: var(--panel);
    border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px;
    max-width: 300px; box-shadow: 0 4px 16px rgba(0,0,0,.25); }
  #graphPanel .stat { color: var(--muted); font-size: 12px; }
  .legend { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
  .legend span { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; }
  .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  #tooltip { position: fixed; pointer-events: none; background: var(--panel);
    border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px;
    display: none; max-width: 360px; box-shadow: 0 4px 16px rgba(0,0,0,.3); z-index: 10; font-size: 12px; }
  #hint { position: absolute; bottom: 10px; left: 12px; color: var(--muted); font-size: 11px; }
  .spinner { color: var(--muted); font-size: 13px; }

  #graphControls { position: absolute; top: 12px; right: 12px; display: flex; gap: 8px;
    align-items: center; z-index: 5; }
  #graphControls select { font-size: 12px; padding: 6px 8px; }
  button.secondary { background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 600; cursor: pointer;
    box-shadow: 0 4px 16px rgba(0,0,0,.2); }
  button.secondary:hover { background: var(--panel-2); }
  button.secondary:disabled { opacity: .5; cursor: default; }
  #nodeDetails { position: absolute; top: 12px; right: 12px; width: 300px; max-height: calc(100% - 24px);
    overflow: auto; background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 14px; box-shadow: 0 4px 16px rgba(0,0,0,.3); display: none; font-size: 12px; z-index: 6; }
  #nodeDetails.open { display: block; }
  #nodeDetails h3 { margin: 0 0 4px; font-size: 14px; }
  #nodeDetails .close { position: absolute; top: 8px; right: 10px; cursor: pointer; color: var(--muted);
    font-size: 16px; line-height: 1; }
  #nodeDetails .file { color: var(--muted); font-family: ui-monospace, Consolas, monospace;
    font-size: 11px; word-break: break-all; margin-bottom: 8px; }
  #nodeDetails .doc { margin-bottom: 10px; line-height: 1.5; }
  #nodeDetails .neigh-heading { font-weight: 600; margin-top: 8px; color: var(--muted);
    text-transform: uppercase; letter-spacing: .04em; font-size: 10px; }
  #nodeDetails ul { margin: 4px 0 0; padding-left: 16px; }
  #nodeDetails li { margin-bottom: 2px; cursor: pointer; }
  #nodeDetails li:hover { color: var(--accent); }
  #summaryPanel { position: absolute; top: 12px; left: 50%; transform: translateX(-50%); width: min(560px, 90%);
    max-height: calc(100% - 24px); overflow: auto; background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px; box-shadow: 0 8px 28px rgba(0,0,0,.35); display: none;
    z-index: 7; }
  #summaryPanel.open { display: block; }
  #summaryPanel h3 { margin: 0 0 8px; font-size: 14px; }
  #summaryPanel .close { position: absolute; top: 10px; right: 12px; cursor: pointer; color: var(--muted);
    font-size: 18px; line-height: 1; }
  #summaryPanel .body { white-space: pre-wrap; line-height: 1.6; font-size: 13px; }

  #repoSelect { font-size: 13px; padding: 5px 8px; max-width: 220px; }
  #addRepoBtn { white-space: nowrap; }
  #uploadOverlay { position: fixed; inset: 0; background: rgba(0,0,0,.5); display: none;
    align-items: center; justify-content: center; z-index: 100; }
  #uploadOverlay.open { display: flex; }
  #uploadOverlay .box { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 20px 24px; min-width: 280px; text-align: center; box-shadow: 0 8px 28px rgba(0,0,0,.4); }
  #uploadOverlay .box .msg { margin-bottom: 4px; font-size: 13px; }
  #uploadOverlay .box .sub { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>gitnexus-py</h1>
    <span class="sub" id="dbPathFallback">__DB_PATH__</span>
    <select id="repoSelect" style="display:none"></select>
    <button class="secondary" id="addRepoBtn">+ Add repo</button>
    <input type="file" id="repoFolderInput" webkitdirectory directory multiple style="display:none">
    <nav>
      <button data-view="stats" class="active">Stats</button>
      <button data-view="explore">Explore</button>
      <button data-view="graph">Graph</button>
      <button data-view="ask">Ask</button>
      <button data-view="cypher">Cypher</button>
    </nav>
  </header>
  <div id="uploadOverlay">
    <div class="box">
      <div class="msg" id="uploadMsg">Indexing repo...</div>
      <div class="sub" id="uploadSub"></div>
    </div>
  </div>
  <main>

    <section id="view-stats" class="view active">
      <div class="card">
        <h2>Node counts</h2>
        <div class="stat-grid" id="nodeStats"><div class="spinner">Loading...</div></div>
      </div>
      <div class="card">
        <h2>Edge counts</h2>
        <div class="stat-grid" id="edgeStats"></div>
      </div>
    </section>

    <section id="view-explore" class="view">
      <div class="card">
        <h2>Function lookup</h2>
        <div class="row">
          <input type="text" id="exploreInput" placeholder="function name, e.g. parse_repo">
          <select id="exploreHops" title="How many CALLS hops back to trace for impact">
            <option value="1">1 hop</option>
            <option value="2">2 hops</option>
            <option value="3" selected>3 hops</option>
            <option value="5">5 hops</option>
            <option value="10">10 hops</option>
          </select>
          <button class="primary" id="exploreBtn">Search</button>
        </div>
        <p class="hint">Shows who calls this function (callers) and the transitive blast radius
          (impact) - same as the <code class="inline">callers</code> /
          <code class="inline">impact --hops N</code> CLI commands.</p>
      </div>
      <div class="card">
        <h2>Callers</h2>
        <div id="callersResult" class="empty">Search a function above.</div>
      </div>
      <div class="card">
        <h2>Impact (transitive)</h2>
        <div id="impactResult" class="empty">Search a function above.</div>
      </div>
    </section>

    <section id="view-graph" class="view wide">
      <div id="graphWrap">
        <canvas id="canvas"></canvas>
        <div id="graphPanel">
          <div class="stat" id="graphStat">Loading graph...</div>
          <div class="legend" id="legend">__LEGEND__</div>
        </div>
        <div id="graphControls">
          <select id="colorMode">
            <option value="kind">Color: node type</option>
            <option value="folder">Color: top-level folder</option>
          </select>
          <button class="secondary" id="explainBtn">Explain this codebase</button>
        </div>
        <div id="nodeDetails">
          <span class="close" id="detailsClose">&times;</span>
          <div id="detailsBody"></div>
        </div>
        <div id="summaryPanel">
          <span class="close" id="summaryClose">&times;</span>
          <h3>Codebase overview</h3>
          <div class="body" id="summaryBody"></div>
        </div>
        <div id="tooltip"></div>
        <div id="hint">drag background to pan &middot; scroll to zoom &middot; drag a node to move it &middot; click a node for details &middot; hover for a quick peek</div>
      </div>
    </section>

    <section id="view-ask" class="view wide">
      <div class="ask-layout">
        <aside class="chat-sidebar">
          <button class="chat-sidebar-new" id="newChatBtn"><span>+</span> New chat</button>
          <div class="chat-list" id="chatList"></div>
        </aside>
        <div class="card ask-main">
          <h2>Ask the graph</h2>
          <p class="hint">Natural-language impact analysis: mention a function name and it pulls its
            <b>real</b> callers + transitive blast radius straight from the CALLS graph (not a guess),
            plus hybrid TF-IDF + semantic search over docstrings for everything else, then sends it
            all to Groq's LLM for a concrete, cited answer. If the context isn't enough to answer
            confidently, it'll ask a clarifying question instead (marked
            <span class="chat-label" style="color:var(--accent)">clarifying</span>) - just reply in the
            box below to continue.
            You can also attach a screenshot (paperclip button, or paste with Ctrl+V) - Gemini reads
            the identifying text in it (page titles, labels, URLs) and feeds that into the same
            pipeline as a typed question.
            Requires <code class="inline">GROQ_API_KEY</code> (and <code class="inline">GEMINI_API_KEY</code>
            for image questions) to be set on the machine running <code class="inline">serve</code>.</p>
          <div id="askResult"><div class="chat-empty">Ask a question about the codebase to get started.</div></div>
          <div id="askImagePreview" style="display:none; margin-top:10px;"></div>
          <div class="row" style="margin-top:10px; align-items:flex-end">
            <button class="secondary" id="askAttachBtn" title="Attach a screenshot" style="flex-shrink:0">📎</button>
            <input type="file" id="askImageInput" accept="image/*" style="display:none">
            <textarea id="askInput" placeholder="What breaks if I change get_app_url? (Enter to send, Shift+Enter for a new line, Ctrl+V to paste a screenshot)" style="min-height:44px"></textarea>
            <button class="primary" id="askBtn">Send</button>
          </div>
        </div>
      </div>
    </section>

    <section id="view-cypher" class="view">
      <div class="card">
        <h2>Raw Cypher</h2>
        <textarea id="cypherInput">MATCH (f:Function) RETURN f.name LIMIT 10</textarea>
        <div class="row" style="margin-top:8px">
          <button class="primary" id="cypherBtn">Run</button>
        </div>
        <div id="cypherResult"></div>
      </div>
    </section>

  </main>
</div>
<script>
const NODE_COLORS = __NODE_COLORS_JSON__;
const EDGE_COLORS = __EDGE_COLORS_JSON__;
const KIND_LEGEND_HTML = __LEGEND_JSON__;
const INITIAL_DB_PATH = __DB_PATH_JSON__;

// ---------------------------------------------------------------- nav ----
const views = document.querySelectorAll('.view');
const navBtns = document.querySelectorAll('nav button');
let graphLoaded = false;
navBtns.forEach(btn => btn.addEventListener('click', () => {
  navBtns.forEach(b => b.classList.remove('active'));
  views.forEach(v => v.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('view-' + btn.dataset.view).classList.add('active');
  if (btn.dataset.view === 'graph' && !graphLoaded) { graphLoaded = true; loadGraph(); }
}));

// ------------------------------------------------------------- repos ----
const SKIP_DIRS = new Set(['.git', '__pycache__', 'node_modules', 'venv', '.venv', 'env', '.env',
  'dist', 'build', '.idea', '.vscode', 'site-packages', '.mypy_cache', '.pytest_cache']);
const CODE_EXT = /[.](py|php)$/i;
const MAX_FILE_BYTES = 2 * 1024 * 1024; // skip anything absurdly large (generated files, etc.)

async function loadRepoList() {
  try {
    const r = await api('/api/repos');
    const sel = document.getElementById('repoSelect');
    sel.innerHTML = r.repos.map(repo =>
      `<option value="${escapeHtml(repo.db_path)}" ${repo.db_path === r.active ? 'selected' : ''}>${escapeHtml(repo.name)}</option>`
    ).join('');
    if (r.repos.length) {
      document.getElementById('dbPathFallback').style.display = 'none';
      sel.style.display = '';
    }
  } catch (e) { /* registry not reachable yet - keep the static fallback text */ }
}
loadRepoList();

document.getElementById('repoSelect').addEventListener('change', async (ev) => {
  try {
    await api('/api/switch-repo', { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ db_path: ev.target.value }) });
    location.reload();
  } catch (e) { alert('Could not switch repo: ' + e.message); }
});

document.getElementById('addRepoBtn').addEventListener('click', () => {
  document.getElementById('repoFolderInput').click();
});
document.getElementById('repoFolderInput').addEventListener('change', async (ev) => {
  const fileList = Array.from(ev.target.files || []);
  ev.target.value = ''; // allow picking the same folder again later
  if (!fileList.length) return;

  const candidates = fileList.filter(f => {
    const rel = f.webkitRelativePath || f.name;
    const parts = rel.split('/');
    if (parts.some(p => SKIP_DIRS.has(p))) return false;
    return CODE_EXT.test(f.name) && f.size <= MAX_FILE_BYTES;
  });
  if (!candidates.length) {
    alert('No .py or .php files found in that folder.');
    return;
  }
  const repoName = (fileList[0].webkitRelativePath || fileList[0].name).split('/')[0];

  const overlay = document.getElementById('uploadOverlay');
  const msg = document.getElementById('uploadMsg');
  const sub = document.getElementById('uploadSub');
  overlay.classList.add('open');
  msg.textContent = `Reading ${candidates.length} file(s) from "${repoName}"...`;
  sub.textContent = '';

  try {
    const files = [];
    for (const f of candidates) {
      const rel = (f.webkitRelativePath || f.name).split('/').slice(1).join('/') || f.name;
      files.push({ path: rel, content: await f.text() });
    }
    msg.textContent = `Indexing "${repoName}"...`;
    sub.textContent = 'Parsing and building the graph - this can take a moment for larger repos.';
    const result = await api('/api/index-repo', { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ repo_name: repoName, files }) });
    msg.textContent = `Indexed ${result.nodes} nodes, ${result.edges} edges.`;
    sub.textContent = 'Reloading...';
    location.reload();
  } catch (e) {
    overlay.classList.remove('open');
    alert('Indexing failed: ' + e.message);
  }
});

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function renderTable(rows, columns) {
  if (!rows.length) return '<div class="empty">No results.</div>';
  const cols = columns || Object.keys(rows[0]);
  let html = '<table><thead><tr>' + cols.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr></thead><tbody>';
  for (const r of rows) {
    html += '<tr>' + cols.map(c => `<td>${escapeHtml(r[c] ?? '')}</td>`).join('') + '</tr>';
  }
  return html + '</tbody></table>';
}

// -------------------------------------------------------------- stats ----
async function loadStats() {
  try {
    const data = await api('/api/stats');
    document.getElementById('nodeStats').innerHTML = Object.entries(data.nodes)
      .map(([k, v]) => `<div class="stat-tile"><div class="n" style="color:${NODE_COLORS[k]||'#888'}">${v}</div><div class="k">${k}</div></div>`)
      .join('');
    document.getElementById('edgeStats').innerHTML = Object.entries(data.edges)
      .map(([k, v]) => `<div class="stat-tile"><div class="n" style="color:${EDGE_COLORS[k]||'#888'}">${v}</div><div class="k">${k}</div></div>`)
      .join('');
  } catch (e) {
    document.getElementById('nodeStats').innerHTML = `<div class="error">${escapeHtml(e.message)}</div>`;
  }
}
loadStats();

// ------------------------------------------------------------ explore ----
async function runExplore() {
  const name = document.getElementById('exploreInput').value.trim();
  if (!name) return;
  const callersEl = document.getElementById('callersResult');
  const impactEl = document.getElementById('impactResult');
  callersEl.innerHTML = '<div class="spinner">Loading...</div>';
  impactEl.innerHTML = '<div class="spinner">Loading...</div>';
  const note = (resolved) => resolved
    ? `<div class="hint">Couldn't find an exact match - showing results for <code class="inline">${escapeHtml(resolved)}</code>.</div>` : '';
  try {
    const c = await api('/api/callers?name=' + encodeURIComponent(name));
    callersEl.innerHTML = note(c.resolved_name) + renderTable(c.rows, ['caller', 'file', 'line']);
  } catch (e) { callersEl.innerHTML = `<div class="error">${escapeHtml(e.message)}</div>`; }
  try {
    const hops = document.getElementById('exploreHops').value;
    const i = await api('/api/impact?name=' + encodeURIComponent(name) + '&hops=' + hops);
    impactEl.innerHTML = note(i.resolved_name) + renderTable(i.rows, ['affected_function', 'file']);
  } catch (e) { impactEl.innerHTML = `<div class="error">${escapeHtml(e.message)}</div>`; }
}
document.getElementById('exploreBtn').addEventListener('click', runExplore);
document.getElementById('exploreInput').addEventListener('keydown', e => { if (e.key === 'Enter') runExplore(); });

// ---------------------------------------------------------------- ask ----
let askHistory = []; // [{question, answer, imageDataUrl?, contextQuestion?, seq?}, ...] - sent back for conversational context
let pendingImage = null; // {base64, mime, dataUrl} - attached but not yet sent
let currentChatId = null; // null = not-yet-persisted "new chat" - created lazily on first send

function activeRepoPath() {
  const sel = document.getElementById('repoSelect');
  return (sel && sel.value) || INITIAL_DB_PATH;
}

let chatListCache = []; // last-loaded [{id, title, ...}] - re-rendered whenever currentChatId changes

async function loadChatList() {
  try {
    const r = await api('/api/chats?repo=' + encodeURIComponent(activeRepoPath()));
    chatListCache = r.chats;
  } catch (e) { chatListCache = []; /* chat store not reachable yet - list just stays empty */ }
  renderChatList();
}

function renderChatList() {
  const el = document.getElementById('chatList');
  if (!chatListCache.length) {
    el.innerHTML = '<div class="chat-list-empty">No past chats yet.</div>';
    return;
  }
  el.innerHTML = chatListCache.map(c => `
    <div class="chat-list-item${c.id === currentChatId ? ' active' : ''}" data-id="${escapeHtml(c.id)}">
      <span class="title">${escapeHtml(c.title || '(untitled)')}</span>
      <span class="del" title="Delete chat" data-id="${escapeHtml(c.id)}">&times;</span>
    </div>`).join('');
  el.querySelectorAll('.chat-list-item').forEach(item => {
    item.addEventListener('click', () => loadChat(item.dataset.id));
  });
  el.querySelectorAll('.del').forEach(btn => {
    btn.addEventListener('click', (ev) => { ev.stopPropagation(); deleteChat(btn.dataset.id); });
  });
}

function startNewChat() {
  currentChatId = null;
  askHistory = [];
  clearPendingImage();
  renderAskHistory();
  renderChatList();
  document.getElementById('askInput').placeholder =
    'What breaks if I change get_app_url? (Enter to send, Shift+Enter for a new line, Ctrl+V to paste a screenshot)';
}

async function loadChat(chatId) {
  if (!chatId || chatId === currentChatId) return;
  try {
    const r = await api('/api/chat-messages?chat_id=' + encodeURIComponent(chatId));
    currentChatId = chatId;
    askHistory = r.messages.map(m => ({
      question: m.question, answer: m.answer, type: m.type, detail: m.detail,
      imageDataUrl: m.image_data_url || null,
      contextQuestion: m.context_question || m.question,
      seq: m.seq,
    }));
    clearPendingImage();
    renderAskHistory();
    renderChatList();
  } catch (e) { alert('Could not load chat: ' + e.message); }
}

async function deleteChat(chatId) {
  if (!confirm('Delete this chat?')) return;
  try {
    await api('/api/chats/delete', { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ chat_id: chatId }) });
    if (chatId === currentChatId) startNewChat();
    loadChatList();
  } catch (e) { alert('Could not delete chat: ' + e.message); }
}

document.getElementById('newChatBtn').addEventListener('click', startNewChat);
loadChatList();

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      resolve({ dataUrl, base64: dataUrl.split(',')[1], mime: file.type || 'image/png' });
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function setPendingImage(file) {
  if (!file || !file.type.startsWith('image/')) return;
  pendingImage = await fileToBase64(file);
  const preview = document.getElementById('askImagePreview');
  preview.style.display = 'flex';
  preview.innerHTML = `<img src="${pendingImage.dataUrl}"><span class="muted">Image attached</span>
    <span class="remove" id="askImageRemove">&times; remove</span>`;
  document.getElementById('askImageRemove').addEventListener('click', clearPendingImage);
  document.getElementById('askAttachBtn').classList.add('attached');
}

function clearPendingImage() {
  pendingImage = null;
  const preview = document.getElementById('askImagePreview');
  preview.style.display = 'none';
  preview.innerHTML = '';
  document.getElementById('askAttachBtn').classList.remove('attached');
}

document.getElementById('askAttachBtn').addEventListener('click', () => {
  document.getElementById('askImageInput').click();
});
document.getElementById('askImageInput').addEventListener('change', (ev) => {
  const file = ev.target.files && ev.target.files[0];
  ev.target.value = ''; // allow attaching the same file again later
  if (file) setPendingImage(file);
});
document.getElementById('askInput').addEventListener('paste', (ev) => {
  const item = Array.from(ev.clipboardData?.items || []).find(i => i.type.startsWith('image/'));
  if (item) { ev.preventDefault(); setPendingImage(item.getAsFile()); }
});

function renderAskHistory() {
  const out = document.getElementById('askResult');
  if (!askHistory.length) {
    out.innerHTML = '<div class="chat-empty">Ask a question about the codebase to get started.</div>';
    return;
  }
  out.innerHTML = askHistory.map((turn, i) => `
    <div class="chat-turn turn-q" data-idx="${i}"><div class="chat-q">
      ${turn.imageDataUrl ? `<img class="chat-q-img" src="${turn.imageDataUrl}">` : ''}
      ${escapeHtml(turn.question)}
      ${turn.seq !== undefined ? `<span class="msg-del" title="Delete this message" data-idx="${i}">&times;</span>` : ''}
      </div></div>
    <div class="chat-turn turn-a">
      ${turn.type === 'clarify' ? '<div class="chat-label clarify-label">clarifying</div>' : ''}
      <div class="chat-a${turn.type === 'clarify' ? ' clarify' : ''}">${escapeHtml(turn.answer)}</div>
      ${turn.detail ? `<details class="chat-detail"><summary></summary>
        <div class="chat-detail-body">${escapeHtml(turn.detail)}</div></details>` : ''}
    </div>`).join('');
  out.querySelectorAll('.msg-del').forEach(btn => {
    btn.addEventListener('click', (ev) => { ev.stopPropagation(); deleteMessage(parseInt(btn.dataset.idx, 10)); });
  });
  out.scrollTop = out.scrollHeight;
}

async function deleteMessage(idx) {
  const turn = askHistory[idx];
  if (!turn || turn.seq === undefined) return; // not yet persisted - nothing to delete server-side
  if (!confirm('Delete this message?')) return;
  try {
    await api('/api/chat-messages/delete', { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ chat_id: currentChatId, seq: turn.seq }) });
    askHistory.splice(idx, 1);
    renderAskHistory();
  } catch (e) { alert('Could not delete message: ' + e.message); }
}

async function sendAsk() {
  const input = document.getElementById('askInput');
  const q = input.value.trim();
  const btn = document.getElementById('askBtn');
  const out = document.getElementById('askResult');
  const image = pendingImage; // snapshot - cleared from the input right away, kept for this request
  if (!q && !image) return;
  btn.disabled = true;
  input.value = '';
  clearPendingImage();
  renderAskHistory();
  out.insertAdjacentHTML('beforeend',
    '<div class="chat-typing"><span></span><span></span><span></span></div>');
  out.scrollTop = out.scrollHeight;
  try {
    // history is sent back so a "clarify" turn's follow-up reply is
    // answered with the earlier question still in context - see
    // _call_groq's history handling in web.py. Each turn's *contextQuestion*
    // (not the raw, often placeholder `question`) is what's replayed here -
    // for an image turn that's the Gemini-extracted description, so an
    // earlier screenshot's content stays in context for follow-ups instead
    // of vanishing after that one turn.
    const historyForApi = askHistory.map(t => ({question: t.contextQuestion || t.question, answer: t.answer}));
    const payload = {question: q, history: historyForApi};
    if (image) { payload.image = image.base64; payload.image_mime = image.mime; }
    const r = await api('/api/ask', { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload) });
    const turn = {question: q || '(image)', answer: r.content, type: r.type, detail: r.detail || '',
      imageDataUrl: image ? image.dataUrl : null,
      contextQuestion: r.resolved_question || q || '(image)'};
    askHistory.push(turn);
    input.placeholder = r.type === 'clarify'
      ? 'Answer the question above, or ask something else...'
      : 'What breaks if I change get_app_url? (Enter to send, Shift+Enter for a new line, Ctrl+V to paste a screenshot)';
    renderAskHistory();
    await persistTurn(turn); // creates a chat on the first turn of a "New chat"
  } catch (e) {
    renderAskHistory();
    out.insertAdjacentHTML('beforeend', `<div class="error">${escapeHtml(e.message)}</div>`);
  } finally { btn.disabled = false; input.focus(); }
}

async function persistTurn(turn) {
  try {
    if (!currentChatId) {
      const chat = await api('/api/chats', { method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ repo_db_path: activeRepoPath() }) });
      currentChatId = chat.id;
    }
    const saved = await api('/api/chat-messages', { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        chat_id: currentChatId, question: turn.question, answer: turn.answer,
        type: turn.type || '', detail: turn.detail || '', image_data_url: turn.imageDataUrl || null,
        context_question: turn.contextQuestion || turn.question,
      }) });
    turn.seq = saved.seq; // lets deleteMessage target this turn once it's on disk
    renderAskHistory();
    loadChatList();
  } catch (e) { /* history still shown live even if persisting to disk failed */ }
}

document.getElementById('askBtn').addEventListener('click', sendAsk);
document.getElementById('askInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendAsk(); }
});

// ------------------------------------------------------------- cypher ----
document.getElementById('cypherBtn').addEventListener('click', async () => {
  const q = document.getElementById('cypherInput').value.trim();
  const btn = document.getElementById('cypherBtn');
  const out = document.getElementById('cypherResult');
  if (!q) return;
  btn.disabled = true;
  out.innerHTML = '<div class="spinner">Running...</div>';
  try {
    const r = await api('/api/cypher', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({query: q}) });
    out.innerHTML = renderTable(r.rows, r.columns);
  } catch (e) {
    out.innerHTML = `<div class="error">${escapeHtml(e.message)}</div>`;
  } finally { btn.disabled = false; }
});

// --------------------------------------------------------------- graph ----
let DATA = null;
const FOLDER_PALETTE = ['#4c6ef5', '#f76707', '#2f9e44', '#e64980', '#7048e8',
  '#1098ad', '#f59f00', '#e03131', '#0ca678', '#495057', '#5c7cfa', '#d6336c'];
function topFolder(file) {
  if (!file) return '(root)';
  const parts = file.replace(/\\\\/g, '/').split('/');
  return parts.length > 1 ? parts[0] : '(root)';
}
function buildFolderColors(nodes) {
  const folders = [...new Set(nodes.map(n => topFolder(n.file)))].sort();
  const map = {};
  folders.forEach((f, i) => { map[f] = FOLDER_PALETTE[i % FOLDER_PALETTE.length]; });
  return map;
}
let folderColors = {};
let colorMode = 'kind';

function colorFor(n) {
  if (colorMode === 'folder') return folderColors[topFolder(n.file)] || '#888';
  return NODE_COLORS[n.kind] || '#ccc';
}
function updateLegend() {
  const el = document.getElementById('legend');
  if (colorMode === 'folder') {
    el.innerHTML = Object.entries(folderColors)
      .map(([f, c]) => `<span><span class="dot" style="background:${c}"></span>${escapeHtml(f)}</span>`)
      .join('');
  } else {
    el.innerHTML = KIND_LEGEND_HTML;
  }
}

async function loadGraph() {
  const statEl = document.getElementById('graphStat');
  try {
    DATA = await api('/api/graph');
    statEl.textContent = `${DATA.nodes.length} nodes, ${DATA.edges.length} edges`;
    folderColors = buildFolderColors(DATA.nodes);
    startSim();
  } catch (e) {
    statEl.innerHTML = `<span class="error">${escapeHtml(e.message)}</span>`;
  }
}

document.getElementById('colorMode').addEventListener('change', (ev) => {
  colorMode = ev.target.value;
  updateLegend();
});

// -------------------------------------------------------- node details ----
function neighborsOf(node, edges) {
  const out = [], incoming = [];
  for (const e of edges) {
    const a = DATA.nodes[e.a], b = DATA.nodes[e.b];
    if (a === node) out.push({ node: b, kind: e.kind });
    else if (b === node) incoming.push({ node: a, kind: e.kind });
  }
  return { out, incoming };
}
function showNodeDetails(node, edges, focusFn) {
  document.getElementById('summaryPanel').classList.remove('open');
  const panel = document.getElementById('nodeDetails');
  const { out, incoming } = neighborsOf(node, edges);
  const listItem = (rec) => `<li data-goto="1">${escapeHtml(rec.kind)} → <b>${escapeHtml(rec.node.name)}</b></li>`;
  let html = `<h3>${escapeHtml(node.name)}</h3>`;
  html += `<div class="muted">${escapeHtml(node.kind)}</div>`;
  html += `<div class="file">${escapeHtml(node.file || '')}</div>`;
  if (node.docstring) html += `<div class="doc">${escapeHtml(node.docstring)}</div>`;
  if (out.length) {
    html += `<div class="neigh-heading">Depends on / contains (${out.length})</div><ul>` +
      out.map(r => listItem(r)).join('') + '</ul>';
  }
  if (incoming.length) {
    html += `<div class="neigh-heading">Used by (${incoming.length})</div><ul>` +
      incoming.map(r => listItem(r)).join('') + '</ul>';
  }
  panel.querySelector('#detailsBody').innerHTML = html;
  panel.classList.add('open');
  const allRecs = [...out, ...incoming];
  panel.querySelectorAll('li[data-goto]').forEach((li, i) => {
    li.addEventListener('click', () => focusFn(allRecs[i].node));
  });
}
document.getElementById('detailsClose').addEventListener('click', () => {
  document.getElementById('nodeDetails').classList.remove('open');
});

// -------------------------------------------------------------- explain ----
document.getElementById('explainBtn').addEventListener('click', async () => {
  document.getElementById('nodeDetails').classList.remove('open');
  const panel = document.getElementById('summaryPanel');
  const body = document.getElementById('summaryBody');
  const btn = document.getElementById('explainBtn');
  panel.classList.add('open');
  body.innerHTML = '<div class="spinner">Reading the codebase...</div>';
  btn.disabled = true;
  try {
    const r = await api('/api/summarize', { method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}' });
    body.innerHTML = escapeHtml(r.summary);
  } catch (e) {
    body.innerHTML = `<div class="error">${escapeHtml(e.message)}</div>`;
  } finally { btn.disabled = false; }
});
document.getElementById('summaryClose').addEventListener('click', () => {
  document.getElementById('summaryPanel').classList.remove('open');
});

function startSim() {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const tooltip = document.getElementById('tooltip');
  const wrap = document.getElementById('graphWrap');

  function resize() {
    canvas.width = wrap.clientWidth * devicePixelRatio;
    canvas.height = wrap.clientHeight * devicePixelRatio;
    canvas.style.width = wrap.clientWidth + 'px';
    canvas.style.height = wrap.clientHeight + 'px';
  }
  window.addEventListener('resize', resize);
  resize();

  const idIndex = new Map();
  DATA.nodes.forEach((n, i) => {
    idIndex.set(n.id, i);
    n.x = (Math.random() - 0.5) * 800;
    n.y = (Math.random() - 0.5) * 800;
    n.vx = 0; n.vy = 0;
  });
  const edges = DATA.edges
    .filter(e => idIndex.has(e.src) && idIndex.has(e.dst))
    .map(e => ({ a: idIndex.get(e.src), b: idIndex.get(e.dst), kind: e.kind }));

  const N = DATA.nodes.length;
  const REPULSION = 2600, SPRING_LEN = 90, SPRING_K = 0.02, CENTER_K = 0.003, DAMPING = 0.85;

  function step() {
    const nodes = DATA.nodes;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const a = nodes[i], b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx * dx + dy * dy || 0.01;
        let d = Math.sqrt(d2);
        let f = REPULSION / d2;
        dx /= d; dy /= d;
        a.vx += dx * f; a.vy += dy * f;
        b.vx -= dx * f; b.vy -= dy * f;
      }
    }
    for (const e of edges) {
      const a = nodes[e.a], b = nodes[e.b];
      let dx = b.x - a.x, dy = b.y - a.y;
      let d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      let f = SPRING_K * (d - SPRING_LEN);
      dx /= d; dy /= d;
      a.vx += dx * f; a.vy += dy * f;
      b.vx -= dx * f; b.vy -= dy * f;
    }
    for (const n of nodes) {
      n.vx -= n.x * CENTER_K;
      n.vy -= n.y * CENTER_K;
      n.vx *= DAMPING; n.vy *= DAMPING;
      if (!n.dragging) { n.x += n.vx; n.y += n.vy; }
    }
  }

  let camX = 0, camY = 0, zoom = 1, panning = false, panStart = null, draggedNode = null;
  let selectedNode = null;
  let mouseDownAt = null;
  function selectNode(n) {
    selectedNode = n;
    showNodeDetails(n, edges, selectNode);
  }
  function toScreen(x, y) { return [(x - camX) * zoom + canvas.width / 2, (y - camY) * zoom + canvas.height / 2]; }
  function toWorld(sx, sy) { return [(sx - canvas.width / 2) / zoom + camX, (sy - canvas.height / 2) / zoom + camY]; }

  const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text').trim();
  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const e of edges) {
      const a = DATA.nodes[e.a], b = DATA.nodes[e.b];
      const [ax, ay] = toScreen(a.x, a.y);
      const [bx, by] = toScreen(b.x, b.y);
      ctx.strokeStyle = (EDGE_COLORS[e.kind] || '#888') + '55';
      ctx.lineWidth = Math.max(1, zoom * devicePixelRatio * 0.8);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
    const r = Math.max(2, 5 * zoom * devicePixelRatio);
    // labels first so node dots stay legible on top of overlapping text
    if (zoom > 0.35) {
      ctx.font = `${Math.max(9, 10 * devicePixelRatio)}px -apple-system, Segoe UI, Roboto, sans-serif`;
      ctx.fillStyle = textColor;
      for (const n of DATA.nodes) {
        if (n.kind !== 'Module' && n.kind !== 'Class') continue;
        const [sx, sy] = toScreen(n.x, n.y);
        ctx.fillText(n.name, sx + r + 3, sy + 3);
      }
    }
    for (const n of DATA.nodes) {
      const [sx, sy] = toScreen(n.x, n.y);
      ctx.fillStyle = n === selectedNode ? '#fff' : colorFor(n);
      ctx.beginPath(); ctx.arc(sx, sy, n === selectedNode ? r * 1.5 : r, 0, Math.PI * 2); ctx.fill();
      if (n === selectedNode) {
        ctx.strokeStyle = colorFor(n); ctx.lineWidth = 2 * devicePixelRatio;
        ctx.stroke();
      }
    }
  }
  function loop() { step(); render(); requestAnimationFrame(loop); }
  loop();

  function nodeAt(sx, sy) {
    const r = Math.max(2, 5 * zoom * devicePixelRatio) + 3;
    for (let i = DATA.nodes.length - 1; i >= 0; i--) {
      const n = DATA.nodes[i];
      const [nx, ny] = toScreen(n.x, n.y);
      if ((nx - sx) ** 2 + (ny - sy) ** 2 <= r * r) return n;
    }
    return null;
  }
  canvas.addEventListener('mousedown', (ev) => {
    mouseDownAt = [ev.clientX, ev.clientY];
    const sx = ev.offsetX * devicePixelRatio, sy = ev.offsetY * devicePixelRatio;
    const n = nodeAt(sx, sy);
    if (n) { draggedNode = n; n.dragging = true; }
    else { panning = true; panStart = [ev.clientX, ev.clientY, camX, camY]; canvas.classList.add('dragging'); }
  });
  window.addEventListener('mousemove', (ev) => {
    if (draggedNode) {
      const [wx, wy] = toWorld(ev.offsetX * devicePixelRatio, ev.offsetY * devicePixelRatio);
      draggedNode.x = wx; draggedNode.y = wy; draggedNode.vx = 0; draggedNode.vy = 0;
    } else if (panning) {
      const [sx0, sy0, cx0, cy0] = panStart;
      camX = cx0 - (ev.clientX - sx0) / zoom;
      camY = cy0 - (ev.clientY - sy0) / zoom;
    } else {
      const sx = ev.offsetX * devicePixelRatio, sy = ev.offsetY * devicePixelRatio;
      const n = nodeAt(sx, sy);
      if (n) {
        tooltip.style.display = 'block';
        tooltip.style.left = (ev.clientX + 14) + 'px';
        tooltip.style.top = (ev.clientY + 14) + 'px';
        tooltip.innerHTML = `<b>${escapeHtml(n.kind)}</b> ${escapeHtml(n.name)}<br><span class="muted">${escapeHtml(n.file || '')}</span>`;
      } else { tooltip.style.display = 'none'; }
    }
  });
  window.addEventListener('mouseup', (ev) => {
    const moved = mouseDownAt && (Math.abs(ev.clientX - mouseDownAt[0]) + Math.abs(ev.clientY - mouseDownAt[1]) > 4);
    if (draggedNode) {
      if (!moved) selectNode(draggedNode);
      draggedNode.dragging = false;
    }
    draggedNode = null; panning = false; canvas.classList.remove('dragging');
  });
  canvas.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const factor = Math.exp(-ev.deltaY * 0.001);
    zoom = Math.min(6, Math.max(0.1, zoom * factor));
  }, { passive: false });
}
</script>
</body>
</html>
"""


def _render_page(db_path: str) -> str:
    legend_items = "".join(
        f'<span><span class="dot" style="background:{color}"></span>{kind}</span>'
        for kind, color in _NODE_COLORS.items()
    )
    html = _PAGE
    html = html.replace("__DB_PATH__", db_path)
    html = html.replace("__DB_PATH_JSON__", json.dumps(db_path))
    html = html.replace("__LEGEND__", legend_items)
    html = html.replace("__NODE_COLORS_JSON__", json.dumps(_NODE_COLORS))
    html = html.replace("__EDGE_COLORS_JSON__", json.dumps(_EDGE_COLORS))
    html = html.replace("__LEGEND_JSON__", json.dumps(legend_items))
    return html


def _query_rows(conn: kuzu.Connection, query: str, params: dict | None = None) -> tuple[list[str], list[dict]]:
    df = conn.execute(query, params or {}).get_as_df()
    columns = list(df.columns)
    rows = df.to_dict(orient="records")
    return columns, rows


def _resolve_function_name(conn: kuzu.Connection, query: str) -> tuple[str, str | None]:
    """Explore tab convenience: if `query` isn't an exact Function name,
    try to spot one mentioned in it (the same detection the Ask tab's NL
    flow uses - retrieval.find_mentioned_names), so typing a full question
    like "what happens if I don't pass the url in get media url" still
    resolves to get_media_url instead of just returning no rows.

    Returns (name_to_query, resolved_name) - resolved_name is None when
    `query` was already an exact match (nothing to tell the user about),
    otherwise it's the name that was matched, for the UI to show a note.
    """
    query = query.strip()
    names = list_function_names(conn)
    if query in names:
        return query, None
    matches = find_mentioned_names(query, names, limit=1)
    if matches:
        return matches[0], matches[0]
    return query, None


def _registry_path() -> Path:
    """Where the dashboard's list of known repos (name + db path) lives -
    a small JSON file in the current directory, so the repo dropdown
    remembers what you've indexed across `serve` restarts without needing
    a real database of its own."""
    return Path.cwd() / ".gitnexus_repos.json"


def _load_registry() -> list[dict]:
    p = _registry_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_registry(repos: list[dict]) -> None:
    _registry_path().write_text(json.dumps(repos, indent=2), encoding="utf-8")


def register_repo(name: str, db_path: str, source: str = "") -> None:
    """Add (or refresh) a repo in the registry the dashboard's repo
    dropdown reads from. Keyed by resolved db_path, so re-indexing the
    same repo updates its entry instead of duplicating it."""
    resolved = str(Path(db_path).resolve())
    repos = [r for r in _load_registry() if r["db_path"] != resolved]
    repos.append({"name": name, "db_path": resolved, "source": source, "added_at": time.time()})
    _save_registry(repos)


class ServerState:
    """Holds the single active Kuzu connection the dashboard is currently
    pointed at, and lets it be swapped for a different repo's database
    without restarting the HTTP server. Kuzu only allows one process to
    hold a given database open at a time (see the concurrency note in
    db.py), so switching means fully closing the old connection/database
    before opening the new one - there is deliberately never more than
    one open connection here."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db, self.conn = open_db(db_path, fresh=False)

    def switch(self, db_path: str) -> None:
        logger.info("ServerState.switch: closing %s, opening %s", self.db_path, db_path)
        self.conn.close()
        self.db.close()

        # Windows can lag releasing the just-closed db's OS-level file lock
        # by a beat, and if `db_path` happens to be the same file (or
        # anything else transiently touches it - AV scan, etc.) the very
        # next open can fail with "Could not set lock on file" even though
        # nothing is genuinely still holding it. Short retry instead of
        # failing the whole switch on a one-off timing race.
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                self.db, self.conn = open_db(db_path, fresh=False)
                self.db_path = db_path
                logger.info("ServerState.switch: opened %s (attempt %d)", db_path, attempt + 1)
                return
            except RuntimeError as exc:
                last_exc = exc
                logger.warning("ServerState.switch: attempt %d failed for %s: %s", attempt + 1, db_path, exc)
                time.sleep(0.5)
        logger.error("ServerState.switch: giving up on %s after retries", db_path)
        raise last_exc


_SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "env", ".env",
              "dist", "build", ".idea", ".vscode", "site-packages", ".mypy_cache", ".pytest_cache"}


def _safe_repo_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return (slug or "repo")[:60]


def index_uploaded_repo(repo_name: str, files: list[dict]) -> dict:
    """Materialize browser-uploaded files (the dashboard's "+ Add repo"
    folder picker - see the Graph tab JS) onto disk under
    .gitnexus_uploads/, then run the same parse_repo -> Kuzu pipeline the
    `index` CLI command uses, into a fresh per-repo database. Returns what
    got indexed plus the new db_path so the caller can register it and
    switch the dashboard's active connection to it."""
    slug = _safe_repo_slug(repo_name)
    base = Path.cwd() / ".gitnexus_uploads"
    src_dir = base / slug
    db_path = str(base / f"{slug}.db")

    if src_dir.exists():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for f in files:
        rel = f.get("path", "")
        parts = Path(rel).parts
        if not rel or any(p in _SKIP_DIRS for p in parts):
            continue
        dest = src_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f.get("content", ""), encoding="utf-8", errors="ignore")
        written += 1

    if written == 0:
        raise RuntimeError("No .py/.php files found in the selected folder.")

    graph = parse_repo(str(src_dir))
    db, conn = open_db(db_path, fresh=True)
    load_graph(conn, graph)
    ensure_vector_index(conn)
    conn.close()
    db.close()  # release the lock before ServerState reopens it

    register_repo(repo_name, db_path, source=repo_name)
    return {"db_path": db_path, "name": repo_name, "nodes": len(graph.nodes), "edges": len(graph.edges)}


def make_handler(state: ServerState) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to a ServerState. All requests are
    served from one thread (see run_server) since a Kuzu Connection isn't
    safe to share across threads - state.conn may be swapped out from
    under the handler by a /api/switch-repo or /api/index-repo call, but
    never while a request is mid-flight since everything's single-threaded."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A003 - quiet the default access log
            pass

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw or b"{}")

        def do_GET(self):  # noqa: N802 - stdlib method name
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self._send_html(_render_page(state.db_path))
                elif parsed.path == "/api/stats":
                    conn = state.conn
                    nodes = {k: _query_rows(conn, f"MATCH (n:{k}) RETURN count(n) AS c")[1][0]["c"] for k in NODE_KINDS}
                    edges = {k: _query_rows(conn, f"MATCH ()-[r:{k}]->() RETURN count(r) AS c")[1][0]["c"] for k in REL_KINDS}
                    self._send_json({"nodes": nodes, "edges": edges})
                elif parsed.path == "/api/graph":
                    self._send_json(export_graph_json(state.conn))
                elif parsed.path == "/api/callers":
                    name, resolved = _resolve_function_name(state.conn, (qs.get("name") or [""])[0])
                    self._send_json({"rows": get_callers(state.conn, name), "resolved_name": resolved})
                elif parsed.path == "/api/impact":
                    hops = int((qs.get("hops") or ["3"])[0])
                    name, resolved = _resolve_function_name(state.conn, (qs.get("name") or [""])[0])
                    self._send_json({"rows": get_impact(state.conn, name, max_hops=hops), "resolved_name": resolved})
                elif parsed.path == "/api/repos":
                    self._send_json({"repos": _load_registry(), "active": str(Path(state.db_path).resolve())})
                elif parsed.path == "/api/chats":
                    repo = (qs.get("repo") or [state.db_path])[0]
                    self._send_json({"chats": chat_store.list_chats(repo)})
                elif parsed.path == "/api/chat-messages":
                    chat_id = (qs.get("chat_id") or [""])[0]
                    if not chat_id:
                        raise RuntimeError("chat_id required")
                    self._send_json({"messages": chat_store.get_messages(chat_id)})
                else:
                    self._send_json({"error": "not found"}, status=404)
            except Exception as exc:  # noqa: BLE001 - surface any DB/query error to the UI
                logger.error("GET %s failed: %s", parsed.path, exc, exc_info=True)
                self._send_json({"error": str(exc)}, status=400)

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            try:
                body = self._read_json_body()
                if parsed.path == "/api/cypher":
                    query = body.get("query", "")
                    columns, rows = _query_rows(state.conn, query)
                    self._send_json({"columns": columns, "rows": rows})
                elif parsed.path == "/api/ask":
                    question = body.get("question", "")
                    image_b64 = body.get("image")
                    if image_b64:
                        logger.info("POST /api/ask (with image): %r image_mime=%s",
                                     question[:200], body.get("image_mime"))
                        self._send_json(_ask_with_image(
                            state.conn, question, image_b64, body.get("image_mime", "image/png"),
                            body.get("model"), body.get("history"), db_path=state.db_path))
                    else:
                        logger.info("POST /api/ask: %r", question[:200])
                        self._send_json(_ask(state.conn, question, body.get("model"),
                                              body.get("history"), db_path=state.db_path))
                elif parsed.path == "/api/summarize":
                    self._send_json({"summary": _summarize(state.conn, body.get("model"))})
                elif parsed.path == "/api/switch-repo":
                    db_path = body.get("db_path", "")
                    if not db_path:
                        raise RuntimeError("db_path required")
                    logger.info("Switching repo: %s -> %s", state.db_path, db_path)
                    state.switch(db_path)
                    self._send_json({"ok": True, "active": state.db_path})
                elif parsed.path == "/api/index-repo":
                    result = index_uploaded_repo(body.get("repo_name", "repo"), body.get("files", []))
                    state.switch(result["db_path"])
                    self._send_json(result)
                elif parsed.path == "/api/chats":
                    repo = body.get("repo_db_path") or state.db_path
                    self._send_json(chat_store.create_chat(repo, body.get("title", "")))
                elif parsed.path == "/api/chat-messages":
                    chat_id = body.get("chat_id", "")
                    if not chat_id:
                        raise RuntimeError("chat_id required")
                    question, answer = body.get("question", ""), body.get("answer", "")
                    chat = chat_store.get_chat(chat_id)
                    # Only the chat's first turn needs a generated title -
                    # every later turn already has one and just keeps it
                    # (see add_message), so skip the extra LLM round-trip.
                    title = (_generate_chat_title(question, answer, body.get("model"))
                             if chat and not chat["title"] else None)
                    self._send_json(chat_store.add_message(
                        chat_id, question, answer, body.get("type", ""), body.get("detail", ""),
                        body.get("image_data_url"), title=title,
                        context_question=body.get("context_question", "")))
                elif parsed.path == "/api/chats/delete":
                    chat_id = body.get("chat_id", "")
                    if not chat_id:
                        raise RuntimeError("chat_id required")
                    chat_store.delete_chat(chat_id)
                    self._send_json({"ok": True})
                elif parsed.path == "/api/chat-messages/delete":
                    chat_id = body.get("chat_id", "")
                    seq = body.get("seq")
                    if not chat_id or seq is None:
                        raise RuntimeError("chat_id and seq required")
                    chat_store.delete_message(chat_id, int(seq))
                    self._send_json({"ok": True})
                else:
                    self._send_json({"error": "not found"}, status=404)
            except Exception as exc:  # noqa: BLE001
                logger.error("POST %s failed: %s", parsed.path, exc, exc_info=True)
                self._send_json({"error": str(exc)}, status=400)

    return Handler


_DB_ISH_PATTERNS = [
    re.compile(r"\$this\s*->\s*db\b"),
    re.compile(r"->\s*table\s*\("),
    re.compile(r"->\s*query\s*\("),
    re.compile(r"\bModel\s*::"),
    re.compile(r"\bextends\s+\w*Model\b"),
    re.compile(r"App\\Models\\\w+"),
    re.compile(r"\bfind\s*\(|->\s*where\s*\(|->\s*get\s*\("),
    re.compile(r"\$_(GET|POST|REQUEST|SESSION)\b"),
]


def _impact_facts(conn: kuzu.Connection, question: str) -> str:
    """Impact-analysis questions ("what breaks if I change X?", "who calls
    Y?") need real CALLS-graph traversal, not docstring similarity - TF-IDF
    over docstrings has no idea who calls whom. This spots function names
    named in the question (retrieval.find_mentioned_names) and pulls their
    actual callers + transitive impact straight from the graph, formatted
    as a context block the LLM can quote instead of guessing. The impact
    hop depth defaults to 3 but honors an explicit number in the question
    (see retrieval.find_hop_count), e.g. "impact of X within 5 hops"."""
    names = find_mentioned_names(question, list_function_names(conn))
    if not names:
        return ""
    hops = find_hop_count(question)

    sections = []
    for name in names:
        callers = get_callers(conn, name)
        impact = get_impact(conn, name, max_hops=hops)
        lines = [f"Graph facts for function `{name}`:"]
        if callers:
            lines.append("  Direct callers: " + ", ".join(
                f"{c['caller']} ({c['file']}:{c['line']})" for c in callers))
        else:
            lines.append("  Direct callers: none found in the graph.")
        if impact:
            lines.append(f"  Transitive impact (up to {hops} hops) if changed: " + ", ".join(
                f"{i['affected_function']} ({i['file']})" for i in impact))
        else:
            lines.append("  Transitive impact if changed: none found - safe to change in isolation.")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _roles_facts(conn: kuzu.Connection) -> str:
    """Deterministic answer to "what roles/permissions are there" - every
    Role Concept extracted by rbac_extractor.py. Same data the `roles` CLI
    command reports, just reformatted as an LLM context block."""
    rows = list_concepts(conn, kind="Role")
    if not rows:
        return ""
    lines = ["Graph facts - Roles defined in this codebase:"]
    for r in rows:
        lines.append(f"  {r['code']} ({r['name']}) - app: {r['app']}, defined at {r['source_file']}:{r['lineno']}")
    return "\n".join(lines)


def _rules_facts(conn: kuzu.Connection, resolved_file: str | None) -> str:
    """Deterministic answer to "what rules gate this page/function" - RULE
    edges (role -> effect/condition) attached to nodes in `resolved_file`.
    Scoped by file so this doesn't dump every rule in the repo when the
    question is asking about one specific page."""
    if not resolved_file:
        return ("Note: the question seems to be about access rules, but no specific file/page was "
                "recognized in it - name the page/file to get rules scoped to it.")
    rows = get_rules_for_file(conn, resolved_file)
    if not rows:
        return f"Graph facts - no RULE edges found gating anything in {resolved_file}."
    lines = [f"Graph facts - Rules gating {resolved_file}:"]
    for r in rows:
        lines.append(f"  role {r['role']} -> {r['effect']} `{r['target']}` (line {r['lineno']}): {r['condition']}")
    return "\n".join(lines)


def _inherits_facts(conn: kuzu.Connection, question: str) -> str:
    """Deterministic answer to "what does X extend/inherit" *and* "what
    extends/subclasses X" - walks the INHERITS edge (already resolved
    cross-file at index time) both directions instead of guessing from
    docstrings. Both directions are always included rather than trying to
    parse which way the English question points ("what does X extend" vs
    "what extends X") - the LLM reads both facts and quotes whichever
    actually answers the question; this can't misfire the way a direction
    heuristic could."""
    _, class_rows = _query_rows(conn, "MATCH (c:Class) RETURN DISTINCT c.name AS name")
    class_names = [str(r["name"]) for r in class_rows]
    names = find_mentioned_names(question, class_names, limit=3)
    sections = []
    for name in names:
        ancestors = get_inherits(conn, name)
        if ancestors:
            sections.append(f"Graph facts - `{name}` inherits from: " +
                             ", ".join(f"{a['ancestor']} ({a['file']})" for a in ancestors))
        else:
            sections.append(f"Graph facts - `{name}` has no INHERITS edges found (doesn't extend anything indexed).")

        children = get_subclasses(conn, name)
        if children:
            sections.append(f"Graph facts - Classes that extend `{name}`: " +
                             ", ".join(f"{c['child']} ({c['file']})" for c in children))
        else:
            sections.append(f"Graph facts - No classes found that extend `{name}`.")
    return "\n\n".join(sections)


def _matrix_facts(conn: kuzu.Connection) -> str:
    """Deterministic answer to "give me the role x page permission matrix"
    - the same RULE data get_rules_for_file exposes per-page, aggregated
    across every page at once."""
    rows = get_role_page_matrix(conn)
    if not rows:
        return "Graph facts - no RULE edges found in this codebase (nothing to build a matrix from)."
    by_role: dict[str, list[str]] = {}
    for r in rows:
        by_role.setdefault(str(r["role"]), []).append(f"{r['file']} ({r['effect']})")
    lines = ["Graph facts - Role x page permission matrix (file: effect):"]
    for role, pages in sorted(by_role.items()):
        lines.append(f"  {role}: " + "; ".join(pages))
    return "\n".join(lines)


def _views_facts(conn: kuzu.Connection, resolved_file: str | None) -> str:
    """Deterministic (where resolved) or honest best-effort (where not)
    answer to "which view(s) does this page render" - RENDERS facts from
    view_extractor.py, scoped to `resolved_file`."""
    if not resolved_file:
        return ("Note: the question seems to be about view rendering, but no specific file/page "
                "was recognized in it - name the page/file to get facts scoped to it.")
    rows = get_views_rendered(conn, resolved_file)
    if not rows:
        return f"Graph facts - no view(...) calls found in {resolved_file}."
    lines = [f"Graph facts - Views rendered by {resolved_file}:"]
    for r in rows:
        if r["resolved"]:
            lines.append(f"  line {r['lineno']}: `{r['view_arg']}` -> {r['target_file']}")
        else:
            lines.append(f"  line {r['lineno']}: `{r['view_arg']}` (referenced but not found in the indexed graph)")
    return "\n".join(lines)


def _resolve_repo_root(db_path: str) -> Path | None:
    """Best-effort: find the source tree a db was indexed from, so
    _data_source_facts can read a page's raw source for DB-ish evidence.
    Only reliable for dashboard-uploaded repos (a fixed .gitnexus_uploads/
    <slug>.db <-> .gitnexus_uploads/<slug>/ convention - see
    index_uploaded_repo); CLI-indexed repos don't record their source root
    anywhere, so this returns None for those rather than guessing, and the
    caller degrades gracefully (skips the raw-source scan)."""
    p = Path(db_path)
    if p.parent.name == ".gitnexus_uploads" and p.suffix == ".db":
        candidate = p.parent / p.stem
        if candidate.is_dir():
            return candidate
    return None


def _data_source_facts(conn: kuzu.Connection, db_path: str, resolved_file: str | None) -> str:
    """Best-effort answer to "where does this page's data come from".
    There's no real data-lineage extractor (see README limitations - calls
    through an object property like $this->someModel->method() aren't
    resolved), so this surfaces what *is* known: the page's own IMPORTS
    (which Models/Helpers/Libraries it pulls in) plus a lightweight,
    non-persisted regex scan of the file's raw source for DB-ish lines
    (query builder calls, $this->db, $_GET/$_POST, ...), each with a line
    number so it reads as evidence, not a claimed-complete trace."""
    if not resolved_file:
        return ("Note: the question seems to be about where data comes from, but no specific "
                "file/page was recognized in it - name the page/file to get evidence scoped to it.")

    lines = [f"Evidence for data sources in {resolved_file} (best-effort, not a guaranteed-complete trace):"]

    _, imp_rows = _query_rows(
        conn,
        "MATCH (m:Module {file: $file})-[:IMPORTS]->(t) RETURN t.name AS name",
        {"file": resolved_file},
    )
    if imp_rows:
        lines.append("  Imports: " + ", ".join(str(r["name"]) for r in imp_rows))

    root = _resolve_repo_root(db_path)
    if root is not None:
        src = root / resolved_file
        if src.is_file():
            try:
                text = src.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
            hits = []
            for lineno, line in enumerate(text.splitlines(), start=1):
                if any(pat.search(line) for pat in _DB_ISH_PATTERNS):
                    hits.append(f"    line {lineno}: {line.strip()[:120]}")
                if len(hits) >= 20:
                    break
            if hits:
                lines.append("  DB-ish lines found in the file (not a resolved trace - just pattern matches):")
                lines.extend(hits)
    if len(lines) == 1:
        lines.append("  Nothing found - no imports and no DB-ish patterns detected in this file.")
    return "\n".join(lines)


_MAX_SOURCE_CHARS = 6000  # keep a single method's body from blowing up the LLM context


def _extract_method_source(text: str, method_name: str, class_name: str | None = None) -> str | None:
    """Pull one method/function's body (signature through matching closing
    brace) out of raw PHP source by brace-counting from the `function
    name(` occurrence - not a real parse (doesn't know about braces inside
    strings/comments), but good enough for well-formed source, same
    best-effort spirit as the DB-ish regex scan above. Function nodes for
    PHP don't carry real line numbers in this graph (only the Python side
    does - see parser.py vs php_parser.py), so re-scanning the source
    directly, on demand, is the only way to get an exact method body here.

    If `class_name` is given, only considers a `function name(` that comes
    after that class's own `class Name` declaration - guards against
    matching a same-named method on a different class earlier in the file.
    """
    search_from = 0
    if class_name:
        cls_match = re.search(rf"\bclass\s+{re.escape(class_name)}\b", text)
        if cls_match:
            search_from = cls_match.end()

    sig_match = re.search(rf"\bfunction\s+{re.escape(method_name)}\s*\(", text[search_from:])
    if not sig_match:
        return None
    start = search_from + sig_match.start()

    brace_start = text.find("{", start)
    if brace_start == -1:
        return None
    depth = 0
    end = None
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None

    snippet = text[start:end]
    if len(snippet) > _MAX_SOURCE_CHARS:
        snippet = snippet[:_MAX_SOURCE_CHARS] + "\n... (truncated)"
    return snippet


def _source_code_facts(db_path: str, resolved_file: str, class_name: str | None, method_name: str) -> str:
    """The actual source code of the method that handles a resolved
    URL/route - not just "which file/method handles this" but its real
    logic, so the LLM can explain what it does line-by-line instead of
    only pointing at it. Best-effort: needs the source tree on disk
    (_resolve_repo_root - only reliable for dashboard-uploaded repos) and
    a brace-matching extraction (_extract_method_source), so absence here
    just means the block is silently omitted, not an error."""
    root = _resolve_repo_root(db_path)
    if root is None:
        return ""
    src = root / resolved_file
    if not src.is_file():
        return ""
    try:
        text = src.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    snippet = _extract_method_source(text, method_name, class_name)
    if not snippet:
        return ""
    label = f"{class_name}::{method_name}" if class_name else method_name
    return f"Actual source code of `{label}` in {resolved_file}:\n```php\n{snippet}\n```"


def _file_source_facts(db_path: str, file: str, label: str) -> str:
    """The actual (whole-file) source of a view/template file, same
    verbatim-source idiom as _source_code_facts but for a file rather than
    one method - a view has no single "method" to brace-match, and
    templates are usually small enough to include whole. Used by
    _route_facts_and_file so a resolved URL's *view* gets its real markup
    alongside its controller method, not just the "renders -> file" fact
    _views_facts already gives; that fact alone can't answer "what does
    this page actually show/do" the way the real template can."""
    root = _resolve_repo_root(db_path)
    if root is None:
        return ""
    src = root / file
    if not src.is_file():
        return ""
    try:
        text = src.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if len(text) > _MAX_SOURCE_CHARS:
        text = text[:_MAX_SOURCE_CHARS] + "\n... (truncated)"
    return f"Actual source code of {label} `{file}`:\n```php\n{text}\n```"


def _route_facts_and_file(conn: kuzu.Connection, question: str) -> tuple[str, str | None, dict | None]:
    """If the question names a URL/path (retrieval.extract_url_or_path),
    try resolving it to a real Controller file (db.resolve_url) - a URL is
    the most natural way a real user names a page, and once resolved it
    feeds `resolved_file` into the same Rules/Data-source/Views blocks that
    file-name scoping already uses. Returns (route facts block, resolved
    file or None, resolve_url()'s full result dict or None) - the third
    element carries {class, method} on to _source_code_facts so a resolved
    URL question gets the actual method body, not just "this file handles
    it". Block is empty and file/result are None when nothing URL-shaped
    is in the question at all, so the caller falls through to
    find_mentioned_file."""
    url = extract_url_or_path(question)
    if not url:
        return "", None, None
    result = resolve_url(conn, url)
    if result.get("resolved"):
        block = (f"Graph facts - `{url}` matches route `{result['matched_pattern']}` -> "
                 f"{result['class']}::{result['method']} ({result['file']}).")
        return block, result["file"], result
    if "matched_pattern" in result:
        block = (f"Note: `{url}` matches route pattern `{result['matched_pattern']}`, but its target "
                 f"`{result['target']}` wasn't found in the indexed graph (not guessing which class this is).")
        return block, None, None
    return f"Note: `{url}` doesn't match any indexed route.", None, None


def graph_facts_for_question(conn: kuzu.Connection, question: str, db_path: str = "") -> str:
    """Router: classify which fact type(s) the question is asking for
    (retrieval.classify_intents) and which page/file it names - either a
    URL (db.resolve_url, tried first since that's the most natural way a
    real user names a page) or a file name (retrieval.find_mentioned_file)
    - then pull only the relevant deterministic graph fact blocks - so
    "what roles exist" gets real Concept data, "what rules gate this page"
    gets real RULE data scoped to that page, etc., instead of everything
    falling back to docstring similarity the way it used to before this
    router existed."""
    intents = classify_intents(question)
    route_block, resolved_file, route_result = _route_facts_and_file(conn, question)
    logger.info("graph_facts_for_question: route_result=%r db_path=%r", route_result, db_path)
    if resolved_file is None and not route_block:
        resolved_file = find_mentioned_file(question, list_module_files(conn))

    blocks = [_impact_facts(conn, question)]  # always relevant, no intent gate (existing behavior)
    if route_block:
        blocks.append(route_block)
    if route_result:
        # A resolved URL/route is exactly "the user cares about this one
        # method's actual logic" - pull its real source, not just the
        # fact that it's the handler (see the earlier "what handles
        # /icab/configuration" case - that alone isn't an explanation).
        src_block = _source_code_facts(db_path, route_result["file"], route_result["class"], route_result["method"])
        logger.info("graph_facts_for_question: source block chars=%d", len(src_block))
        blocks.append(src_block)

        # Whatever view(s) that controller method renders are just as much
        # "the page" as the controller itself - pull their real markup too
        # (not gated behind the "views" intent keyword: a bare URL like
        # "/icab/configuration, what does it do" should still get the
        # actual template, not just the controller). Best-effort: only
        # fires for the views RENDERS actually resolved to a real Module.
        for view_row in get_views_rendered(conn, route_result["file"]):
            target = view_row.get("target_file")
            if not target:
                continue
            view_block = _file_source_facts(db_path, target, f"view `{view_row['view_arg']}`")
            if view_block:
                blocks.append(view_block)
    if "roles" in intents:
        blocks.append(_roles_facts(conn))
    if "rules" in intents:
        blocks.append(_rules_facts(conn, resolved_file))
    if "inherits" in intents:
        blocks.append(_inherits_facts(conn, question))
    if "data_source" in intents:
        blocks.append(_data_source_facts(conn, db_path, resolved_file))
    if "matrix" in intents:
        blocks.append(_matrix_facts(conn))
    if "views" in intents:
        blocks.append(_views_facts(conn, resolved_file))

    return "\n\n".join(b for b in blocks if b)


def _call_groq(system_prompt: str, user_prompt: str, model: str | None,
                history: list[dict] | None = None, json_mode: bool = False) -> str:
    """history entries are {"question", "answer"} pairs - "answer" here
    means whatever the assistant said last turn, including a clarifying
    question (see _ask), so the model sees its own earlier question and
    the user's reply to it as normal conversation turns.

    json_mode=True asks Groq to constrain output to valid JSON (used by
    _ask for its {"type": "answer"|"clarify", "content": ...} shape,
    rather than trying to detect a clarifying question by parsing free
    text). _summarize doesn't need this - plain prose is fine there."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("Groq call aborted: GROQ_API_KEY not set on the server process.")
        raise RuntimeError("GROQ_API_KEY not set on the server process - export it and restart `serve`.")

    import urllib.error
    import urllib.request

    messages = [{"role": "system", "content": system_prompt}]
    for turn in (history or []):
        q, a = str(turn.get("question", "")), str(turn.get("answer", ""))
        if q and a:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_prompt})

    resolved_model = model or "openai/gpt-oss-120b"
    payload = {
        "model": resolved_model,
        "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode(),
        # Explicit User-Agent: Groq's Cloudflare front-end 403s the default
        # "Python-urllib/x.y" UA as a suspected bot request.
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "gitnexus-py/1.0",
        },
    )
    logger.info("Groq request: model=%s prompt_chars=%d history_turns=%d",
                resolved_model, len(user_prompt), len(history or []))
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # str(exc) alone is just "HTTP Error 404: Not Found" - useless for
        # diagnosis. Groq's actual error body (e.g. {"error": {"message":
        # "model `x` does not exist or you do not have access to it", ...}})
        # is on exc.read() and only readable once, here - log it AND fold
        # it into the raised message so it also reaches the browser instead
        # of vanishing into the generic exception text.
        body = exc.read().decode("utf-8", errors="replace")
        logger.error("Groq call failed: HTTP %s model=%s body=%s", exc.code, resolved_model, body)
        raise RuntimeError(f"Groq API error {exc.code} (model={resolved_model}): {body}") from exc
    except urllib.error.URLError as exc:
        logger.error("Groq call failed: network error model=%s reason=%s", resolved_model, exc.reason)
        raise RuntimeError(f"Could not reach Groq's API ({exc.reason}) - check network/proxy access "
                            f"to api.groq.com.") from exc
    logger.info("Groq response ok: model=%s", resolved_model)
    return result["choices"][0]["message"]["content"]


# Verified against Gemini's live ListModels endpoint (the same discipline
# that caught Groq's silently-deprecated default model earlier this
# session) - gemini-2.0-flash, the first guess, isn't even in the catalog
# anymore. gemini-2.5-flash is: stable (not a -preview/-latest alias),
# supports generateContent + image input. Override via GEMINI_MODEL if
# this goes stale too.
def _generate_chat_title(question: str, answer: str, model: str | None = None) -> str:
    """LLM-summarized chat title (ChatGPT-style) for a new chat's first
    turn, so the sidebar shows a short readable label instead of the raw
    (possibly long, or image-only) question text - see /api/chat-messages.
    Returns "" on any failure (Groq unreachable, GROQ_API_KEY unset, ...);
    add_message() then falls back to plain truncation - a nice title is
    never worth failing the actual answer over."""
    try:
        raw = _call_groq(
            "You title chat conversations. Given the user's question (and the assistant's "
            "answer, for context), reply with ONLY a short title for this chat: 3-6 words, "
            "no surrounding quotes, no trailing punctuation, no leading article "
            "('The'/'A'/'An').",
            f"Question: {question or '(user sent an image)'}\n\nAnswer: {(answer or '')[:500]}",
            model,
        )
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("Chat title generation failed, falling back to truncation: %s", exc)
        return ""
    return raw.strip().strip('"').strip("'")[:80]


_GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"


def _call_gemini_describe(image_b64: str, image_mime: str) -> str:
    """Gemini's one job in this pipeline: turn a screenshot into plain text
    describing anything that could identify a page/file in the codebase -
    visible titles, labels, form fields, error text, a URL if visible. That
    text then flows into the *existing* text-only Ask pipeline (_ask) as if
    the user had typed it - Gemini never touches graph facts or the final
    answer, Groq (already working, already tested) still owns those. Groq's
    own catalog has no vision-capable model (checked against its live
    /models endpoint this session), which is the actual reason this exists
    at all."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("Gemini call aborted: GEMINI_API_KEY not set on the server process.")
        raise RuntimeError("GEMINI_API_KEY not set on the server process - add it to .env and restart `serve`.")

    import urllib.error
    import urllib.request

    model = os.environ.get("GEMINI_MODEL") or _GEMINI_DEFAULT_MODEL
    prompt = (
        "This is a screenshot from a web application. Describe, in plain text, only what could "
        "help identify which page/file in a codebase this is: any visible page title, headings, "
        "form field labels, button text, table column names, error messages, and the URL if one "
        "is visible in the browser chrome. Do not describe colors, layout, or general design - "
        "just the identifying text content, as a short paragraph."
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": image_mime, "data": image_b64}},
            ]
        }]
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "gitnexus-py/1.0"},
    )
    logger.info("Gemini request: model=%s image_mime=%s image_b64_chars=%d", model, image_mime, len(image_b64))
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # Same lesson as _call_groq: str(exc) drops the actual response
        # body, which is exactly where Gemini explains a bad model id, a
        # disabled API, or a bad key - log and surface it instead of the
        # generic "HTTP Error NNN" text.
        body = exc.read().decode("utf-8", errors="replace")
        logger.error("Gemini call failed: HTTP %s model=%s body=%s", exc.code, model, body)
        raise RuntimeError(f"Gemini API error {exc.code} (model={model}): {body}") from exc
    except urllib.error.URLError as exc:
        logger.error("Gemini call failed: network error model=%s reason=%s", model, exc.reason)
        raise RuntimeError(f"Could not reach Gemini's API ({exc.reason}) - check network/proxy access "
                            f"to generativelanguage.googleapis.com.") from exc

    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        logger.error("Gemini call failed: unexpected response shape: %s", result)
        raise RuntimeError(f"Gemini returned an unexpected response shape: {result}") from exc
    logger.info("Gemini response ok: model=%s extracted_chars=%d", model, len(text))
    return text


def _top_folder(file_path: str) -> str:
    """First path component of a repo-relative file path - used to group
    modules into rough "business domains" for the codebase summary, since
    that's usually the nearest thing to a domain boundary a plain file
    layout gives you (e.g. `accom_external/...` vs `backend/...`)."""
    norm = (file_path or "").replace("\\", "/").strip("/")
    return norm.split("/", 1)[0] if norm else "(root)"


def _summarize(conn: kuzu.Connection, model: str | None) -> str:
    """A plain-English, business-level overview of the codebase: what each
    top-level folder seems to be for, based on its modules' names and
    docstrings. This is what the Graph tab's "Explain this codebase" button
    calls - the force-directed view alone shows *structure* (who imports/
    calls whom) but not *purpose*, which is what an LLM summary is for."""
    _, rows = _query_rows(conn, "MATCH (m:Module) RETURN m.name, m.file, m.docstring")
    by_folder: dict[str, list[str]] = {}
    for row in rows:
        name = str(row.get("m.name"))
        file = str(row.get("m.file") or "")
        doc = str(row.get("m.docstring") or "").strip().splitlines()[0] if row.get("m.docstring") else ""
        folder = _top_folder(file)
        line = f"- {name} ({file})" + (f": {doc[:140]}" if doc else "")
        by_folder.setdefault(folder, []).append(line)

    if not by_folder:
        return "No modules found - index a repo first with `gitnexus_py index`."

    sections = [f"## {folder}/\n" + "\n".join(lines) for folder, lines in sorted(by_folder.items())]
    listing = "\n\n".join(sections)

    system_prompt = (
        "You are a senior engineer writing a short onboarding brief for a new teammate. "
        "You're given every module in a codebase, grouped by top-level folder, each with its "
        "docstring's first line (if any). Infer what each folder/business-area is responsible "
        "for and write a concise plain-English overview: 1) a 2-3 sentence high-level summary "
        "of what this codebase does, 2) a short bullet per top-level folder naming its apparent "
        "responsibility. Do not just restate file names - infer purpose. No preamble."
    )
    return _call_groq(system_prompt, f"Modules by folder:\n\n{listing}", model)


def build_context(conn: kuzu.Connection, question: str, top_k: int = 15) -> list[str]:
    """Hybrid docstring retrieval for the Ask tab: TF-IDF (retrieval.
    rank_context - exact vocabulary overlap, free, always available) plus
    semantic vector search (db.vector_search - Kuzu's HNSW index over
    fastembed embeddings, catches paraphrases with zero shared words, e.g.
    "formula quantity" matching a docstring about "qty of formula line
    items"). Merged and de-duplicated so a question surfaces whatever
    either method would find - vector_search degrades to [] (not an
    error) when fastembed/the index isn't available, so this always works
    at least as well as TF-IDF alone."""
    docs: list[tuple[str, str]] = []
    for kind in ("Module", "Class", "Function"):
        _, rows = _query_rows(conn, f"MATCH (n:{kind}) RETURN n.name, n.file, n.docstring")
        for row in rows:
            name, file, doc = str(row.get("n.name")), str(row.get("n.file") or ""), str(row.get("n.docstring") or "")
            label = f"{kind} `{name}` in {file}: {doc[:150]}"
            docs.append((label, f"{name} {doc}"))
    tfidf_hits = rank_context(question, docs, top_k=top_k)

    semantic_hits = [
        f"{r['kind']} `{r['name']}` in {r['file']}: {r['docstring'][:150]}"
        for r in vector_search(conn, question, top_k=top_k)
    ]

    seen: set[str] = set()
    merged: list[str] = []
    for label in tfidf_hits + semantic_hits:
        if label not in seen:
            seen.add(label)
            merged.append(label)
    return merged[:top_k]


def _ask(conn: kuzu.Connection, question: str, model: str | None,
         history: list[dict] | None = None, db_path: str = "") -> dict:
    if not question.strip():
        return {"error": "empty question"}

    context_rows = build_context(conn, question, top_k=15)
    graph_facts = graph_facts_for_question(conn, question, db_path=db_path)

    parts = []
    if graph_facts:
        parts.append(graph_facts)
    if context_rows:
        parts.append("Related module/class/function docs:\n" + "\n".join(context_rows))
    context = "\n\n".join(parts) or "(no relevant graph context found for this question)"

    system_prompt = (
        "You are a code assistant. Answer using only the provided graph context. When "
        "'Graph facts' are given, they come from real graph data (Roles/RULEs extracted "
        "deterministically from the code, a real call-graph/INHERITS traversal, or a resolved "
        "Route/Render match) - exact, not guessed - trust them over anything else for roles, "
        "access rules, who-calls-what, change-impact, inheritance (both 'X extends' and 'what "
        "extends X'), URL-to-page routing, and view-rendering questions. A block starting with "
        "'Evidence for data sources' is different: it's a best-effort regex scan of the page's "
        "raw source, not a resolved trace, so state it with appropriate hedging (e.g. \"the "
        "file references X, Y\") rather than as a certain, complete answer. A block titled "
        "'Actual source code of ...' is the real method body or view/template file, verbatim - "
        "when it's present, use it to explain what the code actually *does* (its real logic or "
        "markup, step by step, in plain English) instead of only naming which method/file "
        "handles the request; that's the whole point of including it. When both a controller "
        "method and a view file are given for the same resolved URL, weave them together into "
        "one end-to-end explanation (what the controller computes, what the view then renders "
        "with it) rather than describing them as two disconnected facts. A 'Note:' block "
        "(e.g. a route pattern matched but its target class wasn't found in the graph) should "
        "also be stated as the honest limitation it is, not smoothed over. Earlier turns in "
        "this conversation may be included for context - prefer the fresh graph context above "
        "over older answers if they conflict.\n\n"
        "Respond with a JSON object: "
        "{\"type\": \"answer\" or \"clarify\", \"content\": \"...\", \"detail\": \"...\"}.\n"
        "- Use \"answer\" when the context gives you enough to respond confidently. Never "
        "answer with just a function/class/module name and a pointer to \"go check it\" - "
        "that is not an answer. Read what that function/class actually does in the "
        "provided context (its logic, conditions, fields, return value) and state the "
        "concrete, definite result: the actual rule, value, limit, or effect the question "
        "asked for.\n"
        "  - \"content\": the direct, definite answer alone - one to two sentences, no "
        "hedging, no citations, no code. Just the plain-English fact the user asked for, "
        "as if a colleague who already read the code told you the answer out loud.\n"
        "  - \"detail\": the supporting evidence for that answer - the specific "
        "function/class/module name(s) the fact came from, the relevant logic/conditions/"
        "fields from the context that back it up, and any caveats or edge cases. This is "
        "what proves \"content\" is correct; leave it out of \"content\" itself.\n"
        "- Use \"clarify\" when the context is genuinely insufficient or the question is "
        "ambiguous - e.g. a term in the question could refer to multiple unrelated things "
        "found in the context, or nothing relevant turned up at all. Put the single, "
        "specific, targeted clarifying question in \"content\" and leave \"detail\" as an "
        "empty string. Do not use \"clarify\" just because the honest answer is \"no\" or "
        "\"not found\" - only when a follow-up question from the user would actually let "
        "you find a better answer."
    )
    raw = _call_groq(system_prompt, f"Codebase graph context:\n{context}\n\nQuestion: {question}",
                      model, history, json_mode=True)
    return _parse_ask_response(raw)


def _ask_with_image(conn: kuzu.Connection, question: str, image_b64: str, image_mime: str,
                     model: str | None, history: list[dict] | None = None, db_path: str = "") -> dict:
    """Image variant of _ask: Gemini turns the screenshot into text
    (_call_gemini_describe), then that text is handed to the *unmodified*
    _ask() exactly like a typed question - every existing capability (URL
    resolution, roles/rules/routes/views/actual-source-code blocks, Groq
    answer synthesis) applies to it for free. If the user also typed
    something, it's kept alongside the extracted description rather than
    replaced by it."""
    extracted = _call_gemini_describe(image_b64, image_mime)
    synthesized = f"{question.strip()}\n\n[Image content]: {extracted}" if question.strip() else extracted
    result = _ask(conn, synthesized, model, history, db_path=db_path)
    # Carry the Gemini-extracted description back to the caller so it can be
    # stored as this turn's *context* question (see chat_store.add_message's
    # context_question column and sendAsk's use of it in web.py's HTML/JS).
    # Without this, a follow-up turn's history only replays the original
    # raw question (often just "(image)") and the image's content is lost
    # from the conversation after this one turn.
    result["resolved_question"] = synthesized
    return result


def _parse_ask_response(raw: str) -> dict:
    """Groq's JSON mode (see _call_groq's json_mode) should always return
    valid {"type", "content", "detail"}, but this is a defensive fallback in
    case a response ever doesn't comply - degrades to treating the raw text
    as a plain answer instead of erroring out."""
    try:
        parsed = json.loads(raw)
        kind = parsed.get("type") if parsed.get("type") in ("answer", "clarify") else "answer"
        content = str(parsed.get("content", "")).strip() or raw
        detail = str(parsed.get("detail", "")).strip()
        return {"type": kind, "content": content, "detail": detail}
    except (json.JSONDecodeError, AttributeError):
        return {"type": "answer", "content": raw, "detail": ""}


def run_server(db_path: str, host: str = "127.0.0.1", port: int = 8765,
               open_browser: bool = True) -> None:
    """Start the dashboard server and block until interrupted (Ctrl+C).
    Opens db_path itself (via ServerState) rather than taking an
    already-open connection, since the repo dropdown / "+ Add repo" flow
    needs to be able to close and reopen a different database later."""
    state = ServerState(db_path)
    register_repo(Path(db_path).stem, db_path, source=db_path)
    handler_cls = make_handler(state)

    class Server(socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    httpd = Server((host, port), handler_cls)
    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    log_path = Path.cwd() / "gitnexus_py.log"
    print(f"gitnexus-py dashboard running at {url} (Ctrl+C to stop) - logging to {log_path}")
    logger.info("serve started: db=%s host=%s port=%s pid=%s", db_path, host, port, os.getpid())
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("serve stopped")
        httpd.server_close()
