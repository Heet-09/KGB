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

from .db import (
    export_graph_json, get_callers, get_impact, list_function_names,
    load_graph, open_db, ensure_vector_index, vector_search, NODE_KINDS, REL_KINDS,
)
from .parser import parse_repo
from .retrieval import rank_context, find_mentioned_names, find_hop_count

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
    max-height: calc(100vh - 340px); min-height: 80px; overflow-y: auto; padding-right: 4px; }
  #askResult:empty { display: none; }
  .chat-turn { display: flex; flex-direction: column; gap: 6px; max-width: 80%; }
  .chat-turn.turn-q { align-self: flex-end; align-items: flex-end; }
  .chat-turn.turn-a { align-self: flex-start; align-items: flex-start; }
  .chat-q { background: var(--accent); color: #fff; border-radius: 12px 12px 2px 12px;
    padding: 9px 14px; white-space: pre-wrap; line-height: 1.5; }
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

    <section id="view-ask" class="view">
      <div class="card">
        <h2>Ask the graph</h2>
        <p class="hint">Natural-language impact analysis: mention a function name and it pulls its
          <b>real</b> callers + transitive blast radius straight from the CALLS graph (not a guess),
          plus hybrid TF-IDF + semantic search over docstrings for everything else, then sends it
          all to Groq's LLM for a concrete, cited answer. If the context isn't enough to answer
          confidently, it'll ask a clarifying question instead (marked
          <span class="chat-label" style="color:var(--accent)">clarifying</span>) - just reply in the
          box below to continue.
          Requires <code class="inline">GROQ_API_KEY</code> to be set on the machine running
          <code class="inline">serve</code>.</p>
        <div id="askResult"><div class="chat-empty">Ask a question about the codebase to get started.</div></div>
        <div class="row" style="margin-top:10px; align-items:flex-end">
          <textarea id="askInput" placeholder="What breaks if I change get_app_url? (Enter to send, Shift+Enter for a new line)" style="min-height:44px"></textarea>
          <button class="primary" id="askBtn">Send</button>
          <button class="chat-clear" id="askClearBtn">Clear history</button>
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
let askHistory = []; // [{question, answer}, ...] - sent back for conversational context

function renderAskHistory() {
  const out = document.getElementById('askResult');
  if (!askHistory.length) {
    out.innerHTML = '<div class="chat-empty">Ask a question about the codebase to get started.</div>';
    return;
  }
  out.innerHTML = askHistory.map(turn => `
    <div class="chat-turn turn-q"><div class="chat-q">${escapeHtml(turn.question)}</div></div>
    <div class="chat-turn turn-a">
      ${turn.type === 'clarify' ? '<div class="chat-label clarify-label">clarifying</div>' : ''}
      <div class="chat-a${turn.type === 'clarify' ? ' clarify' : ''}">${escapeHtml(turn.answer)}</div>
      ${turn.detail ? `<details class="chat-detail"><summary></summary>
        <div class="chat-detail-body">${escapeHtml(turn.detail)}</div></details>` : ''}
    </div>`).join('');
  out.scrollTop = out.scrollHeight;
}

async function sendAsk() {
  const input = document.getElementById('askInput');
  const q = input.value.trim();
  const btn = document.getElementById('askBtn');
  const out = document.getElementById('askResult');
  if (!q) return;
  btn.disabled = true;
  input.value = '';
  renderAskHistory();
  out.insertAdjacentHTML('beforeend',
    '<div class="chat-typing"><span></span><span></span><span></span></div>');
  out.scrollTop = out.scrollHeight;
  try {
    // history is sent back so a "clarify" turn's follow-up reply is
    // answered with the earlier question still in context - see
    // _call_groq's history handling in web.py.
    const r = await api('/api/ask', { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question: q, history: askHistory}) });
    askHistory.push({question: q, answer: r.content, type: r.type, detail: r.detail || ''});
    input.placeholder = r.type === 'clarify'
      ? 'Answer the question above, or ask something else...'
      : 'What breaks if I change get_app_url? (Enter to send, Shift+Enter for a new line)';
    renderAskHistory();
  } catch (e) {
    renderAskHistory();
    out.insertAdjacentHTML('beforeend', `<div class="error">${escapeHtml(e.message)}</div>`);
  } finally { btn.disabled = false; input.focus(); }
}

document.getElementById('askBtn').addEventListener('click', sendAsk);
document.getElementById('askInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendAsk(); }
});

document.getElementById('askClearBtn').addEventListener('click', () => {
  askHistory = [];
  renderAskHistory();
  document.getElementById('askInput').placeholder =
    'What breaks if I change get_app_url? (Enter to send, Shift+Enter for a new line)';
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
        self.conn.close()
        self.db.close()
        self.db_path = db_path
        self.db, self.conn = open_db(db_path, fresh=False)


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
                else:
                    self._send_json({"error": "not found"}, status=404)
            except Exception as exc:  # noqa: BLE001 - surface any DB/query error to the UI
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
                    self._send_json(_ask(state.conn, body.get("question", ""), body.get("model"), body.get("history")))
                elif parsed.path == "/api/summarize":
                    self._send_json({"summary": _summarize(state.conn, body.get("model"))})
                elif parsed.path == "/api/switch-repo":
                    db_path = body.get("db_path", "")
                    if not db_path:
                        raise RuntimeError("db_path required")
                    state.switch(db_path)
                    self._send_json({"ok": True, "active": state.db_path})
                elif parsed.path == "/api/index-repo":
                    result = index_uploaded_repo(body.get("repo_name", "repo"), body.get("files", []))
                    state.switch(result["db_path"])
                    self._send_json(result)
                else:
                    self._send_json({"error": "not found"}, status=404)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)

    return Handler


def graph_facts_for_question(conn: kuzu.Connection, question: str) -> str:
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
        raise RuntimeError("GROQ_API_KEY not set on the server process - export it and restart `serve`.")

    import urllib.request

    messages = [{"role": "system", "content": system_prompt}]
    for turn in (history or []):
        q, a = str(turn.get("question", "")), str(turn.get("answer", ""))
        if q and a:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model or "llama-3.3-70b-versatile",
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
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


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
         history: list[dict] | None = None) -> dict:
    if not question.strip():
        return {"error": "empty question"}

    context_rows = build_context(conn, question, top_k=15)
    graph_facts = graph_facts_for_question(conn, question)

    parts = []
    if graph_facts:
        parts.append(graph_facts)
    if context_rows:
        parts.append("Related module/class/function docs:\n" + "\n".join(context_rows))
    context = "\n\n".join(parts) or "(no relevant graph context found for this question)"

    system_prompt = (
        "You are a code assistant. Answer using only the provided graph context. When "
        "'Graph facts' are given, they come from a real call-graph traversal (exact, not "
        "guessed) - trust them over anything else for who-calls-what / change-impact "
        "questions. Earlier turns in this conversation may be included for context - prefer "
        "the fresh graph context above over older answers if they conflict.\n\n"
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
    print(f"gitnexus-py dashboard running at {url} (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
