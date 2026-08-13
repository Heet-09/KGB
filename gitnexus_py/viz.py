"""
viz.py
------
Renders a graph (as plain node/edge dicts, see db.export_graph_json) into a
single self-contained HTML file: inline CSS + vanilla JS, a canvas-based
force-directed layout, no CDN script tags and no separate server process to
run. Keeps the same zero-server ethos as the rest of the project - you just
open the file in a browser.
"""

from __future__ import annotations

import json

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

_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>gitnexus-py graph</title>
<style>
  :root { --bg: #0f1115; --panel: #1a1d24; --text: #e8e8e8; --muted: #9aa0aa; --border: #2a2e37; }
  @media (prefers-color-scheme: light) {
    :root { --bg: #f7f7f8; --panel: #ffffff; --text: #1a1a1a; --muted: #666; --border: #e0e0e0; }
  }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font: 13px/1.4 -apple-system, Segoe UI, Roboto, sans-serif; height: 100%; overflow: hidden; }
  #canvas { display: block; width: 100vw; height: 100vh; cursor: grab; }
  #canvas.dragging { cursor: grabbing; }
  #panel { position: fixed; top: 12px; left: 12px; background: var(--panel);
    border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px;
    max-width: 320px; box-shadow: 0 4px 16px rgba(0,0,0,.25); }
  #panel h1 { font-size: 13px; margin: 0 0 6px; }
  #panel .stat { color: var(--muted); }
  .legend { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
  .legend span { display: inline-flex; align-items: center; gap: 4px; }
  .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  #tooltip { position: fixed; pointer-events: none; background: var(--panel);
    border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px;
    display: none; max-width: 360px; box-shadow: 0 4px 16px rgba(0,0,0,.3); z-index: 10; }
  #tooltip b { color: var(--text); }
  #tooltip .muted { color: var(--muted); }
  #hint { position: fixed; bottom: 10px; left: 12px; color: var(--muted); font-size: 11px; }
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div id="panel">
  <h1>gitnexus-py graph</h1>
  <div class="stat">__NODE_COUNT__ nodes, __EDGE_COUNT__ edges</div>
  <div class="legend">__LEGEND__</div>
</div>
<div id="tooltip"></div>
<div id="hint">drag background to pan · scroll to zoom · drag a node to move it · hover for details</div>
<script>
const DATA = __DATA_JSON__;
const NODE_COLORS = __NODE_COLORS_JSON__;
const EDGE_COLORS = __EDGE_COLORS_JSON__;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');

function resize() {
  canvas.width = window.innerWidth * devicePixelRatio;
  canvas.height = window.innerHeight * devicePixelRatio;
  canvas.style.width = window.innerWidth + 'px';
  canvas.style.height = window.innerHeight + 'px';
}
window.addEventListener('resize', resize);
resize();

// --- build simulation state ---------------------------------------------
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
const REPULSION = 2600;
const SPRING_LEN = 90;
const SPRING_K = 0.02;
const CENTER_K = 0.003;
const DAMPING = 0.85;

function step() {
  const nodes = DATA.nodes;
  // repulsion (all pairs - fine up to a few hundred nodes)
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
  // spring attraction along edges
  for (const e of edges) {
    const a = nodes[e.a], b = nodes[e.b];
    let dx = b.x - a.x, dy = b.y - a.y;
    let d = Math.sqrt(dx * dx + dy * dy) || 0.01;
    let f = SPRING_K * (d - SPRING_LEN);
    dx /= d; dy /= d;
    a.vx += dx * f; a.vy += dy * f;
    b.vx -= dx * f; b.vy -= dy * f;
  }
  // gentle pull to center
  for (const n of nodes) {
    n.vx -= n.x * CENTER_K;
    n.vy -= n.y * CENTER_K;
    n.vx *= DAMPING; n.vy *= DAMPING;
    if (!n.dragging) { n.x += n.vx; n.y += n.vy; }
  }
}

// --- camera (pan/zoom) ----------------------------------------------------
let camX = 0, camY = 0, zoom = 1;
let panning = false, panStart = null;
let draggedNode = null;

function toScreen(x, y) {
  return [
    (x - camX) * zoom + canvas.width / 2,
    (y - camY) * zoom + canvas.height / 2,
  ];
}
function toWorld(sx, sy) {
  return [
    (sx - canvas.width / 2) / zoom + camX,
    (sy - canvas.height / 2) / zoom + camY,
  ];
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const bg = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim();

  // edges
  for (const e of edges) {
    const a = DATA.nodes[e.a], b = DATA.nodes[e.b];
    const [ax, ay] = toScreen(a.x, a.y);
    const [bx, by] = toScreen(b.x, b.y);
    ctx.strokeStyle = (EDGE_COLORS[e.kind] || '#888') + '55';
    ctx.lineWidth = Math.max(1, zoom * devicePixelRatio * 0.8);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
  // nodes
  const r = Math.max(2, 5 * zoom * devicePixelRatio);
  for (const n of DATA.nodes) {
    const [sx, sy] = toScreen(n.x, n.y);
    ctx.fillStyle = NODE_COLORS[n.kind] || '#ccc';
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function loop() {
  step();
  render();
  requestAnimationFrame(loop);
}
loop();

// --- interaction ------------------------------------------------------
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
  const sx = ev.offsetX * devicePixelRatio, sy = ev.offsetY * devicePixelRatio;
  const n = nodeAt(sx, sy);
  if (n) {
    draggedNode = n; n.dragging = true;
  } else {
    panning = true; panStart = [ev.clientX, ev.clientY, camX, camY];
    canvas.classList.add('dragging');
  }
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
      tooltip.innerHTML = `<b>${n.kind}</b> ${n.name}<br><span class="muted">${n.file || ''}</span>`;
    } else {
      tooltip.style.display = 'none';
    }
  }
});
window.addEventListener('mouseup', () => {
  if (draggedNode) draggedNode.dragging = false;
  draggedNode = null; panning = false;
  canvas.classList.remove('dragging');
});
canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const factor = Math.exp(-ev.deltaY * 0.001);
  zoom = Math.min(6, Math.max(0.1, zoom * factor));
}, { passive: false });
</script>
</body>
</html>
"""


def render_html(data: dict) -> str:
    """Render `data` (see db.export_graph_json) into a self-contained HTML
    page with an inline force-directed canvas visualization."""
    legend_items = "".join(
        f'<span><span class="dot" style="background:{color}"></span>{kind}</span>'
        for kind, color in _NODE_COLORS.items()
    )
    html = _TEMPLATE
    html = html.replace("__NODE_COUNT__", str(len(data["nodes"])))
    html = html.replace("__EDGE_COUNT__", str(len(data["edges"])))
    html = html.replace("__LEGEND__", legend_items)
    # json.dumps output can't contain a literal "</script>" sequence, or it
    # would prematurely close the inline <script> tag when embedded.
    html = html.replace("__DATA_JSON__", json.dumps(data).replace("</", "<\\/"))
    html = html.replace("__NODE_COLORS_JSON__", json.dumps(_NODE_COLORS))
    html = html.replace("__EDGE_COLORS_JSON__", json.dumps(_EDGE_COLORS))
    return html
