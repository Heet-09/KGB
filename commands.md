# gitnexus-py — CLI Commands

All commands are run as `python -m gitnexus_py.cli <command> ...`.

## Install

```bash
pip install -r requirements.txt
```

## 1. Index a repo (builds/overwrites the graph DB)

```bash
python -m gitnexus_py.cli index /path/to/repo --db ./graph.db
```

### 1b. Incremental index (skip DB wipe, only touch changed files)

```bash
python -m gitnexus_py.cli index /path/to/repo --db ./graph.db --incremental
```

## 2. Stats — see what got indexed

```bash
python -m gitnexus_py.cli stats --db ./graph.db
```

## 3. Callers — who calls a given function?

```bash
python -m gitnexus_py.cli callers my_function --db ./graph.db
```

## 4. Impact — blast radius of changing a function (before refactoring)

```bash
python -m gitnexus_py.cli impact my_function --db ./graph.db
```

## 5. Cypher — run raw Cypher queries directly

```bash
python -m gitnexus_py.cli cypher "MATCH (f:Function) RETURN f.name LIMIT 10" --db ./graph.db
```

### 5b. Serve — full web dashboard (Stats / Explore / Graph / Ask / Cypher)

```bash
python -m gitnexus_py.cli serve --db ./graph.db
```

Opens automatically at http://127.0.0.1:8765

## 6. Visualize — export a self-contained HTML graph view

```bash
python -m gitnexus_py.cli visualize --db ./graph.db --out ./graph.html
```

### 6b. Obsidian — export the graph as an Obsidian vault

```bash
python -m gitnexus_py.cli obsidian --db ./graph.db --out ./obsidian_vault
```

## 7. Ask — natural-language question over the graph (needs GROQ_API_KEY)

```bash
export GROQ_API_KEY=your_key_here
python -m gitnexus_py.cli ask "what does the parser module do?" --db ./graph.db
```

---

**Full command list:** `index`, `stats`, `cypher`, `callers`, `impact`, `ask`, `visualize`, `obsidian`, `serve` (defined in `cli.py`).
