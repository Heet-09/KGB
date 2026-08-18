"""
retrieval.py
------------
Local, dependency-free context retrieval for the `ask` command's Graph RAG.

Replaces naive substring/keyword matching with a small TF-IDF + cosine
similarity ranker over graph node text (name + docstring). This is still
not real embeddings-based retrieval (see README limitations), but it's a
step up: it scores relevance instead of doing boolean "is this word a
substring of that field" matching, and it ranks results instead of just
taking the first N in whatever order Kuzu returned them.

Everything here is stdlib-only (math, re, collections) - no vector DB, no
embedding model download, consistent with the project's zero-server,
zero-extra-infra approach.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "and", "or", "for", "with", "this", "that", "what", "does", "how",
    "do", "did", "it", "its", "be", "been", "as", "at", "by", "from",
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2 and t.lower() not in _STOPWORDS]


def rank_context(question: str, docs: list[tuple[str, str]], top_k: int = 15) -> list[str]:
    """Rank `docs` by TF-IDF cosine similarity to `question`.

    docs: list of (label, text) pairs - `label` is what gets returned (the
    human-readable context line to send to the LLM), `text` is what gets
    scored against the question (e.g. name + docstring).

    Returns the top_k labels, most relevant first, dropping anything with
    zero overlap with the question's vocabulary.
    """
    q_tokens = _tokenize(question)
    if not q_tokens or not docs:
        return []

    doc_tokens = [_tokenize(text) for _, text in docs]

    # document frequency, for IDF weighting
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))
    n_docs = len(docs)

    def vectorize(tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        return {t: count * math.log((n_docs + 1) / (df.get(t, 0) + 1)) for t, count in tf.items()}

    q_vec = vectorize(q_tokens)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    scored: list[tuple[float, str]] = []
    for (label, _text), tokens in zip(docs, doc_tokens):
        d_vec = vectorize(tokens)
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        dot = sum(weight * d_vec.get(term, 0.0) for term, weight in q_vec.items())
        score = dot / (q_norm * d_norm)
        if score > 0:
            scored.append((score, label))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [label for _, label in scored[:top_k]]


_HOPS_RE = re.compile(r"(\d+)\s+(hops?|levels?|degrees?)\b", re.IGNORECASE)


def find_hop_count(question: str, default: int = 3, max_hops: int = 10) -> int:
    """Pick up an explicit depth from phrasing like "impact within 5 hops"
    or "3 levels deep" for the Ask tab's impact analysis, falling back to
    `default` (matches get_impact's own default) when the question doesn't
    name one. Clamped the same way db.get_impact clamps it, so a wild
    number in the question can't blow up the query."""
    m = _HOPS_RE.search(question)
    if not m:
        return default
    return max(1, min(int(m.group(1)), max_hops))


def find_mentioned_names(question: str, known_names: list[str], limit: int = 5) -> list[str]:
    """Pick out which of `known_names` (e.g. every Function name in the
    graph) the question is actually naming, so natural-language questions
    like "what breaks if I change get_app_url?" or "who calls parse_repo?"
    can be answered with real graph facts (see cli.ask / web._ask) instead
    of relying on docstring similarity alone.

    Matches case-insensitively against whole tokens in the question (an
    identifier like `get_app_url` is one token by _TOKEN_RE, so this finds
    it even though the tokenizer doesn't split snake_case). Also checks
    space-joined tokens against snake_case names split on "_", so a
    question phrased as "get app url" still matches `get_app_url`.

    All three strategies below run unconditionally (not "first non-empty
    wins") and results are sorted longest-name-first: a short, generic
    name like `get` (a common Django view method name) would otherwise
    win as an exact-token match before a much more specific but
    space-phrased `get_media_url` ever got a chance to be considered.
    """
    if not known_names:
        return []
    q_tokens = {t.lower() for t in _TOKEN_RE.findall(question)}
    q_tokens |= {t.lower() for t in re.split(r"[^a-zA-Z0-9]+", question) if t}
    q_lower = question.lower()

    exact = [name for name in known_names if name.lower() in q_tokens]
    # snake_case name written as separate words in the question, e.g.
    # "...in get media url" naming get_media_url - split the name on "_"
    # and check the space-joined phrase appears in the question.
    spaced = [name for name in known_names
              if "_" in name and name.lower().replace("_", " ") in q_lower]
    # loose substring containment, for multi-word phrasing neither above
    # strategy caught
    substr = [name for name in known_names if len(name) > 3 and name.lower() in q_lower]

    matches = sorted({*exact, *spaced, *substr}, key=len, reverse=True)
    # de-dupe, keep first-seen (longest-first) order, cap the count
    seen: set[str] = set()
    out = []
    for name in matches:
        if name not in seen:
            seen.add(name)
            out.append(name)
        if len(out) >= limit:
            break
    return out


def find_mentioned_file(question: str, known_files: list[str]) -> str | None:
    """Which known Module `.file` path (if any) the question is naming, so
    "page"-scoped questions ("what rules gate this page", "where does the
    data on X come from") can be resolved to a real file instead of only
    ever answering globally. Same longest-match spirit as
    find_mentioned_names, but matched against file basenames (a question
    names a file, not a repo-relative path).

    Tries, in order: exact basename mention (e.g. "IcabPermissionChecker.php"
    or just "IcabPermissionChecker"), then a "page name" guess - the
    basename with its extension stripped and CamelCase/underscores split
    into words (e.g. "budget_controller.php" -> "budget controller"),
    matched as a phrase in the question. Returns the single longest/most
    specific match, or None if nothing in the question names a known file.
    """
    if not known_files:
        return None
    q_lower = question.lower()

    def _basename(path: str) -> str:
        return re.split(r"[\\/]", path)[-1]

    def _stem(basename: str) -> str:
        return re.sub(r"\.(php|py)$", "", basename, flags=re.IGNORECASE)

    def _words(stem: str) -> str:
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem)  # camelCase -> spaced
        spaced = re.sub(r"[_\-]+", " ", spaced)
        return spaced.strip().lower()

    candidates: list[tuple[int, str]] = []  # (match length, file)
    for f in known_files:
        base = _basename(f)
        stem = _stem(base)
        if base.lower() in q_lower:
            candidates.append((len(base), f))
            continue
        if len(stem) > 3 and stem.lower() in q_lower:
            candidates.append((len(stem), f))
            continue
        words = _words(stem)
        if words and len(words) > 3 and words in q_lower:
            candidates.append((len(words), f))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


_INTENT_KEYWORDS = {
    "roles": re.compile(r"\brole|\bpermission|\baccess level|\bwho can\b", re.IGNORECASE),
    "rules": re.compile(r"\brule|\bgate|\ballow|\bdeny|\bvalidat|\bcondition|\brestrict", re.IGNORECASE),
    "data_source": re.compile(
        r"\bdata\b.*\bcome|\bwhere.*data|\bdatabase|\btable\b|\bsql\b|\bmodel\b|\bsource\b|\bfetch|\bquery",
        re.IGNORECASE,
    ),
    "inherits": re.compile(r"\bextend|\binherit|\bparent class|\bsubclass|\bbase class", re.IGNORECASE),
    "impact": re.compile(r"\bcall|\bbreak|\bimpact|\bdepend|\baffect", re.IGNORECASE),
    "matrix": re.compile(r"\bmatrix|\bwhich pages?\b|\bevery page\b|\ball pages\b", re.IGNORECASE),
    "views": re.compile(r"\bview\b|\brender|\btemplate|\bpartial", re.IGNORECASE),
}


_URL_RE = re.compile(r"https?://\S+|(?<![\w.])/[\w][\w./-]*")


def extract_url_or_path(question: str) -> str | None:
    """Pull the first URL or path-shaped token (e.g. `/icab/configuration`
    or `https://app/icab/configuration?x=1`) out of a question, so
    graph_facts_for_question can try resolving it as a page URL
    (db.resolve_url) before falling back to find_mentioned_file. A leading
    `/` with no preceding word character avoids matching things like
    "N/A" or a fraction as a path."""
    m = _URL_RE.search(question)
    return m.group(0) if m else None


def classify_intents(question: str) -> set[str]:
    """Which fact types a question is plausibly asking for - a cheap
    keyword router (no extra LLM call, consistent with this project's
    zero-extra-infra style) that lets graph_facts_for_question pull only
    the graph facts relevant to the question instead of either everything
    or nothing. Multiple intents can fire on one question (e.g. "what
    rules and roles gate this page" is both "rules" and "roles")."""
    return {intent for intent, pattern in _INTENT_KEYWORDS.items() if pattern.search(question)}
