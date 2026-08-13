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
