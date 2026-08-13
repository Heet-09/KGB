"""
embeddings.py
-------------
Thin wrapper around fastembed for turning text (a Module/Class/Function's
name + docstring) into vectors for Kuzu's native HNSW vector index (see
db.ensure_vector_index / db.vector_search).

Why this exists: retrieval.rank_context (TF-IDF) only matches questions
against docstrings that share actual vocabulary - a question like "formula
quantity" won't match a docstring that says "qty of formula line items"
because there's zero literal word overlap. Embeddings match by meaning
instead, which is what makes that kind of paraphrase work. The two are
combined (see web.build_context) rather than one replacing the other -
TF-IDF is exact and free, embeddings are semantic and catch what TF-IDF
misses.

fastembed (https://github.com/qdrant/fastembed) was chosen over
sentence-transformers specifically for its size: it runs on ONNX Runtime
instead of pulling in all of PyTorch, so the extra install is ~150-250MB
instead of 1-2GB, while using the same underlying embedding model
(BAAI/bge-small-en-v1.5) and comparable accuracy. Everything still runs
locally - no embeddings API, no extra network calls beyond the one-time
model download on first use.
"""

from __future__ import annotations

EMBED_DIM = 384  # BAAI/bge-small-en-v1.5's output size - fastembed's default model

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding()  # defaults to BAAI/bge-small-en-v1.5
    return _model


def is_available() -> bool:
    """Whether fastembed is installed. Callers should treat semantic
    search as an optional enhancement and fall back to TF-IDF alone when
    this is False, rather than hard-failing indexing/Ask."""
    try:
        import fastembed  # noqa: F401
        return True
    except ImportError:
        return False


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts in one model call - much faster than
    embedding one string at a time when indexing hundreds of nodes."""
    if not texts:
        return []
    model = _get_model()
    # fastembed expects non-empty strings; blank docstrings still need a
    # real vector so every node can be indexed uniformly.
    safe_texts = [t if t.strip() else "(no description)" for t in texts]
    return [vec.tolist() for vec in model.embed(safe_texts)]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
