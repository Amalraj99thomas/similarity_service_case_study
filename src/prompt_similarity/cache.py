"""In-memory vector cache for zero-latency similarity search.

Loads prompt IDs, content strings, and embedding vectors from SQLite into
NumPy arrays.  Rebuilt at startup and after every embedding generation call.
"""

import numpy as np

from prompt_similarity.db import get_db


# ── Module-level cache state ───────────────────────────────────────────────────
_vec_cache: np.ndarray | None = None   # (n, DIM) L2-normalised
_id_cache: list[str] = []              # row i → prompt_id
_content_cache: list[str] = []         # row i → raw content (for previews)


def rebuild() -> None:
    """Reload prompt IDs, vectors, and content from SQLite into memory.

    Called once at startup and after every ``/api/embeddings/generate`` call.
    """
    global _vec_cache, _id_cache, _content_cache

    with get_db() as conn:
        rows = conn.execute(
            "SELECT prompt_id, content, embedding FROM prompts WHERE embedding IS NOT NULL"
        ).fetchall()

    if not rows:
        _id_cache = []
        _content_cache = []
        _vec_cache = None
        return

    _id_cache = [r["prompt_id"] for r in rows]
    _content_cache = [r["content"] for r in rows]
    _vec_cache = np.stack([
        np.frombuffer(r["embedding"], dtype="float32") for r in rows
    ])


def get_vectors() -> np.ndarray | None:
    """Return the cached (n, DIM) embedding matrix, or ``None`` if empty."""
    return _vec_cache


def get_ids() -> list[str]:
    """Return the list of prompt IDs corresponding to cache rows."""
    return _id_cache


def get_contents() -> list[str]:
    """Return the list of raw content strings corresponding to cache rows."""
    return _content_cache


def content_preview(idx: int, max_len: int = 120) -> str:
    """Return a truncated content preview for the prompt at cache index *idx*."""
    text = _content_cache[idx]
    return (text[:max_len] + "...") if len(text) > max_len else text
