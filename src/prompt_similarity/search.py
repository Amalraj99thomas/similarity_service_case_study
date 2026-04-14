"""Brute-force cosine similarity search on the in-memory vector cache."""

import numpy as np

from prompt_similarity import cache


def search(query_vec: np.ndarray, k: int, threshold: float) -> list[tuple[int, float]]:
    """Find the top-*k* cache indices whose cosine similarity to *query_vec* is ≥ *threshold*.

    Returns a list of ``(cache_index, score)`` tuples sorted by descending
    similarity.  Works entirely on the in-memory cache — no SQLite reads.
    """
    vec_cache = cache.get_vectors()
    id_cache = cache.get_ids()

    if vec_cache is None or len(id_cache) == 0:
        return []

    scores = vec_cache @ query_vec
    ranked = np.argsort(scores)[::-1]

    return [
        (int(i), float(scores[i]))
        for i in ranked
        if scores[i] >= threshold
    ][:k]
