"""Tests for prompt_similarity.search — brute-force cosine search."""

import numpy as np
import pytest

from prompt_similarity import cache
from prompt_similarity.search import search


@pytest.fixture(autouse=True)
def _mock_cache(monkeypatch):
    """Inject a small synthetic vector cache for all tests in this module."""
    # 4 unit vectors in 3-D — easy to reason about similarities
    vecs = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.436, 0.0],   # cos(~26°) ≈ 0.9 with vec 0
        [0.0, 1.0, 0.0],     # orthogonal to vec 0
        [0.0, 0.0, 1.0],     # orthogonal to vec 0
    ], dtype="float32")
    # L2-normalise rows
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    ids = ["p0", "p1", "p2", "p3"]
    contents = ["prompt zero", "prompt one", "prompt two", "prompt three"]

    monkeypatch.setattr(cache, "_vec_cache", vecs)
    monkeypatch.setattr(cache, "_id_cache", ids)
    monkeypatch.setattr(cache, "_content_cache", contents)


class TestSearch:
    """Tests for the brute-force cosine search function."""

    def test_returns_matches_above_threshold(self):
        query = np.array([1.0, 0.0, 0.0], dtype="float32")
        hits = search(query, k=10, threshold=0.8)
        hit_ids = [cache.get_ids()[i] for i, _ in hits]
        assert "p0" in hit_ids
        assert "p1" in hit_ids

    def test_excludes_below_threshold(self):
        query = np.array([1.0, 0.0, 0.0], dtype="float32")
        hits = search(query, k=10, threshold=0.95)
        hit_ids = [cache.get_ids()[i] for i, _ in hits]
        assert "p0" in hit_ids
        assert "p2" not in hit_ids
        assert "p3" not in hit_ids

    def test_respects_k_limit(self):
        query = np.array([1.0, 0.0, 0.0], dtype="float32")
        hits = search(query, k=1, threshold=0.0)
        assert len(hits) == 1

    def test_sorted_descending(self):
        query = np.array([1.0, 0.0, 0.0], dtype="float32")
        hits = search(query, k=10, threshold=0.0)
        scores = [sc for _, sc in hits]
        assert scores == sorted(scores, reverse=True)

    def test_empty_cache_returns_empty(self, monkeypatch):
        monkeypatch.setattr(cache, "_vec_cache", None)
        monkeypatch.setattr(cache, "_id_cache", [])
        query = np.array([1.0, 0.0, 0.0], dtype="float32")
        assert search(query, k=5, threshold=0.0) == []
