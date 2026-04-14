"""Tests for prompt_similarity.clustering — duplicate cluster detection."""

import numpy as np
import pytest

from prompt_similarity import cache
from prompt_similarity.clustering import find_duplicate_clusters


@pytest.fixture
def _two_cluster_cache(monkeypatch):
    """Set up a cache with two obvious clusters and one outlier.

    Vectors 0 and 1 are near-identical (cluster A).
    Vectors 2 and 3 are near-identical (cluster B).
    Vector 4 is orthogonal to everything (singleton).
    """
    vecs = np.array([
        [1.0, 0.0, 0.0],     # cluster A
        [0.99, 0.14, 0.0],   # cluster A — cos ≈ 0.99
        [0.0, 1.0, 0.0],     # cluster B
        [0.0, 0.99, 0.14],   # cluster B — cos ≈ 0.99
        [0.0, 0.0, 1.0],     # outlier
    ], dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    monkeypatch.setattr(cache, "_vec_cache", vecs)
    monkeypatch.setattr(cache, "_id_cache", ["a1", "a2", "b1", "b2", "outlier"])
    monkeypatch.setattr(cache, "_content_cache", [
        "Content A1",
        "Content A2",
        "Content B with {{var_x}}",
        "Content B with {{var_y}}",
        "Unrelated content",
    ])


class TestFindDuplicateClusters:
    """Tests for the clustering function."""

    def test_finds_expected_clusters(self, _two_cluster_cache):
        clusters, ms = find_duplicate_clusters(threshold=0.9)
        assert len(clusters) == 2
        assert ms > 0

    def test_cluster_members_correct(self, _two_cluster_cache):
        clusters, _ = find_duplicate_clusters(threshold=0.9)
        all_ids = set()
        for c in clusters:
            for p in c["prompts"]:
                all_ids.add(p["prompt_id"])
        assert "a1" in all_ids
        assert "a2" in all_ids
        assert "b1" in all_ids
        assert "b2" in all_ids
        assert "outlier" not in all_ids

    def test_outlier_excluded(self, _two_cluster_cache):
        clusters, _ = find_duplicate_clusters(threshold=0.9)
        all_ids = {p["prompt_id"] for c in clusters for p in c["prompts"]}
        assert "outlier" not in all_ids

    def test_merge_suggestion_includes_vars(self, _two_cluster_cache):
        clusters, _ = find_duplicate_clusters(threshold=0.9)
        # Find the cluster containing b1/b2 (which have template vars)
        b_cluster = next(
            c for c in clusters
            if any(p["prompt_id"] in ("b1", "b2") for p in c["prompts"])
        )
        unified = b_cluster["merge_suggestion"]["unified_variables"]
        assert "var_x" in unified
        assert "var_y" in unified

    def test_high_threshold_splits_more(self, _two_cluster_cache):
        clusters, _ = find_duplicate_clusters(threshold=0.999)
        # At a very high threshold, clusters may split or disappear
        all_ids = {p["prompt_id"] for c in clusters for p in c["prompts"]}
        assert "outlier" not in all_ids

    def test_empty_cache_raises(self, monkeypatch):
        monkeypatch.setattr(cache, "_vec_cache", None)
        monkeypatch.setattr(cache, "_id_cache", [])
        monkeypatch.setattr(cache, "_content_cache", [])
        with pytest.raises(ValueError, match="No embeddings found"):
            find_duplicate_clusters(threshold=0.85)
