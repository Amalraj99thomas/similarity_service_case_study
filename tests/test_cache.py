"""Tests for prompt_similarity.cache — in-memory vector cache."""

import numpy as np
import pytest

from prompt_similarity import cache


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    """Reset cache to a known state before each test."""
    monkeypatch.setattr(cache, "_vec_cache", None)
    monkeypatch.setattr(cache, "_id_cache", [])
    monkeypatch.setattr(cache, "_content_cache", [])


class TestCacheAccessors:
    """Tests for cache getter functions."""

    def test_empty_cache_vectors(self):
        assert cache.get_vectors() is None

    def test_empty_cache_ids(self):
        assert cache.get_ids() == []

    def test_empty_cache_contents(self):
        assert cache.get_contents() == []

    def test_populated_cache(self, monkeypatch):
        vecs = np.ones((2, 3), dtype="float32")
        monkeypatch.setattr(cache, "_vec_cache", vecs)
        monkeypatch.setattr(cache, "_id_cache", ["a", "b"])
        monkeypatch.setattr(cache, "_content_cache", ["text a", "text b"])

        assert cache.get_vectors() is not None
        assert len(cache.get_ids()) == 2
        assert len(cache.get_contents()) == 2


class TestContentPreview:
    """Tests for the content preview helper."""

    def test_short_text_returned_fully(self, monkeypatch):
        monkeypatch.setattr(cache, "_content_cache", ["Hello world"])
        assert cache.content_preview(0) == "Hello world"

    def test_long_text_truncated(self, monkeypatch):
        long_text = "A" * 200
        monkeypatch.setattr(cache, "_content_cache", [long_text])
        preview = cache.content_preview(0, max_len=120)
        assert len(preview) == 123  # 120 chars + "..."
        assert preview.endswith("...")

    def test_custom_max_len(self, monkeypatch):
        monkeypatch.setattr(cache, "_content_cache", ["Hello world, this is a test"])
        preview = cache.content_preview(0, max_len=5)
        assert preview == "Hello..."
