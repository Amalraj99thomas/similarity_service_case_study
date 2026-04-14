"""Tests for API routes via FastAPI TestClient.

The OpenAI client and database are mocked so tests run without external
dependencies.
"""

import numpy as np
import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from prompt_similarity import cache


@pytest.fixture
def client():
    """Return a TestClient with the cache pre-populated and OpenAI mocked."""
    vecs = np.array([
        [1.0, 0.0, 0.0],
        [0.95, 0.31, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    cache._vec_cache = vecs
    cache._id_cache = ["prompt_a", "prompt_b", "prompt_c"]
    cache._content_cache = [
        "Greet the patient warmly.",
        "Welcome the caller with a friendly opener.",
        "Verify the patient's date of birth.",
    ]

    # Patch away lifespan side-effects that need real OpenAI credentials / DB
    with patch("prompt_similarity.app.init_client"), \
         patch("prompt_similarity.app.init_db"), \
         patch("prompt_similarity.app.cache.rebuild"):
        from prompt_similarity.app import app
        with TestClient(app, raise_server_exceptions=False) as tc:
            yield tc

    cache._vec_cache = None
    cache._id_cache = []
    cache._content_cache = []


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["prompts_indexed"] == 3


class TestFindSimilarEndpoint:
    """Tests for GET /api/prompts/{prompt_id}/similar."""

    def test_returns_results(self, client):
        resp = client.get("/api/prompts/prompt_a/similar?limit=5&threshold=0.5")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "latency" in data

    def test_excludes_self(self, client):
        resp = client.get("/api/prompts/prompt_a/similar?limit=10&threshold=0.0")
        data = resp.json()
        result_ids = [r["prompt_id"] for r in data["results"]]
        assert "prompt_a" not in result_ids

    def test_unknown_prompt_returns_404(self, client):
        resp = client.get("/api/prompts/nonexistent/similar")
        assert resp.status_code == 404


class TestDuplicatesEndpoint:
    """Tests for GET /api/analysis/duplicates."""

    def test_returns_clusters(self, client):
        resp = client.get("/api/analysis/duplicates?threshold=0.8")
        assert resp.status_code == 200
        data = resp.json()
        assert "clusters" in data
        assert "latency" in data

    def test_high_threshold_may_return_empty(self, client):
        resp = client.get("/api/analysis/duplicates?threshold=0.999")
        data = resp.json()
        assert isinstance(data["clusters"], list)
