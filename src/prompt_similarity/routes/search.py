"""Search endpoints: find similar by prompt ID and semantic free-text search."""

import time

from fastapi import APIRouter, HTTPException

from prompt_similarity import cache
from prompt_similarity.search import search
from prompt_similarity.embeddings import normalize, embed
from prompt_similarity.models import SemanticSearchRequest
from prompt_similarity.app_state import get_openai_client

router = APIRouter(tags=["search"])


@router.get("/api/prompts/{prompt_id}/similar", summary="Find prompts similar to a given ID")
def find_similar(prompt_id: str, limit: int = 5, threshold: float = 0.8):
    """Return up to *limit* prompts whose cosine similarity to *prompt_id*
    is ≥ *threshold*.  The source prompt itself is excluded.

    Entirely in-memory — zero SQLite reads.
    """
    id_cache = cache.get_ids()
    vec_cache = cache.get_vectors()

    if prompt_id not in id_cache:
        raise HTTPException(
            404,
            f"No embedding found for '{prompt_id}'. Run /api/embeddings/generate first.",
        )

    src_idx = id_cache.index(prompt_id)
    vec = vec_cache[src_idx]

    t0 = time.perf_counter()
    hits = search(vec, limit + 1, threshold)
    search_ms = round((time.perf_counter() - t0) * 1000, 3)

    # Drop self-match and trim to requested limit
    hits = [(i, sc) for i, sc in hits if id_cache[i] != prompt_id][:limit]

    return {
        "query_prompt_id": prompt_id,
        "results": [
            {
                "prompt_id":        id_cache[i],
                "similarity_score": round(sc, 4),
                "content_preview":  cache.content_preview(i),
            }
            for i, sc in hits
        ],
        "latency": {"search_ms": search_ms},
    }


@router.post("/api/search/semantic", summary="Semantic search by free-text query")
def semantic_search(req: SemanticSearchRequest):
    """Embed *query* on the fly and return the closest *limit* prompts.

    Search is fully in-memory; only the embedding API call hits the network.
    """
    client = get_openai_client()

    t0 = time.perf_counter()
    query_vec = embed([normalize(req.query)], client)[0]
    embed_ms = round((time.perf_counter() - t0) * 1000, 3)

    t1 = time.perf_counter()
    hits = search(query_vec, req.limit, req.threshold)
    search_ms = round((time.perf_counter() - t1) * 1000, 3)

    id_cache = cache.get_ids()

    return {
        "results": [
            {
                "prompt_id":        id_cache[i],
                "similarity_score": round(sc, 4),
                "content_preview":  cache.content_preview(i),
            }
            for i, sc in hits
        ],
        "latency": {
            "embed_ms":  embed_ms,
            "search_ms": search_ms,
            "total_ms":  round(embed_ms + search_ms, 3),
        },
    }
