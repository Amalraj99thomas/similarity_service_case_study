"""Health check endpoint."""

from fastapi import APIRouter

from prompt_similarity import cache
from prompt_similarity.config import MODEL_NAME

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Return service status, index size, and memory usage."""
    id_cache = cache.get_ids()
    vec_cache = cache.get_vectors()

    return {
        "status":          "ok",
        "prompts_indexed": len(id_cache),
        "model":           MODEL_NAME,
        "cache_memory_mb": round(vec_cache.nbytes / 1_048_576, 2) if vec_cache is not None else 0,
        "client":          "openai",
    }
