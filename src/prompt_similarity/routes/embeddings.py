"""Embedding generation endpoint."""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from prompt_similarity import cache
from prompt_similarity.db import get_db
from prompt_similarity.embeddings import normalize, embed
from prompt_similarity.models import PromptInput
from prompt_similarity.app_state import get_openai_client

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


@router.post("/generate", summary="Generate (or refresh) embeddings")
def generate_embeddings(
    prompts: Optional[list[PromptInput]] = None,
    regenerate_all: bool = False,
):
    """Generate embeddings for prompts.

    Two modes:

    1. **With prompts list** — upserts prompts and generates embeddings.
    2. **Re-embed all** — empty body + ``?regenerate_all=true``.
    """
    with get_db() as conn:
        if prompts:
            conn.executemany(
                """INSERT OR REPLACE INTO prompts
                   (prompt_id, category, layer, name, content, content_normalized)
                   VALUES (?,?,?,?,?,?)""",
                [
                    (p.prompt_id, p.category, p.layer, p.name or p.prompt_id,
                     p.content, normalize(p.content))
                    for p in prompts
                ],
            )
            ids = [p.prompt_id for p in prompts]
            texts = [normalize(p.content) for p in prompts]
        elif regenerate_all:
            rows = conn.execute(
                "SELECT prompt_id, content_normalized FROM prompts"
            ).fetchall()
            if not rows:
                raise HTTPException(404, "No prompts in DB. Pass a prompts list to seed.")
            ids = [r["prompt_id"] for r in rows]
            texts = [r["content_normalized"] for r in rows]
        else:
            raise HTTPException(
                400,
                "Send a JSON array of prompts, or use ?regenerate_all=true "
                "to re-embed existing prompts.",
            )

    client = get_openai_client()

    t0 = time.perf_counter()
    vecs = embed(texts, client)
    embed_ms = round((time.perf_counter() - t0) * 1000, 3)

    with get_db() as conn:
        conn.executemany(
            "UPDATE prompts SET embedding = ? WHERE prompt_id = ?",
            [(vec.tobytes(), pid) for pid, vec in zip(ids, vecs)],
        )

    t1 = time.perf_counter()
    cache.rebuild()
    cache_ms = round((time.perf_counter() - t1) * 1000, 3)

    id_cache = cache.get_ids()
    vec_cache = cache.get_vectors()

    return {
        "generated":  len(ids),
        "prompt_ids": ids,
        "cache": {
            "total_vectors": len(id_cache),
            "memory_bytes":  vec_cache.nbytes if vec_cache is not None else 0,
        },
        "latency": {
            "embed_ms":         embed_ms,
            "cache_rebuild_ms": cache_ms,
            "num_texts":        len(texts),
            "ms_per_text":      round(embed_ms / len(texts), 3) if texts else 0,
        },
    }
