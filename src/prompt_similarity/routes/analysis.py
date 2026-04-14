"""Duplicate detection endpoint."""

from fastapi import APIRouter, HTTPException

from prompt_similarity.clustering import find_duplicate_clusters

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.get("/duplicates", summary="Cluster likely duplicate prompts")
def find_duplicates(threshold: float = 0.85):
    """All-pairs cosine similarity on the cached matrix.

    Complete-linkage agglomerative clustering: two clusters only merge when
    ALL pairwise similarities exceed the threshold, preventing the chaining
    problem where A≈B and B≈C causes A and C to merge even when A≉C directly.

    Entirely in-memory — no SQLite reads.
    """
    try:
        clusters, cluster_ms = find_duplicate_clusters(threshold)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "clusters": clusters,
        "latency":  {"cluster_ms": cluster_ms},
    }
