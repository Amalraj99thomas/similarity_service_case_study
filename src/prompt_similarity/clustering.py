"""Complete-linkage agglomerative clustering for duplicate prompt detection.

Uses the precomputed all-pairs cosine similarity matrix from the in-memory
cache.  Complete linkage ensures that two clusters only merge when *all*
pairwise similarities exceed the threshold, preventing the chaining problem
where A≈B and B≈C causes A and C to merge even when A≉C directly.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from prompt_similarity import cache
from prompt_similarity.embeddings import extract_vars


def find_duplicate_clusters(threshold: float) -> tuple[list[dict], float]:
    """Cluster all cached prompts by cosine similarity.

    Args:
        threshold: Minimum cosine similarity for two prompts to be considered
            duplicates (e.g. 0.85).

    Returns:
        A tuple of ``(clusters, cluster_ms)`` where *clusters* is a list of
        cluster dicts ready for the API response and *cluster_ms* is the wall
        time spent clustering in milliseconds.

    Raises:
        ValueError: If the vector cache is empty.
    """
    import time

    vec_cache = cache.get_vectors()
    id_cache = cache.get_ids()
    content_cache = cache.get_contents()

    if vec_cache is None or len(id_cache) == 0:
        raise ValueError("No embeddings found. Run /api/embeddings/generate first.")

    t0 = time.perf_counter()

    scores_mat = vec_cache @ vec_cache.T

    # Convert to cosine distance and clamp floating-point noise to [0, 2]
    distance_mat = np.clip(1.0 - scores_mat, 0.0, 2.0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=1.0 - threshold,
    )
    labels = clustering.fit_predict(distance_mat)

    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    multi = {label: members for label, members in clusters.items() if len(members) > 1}

    cluster_ms = round((time.perf_counter() - t0) * 1000, 3)

    result: list[dict] = []
    for cluster_id, (label, members) in enumerate(multi.items()):
        rep_idx = members[0]

        prompts_out: list[dict] = []
        for i in members:
            sim = float(scores_mat[i, rep_idx]) if i != rep_idx else 1.0
            prompts_out.append({
                "prompt_id":  id_cache[i],
                "similarity": round(sim, 4),
            })

        # Union of template variables across cluster members
        all_vars: list[str] = []
        for i in members:
            all_vars.extend(extract_vars(content_cache[i]))
        unified_vars = sorted(set(all_vars))

        result.append({
            "cluster_id": cluster_id,
            "prompts":    prompts_out,
            "merge_suggestion": {
                "unified_variables": unified_vars,
                "note": (
                    f"{len(members)} prompts appear to be duplicates "
                    f"(threshold={threshold}). Consider merging into a single "
                    f"template with variables: {unified_vars or 'none'}."
                ),
            },
        })

    return result, cluster_ms
