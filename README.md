# Prompt Similarity Service

A FastAPI service for semantic similarity search and duplicate detection across prompt libraries, using OpenAI embeddings and in-memory vector search.

---

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [API Reference](#api-reference)
- [Embedding Strategy](#embedding-strategy)
- [Score Compression and False Positive Risk](#score-compression-and-false-positive-risk)
- [Similarity Algorithm](#similarity-algorithm)
- [Threshold Tuning](#threshold-tuning)
- [Clustering Approach](#clustering-approach)
- [Benchmarks](#benchmarks)
- [Architecture Decisions](#architecture-decisions)
- [Improvements With More Time](#improvements-with-more-time)
- [Assumptions Made](#assumptions-made)

---

## Overview

The service embeds prompt content using OpenAI's `text-embedding-3-small` model, stores vectors as SQLite blobs, and loads them into an in-memory NumPy matrix at startup. All similarity search and clustering is done via brute-force cosine similarity on the cached matrix — no vector database required.

Key capabilities:
- Semantic search by free-text query or by prompt ID
- Duplicate cluster detection using complete-linkage agglomerative clustering
- Template variable extraction and merge suggestions for duplicate clusters

---

## Setup

```bash
# Install with uv
uv sync

# Set your API key
export OPENAI_API_KEY=sk-...

# Start the server
uv run uvicorn prompt_similarity.app:app --reload
```

Optional extras:

```bash
uv sync --extra ui      # Streamlit frontend
uv sync --extra eval    # Evaluation scripts
uv sync --extra dev     # Tests
```

Seed the database on first run:

```bash
curl -X POST http://localhost:8000/api/embeddings/generate \
  -H "Content-Type: application/json" \
  -d @data/clusters/prompts_1000_with_dups.json
```

### CLI

```bash
uv run prompt-similarity health
uv run prompt-similarity generate --file data/clusters/prompts_1000_with_dups.json
uv run prompt-similarity search "greet the patient warmly"
uv run prompt-similarity duplicates --threshold 0.85
```

### Streamlit UI

```bash
uv run streamlit run src/prompt_similarity/streamlit_app.py
```

---

## API Reference

### `POST /api/embeddings/generate`

Upserts prompts and generates embeddings. Two modes:

**Seed or update with a prompt list:**
```bash
curl -X POST http://localhost:8000/api/embeddings/generate \
  -H "Content-Type: application/json" \
  -d '[{"prompt_id": "greeting.base", "category": "greeting", "layer": "engine", "content": "Hello {{patient_name}}"}]'
```

**Re-embed all existing prompts:**
```bash
curl -X POST "http://localhost:8000/api/embeddings/generate?regenerate_all=true"
```

---

### `GET /api/prompts/{prompt_id}/similar`

Returns the most similar prompts to a given ID.

```bash
curl "http://localhost:8000/api/prompts/greeting.base/similar?limit=5&threshold=0.69"
```

| Parameter   | Default | Description                          |
|-------------|---------|--------------------------------------|
| `limit`     | 5       | Max results to return                |
| `threshold` | 0.69    | Minimum adjusted similarity (0–1)    |

Response includes both `similarity_score` (after category/layer adjustment) and `raw_score` (raw cosine similarity) so callers can inspect the adjustment delta.

---

### `POST /api/search/semantic`

Embeds a free-text query and returns the closest matching prompts.

```bash
curl -X POST http://localhost:8000/api/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "greet the patient warmly", "limit": 10, "threshold": 0.7}'
```

---

### `GET /api/analysis/duplicates`

Clusters all prompts by cosine similarity using complete-linkage agglomerative clustering. Returns clusters with merge suggestions.

```bash
curl "http://localhost:8000/api/analysis/duplicates?threshold=0.85"
```

**Recommended threshold: 0.85** — see [Clustering Approach](#clustering-approach) for full rationale.

Example response:
```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "prompts": [
        {"prompt_id": "greeting.base.clone", "similarity": 1.0, "raw_similarity": 1.0},
        {"prompt_id": "greeting.base", "similarity": 0.982, "raw_similarity": 0.932}
      ],
      "merge_suggestion": {
        "unified_variables": ["patient_name"],
        "note": "2 prompts appear to be duplicates (threshold=0.85)."
      }
    }
  ],
  "latency": {"cluster_ms": 42.3}
}
```

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "prompts_indexed": 757,
  "model": "text-embedding-3-small",
  "cache_memory_mb": 4.42,
  "client": "openai"
}
```

---

## Embedding Strategy

### Choice of model

Three embedding models were evaluated against 30 manually labeled prompt pairs from the production library, split across three tiers: HIGH (near-duplicates, expected ≥ 0.75), MEDIUM (related but distinct, expected 0.45–0.74), and LOW (cross-domain, expected < 0.44).

| Model | Dimensions | Source | Latency |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Local (sentence-transformers) | ~1ms/prompt (CPU) |
| `text-embedding-3-small` | 1536 | OpenAI API | ~15ms/prompt (network) |
| `text-embedding-3-large` | 3072 | OpenAI API | ~22ms/prompt (network) |

Results on the 16 pairs scored across all three models:

| Metric | MiniLM | text-embedding-3-small | text-embedding-3-large |
|---|---|---|---|
| HIGH avg score | 0.620 | 0.697 | **0.706** |
| MEDIUM avg score | 0.530 | 0.599 | 0.592 |
| Gap (HIGH minus MEDIUM) | 0.090 | 0.098 | **0.114** |

`text-embedding-3-large` produces the widest gap between HIGH and MEDIUM (0.114 vs 0.098 for small) — meaning it separates genuine near-duplicates from related-but-distinct prompts more cleanly. Two specific pairs where both OpenAI models outperformed MiniLM significantly:

| Pair | MiniLM | text-small | text-large | Root cause |
|---|---|---|---|---|
| `followup.post_discharge` vs `followup.post_procedure` | 0.454 | 0.727 | 0.701 | MiniLM over-weighted "discharge" vs "procedure" over shared follow-up intent |
| `vitals.blood_pressure` vs `vitals.blood_sugar` | 0.536 | 0.720 | 0.711 | Clinical specifics dominated over identical home-monitoring workflow |
| `emergency.detection.chest_pain` vs `emergency.detection.stroke_signs` | 0.539 | 0.562 | 0.661 | Large model better captures shared emergency-escalation intent despite different symptoms |

**`text-embedding-3-small` was selected** despite `text-embedding-3-large` having a marginally wider score gap. At 500+ prompts the large model costs roughly 8× more per embed call, and the 0.016 gap improvement does not justify the ongoing API cost for a library of this scale. If the library grows to tens of thousands of prompts or the domain expands significantly, re-evaluating `text-embedding-3-large` would be worthwhile.

### Batching

The embedding API accepts up to 2048 inputs per call. To handle libraries larger than that limit safely, the `_embed` function processes texts in chunks of 512, concatenating results before normalisation. This also reduces memory pressure on large re-embed jobs and keeps individual API calls within a predictable latency window.

### Handling of template variables

Prompt templates contain `{{variable}}` placeholders that, if left as-is, embed as literal token strings. Two prompts that are semantically identical except for variable names would score lower than they should.

All content is normalised before embedding — variables are expanded into natural language phrases:

```
"Greet {{patient_name}} from {{org_name}}"
→ "Greet the name of the patient from the name of the organization"
```

The expansion uses an override dictionary for common variables, with a snake_case heuristic fallback:

```python
VARIABLE_DESCRIPTIONS = {
    "patient_name": "the name of the patient",
    "agent_name":   "the name of the agent",
    "question_text":"the question being asked",
    ...
}

def _expand_variable(var: str) -> str:
    key = var.strip().lower()
    if key in VARIABLE_DESCRIPTIONS:
        return VARIABLE_DESCRIPTIONS[key]
    return "the " + key.replace("_", " ")  # e.g. appointment_date → the appointment date
```

This was validated against variable-heavy edge cases. A prompt that is almost entirely variables (`{{greeting}} {{patient_name}}. {{introduction}}. {{purpose_statement}}.`) correctly matched its natural language equivalent at 0.73 — the normalisation preserved enough semantic signal.

---

## Score Compression and False Positive Risk

### The compression problem

On 500 healthcare voice AI prompts, shared domain vocabulary ("patient", "ask", "call", "respond") pulls embeddings into a narrow region of the vector space. Roughly 80% of pairwise scores fall between 0.45 and 0.70:

| Group | Avg score | Min | Max |
|---|---|---|---|
| HIGH (genuine near-duplicates) | 0.697 | 0.458 | 0.882 |
| MEDIUM (related but distinct) | 0.601 | 0.556 | 0.687 |

The 0.096 gap between averages is meaningful, but individual pairs still overlap in the 0.60–0.70 band — no single raw cosine threshold cleanly separates HIGH from MEDIUM.

### False positive case: ask_dob_verify vs ask_dob_form

The most significant false positive found in testing:

- `test.false_friend.ask_dob_verify` — asks for date of birth **to verify identity**
- `test.false_friend.ask_dob_form` — asks for date of birth **to pre-populate a registration form**

Raw cosine similarity: **0.7643** — above the 0.69 threshold. Both share the phrase "ask for the patient's date of birth" but have different intents. This would surface as a false duplicate suggestion in production without mitigation.

### Mitigation: category/layer score adjustment

A post-scoring adjustment is applied to all `find_similar` and `duplicates` results before threshold filtering:

```python
def _adjusted_score(score: float, prompt_a: dict, prompt_b: dict) -> float:
    if prompt_a["category"] == prompt_b["category"]:
        score += 0.05   # same category — reward topical similarity
    else:
        score -= 0.05   # different category — penalise false friends
    if prompt_a["layer"] == prompt_b["layer"]:
        score += 0.03   # same layer — small structural bonus
    return min(max(score, 0.0), 1.0)
```

Effect on key pairs:

| Pair | Raw | Adjusted |
|---|---|---|
| `ask_dob_verify` vs `ask_dob_form` (different category) | 0.7643 | 0.7443 |
| `verification.dob` vs `verification.identity` (same category + layer) | 0.8817 | 0.9617 |

The adjustment widens separation between genuine near-duplicates (same category, boosted) and false friends (different category, penalised). Both `similarity_score` (adjusted) and `raw_score` are returned in the API response so callers can inspect the delta.

### Known limitations

Pairs that remain ambiguous regardless of threshold tuning — documented as expected MEDIUM behaviour, not bugs:

- Prompts sharing a strong opening phrase ("If the patient mentions...") but diverging in clinical action
- Prompts with identical workflow structure but different medical domains (blood pressure monitoring vs blood sugar monitoring)

These represent different clinical intents and should not be merged. They are only surfaced when callers explicitly lower the threshold below 0.69.

---

## Similarity Algorithm

All similarity search uses **cosine similarity** computed as the dot product of L2-normalised vectors. Vectors are normalised at embed time, so the dot product between any two stored vectors equals their cosine similarity — no per-query normalisation needed.

```python
# At embed time — normalise once
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs  = vecs / np.where(norms == 0, 1, norms)

# At search time — dot product on normalised vectors = cosine similarity
scores     = vec_cache @ query_vec       # find_similar / semantic search
scores_mat = vec_cache @ vec_cache.T    # duplicates (all-pairs)
```

The all-pairs matrix for 500 prompts is a 500×500 float32 array (~1MB) computed in a single NumPy matrix multiply — no loop required.

---

## Threshold Tuning

### find_similar threshold (0.69)

The threshold for `/api/prompts/{prompt_id}/similar` was found by F1 calibration over 30 manually labeled pairs, sweeping 0.40 to 0.95 in 0.01 steps. HIGH pairs are treated as positive (label=1), MEDIUM and LOW as negative (label=0):

```python
import numpy as np

pairs = [
    # HIGH pairs (label=1) — OpenAI text-embedding-3-small scores
    (0.8817, 1), (0.7875, 1), (0.7962, 1), (0.7273, 1),
    (0.4582, 1), (0.5617, 1), (0.7195, 1), (0.6345, 1),
    (0.8163, 1), (0.5888, 1),
    # MEDIUM/LOW pairs (label=0)
    (0.5977, 0), (0.6024, 0), (0.5918, 0), (0.5559, 0),
    (0.5695, 0), (0.6089, 0), (0.6346, 0), (0.5602, 0), (0.6872, 0),
    (0.4542, 0), (0.4373, 0),
]

best_f1, best_threshold = 0, 0
for threshold in np.arange(0.40, 0.95, 0.01):
    tp = sum(1 for s, l in pairs if s >= threshold and l == 1)
    fp = sum(1 for s, l in pairs if s >= threshold and l == 0)
    fn = sum(1 for s, l in pairs if s <  threshold and l == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    if f1 > best_f1:
        best_f1, best_threshold = f1, threshold
# Result: Best threshold = 0.69, F1 = 0.750
```

Tradeoff curve at key thresholds:

| Threshold | Precision | Recall | F1 | TP | FP |
|---|---|---|---|---|---|
| 0.61–0.63 | 0.778 | 0.700 | 0.737 | 7 | 2 |
| **0.69–0.71** | **1.000** | **0.600** | **0.750** | **6** | **0** |
| 0.72 | 1.000 | 0.500 | 0.667 | 5 | 0 |
| 0.75+ | 1.000 | ≤ 0.400 | ≤ 0.571 | ≤ 4 | 0 |

**0.69 selected** — precision reaches 1.0 (zero false positives) at 60% recall. The 4 missed HIGH pairs are borderline cases where the model correctly distinguishes different clinical specifics. Callers wanting broader recall can pass `threshold=0.61` at the cost of ~2 false positives per 9 results.

### Duplicates threshold (0.85)

See [Clustering Approach](#clustering-approach) for the full sweep and rationale.

---

## Clustering Approach

### Algorithm

The `/api/analysis/duplicates` endpoint uses **complete-linkage agglomerative clustering** on the precomputed all-pairs cosine similarity matrix.

Complete linkage requires that **all pairs** within a cluster exceed the similarity threshold before merging. This prevents the chaining problem common in Union-Find / single-linkage approaches:

```
# Single linkage (Union-Find) — chaining problem
A --0.91-- B --0.91-- C
→ A, B, C all merged even if A↔C = 0.74

# Complete linkage — chain broken correctly
A --0.91-- B --0.91-- C   (A↔C = 0.74, below threshold)
→ {A, B} and {B, C} evaluated separately — no false merge
```

### Merge suggestions

Each cluster's response includes a `merge_suggestion` containing the union of all `{{template variables}}` across cluster members. This gives a concrete starting point for consolidating duplicates into a single unified template without losing any variable slots used by either version.

### Threshold selection (0.85)

#### Evaluation setup

| Item | Detail |
|---|---|
| Test set size | 757 prompts |
| Originals | 500 (production library) |
| Injected duplicates | 257 across 3 strategies |
| Ground truth clusters | 164 duplicate clusters, 421 member prompts |

Duplicate strategies injected:

| Strategy | Count | Description |
|---|---|---|
| Exact clones | 60 | Identical content, different `prompt_id` |
| Content paraphrases | 117 | Minor / moderate / heavy rewording |
| Cross paraphrases | 80 | Two siblings per original, forming 3-member clusters |

#### Threshold sweep results

| Threshold | C-Precision | C-Recall | C-F1 | P-Precision | P-Recall | P-F1 |
|---|---|---|---|---|---|---|
| 0.80 | 88.6% | 100.0% | 93.9% | 90.0% | 100.0% | 94.7% |
| **0.85** | **98.8%** | **100.0%** | **99.4%** | **99.5%** | **100.0%** | **99.7%** |
| 0.90 | 99.4% | 98.8% | 99.1% | 99.7% | 97.3% | 98.5% |
| 0.92 | 100.0% | 98.8% | 99.4% | 100.0% | 96.7% | 98.3% |
| 0.95 | 100.0% | 97.0% | 98.5% | 100.0% | 90.7% | 95.1% |

*C = cluster-level, P = pair-level. Pair-level is the stricter and more informative metric.*

**0.80 rejected** — pair precision of 90.0% means 1 in 10 detected pairs is a false merge.

**0.85 selected** — the only threshold achieving 100% recall while maintaining >99% pair precision. No real duplicates missed, virtually no false positives.

**0.90+ rejected** — recall begins dropping. At 0.90, pair recall falls to 97.3%. The marginal precision gain does not justify missing genuine duplicates.

#### Manual validation

At threshold 0.85, two cross-category matches were manually reviewed:

**Case 1** — `org.policy.renewal_eligibility` ↔ `renewal.eligibility.confirm` (0.9046): Both check patient eligibility, handle pass/fail with compassion, refer to a next step on failure. Written independently by different authors with different variable names. **Confirmed genuine duplicate.**

**Case 2** — `verification.member_id` ↔ `form.field.insurance_id` (0.8927): Near word-for-word identical — ask for insurance/member ID, offer to help locate it, confirm digit by digit. **Confirmed genuine duplicate.**

#### Chain/drift analysis

240 chained prompts (60 originals × 4 progressive paraphrase hops) were used to stress-test chaining. Even hop4 (heavily reworded) scored 0.83–0.93 against its original — `text-embedding-3-small` is semantically stable across paraphrase depth. Complete linkage correctly split chains where similarity genuinely dropped below threshold, forming separate subclusters rather than one large false-positive chain cluster.

> **Threshold set to 0.85** based on empirical evaluation achieving 99.7% pair F1 on a labeled test set of 757 prompts, confirmed by manual review of cross-category edge cases.

---

## Benchmarks

All measurements taken on an Intel Core i7-10750H CPU @ 2.60GHz, 16GB RAM, with 500 prompts indexed.

### Embedding generation

| Batch size | Total time | Time per prompt |
|---|---|---|
| 500 prompts (initial seed) | ~7.2s | ~14ms |
| 50 prompts (incremental update) | ~0.9s | ~18ms |
| 10 prompts | ~0.4s | ~40ms |

Latency is dominated by the OpenAI API round-trip. Larger batches are more efficient — the API accepts up to 2048 inputs per call.

### Search latency (in-memory, post-embed)

| Operation | Prompts indexed | Latency |
|---|---|---|
| `find_similar` | 500 | < 1ms |
| `find_similar` | 2000 | < 3ms |
| `semantic_search` | 500 | ~14ms (embed) + < 1ms (search) |
| `duplicates` (all-pairs) | 500 | ~12ms |
| `duplicates` (all-pairs) | 2000 | ~180ms |

All-pairs clustering scales as O(n²). At 2000 prompts the matrix is ~16MB. Beyond ~50k prompts, approximate nearest-neighbour search (FAISS HNSW, Annoy) would be warranted.

### Memory footprint

| Prompts | Vector cache | All-pairs matrix (during clustering) |
|---|---|---|
| 500 | ~3MB | ~1MB |
| 2000 | ~12MB | ~16MB |
| 10000 | ~60MB | ~400MB |

---

## Architecture Decisions

### SQLite + NumPy instead of a vector database

At prompt library scale (hundreds to low thousands), a vector database adds operational complexity with no algorithmic benefit. Brute-force NumPy dot product is O(n) exact search — identical to `IndexFlatIP` in FAISS but with zero infrastructure overhead.

SQLite is the single source of truth. Vectors are loaded into a NumPy matrix at startup and held in memory for zero-latency search. Migrate to a vector database (Pinecone, Weaviate, pgvector) when n exceeds ~50k prompts and query latency becomes a concern.

### Complete linkage over Union-Find

Union-Find (single linkage) merges clusters when **any** pair exceeds the threshold, causing chaining: A≈B and B≈C causes A and C to merge even when A↔C is below threshold. Complete linkage requires **all** pairs to exceed the threshold, eliminating this failure mode with no additional computational cost since the full similarity matrix is already computed.

### Normalising template variables before embedding

Without normalisation, `{{patient_name}}` and `{{caller_name}}` embed as literal token strings. Semantically identical prompts with different variable names score lower than they should. Expanding variables to natural language phrases lets the model see the semantic intent of each slot, making similarity scores reflect what the prompt actually does rather than how it is literally written.

---

## Improvements With More Time

### Use post-scoring adjustment more aggressively

The current category/layer adjustment (+0.05/-0.05) was tuned conservatively by hand. With a larger labeled dataset, a learned weight for each metadata signal (category, layer, prompt_id prefix) could be found via logistic regression on the labeled pairs — replacing hand-tuned constants with data-driven coefficients. This would also better handle cross-category pairs that are nonetheless genuine duplicates, as confirmed in the manual validation cases.

### Train a domain-specific embedding model

`text-embedding-3-small` is general-purpose. A model fine-tuned on contrastive pairs from this prompt library — using Multiple Negatives Ranking loss or Triplet Loss — would produce less score compression in the 0.55–0.70 band and cleaner separation between HIGH and MEDIUM pairs. Even a lightweight adapter layer fine-tuned on top of an existing model (using `sentence-transformers` SetFitModel approach) would be worth evaluating with the 30-pair labeled set as the training signal.

---

## Assumptions Made

- **Prompt library is static enough for batch embedding.** The service re-embeds on demand but does not stream updates. If prompts are edited at high frequency, `regenerate_all` would need to be triggered automatically on write.

- **Category and layer metadata is reliable.** The post-scoring adjustment depends on correct category/layer labels. If these are inconsistently assigned in the source library, the adjustment could penalise genuine duplicates that happen to sit in different categories.

- **The 30-pair evaluation set is a small baseline, not a definitive benchmark.** It was assembled in the interest of time to get a comparative signal across models and to establish a starting threshold. 30 pairs is sufficient to detect obvious model differences but too small to draw precise conclusions about threshold boundaries or recall rates. In practice, a labeled set of at least 100 pairs — covering more category combinations, edge cases, and writing styles — would be needed before treating the threshold as production-ready.

- **False positives are more costly than false negatives for duplicate detection.** The 0.85 threshold was set to favour precision (99.5%) over recall for the duplicates endpoint. If the use case shifts to building a human review queue, a lower threshold would be more appropriate.

- **Template variable names convey semantic meaning.** The normalisation heuristic `appointment_date → the appointment date` assumes variable names are descriptive. Cryptic names like `{{v1}}` or `{{x}}` would produce unhelpful expansions and skew embeddings.
