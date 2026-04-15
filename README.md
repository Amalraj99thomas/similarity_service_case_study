# Prompt Similarity Service

A FastAPI service for semantic similarity search and duplicate detection across prompt libraries, using OpenAI embeddings and in-memory vector search.

---

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [API Reference](#api-reference)
- [Embedding Strategy](#embedding-strategy)
- [Similarity Algorithm](#similarity-algorithm)
- [Clustering Approach](#clustering-approach)
- [Threshold Tuning](#threshold-tuning)
- [Benchmarks](#benchmarks)
- [Architecture Decisions](#architecture-decisions)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
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

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

```bash
git clone https://github.com/Amalraj99thomas/similarity_service_case_study.git
cd similarity_service_case_study

uv sync
```

### Configuration

Create a `.env` file in the project root (or export the variable directly):

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

Or export it directly:

```bash
export OPENAI_API_KEY=sk-...
```

### Start the server

```bash
uv run uvicorn prompt_similarity.app:app --reload
```

The service will be available at `http://localhost:8000`. You can verify it's running:

```bash
curl http://localhost:8000/health
```

### Seed the database

On first run, load prompts to generate embeddings:

```bash
curl -X POST http://localhost:8000/api/embeddings/generate \
  -H "Content-Type: application/json" \
  -d @data/clustering/prompts_with_dups.json
```

Or use the CLI:

```bash
uv run prompt-similarity generate --file data/clustering/prompts_with_dups.json
```

### CLI

```bash
uv run prompt-similarity health
uv run prompt-similarity search "greet the patient warmly"
uv run prompt-similarity similar receptionist.greeting --threshold 0.69
uv run prompt-similarity duplicates --threshold 0.85
```

### Streamlit UI

```bash
uv sync --extra ui
uv run streamlit run prompt_similarity/streamlit_app.py
```

### Tests

```bash
uv sync --extra dev
uv run pytest
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

**Recommended threshold: 0.85** — see [Threshold Tuning](#threshold-tuning) for full rationale.

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

Three embedding models were evaluated against 30 manually labeled prompt pairs with `prompts_500` synthetically generated set, split across three tiers: HIGH (near-duplicates, expected ≥ 0.75), MEDIUM (related but distinct, expected 0.45–0.74), and LOW (cross-domain, expected < 0.44).

| Model | Dimensions | Source | Latency |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Local (sentence-transformers) | ~1ms/prompt (CPU) |
| `text-embedding-3-small` | 1536 | OpenAI API | ~15ms/prompt (network) |
| `text-embedding-3-large` | 3072 | OpenAI API | ~22ms/prompt (network) |

Results on the pairs scored across all three models:

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

**`text-embedding-3-small` was selected** despite `text-embedding-3-large` having a marginally wider score gap. At this scale the large model costs roughly 8× more per embed call, and the 0.016 gap improvement does not justify the ongoing API costs. If the library grows to tens of thousands of prompts or the domain expands significantly, re-evaluating `text-embedding-3-large` would be worthwhile.

### Batching

The `text-embedding-3-small` API accepts up to 2048 inputs per call. To handle libraries larger than that limit safely, the `_embed` function processes texts in chunks of 512, concatenating results before normalisation. This also reduces memory pressure on large re-embed jobs and keeps individual API calls within a predictable latency window.

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

## Similarity Algorithm

All similarity search uses **cosine similarity** computed as the dot product of L2-normalised vectors. Vectors are normalised at embed time, so the dot product between any two stored vectors equals their cosine similarity.

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

## Clustering Approach

### Algorithm

The `/api/analysis/duplicates` endpoint uses **complete-linkage agglomerative clustering** on the precomputed all-pairs cosine similarity matrix.

Complete linkage requires that **all pairs** within a cluster exceed the similarity threshold before merging. This prevents the chaining problem common in Union-Find / single-linkage approaches


### Merge suggestions

Each cluster's response includes a `merge_suggestion` containing the union of all `{{template variables}}` across cluster members. This gives a concrete starting point for consolidating duplicates into a single unified template without losing any variable slots used by either version.

---

## Threshold Tuning

### find_similar threshold (0.69)

#### Evaluation Setup

| Item | Detail |
|---|---|
| Test set source | `similarity/prompts_500.json` |
| Test set size | 500 prompts |
| Eval set size | 30 (manually labelled) |


The threshold for `/api/prompts/{prompt_id}/similar` was approximated by F1 calibration over 30 manually labeled pairs, sweeping 0.40 to 0.95 in 0.01 steps. HIGH pairs are treated as positive (label=1), MEDIUM and LOW as negative (label=0):


Tradeoff curve at key thresholds:

| Threshold | Precision | Recall | F1 | TP | FP |
|---|---|---|---|---|---|
| 0.61–0.63 | 0.778 | 0.700 | 0.737 | 7 | 2 |
| **0.69–0.71** | **1.000** | **0.600** | **0.750** | **6** | **0** |
| 0.72 | 1.000 | 0.500 | 0.667 | 5 | 0 |
| 0.75+ | 1.000 | ≤ 0.400 | ≤ 0.571 | ≤ 4 | 0 |

**0.69 selected** — precision reaches 1.0 (zero false positives) at 60% recall. The 4 missed HIGH pairs are borderline cases where the model correctly distinguishes different clinical specifics. Callers wanting broader recall can pass `threshold=0.61` at the cost of ~2 false positives per 9 results.


### find_duplicates threshold selection (0.85)

#### Evaluation setup

| Item | Detail |
|---|---|
| Test set size | 757 prompts |
| Test set source | `clustering/prompts_with_dups.json` |
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


### Edge cases

32 handcrafted prompts (`edge_case_prompts.json`) were used to stress-test similarity scoring across six categories that commonly cause problems in production:

| Category | Pair example | Score | Observation |
|---|---|---|---|
| Exact duplicates | `error_recovery.A` ↔ `error_recovery.B` | 1.00 | Identical content correctly scores 1.0 |
| Paraphrases | `verify_dob.original` ↔ `verify_dob.rephrased` | 0.81 | Moderate rewording stays well above the 0.69 threshold |
| Typo variants | `verify_dob.clean` ↔ `verify_dob.typos` | 0.88 | Heavy misspellings have minimal impact on embedding similarity |
| Short prompts | `greet_1` ↔ `greet_2` | 0.63 | Short prompts lack enough semantic signal to score above threshold — expected behaviour |
| Variable-heavy | `booking.lots_of_vars` ↔ `booking.few_vars` | 0.75 | Variable expansion preserves enough intent to score above threshold |
| Variable-heavy | `almost_all_vars` ↔ `no_vars_equivalent` | 0.72 | A near-entirely-variable prompt still matches its natural language equivalent |
| False friends | `ask_dob_verify` ↔ `ask_dob_form` | 0.76 | Shared surface text but different intent — see [Known Limitations](#known-limitations) |

The typo variant result (0.88) confirms that `text-embedding-3-small` is robust to misspellings — the model embeds semantic intent rather than exact character sequences. The short prompt result (0.63) is a known limitation: prompts under ~10 words lack enough context for the model to distinguish related intents, so they fall below the similarity threshold even when semantically close.


> **Threshold set to 0.85** based on empirical evaluation achieving 99.7% pair F1 on a labeled test set of 757 prompts, confirmed by manual review of cross-category edge cases.

---

## Benchmarks

All measurements taken on an Intel Core i7-10750H CPU @ 2.60GHz, 16GB RAM, with 500 prompts indexed.

### Embedding generation

| Batch size | Total time | Time per prompt |
|---|---|---|
| 500 prompts | ~2.3s | ~4ms |
| 257 prompts | ~2.8s | ~4ms |
| 32 prompts | ~1.7s | ~54ms |


Latency is dominated by the OpenAI API round-trip. Larger batches are more efficient — the API accepts up to 2048 inputs per call.

### Search latency (in-memory, post-embed)

| Operation | Prompts indexed | Latency |
|---|---|---|
| `find_similar` | 500 | < 1ms |
| `find_similar` | 1000 | < 1ms |
| `semantic_search` | 500 | ~800ms (embed) + < 1ms (search) |
| `semantic_search` | 1000 | ~280ms (embed) + < 1ms (search) |
| `duplicates` (all-pairs) | 500 | ~38ms |
| `duplicates` (all-pairs) | 1000 | ~57ms |

---

## Architecture Decisions

### SQLite + NumPy instead of a vector database

For current prompt library scale (hundreds to low thousands), a vector database adds operational complexity with no algorithmic benefit. Brute-force NumPy dot product is O(n) exact search — identical to `IndexFlatIP` in FAISS but with zero infrastructure overhead.

SQLite is the single source of truth. Vectors are loaded into a NumPy matrix at startup and held in memory for zero-latency search. Migrate to a vector database (Pinecone, Weaviate, pgvector) when n exceeds ~50k prompts and query latency becomes a concern.

### Normalising template variables before embedding

Without normalisation, `{{patient_name}}` and `{{caller_name}}` embed as literal token strings. Semantically identical prompts with different variable names score lower than they should. Expanding variables to natural language phrases lets the model see the semantic intent of each slot, making similarity scores reflect what the prompt actually does rather than how it is literally written.

### Complete linkage over Union-Find

Union-Find (single linkage) merges clusters when **any** pair exceeds the threshold, causing chaining: A≈B and B≈C causes A and C to merge even when A↔C is below threshold. Complete linkage requires **all** pairs to exceed the threshold, eliminating this failure mode with no additional computational cost since the full similarity matrix is already computed.

---

## Known Limitations

### The compression problem

On 500 healthcare voice AI prompts, shared domain vocabulary ("patient", "ask", "call", "respond") pulls embeddings into a narrow region of the vector space. Roughly 80% of pairwise scores fall between 0.45 and 0.70. No single raw cosine threshold can confidently separate the HIGH and MEDIUM tiers.

### False positive case: ask_dob_verify vs ask_dob_form

The most significant false positive found in testing:

- `test.false_friend.ask_dob_verify` — asks for date of birth **to verify identity**
- `test.false_friend.ask_dob_form` — asks for date of birth **to pre-populate a registration form**

Raw cosine similarity: **0.7643** — above the 0.69 threshold. Both share the phrase "ask for the patient's date of birth" but have different intents. This would surface as a false duplicate suggestion in production without mitigation.

#### Mitigation: category/layer score adjustment

A post-scoring adjustment applied to all `find_similar` and `duplicates` results before threshold filtering:

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

The adjustment widens separation between genuine near-duplicates (same category, boosted) and false friends (different category, penalised). Both `similarity_score` (adjusted) and `raw_score` can be returned in the API response so callers can inspect the delta.

### Storage of O(n^2)

All-pairs clustering scales as O(n²). At 2000 prompts the matrix is ~16MB. Beyond ~50k prompts, approximate nearest-neighbour search (FAISS HNSW, Annoy) would be warranted.

### Memory footprint

| Prompts | Vector cache | All-pairs matrix (during clustering) |
|---|---|---|
| 500 | ~3MB | ~1MB |
| 2000 | ~12MB | ~16MB |
| 10000 | ~60MB | ~400MB |

---

## Future Improvements

### Train a domain-specific embedding model

`text-embedding-3-small` is general-purpose. A model fine-tuned on contrastive pairs from this prompt library — using Multiple Negatives Ranking loss or Triplet Loss — would produce less score compression in the 0.55–0.70 band and cleaner separation between HIGH and MEDIUM pairs. Even a lightweight adapter layer fine-tuned on top of an existing model (using `sentence-transformers` SetFitModel approach) would be worth evaluating with the 30-pair labeled set as the training signal.

### Use post-scoring adjustment

Instead of tuning the category/layer adjustment (+0.05/-0.05) by hand, for a larger labeled dataset, a learned weight for each metadata signal (category, layer, prompt_id prefix) could be found via logistic regression on the labeled pairs — replacing hand-tuned constants with data-driven coefficients. This would also better handle cross-category pairs that are nonetheless genuine duplicates, as confirmed in the manual validation cases.

### Use an LLM to generate merge suggestions

The current merge suggestion is mechanical — it lists the union of template variables across cluster members and recommends consolidation. A more useful approach would pipe the full content of each duplicate prompt into an LLM (e.g. Claude or GPT-4) with instructions to produce a single unified prompt that preserves the intent of all versions, reconciles variable names, and retains the strongest phrasing from each. This would give authors a concrete draft to review rather than a list of variable names to reconcile manually.

### Structured logging and metrics storage

Operational data — embedding latencies, cluster counts per threshold, search hit rates, cache rebuild times — is currently only visible in stdout during the request lifecycle. Writing these metrics to a structured log file (JSON lines or similar) on each API call would enable post-hoc analysis: tracking how similarity distributions shift as the prompt library grows, comparing clustering results across threshold changes over time, and identifying slow embedding batches. This data would also feed into the automated evaluation system described below.

### Automated evaluation pipeline

Threshold tuning and clustering quality are currently validated by running eval scripts manually after code changes. A better approach would be a single `uv run evaluate` command that seeds the test prompts, runs the service, executes all eval suites, and produces a summary report with pass/fail thresholds. This could be integrated into CI so that any code change that degrades pair F1 below 99% or introduces over-merge cases is caught before merge.

### User feedback collection

The Streamlit UI currently displays results but provides no way to capture whether they were useful. Adding a thumbs-up/thumbs-down control on each search result and duplicate cluster card would create a lightweight feedback loop. Storing these ratings alongside the prompt pair and similarity score would build a growing labeled dataset that could replace the current 30-pair calibration set, inform threshold adjustments with real usage data, and surface prompt categories where the model consistently under- or over-scores.

---

## Assumptions Made

- **Prompt library is static enough for batch embedding.** The service re-embeds on demand but does not stream updates. If prompts are edited at high frequency, `regenerate_all` would need to be triggered automatically on write.

- **The 30-pair evaluation set is a small baseline, not a definitive benchmark.** It was assembled in the interest of time to get a comparative signal across models and to establish a starting threshold. 30 pairs is sufficient to detect obvious model differences but too small to draw precise conclusions about threshold boundaries or recall rates. In practice, a labeled set of at least 100 pairs — covering more category combinations, edge cases, and writing styles — would be needed before treating the threshold as production-ready.

- **False positives are more costly than false negatives for duplicate detection.** The 0.85 threshold was set to favour precision (99.5%) over recall for the duplicates endpoint. If the use case shifts to building a human review queue, a lower threshold would be more appropriate.

- **Template variable names convey semantic meaning.** The normalisation heuristic `appointment_date → the appointment date` assumes variable names are descriptive. Cryptic names like `{{v1}}` or `{{x}}` would produce unhelpful expansions and skew embeddings.