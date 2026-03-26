# SKILL: Project Roadmap — Skincare AI System

## Purpose

This file is the **single source of truth** for the full implementation plan of the Skincare AI system.
It replaces the HTML roadmap (`skincare_ai_roadmap.html`) with a structured, AI-consumable format.

Read this file to:

- Understand what exists and what needs to be built
- Know the correct dependency order for tasks
- Get actionable prompts for each pending task
- See best-practice improvements queued for each component

For deep-dive into each domain, follow the references to dedicated SKILL files.

---

## System Overview

**Goal**: Evidence-based, personalized skincare regimen generator using RAG + Vision + LLM.

**Data flow**:

```
User Input (questionnaire + photo)
  → VisionAnalyzer (GPT-4o)              → skin image analysis
  → Profile Builder                       → merge vision + questionnaire
  → RAGRetriever (hybrid search)          → top-K evidence chunks from knowledge base
  → RegimenGenerator (Claude Sonnet 4)    → structured AM/PM regimen with citations
  → SafetyGuard (rule-based)              → pregnancy / drug / phototoxicity checks
  → Output (Streamlit UI or API)          → regimen + warnings + evidence grades
```

**Architecture layers**:

```
Layer 4: UI              — Streamlit (MVP) → Next.js (future)
Layer 3: AI Agents       — vision, retrieval, generation, safety
Layer 2: Vector DB       — ChromaDB (dev) → Qdrant (prod)
Layer 1: Data Pipeline   — collection, tagging, chunking, embedding
Layer 0: Knowledge Base  — Semantic Scholar + PubMed + PMC OA + clinical guidelines
```

**Key tech**: Python 3.12, ChromaDB, text-embedding-3-small, Claude Sonnet 4, GPT-4o Vision, FastAPI, Streamlit, Pydantic.
Full dependency list: `requirements.txt`.

---

## Implementation Status

### Legend

| Status           | Meaning                                                       |
| ---------------- | ------------------------------------------------------------- |
| **DONE**         | Code complete, tested                                         |
| **DONE+IMPROVE** | Working but needs best-practice improvements (details below)  |
| **READY**        | Code complete, needs execution (e.g., run collection scripts) |
| **PENDING**      | Not yet implemented                                           |

---

## Phase 0 — Foundation & Data Pipeline

### Task 0.1: Knowledge Harvesting

**Status**: DONE

**What exists**:

- `src/collectors/semantic_scholar.py` — `SemanticScholarCollector` with pagination, rate limiting, retry
- `src/collectors/pubmed.py` — `PubMedCollector` with esearch→efetch, XML parsing
- `src/collectors/pmc_oa.py` — `PMCOpenAccessCollector` with E-utilities, full-text extraction, section markers
- `scripts/run_collection.py` — CLI tool (--sources semantic_scholar,pubmed,pmc_oa, --max, --dry-run)
- `config/search_queries.yaml` — 30+ Semantic Scholar queries, 10+ PubMed queries, 15 PMC OA queries

**Remaining (nice-to-have)**:

1. Run `scripts/run_collection.py` to populate `data/raw/combined.jsonl`
2. Curate 20-30 clinical guideline documents (AAD acne, rosacea, eczema guidelines; Cochrane reviews) as markdown files in `data/raw/guidelines/`

**Depends on**: Nothing — start here.
**Deep dive**: `skills/SKILL_data_collection.md`

**Prompt (run existing collection)**:

```
Run scripts/run_collection.py with default settings.
Expected output: data/raw/combined.jsonl with ~1000 papers.
Verify: count papers per source, log any API errors.
```

**Prompt (add PMC OA collector)**:

```
Read skills/SKILL_data_collection.md for API patterns and quality filters.
Create src/collectors/pmc_oa.py extending BaseCollector.
Use PMC OA Web Service (https://www.ncbi.nlm.nih.gov/pmc/tools/oa-service-root/).
For each paper, extract sections: abstract, methods, results, conclusion.
Store section text in Paper.abstract field with section markers:
  [ABSTRACT] ... [METHODS] ... [RESULTS] ... [CONCLUSION] ...
This enables section-aware chunking in Task 0.3.
```

---

### Task 0.2: Metadata Schema

**Status**: DONE

**What exists**:

- `config/skin_conditions.yaml` — 10 condition categories, 20+ ingredient families, Fitzpatrick scale, interactions, pregnancy list, evidence levels

**What needs improvement**:

1. **INCI synonyms** — each ingredient maps to its synonym group (retinol → retinol, retinal, retinoic acid, tretinoin, adapalene). Required for allergen matching in safety checks.
2. **Phototoxicity flags** — mark ingredients that increase UV sensitivity (AHA, retinoids, benzoyl peroxide, vitamin C at high %). SafetyGuard needs this.
3. **Concentration limits** — regulatory OTC max per ingredient (retinol ≤1%, hydroquinone ≤2%, benzoyl peroxide ≤10%, glycolic acid ≤10% daily use). RegimenGenerator needs this for validation.
4. **Expanded pregnancy list** — add salicylic acid (oral/high-dose), hydroquinone, formaldehyde-releasing preservatives, essential oils (clary sage, rosemary, cinnamon).
5. **Expanded drug interactions** — add tetracyclines+retinoids, immunosuppressants+exfoliants, blood thinners+vitamin E, photosensitizing medications+UV-active ingredients.

**Depends on**: Nothing — parallel with 0.1.

**Prompt**:

```
Read config/skin_conditions.yaml.
Add these new sections:
1. inci_synonyms: mapping ingredient → [all INCI name variants]
2. phototoxicity: list of ingredients with UV sensitivity flag + SPF requirement note
3. concentration_limits: mapping ingredient → {otc_max: "X%", professional_max: "Y%", note: "..."}
4. Expand pregnancy_avoid with: salicylic_acid_oral, hydroquinone, essential_oils, formaldehyde_donors
5. Expand known_interactions.avoid_together with 5+ new pairs (see plan audit section D)
```

---

### Task 0.3: Chunking Pipeline

**Status**: DONE

**What exists**:

- `src/pipeline/chunker.py` — `PaperChunker` with 512-token sliding window, 64-token overlap
- Abstracts kept whole (high signal), title prepended to all chunks
- `Chunk` dataclass with full metadata + `to_chroma_dict()` for flat storage

**What needs improvement**:

1. **Section-aware chunking** — when full-text papers (from PMC OA) contain `[ABSTRACT]`, `[METHODS]`, `[RESULTS]`, `[CONCLUSION]` markers, chunk each section independently. This preserves PICO structure.
2. **Enriched chunk headers** — prepend study metadata to each chunk: `"{title} ({year}) | {study_type} | Evidence: {evidence_level}"` so chunks are self-contained during retrieval.

**Depends on**: Task 0.1 (needs full-text data format finalized).
**Deep dive**: `skills/SKILL_rag_pipeline.md`

**Prompt**:

```
Read src/pipeline/chunker.py.
Modify PaperChunker.chunk_paper() to:
1. Detect section markers [ABSTRACT], [METHODS], [RESULTS], [CONCLUSION] in paper.abstract
2. If markers present: split by section, chunk each section independently
3. If no markers: use existing abstract-only logic (backward compatible)
4. Change title prefix from "Title: {title}" to:
   "{title} ({year}) | {study_type} | Evidence: {evidence_level}"
Keep all existing tests passing.
```

---

### Task 0.4: Vector DB Setup

**Status**: DONE

- `src/pipeline/indexer.py` — `ChromaIndexer` (local, cosine HNSW) + `QdrantIndexer` (production)
- Both share identical interface: `.add(chunks)`, `.query(text, filters, top_k)`
- `scripts/run_indexing.py` — orchestrates tag→chunk→embed→index pipeline

No changes needed.

---

### Task 0.5: BM25 Sparse Index (NEW)

**Status**: DONE

**Why needed**: Medical queries contain precise terms (drug names, concentrations, INCI names) that dense embeddings may miss. BM25 catches exact keyword matches. Combined with dense via Reciprocal Rank Fusion (RRF), this is the standard for medical RAG.

**Implementation**:

- New file: `src/pipeline/bm25_index.py`
- Class `BM25Index` — builds BM25 index from chunk texts at indexing time
- Persists to disk (pickle or JSON) alongside ChromaDB data
- `.search(query, top_k)` returns chunk_ids + BM25 scores
- RRF fusion function: `rrf_fuse(dense_results, bm25_results, k=60) → merged_ranked_list`
- Library: `rank_bm25` (add to requirements.txt)

**Depends on**: Tasks 0.1, 0.3, 0.4 (needs chunked and indexed data).
**Deep dive**: `skills/SKILL_rag_pipeline.md`

**Prompt**:

```
Create src/pipeline/bm25_index.py with:
1. BM25Index class wrapping rank_bm25.BM25Okapi
2. .build(chunks: list[Chunk]) — tokenize texts, build index, persist to disk
3. .search(query: str, top_k: int) → list[dict] with chunk_id + score
4. .load(path) classmethod to reload from disk
5. rrf_fuse(dense_results, sparse_results, k=60) → merged list
   Formula: RRF_score = Σ 1/(k + rank_i) for each result list
Add rank_bm25 to requirements.txt.
Update src/pipeline/__init__.py exports.
```

---

## Phase 1 — RAG Core + Skin Profiling

### Task 1.1: Hybrid Retrieval

**Status**: DONE

**What exists**:

- `src/agents/rag_retriever.py` — `RAGRetriever` with semantic search + re-ranking
- Scoring: `0.7 × semantic + 0.2 × citation_boost + 0.1 × evidence_boost`
- Over-fetch 2× then re-rank, fallback without filters
- `build_query_from_profile()` — concatenates profile fields

**What needs improvement**:

1. **Enable skin_conditions filter** — currently commented out in `_build_filters()`. Uncomment and activate.
2. **Cross-encoder reranking** — replace rule-based formula with a cross-encoder model (e.g., `BAAI/bge-reranker-v2-m3` or `cross-encoder/ms-marco-MiniLM-L-6-v2`). Run on top-2K candidates, much higher precision than fixed-weight formula.
3. **Multi-query expansion** — generate 2-3 query variants from profile (e.g., "retinol anti-aging wrinkles", "vitamin A derivative fine lines RCT"). Retrieves broader evidence.
4. **BM25 fusion** — integrate BM25Index.search() results via RRF with dense results before reranking.

**Depends on**: Tasks 0.4, 0.5.
**Deep dive**: `skills/SKILL_rag_pipeline.md`

**Prompt**:

```
Read src/agents/rag_retriever.py.
Improvements:
1. In _build_filters(): add skin_conditions filter using $contains operator
2. Add _expand_query(profile) → list[str] that generates 2-3 query variants
3. Add _cross_encoder_rerank(query, candidates) using sentence-transformers CrossEncoder
4. In retrieve(): fuse BM25 + dense via RRF, then cross-encoder rerank top candidates
5. Keep fallback logic (retry without filters if empty)
Add sentence-transformers to requirements.txt.
```

---

### Task 1.2: Skin Questionnaire

**Status**: DONE

- `app.py` — Streamlit form with age, skin type, Fitzpatrick, concerns, allergies, medications
- No changes needed.

---

### Task 1.3: LLM Prompt Engineering

**Status**: DONE

**What exists**:

- `src/agents/regimen_generator.py` — `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` hardcoded
- Claude Sonnet 4 integration, Pydantic validation, markdown fence stripping

**What needs improvement**:

1. **Extract prompts to config files** — `config/prompts/regimen_system.txt` and `config/prompts/regimen_user.txt` for version tracking
2. **Add constraints** to system prompt:
   - Maximum 7 steps per routine (AM or PM)
   - Stay within OTC concentration limits (reference `config/skin_conditions.yaml` concentration_limits)
   - If severe condition detected, first recommendation must be "consult dermatologist"
3. **Load prompts at runtime** from config files, fallback to hardcoded if files missing

**Depends on**: Task 1.1 (needs working retrieval for testing).
**Deep dive**: `skills/SKILL_regimen_generation.md`

**Prompt**:

```
Read src/agents/regimen_generator.py.
1. Create config/prompts/regimen_system.txt with current SYSTEM_PROMPT content + new constraints
2. Create config/prompts/regimen_user.txt with current USER_PROMPT_TEMPLATE content
3. Modify RegimenGenerator.__init__() to load from config files with fallback
4. Add to SYSTEM_PROMPT: max 7 steps per routine, respect concentration limits, lead with dermatologist referral for severe conditions
```

---

### Task 1.4: Vision Analysis

**Status**: DONE

- `src/agents/vision_analyzer.py` — `VisionAnalyzer` with GPT-4o, zone analysis, profile merge
- No changes needed.
- **Deep dive**: `skills/SKILL_vision_analysis.md`

---

## Phase 2 — Safety, MVP UI & End-to-End

### Task 2.1: Safety Guardrails

**Status**: DONE

**What exists**:

- `src/agents/safety_guard.py` — `SafetyGuard` with 4 checks:
  1. `_check_pregnancy` — flags ingredients in `pregnancy_avoid` list
  2. `_check_ingredient_conflicts` — flags pairs from `known_interactions.avoid_together`
  3. `_check_severity_escalation` — dermatologist referral for severe/cystic acne, psoriasis, lupus
  4. `_check_medication_interactions` — isotretinoin+retinoids, antibiotics+BP

**Critical gaps to fix** (all are safety-relevant for medical AI):

1. **Phototoxicity check (NEW)** — Flag AHA, retinoids, benzoyl peroxide, high-% vitamin C with mandatory SPF warning. Any routine containing photosensitizers MUST include SPF step or warning.
2. **Age-tier safety (NEW)** — Teens (<16): limit retinoid concentration (no tretinoin >0.025%, no hydroquinone). Elderly (>65): recommend lower concentrations, barrier-focused approach.
3. **Allergen-ingredient matching (FIX)** — Profile allergies are loaded but NEVER checked against ingredient lists. Cross-reference using INCI synonym groups from expanded taxonomy.
4. **Concentration limit validation (NEW)** — Warn if recommended concentration exceeds OTC regulatory max (e.g., retinol >1%, hydroquinone >2%). Uses `concentration_limits` from expanded taxonomy.
5. **Expanded drug interactions** — Add: tetracyclines+retinoids (photosensitivity risk), immunosuppressants+active exfoliants (barrier compromise), blood thinners+vitamin E (bleeding risk), lithium+acne (lithium causes acne, avoid drying agents).
6. **Expanded pregnancy list** — Add: salicylic acid (oral/high-dose), hydroquinone, essential oils, formaldehyde-releasing preservatives.

**Depends on**: Task 0.2 (needs expanded taxonomy data).

**Prompt**:

```
Read src/agents/safety_guard.py and config/skin_conditions.yaml.
Add these new check methods to SafetyGuard:
1. _check_phototoxicity(ingredients, regimen, report) — if photosensitizer present and no SPF step in AM routine, add warning
2. _check_age_safety(ingredients, profile, report) — age <16 or >65 checks
3. _check_allergens(ingredients, profile, report) — match profile.allergies against INCI synonym groups
4. _check_concentration_limits(regimen, report) — parse concentration_range in RoutineStep, compare against limits
5. Add new pairs to _check_medication_interactions
Wire all new checks into check() method.
Add corresponding taxonomy data to skin_conditions.yaml.
```

---

### Task 2.2: Streamlit MVP

**Status**: DONE

- `app.py` — Full UI with questionnaire, photo upload, regimen display, JSON export
- No changes needed.

---

### Task 2.3: Regimen Output Format

**Status**: DONE

- `Regimen` and `RoutineStep` Pydantic models in `src/agents/regimen_generator.py`
- No changes needed.

---

### Task 2.4: FastAPI Backend

**Status**: DONE

**Implementation**:

- New directory: `src/api/`
- Endpoints:
  - `POST /analyze` — accepts image, returns `SkinImageAnalysis`
  - `POST /retrieve` — accepts profile, returns evidence chunks
  - `POST /generate` — accepts profile + evidence, returns `Regimen`
  - `POST /safety-check` — accepts regimen + profile, returns `SafetyReport`
  - `POST /full-pipeline` — end-to-end: image + questionnaire → regimen + safety report
- Uses existing agent classes as backend logic

**Depends on**: Tasks 1.1, 1.3, 1.4, 2.1.

**Prompt**:

```
Read docs/ARCHITECTURE.md for the data flow.
Create src/api/main.py with FastAPI app.
Create src/api/routes.py with endpoints: /analyze, /retrieve, /generate, /safety-check, /full-pipeline.
Create src/api/schemas.py with Pydantic request/response models.
Wire agents: VisionAnalyzer, RAGRetriever, RegimenGenerator, SafetyGuard.
Add health check endpoint at /health.
```

---

### Task 2.5: Deploy

**Status**: DONE

**What exists**:

- `Dockerfile` — FastAPI backend (python:3.12-slim, uvicorn)
- `Dockerfile.streamlit` — Streamlit UI (separate container)
- `railway.toml` + `railway.streamlit.toml` — Railway 2-service deploy configs
- `.github/workflows/ci.yml` — CI/CD: pytest → Railway deploy (api + ui)
- `.dockerignore` — optimized build context
- `config/settings.py` — env-configurable CORS origins, project root
- Qdrant Cloud for production vector DB
- Environment variable management via `.env` + Railway env vars

**Depends on**: Task 2.4.

---

## Phase 3 — Evaluation & Quality

### Task 3.1: RAG Evaluation

**Status**: DONE

**What exists**:

- `config/eval_test_cases.yaml` — 30 test cases: 10 core profiles + 20 edge cases (medications, multi-ethnic, elderly, pediatric, complex combos, stress tests)
- `scripts/run_eval.py` — automated evaluation script with two modes:
  - `--retrieval-only`: fast keyword recall, precision@K, nDCG@K (no LLM calls)
  - Full pipeline: retrieval + generation + safety check, ingredient recall, forbidden violations, safety accuracy
  - Filters: `--cases N`, `--ids 1,3,11`, `--difficulty hard`
  - Output: `data/processed/eval_results.json` + rich console tables
- `docs/EVALUATION.md` — updated with CLI instructions, baseline results table

**Depends on**: Task 0.1 (needs populated knowledge base to get meaningful scores).
**Deep dive**: `docs/EVALUATION.md`

---

### Task 3.2: Safety Regression Tests (NEW)

**Status**: DONE

- Automated pytest suite covering ALL safety scenarios
- Must pass 100% before any deployment
- Covers: pregnancy, isotretinoin, phototoxicity, age tiers, allergens, concentration limits, drug interactions, severity escalation

**Depends on**: Task 2.1.

**Prompt**:

```
Read tests/test_collectors.py for existing test patterns (MockStep, MockRegimen).
Create tests/test_safety_regression.py with:
1. test_pregnancy_retinoids — retinol flagged during pregnancy
2. test_pregnancy_salicylic — high-dose salicylic acid flagged
3. test_pregnancy_safe_routine — niacinamide + azelaic acid passes
4. test_isotretinoin_retinoid — isotretinoin + topical retinoid blocked
5. test_tetracycline_retinoid — tetracycline + retinoid flagged
6. test_phototoxicity_no_spf — AHA without SPF step triggers warning
7. test_phototoxicity_with_spf — AHA with SPF step passes
8. test_teen_high_retinoid — age 14 + tretinoin 0.05% flagged
9. test_elderly_gentle — age 70 gets lower concentration warning
10. test_allergen_fragrance — fragrance allergy + fragrance ingredient flagged
11. test_concentration_limit — retinol 2% exceeds OTC max
12. test_severity_cystic_acne — dermatologist referral triggered
13. test_clean_regimen — no-issue regimen passes all checks
All tests must run without API keys.
```

---

### Task 3.3: Citation Grounding Check (NEW)

**Status**: DONE

- Post-generation verification: check each cited paper in the regimen against retrieved chunks
- Lightweight string matching or LLM verification
- Flag ungrounded citations (hallucinated references)

**Depends on**: Tasks 1.1, 1.3.

---

### Task 3.4: Observability (NEW)

**Status**: DONE

- Pipeline tracing with trace IDs per request
- Log retrieval scores, generation inputs, safety flags
- LRU cache for query embeddings
- Integration: Langfuse or custom structured logging

**Depends on**: Task 2.4.

---

## Phase 4 — Post-MVP Iteration

- **Follow-up tracking** — user photo comparison over time, routine adjustment based on progress
- **KB auto-refresh** — weekly cron: collect new papers → filter → embed → upsert
- **UI upgrade** — Streamlit → Next.js + shadcn/ui when >10 active users
- **LlamaIndex decision** — currently installed (40+ packages) but barely used. Either leverage fully or remove to reduce surface area.
- **Fine-tuned embeddings** — train on dermatology corpus when >1000 users (text-embedding-3-small is adequate until then)

---

## Dependency Graph

```
0.1 Harvest ─────→ 0.3 Chunk ──→ 0.4 Index ──→ 0.5 BM25 ──→ 1.1 Retrieval ──→ 1.3 Prompts ──→ 3.1 Eval
     ║                                                              ║                 ║
0.2 Taxonomy ═══════════════════════════════════════════════→ 2.1 Safety ──→ 3.2 Safety Tests
(parallel)                                                                        ║
1.2 Questionnaire ──→ 1.4 Vision ──→ 2.2 UI ──→ 2.4 API ──→ 2.5 Deploy
(parallel)             (parallel)                                    ║
                                                              3.4 Observability
```

**Critical path**: 0.1 → 0.3 → 0.4 → 0.5 → 1.1 → 1.3 → 3.1 → Deploy

**Parallel tracks**:

- Track A: Data pipeline (0.1 → 0.3 → 0.4 → 0.5)
- Track B: Taxonomy + Safety (0.2 → 2.1 → 3.2)
- Track C: UI + API (1.2 → 1.4 → 2.2 → 2.4 → 2.5)

---

## Quick-Reference: File Registry

| File                                 | Purpose                         | Tasks    |
| ------------------------------------ | ------------------------------- | -------- |
| `src/collectors/semantic_scholar.py` | Semantic Scholar API collector  | 0.1      |
| `src/collectors/pubmed.py`           | PubMed E-utilities collector    | 0.1      |
| `src/collectors/pmc_oa.py`           | PMC Open Access collector (NEW) | 0.1      |
| `config/skin_conditions.yaml`        | Master taxonomy                 | 0.2, 2.1 |
| `config/search_queries.yaml`         | Search query library            | 0.1      |
| `src/pipeline/chunker.py`            | Paper → Chunk conversion        | 0.3      |
| `src/pipeline/indexer.py`            | ChromaDB/Qdrant embed+store     | 0.4      |
| `src/pipeline/bm25_index.py`         | BM25 sparse index (NEW)         | 0.5      |
| `src/agents/rag_retriever.py`        | Hybrid retrieval + reranking    | 1.1      |
| `src/agents/vision_analyzer.py`      | GPT-4o skin photo analysis      | 1.4      |
| `src/agents/regimen_generator.py`    | Claude regimen generation       | 1.3      |
| `src/agents/safety_guard.py`         | Safety checks                   | 2.1      |
| `config/prompts/regimen_system.txt`  | System prompt (NEW)             | 1.3      |
| `config/prompts/regimen_user.txt`    | User prompt template (NEW)      | 1.3      |
| `src/api/main.py`                    | FastAPI backend (NEW)           | 2.4      |
| `app.py`                             | Streamlit MVP                   | 2.2      |
| `scripts/run_collection.py`          | Paper collection CLI            | 0.1      |
| `scripts/run_indexing.py`            | Index pipeline CLI              | 0.4      |
| `scripts/run_eval.py`                | RAGAS evaluation script (NEW)   | 3.1      |
| `tests/test_collectors.py`           | Unit tests (13 passing)         | —        |
| `tests/test_safety_regression.py`    | Safety regression suite (NEW)   | 3.2      |
| `docs/ARCHITECTURE.md`               | System architecture             | —        |
| `docs/EVALUATION.md`                 | Eval methodology + test cases   | 3.1      |

---

## Cost Breakdown (Per Regimen)

| Component                | Cost          | Notes                                  |
| ------------------------ | ------------- | -------------------------------------- |
| Embedding query          | ~$0.000002    | text-embedding-3-small                 |
| Cross-encoder rerank     | ~$0.00        | Local model, no API cost               |
| BM25 search              | ~$0.00        | Local computation                      |
| Claude generation        | ~$0.003–0.006 | Sonnet 4, ~2K input + 2K output tokens |
| GPT-4o Vision            | ~$0.01–0.02   | Per image analysis (optional)          |
| **Total without vision** | **~$0.006**   |                                        |
| **Total with vision**    | **~$0.026**   |                                        |
