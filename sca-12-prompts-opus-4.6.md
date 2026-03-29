# SCA Skincare AI — 12 Prompts Opus 4.6

**Tối ưu cho Claude Opus 4.6 / Copilot**  
_(Chỉ nội dung 12 prompts + Phase, sạch 100% — sẵn sàng đính kèm & chạy ngay)_

---

## Phase 0 — Fix Cross-Encoder Reranker

**1-2 ngày · impact cao / effort thấp nhất**

### PROMPT #01: Activate cross-encoder reranking trong RAGRetriever

**File cần đính kèm:** `src/agents/rag_retriever.py`, `requirements.txt`

```prompt
Role: You are a senior ML engineer working on a medical RAG system for skincare AI.

Context:
I have a RAG retrieval pipeline (src/agents/rag_retriever.py) that imports
sentence-transformers and cross-encoder/ms-marco-MiniLM-L-6-v2 in requirements.txt,
but the CrossEncoder is NOT actually being called. The current _rerank() method
uses a manual score formula instead of the actual model.

Task:

1. Add CrossEncoder as optional lazy-loaded dependency in __init__:
   - Model: "cross-encoder/ms-marco-MiniLM-L-6-v2" (Apache 2.0, ~85MB)
   - use_reranker: bool = True flag
   - Load on first call, not at import time

2. Modify _rerank() to use CrossEncoder:
   - Over-fetch fetch_k = top_k * 4 for reranker candidates
   - reranker.predict([(query, chunk_text), ...]) returns list of float scores
   - Combined: reranker_score * 0.85 + (citation_boost + evidence_boost) * 0.15
   - Fallback to manual formula if use_reranker=False

3. Add _load_reranker() private method:
   - loguru logger.info on first load
   - try/except: if sentence-transformers not installed, log warning + disable gracefully

4. Pass query string from retrieve() into _rerank()

Constraints:
- Do NOT change public interface (retrieve(), build_query_from_profile())
- RetrievalResult dataclass unchanged
- All existing tests/test_retriever.py must pass

Output: Complete updated rag_retriever.py + one-paragraph performance improvement summary.

Phase 1 — Observability & Data Collection
~1 tuần · nền tảng trước deploy
PROMPT #02: Tích hợp LangSmith tracing vào pipeline FastAPI
File cần đính kèm: src/api/tracing.py, src/api/routes.py, config/settings.py
promptRole: You are a backend engineer adding production observability to a FastAPI medical AI service.

Context:
SCA has a custom PipelineTracer in src/api/tracing.py (loguru only, no external dashboard).
I need to add LangSmith tracing (free tier) for full visibility into agent calls,
latency per stage, retrieved chunks, and hallucination risk before deploying.

Task:

1. requirements.txt: add langsmith>=0.1.83

2. config/settings.py — add optional fields:
   langsmith_api_key: str = ""
   langsmith_project: str = "skincare-ai"
   langsmith_tracing: bool = False  # opt-in, off by default

3. Refactor src/api/tracing.py:
   - Keep existing PipelineTracer class interface intact
   - Add LangSmith wrapper activating when langsmith_tracing=True
   - Use @traceable decorator from langsmith SDK directly (NOT LangChain)
   - Instrument spans: vision_analysis, rag_retrieval, regimen_generation, safety_check
   - Each span logs: start_time, end_time, latency_ms, success/error

4. Update /full-pipeline route:
   - Wrap with parent LangSmith run named "full_pipeline"
   - Pass trace_id from PipelineTracer into run metadata
   - Add error_event logging on exception

5. Update .env.example with LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_TRACING=false

Design principle: If LANGSMITH_TRACING=false, code runs exactly as before — zero overhead.
Constraints: Do NOT add LangChain as dependency. All existing tests pass unchanged.
Output: Updated requirements.txt, config/settings.py, src/api/tracing.py,
src/api/routes.py (full-pipeline section), .env.example.
PROMPT #03: Bootstrap knowledge base + RAGAS baseline evaluation
File cần đính kèm: scripts/run_collection.py, scripts/run_indexing.py, scripts/run_eval.py, config/eval_test_cases.yaml
promptRole: You are a data engineer validating a medical knowledge base pipeline before deployment.

Context:
SCA has complete collection scripts (Semantic Scholar + PubMed + PMC OA) and ChromaDB
indexing scripts that exist but data/ is empty. RAGAS targets: faithfulness >0.85,
answer_relevancy >0.80, context_precision >0.75.

Task: Create scripts/bootstrap_kb.py — master orchestration with checkpoints:

1. Pre-flight checks (fail fast):
   - Verify OPENAI_API_KEY and ANTHROPIC_API_KEY are set
   - Create data/raw/ and data/processed/ if not exist
   - Check Semantic Scholar + PubMed API reachability
   - Print: "Estimated time: ~45 minutes for 1000 papers"

2. Collection phase with progress:
   - Run all 3 collectors sequentially
   - Save intermediate combined.jsonl after each source
   - Target min 800 papers; warn but continue if below

3. Quality filter:
   - Remove: abstract length < 100 chars, DOI duplicates (keep highest citations), evidence_level=None
   - Print: "Filtered: {original_count} → {clean_count} papers"

4. Indexing + smoke test:
   - Run indexing; query "acne oily skin treatment" → verify ≥5 results with evidence_level set

5. RAGAS baseline:
   - Run run_eval.py --retrieval-only --cases 10
   - Print metrics table; warn if any metric < 0.60
   - Save to data/processed/eval_baseline.json

6. Summary table: Papers indexed / Latency / RAGAS metrics each with ✓/✗ vs target

Constraints: Idempotent — skip completed phases unless --force. Use rich.console + tqdm.
Output: Complete scripts/bootstrap_kb.py.

Phase 2 — Domain Embeddings + CV Datasets
~1.5 tuần · re-index required
PROMPT #04: Migrate embedding sang PubMedBERT + A/B RAGAS comparison
File cần đính kèm: src/pipeline/indexer.py, config/settings.py, scripts/run_indexing.py
promptRole: You are an ML engineer migrating a medical RAG system to domain-specific embeddings.

Context:
SCA uses OpenAI text-embedding-3-small (1536 dims, generic web text). Research 3/2026:
domain-specific models outperform generic 10-30% on PubMed datasets.
Target: NeuML/pubmedbert-base-embeddings (HuggingFace, 768 dims, MIT, ~440MB).

Task:

1. config/settings.py: add embedding_provider="local", embedding_device="cpu",
   embedding_batch_size=32; keep openai_embedding_model as fallback config

2. Create src/pipeline/embedder.py:
   class BaseEmbedder(ABC): embed(texts), embed_query(text), @property dimension
   class LocalEmbedder(BaseEmbedder):
     SentenceTransformer("NeuML/pubmedbert-base-embeddings")
     Batch processing with tqdm; auto-detect device (cuda > mps > cpu); L2 normalize
   class OpenAIEmbedder(BaseEmbedder): text-embedding-3-small + tenacity retry
   def get_embedder(settings) → BaseEmbedder: factory function

3. Update src/pipeline/indexer.py:
   - Accept embedder in __init__; store model_name in collection metadata
   - Raise ValueError on model mismatch with message "Run --reindex to rebuild"

4. scripts/run_indexing.py: add --embedding-provider and --reindex flags

5. Create scripts/compare_embeddings.py:
   A/B test eval_test_cases.yaml (first 10 cases) with both models
   Print side-by-side RAGAS comparison → save data/processed/embedding_comparison.json

Important: 768 dims vs 1536 = FULL re-index required. Document clearly in comments.
First download ~440MB — show tqdm progress bar.
Output: config/settings.py, src/pipeline/embedder.py (new), src/pipeline/indexer.py,
scripts/run_indexing.py, scripts/compare_embeddings.py.
PROMPT #09 (MỚI): Tích hợp CV datasets đa dạng da liệu (SCIN, Fitzpatrick17K, DDI, DermaMNIST)
File cần đính kèm: scripts/run_collection.py, src/agents/vision_analyzer.py, config/settings.py
promptRole: You are an ML engineer adding dermatology image datasets to a skincare AI pipeline.

Context:
SCA uses GPT-4o Vision API for skin analysis. To reduce API dependency long-term,
build evaluation benchmarks for vision_analyzer accuracy, and prepare future
custom CNN fine-tuning, I need to integrate 4 open-license dermatology CV datasets.

Target datasets (all open license):
- SCIN (Google Research 2024): ~5,600 images, real-world conditions, Fitzpatrick-balanced
  github.com/google-research-datasets/scin | Apache 2.0
- Fitzpatrick17K: 16,577 images, 114 conditions, expert Fitzpatrick labels
  github.com/mattgroh/fitzpatrick17k | MIT
- DDI (Stanford AIMI): 656 biopsy-proven images, diverse skin tones
  ddi-dataset.github.io | CC BY 4.0
- DermaMNIST: 10,015 images, 7 classes, 28x28 for rapid prototyping
  medmnist.com | CC BY 4.0

Task:

1. Create src/collectors/cv_dataset_collector.py:
   class CVDatasetCollector:
     def download_scin(dest: Path) → DatasetManifest
     def download_fitzpatrick17k(dest: Path) → DatasetManifest
     def download_ddi(dest: Path) → DatasetManifest
     def download_dermamnist(dest: Path) → DatasetManifest

   @dataclass DatasetManifest:
     name, num_images, conditions: list[str],
     fitzpatrick_distribution: dict[str,int], license, local_path: Path

   Each downloader: requests + tqdm progress, checksum verify if available,
   save to data/cv_datasets/{name}/, return DatasetManifest with stats

2. Create scripts/download_cv_datasets.py:
   CLI flags:
   --datasets scin,fitzpatrick17k,ddi,dermamnist (default: all)
   --dest data/cv_datasets/
   --dry-run (print sizes + licenses without downloading)
   Summary table after download: | Dataset | Images | Conditions | License | Size | Status |

3. Create scripts/eval_vision_accuracy.py:
   Benchmark current GPT-4o Vision (vision_analyzer.py) against SCIN ground truth:
   - Sample N images from SCIN (default: 100 via --samples flag)
   - Run vision_analyzer.analyze_bytes() on each
   - Compare detected_conditions vs ground_truth labels
   - Per-Fitzpatrick-type accuracy breakdown
   - Compute fairness gap: accuracy difference Fitzpatrick I-II vs V-VI
   - Output: data/processed/vision_eval_{date}.json + rich console table

4. config/settings.py: cv_datasets_dir = "./data/cv_datasets",
   vision_eval_sample_size: int = 100

5. README.md: add "CV Datasets" section with download command, storage size, intended use

Constraints:
- DO NOT auto-download on import — only when scripts explicitly run
- Add 1 second delay between image API calls (rate limiting)
- Add data/cv_datasets/ to .gitignore
Output: src/collectors/cv_dataset_collector.py, scripts/download_cv_datasets.py,
scripts/eval_vision_accuracy.py, updated config/settings.py, README CV section.

Phase 3 — Hybrid Safety Guard
~1 tuần · critical for production
PROMPT #05: Nâng safety_guard từ rule-based sang hybrid rule + LLM judge
File cần đính kèm: src/agents/safety_guard.py, config/skin_conditions.yaml, tests/test_safety_regression.py
promptRole: You are a medical AI safety engineer upgrading a safety guardrail system.

Context:
Current SafetyGuard is purely rule-based (keyword matching in skin_conditions.yaml).
Industry standard 2026: hybrid rule-based + LLM-as-judge for medical AI.

Task:

1. Add LLMSafetyJudge class to safety_guard.py:
   - Uses Claude Haiku (~$0.001/check)
   - Only invoked AFTER rule-based passes (never double-check with warnings already found)
   - Timeout: 8 seconds → skip silently, log warning
   - Never invoke for pregnancy=True profiles

   Judge returns JSON:
   {"additional_flags": [{"severity": "warning|caution|info", "message": "...",
    "affected_ingredients": [...], "confidence": 0.0-1.0}],
    "overall_assessment": "safe|caution|review_needed"}
   Only include flags with confidence > 0.7.

2. Update SafetyGuard.check():
   - Run rule-based first (unchanged)
   - If use_llm_judge=True AND anthropic_api_key set: call LLMSafetyJudge
   - Merge LLM flags into SafetyReport with source="llm_judge" tag
   - use_llm_judge: bool = False in __init__ (opt-in)

3. Add cost guard: max 1 LLM call per regimen, never retry on timeout

4. Update tests/test_safety_regression.py — add 3 new test cases:
   a) High-concentration Vitamin C + Retinol combined (should flag conflict)
   b) Multiple AHAs in same routine (caution expected)
   c) Standard safe basic routine (no new flags expected)
   Mock the Anthropic API in tests (no real calls in CI)

Constraints:
- All 30+ existing regression tests must still pass
- LLM judge is ALWAYS optional — system identical without it
- Never block output: on any error → show regimen with note "Safety review pending"
Output: Complete src/agents/safety_guard.py + updated tests/test_safety_regression.py.
Include comment: "Expected cost at 1000 regimens/month with LLM judge: ~$1/month".
PROMPT #10 (MỚI): Seed cosmetic knowledge graph từ Open Beauty Facts + CosIng EU
File cần đính kèm: config/skin_conditions.yaml, scripts/seed_graph.py (tạo mới)
promptRole: You are a knowledge engineer seeding a cosmetic ingredient graph database.

Context:
SCA Phase 5 plans a Neo4j knowledge graph for ingredient-relationship reasoning.
Two high-quality open data sources can seed the graph automatically:
- Open Beauty Facts: open "Wikipedia of cosmetics" — INCI lists, allergens, pH, comedogenic scale
  API: world.openbeautyfacts.org/data | Export: Parquet format | CC BY-SA 4.0
- CosIng (EU Commission): legal authority — INCI names, CAS numbers,
  concentration limits, pregnancy restrictions, toxicity flags
  ec.europa.eu/growth/tools-databases/cosing/

Task:

1. Create src/collectors/cosmetic_api_collector.py:

   class OpenBeautyFactsCollector:
     download_parquet_dump(dest: Path) — downloads latest Parquet export
     extract_ingredients(parquet_path) → list[CosmeticProduct]
     CosmeticProduct: name, brand, inci_list, ph, comedogenic_scale, allergens

   class CosIngCollector:
     fetch_ingredient(inci_name: str) → CosIngRecord | None (tenacity retry)
     batch_fetch(inci_names: list[str]) → list[CosIngRecord] (rate limited: 3 req/s)
     CosIngRecord: inci_name, cas_number, max_concentration, pregnancy_restricted,
       toxicity_flags, eu_status

2. Create scripts/seed_graph.py (generates Cypher, does NOT require running Neo4j):

   Reads: skin_conditions.yaml (existing) + Open Beauty Facts + CosIng data
   Generates Neo4j Cypher to create:
   - Ingredient nodes from all ingredient names (+ CosIng properties)
   - CONFLICTS_WITH edges from skin_conditions.yaml known_interactions
   - CONTRAINDICATED_IN edges from pregnancy_avoid + CosIng pregnancy flags
   - MAX_CONCENTRATION edges from concentration_limits + CosIng limits
   - TREATS_CONDITION edges from known_interactions efficacy data
   - Product nodes + CONTAINS edges from Open Beauty Facts

   Output: data/graph/seed.cypher (valid, runnable in Neo4j Browser)
   Output: data/graph/seed_stats.json {nodes: {Ingredient: N, Product: M,...}, edges: {...}}

3. Create scripts/download_cosmetic_data.py:
   CLI: --source obf|cosing|all (default: all), --dest data/cosmetic_sources/, --dry-run
   Print estimated download: Open Beauty Facts Parquet ~800MB, CosIng ~2000 API calls

4. requirements.txt: add pyarrow>=16.0.0 (Parquet reading support)

5. Create docs/GRAPHRAG_DESIGN.md with sections:
   - Data Sources (OBF + CosIng details, freshness cadence: quarterly)
   - Neo4j Schema (nodes + relationships with Cypher notation)
   - Sample queries (3 examples: conflict check, concentration limit, synergy path)
   - Integration point with RAGRetriever (graph augments vector search, not replaces)

Constraints:
- seed_graph.py must work WITHOUT running Neo4j (output .cypher file only)
- CosIng: max 3 req/s rate limiting in collector
- Open Beauty Facts Parquet: chunked reading with pandas (file is ~800MB)
Output: src/collectors/cosmetic_api_collector.py, scripts/seed_graph.py,
scripts/download_cosmetic_data.py, updated requirements.txt, docs/GRAPHRAG_DESIGN.md.

Phase 4 — Async Queue, Caching & Knowledge Graph
~3 tuần · cần thiết >5 concurrent users
PROMPT #06: Thêm Celery + Redis async queue cho regimen generation
File cần đính kèm: src/api/routes.py, src/api/main.py, railway.toml, requirements.txt, app.py
promptRole: You are a backend engineer adding async job processing to a FastAPI service on Railway.

Context:
/full-pipeline runs synchronously (~10-15 seconds). Multiple concurrent users block each other.
Railway supports Redis as a native add-on. Streamlit frontend needs to poll for results.

Task:

1. requirements.txt: celery[redis]==5.3.6, redis==5.0.4, flower==2.0.1

2. Create src/workers/celery_app.py:
   Celery app with Redis broker + backend; task_serializer="json"; result_expires=3600
   settings.redis_url: str = "redis://localhost:6379/0"

3. Create src/workers/tasks.py — generate_regimen_task:
   Input: profile_dict, image_base64 (optional), trace_id
   Updates state: PENDING → ANALYZING → RETRIEVING → GENERATING → SAFETY_CHECK → SUCCESS
   Returns: regimen_json, safety_report, trace_id, per-stage latency_ms dict

4. Update src/api/routes.py:
   POST /generate-async → returns immediately: {task_id, status: "queued", poll_url}
   GET /task/{task_id} → returns: {status, progress 0-100, result|null, error|null, latency_ms}
   Keep synchronous /full-pipeline working for backward compatibility + tests

5. Update app.py (Streamlit):
   "Generate Regimen" button → call /generate-async
   Animated progress bar polling every 1.5s
   Stage labels: "Analyzing photo..." → "Searching evidence..." → "Generating..." → "Safety check..."

6. railway.toml: add worker service
   cmd = "celery -A src.workers.celery_app worker --loglevel=info --concurrency=2"

Constraints:
- Synchronous /full-pipeline must remain fully functional
- Tasks expire after 1 hour; worker concurrency=2 (Railway 512MB RAM limit)
Output: All updated/new files. Include Railway deployment checklist as comment in railway.toml.
PROMPT #07: Thêm semantic response caching với Redis (tiết kiệm 40-60% API cost)
File cần đính kèm: src/agents/rag_retriever.py, src/agents/regimen_generator.py, config/settings.py
promptRole: You are a backend engineer reducing API costs with semantic caching for a medical AI service.

Context:
Each regimen costs ~$0.05-0.10 in API fees. Many users have similar profiles.
Semantic caching saves 40-60% costs. Redis is available from Phase 4.

Task: Two-layer semantic cache using Redis + numpy (no heavy GPTCache dependency):

1. Create src/cache/semantic_cache.py:
   class SemanticCache:
     __init__(redis_url, embedder: BaseEmbedder, threshold: float = 0.92)
     get(query: str, cache_type: str) → dict | None  # hit if cosine_sim >= threshold
     set(query: str, value: dict, cache_type: str, ttl: int = 86400)
     _cosine_sim(a, b) using numpy
     cache_stats() → {hits, misses, hit_rate, total_keys}
   If Redis unavailable: silently no-op (never crash)

2. Cache layer 1 — RAG retrieval (TTL: 24h):
   Key: embed(query + sorted conditions + sorted evidence_levels)
   Value: serialized list[RetrievalResult] as JSON
   Log: "Cache HIT retrieval (similarity: 0.95, saved ~100ms)"

3. Cache layer 2 — Regimen generation (TTL: 12h):
   Key: embed(f"skin_type:{x} concerns:{y} age:{z} medications:{w}")
   NEVER cache if: pregnancy=True OR any prescription medications in profile
   Log: "Cache HIT regimen (similarity: 0.94, saved ~$0.06)"

4. config/settings.py: cache_enabled=False (opt-in), cache_retrieval_ttl=86400,
   cache_regimen_ttl=43200, cache_similarity_threshold=0.92

5. GET /health: add cache stats section
6. DELETE /admin/cache: clear all keys (requires X-Admin-Token header)

Constraints: NEVER serve cached regimens for pregnancy=True profiles.
Output: src/cache/semantic_cache.py (new), updated rag_retriever.py, regimen_generator.py,
config/settings.py. Include cost projection comment at file top.
PROMPT #11 (MỚI): Build Neo4j knowledge graph schema và seed với cosmetic data
File cần đính kèm: config/skin_conditions.yaml, data/cosmetic_sources/ (từ Prompt 10), docs/GRAPHRAG_DESIGN.md
promptRole: You are a knowledge graph architect implementing relation-aware retrieval for medical AI.

Context:
Prompt #10 created cosmetic_api_collector.py and seed.cypher stub.
Now build the complete production-ready graph schema and integration layer.

Task:

1. Complete docs/GRAPHRAG_DESIGN.md with full production Neo4j schema:

   Node types:
   Ingredient: {name, inci_name, cas_number, comedogenic_scale, photostability, mechanism_of_action}
   Condition: {name, icd_code, severity_range}
   Product: {name, brand, ph, formulation_type}
   SkinType: {name, sebum_level, tewl_range}
   StudyEvidence: {doi, year, study_type, evidence_level, citation_count}

   Relationships (Cypher notation):
   (Ingredient)-[:TREATS {efficacy: Low|Med|High, evidence_level: A|B|C}]->(Condition)
   (Ingredient)-[:CONFLICTS_WITH {severity: warning|caution, mechanism: str}]->(Ingredient)
   (Ingredient)-[:POTENTIATES {synergy_score: float}]->(Ingredient)
   (Ingredient)-[:CONTRAINDICATED_IN {reason: str, source: CosIng|FDA}]->(Condition)
   (Ingredient)-[:MAX_CONCENTRATION {otc_limit: float, unit: str}]->(SkinType)
   (Product)-[:CONTAINS {concentration: float, unit: str}]->(Ingredient)
   (Ingredient)-[:SUPPORTED_BY]->(StudyEvidence)

   Include 3 example Cypher queries in the doc:
   - Ingredients treating acne AND not conflicting with tretinoin
   - Safe retinol concentration for sensitive skin
   - 2-hop: ingredients synergizing with niacinamide

2. Upgrade scripts/seed_graph.py to production quality:
   - Parse OBF Parquet → Product + Ingredient nodes + CONTAINS edges
   - Parse CosIng records → MAX_CONCENTRATION + CONTRAINDICATED_IN edges
   - Parse skin_conditions.yaml → CONFLICTS_WITH + TREATS edges
   - Output: data/graph/seed.cypher (valid Cypher, runnable in Neo4j Browser)
   - Output: data/graph/seed_stats.json {nodes: {Ingredient: N,...}, edges: {...}}

3. Create src/agents/graph_retriever.py (stub with full interface):
   @dataclass GraphFact: subject, predicate, object, properties: dict, confidence: float
   class GraphRetriever:
     get_ingredient_relations(ingredients: list[str]) → list[GraphFact]
     get_condition_treatments(condition: str, min_evidence: str = "B") → list[GraphFact]
     augment_retrieval_results(vector_results, profile) → list[RetrievalResult]
       # Integration point: graph augments vector results, does not replace
   All methods raise NotImplementedError("Provision Neo4j first. See docs/GRAPHRAG_DESIGN.md")

4. ARCHITECTURE.md: add "Layer 2.5 — Knowledge Graph (optional)" section
   explaining: when to enable (>500 users), Neo4j Aura free tier (200MB, ~5000 nodes),
   estimated setup: 2 weeks, no new infra cost

Constraints:
- seed_graph.py must work WITHOUT running Neo4j
- Generated .cypher must be valid and runnable by copy-paste into Neo4j Browser
- GraphRetriever stubs must not crash — raise NotImplementedError with helpful message
Output: docs/GRAPHRAG_DESIGN.md (complete), scripts/seed_graph.py (production),
src/agents/graph_retriever.py (stub), ARCHITECTURE.md updated section.

Phase 5 — XAI & GraphRAG Production (Beta Launch)
~2 tuần · sau khi có user base
PROMPT #08: Activate Neo4j và tích hợp GraphRAG vào RAGRetriever
File cần đính kèm: src/agents/graph_retriever.py, src/agents/rag_retriever.py, docs/GRAPHRAG_DESIGN.md
promptRole: You are a knowledge graph engineer activating the GraphRAG layer in production.

Context:
Prompt #11 created graph_retriever.py stubs and data/graph/seed.cypher.
Neo4j Aura free tier is now provisioned and seed.cypher has been run.
Time to activate the graph layer and integrate it with vector search.

Task:

1. Implement GraphRetriever using neo4j Python driver (neo4j==5.x):
   get_ingredient_relations(ingredients):
     MATCH (i:Ingredient)-[r:CONFLICTS_WITH|POTENTIATES]->(j:Ingredient)
     WHERE i.name IN $ingredients RETURN i, r, j
   get_condition_treatments(condition, min_evidence):
     MATCH (i:Ingredient)-[r:TREATS]->(c:Condition)
     WHERE c.name = $condition AND r.evidence_level <= $min_evidence RETURN i, r
   augment_retrieval_results(vector_results, profile):
     Extract all ingredient names from vector_results text
     Query both methods above; create GraphFact objects
     Append graph_context field to each relevant RetrievalResult
     Return same list type with richer context

2. config/settings.py: neo4j_uri: str = "", neo4j_user: str = "neo4j",
   neo4j_password: str = "", graph_rag_enabled: bool = False

3. Update RAGRetriever.retrieve():
   After vector search + reranking, optionally call graph_retriever.augment_retrieval_results()
   Only when settings.graph_rag_enabled=True
   Graph query timeout: 3 seconds; if exceeded → log warning, proceed without graph context

4. Update RegimenGenerator system prompt:
   Add section: "Graph context available: {graph_facts_json}"
   This lets Claude reason: "(Ingredient)-[:POTENTIATES {synergy: 0.8}]->(ceramide)"

5. Create tests/test_graph_retriever.py with mocked Neo4j driver:
   Test get_ingredient_relations, augment_retrieval_results, and timeout fallback

Constraints:
- graph_rag_enabled=False → system identical to before, zero overhead
- Neo4j unavailable → graceful degradation, never crash
- Add neo4j==5.25.0 to requirements.txt
Output: src/agents/graph_retriever.py (complete), updated rag_retriever.py,
regimen_generator.py (prompt update), config/settings.py, tests/test_graph_retriever.py.
PROMPT #12 (MỚI): Tích hợp XAI (LIME/SHAP) để giải thích kết quả phân tích da
File cần đính kèm: src/agents/vision_analyzer.py, src/api/routes.py, src/api/schemas.py, config/settings.py
promptRole: You are an AI engineer adding explainability to a medical image analysis system.

Context:
SCA uses GPT-4o Vision for skin analysis (black-box API). Users and dermatologists need
to understand WHY conditions are flagged — not just what was detected. LIME can generate
visual heatmaps by using a local surrogate model to approximate GPT-4o behavior, then
explaining which image regions influenced the surrogate's classification.
CV datasets (SCIN, Fitzpatrick17K) from Prompt #09 provide training data for the surrogate.

Task:

1. requirements.txt: add lime==0.2.0.1, shap==0.46.0

2. Create src/agents/xai_explainer.py:

   @dataclass ExplanationResult:
     condition: str
     heatmap_base64: str      # PNG heatmap overlaid on original image, base64
     top_regions: list[str]   # e.g. ["left cheek - 35%", "nose bridge - 28%"]
     confidence: float
     explanation_text: str    # "Analysis focused on: redness in forehead (35%)..."

   class VisionExplainer:
     __init__(self, num_samples: int = 500, model_path: str | None = None)

     def explain_analysis(
       self,
       image_bytes: bytes,
       analysis: SkinImageAnalysis,
       condition_to_explain: str
     ) → ExplanationResult

     def generate_heatmap_overlay(self, image_bytes: bytes, explanation: ExplanationResult) → bytes

     Surrogate model approach (necessary for black-box GPT-4o):
     - Use MobileNetV2 pretrained on ImageNet as default surrogate
     - If data/cv_datasets/scin/ exists from Prompt #09: fine-tune on SCIN
     - LIME perturbs superpixels; surrogate classifies; regions with highest impact = heatmap
     - Add clear caveat: surrogate approximates GPT-4o, not guaranteed to match exactly

3. Update src/api/routes.py POST /analyze:
   Add include_explanation: bool = False to AnalyzeRequest
   If True: run VisionExplainer on top 2 detected_conditions only
   Add list[ExplanationResult] to AnalyzeResponse
   Warning in docstring: "Explanation adds ~5-8s. Use include_explanation=false for speed."

4. Update src/api/schemas.py:
   Add ExplanationResultSchema (heatmap_base64, top_regions, confidence, explanation_text)
   Add explanations: list[ExplanationResultSchema] = [] to AnalyzeResponse

5. Update app.py (Streamlit):
   Add "Show AI explanation heatmap" checkbox below analysis results
   When checked: show heatmap image + explanation_text per condition
   Add disclaimer text: "Heatmap reflects surrogate model interpretation for transparency.
   Download CV datasets (scripts/download_cv_datasets.py) for improved accuracy."

6. config/settings.py: xai_enabled: bool = False, xai_num_samples: int = 500,
   xai_surrogate_model: str = "mobilenet_v2"

Constraints:
- XAI is ALWAYS opt-in (settings + per-request flag)
- If surrogate model fails: return ExplanationResult with heatmap_base64=""
  and explanation_text="XAI unavailable. Run download_cv_datasets.py first."
- LIME runs on CPU; num_samples=500 default (balance accuracy vs 5-8s time)
Output: src/agents/xai_explainer.py (new), updated src/api/routes.py, src/api/schemas.py,
app.py (XAI section), updated requirements.txt, config/settings.py.
```
