# Architecture — Skincare AI System

## Design philosophy

**Three constraints for a solo developer:**
1. Ship fast — working prototype beats perfect design
2. Use managed services — don't operate infrastructure you don't need to
3. Replace components, not the whole system — each layer is independently swappable

---

## System layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4 — User Interface (Streamlit → Next.js)             │
│  app.py: photo upload, questionnaire, regimen display        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  Layer 3 — AI Agents                                        │
│  vision_analyzer.py   → GPT-4o Vision                      │
│  rag_retriever.py     → Hybrid semantic + metadata search   │
│  regimen_generator.py → Claude claude-sonnet-4-6 reasoning         │
│  safety_guard.py      → Rule-based contraindication check   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  Layer 2 — Knowledge Base                                   │
│  ChromaDB (dev) / Qdrant (prod)                             │
│  ~1,000–5,000 papers → 2,000–10,000 chunks                  │
│  text-embedding-3-small (1536 dims, cosine similarity)      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  Layer 1 — Data Pipeline                                    │
│  Collectors: Semantic Scholar API + PubMed E-utilities      │
│  Processing: metadata tagger → chunker → embedder           │
│  Storage:    data/raw/*.jsonl → data/processed/*.jsonl      │
└─────────────────────────────────────────────────────────────┘
```

---

## Data flow — single user request

```
User uploads photo + submits questionnaire
        │
        ▼
[VisionAnalyzer]
  GPT-4o Vision analyzes photo
  → SkinImageAnalysis (structured JSON)
        │
        ▼
[Profile Builder]
  Merge vision + questionnaire
  → unified skin_profile dict
        │
        ▼
[RAGRetriever]
  Build query from profile
  → ChromaDB/Qdrant hybrid search
  → Top-K chunks (evidence_level A/B/C)
        │
        ▼
[RegimenGenerator]
  System prompt + profile JSON + evidence context
  → Claude claude-sonnet-4-6 API call
  → Regimen JSON (validated Pydantic)
        │
        ▼
[SafetyGuard]
  Check pregnancy contraindications
  Check ingredient conflicts
  Check medication interactions
  Check severity escalation
  → SafetyReport with flags
        │
        ▼
[Streamlit UI]
  Display AM/PM routine
  Show evidence citations
  Show safety warnings
  Export JSON option
```

---

## Layer 2.5 — Knowledge Graph (optional)

A Neo4j knowledge graph sits between the vector knowledge base (Layer 2) and
the AI agents (Layer 3). It encodes **structured relational facts** that are
difficult to capture in unstructured text embeddings:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3 — AI Agents                                        │
│  (receives: RAG chunks + structured graph facts)            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  Layer 2.5 — Knowledge Graph (optional, Neo4j)             │
│  GraphRetriever: augment_retrieval_results()                │
│  src/agents/graph_retriever.py                             │
│  docs/GRAPHRAG_DESIGN.md                                   │
│                                                             │
│  Nodes:  Ingredient, Condition, Product                     │
│  Edges:  CONFLICTS_WITH, CONTRAINDICATED_IN,               │
│           TREATS_CONDITION, CONTAINS                        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│  Layer 2 — Vector Knowledge Base                            │
│  ChromaDB (dev) / Qdrant (prod)                             │
└─────────────────────────────────────────────────────────────┘
```

**Status:** Stub implementation — `GraphRetriever` methods raise `NotImplementedError`
until Neo4j is configured. The pipeline works without Neo4j.

**Setup:** See `docs/GRAPHRAG_DESIGN.md` for full schema, sample queries, and setup.

---

## Component swap guide

Each component can be replaced independently:

### Swap vector DB (local → cloud)
```python
# dev:
indexer = ChromaIndexer(persist_dir="./data/knowledge_base")

# prod:
indexer = QdrantIndexer(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

# Both expose same interface: .add(chunks), .query(text, filters, top_k)
```

### Swap embedding model
```python
# Cheaper:
embedding_model = "text-embedding-3-small"  # $0.02/1M

# More accurate:
embedding_model = "text-embedding-3-large"  # $0.13/1M

# NOTE: Changing embedding model requires re-indexing entire knowledge base
# All chunks must use the same model
```

### Swap reasoning LLM
```python
# Default (best quality):
reasoning_model = "claude-sonnet-4-6-20251101"

# Cheaper (faster, less nuanced):
reasoning_model = "claude-haiku-4-5-20251001"

# OpenAI alternative:
reasoning_model = "gpt-4o"   # requires switching client in regimen_generator.py
```

### Swap UI (Streamlit → Next.js)
The FastAPI backend (src/api/) is UI-agnostic.
When traffic justifies it, keep FastAPI backend, replace Streamlit with Next.js frontend.
All business logic stays in Python agents — never put logic in the UI layer.

---

## Database schema (Supabase / PostgreSQL)

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT now(),
    email TEXT UNIQUE
);

-- Skin profiles (one per consultation session)
CREATE TABLE skin_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    profile_json JSONB NOT NULL,           -- full profile dict
    skin_type TEXT,
    primary_concerns TEXT[],
    fitzpatrick TEXT
);

-- Generated regimens
CREATE TABLE regimens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID REFERENCES skin_profiles(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    regimen_json JSONB NOT NULL,           -- full Regimen pydantic output
    model_used TEXT,
    evidence_chunks_used INTEGER
);

-- Follow-up photos (for progress tracking)
CREATE TABLE follow_up_photos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    regimen_id UUID REFERENCES regimens(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    vision_analysis_json JSONB,            -- SkinImageAnalysis output
    week_number INTEGER                    -- which week of routine
);
```

---

## Scaling milestones

| Users | Action |
|-------|--------|
| 1–10 | ChromaDB local + Streamlit — zero infra cost |
| 10–50 | Migrate to Qdrant Cloud free tier, deploy FastAPI on Railway |
| 50–200 | Add Supabase for user profiles + regimen history |
| 200+ | Add rate limiting, queue regimen generation, consider Next.js UI |
| 1000+ | Fine-tune embedding model on dermatology corpus, add caching |

---

## Security checklist (pre-launch)

- [ ] API keys stored in environment variables, never in code
- [ ] Face images processed in-memory only, never persisted
- [ ] Supabase Row-Level Security enabled (users see only their data)
- [ ] Rate limiting on regimen generation endpoint
- [ ] Medical disclaimer visible on every regimen output
- [ ] Input sanitization on questionnaire text fields
- [ ] No PII logged in application logs
