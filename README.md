# Skincare AI — Personalized Regimen System

AI-powered skincare consultation system built on scientific literature.
Generates evidence-based, personalized skincare regimens from peer-reviewed research.

## Architecture overview

```
User input (questionnaire + photo + medical history)
        ↓
  Skin Profile Builder
        ↓
  Hybrid RAG Retrieval  ←→  Knowledge Base (ChromaDB → Qdrant)
        ↓                         ↑
  LLM Reasoning Core        Scientific Papers
  (Claude / GPT-4o)         (Semantic Scholar + PubMed)
        ↓
  Safety Guardrail
        ↓
  Personalized Regimen (AM/PM + citations + evidence grade)
```

## Project structure

```
skincare-ai/
├── src/
│   ├── collectors/          # Data collection from APIs
│   │   ├── semantic_scholar.py
│   │   ├── pubmed.py
│   │   └── base_collector.py
│   ├── pipeline/            # Processing & indexing
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── indexer.py
│   ├── agents/              # AI reasoning
│   │   ├── skin_profiler.py
│   │   ├── rag_retriever.py
│   │   ├── regimen_generator.py
│   │   ├── vision_analyzer.py
│   │   └── safety_guard.py
│   └── api/                 # FastAPI backend
│       ├── main.py
│       ├── routes/
│       └── schemas.py
├── data/
│   ├── raw/                 # Downloaded papers (JSON)
│   ├── processed/           # Chunked + cleaned text
│   └── knowledge_base/      # ChromaDB files
├── config/
│   ├── settings.py
│   ├── search_queries.yaml  # Search terms for collection
│   └── skin_conditions.yaml # Full condition taxonomy
├── skills/                  # AI skill files (reusable prompts + logic)
│   ├── SKILL_data_collection.md
│   ├── SKILL_rag_pipeline.md
│   ├── SKILL_skin_profiling.md
│   ├── SKILL_regimen_generation.md
│   └── SKILL_vision_analysis.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_SCHEMA.md
│   ├── PROMPTS.md
│   └── EVALUATION.md
├── tests/
│   ├── test_collectors.py
│   ├── test_retrieval.py
│   └── fixtures/
├── scripts/
│   ├── run_collection.py    # One-shot: collect all papers
│   ├── run_indexing.py      # One-shot: embed + index
│   └── run_eval.py          # Evaluate retrieval quality
├── app.py                   # Streamlit UI (MVP)
├── requirements.txt
├── .env.example
└── README.md
```

## Quickstart

```bash
# 1. Clone and setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: add OPENAI_API_KEY, ANTHROPIC_API_KEY

# 3. Collect scientific papers (~30 min, ~1000 papers)
python scripts/run_collection.py

# 4. Build knowledge base (embed + index)
python scripts/run_indexing.py

# 5. Launch MVP UI
streamlit run app.py
```

## Phase roadmap

| Phase | Goal | Duration | Status |
|-------|------|----------|--------|
| 0 | Data collection + knowledge base | Week 1–2 | 🔨 Building |
| 1 | RAG core + skin profiling | Week 3–5 | ⏳ Planned |
| 2 | MVP UI + end-to-end flow | Week 6–8 | ⏳ Planned |
| 3 | Evaluation + iteration | Week 9–12 | ⏳ Planned |

## Cost estimate (monthly, solo dev)

| Service | Tier | Cost |
|---------|------|------|
| OpenAI API (embeddings + GPT-4o Vision) | Pay-as-you-go | ~$10–30 |
| Anthropic API (Claude reasoning) | Pay-as-you-go | ~$10–20 |
| Qdrant Cloud | Free tier (1GB) | $0 |
| Railway.app (backend) | Hobby | $5 |
| Supabase (database) | Free tier | $0 |
| **Total** | | **~$25–55/month** |

## Safety disclaimer

This system provides evidence-based skincare guidance based on scientific literature.
It does NOT replace dermatologist consultation. Always recommend users consult
a healthcare professional for medical conditions, prescription treatments, or
severe skin disorders.
