# SCA Upgrade Notes — Status & Setup Guide
_Dựa theo kế hoạch sca-12-prompts-opus-4.6.md · Cập nhật: 2026-03-29_

---

## Tóm tắt trạng thái

| Phase | Prompt | Tên | Code | Cần gì để chạy |
|-------|--------|-----|------|----------------|
| 0 | #01 | Cross-encoder reranking | ✅ Xong | `pip install sentence-transformers` (~85MB download) |
| 1 | #02 | LangSmith tracing | ✅ Xong | `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true` |
| 1 | #03 | Bootstrap knowledge base | ✅ Xong | `OPENAI_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY` (optional), mạng |
| 2 | #04 | PubMedBERT embeddings | ✅ Xong | `pip install sentence-transformers` + 440MB download lần đầu |
| 2 | #09 | CV datasets (SCIN, Fitzpatrick17K...) | ✅ Code xong | ~10GB storage + mạng để download |
| 3 | #05 | Hybrid safety guard (LLM judge) | ✅ Xong | `ANTHROPIC_API_KEY` + `use_llm_judge=True` |
| 3 | #10 | Cosmetic knowledge base (OBF + CosIng) | ✅ Code xong | Mạng + 800MB storage (OBF parquet) |
| 4 | #06 | Celery async queue | ✅ Code xong | Redis (local hoặc Railway add-on) |
| 4 | #07 | Semantic caching | ✅ Code xong | Redis + `CACHE_ENABLED=true` |
| 4 | #11 | Neo4j schema + GraphRAG stub | ✅ Code xong | Neo4j Aura (free tier) để activate |
| 5 | #08 | Activate GraphRAG (Neo4j driver) | ✅ Code xong | Neo4j provisioned + seed.cypher đã chạy |
| 5 | #12 | XAI heatmap (LIME/SHAP) | ✅ Code xong | `XAI_ENABLED=true` + optional SCIN data |

**Bugs đã fix (session này):**
- `bm25_index.py`: `rrf_fuse` KeyError khi sparse result thiếu key `metadata`
- `safety_guard.py`: 5 lỗi matching — underscore vs space, case sensitivity, class name expansion

---

## Những gì CÓ THỂ chạy ngay (không cần API key)

### Chạy tests
```bash
pip install pyyaml loguru rank-bm25 pydantic pydantic-settings pytest
python -m pytest tests/test_safety_regression.py tests/test_retriever.py tests/test_collectors.py -v
# Kết quả: 44/44 passed
```

### Chạy pipeline với PubMedBERT (local, không tốn tiền API)
```bash
# Lần đầu: download model ~440MB (tự động)
python scripts/run_indexing.py --embedding-provider local --yes
# Nếu đã có collection cũ với OpenAI:
python scripts/run_indexing.py --embedding-provider local --reindex --yes
```

### So sánh PubMedBERT vs OpenAI embeddings
```bash
# Cần có eval_test_cases.yaml + knowledge base đã index
python scripts/compare_embeddings.py
# Output: data/processed/embedding_comparison.json
```

### Seed knowledge graph (không cần Neo4j)
```bash
# Tạo file Cypher sẵn để paste vào Neo4j Browser sau
python scripts/seed_graph.py
# Output: data/graph/seed.cypher, data/graph/seed_stats.json
```

---

## Hướng dẫn setup theo từng Phase

### Phase 1 — Observability (LangSmith)

**Điều kiện:** Tạo account tại https://smith.langchain.com/ → lấy API key

```bash
pip install langsmith>=0.1.83
```

Trong `.env`:
```
LANGSMITH_API_KEY=ls__your_key_here
LANGSMITH_PROJECT=skincare-ai
LANGSMITH_TRACING=true
```

Sau đó mọi `/full-pipeline` request sẽ auto-trace. Dashboard tại smith.langchain.com.

---

### Phase 1 — Bootstrap Knowledge Base

**Điều kiện:**
- `OPENAI_API_KEY` (để embed + RAGAS eval)
- `ANTHROPIC_API_KEY` (để RAGAS faithfulness check)
- Kết nối mạng (collect từ Semantic Scholar + PubMed)

```bash
pip install -r requirements.txt

# Chạy toàn bộ pipeline: collect → filter → index → eval
python scripts/bootstrap_kb.py

# Nếu muốn force re-run từ đầu:
python scripts/bootstrap_kb.py --force

# Ước tính: ~45 phút cho 1000 papers, chi phí ~$0.50 embedding
```

Kết quả: `data/processed/eval_baseline.json` với RAGAS metrics.

---

### Phase 2 — CV Datasets

**Điều kiện:** ~10GB storage trống, kết nối mạng

```bash
# Xem info mà không download
python scripts/download_cv_datasets.py --dry-run

# Download tất cả datasets
python scripts/download_cv_datasets.py --datasets all

# Download từng cái
python scripts/download_cv_datasets.py --datasets scin,dermamnist

# Benchmark GPT-4o accuracy trên SCIN (cần OPENAI_API_KEY)
python scripts/eval_vision_accuracy.py --samples 100
```

| Dataset | Images | Size | License |
|---------|--------|------|---------|
| SCIN (Google) | ~5,600 | ~2GB | Apache 2.0 |
| Fitzpatrick17K | 16,577 | ~6GB | MIT |
| DDI (Stanford) | 656 | ~500MB | CC BY 4.0 |
| DermaMNIST | 10,015 | ~50MB | CC BY 4.0 |

---

### Phase 3 — LLM Safety Judge

**Điều kiện:** `ANTHROPIC_API_KEY` đã set

```python
# Trong code:
from src.agents.safety_guard import SafetyGuard
guard = SafetyGuard(use_llm_judge=True, anthropic_api_key="sk-ant-...")

# Hoặc qua settings (thêm vào .env):
USE_LLM_JUDGE=true  # chưa implement tự động load — set manually
```

Chi phí ước tính: ~$1/tháng cho 1000 regimens (Claude Haiku ~$0.001/check).

---

### Phase 3 — Cosmetic Knowledge Base (Open Beauty Facts + CosIng)

**Điều kiện:** Mạng + ~800MB storage (OBF Parquet)

```bash
pip install pyarrow>=16.0.0

# Xem info trước
python scripts/download_cosmetic_data.py --dry-run

# Download Open Beauty Facts (~800MB) + fetch CosIng (~2000 API calls)
python scripts/download_cosmetic_data.py --source all

# Generate Cypher file (không cần Neo4j)
python scripts/seed_graph.py
# Output: data/graph/seed.cypher
```

---

### Phase 4 — Async Queue (Celery + Redis)

**Điều kiện:** Redis server

**Option A — Local dev:**
```bash
# Windows: dùng WSL hoặc Docker
docker run -d -p 6379:6379 redis:7-alpine

pip install "celery[redis]==5.3.6" redis==5.0.4 flower==2.0.1

# Khởi động worker
celery -A src.workers.celery_app worker --loglevel=info --concurrency=2

# Monitor (optional)
celery -A src.workers.celery_app flower
```

Trong `.env`: `REDIS_URL=redis://localhost:6379/0`

**Option B — Railway:**
1. Tạo Railway project
2. Add Redis service → copy URL vào `REDIS_URL`
3. Add worker service với command: `celery -A src.workers.celery_app worker --loglevel=info --concurrency=2`

App Streamlit: check "Use async generation (requires Redis)" để thấy progress bar.

---

### Phase 4 — Semantic Caching

**Điều kiện:** Redis đã setup (xem trên)

```
# Trong .env:
CACHE_ENABLED=true
CACHE_SIMILARITY_THRESHOLD=0.92
CACHE_RETRIEVAL_TTL=86400    # 24 giờ
CACHE_REGIMEN_TTL=43200      # 12 giờ
```

Tiết kiệm ước tính: 40-60% API cost khi có user base đủ lớn.
**Lưu ý:** Cache KHÔNG bao giờ được dùng cho profile có `pregnancy=True`.

---

### Phase 4 — Neo4j Knowledge Graph

**Điều kiện:** Neo4j Aura free tier (200MB, ~5000 nodes)

```bash
pip install neo4j==5.25.0

# Bước 1: Generate Cypher (không cần Neo4j)
python scripts/seed_graph.py

# Bước 2: Lên https://aura.neo4j.io → tạo free instance
# Bước 3: Copy seed.cypher → paste vào Neo4j Browser → run

# Bước 4: Set trong .env:
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
GRAPH_RAG_ENABLED=true
```

Tài liệu schema: `docs/GRAPHRAG_DESIGN.md`

---

### Phase 5 — XAI Heatmap (LIME)

**Điều kiện:** `XAI_ENABLED=true` + LIME/SHAP packages

```bash
pip install lime==0.2.0.1 shap==0.46.0 torch torchvision

# Nếu có SCIN data (từ Phase 2), heatmap chính xác hơn
# Không có SCIN: dùng MobileNetV2 pretrained trên ImageNet

# Trong .env:
XAI_ENABLED=true
XAI_NUM_SAMPLES=500    # cao hơn = chính xác hơn nhưng chậm hơn (~5-8s)
```

Trong Streamlit: tick "Show AI explanation heatmap" sau khi upload ảnh.

**Lưu ý:** Heatmap dùng surrogate model (MobileNetV2), không phải GPT-4o thực. Chỉ mang tính minh họa.

---

## Install tất cả dependencies

```bash
pip install -r requirements.txt

# Optional packages (theo phase):
pip install langsmith>=0.1.83          # Phase 1 - LangSmith
pip install pyarrow>=16.0.0            # Phase 3 - OBF Parquet
pip install "celery[redis]==5.3.6" redis==5.0.4  # Phase 4 - Async
pip install neo4j==5.25.0              # Phase 4/5 - GraphRAG
pip install lime==0.2.0.1 shap==0.46.0 # Phase 5 - XAI
```

---

## Cấu trúc files mới đã được tạo

```
src/
  agents/
    safety_guard.py       ← LLMSafetyJudge thêm vào
    graph_retriever.py    ← Stub, activate khi có Neo4j
    xai_explainer.py      ← LIME/SHAP explainer
  api/
    tracing.py            ← LangSmith integration
    routes.py             ← /generate-async, /task/{id}, /admin/cache
  cache/
    semantic_cache.py     ← Redis semantic cache
  collectors/
    cv_dataset_collector.py   ← SCIN, Fitzpatrick17K, DDI, DermaMNIST
    cosmetic_api_collector.py ← Open Beauty Facts + CosIng
  pipeline/
    embedder.py           ← PubMedBERT + OpenAI embedders
  workers/
    celery_app.py         ← Celery + Redis config
    tasks.py              ← generate_regimen_task

scripts/
  bootstrap_kb.py         ← Master pipeline orchestration
  compare_embeddings.py   ← A/B test PubMedBERT vs OpenAI
  download_cv_datasets.py ← CLI downloader for CV datasets
  eval_vision_accuracy.py ← Benchmark GPT-4o vs SCIN ground truth
  seed_graph.py           ← Generate Neo4j Cypher (no DB needed)
  download_cosmetic_data.py ← CLI for OBF + CosIng

docs/
  GRAPHRAG_DESIGN.md      ← Neo4j schema + Cypher examples
  ARCHITECTURE.md         ← Layer 2.5 GraphRAG section

config/
  settings.py             ← All new fields (LangSmith, embedder, cache, XAI...)

.env.example              ← Template với tất cả variables
```
