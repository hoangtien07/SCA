# ĐÁNH GIÁ TOÀN DIỆN DỰ ÁN SCA (Skincare AI)

## TỔNG QUAN

SCA là hệ thống tư vấn skincare cá nhân hóa, kết hợp RAG (Retrieval-Augmented Generation) trên ~1,000 bài nghiên cứu y khoa, phân tích ảnh da bằng GPT-4o Vision, sinh regimen bằng Claude, và hệ thống safety guardrails đa tầng. Giao diện MVP bằng Streamlit, backend FastAPI, hỗ trợ async Celery/Redis.

---

## PHẦN 1: TÍNH KHẢ THI & HIỆU QUẢ KHI TRIỂN KHAI THỰC TẾ

### 1.1 Giá trị sản phẩm — 7.5/10

**Điểm mạnh:**
- Giải quyết nhu cầu thực: skincare cá nhân hóa dựa trên bằng chứng y khoa, không dựa trên quảng cáo
- Evidence-graded recommendations (A/B/C) tạo sự tin cậy hơn hẳn các app skincare hiện tại (Curology, Skin+Me, Proven)
- Safety guardrails 7 tầng là điểm khác biệt lớn nhất — các đối thủ hiện tại thiếu hẳn hoặc rất sơ sài

**Điểm yếu nghiêm trọng:**
- **Knowledge base rỗng** — Đây là blocker #1. Không có dữ liệu = không có sản phẩm. Toàn bộ pipeline dù hoàn thiện vẫn chỉ là "shell" cho đến khi chạy `bootstrap_kb.py`
- **Không có user study/clinical validation** — Trong lĩnh vực health-tech, recommendations chưa qua kiểm chứng lâm sàng hoặc peer-review thì rủi ro pháp lý rất cao
- **Thiếu personalization loop** — Hệ thống là one-shot (hỏi → trả lời), không có feedback mechanism để cải thiện theo thời gian cho từng user (follow-up tracking đã thiết kế schema nhưng chưa implement)

### 1.2 Tính khả thi kỹ thuật — 8/10

**Khả thi cao:**
- Tech stack đã mature (ChromaDB, FastAPI, Claude, GPT-4o) — không có "research risk"
- Pipeline RAG hybrid (dense + sparse + reranking) là state-of-the-art cho medical QA tính đến 3/2026
- Chi phí vận hành thấp (~$25–55/tháng cho solo dev) — khả thi cho MVP/indie

**Rủi ro khả thi:**
- **Embedding model mismatch**: PubMedBERT (768d) vs OpenAI (1536d) — switching giữa dev/prod sẽ cần re-index toàn bộ knowledge base. Đã có check mismatch nhưng migration path chưa rõ
- **Neo4j integration 80% stub** — `GraphRetriever` toàn `NotImplementedError`. Nếu không cần graph features, nên loại bỏ hẳn khỏi architecture diagram thay vì để stub gây nhầm lẫn
- **XAI heatmap dùng MobileNetV2 surrogate cho GPT-4o** — Đây là approximation rất thô. MobileNetV2 trained trên ImageNet không reflect reasoning của GPT-4o Vision trên ảnh da. Kết quả heatmap có thể misleading, đặc biệt trên da tối (Fitzpatrick V-VI). Code đã có caveat nhưng UX chưa làm nổi bật warning này đủ mạnh

### 1.3 Tính khả thi thương mại — 5.5/10

**Thách thức lớn:**

| Thách thức | Mức độ | Chi tiết |
|------------|--------|----------|
| **Pháp lý/Compliance** | Cao | Ở hầu hết thị trường (US/EU/VN), AI skincare advice có thể bị xếp vào medical device nếu đưa ra diagnosis + treatment recommendation. EU MDR 2017/745, FDA digital health guidance đều yêu cầu clinical validation. Disclaimer "not medical advice" không đủ bảo vệ pháp lý |
| **Liability** | Cao | Nếu user bị adverse reaction từ recommendation, liability nằm ở provider. Chưa có indemnification strategy |
| **Data privacy** | Trung bình | Ảnh da = biometric data → GDPR Article 9, CCPA special category. Code xử lý in-memory (tốt) nhưng chưa có data retention policy rõ ràng |
| **Moat/Barrier to entry** | Thấp | RAG pipeline + safety rules có thể replicate trong 2-3 tuần bởi team có kinh nghiệm. Competitive advantage phụ thuộc vào quality of KB + clinical validation |

### 1.4 Đánh giá so với thị trường (3/2026)

| Tiêu chí | SCA | Curology/Skin+Me | Proven Skincare | Tirzepatide Derm AI |
|-----------|-----|-------------------|-----------------|---------------------|
| Evidence-based | Citations + grades | Proprietary | Partial | But opaque |
| Personalization depth | Medium (questionnaire + photo) | High (dermatologist in loop) | High (ML on quiz) | High (longitudinal) |
| Safety guardrails | 7 layers | Human review | Unknown | Clinical-grade |
| Cost to user | Free/Low | $20-40/mo | $30-50 | Insurance |
| Regulatory | None | Licensed | FDA registered | Clinical trial |

**Verdict**: SCA có giá trị như một research project / educational tool / open-source contribution. Để thương mại hóa cần clinical validation + regulatory compliance — chi phí và thời gian cho phần này lớn hơn toàn bộ technical build hiện tại.

---

## PHẦN 2: ĐÁNH GIÁ KIẾN TRÚC & KỸ THUẬT

### 2.1 Kiến trúc tổng thể — 8/10

**Ưu điểm:**
- **Layered architecture rõ ràng**: Data Collection → Processing → Knowledge Base → AI Agents → Safety → API/UI. Mỗi layer independent, swappable
- **Graceful degradation xuất sắc**: Toàn bộ optional components (Redis, Neo4j, LangSmith, XAI, LLM Judge) đều fail silently. Hệ thống vẫn chạy ở mức tối thiểu chỉ cần OpenAI key + Anthropic key
- **Separation of concerns tốt**: Collectors, pipeline, agents, API đều tách biệt. Có thể test từng phần independently
- **Cost-conscious design**: Semantic cache (24h retrieval, 12h regimen), BM25 zero-cost, LLM judge rate-limited (1 call/regimen, 8s timeout)

**Vấn đề kiến trúc:**

1. **Single point of failure: `skin_conditions.yaml` (407 dòng)**
   - Toàn bộ safety logic phụ thuộc vào file YAML duy nhất này
   - Thiếu 1 interaction = recommendation nguy hiểm
   - Không có versioning, audit trail, hay validation pipeline cho taxonomy
   - **Khuyến nghị**: Chuyển sang database với change tracking, hoặc ít nhất thêm schema validation + automated completeness checks

2. **Tight coupling giữa API schemas và internal models**
   - `routes.py` có các hàm `_chunk_to_retrieval_result()`, `_regimen_to_response()`, `_response_to_mock_regimen()` — chuyển đổi qua lại giữa internal models và API models. Đây là code smell cho thấy domain model chưa được thiết kế clean
   - **Khuyến nghị**: Unified domain model với serialization adapters

3. **Streamlit as MVP UI — đúng quyết định nhưng có giới hạn**
   - Streamlit rerun entire script on every interaction → không scalable cho complex UX
   - Session state management sẽ trở thành nightmare khi thêm features
   - **Đã có plan** migrate sang Next.js — đúng hướng

### 2.2 RAG Pipeline — 8.5/10 (điểm cao nhất)

Đây là phần được thiết kế tốt nhất của dự án.

**State-of-the-art elements:**
- **Hybrid retrieval** (dense + BM25 + RRF fusion): Weight 0.6/0.4 với k=60 là balanced. Đây là best practice cho medical domain where keyword matching matters (drug names, INCI terms)
- **Cross-encoder reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` cho second-pass scoring. Formula `0.85 * CE_score + 0.15 * evidence_boost` hợp lý — ưu tiên relevance nhưng bump evidence grade A
- **Section-aware chunking**: Phát hiện `[ABSTRACT]`, `[METHODS]`, `[RESULTS]`, `[CONCLUSION]` markers từ PMC OA. Mỗi chunk mang full metadata. Đây là improvement quan trọng so với naive chunking
- **Multi-query expansion**: Sinh query variants với medical term synonyms trước khi retrieval. Tăng recall đáng kể cho medical terminology
- **Semantic caching với cosine similarity threshold 0.92**: Tránh redundant API calls, TTL hợp lý

**Điểm cần cải thiện:**

1. **Chunk size 512 tokens, overlap 64** — Overlap ratio 12.5% thấp cho medical text. Standard là 20-25% (128 tokens overlap cho 512 chunk). Nguy cơ mất context ở ranh giới chunk

2. **Embedding model choice**: PubMedBERT (768d, free) vs text-embedding-3-small (1536d, paid) — `compare_embeddings.py` script rất hay nhưng chưa chạy. Trong medical domain, PubMedBERT thường outperform general-purpose models trên retrieval relevance. Nên chạy benchmark sớm để quyết định

3. **Chưa có query routing/classification**: Tất cả queries đều đi qua cùng pipeline. Một query đơn giản ("what is retinol?") và query phức tạp ("regimen for 65yo with rosacea + immunosuppressant + pregnancy history") cũng xử lý giống nhau. Query router sẽ tiết kiệm latency và cost

4. **RAGAS evaluation chưa chạy** — Không biết retrieval quality thực tế. Targets (faithfulness > 0.85, precision > 0.75) là hợp lý nhưng chưa có baseline

### 2.3 Safety System — 7.5/10

**Ưu điểm:**
- **7 layers kiểm tra**: pregnancy → medication → conflicts → allergens → phototoxicity → age → concentration → escalation
- **44 regression tests passing** — Coverage tốt cho known scenarios
- **INCI synonym resolution**: `retinol` match với `retinal`, `retinaldehyde`, `retinyl_palmitate`. Critical cho real-world matching
- **Class/group expansion**: "retinoids" → expand ra toàn bộ danh sách. Phòng trường hợp LLM dùng generic term

**Vấn đề nghiêm trọng:**

1. **Rule-based approach có ceiling rõ ràng**
   - 10 interaction pairs trong `avoid_together` là quá ít. Thực tế có hàng trăm known interactions
   - Drug interaction list chỉ cover 6 nhóm thuốc — thiếu nhiều nhóm phổ biến (antidepressants/SSRIs affecting skin, hormonal contraceptives, diabetes medications affecting wound healing)
   - **Đây là rủi ro lớn nhất của dự án**: False negative (miss an interaction) nguy hiểm hơn false positive

2. **LLM Safety Judge chưa auto-enable**
   - `LLMSafetyJudge` dùng Claude Haiku là ý tưởng hay (cheap, fast, second opinion) nhưng disabled by default và không có setting trong `.env`
   - Cost guard (1 call, 8s timeout, confidence 0.7) quá conservative — nên cho chạy mặc định cho mọi regimen, cost chỉ ~$0.001/call

3. **Concentration limits quá đơn giản**
   - Chỉ check OTC max, không xem xét combination effects (retinol 0.5% + glycolic 8% riêng thì OK, nhưng dùng cùng lúc thì quá irritating cho nhiều skin types)
   - Không có "irritation budget" concept — thiếu trong literature nhưng quan trọng trong practice

### 2.4 Vision Analysis — 6.5/10

**Ưu điểm:**
- GPT-4o Vision cho skin analysis là reasonable choice (tốt nhất available 3/2026 cho general vision)
- Zone-based analysis (forehead, cheeks, chin, etc.) là clinical-relevant
- `merge_with_questionnaire()` cho questionnaire take precedence over vision — đúng, vì user self-report reliable hơn cho nhiều conditions

**Vấn đề:**

1. **Không có clinical validation cho GPT-4o trên skin diagnosis**
   - GPT-4o Vision chưa được FDA cleared hay CE marked cho dermatological diagnosis
   - Accuracy trên dark skin (Fitzpatrick V-VI) known to be lower cho tất cả CV models, kể cả GPT-4o
   - `eval_vision_accuracy.py` script tồn tại nhưng chưa chạy — critical gap

2. **XAI heatmap misleading**
   - MobileNetV2 surrogate cho GPT-4o là "explaining the wrong model"
   - LIME trên ImageNet surrogate không reflect GPT-4o's actual attention
   - Có thể gây false confidence ở user
   - **Khuyến nghị**: Hoặc remove XAI hoặc label cực kỳ rõ "this is approximate, not actual model reasoning"

3. **Privacy concern với image processing**
   - Code xử lý in-memory (tốt) nhưng images vẫn gửi lên OpenAI API → data leaves user's control
   - Cần disclosure rõ ràng cho user về data processing

### 2.5 LLM Integration — 7.5/10

**Kiến trúc multi-model hợp lý:**

| Task | Model | Lý do | Đánh giá |
|------|-------|-------|----------|
| Regimen generation | Claude Sonnet 4.6 | Structured output, reasoning | Tốt — Sonnet balance cost/quality |
| Vision analysis | GPT-4o | Best multimodal | Hợp lý |
| Safety judge | Claude Haiku 4.5 | Cheap, fast second opinion | Hay |
| Embeddings | PubMedBERT / text-embedding-3-small | Domain-specific vs general | Cần benchmark |

**Vấn đề:**
1. **Anthropic SDK version cũ** (`anthropic==0.32.0`): Tính đến 3/2026, SDK đã lên 0.50+. Cần upgrade để access latest features (improved structured output, tool use improvements)
2. **OpenAI SDK version cũ** (`openai==1.40.0`): Cũng đã outdated. Missing realtime API, improved vision features
3. **Không có fallback model**: Nếu Claude API down, toàn bộ generation fails. Nên có fallback sang GPT-4o cho regimen generation
4. **Prompt engineering chưa dùng structured output API**: Cả Claude và GPT-4o đều có native JSON mode / tool_use. Hiện tại dựa vào "Output format: Valid JSON" trong prompt — less reliable, cần Pydantic validation as fallback (đã có, nhưng native JSON mode sẽ robust hơn)

### 2.6 Infrastructure & DevOps — 6/10

**Ưu điểm:**
- Docker multi-service setup sẵn sàng
- Railway.toml cho cloud deployment
- GitHub Actions CI cơ bản

**Vấn đề:**

1. **CI pipeline chưa chạy được**: `requirements-dev.txt` tồn tại nhưng thiếu packages cần thiết
2. **Không có rate limiting trên API** — Production risk: 1 user spam endpoint = runaway API costs
3. **Không có authentication/authorization**: API endpoints mở hoàn toàn, chỉ có admin cache clear cần token
4. **Không có monitoring/alerting**: LangSmith tracing là opt-in observability nhưng không có alerting cho errors, latency spikes, cost overruns
5. **Secret management**: API keys trong `.env` file — OK cho dev, nhưng cần proper secret manager cho production (Railway secrets, AWS SSM, etc.)
6. **Database migration**: Supabase schema trong docs nhưng chưa có migration files

### 2.7 Code Quality — 7.5/10

**Ưu điểm:**
- Consistent coding style, Pydantic models throughout
- Loguru structured logging với trace IDs
- Type hints trên hầu hết functions
- Graceful degradation pattern applied consistently

**Vấn đề:**
- Một số file quá dài: `app.py` ~700 lines, `routes.py` ~400 lines — nên split
- `{src` directory rỗng tồn tại ở root — artifact cần xóa
- Inline constants (magic numbers) trong safety checks — nên extract ra config
- Test coverage thiên về safety regression, thiếu integration tests cho full pipeline

### 2.8 Công nghệ & Dependencies — 7/10

| Dependency | Version | Latest (3/2026) | Risk |
|------------|---------|-----------------|------|
| `anthropic` | 0.32.0 | ~0.52+ | Major version drift |
| `openai` | 1.40.0 | ~1.65+ | Missing features |
| `chromadb` | 0.5.3 | ~0.6+ | Breaking changes likely |
| `sentence-transformers` | 3.0.1 | ~3.4+ | Minor |
| `streamlit` | 1.36.0 | ~1.42+ | Minor |
| `fastapi` | 0.111.1 | ~0.115+ | Stable |
| `pydantic` | 2.8.2 | ~2.10+ | Stable |
| `celery` | 5.3.6 | ~5.4+ | Stable |

**Rủi ro**: ChromaDB 0.5 → 0.6 có breaking changes trên persistence format. Cần pin version hoặc upgrade sớm.

---

## ĐIỂM TỔNG KẾT

| Tiêu chí | Điểm | Trọng số | Weighted |
|----------|------|----------|----------|
| **RAG Pipeline Design** | 8.5/10 | 25% | 2.13 |
| **Architecture & Modularity** | 8.0/10 | 20% | 1.60 |
| **Safety System** | 7.5/10 | 20% | 1.50 |
| **Code Quality** | 7.5/10 | 10% | 0.75 |
| **Production Readiness** | 5.0/10 | 15% | 0.75 |
| **Commercial Viability** | 5.5/10 | 10% | 0.55 |
| **TỔNG** | | | **7.28/10** |

---

## KẾT LUẬN

**SCA là một dự án kỹ thuật ấn tượng** với RAG pipeline được thiết kế ở mức chuyên gia, safety system chu đáo, và architecture sạch. Đây dễ dàng là top 5% trong các open-source medical AI projects.

**Tuy nhiên, khoảng cách từ "code hoàn thiện" đến "sản phẩm triển khai thực tế" vẫn còn lớn:**

1. **Blocker tức thời**: Knowledge base rỗng — chạy `bootstrap_kb.py` là bước tiếp theo bắt buộc
2. **Blocker trung hạn**: Clinical validation, regulatory compliance, liability framework
3. **Technical debt**: SDK versions outdated, Neo4j stub, thiếu rate limiting & auth

**Khuyến nghị chiến lược:**
- **Nếu mục tiêu là portfolio/research**: Dự án đã đủ ấn tượng. Chạy KB, chạy eval, deploy demo là xong
- **Nếu mục tiêu là sản phẩm thực**: Cần partner với dermatologist cho clinical validation, luật sư cho regulatory, và ít nhất 6-12 tháng thêm cho compliance trước khi public launch
- **Nếu mục tiêu là B2B tool**: Bán cho dermatology clinics/brands như assistive tool (không phải direct-to-consumer) sẽ giảm regulatory burden đáng kể
