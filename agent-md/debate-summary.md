# SCA Expert Debate Summary

## Date: 2026-03-30
## Participants: Dr. Minh (CTO/AI Architect) vs Lan (Product Strategist)

---

## Executive Summary

Two experts debated the future of the SCA (Skincare AI) project across 6 rounds covering technical architecture, business model, risk/regulatory, role reversal, strategy, and final verdict.

**Starting positions:**
- Dr. Minh: Safety-first, medical AI positioning, regulatory moat
- Lan: Ship fast, cosmetics pivot, validate before perfecting

**Final convergence:** Both experts converged on a **cosmetics-first execution with medical-grade ingredient knowledge as differentiator** — a hybrid of Options B and C. Ship MVP in ~3 weeks (not 2) with expanded safety coverage (30-50 interaction pairs, up from 10). Kill project if MAU < 500 at day 90.

**Key shifts:**
- Dr. Minh accepted that regulatory moat is not actionable for a solo founder in 12 months
- Lan accepted that 91.7% interaction coverage gap (10/120 known pairs) is too risky to ship without expansion
- Both agreed to enable LLM Safety Judge by default ($0.001/call = $50/month at 50K queries)
- Both agreed to remove XAI heatmap (MobileNetV2 surrogate for GPT-4o = "explainability theater")

---

## Decision Matrix (Final)

| Criteria (weight) | Option A: Medical | Option B: Cosmetics | Option C: Hybrid |
|---|---|---|---|
| Technical feasibility (20%) | 7.5 | 8.0 | 8.0 |
| Market size VN (15%) | 2.5 | 8.5 | 7.5 |
| Competitive moat (15%) | 5.0 | 4.0 | 5.5 |
| Time to revenue (15%) | 2.0 | 8.5 | 6.5 |
| Regulatory risk (15%) | 2.5 | 8.0 | 7.0 |
| Code reuse (10%) | 9.0 | 5.5 | 7.0 |
| Team fit (10%) | 3.5 | 7.5 | 6.5 |
| **WEIGHTED TOTAL** | **4.55** | **7.25** | **6.93** |

Option A decisively eliminated. Options B and C close — converge on "cosmetics-first with ingredient intelligence."

---

## Top 5 Actions (Ranked by impact x feasibility)

### 1. Populate Knowledge Base + Run RAGAS Eval
- **Owner:** Developer
- **Deadline:** Day 3
- **Definition of Done:** ≥800 papers indexed, RAGAS faithfulness baseline measured
- **Files:** `scripts/bootstrap_kb.py`, `scripts/run_eval.py`
- **Blocker:** Need to create `config/eval_test_cases.yaml` (20 test cases)

### 2. Expand Interaction Pairs 10 → 50
- **Owner:** Developer + Claude-assisted extraction
- **Deadline:** Day 10
- **Definition of Done:** 50+ pairs in `config/skin_conditions.yaml`, all regression tests pass, 5 new drug classes (SSRIs, statins, hormonal contraceptives, metformin, ACE inhibitors)
- **Files:** `src/agents/safety_guard.py`, `config/skin_conditions.yaml`
- **Blocker:** Dermatologist spot-check (1-2 hour review)

### 3. Enable LLM Safety Judge by Default
- **Owner:** Developer
- **Deadline:** Day 5
- **Definition of Done:** `use_llm_judge=True` as default in `SafetyGuard.__init__()`, ANTHROPIC_API_KEY added to settings
- **Files:** `src/agents/safety_guard.py:222`
- **Blocker:** None

### 4. Add API Rate Limiting + Basic Auth
- **Owner:** Developer
- **Deadline:** Day 7
- **Definition of Done:** 10 req/min per IP, API key required for `/full-pipeline`
- **Files:** `src/api/deps.py`, `src/api/main.py`
- **Blocker:** None

### 5. Scrape 500 SKU + Build Product Recommendation MVP
- **Owner:** Developer
- **Deadline:** Day 21
- **Definition of Done:** 500 products indexed with INCI analysis, `/recommend` endpoint working
- **Files:** New: `src/collectors/product_collector.py`
- **Blocker:** Shopee Affiliate API approval

---

## Kill Criteria

| Metric | Threshold | Timeline | Action if missed |
|---|---|---|---|
| Beta users enrolled | < 50 | Day 28 | Pause, re-evaluate distribution channel |
| User NPS | < 30 | Day 60 | Major product pivot |
| Affiliate CTR | < 1% | Day 60 | Change monetization model |
| MAU | < 500 | Day 90 | **KILL PROJECT** |
| Safety incidents | > 0 reported adverse reactions | Any time | **IMMEDIATE PAUSE** + investigate + fix |
| RAGAS faithfulness | < 0.70 | Day 14 | Re-evaluate RAG pipeline before shipping |

**Positive triggers (accelerate):**

| Metric | Threshold | Timeline | Action |
|---|---|---|---|
| Organic MAU growth | > 20%/week | Month 2-3 | Double down on content, explore seed round |
| Affiliate revenue | > 10M VND/month | Month 3 | Hire content creator full-time |

---

## Unresolved Disagreements

### 1. RAGAS Eval Before vs After Ship
- **Dr. Minh:** Must validate faithfulness > 0.70 before any user touches the product
- **Lan:** User feedback is more reliable than synthetic eval; ship and iterate
- **Resolution data needed:** Run RAGAS eval AND collect beta user ratings. If correlation between RAGAS score and user satisfaction > 0.6, RAGAS is reliable enough to gate shipping.

### 2. Evidence Grading in Consumer UI
- **Dr. Minh:** Display "Evidence A/B/C" for every recommendation — builds trust
- **Lan:** Hide from main UI, show only in "Why this product?" detail view — Gen Z doesn't care
- **Resolution data needed:** A/B test with 100 users. Measure trust score + conversion rate.

### 3. XAI Heatmap
- **Dr. Minh:** Remove completely (MobileNetV2 surrogate for GPT-4o = misleading)
- **Lan:** Keep for premium tier only with prominent caveat
- **Resolution data needed:** User test with 20 people. If >50% say they trust heatmap BUT their understanding of what it shows is incorrect → remove.

### 4. Monetization Priority
- **Dr. Minh:** Premium subscription first (build trust before monetizing via affiliate)
- **Lan:** Affiliate first (prove unit economics from day 1)
- **Resolution data needed:** Run both in beta. Whichever drives more revenue per user at 1,000 MAU wins.

### 5. First Hire
- **Dr. Minh:** Dermatologist consultant (safety validation)
- **Lan:** Content creator (growth/acquisition)
- **Resolution data needed:** Whichever bottleneck hits first during beta.

---

## Appendix: Key Code References

| File | Line(s) | Relevance |
|---|---|---|
| `src/agents/safety_guard.py` | 154-184 | 10 avoid_together interaction pairs — needs expansion to 50+ |
| `src/agents/safety_guard.py` | 187-215 | 7 drug interaction rules, 6 drug classes — needs 5+ new classes |
| `src/agents/safety_guard.py` | 222 | `use_llm_judge=False` — change to True |
| `src/agents/safety_guard.py` | 247-255 | 8 safety check calls — working correctly |
| `src/agents/rag_retriever.py` | 60 | Cross-encoder model: ms-marco-MiniLM-L-6-v2 |
| `src/agents/rag_retriever.py` | 317 | Scoring formula: `norm_ce * 0.85 + boost * 0.15` |
| `src/agents/xai_explainer.py` | 7-8 | Surrogate: MobileNetV2 on ImageNet — misleading for skin |
| `src/agents/xai_explainer.py` | 18-21 | Code acknowledges bias on dark skin (Fitzpatrick V-VI) |
| `src/pipeline/chunker.py` | 96-97 | Chunk 512 tokens, overlap 64 — overlap ratio 12.5% (low) |
| `src/pipeline/bm25_index.py` | 137-161 | RRF fusion: k=60, dense 0.6, sparse 0.4 |
| `src/agents/regimen_generator.py` | 86 | Hard-coded Anthropic client — no fallback model |
| `src/agents/vision_analyzer.py` | 94-97 | Hard-coded OpenAI client — no fallback |
| `config/skin_conditions.yaml` | 245-259 | Pregnancy avoid list: ~10 items |
| `config/skin_conditions.yaml` | 342-379 | Concentration limits — OTC max only, no combo effects |
| `src/api/deps.py` | 1-75 | No authentication on any endpoint |
| `scripts/run_eval.py` | 1-100 | RAGAS eval framework exists but never run (empty KB) |
| `src/api/routes.py` | 374-376 | Only admin cache clear has token protection |

---

## Fermi Estimations Summary

| # | Estimation | Result |
|---|---|---|
| 1 | Cost to populate KB | <$1 and <3 hours |
| 2 | Realistic affiliate revenue at 10K MAU | ~900K VND/month ($36) — 27x lower than plan |
| 3 | CAC for 50K MAU (paid acquisition, VN) | ~2 billion VND ($80,000) |
| 4 | P(adverse event, pregnant teen + retinoid) | ~3 per million users (3% at 10K MAU lifetime) |

---

*Generated from 6-round expert debate. Both experts acknowledged that the final convergence ("cosmetics-first with safety buffer") represents neither's ideal position — which is likely the correct answer for a solo-founder bootstrapped project.*
