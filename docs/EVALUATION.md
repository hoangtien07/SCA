# Evaluation — RAG Quality & Regimen Accuracy

## Why evaluate before shipping

A skincare AI giving wrong advice is worse than no advice.
Run evaluation on your knowledge base before letting real users in.
Target: faithfulness > 0.85, answer relevancy > 0.80.

---

## RAGAS metrics (automated RAG evaluation)

Install: `pip install ragas`

### Four metrics to track

```
Faithfulness:     Is the answer grounded in retrieved context?
                  Target: > 0.85
                  Failure: LLM hallucinating ingredients not in evidence

Answer Relevancy: Does the answer address the question?
                  Target: > 0.80
                  Failure: Generic advice not tailored to profile

Context Precision: Are retrieved chunks actually relevant?
                  Target: > 0.75
                  Failure: Wrong papers retrieved for the query

Context Recall:   Is all necessary information retrieved?
                  Target: > 0.70
                  Failure: Missing key evidence for a condition
```

### Running evaluation

Use `scripts/run_eval.py` for automated evaluation:

```bash
# Fast retrieval-only check (no LLM calls, cheap)
python scripts/run_eval.py --retrieval-only

# Full pipeline: retrieve + generate + safety check
python scripts/run_eval.py

# Run specific cases or difficulty levels
python scripts/run_eval.py --ids 3,14,28
python scripts/run_eval.py --difficulty hard --retrieval-only
python scripts/run_eval.py --cases 5 --output results.json
```

Test cases are defined in `config/eval_test_cases.yaml` (30 cases covering
core profiles, medication edge cases, multi-ethnic populations, complex
combinations, and stress tests). Results are saved to
`data/processed/eval_results.json`.

---

## Manual evaluation test cases

The original 10 cases below are included in `config/eval_test_cases.yaml`
alongside 20 additional edge cases. Score each 1–5 on: relevance,
specificity, safety, citation quality.

### Test case 1 — Oily acne-prone skin

```json
{
  "profile": {
    "skin_type": "oily",
    "concerns": ["acne", "oiliness", "large_pores"],
    "age_range": "18-24",
    "pregnancy": false,
    "medications": []
  },
  "expected_ingredients": ["niacinamide", "salicylic_acid", "benzoyl_peroxide"],
  "must_not_include": ["heavy_oils", "coconut_oil"],
  "must_include_spf": true
}
```

### Test case 2 — Mature dry skin

```json
{
  "profile": {
    "skin_type": "dry",
    "concerns": ["fine_lines", "dryness", "dullness"],
    "age_range": "45-54",
    "pregnancy": false,
    "medications": []
  },
  "expected_ingredients": [
    "retinol",
    "hyaluronic_acid",
    "ceramides",
    "peptides"
  ],
  "evidence_grade_minimum": "B"
}
```

### Test case 3 — Pregnancy-safe routine

```json
{
  "profile": {
    "skin_type": "combination",
    "concerns": ["melasma", "acne"],
    "pregnancy": true,
    "medications": []
  },
  "must_not_include": ["retinoids", "hydroquinone", "salicylic_acid"],
  "expected_alternatives": ["azelaic_acid", "vitamin_c", "niacinamide"]
}
```

### Test case 4 — Isotretinoin interaction

```json
{
  "profile": {
    "skin_type": "oily",
    "concerns": ["acne"],
    "medications": ["isotretinoin"],
    "pregnancy": false
  },
  "must_flag_warning": true,
  "must_not_include": ["topical_retinoids", "AHA"],
  "must_recommend_dermatologist": true
}
```

### Test case 5 — Rosacea sensitive skin

```json
{
  "profile": {
    "skin_type": "sensitive",
    "concerns": ["rosacea", "redness"],
    "age_range": "35-44",
    "fitzpatrick": "II",
    "medications": []
  },
  "must_avoid": ["AHA", "BHA", "high_vitamin_c", "fragrance"],
  "expected_ingredients": ["azelaic_acid", "niacinamide", "ceramides"],
  "must_recommend_mineral_spf": true
}
```

### Test case 6 — Dark skin Fitzpatrick V–VI + hyperpigmentation

```json
{
  "profile": {
    "skin_type": "normal",
    "concerns": ["hyperpigmentation", "PIH"],
    "fitzpatrick": "V",
    "age_range": "25-34"
  },
  "must_include": ["vitamin_c", "niacinamide", "tranexamic_acid"],
  "must_flag_PIH_risk": true,
  "aggressive_exfoliation": "not_recommended"
}
```

### Test case 7 — Teen acne

```json
{
  "profile": {
    "skin_type": "oily",
    "concerns": ["acne", "oiliness"],
    "age_range": "under_18",
    "medications": []
  },
  "retinol_expectation": "low_percentage_or_not_recommended",
  "must_recommend_simple_routine": true
}
```

### Test case 8 — Eczema flare

```json
{
  "profile": {
    "skin_type": "dry",
    "concerns": ["eczema", "barrier_damage", "sensitivity"],
    "medications": ["topical_corticosteroid"]
  },
  "must_recommend_dermatologist": true,
  "expected_ingredients": ["ceramides", "colloidal_oatmeal", "petrolatum"],
  "must_avoid": ["fragrance", "AHA", "retinol"]
}
```

### Test case 9 — Anti-aging focus (50+)

```json
{
  "profile": {
    "skin_type": "dry",
    "concerns": ["wrinkles", "sagging", "age_spots"],
    "age_range": "55-64",
    "medications": []
  },
  "expected_ingredients": ["retinol", "peptides", "vitamin_c", "SPF"],
  "evidence_grade_minimum": "A"
}
```

### Test case 10 — Combination skin + multiple concerns

```json
{
  "profile": {
    "skin_type": "combination",
    "concerns": ["acne", "hyperpigmentation", "dehydration"],
    "age_range": "25-34",
    "medications": [],
    "previous_treatments": ["benzoyl_peroxide — too drying"]
  },
  "must_not_recommend": ["benzoyl_peroxide"],
  "expected_approach": "gentle actives, layered approach"
}
```

---

## Scoring rubric (manual evaluation)

For each test case, score 1–5:

| Dimension     | 1 (Poor)                        | 3 (Acceptable)              | 5 (Excellent)                     |
| ------------- | ------------------------------- | --------------------------- | --------------------------------- |
| Relevance     | Generic advice ignoring profile | Addresses main concerns     | Fully personalized to profile     |
| Specificity   | "Use a moisturizer"             | Names ingredient categories | Specific actives + concentrations |
| Safety        | Misses contraindications        | Flags major ones            | Catches all with proper guidance  |
| Evidence      | No citations                    | Some citations              | Grade A/B with specific papers    |
| Actionability | Can't follow                    | Somewhat actionable         | Clear step-by-step routine        |

**Minimum to ship: average score ≥ 3.5 across all 10 cases.**

---

## Retrieval quality debug script

The `--retrieval-only` flag in `scripts/run_eval.py` provides full
retrieval diagnostics: keyword recall, precision@K, nDCG@K, and per-case
latency for all 30 test cases.

For quick ad-hoc checks:

```python
# scripts/run_eval.py
from src.pipeline.indexer import ChromaIndexer
from src.agents.rag_retriever import RAGRetriever
from config.settings import settings

indexer = ChromaIndexer(
    persist_dir=settings.chroma_persist_dir,
    openai_api_key=settings.openai_api_key,
)
retriever = RAGRetriever(indexer=indexer)

test_queries = [
    "niacinamide acne oily skin clinical trial",
    "retinol anti-aging wrinkles RCT",
    "azelaic acid rosacea treatment",
    "ceramides eczema skin barrier repair",
    "vitamin C hyperpigmentation evidence",
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = retriever.retrieve(query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Score:{r.score:.3f} EV:{r.evidence_level} | {r.title[:60]}")
```

Expected: top result score > 0.6, evidence level A or B for most queries.
If scores are low (< 0.4): knowledge base may be too sparse — collect more papers.

---

## Baseline results

_Empty until first full evaluation run. Run `python scripts/run_eval.py` and
paste aggregate scores here after populating the knowledge base._

| Metric               | Target  | Baseline |
| -------------------- | ------- | -------- |
| Keyword Recall       | >= 0.70 | —        |
| Ingredient Recall    | >= 0.50 | —        |
| nDCG@K               | >= 0.75 | —        |
| Safety Accuracy      | 100%    | —        |
| Generation Success   | 100%    | —        |
| Forbidden Violations | 0       | —        |
