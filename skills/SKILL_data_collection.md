# SKILL: Data Collection — Scientific Literature for Skincare AI

## Purpose
Collect, filter, and normalize peer-reviewed skincare research papers
from public APIs (Semantic Scholar, PubMed) into a structured JSONL format
ready for downstream processing.

## When to use this skill
- Adding new topic areas to the knowledge base
- Refreshing the knowledge base with recent publications
- Expanding coverage to new skin conditions or ingredients
- Debugging low retrieval quality (may indicate sparse coverage)

---

## API reference

### Semantic Scholar
```
Base URL: https://api.semanticscholar.org/graph/v1/paper/search
Rate limit: 100 req / 5 min (no key) | 1 req / 0.1s (with key)
Key signup: https://www.semanticscholar.org/product/api
Fields: paperId, title, abstract, year, authors, venue, citationCount, openAccessPdf
```

Example query:
```python
params = {
    "query": "retinol acne treatment randomized trial",
    "fields": "paperId,title,abstract,year,authors,venue,citationCount",
    "limit": 100,
    "offset": 0,
}
```

### PubMed E-utilities
```
Base URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
Rate limit: 3 req/s (no key) | 10 req/s (with key)
Key signup: https://www.ncbi.nlm.nih.gov/account/
Step 1: esearch.fcgi  → get PMIDs
Step 2: efetch.fcgi   → get full records (XML)
```

---

## Paper quality filters

Always apply these filters before indexing:
```python
def is_quality_paper(paper: Paper) -> bool:
    return (
        len(paper.abstract) > 100          # Has meaningful abstract
        and paper.year and paper.year >= 2000    # Not too old
        and (
            paper.citation_count >= 5      # Has some validation
            or paper.year >= 2022          # OR very recent
        )
    )
```

Evidence hierarchy for skincare:
```
Grade A: RCT, meta-analysis, systematic review
Grade B: cohort study, non-randomized trial, review article
Grade C: case report, expert opinion, in-vitro, animal study
```

---

## Search query best practices

**DO:**
- Include condition + intervention + study type: `"acne vulgaris" retinol "randomized trial"`
- Use MeSH terms for PubMed: `acne vulgaris[MeSH] adapalene`
- Search ingredient + skin type: `"niacinamide" "oily skin" clinical`

**DON'T:**
- Use overly broad queries: `"skincare"` → too much noise
- Skip pagination: most queries return 200–1000 papers
- Ignore citation count: filter `citationCount >= 5` for reliability

**Key query clusters to always cover:**
1. Condition-specific (acne, rosacea, eczema, melasma, aging, psoriasis)
2. Ingredient-specific (retinoids, niacinamide, vitamin C, AHA/BHA, ceramides, SPF)
3. Technology (AI skin analysis, computer vision dermatology, personalized cosmetics)
4. Mechanism (skin barrier, microbiome, sebum, collagen, melanin)

---

## Data schema (Paper object)

```python
@dataclass
class Paper:
    paper_id: str          # "ss_{id}" or "pm_{pmid}"
    title: str
    abstract: str
    year: int | None
    authors: list[str]
    journal: str
    doi: str
    url: str
    citation_count: int
    source: str            # "semantic_scholar" | "pubmed"
    skin_conditions: list[str]    # tagged in pipeline phase
    active_ingredients: list[str] # tagged in pipeline phase
    evidence_level: str    # A | B | C — tagged in pipeline phase
    study_type: str        # RCT | review | cohort | research_article | ...
```

---

## Common issues and fixes

| Issue | Likely cause | Fix |
|-------|-------------|-----|
| 0 results returned | Query too specific / typo | Broaden query, check spelling |
| Rate limit 429 error | Too fast | Add `time.sleep(1.5)` between calls |
| Many irrelevant papers | Query too broad | Add condition + intervention keywords |
| Missing recent papers | Year filter too strict | Allow `year >= 2020` minimum |
| Empty abstracts | Some conference papers | Filter `len(abstract) > 100` |

---

## Improving this skill with AI

Prompt template to expand search queries:
```
You are a dermatology research librarian.
Given these existing search queries:
{existing_queries}

Generate 10 additional queries that would find papers about:
{new_topic}

Rules:
- Include MeSH terms where appropriate for PubMed
- Mix broad and specific queries
- Include study type modifiers (randomized, systematic review, clinical trial)
- Output as a YAML list
```

Prompt template to improve metadata tagging:
```
You are a dermatology AI. Given this paper abstract:
{abstract}

Extract:
1. skin_conditions: list from [acne, pigmentation, aging, barrier_damage,
   redness_vascular, eczema_dermatitis, psoriasis, texture, oiliness, sensitivity]
2. active_ingredients: list of mentioned skincare actives (canonical names)
3. study_type: one of [RCT, meta_analysis, systematic_review, review, cohort,
   case_report, research_article, in_vitro]

Return JSON only.
```
