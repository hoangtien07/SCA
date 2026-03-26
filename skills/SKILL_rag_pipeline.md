# SKILL: RAG Pipeline — Skincare Knowledge Base

## Purpose
Transform raw Paper objects into a queryable vector knowledge base.
Covers chunking strategy, embedding, indexing, and hybrid retrieval.

## Pipeline stages
```
Papers (JSONL) → Metadata Tagger → Chunker → Embedder → Vector DB → Retriever
```

---

## Chunking strategy

### Why chunk by section, not fixed-size?
Scientific papers have a known structure. Section-aware chunking keeps
semantically coherent units together and improves retrieval precision.

```
Abstract  → always 1 chunk (high signal, usually <300 tokens)
Methods   → 1–3 chunks (technique details, study design)
Results   → 1–3 chunks (key findings, statistics)
Conclusion→ 1 chunk (summary of evidence)
```

If full text is not available (most cases — only abstract):
```
title_abstract chunk → 1 chunk with title prepended for context
```

### Token sizing
```
chunk_size = 512 tokens    ← sweet spot for text-embedding-3-small
overlap = 64 tokens        ← preserves cross-boundary context
```

Prepend title to every chunk:
```python
text = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
```
This ensures every chunk is self-contained and title keywords
are always present for keyword matching.

---

## Embedding model selection

| Model | Dims | Cost | Best for |
|-------|------|------|---------|
| `text-embedding-3-small` | 1536 | $0.02/1M tokens | ✅ Default — best cost/quality |
| `text-embedding-3-large` | 3072 | $0.13/1M tokens | Higher accuracy, 6.5× cost |
| `BioBERT` (self-hosted) | 768 | Free | Domain-specific, needs GPU |

For 1,000 papers × 1.5 chunks avg × 400 tokens avg:
→ 600,000 tokens → **$0.012 total** with text-embedding-3-small

---

## Metadata schema for ChromaDB

ChromaDB requires flat metadata (no nested lists).
Lists are stored as comma-separated strings:

```python
metadata = {
    "paper_id": "ss_abc123",
    "title": "Niacinamide for acne...",
    "year": 2022,
    "journal": "JAAD",
    "evidence_level": "A",
    "study_type": "RCT",
    "skin_conditions": "acne,oiliness",      # comma-separated
    "active_ingredients": "niacinamide,zinc", # comma-separated
    "citation_count": 47,
    "source": "semantic_scholar",
    "url": "https://doi.org/...",
}
```

At retrieval, split back to lists:
```python
conditions = metadata["skin_conditions"].split(",")
```

---

## Hybrid retrieval algorithm

Pure semantic search misses exact ingredient mentions.
Hybrid = semantic + metadata pre-filter + citation re-ranking.

```
Score = semantic_score × 0.7
      + citation_boost × 0.2     # (citations / max_citations)
      + evidence_boost × 0.1     # A=0.3, B=0.2, C=0.0
```

### Filter examples (ChromaDB where-clause syntax)

Filter by evidence level:
```python
{"evidence_level": {"$in": ["A", "B"]}}
```

Filter by skin condition (string contains):
```python
{"skin_conditions": {"$contains": "acne"}}
```

Combined (AND):
```python
{
    "$and": [
        {"evidence_level": {"$in": ["A", "B"]}},
        {"year": {"$gte": 2015}},
    ]
}
```

---

## Query construction from skin profile

Build a retrieval query from profile fields:
```python
def build_query(profile: dict) -> str:
    parts = []
    parts.append(profile["skin_type"] + " skin")
    parts.extend(profile["concerns"][:3])   # top 3 concerns
    parts.append(profile.get("primary_goal", ""))
    return " ".join(parts) + " treatment evidence"

# Example output:
# "oily skin acne dark spots clear skin treatment evidence"
```

---

## Retrieval quality checklist

Run this manually to debug low-quality retrieval:

```python
# Test query
results = retriever.retrieve(
    query="niacinamide acne oily skin",
    evidence_levels=["A", "B"],
    top_k=5,
)

for r in results:
    print(f"Score: {r.score:.3f} | EV: {r.evidence_level} | {r.title[:60]}")
    print(f"  Conditions: {r.skin_conditions}")
    print()
```

Expected: top results should be relevant to both niacinamide AND acne.
If not: check that tagging is working, re-run `run_indexing.py`.

---

## Improving retrieval with AI

Prompt for query expansion at retrieval time:
```
Given this skin profile:
{profile_json}

Generate 3 different search queries optimized for retrieving
evidence-based skincare recommendations. Each query should emphasize
a different aspect: conditions, ingredients, mechanisms.

Format: JSON array of 3 strings.
```

Use all 3 queries, retrieve top-K each, merge + deduplicate by paper_id,
then re-rank the combined set. This significantly improves recall.
