# GraphRAG Design — Skincare AI Knowledge Graph

## Overview

The knowledge graph adds **structured relational reasoning** on top of the existing vector similarity search. While the vector knowledge base excels at semantic similarity ("find papers about niacinamide for acne"), the knowledge graph encodes hard rules ("retinol CONFLICTS_WITH AHA — irritation risk").

**Status:** Stub implementation (v1). Full Neo4j integration planned for v2.0.

---

## Data Sources

| Source | Content | Format | License |
|--------|---------|--------|---------|
| `config/skin_conditions.yaml` | Ingredient interactions, concentrations, INCI synonyms | YAML | Internal |
| Open Beauty Facts | 3M+ cosmetic products + INCI ingredient lists | Parquet | ODbL |
| EU CosIng | Cosmetic ingredient functions, restrictions, EC/CAS numbers | REST API | Public |
| PubMed / Semantic Scholar | Evidence-backed ingredient-condition relationships | JSONL | Various |

---

## Neo4j Schema

### Node Labels

```cypher
(:Ingredient {
    name: String,             -- INCI name (lowercase, spaces)
    category: String,         -- retinoid | AHA | antioxidant | humectant | ...
    otc_max_concentration: String,  -- e.g. "1%"
    concentration_note: String,
    prohibited: Boolean       -- EU CosIng prohibited flag
})

(:Condition {
    name: String,             -- acne | rosacea | eczema | hyperpigmentation | ...
    subtypes: List<String>,
    keywords: List<String>
})

(:Product {
    id: String,               -- Open Beauty Facts barcode
    name: String,
    brand: String,
    categories: List<String>
})
```

### Relationship Types

```cypher
(:Ingredient)-[:CONFLICTS_WITH {
    reason: String,           -- e.g. "irritation risk — use on alternating nights"
    severity: String          -- warning | caution | info
}]->(:Ingredient)

(:Ingredient)-[:CONTRAINDICATED_IN {
    reason: String,           -- e.g. "pregnancy safety"
    category: String          -- pregnancy | pediatric | geriatric | drug_interaction
}]->(:Condition)

(:Ingredient)-[:TREATS_CONDITION {
    evidence_grade: String,   -- A | B | C
    mechanism: String,
    typical_concentration: String
}]->(:Condition)

(:Ingredient)-[:SYNERGISTIC_WITH {
    reason: String
}]->(:Ingredient)

(:Product)-[:CONTAINS {
    inci_name: String,
    position: Integer         -- INCI list order (1 = highest concentration)
}]->(:Ingredient)
```

---

## Sample Queries

### Find all conflicts for retinol
```cypher
MATCH (i:Ingredient {name: 'retinol'})-[r:CONFLICTS_WITH]->(other)
RETURN i.name, type(r), other.name, r.reason
```

### Find evidence-A ingredients for acne treatment
```cypher
MATCH (i:Ingredient)-[r:TREATS_CONDITION {evidence_grade: 'A'}]->(c:Condition {name: 'acne'})
RETURN i.name, r.mechanism, r.typical_concentration
ORDER BY i.name
```

### Check if ingredient is safe during pregnancy
```cypher
MATCH (i:Ingredient {name: 'retinol'})-[r:CONTRAINDICATED_IN]->(c:Condition {name: 'pregnancy'})
RETURN r.reason
```

### Find products containing niacinamide without parabens
```cypher
MATCH (p:Product)-[:CONTAINS]->(n:Ingredient {name: 'niacinamide'})
WHERE NOT EXISTS {
    MATCH (p)-[:CONTAINS]->(x:Ingredient)
    WHERE x.name CONTAINS 'paraben'
}
RETURN p.name, p.brand
LIMIT 20
```

### Graph path: ingredient → conditions it treats
```cypher
MATCH path = (i:Ingredient)-[:TREATS_CONDITION*1..2]->(c:Condition)
WHERE i.name = 'azelaic acid'
RETURN path
```

---

## Integration Point

The `GraphRetriever` in `src/agents/graph_retriever.py` is the bridge between
the vector knowledge base and the knowledge graph.

### Pipeline position (Layer 2.5)

```
[RAGRetriever] → vector results
       ↓
[GraphRetriever.augment_retrieval_results()] ← Layer 2.5
       ↓
[RegimenGenerator] ← richer context: vector evidence + graph facts
```

### Code example (once Neo4j is running)

```python
from src.agents.rag_retriever import RAGRetriever
from src.agents.graph_retriever import GraphRetriever

retriever = RAGRetriever(indexer=indexer, bm25=bm25)
rag_results = retriever.retrieve(query, skin_conditions=["acne"])

# Layer 2.5: augment with graph facts
graph = GraphRetriever(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password",
    enabled=True,
)
augmented_results = graph.augment_retrieval_results(rag_results, profile)

# Generator receives both vector evidence and structured graph facts
regimen = generator.generate(profile, augmented_results)
```

---

## Setup Instructions

### 1. Install Neo4j

**Option A: Docker (recommended for development)**
```bash
docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/your_password \
    neo4j:5.15
```

**Option B: Neo4j AuraDB (cloud, free tier)**
- Visit https://neo4j.com/cloud/platform/aura-graph-database/
- Create a free instance
- Copy URI, username, password to .env

### 2. Install Python driver
```bash
pip install neo4j>=5.0.0
```

### 3. Seed the graph
```bash
# Download cosmetic data first (optional, for Product nodes)
python scripts/download_cosmetic_data.py --source openbeautyfacts

# Generate + run Cypher
python scripts/seed_graph.py
cypher-shell -u neo4j -p your_password --file data/graph/seed.cypher
```

### 4. Add to .env
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### 5. Enable in code
```python
from src.agents.graph_retriever import GraphRetriever
graph = GraphRetriever(
    neo4j_uri=settings.neo4j_uri,
    enabled=True,
)
```

---

## Fairness & Bias Considerations

The knowledge graph should reflect **evidence-based** ingredient-condition relationships,
not marketing claims. When adding new TREATS_CONDITION edges:

- Set `evidence_grade` to A/B/C based on clinical evidence level
- Cite the PubMed paper or systematic review in `mechanism` or a separate `citation` property
- Avoid adding relationships for ingredients with only in-vitro or animal evidence without flagging `evidence_grade: C`

---

## Future Work

- [ ] Full `GraphRetriever.get_ingredient_relations()` implementation
- [ ] Full `GraphRetriever.augment_retrieval_results()` implementation
- [ ] TREATS_CONDITION edges with evidence grades from PubMed metadata
- [ ] Product safety scoring using graph path analysis
- [ ] Graph embeddings (node2vec) for ingredient similarity
- [ ] Periodic sync from Open Beauty Facts (monthly dumps)
