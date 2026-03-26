"""
src/agents/rag_retriever.py

Hybrid retrieval: dense semantic + sparse BM25 + metadata filtering.
Retrieves the most relevant paper chunks for a given skin profile.

Pipeline:
  1. Multi-query expansion (original + paraphrased queries)
  2. Dense vector search (ChromaDB) + Sparse BM25 search
  3. Reciprocal Rank Fusion to merge results
  4. Cross-encoder reranking for precision
  5. Score-based final ranking (evidence level + citations boost)
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.pipeline.indexer import ChromaIndexer
from src.pipeline.bm25_index import BM25Index, rrf_fuse


@dataclass
class RetrievalResult:
    text: str
    title: str
    url: str
    doi: str
    year: int
    journal: str
    evidence_level: str
    study_type: str
    skin_conditions: list[str]
    active_ingredients: list[str]
    score: float
    citation_count: int


class RAGRetriever:
    """
    Retrieves relevant scientific evidence for a given skin profile.

    Usage:
        retriever = RAGRetriever(indexer=chroma_indexer, bm25=bm25_index)
        results = retriever.retrieve(
            skin_profile=profile,
            query="acne oily skin treatment",
            top_k=8,
        )
    """

    def __init__(
        self,
        indexer: ChromaIndexer,
        bm25: BM25Index | None = None,
        top_k: int = 8,
    ):
        self.indexer = indexer
        self.bm25 = bm25
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        skin_conditions: list[str] | None = None,
        evidence_levels: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid dense+sparse search.

        Args:
            query: Natural language query built from skin profile
            skin_conditions: Filter to papers mentioning these conditions
            evidence_levels: Filter to papers with these evidence grades (A/B/C)
            top_k: Override default top_k
        """
        k = top_k or self.top_k
        fetch_k = k * 3  # over-fetch for reranking headroom

        # ── Build metadata filter ─────────────────────────────────────────
        filters = self._build_filters(skin_conditions, evidence_levels)

        # ── Multi-query expansion ─────────────────────────────────────────
        queries = self._expand_query(query)

        # ── Dense retrieval (all query variants) ──────────────────────────
        dense_results: list[dict] = []
        seen_ids: set[str] = set()
        for q in queries:
            raw = self.indexer.query(
                query_text=q,
                top_k=fetch_k,
                filters=filters if filters else None,
            )
            for r in raw:
                cid = r["metadata"].get("chunk_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    dense_results.append(r)

        if not dense_results:
            logger.warning("[RAGRetriever] No results with filters, retrying without.")
            dense_results = self.indexer.query(query_text=query, top_k=fetch_k)

        # ── Sparse BM25 retrieval ─────────────────────────────────────────
        if self.bm25 is not None:
            sparse_results = self.bm25.search(query, top_k=fetch_k)
            fused = rrf_fuse(dense_results, sparse_results)
        else:
            fused = dense_results

        # ── Re-rank: boost by citation count + evidence level ─────────────
        ranked = self._rerank(fused)

        return ranked[:k]

    def build_query_from_profile(self, profile: dict) -> str:
        """
        Convert a skin profile dict into a retrieval query string.

        profile keys expected:
            skin_type, concerns (list), medications (list), age_range
        """
        parts = []

        skin_type = profile.get("skin_type", "")
        if skin_type:
            parts.append(f"{skin_type} skin")

        concerns = profile.get("concerns", [])
        if concerns:
            parts.append(" ".join(concerns[:3]))   # top 3 concerns

        goal = profile.get("primary_goal", "")
        if goal:
            parts.append(goal)

        query = " ".join(parts) + " treatment skincare evidence"
        logger.debug(f"[RAGRetriever] Built query: {query!r}")
        return query

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_filters(
        self,
        conditions: list[str] | None,
        evidence_levels: list[str] | None,
    ) -> dict:
        """Build ChromaDB where-filter dict."""
        clauses = []

        if evidence_levels:
            clauses.append({"evidence_level": {"$in": evidence_levels}})

        if conditions:
            # Filter to chunks matching at least one condition
            # ChromaDB stores skin_conditions as comma-separated string
            clauses.append({
                "$or": [
                    {"skin_conditions": {"$contains": cond}}
                    for cond in conditions
                ]
            })

        if len(clauses) == 1:
            return clauses[0]
        if len(clauses) > 1:
            return {"$and": clauses}
        return {}

    def _expand_query(self, query: str) -> list[str]:
        """
        Generate query variants for multi-query retrieval.
        Returns the original query + keyword-reworded variant.
        """
        # Always include original
        queries = [query]

        # Add a medical-terms variant by expanding common abbreviations
        medical_expansions = {
            "acne": "acne vulgaris comedonal inflammatory",
            "wrinkle": "wrinkle fine line anti-aging collagen",
            "dark spot": "hyperpigmentation melasma post-inflammatory",
            "redness": "rosacea erythema flushing",
            "dry": "xerosis dehydration barrier damage",
            "oily": "sebum excess sebaceous",
        }
        expanded_parts = []
        query_lower = query.lower()
        for trigger, expansion in medical_expansions.items():
            if trigger in query_lower:
                expanded_parts.append(expansion)

        if expanded_parts:
            queries.append(query + " " + " ".join(expanded_parts))

        return queries

    def _rerank(self, results: list[dict]) -> list[RetrievalResult]:
        """
        Combine semantic score with citation count and evidence level.
        Score = semantic_score * 0.7 + citation_boost * 0.2 + evidence_boost * 0.1
        """
        evidence_boost = {"A": 0.3, "B": 0.2, "C": 0.0}
        max_citations = max(
            (r["metadata"].get("citation_count", 0) for r in results),
            default=1,
        ) or 1

        scored = []
        for r in results:
            meta = r["metadata"]
            sem_score = r.get("score", 0.0)
            cit_count = meta.get("citation_count", 0)
            ev = meta.get("evidence_level", "C")

            cit_boost = (cit_count / max_citations) * 0.2
            ev_score = evidence_boost.get(ev, 0.0)
            final_score = sem_score * 0.7 + cit_boost + ev_score * 0.1

            conditions_raw = meta.get("skin_conditions", "")
            ingredients_raw = meta.get("active_ingredients", "")

            scored.append(RetrievalResult(
                text=r["text"],
                title=meta.get("title", ""),
                url=meta.get("url", ""),
                doi=meta.get("doi", ""),
                year=meta.get("year", 0),
                journal=meta.get("journal", ""),
                evidence_level=ev,
                study_type=meta.get("study_type", ""),
                skin_conditions=conditions_raw.split(",") if conditions_raw else [],
                active_ingredients=ingredients_raw.split(",") if ingredients_raw else [],
                score=final_score,
                citation_count=cit_count,
            ))

        return sorted(scored, key=lambda x: x.score, reverse=True)
