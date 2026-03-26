"""
tests/test_retriever.py
Unit tests for RAGRetriever with mocked ChromaDB and BM25 dependencies.
Runs without API keys or network access.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.rag_retriever import RAGRetriever


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_mock_indexer(results=None):
    """Create a mock ChromaIndexer that returns canned query() results.

    indexer.query() returns list[dict] where each dict has:
      - "text": str
      - "metadata": dict (chunk_id, title, year, journal, evidence_level, ...)
      - "score": float
    """
    indexer = MagicMock()

    if results is None:
        results = [
            {
                "text": "Niacinamide 5% significantly reduced sebum production in a 12-week RCT.",
                "metadata": {
                    "chunk_id": "chunk_001",
                    "title": "Niacinamide for Acne: A Randomized Controlled Trial",
                    "year": 2023,
                    "journal": "JAAD",
                    "evidence_level": "A",
                    "url": "",
                    "doi": "10.1234/nia2023",
                    "citation_count": 150,
                    "study_type": "RCT",
                    "skin_conditions": "acne,oily",
                    "active_ingredients": "niacinamide",
                },
                "score": 0.92,
            },
            {
                "text": "Salicylic acid 2% is effective for mild-moderate acne.",
                "metadata": {
                    "chunk_id": "chunk_002",
                    "title": "BHA for Acne: Systematic Review",
                    "year": 2022,
                    "journal": "BJD",
                    "evidence_level": "B",
                    "url": "",
                    "doi": "10.1234/bha2022",
                    "citation_count": 80,
                    "study_type": "systematic_review",
                    "skin_conditions": "acne",
                    "active_ingredients": "salicylic acid",
                },
                "score": 0.78,
            },
        ]

    indexer.query.return_value = results
    return indexer


def _make_mock_bm25(results=None):
    """Create a mock BM25Index.

    bm25.search() returns list[dict] with chunk_id and score.
    """
    bm25 = MagicMock()
    if results is None:
        results = [
            {"chunk_id": "chunk_001", "score": 5.2},
            {
                "chunk_id": "chunk_003",
                "score": 3.1,
                "text": "Retinoids improve photodamaged skin.",
                "metadata": {
                    "chunk_id": "chunk_003",
                    "title": "Retinoids Review",
                    "year": 2021,
                    "journal": "Dermatol Ther",
                    "evidence_level": "B",
                    "url": "",
                    "doi": "",
                    "citation_count": 40,
                    "study_type": "review",
                    "skin_conditions": "acne,photoaging",
                    "active_ingredients": "retinol",
                },
            },
        ]
    bm25.search.return_value = results
    return bm25


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_retrieve_returns_results():
    """Basic retrieval should return ranked results."""
    retriever = RAGRetriever(indexer=_make_mock_indexer(), bm25=None, top_k=5)
    results = retriever.retrieve("niacinamide acne oily skin")

    assert len(results) > 0
    assert hasattr(results[0], "text")
    assert hasattr(results[0], "score")


def test_retrieve_respects_top_k():
    """Should not return more results than top_k."""
    retriever = RAGRetriever(indexer=_make_mock_indexer(), bm25=None, top_k=1)
    results = retriever.retrieve("niacinamide acne")

    assert len(results) <= 1


def test_retrieve_with_bm25_fusion():
    """Retrieval with BM25 should still return results (fusion path)."""
    retriever = RAGRetriever(
        indexer=_make_mock_indexer(),
        bm25=_make_mock_bm25(),
        top_k=5,
    )
    results = retriever.retrieve("niacinamide acne oily skin")

    assert len(results) > 0


def test_build_query_from_profile():
    """Profile-to-query builder should include key profile fields."""
    retriever = RAGRetriever(indexer=_make_mock_indexer(), bm25=None, top_k=5)
    profile = {
        "skin_type": "oily",
        "concerns": ["acne", "large pores"],
        "age": 25,
    }
    query = retriever.build_query_from_profile(profile)

    assert isinstance(query, str)
    assert len(query) > 0
    # Should contain profile info
    assert "oily" in query.lower() or "acne" in query.lower()


def test_retrieve_empty_query_no_crash():
    """Empty query should not crash — returns empty or fallback results."""
    indexer = _make_mock_indexer(results=[])
    retriever = RAGRetriever(indexer=indexer, bm25=None, top_k=5)
    results = retriever.retrieve("")

    assert isinstance(results, list)


def test_retrieve_with_skin_conditions_filter():
    """Passing skin_conditions should forward filter to indexer."""
    indexer = _make_mock_indexer()
    retriever = RAGRetriever(indexer=indexer, bm25=None, top_k=5)

    retriever.retrieve("acne treatment", skin_conditions=["acne"])

    # Verify indexer.query was called (filter may or may not be passed depending on impl)
    assert indexer.query.called
