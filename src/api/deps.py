"""
src/api/deps.py

Dependency injection for FastAPI routes.
Initializes shared agent instances once and provides them via Depends().
"""
from __future__ import annotations

from functools import lru_cache

from loguru import logger

from config.settings import settings


@lru_cache(maxsize=1)
def get_indexer():
    """ChromaDB indexer — singleton."""
    from src.pipeline.indexer import ChromaIndexer
    return ChromaIndexer(
        persist_dir=settings.chroma_persist_dir,
        openai_api_key=settings.openai_api_key,
    )


@lru_cache(maxsize=1)
def get_bm25():
    """BM25 index — singleton; returns None if index file doesn't exist."""
    from pathlib import Path
    from src.pipeline.bm25_index import BM25Index

    bm25_path = Path(settings.chroma_persist_dir) / "bm25_index.json"
    if bm25_path.exists():
        return BM25Index.load(str(bm25_path))
    logger.warning("[deps] BM25 index not found, sparse search disabled")
    return None


@lru_cache(maxsize=1)
def get_retriever():
    """RAG retriever — singleton."""
    from src.agents.rag_retriever import RAGRetriever
    return RAGRetriever(
        indexer=get_indexer(),
        bm25=get_bm25(),
        top_k=settings.retrieval_top_k,
    )


@lru_cache(maxsize=1)
def get_generator():
    """Regimen generator — singleton."""
    from src.agents.regimen_generator import RegimenGenerator
    return RegimenGenerator(
        api_key=settings.anthropic_api_key,
        model=settings.reasoning_model,
    )


@lru_cache(maxsize=1)
def get_vision_analyzer():
    """Vision analyzer — singleton."""
    from src.agents.vision_analyzer import VisionAnalyzer
    return VisionAnalyzer(
        api_key=settings.openai_api_key,
        model=settings.vision_model,
    )


@lru_cache(maxsize=1)
def get_safety_guard():
    """Safety guard — singleton."""
    from src.agents.safety_guard import SafetyGuard
    return SafetyGuard()
