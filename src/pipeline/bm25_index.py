"""
src/pipeline/bm25_index.py

Sparse BM25 index for keyword-based retrieval.
Used alongside dense vector search (ChromaDB/Qdrant) in a hybrid pipeline.

Fusion strategy: Reciprocal Rank Fusion (RRF) to combine sparse + dense results.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

from src.pipeline.chunker import Chunk

# Lazy import — rank_bm25 only needed when building/querying
_bm25_cls = None


def _get_bm25():
    global _bm25_cls
    if _bm25_cls is None:
        from rank_bm25 import BM25Okapi
        _bm25_cls = BM25Okapi
    return _bm25_cls


# ── Simple tokenizer ─────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization (no stemming needed for medical terms)."""
    return _WORD_RE.findall(text.lower())


# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Index:
    """
    In-memory BM25 index over chunk texts.

    Usage:
        idx = BM25Index()
        idx.build(chunks)
        results = idx.search("retinol acne treatment", top_k=10)
        idx.save("data/knowledge_base/bm25_index.json")

        # Later:
        idx2 = BM25Index.load("data/knowledge_base/bm25_index.json")
    """

    def __init__(self):
        self._bm25 = None
        self._corpus: list[list[str]] = []
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from a list of Chunk objects."""
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._texts = [c.text for c in chunks]
        self._metadatas = [c.to_chroma_dict() for c in chunks]
        self._corpus = [_tokenize(t) for t in self._texts]

        BM25Okapi = _get_bm25()
        self._bm25 = BM25Okapi(self._corpus)
        logger.info(f"[BM25Index] Built index over {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search the BM25 index.

        Returns list of dicts with keys: text, metadata, score, chunk_id
        """
        if self._bm25 is None:
            logger.warning("[BM25Index] Index not built yet")
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                break
            results.append({
                "text": self._texts[idx],
                "metadata": self._metadatas[idx],
                "score": float(scores[idx]),
                "chunk_id": self._chunk_ids[idx],
            })

        return results

    def save(self, path: str) -> None:
        """Persist index data to JSON (BM25 model is rebuilt on load)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunk_ids": self._chunk_ids,
            "texts": self._texts,
            "metadatas": self._metadatas,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info(f"[BM25Index] Saved {len(self._chunk_ids)} entries to {path}")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load index data from JSON and rebuild BM25 model."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        idx = cls()
        idx._chunk_ids = data["chunk_ids"]
        idx._texts = data["texts"]
        idx._metadatas = data["metadatas"]
        idx._corpus = [_tokenize(t) for t in idx._texts]

        BM25Okapi = _get_bm25()
        idx._bm25 = BM25Okapi(idx._corpus)
        logger.info(f"[BM25Index] Loaded {len(idx._chunk_ids)} entries from {path}")
        return idx


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def rrf_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[dict]:
    """
    Reciprocal Rank Fusion to merge dense (vector) and sparse (BM25) results.

    RRF score = weight / (k + rank)  summed across result lists.

    Args:
        dense_results:  Results from vector search (must have 'metadata' with 'chunk_id')
        sparse_results: Results from BM25 search (must have 'chunk_id')
        k:              RRF constant (default 60, per original paper)
        dense_weight:   Weight for dense results
        sparse_weight:  Weight for sparse results

    Returns:
        Merged list sorted by fused score, with original metadata preserved.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    # Score dense results
    for rank, r in enumerate(dense_results, 1):
        cid = r["metadata"].get("chunk_id", f"dense_{rank}")
        scores[cid] = scores.get(cid, 0) + dense_weight / (k + rank)
        doc_map[cid] = r

    # Score sparse results
    for rank, r in enumerate(sparse_results, 1):
        cid = r.get("chunk_id", r["metadata"].get("chunk_id", f"sparse_{rank}"))
        scores[cid] = scores.get(cid, 0) + sparse_weight / (k + rank)
        if cid not in doc_map:
            doc_map[cid] = r

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused = []
    for cid, score in ranked:
        entry = doc_map[cid].copy()
        entry["score"] = score
        fused.append(entry)

    return fused
