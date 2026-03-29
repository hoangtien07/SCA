"""
src/pipeline/indexer.py

Embeds chunks and stores in ChromaDB (local dev) or Qdrant (production).

Two backends:
    ChromaIndexer   — zero-config local, ideal for development
    QdrantIndexer   — production-ready, supports filtering + cloud

Both expose the same interface:
    indexer.add(chunks)
    indexer.query(text, filters, top_k) -> list[Chunk]

Prompt #04 upgrade:
    - Both indexers now accept an optional `embedder` (BaseEmbedder) in __init__
    - When embedder is provided, it is used instead of the built-in embedding function
    - model_name is stored in collection metadata
    - ValueError raised if existing collection used a different model
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from src.pipeline.chunker import Chunk

if TYPE_CHECKING:
    from src.pipeline.embedder import BaseEmbedder


# ── ChromaDB Indexer (local dev) ──────────────────────────────────────────────

class ChromaIndexer:
    """
    Persists embeddings locally via ChromaDB.
    No API key required. Data lives in ./data/knowledge_base/

    Usage:
        indexer = ChromaIndexer()
        indexer.add(chunks)
        results = indexer.query("retinol for acne", top_k=5)

    With custom embedder (Prompt #04):
        from src.pipeline.embedder import get_embedder
        embedder = get_embedder(settings)
        indexer = ChromaIndexer(embedder=embedder)
    """

    COLLECTION_NAME = "skincare_papers"

    def __init__(
        self,
        persist_dir: str = "./data/knowledge_base",
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: str = "",
        embedder: "BaseEmbedder | None" = None,
    ):
        import chromadb
        from chromadb.utils import embedding_functions

        self.client = chromadb.PersistentClient(path=persist_dir)
        self._custom_embedder = embedder

        if embedder is not None:
            # Use custom embedder — wrap it as a ChromaDB-compatible function
            _emb = embedder
            class _CustomEmbedFn:
                def __call__(self, input: list[str]) -> list[list[float]]:
                    return _emb.embed(input)

            self.embed_fn = _CustomEmbedFn()
            resolved_model_name = embedder.model_name
        else:
            # Default: OpenAI embeddings
            self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name=embedding_model,
            )
            resolved_model_name = embedding_model

        # Check for model mismatch in existing collection
        existing_collections = [c.name for c in self.client.list_collections()]
        if self.COLLECTION_NAME in existing_collections:
            existing_coll = self.client.get_collection(name=self.COLLECTION_NAME)
            stored_model = (existing_coll.metadata or {}).get("embedding_model", "")
            if stored_model and stored_model != resolved_model_name:
                raise ValueError(
                    f"[ChromaIndexer] Embedding model mismatch! "
                    f"Collection was built with '{stored_model}' but "
                    f"you are using '{resolved_model_name}'. "
                    "Use --reindex to rebuild the collection from scratch."
                )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": resolved_model_name,
            },
        )
        logger.info(
            f"[ChromaIndexer] Collection '{self.COLLECTION_NAME}' "
            f"has {self.collection.count()} chunks (model: {resolved_model_name})"
        )

    def add(self, chunks: list[Chunk], batch_size: int = 50) -> None:
        """Embed and add chunks in batches (avoids rate limits)."""
        # Filter already-indexed chunks
        existing_ids = set(self.collection.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("[ChromaIndexer] All chunks already indexed, skipping.")
            return

        logger.info(f"[ChromaIndexer] Indexing {len(new_chunks)} new chunks...")

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.to_chroma_dict() for c in batch],
            )
            logger.info(
                f"[ChromaIndexer] Batch {i // batch_size + 1} / "
                f"{(len(new_chunks) - 1) // batch_size + 1} indexed"
            )

        logger.info(f"[ChromaIndexer] Done. Total chunks: {self.collection.count()}")

    def query(
        self,
        query_text: str,
        top_k: int = 8,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Semantic search with optional metadata filters.

        filters example:
            {"evidence_level": {"$in": ["A", "B"]}}
            {"skin_conditions": {"$contains": "acne"}}
        """
        kwargs: dict = {
            "query_texts": [query_text],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)

        return [
            {
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,          # cosine distance → similarity score
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def stats(self) -> dict:
        return {
            "total_chunks": self.collection.count(),
            "collection": self.COLLECTION_NAME,
        }


# ── Qdrant Indexer (production) ───────────────────────────────────────────────

class QdrantIndexer:
    """
    Production indexer using Qdrant Cloud.
    Supports advanced filtering, scalable to millions of chunks.

    Usage:
        indexer = QdrantIndexer(url="https://xxx.qdrant.io", api_key="...")
        indexer.add(chunks)
        results = indexer.query("niacinamide oily skin", top_k=8)
    """

    COLLECTION_NAME = "skincare_papers"
    VECTOR_SIZE = 1536                      # text-embedding-3-small output dim

    def __init__(
        self,
        url: str,
        api_key: str,
        embedding_model: str = "text-embedding-3-small",
        openai_api_key: str = "",
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from openai import OpenAI

        self.client = QdrantClient(url=url, api_key=api_key)
        self.openai = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model

        # Create collection if not exists
        existing = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"[QdrantIndexer] Created collection '{self.COLLECTION_NAME}'")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.openai.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [r.embedding for r in resp.data]

    def add(self, chunks: list[Chunk], batch_size: int = 50) -> None:
        from qdrant_client.models import PointStruct

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = self._embed([c.text for c in batch])
            points = [
                PointStruct(
                    id=abs(hash(c.chunk_id)) % (2**63),
                    vector=vec,
                    payload=c.to_chroma_dict(),
                )
                for c, vec in zip(batch, vectors)
            ]
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points,
            )
            logger.info(f"[QdrantIndexer] Batch {i // batch_size + 1} upserted")

    def query(
        self,
        query_text: str,
        top_k: int = 8,
        filters: dict | None = None,
    ) -> list[dict]:
        from qdrant_client.models import Filter

        query_vector = self._embed([query_text])[0]
        qdrant_filter = Filter(**filters) if filters else None

        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {
                "text": r.payload.get("text", ""),
                "metadata": r.payload,
                "score": r.score,
            }
            for r in results
        ]
