"""
src/cache/semantic_cache.py

Semantic similarity-based caching for expensive AI pipeline steps.

Architecture:
    SemanticCache(redis_url, embedder, threshold=0.92)
        - get(query, cache_type)  → cached value or None
        - set(query, value, cache_type, ttl)
        - cache_stats()           → {hits, misses, hit_rate, total_keys}

Cache types:
    "retrieval"  — RAG results (TTL 24h)
    "regimen"    — Generated regimen (TTL 12h)

Similarity matching:
    - Query is embedded using the provided embedder
    - Cosine similarity computed against stored query vectors
    - Hit if similarity >= threshold (default 0.92)

Graceful degradation:
    - Redis unavailable → silently returns None / no-op
    - Embedding failure → silently returns None / no-op
    - Never caches pregnancy=True requests (enforced at usage sites)

Redis key schema:
    sca:cache:{cache_type}:vectors — sorted set of (vector_key → timestamp)
    sca:cache:{cache_type}:vec:{key} — stored query vector (JSON)
    sca:cache:{cache_type}:val:{key} — stored value (JSON)
    sca:cache:stats:hits — counter
    sca:cache:stats:misses — counter
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.pipeline.embedder import BaseEmbedder


class SemanticCache:
    """
    Semantic similarity cache backed by Redis.

    Usage:
        from config.settings import settings
        from src.pipeline.embedder import get_embedder
        from src.cache.semantic_cache import SemanticCache

        embedder = get_embedder(settings)
        cache = SemanticCache(
            redis_url=settings.redis_url,
            embedder=embedder,
            threshold=settings.cache_similarity_threshold,
        )

        # Check cache
        cached = cache.get("best niacinamide for oily skin", "retrieval")
        if cached is None:
            result = retriever.retrieve(query)
            cache.set(query, result, "retrieval", ttl=86400)
    """

    KEY_PREFIX = "sca:cache"

    def __init__(
        self,
        redis_url: str,
        embedder: "BaseEmbedder",
        threshold: float = 0.92,
    ):
        self._redis_url = redis_url
        self._embedder = embedder
        self._threshold = threshold
        self._redis = None
        self._available = False

        self._connect()

    def _connect(self) -> None:
        """Attempt to connect to Redis. Silently no-op if unavailable."""
        try:
            import redis
            client = redis.from_url(self._redis_url, socket_connect_timeout=2)
            client.ping()
            self._redis = client
            self._available = True
            logger.debug(f"[SemanticCache] Connected to Redis: {self._redis_url}")
        except Exception as e:
            logger.warning(f"[SemanticCache] Redis unavailable, caching disabled: {e}")
            self._available = False

    def get(self, query: str, cache_type: str) -> dict | None:
        """
        Look up query in cache. Returns cached value if similarity >= threshold.

        Args:
            query: The query string to look up
            cache_type: "retrieval" or "regimen"

        Returns:
            Cached value dict, or None if no similar query found
        """
        if not self._available:
            return None

        try:
            query_vec = self._embedder.embed_query(query)
            return self._find_similar(query_vec, cache_type)
        except Exception as e:
            logger.debug(f"[SemanticCache] get() error: {e}")
            self._increment_stat("misses")
            return None

    def set(
        self,
        query: str,
        value: dict,
        cache_type: str,
        ttl: int = 86400,
    ) -> None:
        """
        Store a query-value pair in the cache.

        Args:
            query: The query string
            value: The value to cache (must be JSON-serializable)
            cache_type: "retrieval" or "regimen"
            ttl: Time-to-live in seconds
        """
        if not self._available:
            return

        try:
            query_vec = self._embedder.embed_query(query)
            key = self._make_key(query)

            pipe = self._redis.pipeline()

            # Store vector
            vec_key = f"{self.KEY_PREFIX}:{cache_type}:vec:{key}"
            pipe.setex(vec_key, ttl, json.dumps(query_vec))

            # Store value
            val_key = f"{self.KEY_PREFIX}:{cache_type}:val:{key}"
            pipe.setex(val_key, ttl, json.dumps(value))

            # Track keys for similarity search (use a set of active keys)
            index_key = f"{self.KEY_PREFIX}:{cache_type}:index"
            pipe.sadd(index_key, key)
            pipe.expire(index_key, ttl)

            pipe.execute()
            logger.debug(f"[SemanticCache] Cached {cache_type} query: {query[:60]}...")

        except Exception as e:
            logger.debug(f"[SemanticCache] set() error: {e}")

    def cache_stats(self) -> dict:
        """
        Return cache statistics.

        Returns:
            {hits, misses, hit_rate, total_keys}
        """
        if not self._available:
            return {"available": False, "hits": 0, "misses": 0, "hit_rate": 0.0, "total_keys": 0}

        try:
            hits = int(self._redis.get(f"{self.KEY_PREFIX}:stats:hits") or 0)
            misses = int(self._redis.get(f"{self.KEY_PREFIX}:stats:misses") or 0)
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0.0

            # Count cache keys
            all_keys = self._redis.keys(f"{self.KEY_PREFIX}:*:val:*")

            return {
                "available": True,
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hit_rate, 4),
                "total_keys": len(all_keys),
            }
        except Exception as e:
            logger.debug(f"[SemanticCache] cache_stats() error: {e}")
            return {"available": False, "hits": 0, "misses": 0, "hit_rate": 0.0, "total_keys": 0}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_similar(self, query_vec: list[float], cache_type: str) -> dict | None:
        """
        Search Redis for a cached query vector with cosine similarity >= threshold.
        """
        import numpy as np

        index_key = f"{self.KEY_PREFIX}:{cache_type}:index"
        keys = self._redis.smembers(index_key)
        if not keys:
            self._increment_stat("misses")
            return None

        query_arr = np.array(query_vec, dtype=float)
        query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-10)

        best_sim = -1.0
        best_key = None

        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            vec_key = f"{self.KEY_PREFIX}:{cache_type}:vec:{key_str}"
            raw = self._redis.get(vec_key)
            if raw is None:
                # Expired — clean up index
                self._redis.srem(index_key, key_str)
                continue

            stored_vec = json.loads(raw)
            stored_arr = np.array(stored_vec, dtype=float)
            stored_norm = stored_arr / (np.linalg.norm(stored_arr) + 1e-10)

            sim = float(np.dot(query_norm, stored_norm))
            if sim > best_sim:
                best_sim = sim
                best_key = key_str

        if best_sim >= self._threshold and best_key is not None:
            val_key = f"{self.KEY_PREFIX}:{cache_type}:val:{best_key}"
            raw_val = self._redis.get(val_key)
            if raw_val is not None:
                self._increment_stat("hits")
                logger.debug(f"[SemanticCache] Cache HIT (sim={best_sim:.4f})")
                return json.loads(raw_val)

        self._increment_stat("misses")
        return None

    def _make_key(self, query: str) -> str:
        """Generate a short deterministic key from query text."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _increment_stat(self, stat: str) -> None:
        """Increment a counter stat in Redis."""
        try:
            self._redis.incr(f"{self.KEY_PREFIX}:stats:{stat}")
        except Exception:
            pass
