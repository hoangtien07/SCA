"""
src/cache

Semantic caching layer for RAG retrieval and regimen generation.

Layer 1: RAG retrieval cache (TTL 24h)
Layer 2: Regimen generation cache (TTL 12h)

Requires Redis. Falls back to no-op silently when Redis is unavailable.
Never caches pregnancy=True requests.
"""
