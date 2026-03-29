"""
src/workers

Celery async task queue for long-running regimen generation.
Requires Redis: redis://localhost:6379/0 (set REDIS_URL in .env)

Workers are opt-in — the synchronous /full-pipeline endpoint continues to work
without Redis.
"""
