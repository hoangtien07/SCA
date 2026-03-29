"""
src/workers/celery_app.py

Celery application configuration.
Uses Redis as both broker and result backend.

Graceful degradation: if Redis is unavailable at import time,
the module still imports successfully (workers simply won't work).

Usage:
    # Start worker
    celery -A src.workers.celery_app worker --loglevel=info

    # Start Flower monitoring UI
    celery -A src.workers.celery_app flower --port=5555
"""
from __future__ import annotations

from loguru import logger


def _create_celery_app():
    """Create and configure the Celery application."""
    try:
        from celery import Celery
        from config.settings import settings

        app = Celery(
            "skincare_ai",
            broker=settings.redis_url,
            backend=settings.redis_url,
        )

        app.conf.update(
            task_serializer="json",
            accept_content=["json"],
            result_serializer="json",
            timezone="UTC",
            enable_utc=True,
            task_track_started=True,
            task_acks_late=True,
            worker_prefetch_multiplier=1,
            result_expires=3600,  # results expire after 1 hour
            # Rate limiting for regimen generation (expensive LLM call)
            task_routes={
                "src.workers.tasks.generate_regimen_task": {
                    "queue": "regimen",
                    "rate_limit": "10/m",   # max 10 generations/minute per worker
                },
            },
        )

        logger.info(f"[celery_app] Celery configured with broker: {settings.redis_url}")
        return app

    except ImportError as e:
        logger.warning(f"[celery_app] Celery not available: {e}")
        return None


celery_app = _create_celery_app()
