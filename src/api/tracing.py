"""
src/api/tracing.py

Request-level observability for the skincare AI pipeline.

Assigns a trace_id per request and logs structured events for:
  - Retrieval (query, filters, result count, top scores)
  - Generation (model, input tokens estimate, latency)
  - Safety (flags triggered, severity counts)
  - Citation grounding (grounding rate, ungrounded list)

Uses loguru structured logging — compatible with any log aggregator
(Datadog, Grafana Loki, CloudWatch) via JSON output.

Optional LangSmith integration (requires langsmith_tracing=True in settings):
  - Wraps pipeline spans with @traceable decorator
  - Parent run named "full_pipeline" on the /full-pipeline route
  - Instruments: vision_analysis, rag_retrieval, regimen_generation, safety_check
  - Falls back to standard logging when tracing is disabled or SDK unavailable
"""
from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable

from loguru import logger

# Per-request trace ID (set by middleware, accessible anywhere in the call chain)
_trace_id: ContextVar[str] = ContextVar("trace_id", default="no-trace")


def get_trace_id() -> str:
    return _trace_id.get()


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


# ── LangSmith integration ─────────────────────────────────────────────────────

def _get_langsmith_settings():
    """Lazy-load settings to avoid circular imports."""
    from config.settings import settings
    return settings


def _is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is configured and enabled."""
    try:
        s = _get_langsmith_settings()
        return bool(s.langsmith_tracing and s.langsmith_api_key)
    except Exception:
        return False


def _configure_langsmith() -> bool:
    """
    Configure LangSmith environment variables if tracing is enabled.
    Returns True if successfully configured.
    """
    if not _is_tracing_enabled():
        return False
    try:
        import os
        s = _get_langsmith_settings()
        os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        return True
    except Exception as e:
        logger.warning(f"[tracing] LangSmith configuration failed: {e}")
        return False


def traceable_span(name: str):
    """
    Decorator factory that wraps a function in a LangSmith span named `name`.
    When tracing is disabled (default), acts as a no-op decorator.

    Usage:
        @traceable_span("rag_retrieval")
        def my_fn(query):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not _is_tracing_enabled():
                return fn(*args, **kwargs)
            try:
                from langsmith import traceable
                traced = traceable(name=name)(fn)
                return traced(*args, **kwargs)
            except ImportError:
                logger.debug("[tracing] langsmith not installed, skipping span")
                return fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"[tracing] LangSmith span '{name}' failed: {e}")
                return fn(*args, **kwargs)
        return wrapper
    return decorator


class LangSmithRun:
    """
    Context manager that creates a parent LangSmith run for the full pipeline.
    When tracing is disabled, acts as a no-op context manager.

    Usage:
        with LangSmithRun("full_pipeline", inputs={"profile": ...}) as run:
            ...
    """

    def __init__(self, name: str, inputs: dict | None = None):
        self.name = name
        self.inputs = inputs or {}
        self._client = None
        self._run_id = None

    def __enter__(self):
        if not _is_tracing_enabled():
            return self
        try:
            _configure_langsmith()
            from langsmith import Client
            self._client = Client()
            import uuid as _uuid
            self._run_id = str(_uuid.uuid4())
            self._client.create_run(
                name=self.name,
                run_type="chain",
                inputs=self.inputs,
                id=self._run_id,
            )
            logger.debug(f"[tracing] LangSmith run '{self.name}' started: {self._run_id}")
        except ImportError:
            logger.debug("[tracing] langsmith not installed")
        except Exception as e:
            logger.warning(f"[tracing] Failed to start LangSmith run: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._client and self._run_id:
            try:
                self._client.update_run(
                    run_id=self._run_id,
                    end_time=None,  # auto
                    error=str(exc_val) if exc_val else None,
                )
            except Exception as e:
                logger.debug(f"[tracing] Failed to end LangSmith run: {e}")
        return False  # don't suppress exceptions


# ── Pipeline tracer ───────────────────────────────────────────────────────────

class PipelineTracer:
    """
    Collects structured events during a single pipeline execution.

    Usage:
        tracer = PipelineTracer()
        tracer.log_retrieval(query="...", n_results=8, top_score=0.92)
        tracer.log_generation(model="claude-sonnet-4-6", latency_ms=1200)
        tracer.log_safety(flags=["pregnancy_warning"], severity_counts={"warning": 1})
        tracer.log_citation(grounding_rate=0.85, ungrounded=["Paper X"])
        tracer.finish()
    """

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or get_trace_id()
        self.start_time = time.monotonic()
        self.events: list[dict] = []
        logger.info(f"[trace:{self.trace_id}] Pipeline started")

    def log_retrieval(
        self,
        query: str,
        n_results: int,
        top_score: float = 0.0,
        filters: dict | None = None,
        bm25_used: bool = False,
    ) -> None:
        event = {
            "stage": "retrieval",
            "query": query[:200],
            "n_results": n_results,
            "top_score": round(top_score, 4),
            "filters": bool(filters),
            "bm25_used": bm25_used,
        }
        self.events.append(event)
        logger.info(
            f"[trace:{self.trace_id}] Retrieval: {n_results} results, "
            f"top_score={top_score:.3f}, bm25={bm25_used}"
        )

    def log_generation(
        self,
        model: str,
        latency_ms: float,
        input_tokens_est: int = 0,
    ) -> None:
        event = {
            "stage": "generation",
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "input_tokens_est": input_tokens_est,
        }
        self.events.append(event)
        logger.info(
            f"[trace:{self.trace_id}] Generation: {model}, "
            f"{latency_ms:.0f}ms, ~{input_tokens_est} input tokens"
        )

    def log_safety(
        self,
        flags: list[str],
        severity_counts: dict[str, int] | None = None,
    ) -> None:
        event = {
            "stage": "safety",
            "flags": flags,
            "severity_counts": severity_counts or {},
        }
        self.events.append(event)
        logger.info(
            f"[trace:{self.trace_id}] Safety: {len(flags)} flags "
            f"({severity_counts or 'none'})"
        )

    def log_citation(
        self,
        grounding_rate: float,
        ungrounded: list[str] | None = None,
    ) -> None:
        event = {
            "stage": "citation",
            "grounding_rate": round(grounding_rate, 3),
            "ungrounded_count": len(ungrounded) if ungrounded else 0,
        }
        self.events.append(event)
        if ungrounded:
            logger.warning(
                f"[trace:{self.trace_id}] Citations: {grounding_rate:.0%} grounded, "
                f"{len(ungrounded)} ungrounded"
            )
        else:
            logger.info(
                f"[trace:{self.trace_id}] Citations: {grounding_rate:.0%} grounded"
            )

    def finish(self) -> dict:
        """Finalize trace and return summary."""
        total_ms = (time.monotonic() - self.start_time) * 1000
        summary = {
            "trace_id": self.trace_id,
            "total_ms": round(total_ms, 1),
            "stages": len(self.events),
            "events": self.events,
        }
        logger.info(
            f"[trace:{self.trace_id}] Pipeline complete in {total_ms:.0f}ms "
            f"({len(self.events)} stages)"
        )
        return summary
