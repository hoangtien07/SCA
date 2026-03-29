"""
src/workers/tasks.py

Celery tasks for async regimen generation.

Task lifecycle:
    PENDING → ANALYZING → RETRIEVING → GENERATING → SAFETY_CHECK → SUCCESS
                                                                  → FAILURE

State updates are stored in Redis so the polling endpoint can return progress.
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger


# ── Task state constants ──────────────────────────────────────────────────────

class TaskState:
    PENDING = "PENDING"
    ANALYZING = "ANALYZING"
    RETRIEVING = "RETRIEVING"
    GENERATING = "GENERATING"
    SAFETY_CHECK = "SAFETY_CHECK"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"

    # Progress percentages for each state
    PROGRESS = {
        PENDING: 0,
        ANALYZING: 15,
        RETRIEVING: 35,
        GENERATING: 65,
        SAFETY_CHECK: 90,
        SUCCESS: 100,
        FAILURE: 0,
    }


def _update_state(task, state: str, meta: dict | None = None) -> None:
    """Update Celery task state with progress metadata."""
    progress = TaskState.PROGRESS.get(state, 0)
    task.update_state(
        state=state,
        meta={
            "status": state,
            "progress": progress,
            **(meta or {}),
        },
    )


# ── Main task ─────────────────────────────────────────────────────────────────

def generate_regimen_task_fn(
    task,
    profile_dict: dict,
    image_base64: str | None = None,
    media_type: str = "image/jpeg",
) -> dict:
    """
    Core implementation of the regimen generation task.
    Called by the Celery task wrapper.
    """
    from config.settings import settings
    start_time = time.monotonic()

    _update_state(task, TaskState.ANALYZING)

    # ── Step 1: Vision analysis (optional) ───────────────────────────────────
    vision_result = None
    if image_base64:
        try:
            import base64
            from src.agents.vision_analyzer import VisionAnalyzer

            image_bytes = base64.b64decode(image_base64)
            analyzer = VisionAnalyzer(
                api_key=settings.openai_api_key,
                model=settings.vision_model,
            )
            analysis = analyzer.analyze_bytes(image_bytes, media_type=media_type)
            vision_result = {
                "overall_skin_type": analysis.overall_skin_type,
                "fitzpatrick_estimate": analysis.fitzpatrick_estimate,
                "detected_conditions": analysis.detected_conditions,
                "acne_severity": analysis.acne_severity,
                "hyperpigmentation": analysis.hyperpigmentation,
                "redness_level": analysis.redness_level,
                "texture_notes": analysis.texture_notes,
                "confidence_note": analysis.confidence_note,
            }
            profile_dict.update({
                "detected_conditions": analysis.detected_conditions,
                "acne_severity": analysis.acne_severity,
                "hyperpigmentation": analysis.hyperpigmentation,
                "redness_level": analysis.redness_level,
            })
            if not profile_dict.get("skin_type"):
                profile_dict["skin_type"] = analysis.overall_skin_type
        except Exception as e:
            logger.warning(f"[task] Vision analysis failed: {e}")

    # ── Step 2: Retrieve evidence ─────────────────────────────────────────────
    _update_state(task, TaskState.RETRIEVING)

    from src.pipeline.indexer import ChromaIndexer
    from src.pipeline.bm25_index import BM25Index
    from src.agents.rag_retriever import RAGRetriever
    from pathlib import Path

    indexer = ChromaIndexer(
        persist_dir=settings.chroma_persist_dir,
        openai_api_key=settings.openai_api_key,
    )
    bm25_path = Path(settings.chroma_persist_dir) / "bm25_index.json"
    bm25 = BM25Index.load(str(bm25_path)) if bm25_path.exists() else None
    retriever = RAGRetriever(indexer=indexer, bm25=bm25, top_k=settings.retrieval_top_k)

    query = retriever.build_query_from_profile(profile_dict)
    results = retriever.retrieve(
        query=query,
        skin_conditions=profile_dict.get("concerns") or None,
        top_k=settings.retrieval_top_k,
    )

    # ── Step 3: Generate regimen ──────────────────────────────────────────────
    _update_state(task, TaskState.GENERATING, {"query_used": query, "evidence_count": len(results)})

    from src.agents.regimen_generator import RegimenGenerator
    generator = RegimenGenerator(
        api_key=settings.anthropic_api_key,
        model=settings.reasoning_model,
    )
    regimen = generator.generate(profile=profile_dict, evidence_chunks=results)

    # ── Step 4: Safety check ──────────────────────────────────────────────────
    _update_state(task, TaskState.SAFETY_CHECK)

    from src.agents.safety_guard import SafetyGuard
    guard = SafetyGuard()
    safety_report = guard.check(regimen=regimen, profile=profile_dict)

    if safety_report.modified_contraindications:
        existing = list(regimen.contraindications)
        existing.extend(safety_report.modified_contraindications)
        regimen.contraindications = existing

    # ── Build response ────────────────────────────────────────────────────────
    latency_ms = (time.monotonic() - start_time) * 1000

    def step_dict(step) -> dict:
        return {
            "step_number": step.step_number,
            "product_type": step.product_type,
            "active_ingredients": step.active_ingredients,
            "concentration_range": step.concentration_range,
            "application_notes": step.application_notes,
            "evidence_grade": step.evidence_grade,
            "citations": step.citations,
        }

    return {
        "status": TaskState.SUCCESS,
        "progress": 100,
        "latency_ms": round(latency_ms, 1),
        "result": {
            "regimen": {
                "profile_summary": regimen.profile_summary,
                "am_routine": [step_dict(s) for s in regimen.am_routine],
                "pm_routine": [step_dict(s) for s in regimen.pm_routine],
                "weekly_treatments": [step_dict(s) for s in regimen.weekly_treatments],
                "ingredients_to_avoid": regimen.ingredients_to_avoid,
                "contraindications": regimen.contraindications,
                "lifestyle_notes": regimen.lifestyle_notes,
                "follow_up_weeks": regimen.follow_up_weeks,
                "disclaimer": regimen.disclaimer,
            },
            "safety_report": {
                "flags": [
                    {
                        "severity": f.severity,
                        "message": f.message,
                        "affected_ingredients": f.affected_ingredients,
                    }
                    for f in safety_report.flags
                ],
                "is_safe_to_show": safety_report.is_safe_to_show,
                "modified_contraindications": safety_report.modified_contraindications,
            },
            "vision_analysis": vision_result,
            "evidence_count": len(results),
            "query_used": query,
        },
    }


# ── Register Celery task ──────────────────────────────────────────────────────

try:
    from src.workers.celery_app import celery_app

    if celery_app is not None:
        @celery_app.task(
            bind=True,
            name="src.workers.tasks.generate_regimen_task",
            max_retries=0,  # never retry — LLM calls are expensive
            soft_time_limit=120,
            time_limit=150,
        )
        def generate_regimen_task(
            self,
            profile_dict: dict,
            image_base64: str | None = None,
            media_type: str = "image/jpeg",
        ) -> dict:
            """
            Async Celery task: profile → regimen + safety report.

            State transitions: PENDING → ANALYZING → RETRIEVING → GENERATING → SAFETY_CHECK → SUCCESS
            """
            try:
                return generate_regimen_task_fn(self, profile_dict, image_base64, media_type)
            except Exception as e:
                logger.error(f"[task] generate_regimen_task failed: {e}")
                self.update_state(
                    state=TaskState.FAILURE,
                    meta={"status": TaskState.FAILURE, "progress": 0, "error": str(e)},
                )
                raise
    else:
        # Celery unavailable — provide a stub
        def generate_regimen_task(*args, **kwargs):
            raise RuntimeError("Celery is not available. Ensure Redis is running.")

except ImportError:
    def generate_regimen_task(*args, **kwargs):
        raise RuntimeError("Celery is not available. Install celery[redis] and start Redis.")
