"""
src/api/routes.py

FastAPI route handlers for the skincare AI pipeline.

Endpoints:
    POST /analyze         — Analyze a skin photo (GPT-4o Vision)
    POST /retrieve        — Retrieve relevant evidence chunks
    POST /generate        — Generate a skincare regimen (Claude)
    POST /safety-check    — Run safety guardrails on a regimen
    POST /full-pipeline   — End-to-end: profile (+ optional image) → regimen + safety
    GET  /health          — Health check with KB stats
"""
from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas import (
    AnalyzeRequest, AnalyzeResponse,
    RetrieveRequest, RetrieveResponse,
    GenerateRequest, RegimenResponse,
    SafetyCheckRequest, SafetyReportResponse,
    FullPipelineRequest, FullPipelineResponse,
    HealthResponse, EvidenceChunk,
    RoutineStepResponse,
)
from src.api.deps import (
    get_retriever, get_generator, get_vision_analyzer,
    get_safety_guard, get_indexer,
)
from src.api.tracing import PipelineTracer, new_trace_id
from src.agents.citation_checker import CitationChecker

router = APIRouter()


# ── POST /analyze ─────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_image(req: AnalyzeRequest):
    """Analyze a skin photo and return structured assessment."""
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    analyzer = get_vision_analyzer()
    analysis = analyzer.analyze_bytes(image_bytes, media_type=req.media_type)

    return AnalyzeResponse(
        overall_skin_type=analysis.overall_skin_type,
        fitzpatrick_estimate=analysis.fitzpatrick_estimate,
        detected_conditions=analysis.detected_conditions,
        acne_severity=analysis.acne_severity,
        hyperpigmentation=analysis.hyperpigmentation,
        redness_level=analysis.redness_level,
        texture_notes=analysis.texture_notes,
        confidence_note=analysis.confidence_note,
    )


# ── POST /retrieve ────────────────────────────────────────────────────────────

@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_evidence(req: RetrieveRequest):
    """Retrieve relevant scientific evidence for a skin profile."""
    retriever = get_retriever()

    # Build query from profile if not provided
    profile_dict = req.profile.model_dump()
    query = req.query or retriever.build_query_from_profile(profile_dict)

    results = retriever.retrieve(
        query=query,
        skin_conditions=req.profile.concerns or None,
        evidence_levels=req.evidence_levels,
        top_k=req.top_k,
    )

    chunks = [
        EvidenceChunk(
            text=r.text,
            title=r.title,
            year=r.year,
            journal=r.journal,
            evidence_level=r.evidence_level,
            url=r.url,
            doi=r.doi,
            score=r.score,
        )
        for r in results
    ]

    return RetrieveResponse(chunks=chunks, query_used=query)


# ── POST /generate ────────────────────────────────────────────────────────────

@router.post("/generate", response_model=RegimenResponse)
def generate_regimen(req: GenerateRequest):
    """Generate a personalized skincare regimen from profile + evidence."""
    generator = get_generator()
    profile_dict = req.profile.model_dump()

    # Convert evidence chunks to objects with expected attributes
    evidence = [_chunk_to_retrieval_result(c) for c in req.evidence_context]

    regimen = generator.generate(profile=profile_dict, evidence_chunks=evidence)
    return _regimen_to_response(regimen)


# ── POST /safety-check ───────────────────────────────────────────────────────

@router.post("/safety-check", response_model=SafetyReportResponse)
def safety_check(req: SafetyCheckRequest):
    """Run safety guardrails on a regimen + profile."""
    guard = get_safety_guard()
    profile_dict = req.profile.model_dump()

    # Convert API regimen to mock object with expected attributes
    mock_regimen = _response_to_mock_regimen(req.regimen)
    report = guard.check(regimen=mock_regimen, profile=profile_dict)

    return _safety_report_to_response(report)


# ── POST /full-pipeline ──────────────────────────────────────────────────────

@router.post("/full-pipeline", response_model=FullPipelineResponse)
def full_pipeline(req: FullPipelineRequest):
    """End-to-end: profile (+ optional image) → regimen + safety report."""
    trace_id = new_trace_id()
    tracer = PipelineTracer(trace_id=trace_id)
    profile_dict = req.profile.model_dump()
    vision_response = None

    # ── Step 1: Vision analysis (optional) ────────────────────────────────
    if req.image_base64:
        try:
            image_bytes = base64.b64decode(req.image_base64)
            analyzer = get_vision_analyzer()
            analysis = analyzer.analyze_bytes(image_bytes, media_type=req.media_type)
            vision_response = AnalyzeResponse(
                overall_skin_type=analysis.overall_skin_type,
                fitzpatrick_estimate=analysis.fitzpatrick_estimate,
                detected_conditions=analysis.detected_conditions,
                acne_severity=analysis.acne_severity,
                hyperpigmentation=analysis.hyperpigmentation,
                redness_level=analysis.redness_level,
                texture_notes=analysis.texture_notes,
                confidence_note=analysis.confidence_note,
            )
            # Merge vision results into profile
            profile_dict["detected_conditions"] = analysis.detected_conditions
            profile_dict["acne_severity"] = analysis.acne_severity
            profile_dict["hyperpigmentation"] = analysis.hyperpigmentation
            profile_dict["redness_level"] = analysis.redness_level
            if not profile_dict.get("skin_type"):
                profile_dict["skin_type"] = analysis.overall_skin_type
        except Exception as e:
            logger.warning(f"[full-pipeline] Vision analysis failed: {e}")

    # ── Step 2: Retrieve evidence ─────────────────────────────────────────
    retriever = get_retriever()
    query = retriever.build_query_from_profile(profile_dict)
    results = retriever.retrieve(
        query=query,
        skin_conditions=profile_dict.get("concerns") or None,
        top_k=8,
    )
    tracer.log_retrieval(
        query=query,
        n_results=len(results),
        top_score=results[0].score if results else 0.0,
        bm25_used=retriever.bm25 is not None,
    )

    # ── Step 3: Generate regimen ──────────────────────────────────────────
    import time as _time
    _gen_start = _time.monotonic()
    generator = get_generator()
    regimen = generator.generate(profile=profile_dict, evidence_chunks=results)
    _gen_ms = (_time.monotonic() - _gen_start) * 1000
    tracer.log_generation(model=generator.model, latency_ms=_gen_ms)

    # ── Step 4: Safety check ──────────────────────────────────────────────
    guard = get_safety_guard()
    safety_report = guard.check(regimen=regimen, profile=profile_dict)
    tracer.log_safety(
        flags=[f.message for f in safety_report.flags],
        severity_counts=_count_severities(safety_report),
    )

    # ── Step 5: Citation grounding ────────────────────────────────────────
    checker = CitationChecker()
    citation_report = checker.check(regimen, results)
    tracer.log_citation(
        grounding_rate=citation_report.grounding_rate,
        ungrounded=citation_report.ungrounded_citations,
    )

    # Append safety contraindications to regimen
    if safety_report.modified_contraindications:
        existing = list(regimen.contraindications)
        existing.extend(safety_report.modified_contraindications)
        regimen.contraindications = existing

    tracer.finish()

    return FullPipelineResponse(
        regimen=_regimen_to_response(regimen),
        safety_report=_safety_report_to_response(safety_report),
        vision_analysis=vision_response,
        evidence_count=len(results),
        query_used=query,
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health():
    """Health check with KB statistics."""
    from config.settings import settings

    try:
        indexer = get_indexer()
        stats = indexer.stats()
        chunk_count = stats.get("total_chunks", 0)
    except Exception:
        chunk_count = -1

    return HealthResponse(
        status="ok",
        environment=settings.environment,
        knowledge_base_chunks=chunk_count,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _count_severities(report) -> dict[str, int]:
    """Count flags by severity level."""
    counts: dict[str, int] = {}
    for f in report.flags:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return counts


class _SimpleNamespace:
    """Lightweight object to carry attributes for agents expecting object access."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _chunk_to_retrieval_result(chunk: EvidenceChunk) -> _SimpleNamespace:
    """Convert API EvidenceChunk to an object with attrs expected by RegimenGenerator."""
    return _SimpleNamespace(
        text=chunk.text,
        title=chunk.title,
        year=chunk.year,
        journal=chunk.journal,
        evidence_level=chunk.evidence_level,
        url=chunk.url,
        doi=chunk.doi,
    )


def _regimen_to_response(regimen) -> RegimenResponse:
    """Convert internal Regimen model to API response model."""
    def step_to_resp(step) -> RoutineStepResponse:
        return RoutineStepResponse(
            step_number=step.step_number,
            product_type=step.product_type,
            active_ingredients=step.active_ingredients,
            concentration_range=step.concentration_range,
            application_notes=step.application_notes,
            evidence_grade=step.evidence_grade,
            citations=step.citations,
        )

    return RegimenResponse(
        profile_summary=regimen.profile_summary,
        am_routine=[step_to_resp(s) for s in regimen.am_routine],
        pm_routine=[step_to_resp(s) for s in regimen.pm_routine],
        weekly_treatments=[step_to_resp(s) for s in regimen.weekly_treatments],
        ingredients_to_avoid=regimen.ingredients_to_avoid,
        contraindications=regimen.contraindications,
        lifestyle_notes=regimen.lifestyle_notes,
        follow_up_weeks=regimen.follow_up_weeks,
        disclaimer=regimen.disclaimer,
    )


def _response_to_mock_regimen(resp: RegimenResponse) -> _SimpleNamespace:
    """Convert API RegimenResponse to a mock object for SafetyGuard."""
    def resp_to_step(s: RoutineStepResponse) -> _SimpleNamespace:
        return _SimpleNamespace(
            active_ingredients=s.active_ingredients,
            product_type=s.product_type,
            concentration_range=s.concentration_range,
        )

    return _SimpleNamespace(
        am_routine=[resp_to_step(s) for s in resp.am_routine],
        pm_routine=[resp_to_step(s) for s in resp.pm_routine],
        weekly_treatments=[resp_to_step(s) for s in resp.weekly_treatments],
        contraindications=list(resp.contraindications),
    )


def _safety_report_to_response(report) -> SafetyReportResponse:
    """Convert internal SafetyReport to API response."""
    from src.api.schemas import SafetyFlagResponse

    return SafetyReportResponse(
        flags=[
            SafetyFlagResponse(
                severity=f.severity,
                message=f.message,
                affected_ingredients=f.affected_ingredients,
            )
            for f in report.flags
        ],
        is_safe_to_show=report.is_safe_to_show,
        modified_contraindications=report.modified_contraindications,
        summary=report.summary(),
    )
