"""
src/api/schemas.py

Pydantic request/response models for the FastAPI endpoints.
These are the API boundary types — separate from internal models.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────

class SkinProfile(BaseModel):
    """Skin profile from questionnaire + optional vision analysis."""
    skin_type: str = ""                     # oily | dry | combination | normal | sensitive
    fitzpatrick: str = ""                   # I–VI
    age: int | None = None
    concerns: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    pregnancy: bool = False
    acne_severity: str = "none"             # none | mild | moderate | severe | cystic
    primary_goal: str = ""
    # Vision-augmented fields (optional, populated by /analyze)
    detected_conditions: list[str] = Field(default_factory=list)
    hyperpigmentation: str = "none"
    redness_level: str = "none"
    texture_notes: str = ""


class AnalyzeRequest(BaseModel):
    """Request body for /analyze — image as base64."""
    image_base64: str
    media_type: str = "image/jpeg"


class RetrieveRequest(BaseModel):
    """Request body for /retrieve."""
    profile: SkinProfile
    query: str | None = None                # optional override; built from profile if missing
    evidence_levels: list[str] | None = None
    top_k: int = 8


class GenerateRequest(BaseModel):
    """Request body for /generate."""
    profile: SkinProfile
    evidence_context: list[EvidenceChunk] = Field(default_factory=list)


class EvidenceChunk(BaseModel):
    """A single evidence chunk passed to the generator."""
    text: str
    title: str = ""
    year: int = 0
    journal: str = ""
    evidence_level: str = "C"
    url: str = ""
    doi: str = ""
    score: float = 0.0


# Fix forward reference for GenerateRequest
GenerateRequest.model_rebuild()


class SafetyCheckRequest(BaseModel):
    """Request body for /safety-check."""
    regimen: RegimenResponse
    profile: SkinProfile


class FullPipelineRequest(BaseModel):
    """Request body for /full-pipeline — end-to-end."""
    profile: SkinProfile
    image_base64: str | None = None
    media_type: str = "image/jpeg"


# ── Response Models ───────────────────────────────────────────────────────────

class RoutineStepResponse(BaseModel):
    step_number: int
    product_type: str
    active_ingredients: list[str]
    concentration_range: str
    application_notes: str
    evidence_grade: str
    citations: list[str]


class RegimenResponse(BaseModel):
    profile_summary: str
    am_routine: list[RoutineStepResponse]
    pm_routine: list[RoutineStepResponse]
    weekly_treatments: list[RoutineStepResponse] = Field(default_factory=list)
    ingredients_to_avoid: list[str]
    contraindications: list[str]
    lifestyle_notes: list[str]
    follow_up_weeks: int = 8
    disclaimer: str = ""


# Fix forward reference for SafetyCheckRequest
SafetyCheckRequest.model_rebuild()


class SafetyFlagResponse(BaseModel):
    severity: str
    message: str
    affected_ingredients: list[str] = Field(default_factory=list)


class SafetyReportResponse(BaseModel):
    flags: list[SafetyFlagResponse]
    is_safe_to_show: bool = True
    modified_contraindications: list[str] = Field(default_factory=list)
    summary: str = ""


class RetrieveResponse(BaseModel):
    chunks: list[EvidenceChunk]
    query_used: str


class AnalyzeResponse(BaseModel):
    overall_skin_type: str
    fitzpatrick_estimate: str
    detected_conditions: list[str]
    acne_severity: str
    hyperpigmentation: str
    redness_level: str
    texture_notes: str
    confidence_note: str


class FullPipelineResponse(BaseModel):
    """Full pipeline output — regimen + safety report + evidence metadata."""
    regimen: RegimenResponse
    safety_report: SafetyReportResponse
    vision_analysis: AnalyzeResponse | None = None
    evidence_count: int = 0
    query_used: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str = ""
    knowledge_base_chunks: int = 0
