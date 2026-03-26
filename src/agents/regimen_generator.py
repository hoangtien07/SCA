"""
src/agents/regimen_generator.py

Core AI reasoning agent.
Takes skin profile + retrieved evidence → outputs structured skincare regimen.

Uses Claude claude-sonnet-4-6 for long-context synthesis and citation grounding.
Output is a validated Pydantic model — safe to serialize and display.

Prompts loaded from config/prompts/ for easy iteration without code changes.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from loguru import logger


# ── Prompt loading ────────────────────────────────────────────────────────────
_PROMPTS_DIR = Path(__file__).parent.parent.parent / "config" / "prompts"


def _load_prompt(filename: str, fallback: str) -> str:
    """Load prompt from file; return fallback if file missing."""
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    logger.warning(f"[RegimenGenerator] Prompt file {path} not found, using fallback")
    return fallback


# ── Output schema ─────────────────────────────────────────────────────────────

class RoutineStep(BaseModel):
    step_number: int
    product_type: str                           # e.g. "Cleanser", "Vitamin C Serum"
    active_ingredients: list[str]
    concentration_range: str                    # e.g. "10–15%"
    application_notes: str
    evidence_grade: str                         # A | B | C
    citations: list[str]                        # paper titles / DOIs

class Regimen(BaseModel):
    profile_summary: str
    am_routine: list[RoutineStep]
    pm_routine: list[RoutineStep]
    weekly_treatments: list[RoutineStep] = Field(default_factory=list)
    ingredients_to_avoid: list[str]
    contraindications: list[str]
    lifestyle_notes: list[str]
    follow_up_weeks: int = 8
    disclaimer: str = (
        "This regimen is based on published scientific literature and is "
        "for educational purposes only. Consult a dermatologist before "
        "starting new treatments, especially for prescription actives."
    )


# ── Generator ─────────────────────────────────────────────────────────────────

_SYSTEM_FALLBACK = (
    "You are a clinical skincare AI assistant trained on peer-reviewed "
    "dermatology research. Output valid JSON matching the Regimen schema."
)

_USER_FALLBACK = (
    "SKIN PROFILE:\n{profile_json}\n\n"
    "RETRIEVED SCIENTIFIC EVIDENCE:\n{evidence_context}\n\n"
    "Generate a complete personalized skincare regimen as valid JSON."
)

SYSTEM_PROMPT = _load_prompt("regimen_system.txt", _SYSTEM_FALLBACK)
USER_PROMPT_TEMPLATE = _load_prompt("regimen_user.txt", _USER_FALLBACK)


class RegimenGenerator:
    """
    Generates personalized skincare regimens using Claude.

    Usage:
        generator = RegimenGenerator(api_key="sk-ant-...")
        regimen = await generator.generate(profile, retrieved_chunks)
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6-20251101"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        profile: dict,
        evidence_chunks: list,                 # list[RetrievalResult]
        max_tokens: int = 4096,
    ) -> Regimen:
        """
        Generate a regimen. Synchronous.

        Args:
            profile: Skin profile dict from questionnaire + vision analysis
            evidence_chunks: Retrieved chunks from RAGRetriever
            max_tokens: Max tokens for LLM response
        """
        import json

        profile_json = json.dumps(profile, indent=2)
        evidence_context = self._format_evidence(evidence_chunks)

        user_message = USER_PROMPT_TEMPLATE.format(
            profile_json=profile_json,
            evidence_context=evidence_context,
        )

        logger.info(f"[RegimenGenerator] Calling {self.model}...")

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_json = message.content[0].text.strip()

        # Strip markdown code fences if present
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]

        try:
            data = json.loads(raw_json)
            regimen = Regimen(**data)
            logger.info("[RegimenGenerator] Regimen generated successfully.")
            return regimen
        except Exception as e:
            logger.error(f"[RegimenGenerator] Parse error: {e}\nRaw: {raw_json[:500]}")
            raise ValueError(f"Failed to parse regimen JSON: {e}") from e

    # ── Internal ──────────────────────────────────────────────────────────────

    def _format_evidence(self, chunks: list) -> str:
        """Format retrieved chunks into a readable context block for the LLM."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            title = getattr(chunk, "title", "Unknown")
            year = getattr(chunk, "year", "")
            journal = getattr(chunk, "journal", "")
            ev = getattr(chunk, "evidence_level", "C")
            text = getattr(chunk, "text", "")

            parts.append(
                f"[{i}] \"{title}\" ({year}) — {journal} | Evidence: {ev}\n"
                f"{text[:600]}\n"
            )

        return "\n---\n".join(parts)
