"""
src/agents/vision_analyzer.py

Analyzes skin images using GPT-4o Vision.
Returns a structured SkinImageAnalysis with detected conditions and severity.

Cost estimate: ~$0.01–0.02 per image (GPT-4o vision pricing, Mar 2026)
"""
from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel, Field
from loguru import logger


class ZoneAnalysis(BaseModel):
    """Analysis of a specific facial zone."""
    zone: str                               # forehead | nose | cheeks | chin | under_eyes
    conditions: list[str]
    severity: str                           # none | mild | moderate | severe
    notes: str = ""


class SkinImageAnalysis(BaseModel):
    """Structured output from vision analysis of a skin photo."""
    overall_skin_type: str                  # oily | dry | combination | normal | sensitive
    fitzpatrick_estimate: str               # I–VI
    zones: list[ZoneAnalysis]
    detected_conditions: list[str]          # flattened list of all detected conditions
    texture_notes: str
    hyperpigmentation: str                  # none | mild | moderate | severe
    visible_pores: str                      # minimal | moderate | enlarged
    redness_level: str                      # none | mild | moderate | severe
    acne_severity: str                      # none | mild | moderate | severe | cystic
    estimated_age_range: str               # e.g. "25–35"
    confidence_note: str                    # any caveats about image quality
    raw_description: str                    # free-text summary from GPT-4o


VISION_SYSTEM_PROMPT = """You are a dermatology AI assistant specialized in skin image analysis.

Analyze the provided skin photograph and return a structured assessment.
Focus on visible skin conditions, texture, pigmentation, and hydration levels.

Important caveats to communicate:
- This is NOT a medical diagnosis
- Lighting and photo quality significantly affect accuracy
- Professional dermatologist evaluation is recommended for medical concerns

Be precise about what you CAN and CANNOT determine from the image.
Return valid JSON only — no prose outside the JSON object.
"""

VISION_USER_PROMPT = """Analyze this skin photograph and return a JSON assessment.

Schema:
{
  "overall_skin_type": "oily|dry|combination|normal|sensitive",
  "fitzpatrick_estimate": "I|II|III|IV|V|VI",
  "zones": [
    {
      "zone": "forehead|nose|cheeks|chin|under_eyes",
      "conditions": ["acne", "dryness", ...],
      "severity": "none|mild|moderate|severe",
      "notes": "..."
    }
  ],
  "detected_conditions": ["list of all detected skin conditions"],
  "texture_notes": "description of skin texture",
  "hyperpigmentation": "none|mild|moderate|severe",
  "visible_pores": "minimal|moderate|enlarged",
  "redness_level": "none|mild|moderate|severe",
  "acne_severity": "none|mild|moderate|severe|cystic",
  "estimated_age_range": "e.g. 25-35",
  "confidence_note": "any caveats about image quality or limitations",
  "raw_description": "2-3 sentence overall skin assessment"
}
"""


class VisionAnalyzer:
    """
    Analyzes skin photos using GPT-4o Vision.

    Usage:
        analyzer = VisionAnalyzer(api_key="sk-...")
        analysis = analyzer.analyze(image_path="face.jpg")
        # or from bytes:
        analysis = analyzer.analyze_bytes(image_bytes, media_type="image/jpeg")
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def analyze(self, image_path: str | Path) -> SkinImageAnalysis:
        """Analyze a skin image from a file path."""
        path = Path(image_path)
        media_type = _infer_media_type(path.suffix)
        image_bytes = path.read_bytes()
        return self.analyze_bytes(image_bytes, media_type)

    def analyze_bytes(
        self,
        image_bytes: bytes,
        media_type: str = "image/jpeg",
    ) -> SkinImageAnalysis:
        """Analyze a skin image from raw bytes (e.g. uploaded via Streamlit)."""
        import json

        b64 = base64.b64encode(image_bytes).decode("utf-8")

        logger.info(f"[VisionAnalyzer] Analyzing image ({len(image_bytes) // 1024}KB)...")

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": VISION_USER_PROMPT},
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw)
            analysis = SkinImageAnalysis(**data)
            logger.info("[VisionAnalyzer] Analysis complete.")
            return analysis
        except Exception as e:
            logger.error(f"[VisionAnalyzer] Parse error: {e}\nRaw: {raw[:300]}")
            raise ValueError(f"Failed to parse vision analysis: {e}") from e

    def merge_with_questionnaire(
        self,
        vision: SkinImageAnalysis,
        questionnaire: dict,
    ) -> dict:
        """
        Merge vision analysis with questionnaire data into a unified skin profile.
        Vision data augments but doesn't override self-reported data.
        """
        # Self-reported concerns take precedence; vision adds detected ones
        reported_concerns = questionnaire.get("concerns", [])
        detected = vision.detected_conditions

        all_concerns = list(dict.fromkeys(reported_concerns + detected))

        return {
            # From questionnaire (self-reported — higher trust)
            "skin_type": questionnaire.get("skin_type") or vision.overall_skin_type,
            "age_range": questionnaire.get("age_range") or vision.estimated_age_range,
            "fitzpatrick": questionnaire.get("fitzpatrick") or vision.fitzpatrick_estimate,
            "allergies": questionnaire.get("allergies", []),
            "medications": questionnaire.get("medications", []),
            "pregnancy": questionnaire.get("pregnancy", False),
            "previous_treatments": questionnaire.get("previous_treatments", []),

            # Merged concerns
            "concerns": all_concerns,
            "primary_goal": questionnaire.get("primary_goal", ""),

            # From vision (objective observations)
            "vision_skin_type": vision.overall_skin_type,
            "vision_detected_conditions": detected,
            "acne_severity": vision.acne_severity,
            "hyperpigmentation": vision.hyperpigmentation,
            "redness_level": vision.redness_level,
            "visible_pores": vision.visible_pores,
            "texture_notes": vision.texture_notes,
            "zone_analysis": [z.model_dump() for z in vision.zones],

            # Meta
            "vision_confidence": vision.confidence_note,
            "data_sources": ["questionnaire", "vision_analysis"],
        }


def _infer_media_type(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix.lower(), "image/jpeg")
