"""
tests/test_generator.py
Unit tests for RegimenGenerator with mocked Anthropic client.
Runs without API keys or network access.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.regimen_generator import RegimenGenerator, Regimen


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_REGIMEN_JSON = json.dumps({
    "profile_summary": "25-year-old with oily acne-prone skin",
    "am_routine": [
        {
            "step_number": 1,
            "product_type": "Cleanser",
            "active_ingredients": ["salicylic acid"],
            "concentration_range": "2%",
            "application_notes": "Gentle massage, rinse with lukewarm water",
            "evidence_grade": "A",
            "citations": ["RCT: Salicylic Acid for Acne (2022)"],
            "alternatives": ["benzoyl peroxide"],
            "product_name": "Gentle SA Cleanser",
        },
        {
            "step_number": 2,
            "product_type": "SPF Moisturizer",
            "active_ingredients": ["niacinamide", "zinc oxide SPF 50"],
            "concentration_range": "5%",
            "application_notes": "Apply generously",
            "evidence_grade": "A",
            "citations": ["Niacinamide in Acne (2023)"],
            "alternatives": [],
            "product_name": "SPF 50 Moisturizer",
        },
    ],
    "pm_routine": [
        {
            "step_number": 1,
            "product_type": "Treatment",
            "active_ingredients": ["adapalene"],
            "concentration_range": "0.1%",
            "application_notes": "Pea-sized amount to dry skin",
            "evidence_grade": "A",
            "citations": ["Adapalene Review (2021)"],
            "alternatives": ["tretinoin"],
            "product_name": "Adapalene Gel",
        },
    ],
    "weekly_treatments": [],
    "ingredients_to_avoid": ["coconut oil", "isopropyl myristate"],
    "contraindications": [],
    "lifestyle_notes": ["Change pillowcases 2x/week", "Avoid touching face"],
    "follow_up_weeks": 8,
    "disclaimer": "This is not medical advice. Consult a dermatologist.",
})


class MockEvidence:
    def __init__(self, text, title, year=2023, evidence_level="A"):
        self.text = text
        self.title = title
        self.year = year
        self.journal = "JAAD"
        self.evidence_level = evidence_level
        self.url = ""
        self.doi = ""


def _make_mock_anthropic_response(content: str):
    """Create a mock Anthropic Messages.create() response."""
    mock_block = MagicMock()
    mock_block.text = content
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    return mock_response


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_generate_parses_valid_json():
    """Generator should parse LLM JSON output into a Regimen object."""
    with patch("anthropic.Anthropic") as MockClient:
        client_instance = MockClient.return_value
        client_instance.messages.create.return_value = _make_mock_anthropic_response(
            f"```json\n{MOCK_REGIMEN_JSON}\n```"
        )

        gen = RegimenGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        gen.client = client_instance

        profile = {"skin_type": "oily", "concerns": ["acne"], "age": 25}
        evidence = [
            MockEvidence("Salicylic acid reduces acne lesions.", "SA for Acne"),
            MockEvidence("Niacinamide reduces sebum.", "Niacinamide Study"),
        ]

        regimen = gen.generate(profile=profile, evidence_chunks=evidence)

        assert isinstance(regimen, Regimen)
        assert len(regimen.am_routine) == 2
        assert len(regimen.pm_routine) == 1
        assert regimen.follow_up_weeks == 8
        assert "salicylic acid" in regimen.am_routine[0].active_ingredients


def test_generate_strips_markdown_fences():
    """Generator should strip ```json fences from LLM output."""
    with patch("anthropic.Anthropic") as MockClient:
        client_instance = MockClient.return_value
        client_instance.messages.create.return_value = _make_mock_anthropic_response(
            f"Here is the regimen:\n```json\n{MOCK_REGIMEN_JSON}\n```\nDone."
        )

        gen = RegimenGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        gen.client = client_instance

        profile = {"skin_type": "oily", "concerns": ["acne"]}
        evidence = [MockEvidence("Test evidence.", "Test Paper")]

        regimen = gen.generate(profile=profile, evidence_chunks=evidence)

        assert isinstance(regimen, Regimen)
        assert regimen.disclaimer != ""


def test_generate_regimen_schema_fields():
    """Regimen object should have all expected fields."""
    with patch("anthropic.Anthropic") as MockClient:
        client_instance = MockClient.return_value
        client_instance.messages.create.return_value = _make_mock_anthropic_response(
            MOCK_REGIMEN_JSON
        )

        gen = RegimenGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
        gen.client = client_instance

        profile = {"skin_type": "dry", "concerns": ["dryness"]}
        evidence = [MockEvidence("Ceramides restore skin barrier.", "Ceramide Study")]

        regimen = gen.generate(profile=profile, evidence_chunks=evidence)

        assert hasattr(regimen, "profile_summary")
        assert hasattr(regimen, "am_routine")
        assert hasattr(regimen, "pm_routine")
        assert hasattr(regimen, "weekly_treatments")
        assert hasattr(regimen, "ingredients_to_avoid")
        assert hasattr(regimen, "contraindications")
        assert hasattr(regimen, "lifestyle_notes")
        assert hasattr(regimen, "disclaimer")
