"""
tests/test_api_routes.py
Integration tests for FastAPI endpoints using TestClient.
Mocks external dependencies (LLM, ChromaDB) so tests run without API keys.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.api.main import app


# ── Mock dependency overrides ─────────────────────────────────────────────────

class MockRetrievalResult:
    def __init__(self):
        self.text = "Niacinamide reduces sebum production in oily skin."
        self.title = "Niacinamide for Acne: A Randomized Controlled Trial"
        self.year = 2023
        self.journal = "JAAD"
        self.evidence_level = "A"
        self.url = ""
        self.doi = "10.1234/test"
        self.score = 0.85


class MockStep:
    def __init__(self):
        self.step_number = 1
        self.product_type = "Serum"
        self.active_ingredients = ["niacinamide"]
        self.concentration_range = "5%"
        self.application_notes = "Apply to clean skin"
        self.evidence_grade = "A"
        self.citations = ["Niacinamide for Acne (2023)"]
        self.alternatives = []
        self.product_name = "Niacinamide Serum"


class MockRegimen:
    def __init__(self):
        step = MockStep()
        self.profile_summary = "Oily acne-prone skin"
        self.am_routine = [step]
        self.pm_routine = [step]
        self.weekly_treatments = []
        self.ingredients_to_avoid = ["coconut oil"]
        self.contraindications = []
        self.lifestyle_notes = ["Wash pillowcases weekly"]
        self.follow_up_weeks = 8
        self.disclaimer = "Consult a dermatologist."


def _mock_retriever():
    retriever = MagicMock()
    retriever.build_query_from_profile.return_value = "niacinamide acne oily skin"
    retriever.retrieve.return_value = [MockRetrievalResult()]
    retriever.bm25 = None
    return retriever


def _mock_generator():
    gen = MagicMock()
    gen.generate.return_value = MockRegimen()
    gen.model = "claude-sonnet-4-20250514"
    return gen


def _mock_indexer():
    indexer = MagicMock()
    indexer.stats.return_value = {"total_chunks": 1500}
    return indexer


# ── Apply mocks ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_deps():
    """Override dependency injection so routes don't need real services."""
    with (
        patch("src.api.routes.get_retriever", return_value=_mock_retriever()),
        patch("src.api.routes.get_generator", return_value=_mock_generator()),
        patch("src.api.routes.get_indexer", return_value=_mock_indexer()),
        patch("src.api.routes.get_vision_analyzer", return_value=MagicMock()),
        patch("src.api.routes.get_safety_guard") as mock_guard,
    ):
        # Use real SafetyGuard for /safety-check tests
        from src.agents.safety_guard import SafetyGuard
        mock_guard.return_value = SafetyGuard()
        yield


client = TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "knowledge_base_chunks" in data


# ── /retrieve ─────────────────────────────────────────────────────────────────

def test_retrieve_returns_chunks():
    resp = client.post("/retrieve", json={
        "profile": {
            "skin_type": "oily",
            "concerns": ["acne"],
            "pregnancy": False,
        },
        "top_k": 5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "chunks" in data
    assert len(data["chunks"]) > 0
    assert "query_used" in data


def test_retrieve_with_custom_query():
    resp = client.post("/retrieve", json={
        "profile": {"skin_type": "oily", "concerns": ["acne"]},
        "query": "niacinamide for oily skin",
        "top_k": 3,
    })
    assert resp.status_code == 200


# ── /safety-check ────────────────────────────────────────────────────────────

def test_safety_check_pregnancy_warning():
    """Regimen with retinol + pregnant profile should flag warning."""
    resp = client.post("/safety-check", json={
        "profile": {
            "skin_type": "oily",
            "concerns": ["acne"],
            "pregnancy": True,
        },
        "regimen": {
            "profile_summary": "test",
            "am_routine": [],
            "pm_routine": [{
                "step_number": 1,
                "product_type": "Serum",
                "active_ingredients": ["retinol 0.5%"],
                "concentration_range": "0.5%",
                "application_notes": "PM only",
                "evidence_grade": "A",
                "citations": [],
            }],
            "weekly_treatments": [],
            "ingredients_to_avoid": [],
            "contraindications": [],
            "lifestyle_notes": [],
        },
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["flags"]) > 0
    assert any("pregnancy" in f["message"].lower() for f in data["flags"])


def test_safety_check_clean_regimen():
    resp = client.post("/safety-check", json={
        "profile": {
            "skin_type": "normal",
            "concerns": [],
            "pregnancy": False,
        },
        "regimen": {
            "profile_summary": "test",
            "am_routine": [{
                "step_number": 1,
                "product_type": "SPF Moisturizer",
                "active_ingredients": ["niacinamide", "zinc SPF 50"],
                "concentration_range": "",
                "application_notes": "",
                "evidence_grade": "A",
                "citations": [],
            }],
            "pm_routine": [{
                "step_number": 1,
                "product_type": "Moisturizer",
                "active_ingredients": ["ceramides", "hyaluronic acid"],
                "concentration_range": "",
                "application_notes": "",
                "evidence_grade": "B",
                "citations": [],
            }],
            "weekly_treatments": [],
            "ingredients_to_avoid": [],
            "contraindications": [],
            "lifestyle_notes": [],
        },
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_safe_to_show"] is True


# ── /generate ─────────────────────────────────────────────────────────────────

def test_generate_returns_regimen():
    resp = client.post("/generate", json={
        "profile": {
            "skin_type": "oily",
            "concerns": ["acne"],
        },
        "evidence_context": [{
            "text": "Niacinamide reduces sebum in RCT.",
            "title": "Test Paper",
            "year": 2023,
            "evidence_level": "A",
        }],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "am_routine" in data
    assert "pm_routine" in data
    assert "disclaimer" in data


# ── Validation errors ─────────────────────────────────────────────────────────

def test_retrieve_missing_profile_fails():
    resp = client.post("/retrieve", json={})
    assert resp.status_code == 422
