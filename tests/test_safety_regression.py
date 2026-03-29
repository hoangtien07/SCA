"""
tests/test_safety_regression.py
Safety regression tests — covers all SafetyGuard check paths.
Run without API keys, no network required.

Prompt #05 additions:
  - test_llm_judge_high_concentration_vitamin_c_retinol
  - test_llm_judge_multiple_ahas
  - test_llm_judge_standard_safe_routine
  All three mock the Anthropic API — no real API calls made.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.safety_guard import SafetyGuard, LLMSafetyJudge


# ── Test fixtures ─────────────────────────────────────────────────────────────

class MockStep:
    def __init__(self, ingredients, product_type="Serum", concentration_range=""):
        self.active_ingredients = ingredients
        self.product_type = product_type
        self.concentration_range = concentration_range


class MockRegimen:
    def __init__(self, am=None, pm=None, weekly=None):
        self.am_routine = am or []
        self.pm_routine = pm or []
        self.weekly_treatments = weekly or []
        self.contraindications = []


# ── Pregnancy tests ───────────────────────────────────────────────────────────

def test_pregnancy_flags_retinol():
    guard = SafetyGuard()
    regimen = MockRegimen(pm=[MockStep(["retinol 0.5%"])])
    profile = {"pregnancy": True, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)
    assert report.has_warnings
    assert any("pregnancy" in f.message.lower() for f in report.flags)


def test_pregnancy_flags_hydroquinone():
    guard = SafetyGuard()
    regimen = MockRegimen(pm=[MockStep(["hydroquinone 2%"])])
    profile = {"pregnancy": True, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)
    assert report.has_warnings


def test_pregnancy_no_flag_safe_ingredients():
    guard = SafetyGuard()
    regimen = MockRegimen(
        am=[MockStep(["niacinamide", "zinc SPF"], product_type="SPF Moisturizer")],
        pm=[MockStep(["ceramides", "hyaluronic acid"])],
    )
    profile = {"pregnancy": True, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)
    # Should not have pregnancy-related warnings (ceramides and HA are safe)
    preg_flags = [f for f in report.flags if "pregnancy" in f.message.lower()]
    assert len(preg_flags) == 0


# ── Phototoxicity tests ──────────────────────────────────────────────────────

def test_phototoxicity_flags_retinol_no_spf():
    guard = SafetyGuard()
    regimen = MockRegimen(
        am=[MockStep(["niacinamide"])],
        pm=[MockStep(["retinol"])],
    )
    profile = {"pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)
    phototox = [f for f in report.flags if "photosensitiz" in f.message.lower()]
    assert len(phototox) > 0


def test_phototoxicity_no_flag_with_spf():
    guard = SafetyGuard()
    regimen = MockRegimen(
        am=[MockStep(["niacinamide"], product_type="Moisturizer"),
            MockStep(["zinc oxide"], product_type="SPF 50 Sunscreen")],
        pm=[MockStep(["retinol"])],
    )
    profile = {"pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)
    # retinol + SPF present → should NOT flag phototoxicity warning
    phototox_warnings = [
        f for f in report.flags
        if "photosensitiz" in f.message.lower() and f.severity == "warning"
    ]
    assert len(phototox_warnings) == 0


def test_phototoxicity_glycolic_acid():
    guard = SafetyGuard()
    regimen = MockRegimen(
        am=[MockStep(["glycolic acid 8%"])],
        pm=[MockStep(["ceramides"])],
    )
    profile = {"pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)
    phototox = [f for f in report.flags if "photosensitiz" in f.message.lower()]
    assert len(phototox) > 0


# ── Age safety tests ─────────────────────────────────────────────────────────

def test_pediatric_flags_tretinoin():
    guard = SafetyGuard()
    regimen = MockRegimen(pm=[MockStep(["tretinoin 0.025%"])])
    profile = {"age": 14, "pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)
    age_flags = [f for f in report.flags if "pediatric" in f.message.lower()]
    assert len(age_flags) > 0


def test_geriatric_caution_benzoyl_peroxide():
    guard = SafetyGuard()
    regimen = MockRegimen(am=[MockStep(["benzoyl peroxide 5%"])])
    profile = {"age": 70, "pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)
    age_flags = [f for f in report.flags if "geriatric" in f.message.lower()]
    assert len(age_flags) > 0


def test_adult_no_age_flag():
    guard = SafetyGuard()
    regimen = MockRegimen(pm=[MockStep(["retinol"])])
    profile = {"age": 35, "pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)
    age_flags = [f for f in report.flags if "pediatric" in f.message.lower() or "geriatric" in f.message.lower()]
    assert len(age_flags) == 0


# ── Drug interaction tests ────────────────────────────────────────────────────

def test_isotretinoin_blocks_retinoids():
    guard = SafetyGuard()
    regimen = MockRegimen(pm=[MockStep(["retinol 0.5%"])])
    profile = {"pregnancy": False, "medications": ["isotretinoin"], "concerns": []}
    report = guard.check(regimen, profile)
    drug_flags = [f for f in report.flags if "isotretinoin" in f.message.lower()]
    assert len(drug_flags) > 0


def test_tetracycline_flags_aha():
    guard = SafetyGuard()
    regimen = MockRegimen(pm=[MockStep(["AHA glycolic acid 8%"])])
    profile = {"pregnancy": False, "medications": ["doxycycline"], "concerns": []}
    report = guard.check(regimen, profile)
    # doxycycline is a tetracycline — should match drug_interactions rule
    drug_flags = [f for f in report.flags if "tetracycline" in f.message.lower()]
    assert len(drug_flags) > 0


# ── Concentration limit tests ────────────────────────────────────────────────

def test_concentration_retinol_over_limit():
    guard = SafetyGuard()
    regimen = MockRegimen(
        pm=[MockStep(["retinol"], concentration_range="2%")]
    )
    profile = {"pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)
    conc_flags = [f for f in report.flags if "concentration" in f.message.lower()]
    assert len(conc_flags) > 0


def test_concentration_within_limit():
    guard = SafetyGuard()
    regimen = MockRegimen(
        pm=[MockStep(["retinol"], concentration_range="0.5%")]
    )
    profile = {"pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)
    conc_flags = [f for f in report.flags if "concentration" in f.message.lower()]
    assert len(conc_flags) == 0


# ── Severity escalation tests ────────────────────────────────────────────────

def test_severe_acne_referral():
    guard = SafetyGuard()
    regimen = MockRegimen(am=[MockStep(["niacinamide"])])
    profile = {"pregnancy": False, "medications": [], "concerns": [], "acne_severity": "cystic"}
    report = guard.check(regimen, profile)
    assert any("dermatologist" in f.message.lower() for f in report.flags)


def test_psoriasis_referral():
    guard = SafetyGuard()
    regimen = MockRegimen(am=[MockStep(["ceramides"])])
    profile = {"pregnancy": False, "medications": [], "concerns": ["psoriasis"]}
    report = guard.check(regimen, profile)
    assert any("psoriasis" in f.message.lower() or "autoimmune" in f.message.lower() for f in report.flags)


# ── Ingredient conflict tests ────────────────────────────────────────────────

def test_retinol_aha_conflict():
    guard = SafetyGuard()
    regimen = MockRegimen(pm=[MockStep(["retinol", "AHA glycolic acid"])])
    profile = {"pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)
    conflict_flags = [f for f in report.flags if "conflict" in f.message.lower()]
    assert len(conflict_flags) > 0


# ── Allergen tests ────────────────────────────────────────────────────────────

def test_allergen_fragrance_flagged():
    guard = SafetyGuard()
    regimen = MockRegimen(am=[MockStep(["fragrance", "niacinamide"])])
    profile = {"pregnancy": False, "medications": [], "concerns": [], "allergies": ["fragrance"]}
    report = guard.check(regimen, profile)
    allergen_flags = [f for f in report.flags if "allergen" in f.message.lower()]
    assert len(allergen_flags) > 0
    assert any("fragrance" in f.message.lower() for f in report.flags)


def test_allergen_inci_synonym_match():
    """Allergy to 'retinol' should flag 'retinaldehyde' via INCI synonym group."""
    guard = SafetyGuard()
    regimen = MockRegimen(pm=[MockStep(["retinaldehyde 0.05%"])])
    profile = {"pregnancy": False, "medications": [], "concerns": [], "allergies": ["retinol"]}
    report = guard.check(regimen, profile)
    allergen_flags = [f for f in report.flags if "allergen" in f.message.lower()]
    assert len(allergen_flags) > 0


def test_allergen_no_flag_when_no_allergy():
    guard = SafetyGuard()
    regimen = MockRegimen(am=[MockStep(["niacinamide", "ceramides"])])
    profile = {"pregnancy": False, "medications": [], "concerns": [], "allergies": []}
    report = guard.check(regimen, profile)
    allergen_flags = [f for f in report.flags if "allergen" in f.message.lower()]
    assert len(allergen_flags) == 0


def test_allergen_no_false_positive():
    """Allergy to 'fragrance' should NOT flag 'niacinamide'."""
    guard = SafetyGuard()
    regimen = MockRegimen(am=[MockStep(["niacinamide", "hyaluronic acid"])])
    profile = {"pregnancy": False, "medications": [], "concerns": [], "allergies": ["fragrance"]}
    report = guard.check(regimen, profile)
    allergen_flags = [f for f in report.flags if "allergen" in f.message.lower()]
    assert len(allergen_flags) == 0


# ── Prompt #05: LLM judge tests (Anthropic API mocked) ───────────────────────

def _make_mock_judge(response_dict: dict):
    """Create a mock LLMSafetyJudge that returns a fixed response without API calls."""
    from unittest.mock import MagicMock, patch

    class MockJudge(LLMSafetyJudge):
        def __init__(self):
            super().__init__(api_key="mock-key")

        def judge(self, ingredients, profile):
            return response_dict

    return MockJudge()


def test_llm_judge_high_concentration_vitamin_c_retinol():
    """
    High-concentration Vitamin C + Retinol: LLM judge should detect pH
    incompatibility concern. Mock API returns a high-confidence warning.
    """
    # Build guard with mocked LLM judge
    guard = SafetyGuard(use_llm_judge=True, anthropic_api_key="mock")
    guard._llm_judge = _make_mock_judge({
        "additional_flags": [
            {
                "severity": "caution",
                "message": "High-dose Vitamin C (20%) combined with retinol may cause pH incompatibility and oxidation instability",
                "affected_ingredients": ["ascorbic acid", "retinol"],
                "confidence": 0.85,
            }
        ],
        "overall_assessment": "caution",
    })

    regimen = MockRegimen(
        am=[MockStep(["ascorbic acid 20%"])],
        pm=[MockStep(["retinol 0.5%"])],
    )
    profile = {"pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)

    llm_flags = [f for f in report.flags if getattr(f, "source", "") == "llm_judge"]
    assert len(llm_flags) > 0
    assert any("vitamin c" in f.message.lower() or "ascorbic" in f.message.lower()
               or "ph" in f.message.lower() for f in llm_flags)


def test_llm_judge_multiple_ahas():
    """
    Multiple AHAs stacked: LLM judge should flag over-exfoliation risk.
    Mock API returns a warning with confidence > 0.7.
    """
    guard = SafetyGuard(use_llm_judge=True, anthropic_api_key="mock")
    guard._llm_judge = _make_mock_judge({
        "additional_flags": [
            {
                "severity": "warning",
                "message": "Stacking multiple AHAs (glycolic, lactic, mandelic) significantly increases risk of over-exfoliation and barrier damage",
                "affected_ingredients": ["glycolic acid", "lactic acid", "mandelic acid"],
                "confidence": 0.92,
            }
        ],
        "overall_assessment": "review_needed",
    })

    regimen = MockRegimen(
        pm=[MockStep(["glycolic acid 8%", "lactic acid 5%", "mandelic acid 5%"])],
    )
    profile = {"pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)

    llm_flags = [f for f in report.flags if getattr(f, "source", "") == "llm_judge"]
    assert len(llm_flags) > 0


def test_llm_judge_standard_safe_routine():
    """
    Standard safe routine: LLM judge returns empty flags.
    Existing rule-based tests should still pass (no regression).
    Mock API returns no additional flags.
    """
    guard = SafetyGuard(use_llm_judge=True, anthropic_api_key="mock")
    guard._llm_judge = _make_mock_judge({
        "additional_flags": [],
        "overall_assessment": "safe",
    })

    regimen = MockRegimen(
        am=[
            MockStep(["niacinamide 5%"], product_type="Serum"),
            MockStep(["zinc oxide"], product_type="SPF 50 Sunscreen"),
        ],
        pm=[MockStep(["ceramides", "hyaluronic acid"], product_type="Moisturizer")],
    )
    profile = {"pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)

    llm_flags = [f for f in report.flags if getattr(f, "source", "") == "llm_judge"]
    assert len(llm_flags) == 0  # no LLM flags for safe routine

    # All existing rule-based checks should still pass
    all_flags = report.flags
    pregnancy_flags = [f for f in all_flags if "pregnancy" in f.message.lower()]
    assert len(pregnancy_flags) == 0


def test_llm_judge_skipped_for_pregnancy():
    """LLM judge must never be called for pregnancy=True profiles."""
    call_count = [0]

    guard = SafetyGuard(use_llm_judge=True, anthropic_api_key="mock")

    class TrackingJudge(LLMSafetyJudge):
        def __init__(self):
            super().__init__(api_key="mock-key")

        def judge(self, ingredients, profile):
            call_count[0] += 1
            return {"additional_flags": [], "overall_assessment": "safe"}

    guard._llm_judge = TrackingJudge()

    regimen = MockRegimen(pm=[MockStep(["niacinamide 5%"])])
    profile = {"pregnancy": True, "medications": [], "concerns": []}
    guard.check(regimen, profile)

    assert call_count[0] == 0, "LLM judge must not be called for pregnancy=True"


def test_llm_judge_confidence_threshold():
    """Flags with confidence <= 0.7 must be filtered out."""
    guard = SafetyGuard(use_llm_judge=True, anthropic_api_key="mock")
    guard._llm_judge = _make_mock_judge({
        "additional_flags": [
            {
                "severity": "warning",
                "message": "Low confidence flag — should be filtered",
                "affected_ingredients": ["niacinamide"],
                "confidence": 0.5,  # below threshold
            },
            {
                "severity": "caution",
                "message": "High confidence flag — should appear",
                "affected_ingredients": ["retinol"],
                "confidence": 0.85,  # above threshold
            },
        ],
        "overall_assessment": "caution",
    })

    regimen = MockRegimen(pm=[MockStep(["niacinamide 5%", "retinol 0.5%"])])
    profile = {"pregnancy": False, "medications": [], "concerns": []}
    report = guard.check(regimen, profile)

    llm_flags = [f for f in report.flags if getattr(f, "source", "") == "llm_judge"]
    assert len(llm_flags) == 1
    assert "high confidence" in llm_flags[0].message.lower()


# ── Run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_pregnancy_flags_retinol,
        test_pregnancy_flags_hydroquinone,
        test_pregnancy_no_flag_safe_ingredients,
        test_phototoxicity_flags_retinol_no_spf,
        test_phototoxicity_no_flag_with_spf,
        test_phototoxicity_glycolic_acid,
        test_pediatric_flags_tretinoin,
        test_geriatric_caution_benzoyl_peroxide,
        test_adult_no_age_flag,
        test_isotretinoin_blocks_retinoids,
        test_tetracycline_flags_aha,
        test_concentration_retinol_over_limit,
        test_concentration_within_limit,
        test_severe_acne_referral,
        test_psoriasis_referral,
        test_retinol_aha_conflict,
        test_allergen_fragrance_flagged,
        test_allergen_inci_synonym_match,
        test_allergen_no_flag_when_no_allergy,
        test_allergen_no_false_positive,
        # Prompt #05 — LLM judge
        test_llm_judge_high_concentration_vitamin_c_retinol,
        test_llm_judge_multiple_ahas,
        test_llm_judge_standard_safe_routine,
        test_llm_judge_skipped_for_pregnancy,
        test_llm_judge_confidence_threshold,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
