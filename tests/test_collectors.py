"""
tests/test_collectors.py
Basic sanity tests — run without API keys, no network required.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collectors.base_collector import Paper
from src.pipeline.metadata_tagger import tag_paper, _find_conditions, _find_ingredients
from src.pipeline.chunker import PaperChunker, count_tokens
from src.agents.safety_guard import SafetyGuard


# ── Paper dataclass tests ─────────────────────────────────────────────────────

def test_paper_is_valid():
    p = Paper(
        paper_id="ss_123", title="Test paper",
        abstract="A" * 200, year=2022, source="semantic_scholar"
    )
    assert p.is_valid()

def test_paper_invalid_short_abstract():
    p = Paper(paper_id="ss_1", title="T", abstract="short", year=2022, source="ss")
    assert not p.is_valid()

def test_paper_to_from_dict():
    p = Paper(paper_id="ss_abc", title="Title", abstract="Abstract text " * 20,
              year=2021, source="semantic_scholar", evidence_level="A")
    d = p.to_dict()
    p2 = Paper.from_dict(d)
    assert p2.paper_id == p.paper_id
    assert p2.evidence_level == "A"


# ── Metadata tagger tests ──────────────────────────────────────────────────────

def test_tag_paper_detects_acne():
    p = Paper(
        paper_id="ss_1", title="Acne vulgaris treatment with niacinamide",
        abstract="This study examines the effect of niacinamide on acne and pimples "
                 "in patients with oily skin. Results showed significant reduction in "
                 "inflammatory lesions after 8 weeks of treatment. " * 3,
        year=2022, source="semantic_scholar", study_type="RCT"
    )
    tagged = tag_paper(p)
    assert "acne" in tagged.skin_conditions
    assert tagged.evidence_level == "A"

def test_tag_paper_detects_ingredients():
    p = Paper(
        paper_id="ss_2", title="Vitamin C and niacinamide for hyperpigmentation",
        abstract="Ascorbic acid (vitamin C) and niacinamide were applied topically "
                 "to assess effects on melanin and dark spots over 12 weeks. " * 3,
        year=2021, source="semantic_scholar", study_type="review"
    )
    tagged = tag_paper(p)
    assert tagged.evidence_level == "B"

def test_tag_evidence_levels():
    for study_type, expected in [
        ("RCT", "A"), ("meta_analysis", "A"), ("systematic_review", "A"),
        ("review", "B"), ("cohort", "B"),
        ("case_report", "C"), ("research_article", "C"), ("in_vitro", "C"),
    ]:
        p = Paper(paper_id="x", title="t", abstract="a" * 200,
                  year=2020, source="ss", study_type=study_type)
        tagged = tag_paper(p)
        assert tagged.evidence_level == expected, f"Failed for {study_type}"


# ── Chunker tests ─────────────────────────────────────────────────────────────

def test_chunker_basic():
    p = Paper(
        paper_id="ss_c1", title="Retinol efficacy review",
        abstract="Retinol has been shown to reduce fine lines and wrinkles. " * 30,
        year=2020, source="semantic_scholar"
    )
    chunker = PaperChunker(chunk_size=512, overlap=64)
    chunks = chunker.chunk_paper(p)
    assert len(chunks) >= 1
    assert all(c.paper_id == "ss_c1" for c in chunks)
    assert all(c.token_count > 0 for c in chunks)

def test_chunker_title_prepended():
    p = Paper(paper_id="ss_c2", title="My Test Title",
              abstract="Some abstract text. " * 20, year=2021, source="ss")
    chunker = PaperChunker()
    chunks = chunker.chunk_paper(p)
    assert "My Test Title" in chunks[0].text

def test_chunker_chunk_ids_unique():
    p = Paper(paper_id="ss_c3", title="Title",
              abstract="Long abstract. " * 100, year=2022, source="ss")
    chunker = PaperChunker(chunk_size=200, overlap=30)
    chunks = chunker.chunk_paper(p)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

def test_count_tokens():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


# ── Safety guard tests ────────────────────────────────────────────────────────

class MockStep:
    def __init__(self, ingredients):
        self.active_ingredients = ingredients

class MockRegimen:
    def __init__(self, am, pm, weekly=None):
        self.am_routine = am
        self.pm_routine = pm
        self.weekly_treatments = weekly or []
        self.contraindications = []

def test_safety_pregnancy_retinol():
    guard = SafetyGuard()
    regimen = MockRegimen(
        am=[MockStep(["vitamin C"])],
        pm=[MockStep(["retinol 0.5%", "niacinamide"])],
    )
    profile = {"pregnancy": True, "medications": [], "concerns": [], "acne_severity": "none"}
    report = guard.check(regimen, profile)
    assert report.has_warnings
    assert any("retinol" in f.message.lower() or "pregnancy" in f.message.lower()
               for f in report.flags)

def test_safety_no_issues():
    guard = SafetyGuard()
    regimen = MockRegimen(
        am=[MockStep(["niacinamide", "zinc SPF"])],
        pm=[MockStep(["ceramides", "hyaluronic acid"])],
    )
    profile = {"pregnancy": False, "medications": [], "concerns": [], "acne_severity": "mild"}
    report = guard.check(regimen, profile)
    assert not report.has_warnings

def test_safety_isotretinoin_retinoid():
    guard = SafetyGuard()
    regimen = MockRegimen(
        am=[MockStep(["vitamin C"])],
        pm=[MockStep(["retinol", "niacinamide"])],
    )
    profile = {"pregnancy": False, "medications": ["isotretinoin"],
               "concerns": [], "acne_severity": "severe"}
    report = guard.check(regimen, profile)
    assert any("isotretinoin" in f.message.lower() for f in report.flags)


# ── Run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_paper_is_valid,
        test_paper_invalid_short_abstract,
        test_paper_to_from_dict,
        test_tag_paper_detects_acne,
        test_tag_paper_detects_ingredients,
        test_tag_evidence_levels,
        test_chunker_basic,
        test_chunker_title_prepended,
        test_chunker_chunk_ids_unique,
        test_count_tokens,
        test_safety_pregnancy_retinol,
        test_safety_no_issues,
        test_safety_isotretinoin_retinoid,
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
