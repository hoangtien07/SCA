"""
src/pipeline/metadata_tagger.py

Auto-tags Paper objects with:
  - skin_conditions  (from taxonomy)
  - active_ingredients (from taxonomy)
  - evidence_level   (A / B / C based on study_type)

Uses keyword matching — fast, zero API cost.
Optional: upgrade to LLM-based tagging for higher accuracy (see docstring).
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.collectors.base_collector import Paper


# ── Load taxonomy once at import ─────────────────────────────────────────────
_TAXONOMY_PATH = Path(__file__).parent.parent.parent / "config" / "skin_conditions.yaml"

with open(_TAXONOMY_PATH, encoding="utf-8") as f:
    _TAXONOMY = yaml.safe_load(f)


def _build_condition_index() -> dict[str, list[str]]:
    """Build {keyword_lower: [condition_label, ...]} lookup."""
    index: dict[str, list[str]] = {}
    for condition, data in _TAXONOMY.get("conditions", {}).items():
        for kw in data.get("keywords", []):
            index.setdefault(kw.lower(), []).append(condition)
    return index


def _build_ingredient_index() -> dict[str, str]:
    """Build {ingredient_lower: canonical_name} lookup."""
    index: dict[str, str] = {}
    actives = _TAXONOMY.get("active_ingredients", {})

    def _add(name: str) -> None:
        canonical = name.replace("_", " ").lower()
        index[canonical] = name

    for category, value in actives.items():
        if isinstance(value, list):
            for item in value:
                _add(item)
        elif isinstance(value, dict):
            for sub_list in value.values():
                if isinstance(sub_list, list):
                    for item in sub_list:
                        _add(item)

    return index


_CONDITION_INDEX = _build_condition_index()
_INGREDIENT_INDEX = _build_ingredient_index()

_EVIDENCE_MAP = {
    "RCT": "A",
    "meta_analysis": "A",
    "systematic_review": "A",
    "clinical_trial": "B",
    "review": "B",
    "cohort": "B",
    "case_control": "B",
    "case_report": "C",
    "research_article": "C",
    "in_vitro": "C",
}


# ── Public API ────────────────────────────────────────────────────────────────

def tag_paper(paper: Paper) -> Paper:
    """
    In-place tag a Paper with conditions, ingredients, and evidence level.
    Returns the same Paper object (mutated).
    """
    text = f"{paper.title} {paper.abstract}".lower()
    text = re.sub(r"[^\w\s\-]", " ", text)   # normalize punctuation

    paper.skin_conditions = _find_conditions(text)
    paper.active_ingredients = _find_ingredients(text)
    paper.evidence_level = _EVIDENCE_MAP.get(paper.study_type, "C")

    return paper


def tag_papers(papers: list[Paper]) -> list[Paper]:
    """Tag a list of papers. Returns same list with tags applied."""
    return [tag_paper(p) for p in papers]


# ── Internal ─────────────────────────────────────────────────────────────────

def _find_conditions(text: str) -> list[str]:
    found: set[str] = set()
    for keyword, conditions in _CONDITION_INDEX.items():
        if keyword in text:
            found.update(conditions)
    return sorted(found)


def _find_ingredients(text: str) -> list[str]:
    found: set[str] = set()
    for canonical, name in _INGREDIENT_INDEX.items():
        if canonical in text:
            found.add(name)
    return sorted(found)


# ── Optional: LLM-based tagging (more accurate, costs API calls) ─────────────
# Uncomment and use when keyword matching isn't precise enough.
#
# async def tag_paper_llm(paper: Paper, client: AsyncOpenAI) -> Paper:
#     """Use GPT-4o-mini to extract conditions and ingredients from abstract."""
#     prompt = f"""
#     Extract from this dermatology paper abstract:
#     1. skin_conditions: list from {list(_TAXONOMY['conditions'].keys())}
#     2. active_ingredients: list of mentioned skincare actives
#     3. study_type: one of RCT|meta_analysis|systematic_review|review|cohort|case_report|research_article|in_vitro
#
#     Abstract: {paper.abstract[:1000]}
#
#     Respond only with valid JSON: {{"skin_conditions": [], "active_ingredients": [], "study_type": ""}}
#     """
#     resp = await client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         response_format={"type": "json_object"},
#         max_tokens=200,
#     )
#     data = json.loads(resp.choices[0].message.content)
#     paper.skin_conditions = data.get("skin_conditions", [])
#     paper.active_ingredients = data.get("active_ingredients", [])
#     paper.study_type = data.get("study_type", paper.study_type)
#     paper.evidence_level = _EVIDENCE_MAP.get(paper.study_type, "C")
#     return paper
