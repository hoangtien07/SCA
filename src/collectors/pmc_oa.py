"""
src/collectors/pmc_oa.py

PMC Open Access collector — retrieves full-text articles from PubMed Central.
- Same NCBI E-utilities as PubMed, but queries the "pmc" database
- Returns structured full-text with section markers for section-aware chunking
- Rate limits identical to PubMed: 3 req/s (no key), 10 req/s (with key)

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
PMC OA: https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from typing import Iterator

from loguru import logger

from .base_collector import BaseCollector, Paper


class PMCOpenAccessCollector(BaseCollector):
    """
    Collector for PMC Open Access full-text articles via NCBI E-utilities.

    Retrieves articles from the ``pmc`` database and extracts structured
    section text (abstract, methods, results, conclusion) with markers
    that the section-aware chunker can split on.

    Usage:
        collector = PMCOpenAccessCollector(api_key="optional_ncbi_key")
        papers = collector.collect_all_queries(queries, max_per_query=100)
    """

    name = "pmc_oa"
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    rate_limit_delay = 0.35

    def __init__(self, api_key: str = "", output_dir: str = "./data/raw"):
        super().__init__(output_dir)
        self.api_key = api_key
        if api_key:
            self.rate_limit_delay = 0.1

    def search(self, query: str, max_results: int = 100) -> Iterator[Paper]:
        """
        Two-step flow: esearch on ``pmc`` DB → efetch full-text XML.
        """
        pmcids = self._esearch(query, max_results)
        if not pmcids:
            return

        batch_size = 20  # full-text XML is large, use smaller batches
        for i in range(0, len(pmcids), batch_size):
            batch = pmcids[i : i + batch_size]
            papers = self._efetch(batch)
            for paper in papers:
                yield paper
            time.sleep(self.rate_limit_delay)

    def _esearch(self, query: str, max_results: int) -> list[str]:
        """Return list of PMC IDs matching the query (open access filter)."""
        params: dict = {
            "db": "pmc",
            "term": f"{query} AND open access[filter]",
            "retmax": min(max_results, 5000),
            "retmode": "json",
            "usehistory": "n",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            data = self._get(f"{self.base_url}/esearch.fcgi", params=params)
        except Exception as e:
            logger.warning(f"[pmc_oa] esearch failed: {e}")
            return []

        return data.get("esearchresult", {}).get("idlist", [])

    def _efetch(self, pmcids: list[str]) -> list[Paper]:
        """Fetch full-text XML for a batch of PMC IDs."""
        params: dict = {
            "db": "pmc",
            "id": ",".join(pmcids),
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            import requests

            resp = requests.get(
                f"{self.base_url}/efetch.fcgi",
                params=params,
                timeout=60,  # full-text XML can be large
            )
            resp.raise_for_status()
            return self._parse_xml(resp.text)
        except Exception as e:
            logger.warning(f"[pmc_oa] efetch failed for {len(pmcids)} articles: {e}")
            return []

    def _parse_xml(self, xml_text: str) -> list[Paper]:
        """Parse PMC full-text XML into Paper objects."""
        papers: list[Paper] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning(f"[pmc_oa] XML parse error: {e}")
            return []

        for article in root.findall(".//article"):
            paper = self._parse_article(article)
            if paper:
                papers.append(paper)

        return papers

    def _parse_article(self, article: ET.Element) -> Paper | None:
        """Extract structured fields from a PMC <article> element."""
        # PMC ID
        pmcid = ""
        for aid in article.findall(".//article-id"):
            if aid.get("pub-id-type") == "pmc":
                pmcid = (aid.text or "").strip()
                break
        if not pmcid:
            return None

        # Title
        title_el = article.find(".//article-title")
        title = _xml_text(title_el).strip()
        if not title:
            return None

        # Abstract
        abstract_parts: list[str] = []
        for abstract_el in article.findall(".//abstract"):
            for sec in abstract_el.findall(".//sec"):
                sec_title = _xml_text(sec.find("title")).strip()
                sec_body = " ".join(
                    _xml_text(p) for p in sec.findall("p")
                ).strip()
                if sec_body:
                    abstract_parts.append(
                        f"{sec_title}: {sec_body}" if sec_title else sec_body
                    )
            # Some abstracts have no <sec> children — just <p> elements
            if not abstract_parts:
                for p in abstract_el.findall("p"):
                    text = _xml_text(p).strip()
                    if text:
                        abstract_parts.append(text)

        abstract_text = " ".join(abstract_parts).strip()

        # Full-text body sections
        sections = self._extract_body_sections(article)

        # Build combined text with section markers
        combined = _build_sectioned_text(abstract_text, sections)
        if len(combined) < 100:
            return None

        # Authors
        authors: list[str] = []
        for contrib in article.findall(".//contrib[@contrib-type='author']"):
            surname = _xml_text(contrib.find(".//surname"))
            given = _xml_text(contrib.find(".//given-names"))
            if surname:
                authors.append(f"{surname} {given}".strip())

        # Journal
        journal = _xml_text(article.find(".//journal-title"))

        # Year
        year = None
        year_el = article.find(".//pub-date/year")
        if year_el is not None and year_el.text:
            try:
                year = int(year_el.text[:4])
            except ValueError:
                pass

        # DOI
        doi = ""
        for aid in article.findall(".//article-id"):
            if aid.get("pub-id-type") == "doi":
                doi = (aid.text or "").strip()

        # Study type from article-type attribute or subject groups
        study_type = _detect_study_type(article)

        return Paper(
            paper_id=f"pmc_{pmcid}",
            title=title,
            abstract=combined,
            year=year,
            authors=authors,
            journal=journal,
            doi=doi,
            url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
            citation_count=0,
            source="pmc_oa",
            study_type=study_type,
        )

    def _extract_body_sections(self, article: ET.Element) -> dict[str, str]:
        """
        Extract body sections, mapping normalized section names to text.
        Returns dict like {"methods": "...", "results": "...", "discussion": "..."}.
        """
        sections: dict[str, str] = {}
        body = article.find(".//body")
        if body is None:
            return sections

        for sec in body.findall("sec"):
            sec_title_el = sec.find("title")
            raw_title = _xml_text(sec_title_el).strip().lower()
            normalized = _normalize_section_name(raw_title)
            if not normalized:
                continue

            paragraphs = " ".join(_xml_text(p) for p in sec.findall(".//p")).strip()
            if paragraphs:
                if normalized in sections:
                    sections[normalized] += " " + paragraphs
                else:
                    sections[normalized] = paragraphs

        return sections


# ── Helpers ──────────────────────────────────────────────────────────────────


def _xml_text(el: ET.Element | None) -> str:
    """Safely extract all text from an XML element, including tail text."""
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


_SECTION_MAP: dict[str, str] = {
    "method": "methods",
    "methods": "methods",
    "materials and methods": "methods",
    "material and methods": "methods",
    "patients and methods": "methods",
    "experimental": "methods",
    "study design": "methods",
    "result": "results",
    "results": "results",
    "findings": "results",
    "outcome": "results",
    "outcomes": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "summary": "conclusion",
    "introduction": "introduction",
    "background": "introduction",
}


def _normalize_section_name(raw: str) -> str:
    """Map a raw section title to one of our standard names."""
    raw_lower = raw.lower().strip()
    # Direct match
    if raw_lower in _SECTION_MAP:
        return _SECTION_MAP[raw_lower]
    # Substring match for headings like "2. Methods" or "Materials and Methods"
    for key, value in _SECTION_MAP.items():
        if key in raw_lower:
            return value
    return ""


def _build_sectioned_text(abstract: str, sections: dict[str, str]) -> str:
    """
    Build combined text with section markers for the chunker.
    Format: [ABSTRACT] ... [METHODS] ... [RESULTS] ... [CONCLUSION] ...
    """
    parts: list[str] = []

    if abstract:
        parts.append(f"[ABSTRACT] {abstract}")

    # Ordered section output
    for key in ("introduction", "methods", "results", "discussion", "conclusion"):
        if key in sections:
            marker = key.upper()
            parts.append(f"[{marker}] {sections[key]}")

    if not parts:
        return abstract or ""

    return "\n\n".join(parts)


def _detect_study_type(article: ET.Element) -> str:
    """Detect study type from article-type attribute or subject groups."""
    # Check article-type attribute
    article_type = article.get("article-type", "").lower()
    if "review" in article_type:
        return "review"
    if "research" in article_type:
        return "research_article"
    if "case" in article_type:
        return "case_report"

    # Check subject groups for hints
    subjects = " ".join(
        _xml_text(s) for s in article.findall(".//subject")
    ).lower()

    if "randomized" in subjects or "rct" in subjects:
        return "RCT"
    if "meta-analysis" in subjects:
        return "meta_analysis"
    if "systematic review" in subjects:
        return "systematic_review"
    if "review" in subjects:
        return "review"
    if "clinical trial" in subjects:
        return "clinical_trial"
    if "case" in subjects:
        return "case_report"

    # Check title/abstract text as last resort
    title = _xml_text(article.find(".//article-title")).lower()
    if "systematic review" in title or "meta-analysis" in title:
        return "systematic_review"
    if "randomized" in title or "randomised" in title:
        return "RCT"
    if "case report" in title:
        return "case_report"

    return "research_article"
