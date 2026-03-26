"""
src/collectors/pubmed.py

PubMed / NCBI E-utilities collector.
- Free, no API key required (3 req/s); with key: 10 req/s
- Register free API key at: https://www.ncbi.nlm.nih.gov/account/
- Especially good for clinical dermatology + RCT studies

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Iterator

from loguru import logger

from .base_collector import BaseCollector, Paper


class PubMedCollector(BaseCollector):
    """
    Collector for PubMed via NCBI E-utilities.

    Usage:
        collector = PubMedCollector(api_key="optional_ncbi_key")
        papers = collector.collect_all_queries(queries, max_per_query=200)
    """

    name = "pubmed"
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    rate_limit_delay = 0.35                # ~3 req/s without key

    def __init__(self, api_key: str = "", output_dir: str = "./data/raw"):
        super().__init__(output_dir)
        self.api_key = api_key
        if api_key:
            self.rate_limit_delay = 0.1    # 10 req/s with key

    def search(self, query: str, max_results: int = 200) -> Iterator[Paper]:
        """
        Two-step E-utilities flow: esearch (get IDs) → efetch (get records).
        """
        pmids = self._esearch(query, max_results)
        if not pmids:
            return

        # Fetch in batches of 100 (API limit)
        batch_size = 100
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            records = self._efetch(batch)
            for paper in records:
                yield paper
            time.sleep(self.rate_limit_delay)

    def _esearch(self, query: str, max_results: int) -> list[str]:
        """Return list of PMIDs matching the query."""
        params: dict = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 10_000),
            "retmode": "json",
            "usehistory": "n",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            data = self._get(f"{self.base_url}/esearch.fcgi", params=params)
        except Exception as e:
            logger.warning(f"[pubmed] esearch failed: {e}")
            return []

        return data.get("esearchresult", {}).get("idlist", [])

    def _efetch(self, pmids: list[str]) -> list[Paper]:
        """Fetch full records for a batch of PMIDs, parse XML."""
        params: dict = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            import requests
            resp = requests.get(
                f"{self.base_url}/efetch.fcgi",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            return self._parse_xml(resp.text)
        except Exception as e:
            logger.warning(f"[pubmed] efetch failed: {e}")
            return []

    def _parse_xml(self, xml_text: str) -> list[Paper]:
        """Parse PubMed XML response into Paper objects."""
        papers = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning(f"[pubmed] XML parse error: {e}")
            return []

        for article in root.findall(".//PubmedArticle"):
            paper = self._parse_article(article)
            if paper:
                papers.append(paper)

        return papers

    def _parse_article(self, article: ET.Element) -> Paper | None:
        """Extract fields from a single PubmedArticle XML element."""
        # PMID
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            return None
        pmid = pmid_el.text.strip()

        # Title
        title_el = article.find(".//ArticleTitle")
        title = _xml_text(title_el).strip()
        if not title:
            return None

        # Abstract (may have multiple AbstractText sections)
        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join(_xml_text(a) for a in abstract_parts).strip()

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last = _xml_text(author.find("LastName"))
            first = _xml_text(author.find("ForeName"))
            if last:
                authors.append(f"{last} {first}".strip())

        # Journal
        journal_el = article.find(".//Journal/Title")
        journal = _xml_text(journal_el)

        # Year
        year = None
        year_el = (
            article.find(".//PubDate/Year")
            or article.find(".//PubDate/MedlineDate")
        )
        if year_el is not None and year_el.text:
            try:
                year = int(year_el.text[:4])
            except ValueError:
                pass

        # DOI
        doi = ""
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = (id_el.text or "").strip()

        # Study type from publication type list
        pub_types = [
            _xml_text(pt)
            for pt in article.findall(".//PublicationType")
        ]
        study_type = _map_pub_types(pub_types)

        return Paper(
            paper_id=f"pm_{pmid}",
            title=title,
            abstract=abstract,
            year=year,
            authors=authors,
            journal=journal,
            doi=doi,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            citation_count=0,              # PubMed doesn't expose citation count
            source="pubmed",
            study_type=study_type,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _xml_text(el: ET.Element | None) -> str:
    """Safely extract all text from an XML element, including tail text."""
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _map_pub_types(pub_types: list[str]) -> str:
    types_lower = [t.lower() for t in pub_types]
    if any("randomized" in t for t in types_lower):
        return "RCT"
    if any("meta-analysis" in t for t in types_lower):
        return "meta_analysis"
    if any("systematic review" in t for t in types_lower):
        return "systematic_review"
    if any("review" in t for t in types_lower):
        return "review"
    if any("case report" in t for t in types_lower):
        return "case_report"
    if any("clinical trial" in t for t in types_lower):
        return "clinical_trial"
    return "research_article"
