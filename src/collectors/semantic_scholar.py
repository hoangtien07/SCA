"""
src/collectors/semantic_scholar.py

Semantic Scholar Academic Graph API collector.
- Free, no API key required for basic use (100 req / 5 min)
- Optional API key for higher rate limits (get at semanticscholar.org)
- Returns up to 10,000 results per query with pagination

Docs: https://api.semanticscholar.org/api-docs/
"""
from __future__ import annotations

import time
from typing import Iterator

from loguru import logger

from .base_collector import BaseCollector, Paper


FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "externalIds",
    "citationCount",
    "openAccessPdf",
    "publicationTypes",
])


class SemanticScholarCollector(BaseCollector):
    """
    Collector for Semantic Scholar API.

    Usage:
        collector = SemanticScholarCollector(api_key="optional")
        papers = collector.collect_all_queries(queries, max_per_query=200)
    """

    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1"
    rate_limit_delay = 1.2                 # ~50 req/min without key; ~100/min with key

    def __init__(self, api_key: str = "", output_dir: str = "./data/raw"):
        super().__init__(output_dir)
        self.headers = {"x-api-key": api_key} if api_key else {}
        if api_key:
            self.rate_limit_delay = 0.5    # faster with key

    def search(self, query: str, max_results: int = 200) -> Iterator[Paper]:
        """
        Search Semantic Scholar and yield Paper objects.
        Paginates automatically up to max_results.
        """
        offset = 0
        limit = min(100, max_results)      # API max per page = 100
        fetched = 0

        while fetched < max_results:
            params = {
                "query": query,
                "offset": offset,
                "limit": limit,
                "fields": FIELDS,
            }
            try:
                data = self._get(
                    f"{self.base_url}/paper/search",
                    params=params,
                    headers=self.headers,
                )
            except Exception as e:
                logger.warning(f"[semantic_scholar] Request failed at offset {offset}: {e}")
                break

            papers_raw = data.get("data", [])
            if not papers_raw:
                break

            for raw in papers_raw:
                paper = self._parse(raw)
                if paper:
                    yield paper
                    fetched += 1

            # Check if more pages exist
            total = data.get("total", 0)
            offset += len(papers_raw)
            if offset >= total or offset >= max_results:
                break

            time.sleep(self.rate_limit_delay)

    def _parse(self, raw: dict) -> Paper | None:
        """Convert raw API response to normalized Paper."""
        paper_id = raw.get("paperId", "")
        title = raw.get("title", "").strip()
        abstract = (raw.get("abstract") or "").strip()

        if not paper_id or not title:
            return None

        authors = [
            a.get("name", "") for a in raw.get("authors", [])
            if a.get("name")
        ]

        doi = raw.get("externalIds", {}).get("DOI", "")
        pmid = raw.get("externalIds", {}).get("PubMed", "")

        # Prefer open access PDF url for later full-text retrieval
        oa = raw.get("openAccessPdf") or {}
        url = oa.get("url", "")
        if not url and doi:
            url = f"https://doi.org/{doi}"

        # Map publication types to our study_type vocabulary
        pub_types = raw.get("publicationTypes") or []
        study_type = _map_study_type(pub_types)

        return Paper(
            paper_id=f"ss_{paper_id}",
            title=title,
            abstract=abstract,
            year=raw.get("year"),
            authors=authors,
            journal=raw.get("venue", ""),
            doi=doi,
            url=url,
            citation_count=raw.get("citationCount", 0),
            source="semantic_scholar",
            study_type=study_type,
        )


def _map_study_type(pub_types: list[str]) -> str:
    """Map Semantic Scholar publication types to our evidence vocabulary."""
    types_lower = [t.lower() for t in pub_types]
    if any("review" in t for t in types_lower):
        return "review"
    if any("clinical" in t or "trial" in t for t in types_lower):
        return "RCT"
    if any("case" in t for t in types_lower):
        return "case_report"
    return "research_article"
