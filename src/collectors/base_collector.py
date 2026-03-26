"""
src/collectors/base_collector.py
Abstract base class for all data source collectors.
Subclass this for each API (Semantic Scholar, PubMed, OpenAlex).
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class Paper:
    """
    Normalized paper record — same schema regardless of source API.
    All collectors must return List[Paper].
    """
    paper_id: str                          # source-prefixed: "ss_<id>" or "pm_<pmid>"
    title: str
    abstract: str
    year: int | None
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    doi: str = ""
    url: str = ""
    citation_count: int = 0
    source: str = ""                       # "semantic_scholar" | "pubmed" | "openalex"

    # Skincare-specific metadata (filled during processing phase)
    skin_conditions: list[str] = field(default_factory=list)
    active_ingredients: list[str] = field(default_factory=list)
    evidence_level: str = ""               # A | B | C
    study_type: str = ""                   # RCT | review | cohort | in_vitro | ...

    def is_valid(self) -> bool:
        """Minimum quality check before indexing."""
        return (
            bool(self.title)
            and bool(self.abstract)
            and len(self.abstract) > 100
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class BaseCollector(ABC):
    """
    Abstract base for all source collectors.

    Subclasses must implement:
        search(query, max_results) -> Iterator[Paper]
    """

    name: str = "base"
    base_url: str = ""
    rate_limit_delay: float = 0.5          # seconds between requests

    def __init__(self, output_dir: str | Path = "./data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._session = None

    @abstractmethod
    def search(self, query: str, max_results: int = 100) -> Iterator[Paper]:
        """Yield Paper objects for a given query string."""
        ...

    def collect_all_queries(
        self,
        queries: list[str],
        max_per_query: int = 200,
    ) -> list[Paper]:
        """
        Run all queries, deduplicate by paper_id, save to disk.
        Returns deduplicated list of Paper objects.
        """
        seen_ids: set[str] = set()
        all_papers: list[Paper] = []

        for i, query in enumerate(queries, 1):
            logger.info(f"[{self.name}] Query {i}/{len(queries)}: {query!r}")
            try:
                for paper in self.search(query, max_results=max_per_query):
                    if paper.paper_id not in seen_ids and paper.is_valid():
                        seen_ids.add(paper.paper_id)
                        all_papers.append(paper)
            except Exception as e:
                logger.warning(f"[{self.name}] Query failed: {e}")

            time.sleep(self.rate_limit_delay)

        logger.info(f"[{self.name}] Collected {len(all_papers)} unique papers")
        self._save(all_papers)
        return all_papers

    def _save(self, papers: list[Paper]) -> Path:
        """Persist papers as JSONL to data/raw/<source>.jsonl"""
        output_path = self.output_dir / f"{self.name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for p in papers:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"[{self.name}] Saved {len(papers)} papers → {output_path}")
        return output_path

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _get(url: str, params: dict | None = None, headers: dict | None = None) -> dict:
        """Retry-wrapped GET request."""
        import requests
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
