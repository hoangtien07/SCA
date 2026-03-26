"""
src/agents/citation_checker.py

Post-generation verification: checks that each citation in the regimen
actually appears in the retrieved evidence chunks.

Flags ungrounded citations (possible hallucinated references) so the
SafetyGuard or UI can warn users.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger


@dataclass
class CitationCheckResult:
    """Result of checking one citation."""
    citation: str
    grounded: bool
    matched_title: str = ""
    similarity: float = 0.0


@dataclass
class CitationReport:
    """Aggregate report for all citations in a regimen."""
    results: list[CitationCheckResult] = field(default_factory=list)
    total_citations: int = 0
    grounded_count: int = 0
    ungrounded_citations: list[str] = field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        if self.total_citations == 0:
            return 1.0
        return self.grounded_count / self.total_citations

    def summary(self) -> str:
        if not self.ungrounded_citations:
            return f"All {self.total_citations} citations grounded in evidence."
        return (
            f"{self.grounded_count}/{self.total_citations} citations grounded. "
            f"Ungrounded: {', '.join(self.ungrounded_citations)}"
        )


class CitationChecker:
    """
    Verifies that citations in a generated regimen can be traced
    back to retrieved evidence chunks.

    Uses fuzzy title matching — exact match not required since
    the LLM may paraphrase or abbreviate paper titles.

    Usage:
        checker = CitationChecker()
        report = checker.check(regimen, evidence_chunks)
        if report.ungrounded_citations:
            logger.warning(f"Ungrounded: {report.ungrounded_citations}")
    """

    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold

    def check(self, regimen, evidence_chunks: list) -> CitationReport:
        """
        Check all citations in a regimen against evidence chunks.

        Args:
            regimen: Regimen object with am_routine, pm_routine, weekly_treatments
            evidence_chunks: List of RetrievalResult or similar with .title attribute
        """
        # Collect all evidence titles
        evidence_titles = []
        for chunk in evidence_chunks:
            title = getattr(chunk, "title", "") or ""
            if title:
                evidence_titles.append(title.lower().strip())

        # Collect all citations from regimen
        all_citations = self._collect_citations(regimen)

        report = CitationReport(total_citations=len(all_citations))

        for citation in all_citations:
            result = self._check_one(citation, evidence_titles)
            report.results.append(result)
            if result.grounded:
                report.grounded_count += 1
            else:
                report.ungrounded_citations.append(citation)

        if report.ungrounded_citations:
            logger.warning(
                f"[CitationChecker] {len(report.ungrounded_citations)} ungrounded "
                f"citations detected: {report.ungrounded_citations[:3]}"
            )
        else:
            logger.info(
                f"[CitationChecker] All {report.total_citations} citations grounded"
            )

        return report

    def _check_one(
        self, citation: str, evidence_titles: list[str]
    ) -> CitationCheckResult:
        """Check a single citation against evidence titles."""
        citation_lower = citation.lower().strip()

        # Exact substring match first (fast path)
        for title in evidence_titles:
            if citation_lower in title or title in citation_lower:
                return CitationCheckResult(
                    citation=citation, grounded=True,
                    matched_title=title, similarity=1.0,
                )

        # Token overlap similarity (lightweight fuzzy match)
        best_sim = 0.0
        best_title = ""
        citation_tokens = set(citation_lower.split())

        for title in evidence_titles:
            title_tokens = set(title.split())
            if not citation_tokens or not title_tokens:
                continue
            overlap = len(citation_tokens & title_tokens)
            union = len(citation_tokens | title_tokens)
            sim = overlap / union if union > 0 else 0.0

            if sim > best_sim:
                best_sim = sim
                best_title = title

        grounded = best_sim >= self.similarity_threshold
        return CitationCheckResult(
            citation=citation, grounded=grounded,
            matched_title=best_title, similarity=best_sim,
        )

    def _collect_citations(self, regimen) -> list[str]:
        """Extract all unique citations from a regimen."""
        citations = set()
        all_steps = (
            getattr(regimen, "am_routine", [])
            + getattr(regimen, "pm_routine", [])
            + getattr(regimen, "weekly_treatments", [])
        )
        for step in all_steps:
            for cite in getattr(step, "citations", []):
                if cite and cite.strip():
                    citations.add(cite.strip())
        return list(citations)
