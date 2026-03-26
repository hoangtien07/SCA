"""
src/pipeline/chunker.py

Splits Paper objects into smaller overlapping chunks for embedding.

Strategy:
  - Abstract      → 1 chunk (always kept whole, high signal)
  - Full-text     → section-aware chunks (split by [ABSTRACT]/[METHODS]/[RESULTS]/[CONCLUSION] markers)
  - Body sections → sliding window, 512 tokens, 64 token overlap
  - Enriched header with title + year + study_type + evidence_level prepended to every chunk

Each chunk carries full metadata so filters work at retrieval time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

from src.collectors.base_collector import Paper


# ── Token counter (shared across calls) ──────────────────────────────────────
_ENC = tiktoken.get_encoding("cl100k_base")   # matches text-embedding-3-small

# Section markers used by PMC OA collector for full-text papers
_SECTION_PATTERN = re.compile(
    r"\[(ABSTRACT|METHODS|RESULTS|CONCLUSION|DISCUSSION|INTRODUCTION)\]",
    re.IGNORECASE,
)


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single embeddable unit of text, derived from a Paper.
    Carries all metadata needed for filtering at retrieval time.
    """
    chunk_id: str                              # f"{paper_id}_chunk_{n}"
    paper_id: str
    text: str                                  # text to embed
    chunk_type: str                            # "abstract" | "body" | "title_abstract"

    # Inherited from Paper
    title: str = ""
    year: int | None = None
    journal: str = ""
    doi: str = ""
    url: str = ""
    citation_count: int = 0
    source: str = ""
    skin_conditions: list[str] = field(default_factory=list)
    active_ingredients: list[str] = field(default_factory=list)
    evidence_level: str = ""
    study_type: str = ""
    token_count: int = 0

    def to_chroma_dict(self) -> dict:
        """ChromaDB metadata must be flat (no nested lists as values)."""
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "chunk_type": self.chunk_type,
            "title": self.title,
            "year": self.year or 0,
            "journal": self.journal,
            "doi": self.doi,
            "url": self.url,
            "citation_count": self.citation_count,
            "source": self.source,
            # Lists → comma-separated strings for ChromaDB compatibility
            "skin_conditions": ",".join(self.skin_conditions),
            "active_ingredients": ",".join(self.active_ingredients),
            "evidence_level": self.evidence_level,
            "study_type": self.study_type,
            "token_count": self.token_count,
        }


# ── Main chunker ──────────────────────────────────────────────────────────────

class PaperChunker:
    """
    Converts Paper objects into lists of Chunk objects.

    Usage:
        chunker = PaperChunker(chunk_size=512, overlap=64)
        chunks = chunker.chunk_papers(papers)
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_paper(self, paper: Paper) -> list[Chunk]:
        """Split one paper into chunks. Supports section-aware chunking for full-text."""
        chunks: list[Chunk] = []
        header = self._build_header(paper)

        # ── Check for section markers (full-text from PMC OA) ─────────
        sections = self._split_sections(paper.abstract)

        if sections:
            # Section-aware chunking: each section chunked independently
            idx = 0
            for section_name, section_text in sections.items():
                if not section_text.strip():
                    continue
                section_prefix = f"{header}\n[{section_name.upper()}]\n"
                if count_tokens(section_text) <= self.chunk_size:
                    text = f"{section_prefix}{section_text}"
                    chunks.append(self._make_chunk(paper, text, idx, section_name.lower()))
                    idx += 1
                else:
                    for window in self._sliding_window(section_text):
                        text = f"{section_prefix}{window}"
                        chunks.append(self._make_chunk(paper, text, idx, section_name.lower()))
                        idx += 1
        else:
            # ── Abstract-only chunking (original logic) ───────────────
            if paper.abstract:
                abstract_text = f"{header}\n\nAbstract: {paper.abstract}"
                chunks.append(self._make_chunk(paper, abstract_text, 0, "abstract"))

            # ── Sliding window on abstract if very long (>600 tokens) ─
            if count_tokens(paper.abstract) > 600:
                for i, window in enumerate(
                    self._sliding_window(paper.abstract), start=1
                ):
                    text = f"{header}\n\n{window}"
                    chunks.append(self._make_chunk(paper, text, i, "body"))

        return chunks

    def chunk_papers(self, papers: list[Paper]) -> list[Chunk]:
        """Chunk all papers, return flat list."""
        all_chunks: list[Chunk] = []
        for paper in papers:
            all_chunks.extend(self.chunk_paper(paper))
        return all_chunks

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_header(self, paper: Paper) -> str:
        """Build an enriched header with study metadata for self-contained chunks."""
        parts = [paper.title]
        if paper.year:
            parts[0] += f" ({paper.year})"
        if paper.study_type:
            parts.append(f"Study: {paper.study_type}")
        if paper.evidence_level:
            parts.append(f"Evidence: {paper.evidence_level}")
        if paper.journal:
            parts.append(f"Journal: {paper.journal}")
        return " | ".join(parts)

    def _split_sections(self, text: str) -> dict[str, str] | None:
        """
        Split text by section markers [ABSTRACT], [METHODS], [RESULTS], [CONCLUSION].
        Returns None if no markers found (abstract-only paper).
        """
        markers = list(_SECTION_PATTERN.finditer(text))
        if len(markers) < 2:
            return None

        sections: dict[str, str] = {}
        for i, match in enumerate(markers):
            section_name = match.group(1).upper()
            start = match.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            sections[section_name] = text[start:end].strip()

        return sections

    def _make_chunk(
        self, paper: Paper, text: str, idx: int, chunk_type: str
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{paper.paper_id}_chunk_{idx}",
            paper_id=paper.paper_id,
            text=text,
            chunk_type=chunk_type,
            title=paper.title,
            year=paper.year,
            journal=paper.journal,
            doi=paper.doi,
            url=paper.url,
            citation_count=paper.citation_count,
            source=paper.source,
            skin_conditions=paper.skin_conditions,
            active_ingredients=paper.active_ingredients,
            evidence_level=paper.evidence_level,
            study_type=paper.study_type,
            token_count=count_tokens(text),
        )

    def _sliding_window(self, text: str) -> list[str]:
        """Split text into overlapping token windows."""
        tokens = _ENC.encode(text)
        windows: list[str] = []
        step = self.chunk_size - self.overlap

        for start in range(0, len(tokens), step):
            end = start + self.chunk_size
            window_tokens = tokens[start:end]
            windows.append(_ENC.decode(window_tokens))
            if end >= len(tokens):
                break

        return windows
