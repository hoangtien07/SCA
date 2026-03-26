"""
scripts/run_indexing.py

Phase 0 — Step 2: Process raw papers → tag metadata → chunk → embed → index.

Run:
    python scripts/run_indexing.py
    python scripts/run_indexing.py --input data/raw/combined.jsonl
    python scripts/run_indexing.py --backend qdrant   # for production

Prerequisites:
    - data/raw/combined.jsonl must exist (run run_collection.py first)
    - OPENAI_API_KEY must be set in .env

Output:
    data/knowledge_base/   (ChromaDB files)
    data/processed/tagged_papers.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.collectors.base_collector import Paper
from src.pipeline.metadata_tagger import tag_papers
from src.pipeline.chunker import PaperChunker
from src.pipeline.indexer import ChromaIndexer
from src.pipeline.bm25_index import BM25Index

console = Console()


def load_papers(jsonl_path: Path) -> list[Paper]:
    papers = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(Paper.from_dict(json.loads(line)))
    return papers


def save_tagged(papers: list[Paper], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def main(args: argparse.Namespace) -> None:
    console.rule("[bold]Phase 0 — Indexing Pipeline[/bold]")

    # ── 1. Load raw papers ────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_path}[/red]")
        console.print("Run [bold]python scripts/run_collection.py[/bold] first.")
        sys.exit(1)

    console.print(f"\n[1/4] Loading papers from {input_path}...")
    papers = load_papers(input_path)
    console.print(f"      Loaded [green]{len(papers)}[/green] papers")

    # ── 2. Tag metadata ───────────────────────────────────────────────────────
    console.print("\n[2/4] Tagging metadata (conditions, ingredients, evidence)...")
    tagged = tag_papers(papers)

    # Stats
    with_conditions = sum(1 for p in tagged if p.skin_conditions)
    with_ingredients = sum(1 for p in tagged if p.active_ingredients)
    console.print(f"      {with_conditions} papers tagged with conditions")
    console.print(f"      {with_ingredients} papers tagged with ingredients")

    # Save tagged papers
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    tagged_path = processed_dir / "tagged_papers.jsonl"
    save_tagged(tagged, tagged_path)
    console.print(f"      Saved → {tagged_path}")

    # ── 3. Chunk ──────────────────────────────────────────────────────────────
    console.print(f"\n[3/4] Chunking (size={settings.chunk_size}, overlap={settings.chunk_overlap})...")
    chunker = PaperChunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    chunks = chunker.chunk_papers(tagged)
    console.print(f"      Generated [green]{len(chunks)}[/green] chunks from {len(tagged)} papers")
    avg_tokens = sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
    console.print(f"      Average chunk size: {avg_tokens:.0f} tokens")

    # ── 4. Embed + Index ──────────────────────────────────────────────────────
    console.print(f"\n[4/4] Embedding + indexing ({args.backend})...")

    if not settings.openai_api_key:
        console.print("[red]OPENAI_API_KEY not set in .env[/red]")
        sys.exit(1)

    if args.backend == "chroma":
        indexer = ChromaIndexer(
            persist_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    elif args.backend == "qdrant":
        from src.pipeline.indexer import QdrantIndexer
        if not settings.qdrant_url:
            console.print("[red]QDRANT_URL not set in .env[/red]")
            sys.exit(1)
        indexer = QdrantIndexer(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            embedding_model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    else:
        console.print(f"[red]Unknown backend: {args.backend}[/red]")
        sys.exit(1)

    # Estimate cost
    total_tokens = sum(c.token_count for c in chunks)
    cost_estimate = (total_tokens / 1_000_000) * 0.02   # $0.02/1M tokens for text-embedding-3-small
    console.print(f"      Estimated embedding cost: ~${cost_estimate:.3f}")

    if not args.yes:
        confirm = console.input(f"      Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            console.print("Aborted.")
            sys.exit(0)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), console=console) as progress:
        task = progress.add_task("Indexing chunks...", total=len(chunks))
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            indexer.add(batch)
            progress.advance(task, len(batch))

    stats = indexer.stats()
    console.print(f"\n[bold green]✓ Indexing complete![/bold green]")
    console.print(f"  Total chunks in DB: {stats['total_chunks']}")
    # ── 5. Build BM25 sparse index ─────────────────────────────────────────
    console.print("\n[5/5] Building BM25 sparse index...")
    bm25 = BM25Index()
    bm25.build(chunks)
    bm25_path = str(Path(settings.chroma_persist_dir) / "bm25_index.json")
    bm25.save(bm25_path)
    console.print(f"      Saved BM25 index \u2192 {bm25_path}")
    console.print(f"\nNext step: [bold]streamlit run app.py[/bold]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index skincare papers into vector DB")
    parser.add_argument("--input", default="data/raw/combined.jsonl")
    parser.add_argument("--backend", choices=["chroma", "qdrant"], default="chroma")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip cost confirmation")
    args = parser.parse_args()
    main(args)
