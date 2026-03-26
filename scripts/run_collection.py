"""
scripts/run_collection.py

Phase 0 — Step 1: Collect scientific papers from all configured sources.

Run:
    python scripts/run_collection.py
    python scripts/run_collection.py --sources pubmed --max 100
    python scripts/run_collection.py --dry-run   # test queries without saving

Output:
    data/raw/semantic_scholar.jsonl
    data/raw/pubmed.jsonl
    data/raw/combined.jsonl  (deduplicated)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# Make sure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.collectors import SemanticScholarCollector, PubMedCollector, PMCOpenAccessCollector, Paper

console = Console()


def load_queries(source: str) -> list[str]:
    queries_path = Path("config/search_queries.yaml")
    with open(queries_path, encoding="utf-8") as f:
        all_queries = yaml.safe_load(f)
    return all_queries.get(source, [])


def combine_and_deduplicate(raw_dir: Path) -> list[Paper]:
    """Load all .jsonl files, deduplicate by paper_id."""
    seen: set[str] = set()
    combined: list[Paper] = []

    for jsonl_file in raw_dir.glob("*.jsonl"):
        if jsonl_file.stem == "combined":
            continue
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                pid = data.get("paper_id", "")
                if pid and pid not in seen:
                    seen.add(pid)
                    combined.append(Paper.from_dict(data))

    return combined


def save_combined(papers: list[Paper], raw_dir: Path) -> Path:
    output = raw_dir / "combined.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    return output


def print_stats(papers: list[Paper]) -> None:
    table = Table(title="Collection summary", show_header=True)
    table.add_column("Source", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("With abstract", justify="right")
    table.add_column("Avg citations", justify="right")

    by_source: dict[str, list[Paper]] = {}
    for p in papers:
        by_source.setdefault(p.source, []).append(p)

    for source, ps in by_source.items():
        with_abstract = sum(1 for p in ps if len(p.abstract) > 100)
        avg_cit = sum(p.citation_count for p in ps) / len(ps) if ps else 0
        table.add_row(source, str(len(ps)), str(with_abstract), f"{avg_cit:.0f}")

    console.print(table)
    console.print(f"\n[bold green]Total unique papers: {len(papers)}[/bold green]")


def main(args: argparse.Namespace) -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources_to_run = args.sources.split(",") if args.sources else ["semantic_scholar", "pubmed"]

    console.rule("[bold]Phase 0 — Paper Collection[/bold]")
    console.print(f"Sources: {sources_to_run}")
    console.print(f"Max per query: {args.max}")
    if args.dry_run:
        console.print("[yellow]DRY RUN — queries will print but nothing saved[/yellow]")

    # ── Semantic Scholar ──────────────────────────────────────────────────────
    if "semantic_scholar" in sources_to_run:
        queries = load_queries("semantic_scholar")
        console.print(f"\n[cyan]Semantic Scholar:[/cyan] {len(queries)} queries")

        if args.dry_run:
            for q in queries:
                console.print(f"  • {q}")
        else:
            collector = SemanticScholarCollector(
                api_key=getattr(settings, "semantic_scholar_api_key", ""),
                output_dir=str(raw_dir),
            )
            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"),
                BarColumn(), TimeElapsedColumn(), console=console
            ) as progress:
                task = progress.add_task("Collecting...", total=len(queries))
                for i, query in enumerate(queries):
                    progress.update(task, description=f"SS: {query[:50]}...", advance=1)
                    try:
                        papers = list(collector.search(query, max_results=args.max))
                        logger.debug(f"  → {len(papers)} papers")
                    except Exception as e:
                        logger.warning(f"Query failed: {e}")
                    time.sleep(1.2)

            # Save all at once after iteration
            collector.collect_all_queries(queries, max_per_query=args.max)

    # ── PubMed ────────────────────────────────────────────────────────────────
    if "pubmed" in sources_to_run:
        queries = load_queries("pubmed")
        console.print(f"\n[cyan]PubMed:[/cyan] {len(queries)} queries")

        if args.dry_run:
            for q in queries:
                console.print(f"  • {q}")
        else:
            collector = PubMedCollector(
                api_key=getattr(settings, "ncbi_api_key", ""),
                output_dir=str(raw_dir),
            )
            collector.collect_all_queries(queries, max_per_query=args.max)

    # ── PMC Open Access ──────────────────────────────────────────────────────
    if "pmc_oa" in sources_to_run:
        queries = load_queries("pmc_oa")
        console.print(f"\n[cyan]PMC Open Access:[/cyan] {len(queries)} queries")

        if args.dry_run:
            for q in queries:
                console.print(f"  \u2022 {q}")
        else:
            collector = PMCOpenAccessCollector(
                api_key=getattr(settings, "ncbi_api_key", ""),
                output_dir=str(raw_dir),
            )
            collector.collect_all_queries(queries, max_per_query=args.max)

    # ── Combine + deduplicate ─────────────────────────────────────────────────
    if not args.dry_run:
        console.print("\n[bold]Combining and deduplicating...[/bold]")
        combined = combine_and_deduplicate(raw_dir)
        save_combined(combined, raw_dir)
        print_stats(combined)
        console.print(f"\n[green]✓ Saved to data/raw/combined.jsonl[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect skincare research papers")
    parser.add_argument("--sources", default="semantic_scholar,pubmed,pmc_oa",
                        help="Comma-separated: semantic_scholar,pubmed,pmc_oa")
    parser.add_argument("--max", type=int, default=settings.max_papers_per_query,
                        help="Max papers per query")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print queries without collecting")
    args = parser.parse_args()
    main(args)
