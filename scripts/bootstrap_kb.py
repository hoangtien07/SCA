"""
scripts/bootstrap_kb.py

Master orchestration script — builds the skincare knowledge base end-to-end.
Idempotent: completed phases are skipped unless --force is used.

Phases:
  0. Pre-flight   — verify keys, create dirs, check API reachability
  1. Collection   — run all 3 collectors, save combined.jsonl (target: 800 papers)
  2. Quality filter — remove short abstracts, DOI duplicates, missing evidence_level
  3. Indexing     — embed + index + BM25
  4. RAGAS baseline — retrieval-only eval on 10 cases, save eval_baseline.json
  5. Summary      — print final stats table

Run:
    python scripts/bootstrap_kb.py
    python scripts/bootstrap_kb.py --force          # re-run all phases
    python scripts/bootstrap_kb.py --force --phase 3  # re-run only phase 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

_CHECKPOINT_FILE = Path("data/processed/.bootstrap_checkpoints.json")


def _load_checkpoints() -> dict:
    if _CHECKPOINT_FILE.exists():
        try:
            return json.loads(_CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_checkpoint(phase: int, meta: dict | None = None) -> None:
    _CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = _load_checkpoints()
    checkpoints[str(phase)] = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "meta": meta or {},
    }
    _CHECKPOINT_FILE.write_text(
        json.dumps(checkpoints, indent=2), encoding="utf-8"
    )


def _is_completed(phase: int, force: bool) -> bool:
    if force:
        return False
    checkpoints = _load_checkpoints()
    return str(phase) in checkpoints


# ── Phase 0: Pre-flight checks ────────────────────────────────────────────────


def phase_preflight() -> bool:
    """Verify API keys, create directories, check API reachability."""
    console.rule("[bold cyan]Phase 0 — Pre-flight checks[/bold cyan]")

    from config.settings import settings

    ok = True

    # Check API keys
    if not settings.openai_api_key:
        console.print("[red]MISSING: OPENAI_API_KEY not set in .env[/red]")
        ok = False
    else:
        console.print("[green]OK[/green] OPENAI_API_KEY set")

    if not settings.anthropic_api_key:
        console.print("[yellow]WARN: ANTHROPIC_API_KEY not set (needed for full eval)[/yellow]")
    else:
        console.print("[green]OK[/green] ANTHROPIC_API_KEY set")

    # Create required directories
    for d in ["data/raw", "data/processed", "data/knowledge_base"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        console.print(f"[green]OK[/green] Directory: {d}/")

    # Estimate time
    console.print("\n[bold]Estimated time:[/bold]")
    console.print("  Collection: ~10–30 min (rate-limited)")
    console.print("  Indexing:   ~5–15 min (depends on paper count + API speed)")
    console.print("  Evaluation: ~2–5 min")
    console.print("  Total:      ~20–50 min")

    return ok


# ── Phase 1: Collection ────────────────────────────────────────────────────────


def phase_collection(force: bool = False) -> int:
    """Run all 3 collectors and save combined.jsonl."""
    if _is_completed(1, force):
        combined = Path("data/raw/combined.jsonl")
        if combined.exists():
            with open(combined) as f:
                count = sum(1 for l in f if l.strip())
            console.print(f"[dim]Phase 1 already completed ({count} papers). Skipping.[/dim]")
            return count
        console.print("[dim]Phase 1 marked complete but combined.jsonl missing — re-running[/dim]")

    console.rule("[bold cyan]Phase 1 — Collection[/bold cyan]")

    try:
        from tqdm import tqdm
        from src.collectors.pubmed import PubMedCollector
        from src.collectors.semantic_scholar import SemanticScholarCollector
        from src.collectors.pmc_oa import PMCOpenAccessCollector
        from config.settings import settings

        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        search_queries_path = Path("config/search_queries.yaml")
        import yaml
        with open(search_queries_path, encoding="utf-8") as f:
            queries_data = yaml.safe_load(f)
        queries: list[str] = queries_data.get("queries", [])[:10]  # cap for bootstrap

        all_papers = []

        # Collector 1: PubMed
        console.print("\n[1/3] PubMed...")
        try:
            collector = PubMedCollector()
            for q in tqdm(queries[:5], desc="PubMed"):
                papers = collector.search(q, max_results=50)
                all_papers.extend(papers)
            console.print(f"  PubMed: {len(all_papers)} papers so far")
        except Exception as e:
            console.print(f"[yellow]  PubMed failed: {e}[/yellow]")

        prev_count = len(all_papers)

        # Collector 2: Semantic Scholar
        console.print("\n[2/3] Semantic Scholar...")
        try:
            collector2 = SemanticScholarCollector()
            for q in tqdm(queries[:5], desc="SemanticScholar"):
                papers = collector2.search(q, max_results=50)
                all_papers.extend(papers)
            console.print(f"  Semantic Scholar: +{len(all_papers)-prev_count} papers")
        except Exception as e:
            console.print(f"[yellow]  Semantic Scholar failed: {e}[/yellow]")

        prev_count = len(all_papers)

        # Collector 3: PMC Open Access
        console.print("\n[3/3] PMC Open Access...")
        try:
            collector3 = PMCOpenAccessCollector()
            for q in tqdm(queries[:5], desc="PMC OA"):
                papers = collector3.search(q, max_results=50)
                all_papers.extend(papers)
            console.print(f"  PMC OA: +{len(all_papers)-prev_count} papers")
        except Exception as e:
            console.print(f"[yellow]  PMC OA failed: {e}[/yellow]")

        # Save combined
        combined_path = raw_dir / "combined.jsonl"
        with open(combined_path, "w", encoding="utf-8") as f:
            for p in all_papers:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

        n = len(all_papers)
        console.print(f"\n[green]Saved {n} papers → {combined_path}[/green]")
        if n < 800:
            console.print(f"[yellow]Note: target is 800 papers, collected {n}. "
                          f"Consider adding more queries or sources.[/yellow]")

        _save_checkpoint(1, {"papers_collected": n})
        return n

    except ImportError as e:
        console.print(f"[red]Collection failed (import error): {e}[/red]")
        console.print("Ensure collectors are properly installed.")
        return 0


# ── Phase 2: Quality filter ────────────────────────────────────────────────────


def phase_quality_filter(force: bool = False) -> int:
    """Remove low-quality papers: short abstracts, DOI duplicates, missing evidence_level."""
    if _is_completed(2, force):
        filtered_path = Path("data/raw/combined_filtered.jsonl")
        if filtered_path.exists():
            with open(filtered_path) as f:
                count = sum(1 for l in f if l.strip())
            console.print(f"[dim]Phase 2 already completed ({count} papers after filter). Skipping.[/dim]")
            return count
        console.print("[dim]Phase 2 marked complete but filtered file missing — re-running[/dim]")

    console.rule("[bold cyan]Phase 2 — Quality Filter[/bold cyan]")

    input_path = Path("data/raw/combined.jsonl")
    if not input_path.exists():
        console.print("[red]combined.jsonl not found — run Phase 1 first[/red]")
        return 0

    from tqdm import tqdm

    papers: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))

    original_count = len(papers)
    console.print(f"  Input: {original_count} papers")

    # Filter 1: Short abstracts (<100 chars)
    papers = [p for p in papers if len(p.get("abstract", "")) >= 100]
    after_abstract = len(papers)
    console.print(f"  After abstract filter (≥100 chars): {after_abstract} "
                  f"(removed {original_count - after_abstract})")

    # Filter 2: DOI duplicates
    seen_dois: set[str] = set()
    deduped = []
    for p in papers:
        doi = p.get("doi", "").strip()
        if doi and doi in seen_dois:
            continue
        if doi:
            seen_dois.add(doi)
        deduped.append(p)
    after_dedup = len(deduped)
    console.print(f"  After DOI dedup: {after_dedup} "
                  f"(removed {after_abstract - after_dedup})")

    # Filter 3: Missing evidence_level (add default "C" if missing rather than remove)
    for p in deduped:
        if not p.get("evidence_level"):
            p["evidence_level"] = "C"

    # Save filtered
    filtered_path = Path("data/raw/combined_filtered.jsonl")
    with open(filtered_path, "w", encoding="utf-8") as f:
        for p in deduped:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Also update combined.jsonl in-place
    import shutil
    shutil.copy(filtered_path, Path("data/raw/combined.jsonl"))

    n = len(deduped)
    console.print(f"\n[green]Saved {n} filtered papers → {filtered_path}[/green]")
    _save_checkpoint(2, {"papers_after_filter": n})
    return n


# ── Phase 3: Indexing ──────────────────────────────────────────────────────────


def phase_indexing(force: bool = False) -> bool:
    """Embed + index papers into ChromaDB + BM25."""
    if _is_completed(3, force):
        console.print("[dim]Phase 3 already completed. Skipping. Use --force to re-index.[/dim]")
        return True

    console.rule("[bold cyan]Phase 3 — Indexing[/bold cyan]")

    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/run_indexing.py", "--yes"],
        cwd=str(Path(__file__).parent.parent),
        capture_output=False,
    )
    success = result.returncode == 0
    if success:
        _save_checkpoint(3, {"status": "ok"})
        console.print("[green]Indexing complete.[/green]")
    else:
        console.print("[red]Indexing failed. Check output above.[/red]")
    return success


# ── Phase 4: RAGAS baseline ────────────────────────────────────────────────────


def phase_ragas_baseline(force: bool = False) -> dict | None:
    """Run retrieval-only eval on first 10 cases, save eval_baseline.json."""
    if _is_completed(4, force):
        baseline_path = Path("data/processed/eval_baseline.json")
        if baseline_path.exists():
            console.print("[dim]Phase 4 already completed (eval_baseline.json exists). Skipping.[/dim]")
            return json.loads(baseline_path.read_text())
        console.print("[dim]Phase 4 marked complete but eval_baseline.json missing — re-running[/dim]")

    console.rule("[bold cyan]Phase 4 — RAGAS Baseline[/bold cyan]")

    import subprocess
    baseline_output = "data/processed/eval_baseline.json"
    result = subprocess.run(
        [
            sys.executable, "scripts/run_eval.py",
            "--retrieval-only",
            "--cases", "10",
            "--output", baseline_output,
        ],
        cwd=str(Path(__file__).parent.parent),
        capture_output=False,
    )
    if result.returncode == 0:
        _save_checkpoint(4, {"output": baseline_output})
        console.print(f"[green]Baseline saved → {baseline_output}[/green]")
        try:
            return json.loads(Path(baseline_output).read_text())
        except Exception:
            return {}
    else:
        console.print("[red]RAGAS baseline failed. Knowledge base may be empty.[/red]")
        return None


# ── Phase 5: Summary ──────────────────────────────────────────────────────────


def phase_summary(paper_count: int, filtered_count: int, baseline: dict | None) -> None:
    """Print rich summary table."""
    console.rule("[bold cyan]Phase 5 — Summary[/bold cyan]")

    table = Table(title="Bootstrap Summary", show_header=True)
    table.add_column("Phase", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    checkpoints = _load_checkpoints()

    phases_info = [
        ("0 — Pre-flight", "0", "Keys + dirs verified"),
        ("1 — Collection", "1", f"{paper_count} papers collected"),
        ("2 — Quality Filter", "2", f"{filtered_count} papers after filter"),
        ("3 — Indexing", "3", "ChromaDB + BM25 indexed"),
        ("4 — RAGAS Baseline", "4", "eval_baseline.json saved"),
    ]

    for name, key, detail in phases_info:
        status = "[green]Done[/green]" if key in checkpoints else "[yellow]Skipped[/yellow]"
        table.add_row(name, status, detail)

    console.print(table)

    if baseline and "results" in baseline:
        results = baseline["results"]
        if results:
            avg_recall = sum(r.get("keyword_recall", 0) for r in results) / len(results)
            avg_ndcg = sum(r.get("ndcg_at_k", 0) for r in results) / len(results)
            console.print(f"\n[bold]Baseline Retrieval Metrics ({len(results)} cases):[/bold]")
            console.print(f"  Avg Keyword Recall: {avg_recall:.3f}")
            console.print(f"  Avg nDCG@K:         {avg_ndcg:.3f}")

    console.print("\n[bold green]Bootstrap complete![/bold green]")
    console.print("Next step: [bold]streamlit run app.py[/bold]")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    console.rule("[bold]Skincare AI — Knowledge Base Bootstrap[/bold]")

    force_phase = args.phase  # None = all phases, or specific phase number

    def should_force(phase: int) -> bool:
        if force_phase is not None:
            return args.force and phase == force_phase
        return args.force

    # Phase 0: Pre-flight
    ok = phase_preflight()
    if not ok:
        console.print("\n[red]Pre-flight failed. Please fix the issues above and retry.[/red]")
        sys.exit(1)
    _save_checkpoint(0)

    # Phase 1: Collection
    paper_count = phase_collection(force=should_force(1))

    # Phase 2: Quality filter
    filtered_count = phase_quality_filter(force=should_force(2))

    # Phase 3: Indexing
    if filtered_count > 0:
        phase_indexing(force=should_force(3))
    else:
        console.print("[yellow]Skipping indexing — no papers available.[/yellow]")

    # Phase 4: RAGAS baseline
    baseline = phase_ragas_baseline(force=should_force(4))

    # Phase 5: Summary
    phase_summary(paper_count, filtered_count, baseline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap the skincare knowledge base end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/bootstrap_kb.py                    # full run\n"
            "  python scripts/bootstrap_kb.py --force            # re-run all phases\n"
            "  python scripts/bootstrap_kb.py --force --phase 3  # re-run only indexing\n"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of completed phases",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="When used with --force, re-run only this specific phase",
    )
    args = parser.parse_args()
    main(args)
