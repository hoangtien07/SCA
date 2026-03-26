"""
scripts/run_eval.py

Phase 3 — Evaluate RAG pipeline quality using RAGAS metrics + retrieval-only mode.

Usage:
    python scripts/run_eval.py                          # full eval (retrieval + generation)
    python scripts/run_eval.py --retrieval-only          # fast: retrieval metrics only (no LLM)
    python scripts/run_eval.py --cases 5                 # run first 5 cases only
    python scripts/run_eval.py --ids 1,3,11              # run specific case IDs
    python scripts/run_eval.py --difficulty hard          # run only hard cases

Prerequisites:
    - Knowledge base must be populated (run run_collection.py + run_indexing.py first)
    - OPENAI_API_KEY must be set (for embeddings)
    - ANTHROPIC_API_KEY must be set (for generation, unless --retrieval-only)

Output:
    data/processed/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

console = Console()


# ── Load test cases ──────────────────────────────────────────────────────────


def load_test_cases(
    path: str = "config/eval_test_cases.yaml",
    case_ids: list[int] | None = None,
    max_cases: int | None = None,
    difficulty: str | None = None,
) -> list[dict]:
    """Load and filter test cases from YAML."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cases = data.get("test_cases", [])

    if difficulty:
        cases = [c for c in cases if c.get("difficulty") == difficulty]

    if case_ids:
        cases = [c for c in cases if c.get("id") in case_ids]

    if max_cases:
        cases = cases[:max_cases]

    return cases


# ── Retrieval evaluation ─────────────────────────────────────────────────────


def evaluate_retrieval(
    cases: list[dict],
    retriever: Any,
    top_k: int = 8,
) -> list[dict]:
    """
    Run retrieval for each case and compute keyword-based metrics.
    Returns list of per-case result dicts.
    """
    results = []

    for case in cases:
        query = case["query"]
        expected_kw = [kw.lower() for kw in case.get("expected_keywords", [])]

        start = time.time()
        retrieved = retriever.retrieve(query, top_k=top_k)
        latency = time.time() - start

        # Combine all retrieved text
        retrieved_text = " ".join(r.text.lower() for r in retrieved)
        scores = [r.score for r in retrieved]

        # Keyword hit rate (proxy for recall)
        hits = sum(1 for kw in expected_kw if kw in retrieved_text)
        keyword_recall = hits / len(expected_kw) if expected_kw else 1.0

        # Precision@K: fraction of results with score > threshold
        score_threshold = 0.3
        relevant_count = sum(1 for s in scores if s > score_threshold)
        precision_at_k = relevant_count / len(scores) if scores else 0.0

        # nDCG@K (simplified: use score as relevance)
        ndcg = _ndcg(scores)

        result = {
            "case_id": case["id"],
            "case_name": case["name"],
            "difficulty": case.get("difficulty", ""),
            "query": query,
            "n_retrieved": len(retrieved),
            "top_score": max(scores) if scores else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "keyword_recall": keyword_recall,
            "precision_at_k": precision_at_k,
            "ndcg_at_k": ndcg,
            "latency_s": round(latency, 3),
            "retrieved_titles": [r.title[:80] for r in retrieved[:3]],
        }
        results.append(result)

    return results


def _ndcg(scores: list[float], k: int | None = None) -> float:
    """Compute nDCG from a list of relevance scores (higher = better)."""
    import math

    if not scores:
        return 0.0
    if k:
        scores = scores[:k]

    dcg = sum(s / math.log2(i + 2) for i, s in enumerate(scores))
    ideal = sorted(scores, reverse=True)
    idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ── Full pipeline evaluation (retrieval + generation + safety) ───────────────


def evaluate_full(
    cases: list[dict],
    retriever: Any,
    generator: Any,
    safety: Any,
    top_k: int = 8,
) -> list[dict]:
    """
    Run full pipeline: retrieve → generate → safety check.
    Computes retrieval metrics + generation quality signals.
    """
    results = []

    for case in cases:
        query = case["query"]
        profile_data = case.get("profile", {})

        # ── Retrieval ────────────────────────────────────────────────────
        start = time.time()
        retrieved = retriever.retrieve(query, top_k=top_k)
        retrieval_time = time.time() - start

        retrieved_text = " ".join(r.text.lower() for r in retrieved)
        scores = [r.score for r in retrieved]
        expected_kw = [kw.lower() for kw in case.get("expected_keywords", [])]
        kw_hits = sum(1 for kw in expected_kw if kw in retrieved_text)
        keyword_recall = kw_hits / len(expected_kw) if expected_kw else 1.0

        # ── Generation ───────────────────────────────────────────────────
        gen_start = time.time()
        try:
            regimen = generator.generate(
                profile=profile_data,
                evidence=retrieved,
            )
            gen_time = time.time() - gen_start
            gen_success = True
        except Exception as e:
            logger.warning(f"Generation failed for case {case['id']}: {e}")
            gen_time = time.time() - gen_start
            gen_success = False
            regimen = None

        # ── Ingredient check ─────────────────────────────────────────────
        expected_ingredients = case.get("expected_ingredients", [])
        must_not = case.get("must_not_include", [])
        ingredient_hits = 0
        forbidden_present = 0
        all_steps_text = ""

        if regimen:
            all_steps = []
            for routine in [regimen.am_routine, regimen.pm_routine]:
                if routine:
                    all_steps.extend(routine)
            if hasattr(regimen, "weekly_treatments") and regimen.weekly_treatments:
                all_steps.extend(regimen.weekly_treatments)

            all_steps_text = " ".join(
                f"{s.product_name} {s.active_ingredient} {' '.join(s.alternatives or [])}"
                for s in all_steps
            ).lower()

            ingredient_hits = sum(
                1 for ing in expected_ingredients if ing.lower() in all_steps_text
            )
            forbidden_present = sum(
                1 for ing in must_not if ing.lower() in all_steps_text
            )

        ingredient_recall = (
            ingredient_hits / len(expected_ingredients)
            if expected_ingredients
            else 1.0
        )

        # ── Safety check ─────────────────────────────────────────────────
        safety_result = None
        if regimen:
            try:
                safety_result = safety.check(regimen, profile_data)
            except Exception as e:
                logger.warning(f"Safety check failed for case {case['id']}: {e}")

        must_flag = case.get("must_flag_warning", False)
        has_flags = bool(safety_result and safety_result.flags)
        safety_correct = (must_flag == has_flags) or (must_flag and has_flags)

        result = {
            "case_id": case["id"],
            "case_name": case["name"],
            "difficulty": case.get("difficulty", ""),
            # Retrieval metrics
            "keyword_recall": keyword_recall,
            "ndcg_at_k": _ndcg(scores),
            "top_score": max(scores) if scores else 0.0,
            "retrieval_latency_s": round(retrieval_time, 3),
            # Generation metrics
            "generation_success": gen_success,
            "generation_latency_s": round(gen_time, 3),
            "ingredient_recall": ingredient_recall,
            "forbidden_ingredients_found": forbidden_present,
            # Safety metrics
            "safety_correct": safety_correct,
            "expected_warning": must_flag,
            "got_warning": has_flags,
            "n_safety_flags": len(safety_result.flags) if safety_result else 0,
        }
        results.append(result)

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────


def print_retrieval_report(results: list[dict]) -> None:
    """Print a rich table summarizing retrieval-only evaluation."""
    table = Table(title="Retrieval Evaluation Results", show_header=True)
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Name", style="white", max_width=30)
    table.add_column("Diff", style="yellow")
    table.add_column("KW Recall", justify="right")
    table.add_column("P@K", justify="right")
    table.add_column("nDCG", justify="right")
    table.add_column("Top Score", justify="right")
    table.add_column("Latency", justify="right")

    for r in results:
        kw_style = "green" if r["keyword_recall"] >= 0.7 else "red"
        table.add_row(
            str(r["case_id"]),
            r["case_name"],
            r["difficulty"],
            f"[{kw_style}]{r['keyword_recall']:.2f}[/{kw_style}]",
            f"{r['precision_at_k']:.2f}",
            f"{r['ndcg_at_k']:.2f}",
            f"{r['top_score']:.3f}",
            f"{r['latency_s']:.2f}s",
        )

    console.print(table)

    # Aggregates
    avg_recall = sum(r["keyword_recall"] for r in results) / len(results)
    avg_ndcg = sum(r["ndcg_at_k"] for r in results) / len(results)
    avg_prec = sum(r["precision_at_k"] for r in results) / len(results)

    console.print(f"\n[bold]Averages:[/bold]")
    console.print(f"  Keyword Recall: {avg_recall:.3f}  (target: >= 0.70)")
    console.print(f"  Precision@K:    {avg_prec:.3f}  (target: >= 0.75)")
    console.print(f"  nDCG@K:         {avg_ndcg:.3f}  (target: >= 0.75)")

    # Per-difficulty breakdown
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            avg_r = sum(r["keyword_recall"] for r in subset) / len(subset)
            console.print(f"  [{diff}] Keyword Recall: {avg_r:.3f}  (n={len(subset)})")


def print_full_report(results: list[dict]) -> None:
    """Print a rich table summarizing full pipeline evaluation."""
    table = Table(title="Full Pipeline Evaluation Results", show_header=True)
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Name", style="white", max_width=25)
    table.add_column("Diff", style="yellow")
    table.add_column("KW Recall", justify="right")
    table.add_column("Ing Recall", justify="right")
    table.add_column("Forbidden", justify="right")
    table.add_column("Safety OK", justify="right")
    table.add_column("Gen", justify="right")

    for r in results:
        kw_style = "green" if r["keyword_recall"] >= 0.7 else "red"
        ing_style = "green" if r["ingredient_recall"] >= 0.5 else "red"
        safety_style = "green" if r["safety_correct"] else "red"
        forbidden_style = "green" if r["forbidden_ingredients_found"] == 0 else "red"

        table.add_row(
            str(r["case_id"]),
            r["case_name"],
            r["difficulty"],
            f"[{kw_style}]{r['keyword_recall']:.2f}[/{kw_style}]",
            f"[{ing_style}]{r['ingredient_recall']:.2f}[/{ing_style}]",
            f"[{forbidden_style}]{r['forbidden_ingredients_found']}[/{forbidden_style}]",
            f"[{safety_style}]{'OK' if r['safety_correct'] else 'FAIL'}[/{safety_style}]",
            f"{'OK' if r['generation_success'] else 'FAIL'}",
        )

    console.print(table)

    # Aggregates
    n = len(results)
    console.print(f"\n[bold]Aggregates ({n} cases):[/bold]")

    avg_kw = sum(r["keyword_recall"] for r in results) / n
    avg_ing = sum(r["ingredient_recall"] for r in results) / n
    safety_ok = sum(1 for r in results if r["safety_correct"]) / n
    gen_ok = sum(1 for r in results if r["generation_success"]) / n
    total_forbidden = sum(r["forbidden_ingredients_found"] for r in results)

    console.print(f"  Keyword Recall:      {avg_kw:.3f}")
    console.print(f"  Ingredient Recall:   {avg_ing:.3f}")
    console.print(f"  Safety Accuracy:     {safety_ok:.1%}")
    console.print(f"  Generation Success:  {gen_ok:.1%}")
    console.print(f"  Forbidden Violations: {total_forbidden}")


def save_results(results: list[dict], output_path: str) -> None:
    """Save evaluation results to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    console.print(f"\n[green]Results saved → {out}[/green]")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    console.rule("[bold]Phase 3 — RAG Evaluation[/bold]")

    # Parse case IDs if provided
    case_ids = None
    if args.ids:
        case_ids = [int(x.strip()) for x in args.ids.split(",")]

    cases = load_test_cases(
        case_ids=case_ids,
        max_cases=args.cases,
        difficulty=args.difficulty,
    )
    console.print(f"Loaded [green]{len(cases)}[/green] test cases")

    if not cases:
        console.print("[red]No test cases matched filters[/red]")
        sys.exit(1)

    # Check API keys
    if not settings.openai_api_key:
        console.print("[red]OPENAI_API_KEY not set in .env[/red]")
        sys.exit(1)

    # Initialize retriever
    from src.pipeline.indexer import ChromaIndexer
    from src.pipeline.bm25_index import BM25Index
    from src.agents.rag_retriever import RAGRetriever

    indexer = ChromaIndexer(
        persist_dir=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

    bm25_path = Path(settings.chroma_persist_dir) / "bm25_index.json"
    bm25 = BM25Index.load(str(bm25_path)) if bm25_path.exists() else None

    retriever = RAGRetriever(indexer=indexer, bm25=bm25, top_k=settings.retrieval_top_k)

    if args.retrieval_only:
        # ── Retrieval-only mode ──────────────────────────────────────────
        console.print("[cyan]Mode: retrieval-only (no LLM generation)[/cyan]\n")
        results = evaluate_retrieval(cases, retriever, top_k=settings.retrieval_top_k)
        print_retrieval_report(results)
    else:
        # ── Full pipeline mode ───────────────────────────────────────────
        if not settings.anthropic_api_key:
            console.print("[red]ANTHROPIC_API_KEY not set (required for generation)[/red]")
            console.print("Use --retrieval-only to skip generation.")
            sys.exit(1)

        from src.agents.regimen_generator import RegimenGenerator
        from src.agents.safety_guard import SafetyGuard

        console.print("[cyan]Mode: full pipeline (retrieval + generation + safety)[/cyan]\n")

        generator = RegimenGenerator(
            api_key=settings.anthropic_api_key,
            model=settings.reasoning_model,
        )
        safety = SafetyGuard()

        results = evaluate_full(
            cases, retriever, generator, safety, top_k=settings.retrieval_top_k
        )
        print_full_report(results)

    save_results(results, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_eval.py --retrieval-only         # fast retrieval check\n"
            "  python scripts/run_eval.py --cases 5                # first 5 cases\n"
            "  python scripts/run_eval.py --ids 3,14,28            # specific cases\n"
            "  python scripts/run_eval.py --difficulty hard         # hard cases only\n"
        ),
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run retrieval metrics only (no LLM generation — faster, cheaper)",
    )
    parser.add_argument("--cases", type=int, default=None, help="Max number of cases to run")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated case IDs to run")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter by difficulty level",
    )
    parser.add_argument(
        "--output",
        default="data/processed/eval_results.json",
        help="Output path for results JSON",
    )
    args = parser.parse_args()
    main(args)
