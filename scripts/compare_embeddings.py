"""
scripts/compare_embeddings.py

A/B comparison of embedding models on first 10 eval test cases.
Compares PubMedBERT (local) vs text-embedding-3-small (OpenAI) side by side
using RAGAS-style retrieval metrics.

Output:
    data/processed/embedding_comparison.json

Run:
    python scripts/compare_embeddings.py
    python scripts/compare_embeddings.py --cases 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


def run_retrieval_with_provider(
    cases: list[dict],
    embedding_provider: str,
    settings,
    top_k: int = 8,
) -> list[dict]:
    """Run retrieval for all cases using specified embedding provider."""
    from src.pipeline.embedder import get_embedder
    from src.pipeline.indexer import ChromaIndexer
    from src.pipeline.bm25_index import BM25Index
    from src.agents.rag_retriever import RAGRetriever
    import math

    # Create embedder for this provider
    class _SettingsProxy:
        def __init__(self, base, provider):
            self._base = base
            self.embedding_provider = provider
            self.openai_api_key = base.openai_api_key
            self.anthropic_api_key = base.anthropic_api_key
            self.embedding_model = base.embedding_model
            self.embedding_device = getattr(base, "embedding_device", "cpu")
            self.embedding_batch_size = getattr(base, "embedding_batch_size", 32)

    proxy = _SettingsProxy(settings, embedding_provider)
    embedder = get_embedder(proxy)

    # Create indexer with this embedder
    try:
        indexer = ChromaIndexer(
            persist_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
            embedder=embedder,
        )
    except ValueError as e:
        logger.warning(f"[compare] ChromaDB model mismatch for {embedding_provider}: {e}")
        return []

    bm25_path = Path(settings.chroma_persist_dir) / "bm25_index.json"
    bm25 = BM25Index.load(str(bm25_path)) if bm25_path.exists() else None

    retriever = RAGRetriever(indexer=indexer, bm25=bm25, top_k=top_k)

    results = []
    for case in cases:
        query = case["query"]
        expected_kw = [kw.lower() for kw in case.get("expected_keywords", [])]

        start = time.time()
        try:
            retrieved = retriever.retrieve(query, top_k=top_k)
            latency = time.time() - start
        except Exception as e:
            logger.warning(f"Retrieval failed for case {case['id']}: {e}")
            continue

        retrieved_text = " ".join(r.text.lower() for r in retrieved)
        scores = [r.score for r in retrieved]

        hits = sum(1 for kw in expected_kw if kw in retrieved_text)
        keyword_recall = hits / len(expected_kw) if expected_kw else 1.0

        score_threshold = 0.3
        relevant_count = sum(1 for s in scores if s > score_threshold)
        precision_at_k = relevant_count / len(scores) if scores else 0.0

        # nDCG
        def ndcg(sc):
            if not sc:
                return 0.0
            dcg = sum(s / math.log2(i + 2) for i, s in enumerate(sc))
            ideal = sorted(sc, reverse=True)
            idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
            return dcg / idcg if idcg > 0 else 0.0

        results.append({
            "case_id": case["id"],
            "case_name": case["name"],
            "provider": embedding_provider,
            "keyword_recall": round(keyword_recall, 4),
            "precision_at_k": round(precision_at_k, 4),
            "ndcg_at_k": round(ndcg(scores), 4),
            "top_score": round(max(scores) if scores else 0.0, 4),
            "latency_s": round(latency, 3),
        })

    return results


def main(args: argparse.Namespace) -> None:
    console.rule("[bold]Embedding Model Comparison[/bold]")

    import yaml
    from config.settings import settings

    if not settings.openai_api_key:
        console.print("[red]OPENAI_API_KEY not set — cannot run OpenAI embedder comparison[/red]")
        sys.exit(1)

    # Load test cases
    cases_path = Path("config/eval_test_cases.yaml")
    if not cases_path.exists():
        console.print("[red]config/eval_test_cases.yaml not found[/red]")
        sys.exit(1)

    with open(cases_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cases = data.get("test_cases", [])[:args.cases]
    console.print(f"Running comparison on {len(cases)} test cases...\n")

    # Run both providers
    all_results = []

    for provider in ["local", "openai"]:
        console.print(f"[cyan]Testing provider: {provider}...[/cyan]")
        try:
            results = run_retrieval_with_provider(cases, provider, settings)
            all_results.extend(results)
            console.print(f"  Done: {len(results)} results")
        except Exception as e:
            console.print(f"  [yellow]Failed: {e}[/yellow]")

    if not all_results:
        console.print("[red]No results — ensure knowledge base is indexed first[/red]")
        sys.exit(1)

    # Build comparison table
    table = Table(title="Embedding Comparison", show_header=True)
    table.add_column("Case", style="cyan", max_width=25)
    table.add_column("Local KW", justify="right")
    table.add_column("OAI KW", justify="right")
    table.add_column("Local nDCG", justify="right")
    table.add_column("OAI nDCG", justify="right")
    table.add_column("Local P@K", justify="right")
    table.add_column("OAI P@K", justify="right")

    # Organize by case_id
    by_case: dict[int, dict] = {}
    for r in all_results:
        cid = r["case_id"]
        if cid not in by_case:
            by_case[cid] = {}
        by_case[cid][r["provider"]] = r

    for cid, providers in sorted(by_case.items()):
        local = providers.get("local", {})
        oai = providers.get("openai", {})
        name = local.get("case_name", oai.get("case_name", str(cid)))

        def fmt(v, winner_v, losser_v):
            if v is None:
                return "-"
            s = f"{v:.3f}"
            if v is not None and winner_v is not None and v >= winner_v:
                return f"[green]{s}[/green]"
            return s

        lkw = local.get("keyword_recall")
        okw = oai.get("keyword_recall")
        lnd = local.get("ndcg_at_k")
        ond = oai.get("ndcg_at_k")
        lpk = local.get("precision_at_k")
        opk = oai.get("precision_at_k")

        table.add_row(
            name[:25],
            fmt(lkw, lkw if (lkw or 0) >= (okw or 0) else None, None),
            fmt(okw, okw if (okw or 0) >= (lkw or 0) else None, None),
            f"{lnd:.3f}" if lnd is not None else "-",
            f"{ond:.3f}" if ond is not None else "-",
            f"{lpk:.3f}" if lpk is not None else "-",
            f"{opk:.3f}" if opk is not None else "-",
        )

    console.print(table)

    # Averages
    def avg(lst, key):
        vals = [r[key] for r in lst if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    local_results = [r for r in all_results if r["provider"] == "local"]
    oai_results = [r for r in all_results if r["provider"] == "openai"]

    console.print("\n[bold]Averages:[/bold]")
    console.print(f"  {'Metric':<20} {'Local':<12} {'OpenAI':<12}")
    for metric in ["keyword_recall", "ndcg_at_k", "precision_at_k"]:
        lv = avg(local_results, metric)
        ov = avg(oai_results, metric)
        console.print(f"  {metric:<20} {lv:<12.4f} {ov:<12.4f}")

    # Save results
    output_path = Path("data/processed/embedding_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_cases": len(cases),
        "results": all_results,
        "averages": {
            "local": {m: avg(local_results, m) for m in ["keyword_recall", "ndcg_at_k", "precision_at_k"]},
            "openai": {m: avg(oai_results, m) for m in ["keyword_recall", "ndcg_at_k", "precision_at_k"]},
        },
    }
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    console.print(f"\n[green]Results saved → {output_path}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare embedding models on eval cases")
    parser.add_argument("--cases", type=int, default=10, help="Number of test cases (default: 10)")
    args = parser.parse_args()
    main(args)
