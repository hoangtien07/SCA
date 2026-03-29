"""
scripts/eval_vision_accuracy.py

Benchmark GPT-4o Vision against SCIN dataset ground truth.

Metrics:
  - Per-condition accuracy (condition_name → precision/recall)
  - Per-Fitzpatrick accuracy (skin tone bias analysis)
  - Fairness gap metric (max accuracy - min accuracy across Fitzpatrick types)

Output:
    data/processed/vision_accuracy_eval.json

Usage:
    python scripts/eval_vision_accuracy.py
    python scripts/eval_vision_accuracy.py --sample-size 50 --dataset scin
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


def load_scin_samples(dataset_dir: Path, sample_size: int) -> list[dict]:
    """
    Load sample images from SCIN dataset directory.
    Returns list of {image_path, condition, fitzpatrick} dicts.
    """
    samples = []
    metadata_path = dataset_dir / "metadata.json"

    if metadata_path.exists():
        try:
            data = json.loads(metadata_path.read_text())
            for item in data.get("samples", [])[:sample_size]:
                img_path = dataset_dir / item.get("filename", "")
                if img_path.exists():
                    samples.append({
                        "image_path": str(img_path),
                        "condition": item.get("condition", "unknown"),
                        "fitzpatrick": item.get("fitzpatrick", "unknown"),
                    })
        except Exception as e:
            logger.warning(f"Could not load SCIN metadata: {e}")

    if not samples:
        # Fallback: scan directory for images
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = [f for f in dataset_dir.rglob("*") if f.suffix.lower() in image_exts]
        for f in image_files[:sample_size]:
            samples.append({
                "image_path": str(f),
                "condition": f.parent.name,  # dirname as condition label
                "fitzpatrick": "unknown",
            })

    return samples[:sample_size]


def eval_single_image(
    image_path: str,
    ground_truth: dict,
    analyzer,
) -> dict:
    """Run vision analysis on a single image and compare to ground truth."""
    import base64

    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        b64 = base64.b64encode(img_bytes).decode()
        ext = Path(image_path).suffix.lower().replace(".", "")
        media_type = f"image/{ext}" if ext in ("jpg", "jpeg", "png", "webp") else "image/jpeg"

        analysis = analyzer.analyze_bytes(img_bytes, media_type=media_type)

        # Check if predicted condition matches ground truth
        detected = [c.lower() for c in analysis.detected_conditions]
        gt_condition = ground_truth.get("condition", "").lower()
        condition_correct = any(gt_condition in d or d in gt_condition for d in detected)

        # Fitzpatrick prediction comparison
        gt_fitzpatrick = ground_truth.get("fitzpatrick", "unknown")
        pred_fitzpatrick = analysis.fitzpatrick_estimate.split()[0] if analysis.fitzpatrick_estimate else "unknown"
        fitzpatrick_correct = gt_fitzpatrick.upper() == pred_fitzpatrick.upper()

        return {
            "image_path": image_path,
            "ground_truth_condition": gt_condition,
            "ground_truth_fitzpatrick": gt_fitzpatrick,
            "predicted_conditions": detected,
            "predicted_fitzpatrick": pred_fitzpatrick,
            "condition_correct": condition_correct,
            "fitzpatrick_correct": fitzpatrick_correct,
            "acne_severity": analysis.acne_severity,
            "confidence_note": analysis.confidence_note,
        }
    except Exception as e:
        logger.warning(f"Failed to evaluate {image_path}: {e}")
        return {
            "image_path": image_path,
            "error": str(e),
            "condition_correct": False,
            "fitzpatrick_correct": False,
            "ground_truth_condition": ground_truth.get("condition", ""),
            "ground_truth_fitzpatrick": ground_truth.get("fitzpatrick", ""),
        }


def compute_fairness_gap(results: list[dict]) -> dict:
    """
    Compute per-Fitzpatrick accuracy and fairness gap.
    Fairness gap = max_accuracy - min_accuracy across Fitzpatrick types.
    """
    by_fitzpatrick: dict[str, list[bool]] = {}
    for r in results:
        fitz = r.get("ground_truth_fitzpatrick", "unknown")
        correct = r.get("condition_correct", False)
        if fitz not in by_fitzpatrick:
            by_fitzpatrick[fitz] = []
        by_fitzpatrick[fitz].append(correct)

    per_type = {}
    for fitz, correctness in by_fitzpatrick.items():
        if fitz == "unknown":
            continue
        per_type[fitz] = {
            "n": len(correctness),
            "accuracy": sum(correctness) / len(correctness) if correctness else 0.0,
        }

    accuracies = [v["accuracy"] for v in per_type.values() if v["n"] >= 3]
    fairness_gap = max(accuracies) - min(accuracies) if len(accuracies) >= 2 else None

    return {
        "per_fitzpatrick": per_type,
        "fairness_gap": fairness_gap,
        "fairness_gap_note": (
            "Fairness gap = max accuracy - min accuracy across Fitzpatrick types. "
            "Target: < 0.10 (10 percentage points)."
        ),
    }


def main(args: argparse.Namespace) -> None:
    console.rule("[bold]Vision Accuracy Evaluation[/bold]")

    from config.settings import settings

    if not settings.openai_api_key:
        console.print("[red]OPENAI_API_KEY not set[/red]")
        sys.exit(1)

    # Load dataset
    dataset_dir = Path(settings.cv_datasets_dir) / args.dataset
    if not dataset_dir.exists():
        console.print(f"[red]Dataset not found: {dataset_dir}[/red]")
        console.print(f"Run: python scripts/download_cv_datasets.py --datasets {args.dataset}")
        sys.exit(1)

    samples = load_scin_samples(dataset_dir, args.sample_size)
    if not samples:
        console.print(f"[red]No images found in {dataset_dir}[/red]")
        sys.exit(1)

    console.print(f"Evaluating {len(samples)} images from {args.dataset}...")

    from src.agents.vision_analyzer import VisionAnalyzer
    analyzer = VisionAnalyzer(api_key=settings.openai_api_key, model=settings.vision_model)

    results = []
    from tqdm import tqdm

    for sample in tqdm(samples, desc="Evaluating"):
        result = eval_single_image(
            image_path=sample["image_path"],
            ground_truth=sample,
            analyzer=analyzer,
        )
        results.append(result)

    # Compute metrics
    total = len(results)
    condition_correct = sum(1 for r in results if r.get("condition_correct", False))
    fitzpatrick_correct = sum(1 for r in results if r.get("fitzpatrick_correct", False))
    errors = sum(1 for r in results if "error" in r)

    condition_accuracy = condition_correct / total if total > 0 else 0.0
    fitzpatrick_accuracy = fitzpatrick_correct / total if total > 0 else 0.0

    fairness = compute_fairness_gap(results)

    # Per-condition breakdown
    by_condition: dict[str, list[bool]] = {}
    for r in results:
        cond = r.get("ground_truth_condition", "unknown")
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(r.get("condition_correct", False))

    per_condition = {
        cond: {"n": len(vals), "accuracy": sum(vals) / len(vals) if vals else 0.0}
        for cond, vals in by_condition.items()
    }

    # Display results
    console.print(f"\n[bold]Overall Results ({total} images):[/bold]")
    console.print(f"  Condition Accuracy:   {condition_accuracy:.1%}")
    console.print(f"  Fitzpatrick Accuracy: {fitzpatrick_accuracy:.1%}")
    console.print(f"  Errors:               {errors}")

    if fairness["fairness_gap"] is not None:
        gap = fairness["fairness_gap"]
        color = "green" if gap < 0.1 else "yellow" if gap < 0.2 else "red"
        console.print(f"  Fairness Gap:         [{color}]{gap:.1%}[/{color}] (target: <10%)")

    table = Table(title="Per-Condition Accuracy")
    table.add_column("Condition", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Accuracy", justify="right")

    for cond, stats in sorted(per_condition.items(), key=lambda x: -x[1]["accuracy"]):
        color = "green" if stats["accuracy"] >= 0.7 else "yellow" if stats["accuracy"] >= 0.5 else "red"
        table.add_row(
            cond,
            str(stats["n"]),
            f"[{color}]{stats['accuracy']:.1%}[/{color}]",
        )
    console.print(table)

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.dataset,
        "sample_size": total,
        "overall": {
            "condition_accuracy": round(condition_accuracy, 4),
            "fitzpatrick_accuracy": round(fitzpatrick_accuracy, 4),
            "errors": errors,
        },
        "per_condition": per_condition,
        "fairness": fairness,
        "raw_results": results,
    }

    output_path = Path("data/processed/vision_accuracy_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    console.print(f"\n[green]Results saved → {output_path}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GPT-4o Vision accuracy")
    parser.add_argument(
        "--dataset",
        choices=["scin", "fitzpatrick17k", "ddi", "dermamnist"],
        default="scin",
        help="Dataset to evaluate against",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of images to evaluate (default: 100)",
    )
    args = parser.parse_args()
    main(args)
