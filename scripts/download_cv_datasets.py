"""
scripts/download_cv_datasets.py

CLI for downloading computer vision skin datasets.

Usage:
    python scripts/download_cv_datasets.py --dry-run
    python scripts/download_cv_datasets.py --datasets scin fitzpatrick17k
    python scripts/download_cv_datasets.py --datasets all --dest data/cv_datasets
    python scripts/download_cv_datasets.py --datasets dermamnist
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


def main(args: argparse.Namespace) -> None:
    console.rule("[bold]CV Dataset Downloader[/bold]")

    from src.collectors.cv_dataset_collector import CVDatasetCollector, DatasetManifest

    collector = CVDatasetCollector(dest_dir=args.dest)

    # Determine which datasets to download
    available = ["scin", "fitzpatrick17k", "ddi", "dermamnist"]
    if "all" in args.datasets:
        datasets = available
    else:
        datasets = [d for d in args.datasets if d in available]
        unknown = [d for d in args.datasets if d not in available and d != "all"]
        if unknown:
            console.print(f"[yellow]Unknown datasets: {unknown}. Available: {available}[/yellow]")

    if not datasets:
        console.print("[red]No valid datasets specified[/red]")
        sys.exit(1)

    console.print(f"Datasets to download: {datasets}")
    console.print(f"Destination: {args.dest}")
    if args.dry_run:
        console.print("[yellow]DRY RUN — no files will be downloaded[/yellow]\n")

    downloaders = {
        "scin": collector.download_scin,
        "fitzpatrick17k": collector.download_fitzpatrick17k,
        "ddi": collector.download_ddi,
        "dermamnist": collector.download_dermamnist,
    }

    manifests = []
    for name in datasets:
        console.print(f"\n[cyan]Downloading: {name}[/cyan]")
        try:
            manifest = downloaders[name](dry_run=args.dry_run)
            manifests.append(manifest)
            console.print(f"  [green]OK[/green] {manifest.num_images} images")
        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/red]")

    # Summary table
    if manifests:
        console.print("\n")
        table = Table(title="Download Summary")
        table.add_column("Dataset", style="cyan")
        table.add_column("Images", justify="right")
        table.add_column("License")
        table.add_column("Conditions", max_width=40)
        table.add_column("Path")

        for m in manifests:
            table.add_row(
                m.name,
                str(m.num_images),
                m.license,
                ", ".join(m.conditions[:4]) + ("..." if len(m.conditions) > 4 else ""),
                str(m.local_path),
            )

        console.print(table)

    if not args.dry_run:
        console.print("\nNext: python scripts/eval_vision_accuracy.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download CV skin condition datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_cv_datasets.py --dry-run\n"
            "  python scripts/download_cv_datasets.py --datasets dermamnist\n"
            "  python scripts/download_cv_datasets.py --datasets all\n"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["scin", "fitzpatrick17k", "ddi", "dermamnist", "all"],
        metavar="DATASET",
        help="Dataset(s) to download: scin, fitzpatrick17k, ddi, dermamnist, all",
    )
    parser.add_argument(
        "--dest",
        default="./data/cv_datasets",
        help="Destination directory (default: ./data/cv_datasets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check availability and estimate sizes without downloading",
    )
    args = parser.parse_args()
    main(args)
