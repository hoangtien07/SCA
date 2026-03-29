"""
scripts/download_cosmetic_data.py

CLI for downloading cosmetic product and ingredient data.

Sources:
  - openbeautyfacts: Open Beauty Facts Parquet dump (products + ingredients)
  - cosing: EU CosIng database (requires ingredient name list)

Usage:
    python scripts/download_cosmetic_data.py --dry-run
    python scripts/download_cosmetic_data.py --source openbeautyfacts
    python scripts/download_cosmetic_data.py --source cosing --dest data/cosmetic_raw
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
    console.rule("[bold]Cosmetic Data Downloader[/bold]")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    sources = args.source if isinstance(args.source, list) else [args.source]

    if "openbeautyfacts" in sources or "all" in sources:
        console.print("\n[cyan]Downloading Open Beauty Facts...[/cyan]")
        try:
            from src.collectors.cosmetic_api_collector import OpenBeautyFactsCollector
            collector = OpenBeautyFactsCollector(dest_dir=str(dest))
            if args.dry_run:
                collector.download_parquet_dump(dry_run=True)
                console.print("  DRY RUN complete")
            else:
                products = collector.download_parquet_dump()
                console.print(f"  [green]Downloaded {len(products)} products[/green]")
                # Save as JSONL for pipeline
                jsonl_path = dest / "cosmetic_products.jsonl"
                import json
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for p in products:
                        f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
                console.print(f"  Saved → {jsonl_path}")
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")

    if "cosing" in sources or "all" in sources:
        console.print("\n[cyan]Fetching CosIng ingredient data...[/cyan]")
        try:
            from src.collectors.cosmetic_api_collector import CosIngCollector
            import yaml

            # Load ingredient list from taxonomy
            with open("config/skin_conditions.yaml", encoding="utf-8") as f:
                taxonomy = yaml.safe_load(f)

            # Extract all ingredient names from taxonomy
            ingredient_names = []
            for cat, items in taxonomy.get("active_ingredients", {}).items():
                if isinstance(items, list):
                    ingredient_names.extend(items)
                elif isinstance(items, dict):
                    for subcat, subitems in items.items():
                        if isinstance(subitems, list):
                            ingredient_names.extend(subitems)

            ingredient_names = [n.replace("_", " ").upper() for n in ingredient_names[:50]]
            console.print(f"  Fetching {len(ingredient_names)} ingredients from CosIng...")

            if args.dry_run:
                console.print(f"  DRY RUN — would fetch {len(ingredient_names)} CosIng records")
            else:
                collector = CosIngCollector()
                records = collector.batch_fetch(ingredient_names)

                import json
                cosing_path = dest / "cosing_ingredients.jsonl"
                with open(cosing_path, "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
                console.print(f"  [green]Saved {len(records)} CosIng records → {cosing_path}[/green]")

        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")

    # Summary table
    table = Table(title="Download Summary")
    table.add_column("Source", style="cyan")
    table.add_column("Status")
    table.add_column("Path")

    for source in ["openbeautyfacts", "cosing"]:
        if source not in sources and "all" not in sources:
            continue
        if source == "openbeautyfacts":
            path = dest / "cosmetic_products.jsonl"
        else:
            path = dest / "cosing_ingredients.jsonl"

        if args.dry_run:
            status = "[yellow]dry-run[/yellow]"
        elif path.exists():
            status = "[green]OK[/green]"
        else:
            status = "[red]not found[/red]"

        table.add_row(source, status, str(path))

    console.print(table)

    if not args.dry_run:
        console.print("\nNext: python scripts/seed_graph.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download cosmetic product and ingredient data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_cosmetic_data.py --dry-run\n"
            "  python scripts/download_cosmetic_data.py --source openbeautyfacts\n"
            "  python scripts/download_cosmetic_data.py --source all\n"
        ),
    )
    parser.add_argument(
        "--source",
        nargs="+",
        default=["all"],
        choices=["openbeautyfacts", "cosing", "all"],
        metavar="SOURCE",
        help="Data source: openbeautyfacts, cosing, all",
    )
    parser.add_argument(
        "--dest",
        default="./data/cosmetic_raw",
        help="Destination directory (default: ./data/cosmetic_raw)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check availability without downloading",
    )
    args = parser.parse_args()
    main(args)
