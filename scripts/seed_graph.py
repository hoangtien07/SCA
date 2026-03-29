"""
scripts/seed_graph.py

Generates Neo4j Cypher statements for the skincare knowledge graph.
Does NOT require a running Neo4j instance — outputs .cypher file for import.

Graph schema:
  Nodes:
    (:Ingredient {name, inci_name, category, otc_max_concentration})
    (:Condition  {name, skin_type, severity_levels})
    (:Product    {id, name, brand, categories})

  Relationships:
    (:Ingredient)-[:CONFLICTS_WITH {reason}]->(:Ingredient)
    (:Ingredient)-[:CONTRAINDICATED_IN {reason}]->(:Condition)
    (:Ingredient)-[:MAX_CONCENTRATION {otc_max, note}]->(:Ingredient)
    (:Ingredient)-[:TREATS_CONDITION {evidence_grade}]->(:Condition)
    (:Product)-[:CONTAINS {inci_name}]->(:Ingredient)

Data sources:
  - config/skin_conditions.yaml (interactions, concentrations, conditions)
  - OpenBeautyFacts Parquet dump (products + ingredients)

Output:
  data/graph/seed.cypher
  data/graph/seed_stats.json

Run:
    python scripts/seed_graph.py
    python scripts/seed_graph.py --dry-run       # show stats only
    python scripts/seed_graph.py --products 1000 # limit product nodes
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


def _escape_cypher(s: str) -> str:
    """Escape single quotes for Cypher strings."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _esc(s: str) -> str:
    return _escape_cypher(str(s))


class GraphSeeder:
    """Reads YAML + cosmetic data and generates Cypher statements."""

    def __init__(self, skin_conditions_path: str = "config/skin_conditions.yaml"):
        import yaml
        with open(skin_conditions_path, encoding="utf-8") as f:
            self.taxonomy = yaml.safe_load(f)

        self.cypher_lines: list[str] = []
        self.stats: dict = {
            "ingredient_nodes": 0,
            "condition_nodes": 0,
            "product_nodes": 0,
            "conflicts_with_edges": 0,
            "contraindicated_in_edges": 0,
            "treats_condition_edges": 0,
            "contains_edges": 0,
        }

    def generate_all(self, max_products: int = 500) -> None:
        """Generate all Cypher statements."""
        self.cypher_lines = [
            "// ── Skincare AI Knowledge Graph Seed ─────────────────────────────────",
            f"// Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
            "// Source: config/skin_conditions.yaml + OpenBeautyFacts",
            "",
            "// Clear existing data (comment out on production!)",
            "// MATCH (n) DETACH DELETE n;",
            "",
            "// ── Constraints ──────────────────────────────────────────────────────",
            "CREATE CONSTRAINT ingredient_name IF NOT EXISTS",
            "  FOR (i:Ingredient) REQUIRE i.name IS UNIQUE;",
            "CREATE CONSTRAINT condition_name IF NOT EXISTS",
            "  FOR (c:Condition) REQUIRE c.name IS UNIQUE;",
            "CREATE CONSTRAINT product_id IF NOT EXISTS",
            "  FOR (p:Product) REQUIRE p.id IS UNIQUE;",
            "",
        ]

        self._gen_condition_nodes()
        self._gen_ingredient_nodes()
        self._gen_conflict_edges()
        self._gen_pregnancy_contraindicated()
        self._gen_concentration_limits()
        self._gen_product_nodes(max_products)

    def _gen_condition_nodes(self) -> None:
        """Generate Condition nodes from skin_conditions.yaml."""
        self.cypher_lines.append("// ── Condition nodes ──────────────────────────────────────────────────")
        conditions = self.taxonomy.get("conditions", {})
        for name, data in conditions.items():
            subtypes = json.dumps(data.get("subtypes", []))
            keywords = json.dumps(data.get("keywords", []))
            self.cypher_lines.append(
                f"MERGE (:Condition {{name: '{_esc(name)}', "
                f"subtypes: {subtypes}, keywords: {keywords}}});"
            )
            self.stats["condition_nodes"] += 1
        self.cypher_lines.append("")

    def _gen_ingredient_nodes(self) -> None:
        """Generate Ingredient nodes from all active ingredients in taxonomy."""
        self.cypher_lines.append("// ── Ingredient nodes ─────────────────────────────────────────────────")

        ingredients_added: set[str] = set()
        actives = self.taxonomy.get("active_ingredients", {})
        concentration_limits = self.taxonomy.get("concentration_limits", {})

        def add_ingredient(name: str, category: str = "", otc_max: str = "") -> None:
            clean = name.replace("_", " ").lower()
            if clean in ingredients_added:
                return
            ingredients_added.add(clean)
            self.cypher_lines.append(
                f"MERGE (:Ingredient {{name: '{_esc(clean)}', "
                f"category: '{_esc(category)}', "
                f"otc_max_concentration: '{_esc(otc_max)}'}});"
            )
            self.stats["ingredient_nodes"] += 1

        for category, subcats in actives.items():
            if isinstance(subcats, list):
                for ing in subcats:
                    otc = concentration_limits.get(ing.lower(), {}).get("otc_max", "")
                    add_ingredient(ing, category=category, otc_max=otc)
            elif isinstance(subcats, dict):
                for subcat, items in subcats.items():
                    if isinstance(items, list):
                        for ing in items:
                            otc = concentration_limits.get(ing.lower(), {}).get("otc_max", "")
                            add_ingredient(ing, category=f"{category}/{subcat}", otc_max=otc)

        # Add INCI synonyms as ingredient nodes
        for canonical, synonyms in self.taxonomy.get("inci_synonyms", {}).items():
            add_ingredient(canonical)
            for syn in synonyms:
                add_ingredient(syn, category="synonym")

        self.cypher_lines.append("")

    def _gen_conflict_edges(self) -> None:
        """Generate CONFLICTS_WITH edges from known_interactions."""
        self.cypher_lines.append("// ── Ingredient conflicts (CONFLICTS_WITH) ────────────────────────────")
        interactions = self.taxonomy.get("known_interactions", {})
        for pair in interactions.get("avoid_together", []):
            a = pair[0].replace("_", " ").lower()
            b = pair[1].replace("_", " ").lower()
            reason = pair[2] if len(pair) > 2 else ""
            self.cypher_lines.append(
                f"MATCH (a:Ingredient {{name: '{_esc(a)}'}}), "
                f"(b:Ingredient {{name: '{_esc(b)}'}})\n"
                f"  MERGE (a)-[:CONFLICTS_WITH {{reason: '{_esc(reason)}'}}]->(b);"
            )
            self.stats["conflicts_with_edges"] += 1
        self.cypher_lines.append("")

    def _gen_pregnancy_contraindicated(self) -> None:
        """Generate CONTRAINDICATED_IN edges for pregnancy-unsafe ingredients."""
        self.cypher_lines.append("// ── Pregnancy contraindications (CONTRAINDICATED_IN) ─────────────────")
        for ing_raw in self.taxonomy.get("pregnancy_avoid", []):
            ing = ing_raw.replace("_", " ").lower()
            self.cypher_lines.append(
                f"MATCH (i:Ingredient {{name: '{_esc(ing)}'}}), "
                f"(c:Condition {{name: 'pregnancy'}})\n"
                f"  MERGE (i)-[:CONTRAINDICATED_IN {{reason: 'pregnancy safety'}}]->(c);"
            )
            self.stats["contraindicated_in_edges"] += 1

        # Ensure pregnancy is a "condition" node
        self.cypher_lines.insert(
            self.cypher_lines.index("// ── Ingredient conflicts (CONFLICTS_WITH) ────────────────────────────"),
            "MERGE (:Condition {name: 'pregnancy', subtypes: [], keywords: ['pregnancy', 'breastfeeding']});",
        )
        self.cypher_lines.append("")

    def _gen_concentration_limits(self) -> None:
        """Generate MAX_CONCENTRATION properties (stored on Ingredient nodes, not edges)."""
        self.cypher_lines.append("// ── Concentration limits (update Ingredient nodes) ───────────────────")
        for ing_key, limits in self.taxonomy.get("concentration_limits", {}).items():
            ing = ing_key.replace("_", " ").lower()
            otc_max = limits.get("otc_max", "")
            note = limits.get("note", "")
            if otc_max:
                self.cypher_lines.append(
                    f"MATCH (i:Ingredient {{name: '{_esc(ing)}'}})\n"
                    f"  SET i.otc_max = '{_esc(otc_max)}', i.concentration_note = '{_esc(note)}';"
                )
        self.cypher_lines.append("")

    def _gen_product_nodes(self, max_products: int) -> None:
        """Generate Product nodes and CONTAINS edges from OpenBeautyFacts."""
        parquet_path = Path("data/cosmetic_raw/openbeautyfacts.parquet")
        if not parquet_path.exists():
            logger.info("[seed_graph] No OpenBeautyFacts data — skipping Product nodes")
            self.cypher_lines.append("// Product nodes skipped — run: python scripts/download_cosmetic_data.py")
            return

        try:
            from src.collectors.cosmetic_api_collector import OpenBeautyFactsCollector
            collector = OpenBeautyFactsCollector(dest_dir="data/cosmetic_raw")
            products = collector.download_parquet_dump()[:max_products]
        except Exception as e:
            logger.warning(f"[seed_graph] Could not load products: {e}")
            return

        self.cypher_lines.append(f"// ── Product nodes ({len(products)} products) ──────────────────────────────")
        for p in products:
            cats = json.dumps(p.categories[:5])
            self.cypher_lines.append(
                f"MERGE (:Product {{id: '{_esc(p.product_id)}', "
                f"name: '{_esc(p.name[:100])}', "
                f"brand: '{_esc(p.brand[:50])}', "
                f"categories: {cats}}});"
            )
            self.stats["product_nodes"] += 1

            for ing in p.ingredients[:30]:  # cap to first 30 per product
                self.cypher_lines.append(
                    f"MATCH (pr:Product {{id: '{_esc(p.product_id)}'}})\n"
                    f"MERGE (i:Ingredient {{name: '{_esc(ing)}'}})\n"
                    f"MERGE (pr)-[:CONTAINS {{inci_name: '{_esc(ing)}'}}]->(i);"
                )
                self.stats["contains_edges"] += 1

        self.cypher_lines.append("")

    def save(self, output_dir: Path) -> tuple[Path, Path]:
        """Save generated Cypher and stats to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        cypher_path = output_dir / "seed.cypher"
        stats_path = output_dir / "seed_stats.json"

        cypher_content = "\n".join(self.cypher_lines)
        cypher_path.write_text(cypher_content, encoding="utf-8")

        self.stats["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        stats_path.write_text(json.dumps(self.stats, indent=2), encoding="utf-8")

        return cypher_path, stats_path


def main(args: argparse.Namespace) -> None:
    console.rule("[bold]Knowledge Graph Seeder[/bold]")

    seeder = GraphSeeder()
    seeder.generate_all(max_products=args.products)

    if args.dry_run:
        console.print("[yellow]DRY RUN — showing stats only (no files written)[/yellow]")
    else:
        output_dir = Path("data/graph")
        cypher_path, stats_path = seeder.save(output_dir)
        console.print(f"[green]Cypher saved → {cypher_path}[/green]")
        console.print(f"[green]Stats saved  → {stats_path}[/green]")

    # Stats table
    table = Table(title="Generation Stats")
    table.add_column("Entity", style="cyan")
    table.add_column("Count", justify="right", style="green")

    for key, val in seeder.stats.items():
        if key == "timestamp":
            continue
        table.add_row(key.replace("_", " ").title(), str(val))

    console.print(table)

    if not args.dry_run:
        console.print(
            "\nTo load into Neo4j:\n"
            "  cypher-shell -u neo4j -p <pass> --file data/graph/seed.cypher\n"
            "  Or: copy contents to Neo4j Browser and run"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Neo4j Cypher for knowledge graph seeding")
    parser.add_argument(
        "--products",
        type=int,
        default=500,
        help="Max product nodes to generate (default: 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without writing files",
    )
    args = parser.parse_args()
    main(args)
