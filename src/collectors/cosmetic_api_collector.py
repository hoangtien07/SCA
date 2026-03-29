"""
src/collectors/cosmetic_api_collector.py

Cosmetic product data collectors for graph database seeding.

Sources:
  1. OpenBeautyFacts — open cosmetics database (Parquet dump)
  2. CosIng — EU cosmetic ingredient database (rate-limited API)

Usage:
    from src.collectors.cosmetic_api_collector import OpenBeautyFactsCollector, CosIngCollector

    # OpenBeautyFacts
    collector = OpenBeautyFactsCollector(dest_dir="./data/cosmetic_raw")
    products = collector.download_parquet_dump()

    # CosIng
    cosing = CosIngCollector()
    ingredient = cosing.fetch_ingredient("Retinol")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class CosmeticProduct:
    """A cosmetic product with ingredient list."""
    product_id: str
    name: str
    brand: str
    ingredients_raw: str                   # raw INCI string
    ingredients: list[str]                 # parsed INCI names
    categories: list[str]
    country: str = ""
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "brand": self.brand,
            "ingredients_raw": self.ingredients_raw,
            "ingredients": self.ingredients,
            "categories": self.categories,
            "country": self.country,
            "url": self.url,
        }


@dataclass
class CosIngRecord:
    """EU CosIng database record for a cosmetic ingredient."""
    inci_name: str
    ec_number: str = ""
    cas_number: str = ""
    functions: list[str] = field(default_factory=list)
    description: str = ""
    restrictions: str = ""                 # EU regulatory restrictions
    prohibited: bool = False

    def to_dict(self) -> dict:
        return {
            "inci_name": self.inci_name,
            "ec_number": self.ec_number,
            "cas_number": self.cas_number,
            "functions": self.functions,
            "description": self.description,
            "restrictions": self.restrictions,
            "prohibited": self.prohibited,
        }


class OpenBeautyFactsCollector:
    """
    Downloads the Open Beauty Facts dataset (Parquet format).
    Open Beauty Facts is the cosmetic equivalent of Open Food Facts.
    License: Open Database License (ODbL)
    """

    PARQUET_URL = "https://static.openbeautyfacts.org/data/openbeautyfacts.parquet"
    CSV_URL = "https://static.openbeautyfacts.org/data/en.openbeautyfacts.org.products.csv.gz"

    def __init__(self, dest_dir: str = "./data/cosmetic_raw"):
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)

    def download_parquet_dump(self, dry_run: bool = False) -> list[CosmeticProduct]:
        """
        Download the Open Beauty Facts Parquet dump and parse into CosmeticProduct list.

        Args:
            dry_run: If True, only check size without downloading

        Returns:
            List of CosmeticProduct objects
        """
        import requests

        parquet_path = self.dest_dir / "openbeautyfacts.parquet"

        if parquet_path.exists():
            logger.info(f"[OpenBeautyFacts] Using cached file: {parquet_path}")
        else:
            if dry_run:
                try:
                    resp = requests.head(self.PARQUET_URL, timeout=10, allow_redirects=True)
                    size_mb = int(resp.headers.get("content-length", 0)) / (1024 * 1024)
                    logger.info(f"[OpenBeautyFacts] DRY RUN — Parquet size: ~{size_mb:.0f} MB")
                except Exception:
                    logger.info(f"[OpenBeautyFacts] DRY RUN — would download {self.PARQUET_URL}")
                return []

            logger.info(f"[OpenBeautyFacts] Downloading Parquet dump...")
            self._download_file(self.PARQUET_URL, parquet_path)

        return self._parse_parquet(parquet_path)

    def extract_ingredients(self, products: list[CosmeticProduct]) -> dict[str, int]:
        """
        Extract ingredient frequency map from product list.

        Returns:
            {ingredient_inci_name: count_of_products_containing_it}
        """
        freq: dict[str, int] = {}
        for p in products:
            for ing in p.ingredients:
                ing_clean = ing.strip().lower()
                if ing_clean:
                    freq[ing_clean] = freq.get(ing_clean, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: -x[1]))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file with tqdm progress bar."""
        import requests
        from tqdm import tqdm

        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total, unit="iB", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))

        logger.info(f"[OpenBeautyFacts] Downloaded → {dest}")

    def _parse_parquet(self, parquet_path: Path) -> list[CosmeticProduct]:
        """Parse Parquet file into CosmeticProduct list."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("[OpenBeautyFacts] pyarrow not installed. Run: pip install pyarrow>=16.0.0")
            return []

        logger.info(f"[OpenBeautyFacts] Parsing {parquet_path}...")

        table = pq.read_table(
            parquet_path,
            columns=["code", "product_name", "brands", "ingredients_text", "categories", "countries", "url"],
        )
        df = table.to_pandas()

        products = []
        for _, row in df.iterrows():
            ingredients_raw = str(row.get("ingredients_text", "") or "")
            # Parse INCI ingredients: split by comma, clean whitespace
            ingredients = [
                i.strip().lower()
                for i in ingredients_raw.replace(";", ",").split(",")
                if i.strip()
            ][:50]  # cap at 50 ingredients per product

            categories_raw = str(row.get("categories", "") or "")
            categories = [c.strip() for c in categories_raw.split(",") if c.strip()][:10]

            product = CosmeticProduct(
                product_id=str(row.get("code", "")),
                name=str(row.get("product_name", "") or ""),
                brand=str(row.get("brands", "") or ""),
                ingredients_raw=ingredients_raw[:500],  # truncate for storage
                ingredients=ingredients,
                categories=categories,
                country=str(row.get("countries", "") or "")[:100],
                url=str(row.get("url", "") or ""),
            )
            products.append(product)

        logger.info(f"[OpenBeautyFacts] Parsed {len(products)} products")
        return products


class CosIngCollector:
    """
    Fetches cosmetic ingredient data from the EU CosIng database.
    Rate limit: max 3 requests/second (enforced internally).

    Note: CosIng does not provide a public REST API; this collector
    uses a scraped/cached version or the official CosIng search endpoint.
    """

    BASE_URL = "https://ec.europa.eu/growth/tools-databases/cosing/rest/ingredients"
    RATE_LIMIT_INTERVAL = 1.0 / 3  # 3 req/s = 0.333s between requests

    def __init__(self):
        self._last_request_time: float = 0.0

    def fetch_ingredient(self, inci_name: str) -> CosIngRecord | None:
        """
        Fetch a single ingredient from CosIng.

        Args:
            inci_name: INCI name of the ingredient (e.g. "RETINOL")

        Returns:
            CosIngRecord or None if not found
        """
        import requests

        self._rate_limit()

        try:
            resp = requests.get(
                self.BASE_URL,
                params={"name": inci_name.upper(), "format": "json"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("results"):
                return None

            item = data["results"][0]
            return CosIngRecord(
                inci_name=item.get("inci_name", inci_name),
                ec_number=item.get("ec_number", ""),
                cas_number=item.get("cas_number", ""),
                functions=item.get("functions", "").split(", ") if item.get("functions") else [],
                description=item.get("description", "")[:500],
                restrictions=item.get("restrictions", ""),
                prohibited=bool(item.get("prohibited", False)),
            )
        except Exception as e:
            logger.warning(f"[CosIngCollector] fetch_ingredient({inci_name!r}) failed: {e}")
            return None

    def batch_fetch(self, inci_names: list[str]) -> list[CosIngRecord]:
        """
        Fetch multiple ingredients with rate limiting (3 req/s).

        Args:
            inci_names: List of INCI names

        Returns:
            List of CosIngRecord objects (skips failures)
        """
        from tqdm import tqdm

        results = []
        for name in tqdm(inci_names, desc="CosIng fetch"):
            record = self.fetch_ingredient(name)
            if record is not None:
                results.append(record)

        logger.info(f"[CosIngCollector] Fetched {len(results)}/{len(inci_names)} ingredients")
        return results

    def _rate_limit(self) -> None:
        """Enforce 3 requests/second rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.RATE_LIMIT_INTERVAL:
            time.sleep(self.RATE_LIMIT_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()
