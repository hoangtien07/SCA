"""
src/collectors/cv_dataset_collector.py

Computer vision dataset collector for skin condition images.

Supported datasets:
  - SCIN (Skin Condition Image Network) — Google Research
  - Fitzpatrick17k — Fitzpatrick-scale labelled dermoscopy images
  - DDI (Diverse Dermatology Images) — diverse Fitzpatrick representation
  - DermaMNIST — MedMNIST v2 subset for dermatology

Each downloader:
  - Uses requests + tqdm for progress display
  - Verifies checksums where available
  - Saves to data/cv_datasets/{dataset_name}/

Usage:
    from src.collectors.cv_dataset_collector import CVDatasetCollector

    collector = CVDatasetCollector(dest_dir="./data/cv_datasets")
    manifest = collector.download_scin()
    print(manifest.num_images, manifest.fitzpatrick_distribution)
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class DatasetManifest:
    """Metadata about a downloaded CV dataset."""
    name: str
    num_images: int
    conditions: list[str]
    fitzpatrick_distribution: dict[str, int]  # {"I": 12, "II": 45, ...}
    license: str
    local_path: Path
    checksum_verified: bool = False
    download_url: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_images": self.num_images,
            "conditions": self.conditions,
            "fitzpatrick_distribution": self.fitzpatrick_distribution,
            "license": self.license,
            "local_path": str(self.local_path),
            "checksum_verified": self.checksum_verified,
            "download_url": self.download_url,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class CVDatasetCollector:
    """
    Downloads and verifies computer vision datasets for skin condition analysis.

    Usage:
        collector = CVDatasetCollector(dest_dir="./data/cv_datasets")
        manifest = collector.download_fitzpatrick17k(dry_run=True)
    """

    # Dataset registry with metadata
    DATASETS = {
        "scin": {
            "url": "https://storage.googleapis.com/scin-dataset/scin_dataset.zip",
            "description": "Skin Condition Image Network — Google Research",
            "license": "CC BY 4.0",
            "conditions": [
                "acne", "eczema", "psoriasis", "rosacea", "vitiligo",
                "urticaria", "contact dermatitis", "seborrheic dermatitis",
            ],
            "approx_images": 5000,
        },
        "fitzpatrick17k": {
            "url": "https://github.com/mattgroh/fitzpatrick17k/raw/main/fitzpatrick17k.csv",
            "description": "Fitzpatrick scale labelled dermatology images",
            "license": "CC BY-NC 4.0",
            "conditions": ["acne", "eczema", "psoriasis", "melanoma", "basal cell carcinoma"],
            "approx_images": 16577,
        },
        "ddi": {
            "url": "https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965",
            "description": "Diverse Dermatology Images — Stanford AIMI",
            "license": "Stanford AIMI Research Use",
            "conditions": ["melanoma", "melanocytic nevi", "seborrheic keratosis"],
            "approx_images": 656,
        },
        "dermamnist": {
            "url": "https://zenodo.org/record/6496656/files/dermamnist.npz",
            "description": "DermaMNIST — MedMNIST v2 dermatology subset",
            "license": "CC BY 4.0",
            "conditions": ["melanocytic nevi", "melanoma", "basal cell carcinoma",
                           "actinic keratoses", "benign keratosis", "dermatofibroma",
                           "vascular lesions"],
            "approx_images": 10015,
        },
    }

    def __init__(self, dest_dir: str = "./data/cv_datasets"):
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)

    def download_scin(self, dry_run: bool = False) -> DatasetManifest:
        """
        Download the SCIN (Skin Condition Image Network) dataset.

        Args:
            dry_run: If True, only check availability without downloading

        Returns:
            DatasetManifest with dataset metadata
        """
        return self._download_dataset("scin", dry_run=dry_run)

    def download_fitzpatrick17k(self, dry_run: bool = False) -> DatasetManifest:
        """
        Download the Fitzpatrick17k dataset (CSV + image links).

        Args:
            dry_run: If True, only check availability without downloading
        """
        return self._download_dataset("fitzpatrick17k", dry_run=dry_run)

    def download_ddi(self, dry_run: bool = False) -> DatasetManifest:
        """
        Download the Diverse Dermatology Images (DDI) dataset.

        Args:
            dry_run: If True, only check availability without downloading
        """
        return self._download_dataset("ddi", dry_run=dry_run)

    def download_dermamnist(self, dry_run: bool = False) -> DatasetManifest:
        """
        Download the DermaMNIST dataset (compressed numpy archive).

        Args:
            dry_run: If True, only check availability without downloading
        """
        return self._download_dataset("dermamnist", dry_run=dry_run)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _download_dataset(self, name: str, dry_run: bool = False) -> DatasetManifest:
        """Generic downloader with tqdm progress and checksum verification."""
        import requests
        from tqdm import tqdm

        meta = self.DATASETS.get(name)
        if not meta:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(self.DATASETS.keys())}")

        dataset_dir = self.dest_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        url = meta["url"]
        filename = url.split("/")[-1] or f"{name}_data"
        local_file = dataset_dir / filename
        manifest_path = dataset_dir / "manifest.json"

        # Return existing manifest if already downloaded
        if manifest_path.exists():
            logger.info(f"[CVDatasetCollector] {name} already downloaded, loading manifest")
            data = json.loads(manifest_path.read_text())
            return DatasetManifest(
                name=data["name"],
                num_images=data["num_images"],
                conditions=data["conditions"],
                fitzpatrick_distribution=data["fitzpatrick_distribution"],
                license=data["license"],
                local_path=Path(data["local_path"]),
                checksum_verified=data.get("checksum_verified", False),
                download_url=data.get("download_url", url),
            )

        if dry_run:
            logger.info(f"[CVDatasetCollector] DRY RUN — would download {url}")
            # Estimate size via HEAD request
            try:
                resp = requests.head(url, timeout=10, allow_redirects=True)
                size_mb = int(resp.headers.get("content-length", 0)) / (1024 * 1024)
                logger.info(f"  Estimated size: {size_mb:.1f} MB")
            except Exception:
                pass

            return DatasetManifest(
                name=name,
                num_images=meta["approx_images"],
                conditions=meta["conditions"],
                fitzpatrick_distribution={},
                license=meta["license"],
                local_path=dataset_dir,
                download_url=url,
            )

        # Download with streaming + tqdm
        logger.info(f"[CVDatasetCollector] Downloading {name} from {url}")
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            with open(local_file, "wb") as f, tqdm(
                total=total,
                unit="iB",
                unit_scale=True,
                desc=f"Downloading {name}",
            ) as bar:
                for chunk in resp.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

            logger.info(f"[CVDatasetCollector] Downloaded → {local_file}")

        except requests.RequestException as e:
            logger.error(f"[CVDatasetCollector] Download failed for {name}: {e}")
            # Return stub manifest so pipeline continues
            return DatasetManifest(
                name=name,
                num_images=0,
                conditions=meta["conditions"],
                fitzpatrick_distribution={},
                license=meta["license"],
                local_path=dataset_dir,
                download_url=url,
            )

        # Checksum verification (SHA256) if checksum file exists
        checksum_verified = self._verify_checksum(name, local_file)

        # Build manifest
        manifest = DatasetManifest(
            name=name,
            num_images=meta["approx_images"],  # actual count requires parsing
            conditions=meta["conditions"],
            fitzpatrick_distribution=self._estimate_fitzpatrick(name, local_file),
            license=meta["license"],
            local_path=dataset_dir,
            checksum_verified=checksum_verified,
            download_url=url,
        )

        manifest.save(manifest_path)
        logger.info(f"[CVDatasetCollector] Manifest saved → {manifest_path}")
        return manifest

    def _verify_checksum(self, name: str, file_path: Path) -> bool:
        """
        Verify SHA256 checksum of downloaded file.
        Returns False if no expected checksum is available (not an error).
        """
        checksums = {
            "dermamnist": "a29e15ac21f7c1960c33c6965498c3d8b53437e41ffb4e63e43c0e42736bc9df",
        }
        expected = checksums.get(name)
        if not expected:
            return False

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        actual = sha256.hexdigest()
        if actual != expected:
            logger.warning(
                f"[CVDatasetCollector] Checksum mismatch for {name}! "
                f"Expected {expected[:8]}... got {actual[:8]}..."
            )
            return False

        logger.info(f"[CVDatasetCollector] Checksum verified for {name}")
        return True

    def _estimate_fitzpatrick(self, name: str, file_path: Path) -> dict[str, int]:
        """
        Estimate Fitzpatrick distribution from dataset if parseable.
        Returns empty dict for binary downloads.
        """
        if name == "fitzpatrick17k" and file_path.suffix == ".csv":
            try:
                import csv
                distribution: dict[str, int] = {}
                with open(file_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fitz = str(row.get("fitzpatrick_scale", "")).strip()
                        if fitz and fitz.isdigit():
                            roman = ["I", "II", "III", "IV", "V", "VI"]
                            idx = int(fitz) - 1
                            if 0 <= idx < len(roman):
                                key = roman[idx]
                                distribution[key] = distribution.get(key, 0) + 1
                return distribution
            except Exception as e:
                logger.warning(f"[CVDatasetCollector] Could not parse Fitzpatrick distribution: {e}")

        return {}
