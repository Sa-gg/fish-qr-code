# =============================================================================
# download_dataset.py — Auto-download sample fish images for prototyping
# =============================================================================
# This script downloads images from Bing Image Search automatically so you can
# start training without needing to collect photos manually.
#
# USAGE:
#   python download_dataset.py
#
# REQUIREMENTS (install once):
#   pip install icrawler
#
# NOTE: For prototyping only. Downloaded images may vary in quality.
#       For production, curate your own verified dataset.
# =============================================================================

import os
import shutil
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Define species with targeted search keywords for better image quality.
# To add a new species: add an entry to this dictionary — no other changes needed.
SPECIES_CONFIG = {
    "Bangus": {
        "keywords": [
            "fresh raw bangus milkfish whole uncooked",
            "bangus milkfish fresh fish market whole",
            "live bangus milkfish Philippines raw",
            "Chanos chanos whole fresh raw fish",
            "bangus fish raw uncooked whole body",
        ],
        "train_count": 60,  # images to download for training
        "val_count":   15,  # images to download for validation
    },
    "Tilapia": {
        "keywords": [
            "fresh raw tilapia whole fish uncooked",
            "tilapia fish market fresh live whole",
            "Nile tilapia fresh raw whole fish",
            "tilapia Philippines fresh fish raw",
            "tilapia fish uncooked whole body fresh",
        ],
        "train_count": 60,
        "val_count":   15,
    },
    "Galunggong": {
        "keywords": [
            "fresh raw galunggong round scad whole",
            "galunggong fish market fresh uncooked",
            "Decapterus macrosoma fresh whole fish",
            "round scad fish fresh raw whole Philippines",
            "galunggong fresh fish raw uncooked whole body",
        ],
        "train_count": 60,
        "val_count":   15,
    },
}

DATASET_ROOT = Path("dataset")
TRAIN_DIR    = DATASET_ROOT / "train"
VAL_DIR      = DATASET_ROOT / "validation"

# Minimum file size in bytes — filters out tiny broken/thumbnail images
MIN_FILE_SIZE = 8_000    # ~8 KB


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def clean_small_files(folder: Path, min_size: int) -> int:
    """Remove images that are too small (likely corrupt/thumbnail)."""
    removed = 0
    for f in folder.iterdir():
        if f.is_file() and f.stat().st_size < min_size:
            f.unlink()
            removed += 1
    return removed


def split_into_validation(train_folder: Path, val_folder: Path, val_count: int) -> None:
    """Move `val_count` random images from train → validation folder."""
    val_folder.mkdir(parents=True, exist_ok=True)
    images = [
        f for f in train_folder.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    if len(images) < val_count:
        print(f"   [WARN] Only {len(images)} images available; "
              f"moving all to validation.")
        val_count = len(images)

    chosen = random.sample(images, val_count)
    for img in chosen:
        shutil.move(str(img), str(val_folder / img.name))
    print(f"   Moved {len(chosen)} images to validation.")


# ---------------------------------------------------------------------------
# MAIN DOWNLOAD LOGIC
# ---------------------------------------------------------------------------

def download_species(species_name: str, config: dict) -> None:
    """Download images for a single species using Bing Image Search."""
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("\n[ERROR] icrawler is not installed.")
        print("        Run:  pip install icrawler\n")
        raise SystemExit(1)

    train_folder = TRAIN_DIR / species_name
    train_folder.mkdir(parents=True, exist_ok=True)

    total_needed = config["train_count"] + config["val_count"]
    per_keyword  = max(1, total_needed // len(config["keywords"]))

    print(f"\n{'='*55}")
    print(f"  Downloading: {species_name}")
    print(f"  Target      : {total_needed} images  ({len(config['keywords'])} keywords)")
    print(f"{'='*55}")

    for keyword in config["keywords"]:
        print(f"  Keyword: \"{keyword}\"  →  up to {per_keyword} images")
        crawler = BingImageCrawler(
            downloader_threads=4,   # parallel downloads, friendly on CPU
            storage={"root_dir": str(train_folder)},
            log_level=40            # ERROR level — suppresses verbose icrawler logs
        )
        crawler.crawl(
            keyword=keyword,
            max_num=per_keyword,
            min_size=(150, 150),    # skip very small images at the API level
            file_idx_offset="auto"  # avoids overwriting existing files
        )

    # Clean broken / tiny files
    removed = clean_small_files(train_folder, MIN_FILE_SIZE)
    if removed:
        print(f"   Removed {removed} tiny/corrupt files.")

    remaining = list(train_folder.iterdir())
    print(f"   Images after cleanup: {len(remaining)}")

    # Move a portion to validation
    val_folder = VAL_DIR / species_name
    split_into_validation(train_folder, val_folder, config["val_count"])

    train_left = sum(1 for f in train_folder.iterdir() if f.is_file())
    val_left   = sum(1 for f in val_folder.iterdir()   if f.is_file())
    print(f"   Final → Train: {train_left}  |  Validation: {val_left}")


def main() -> None:
    print("\nPhilippine Fish Dataset Downloader")
    print("===================================")
    print(f"Species to download: {list(SPECIES_CONFIG.keys())}\n")

    for species, config in SPECIES_CONFIG.items():
        download_species(species, config)

    # ---- Summary ----
    print("\n\nDataset Download Complete — Summary")
    print("=" * 45)
    for split_name, split_dir in [("Train", TRAIN_DIR), ("Validation", VAL_DIR)]:
        print(f"\n  {split_name}:")
        if split_dir.exists():
            for species_dir in sorted(split_dir.iterdir()):
                if species_dir.is_dir():
                    count = sum(1 for f in species_dir.iterdir() if f.is_file()
                                and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
                    print(f"    {species_dir.name:<15} {count:>3} images")

    print("\nNext step:  python train_fish_classifier.py")


if __name__ == "__main__":
    main()
