"""Download the MS COCO image set used as the content corpus for training.

By default this fetches COCO val2017 (~1 GB, 5 000 images) — enough for good
Fast Style Transfer results with reasonable training time on consumer GPUs.

Pass ``--full`` to download COCO train2014 (~13 GB, 82 783 images) instead, to
match the exact dataset used in Johnson et al. (2016).

Run::

    python scripts/download_coco.py          # val2017 (fast)
    python scripts/download_coco.py --full   # train2014 (best quality, slow)

Extracted files end up under ``data/coco_train/`` regardless of variant, so
the existing ``COCO_TRAIN_DIR`` default in the backend just works.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("download_coco")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "coco_train"
CACHE_DIR = ROOT / ".cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "val": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "archive": "val2017.zip",
        "dir": "val2017",
    },
    "full": {
        "url": "http://images.cocodataset.org/zips/train2014.zip",
        "archive": "train2014.zip",
        "dir": "train2014",
    },
}


def download(url: str, dest: Path) -> None:
    logger.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))


def extract(archive: Path, target_subdir: str) -> int:
    logger.info("extracting %s", archive.name)
    count = 0
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if m.endswith((".jpg", ".jpeg", ".png"))]
        for name in tqdm(members, desc="extract", unit="img"):
            src_path = Path(name)
            if src_path.parts and src_path.parts[0] == target_subdir:
                rel = Path(*src_path.parts[1:])
            else:
                rel = src_path
            out_path = DATA_DIR / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download COCO train2014 (~13 GB, 82k images) instead of val2017.",
    )
    args = parser.parse_args()
    variant = "full" if args.full else "val"
    spec = VARIANTS[variant]

    existing = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(existing) > 100:
        logger.info("already have %d images in %s; skipping download", len(existing), DATA_DIR)
        return 0

    archive = CACHE_DIR / spec["archive"]
    if not archive.exists():
        download(spec["url"], archive)
    else:
        logger.info("using cached %s", archive.name)

    n = extract(archive, spec["dir"])
    logger.info("extracted %d images into %s", n, DATA_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
