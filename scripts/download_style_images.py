"""Download the 10 style reference artworks used to train LiveArt models.

Images are sourced from Wikimedia Commons (public domain / free-license art
works). Files are written to ``backend/style_sources/<style_id>.jpg`` and the
same images are copied to ``backend/thumbnails/<style_id>.jpg`` so the frontend
has a proper preview (replacing the placeholder the download_models.py script
generates).

Run with::

    python scripts/download_style_images.py
"""
from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("download_style_images")

ROOT = Path(__file__).resolve().parent.parent
STYLE_SRC_DIR = ROOT / "backend" / "style_sources"
THUMB_DIR = ROOT / "backend" / "thumbnails"
STYLE_SRC_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)

UA = "LiveArtSemesterProject/1.0 (educational use)"

# (style_id, human name, source URL). All URLs resolve to full-size JPEG/PNG on
# Wikimedia Commons — no hot-linking tricks, these are the canonical file paths.
# Primary URL + fallback list. Pytorch/examples GitHub raw URLs are used for the
# canonical Fast-Style-Transfer reference images (candy, mosaic, rain_princess,
# udnie). Wikimedia is used for famous public-domain artworks with both a
# thumbnail URL and the Special:FilePath redirect as a backup path.
PYTORCH_BASE = (
    "https://raw.githubusercontent.com/pytorch/examples/main/fast_neural_style/"
    "images/style-images"
)

STYLES: list[tuple[str, str, list[str]]] = [
    (
        "starry_night",
        "Starry Night (Van Gogh)",
        [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/"
            "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/"
            "1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg?width=1280",
        ],
    ),
    (
        "the_scream",
        "The Scream (Munch)",
        [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/"
            "The_Scream.jpg/1024px-The_Scream.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg",
        ],
    ),
    (
        "great_wave",
        "The Great Wave off Kanagawa (Hokusai)",
        [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/"
            "Great_Wave_off_Kanagawa2.jpg/1280px-Great_Wave_off_Kanagawa2.jpg",
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            "Great_Wave_off_Kanagawa2.jpg?width=1280",
        ],
    ),
    (
        "composition_viii",
        "Color Study: Squares with Concentric Circles (Kandinsky)",
        [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/"
            "Vassily_Kandinsky%2C_1913_-_Color_Study%2C_Squares_with_Concentric_Circles.jpg/"
            "1024px-Vassily_Kandinsky%2C_1913_-_Color_Study%2C_Squares_with_Concentric_Circles.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/b/b4/"
            "Vassily_Kandinsky%2C_1913_-_Color_Study%2C_Squares_with_Concentric_Circles.jpg",
        ],
    ),
    (
        "udnie",
        "Udnie (Picabia)",
        [
            f"{PYTORCH_BASE}/udnie.jpg",
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            "Francis_Picabia,_1913,_Udnie_(Young_American_Girl,_The_Dance),_oil_on_canvas,_290_x_300_cm,_Mus%C3%A9e_National_d%27Art_Moderne,_Centre_Georges_Pompidou,_Paris..jpg?width=1280",
        ],
    ),
    (
        "la_muse",
        "Mont Sainte-Victoire (Cézanne)",
        [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/"
            "Paul_C%C3%A9zanne_108.jpg/1024px-Paul_C%C3%A9zanne_108.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/9/9a/"
            "Paul_C%C3%A9zanne_108.jpg",
        ],
    ),
    (
        "mosaic",
        "Byzantine Mosaic",
        [
            f"{PYTORCH_BASE}/mosaic.jpg",
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            "Meister_von_San_Vitale_in_Ravenna_008.jpg?width=1024",
        ],
    ),
    (
        "candy",
        "Candy (Fast-Style reference)",
        [
            f"{PYTORCH_BASE}/candy.jpg",
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            "A_Sunday_on_La_Grande_Jatte,_Georges_Seurat,_1884.jpg?width=1280",
        ],
    ),
    (
        "feathers",
        "Strawberry Thief (William Morris — decorative pattern)",
        [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/"
            "Morris_Strawberry_Thief_1883.jpg/1024px-Morris_Strawberry_Thief_1883.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/a/a5/"
            "Morris_Strawberry_Thief_1883.jpg",
        ],
    ),
    (
        "rain_princess",
        "Rain Princess (Fast-Style reference)",
        [
            f"{PYTORCH_BASE}/rain-princess.jpg",
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            "Monet_-_Impression,_Sunrise.jpg?width=1280",
        ],
    ),
]


def download_with_fallback(urls: list[str], dest: Path) -> None:
    last_exc: Exception | None = None
    for idx, url in enumerate(urls):
        try:
            logger.info("-> %s (src %d/%d)", dest.name, idx + 1, len(urls))
            with requests.get(
                url,
                stream=True,
                timeout=120,
                headers={"User-Agent": UA},
                allow_redirects=True,
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(dest, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, leave=False, desc=dest.name
                ) as bar:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                        bar.update(len(chunk))
            return
        except Exception as exc:
            last_exc = exc
            logger.warning("  src %d failed: %s", idx + 1, exc)
            if dest.exists():
                dest.unlink()
            time.sleep(2)  # backoff before next source
    if last_exc is not None:
        raise last_exc


def to_jpeg(src: Path, dst: Path, max_side: int = 1024) -> None:
    img = Image.open(src).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(dst, "JPEG", quality=92)


def main() -> int:
    failed: list[str] = []
    for style_id, display_name, urls in STYLES:
        raw_path = STYLE_SRC_DIR / f"{style_id}_raw"
        final_path = STYLE_SRC_DIR / f"{style_id}.jpg"
        thumb_path = THUMB_DIR / f"{style_id}.jpg"
        if final_path.exists() and thumb_path.exists():
            logger.info("skip %s (already present)", style_id)
            continue
        try:
            download_with_fallback(urls, raw_path)
            to_jpeg(raw_path, final_path, max_side=1024)
            to_jpeg(raw_path, thumb_path, max_side=512)
            raw_path.unlink()
            logger.info("saved %s (%s)", style_id, display_name)
            time.sleep(1.0)  # be gentle to upstream servers
        except Exception as exc:
            logger.error("FAILED %s: %s", style_id, exc)
            failed.append(style_id)
            if raw_path.exists():
                raw_path.unlink()

    ok = len(STYLES) - len(failed)
    logger.info("%d/%d style images downloaded", ok, len(STYLES))
    if failed:
        logger.warning("failed styles: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
