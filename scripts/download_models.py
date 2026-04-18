"""Download pre-trained Fast Neural Style Transfer weights and generate thumbnails.

This script is idempotent — existing files are skipped. It:

1. Downloads `saved_models.zip` from the official `pytorch/examples` release,
   which bundles four pre-trained styles: ``mosaic``, ``candy``,
   ``rain_princess``, ``udnie``.
2. Extracts the matching ``.pth`` files into ``backend/models/``.
3. Writes a ``<style_id>.json`` metadata file for each of the 10 styles in
   ``backend/models/``.
4. Generates a placeholder thumbnail JPEG for each style in
   ``backend/thumbnails/`` using PIL (solid color derived from a hash of the
   style ID, with the style name drawn on a dark strip at the bottom).
5. Logs which of the 10 styles are still missing trained weights and prints
   instructions to train them via the UI.

Run from the repo root::

    python scripts/download_models.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "backend" / "models"
THUMBNAILS_DIR = REPO_ROOT / "backend" / "thumbnails"
CACHE_DIR = REPO_ROOT / ".cache"

# ---------------------------------------------------------------------------
# Remote weights archive
# ---------------------------------------------------------------------------

# Official pytorch/examples release bundling 4 Fast Neural Style weights.
WEIGHTS_ZIP_URL = (
    "https://github.com/pytorch/examples/releases/download/"
    "fast-neural-style/saved_models.zip"
)
WEIGHTS_ZIP_NAME = "saved_models.zip"

# Style IDs included in the upstream archive. The archive contains files like
# ``saved_models/mosaic.pth`` (or sometimes bare ``mosaic.pth``) — we match by
# stem so either layout works.
BUNDLED_STYLE_IDS = {"mosaic", "candy", "rain_princess", "udnie"}

# ---------------------------------------------------------------------------
# Style library — spec §3.4
# ---------------------------------------------------------------------------

STYLES: List[Dict[str, str]] = [
    {"id": "starry_night", "name": "Starry Night", "artist": "Van Gogh"},
    {"id": "the_scream", "name": "The Scream", "artist": "Edvard Munch"},
    {"id": "candy", "name": "Candy", "artist": ""},
    {"id": "mosaic", "name": "Mosaic", "artist": ""},
    {"id": "udnie", "name": "Udnie", "artist": "Francis Picabia"},
    {"id": "rain_princess", "name": "Rain Princess", "artist": "Leonid Afremov"},
    {"id": "la_muse", "name": "La Muse", "artist": "Pablo Picasso"},
    {"id": "feathers", "name": "Feathers", "artist": ""},
    {"id": "great_wave", "name": "The Great Wave", "artist": "Hokusai"},
    {
        "id": "composition_viii",
        "name": "Composition VIII",
        "artist": "Wassily Kandinsky",
    },
]

# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

THUMBNAIL_SIZE = (320, 240)
LABEL_STRIP_HEIGHT = 56


def _color_from_id(style_id: str) -> tuple[int, int, int]:
    """Derive a deterministic, visually distinct RGB color from a style ID."""
    digest = hashlib.md5(style_id.encode("utf-8")).digest()
    # Use the first 3 bytes but push them into the mid-bright range so the
    # white label is legible against the solid background.
    r = 60 + (digest[0] % 160)
    g = 60 + (digest[1] % 160)
    b = 60 + (digest[2] % 160)
    return r, g, b


def _load_font(size: int) -> ImageFont.ImageFont:
    """Best-effort TrueType font load; falls back to PIL default bitmap font."""
    candidates = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
        "arial.ttf",
        "Helvetica.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def generate_thumbnail(style: Dict[str, str], out_path: Path) -> None:
    """Generate a solid-color thumbnail with the style name on a dark strip."""
    bg = _color_from_id(style["id"])
    img = Image.new("RGB", THUMBNAIL_SIZE, bg)
    draw = ImageDraw.Draw(img)

    # Dark label strip along the bottom.
    strip_top = THUMBNAIL_SIZE[1] - LABEL_STRIP_HEIGHT
    draw.rectangle(
        [(0, strip_top), (THUMBNAIL_SIZE[0], THUMBNAIL_SIZE[1])],
        fill=(20, 20, 24),
    )

    # Style name, centered in the strip.
    font = _load_font(22)
    text = style["name"]
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:  # Pillow <9.2 fallback
        text_w, text_h = draw.textsize(text, font=font)

    tx = (THUMBNAIL_SIZE[0] - text_w) // 2
    ty = strip_top + (LABEL_STRIP_HEIGHT - text_h) // 2 - 2
    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=88)


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------


def download_with_progress(url: str, dest: Path) -> None:
    """Stream-download ``url`` to ``dest`` with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        chunk_size = 1024 * 64
        with open(tmp, "wb") as fh, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def extract_bundled_weights(zip_path: Path, models_dir: Path) -> List[str]:
    """Extract ``.pth`` files from the pytorch/examples archive.

    Returns the list of style IDs successfully written to ``models_dir``.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name.endswith(".pth"):
                continue
            stem = Path(name).stem
            if stem not in BUNDLED_STYLE_IDS:
                continue
            target = models_dir / f"{stem}.pth"
            if target.exists():
                extracted.append(stem)
                continue
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(stem)

    return extracted


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def write_metadata(style: Dict[str, str], models_dir: Path) -> Path:
    """Write a ``<style_id>.json`` metadata file. Overwrites if different."""
    out = models_dir / f"{style['id']}.json"
    payload = {"name": style["name"], "artist": style["artist"]}
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            if existing == payload:
                return out
        except (OSError, json.JSONDecodeError):
            pass
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("LiveArt — model & thumbnail bootstrap")
    print(f"  models dir:     {MODELS_DIR}")
    print(f"  thumbnails dir: {THUMBNAILS_DIR}")
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Download + extract bundled weights (idempotent).
    zip_path = CACHE_DIR / WEIGHTS_ZIP_NAME
    already_have = {
        sid for sid in BUNDLED_STYLE_IDS if (MODELS_DIR / f"{sid}.pth").exists()
    }
    need_download = BUNDLED_STYLE_IDS - already_have

    if not need_download:
        print(
            "[weights] All bundled pre-trained styles already present — "
            "skipping download."
        )
        extracted = sorted(already_have)
    else:
        if zip_path.exists():
            print(f"[weights] Using cached archive: {zip_path}")
        else:
            print(f"[weights] Downloading {WEIGHTS_ZIP_URL}")
            try:
                download_with_progress(WEIGHTS_ZIP_URL, zip_path)
            except requests.RequestException as exc:
                print(f"[weights] ERROR: download failed: {exc}", file=sys.stderr)
                print(
                    "          You can manually place .pth files into "
                    f"{MODELS_DIR} and re-run this script.",
                    file=sys.stderr,
                )
                return 1

        print(f"[weights] Extracting .pth files into {MODELS_DIR}")
        extracted = extract_bundled_weights(zip_path, MODELS_DIR)

    print(f"[weights] Available pretrained styles: {sorted(set(extracted))}")
    print()

    # 2) Write per-style metadata JSON.
    print("[metadata] Writing <style_id>.json files...")
    for style in STYLES:
        path = write_metadata(style, MODELS_DIR)
        print(f"  - {path.name}")
    print()

    # 3) Generate placeholder thumbnails (skip if present).
    print("[thumbnails] Generating placeholder thumbnails...")
    for style in STYLES:
        out = THUMBNAILS_DIR / f"{style['id']}.jpg"
        if out.exists():
            print(f"  - {out.name}  (skipped, already exists)")
            continue
        generate_thumbnail(style, out)
        print(f"  - {out.name}  (generated)")
    print()

    # 4) Log which styles still need training.
    have_weights = {
        style["id"]
        for style in STYLES
        if (MODELS_DIR / f"{style['id']}.pth").exists()
    }
    missing = [s for s in STYLES if s["id"] not in have_weights]

    if missing:
        print("[next steps] The following styles do not yet have trained weights:")
        for s in missing:
            label = f"{s['name']}"
            if s["artist"]:
                label += f" ({s['artist']})"
            print(f"  - {s['id']}: {label}")
        print()
        print(
            "  To create weights for these, launch the app and use the Training "
            "page:\n"
            "    1) uvicorn backend.main:app --reload --port 8000\n"
            "    2) cd frontend && npm run dev\n"
            "    3) Visit http://localhost:5173/training, upload a reference\n"
            "       image for each style above, and press Start.\n"
        )
    else:
        print("[next steps] All 10 styles have trained weights. Ready to go!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
