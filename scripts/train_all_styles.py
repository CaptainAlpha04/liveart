"""Batch-train LiveArt style transfer models for all 10 art styles.

For each style listed in ``STYLE_TITLES`` this script runs a full
``TrainingEngine.train()`` cycle against the style reference image at
``backend/style_sources/<style_id>.jpg`` using images from ``COCO_TRAIN_DIR``
(defaults to ``data/coco_train/``) as the content corpus.

Finished models are saved directly to ``backend/models/<style_id>.pth`` along
with a metadata JSON so they appear as "pretrained" in the UI at startup.

Progress is logged to stdout. Skips any style whose ``.pth`` already exists.

Hyperparameters are tuned for fast-but-decent quality on a consumer GPU with
COCO val2017 (5 000 images). For production-quality output, pass ``--full`` —
uses Johnson et al.'s recommended settings over the full corpus.

Usage::

    python scripts/train_all_styles.py                    # fast preset
    python scripts/train_all_styles.py --full             # paper preset
    python scripts/train_all_styles.py --only starry_night the_scream
    python scripts/train_all_styles.py --skip candy mosaic
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train_all_styles")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STYLE_SRC = ROOT / "backend" / "style_sources"
MODELS_DIR = ROOT / "backend" / "models"
DATA_DIR = Path(os.environ.get("COCO_TRAIN_DIR", ROOT / "data" / "coco_train"))

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# style_id -> (display name, artist). Names are chosen so that
# ``slugify(name) == style_id`` — the TrainingEngine saves weights at
# ``<slugify(style_name)>.pth``, so keeping them aligned avoids filename drift.
STYLE_TITLES: dict[str, tuple[str, str]] = {
    "starry_night": ("Starry Night", "Van Gogh"),
    "the_scream": ("The Scream", "Edvard Munch"),
    "great_wave": ("Great Wave", "Hokusai"),
    "composition_viii": ("Composition VIII", "Kandinsky"),
    "udnie": ("Udnie", "Francis Picabia"),
    "la_muse": ("La Muse", "Picasso / Braque"),
    "mosaic": ("Mosaic", "Byzantine"),
    "candy": ("Candy", "Pointillism"),
    "feathers": ("Feathers", "Decorative pattern"),
    "rain_princess": ("Rain Princess", "Afremov / Monet"),
}

FAST_CFG = dict(
    style_weight=1e10,
    content_weight=1e5,
    tv_weight=1e-6,
    learning_rate=1e-3,
    epochs=2,
    batch_size=8,
    image_size=256,
)

PAPER_CFG = dict(
    style_weight=1e10,
    content_weight=1e5,
    tv_weight=1e-6,
    learning_rate=1e-3,
    epochs=2,
    batch_size=4,
    image_size=256,
)


def _progress_cb(style_id: str):
    last_emit = {"t": 0.0}

    def cb(event: dict) -> None:
        if event.get("status") == "done":
            logger.info(
                "[%s] DONE — elapsed=%.1fs model=%s",
                style_id,
                event.get("elapsed_s", 0),
                event.get("model_path"),
            )
            return
        if event.get("status") == "error":
            logger.error("[%s] ERROR — %s", style_id, event.get("error"))
            return
        if event.get("status") == "stopped":
            logger.warning("[%s] STOPPED", style_id)
            return
        now = time.time()
        if now - last_emit["t"] < 5.0:
            return  # throttle console output
        last_emit["t"] = now
        batch = event.get("batch")
        total = event.get("total_batches")
        pct = 100 * batch / total if total else 0
        logger.info(
            "[%s] epoch=%d batch=%d/%d (%.1f%%) content=%.2e style=%.2e total=%.2e eta=%.0fs",
            style_id,
            event.get("epoch", 0),
            batch,
            total,
            pct,
            event.get("content_loss", 0),
            event.get("style_loss", 0),
            event.get("total_loss", 0),
            event.get("eta_s", 0),
        )

    return cb


def train_one(style_id: str, name: str, artist: str, cfg: dict) -> bool:
    # Lazy import so this script can show --help without loading torch.
    import torch

    from backend.core.model_manager import ModelManager
    from backend.core.training_engine import (
        TrainingArtifacts,
        TrainingConfig,
        TrainingEngine,
    )

    out_pth = MODELS_DIR / f"{style_id}.pth"
    if out_pth.exists():
        logger.info("[%s] already trained (%s exists), skipping", style_id, out_pth.name)
        return True

    style_img = STYLE_SRC / f"{style_id}.jpg"
    if not style_img.exists():
        logger.error("[%s] missing style image at %s", style_id, style_img)
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[%s] training on %s", style_id, device.type)

    # Use MODELS_DIR as trained_dir so batch-trained styles show up as
    # "pretrained" in the UI (served from backend/models/ at startup).
    artifacts = TrainingArtifacts(
        trained_dir=MODELS_DIR,
        thumbnails_dir=ROOT / "backend" / "thumbnails",
        dataset_root=str(DATA_DIR),
        project_root=ROOT,
    )
    mm = ModelManager(
        pretrained_dir=MODELS_DIR,
        trained_dir=MODELS_DIR,
        thumbnails_dir=ROOT / "backend" / "thumbnails",
        device=device,
    )
    engine = TrainingEngine(model_manager=mm, device=device, artifacts=artifacts)

    config = TrainingConfig(
        style_name=name,
        style_weight=cfg["style_weight"],
        content_weight=cfg["content_weight"],
        tv_weight=cfg["tv_weight"],
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        image_size=cfg["image_size"],
    )

    t0 = time.time()
    try:
        engine.start(config, style_img, on_event=_progress_cb(style_id))
    except Exception:
        logger.exception("[%s] failed to start", style_id)
        return False

    # Poll until the worker thread finishes.
    while engine.is_running():
        time.sleep(5)

    final = engine.get_status()
    state = final.get("state")
    if state != "done":
        logger.error("[%s] ended in state=%s error=%s", style_id, state, final.get("error"))
        return False

    if not out_pth.exists():
        logger.error("[%s] training finished but %s was not produced", style_id, out_pth)
        return False

    # Write metadata JSON next to the .pth (model_manager reads this).
    meta_path = MODELS_DIR / f"{style_id}.json"
    meta_path.write_text(json.dumps({"name": name, "artist": artist}, indent=2))

    dt = time.time() - t0
    logger.info("[%s] saved in %.1fs (%.1f min)", style_id, dt, dt / 60)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true", help="Use paper-quality preset (batch=4, slower)."
    )
    parser.add_argument("--only", nargs="+", help="Train only these style IDs.")
    parser.add_argument("--skip", nargs="+", help="Skip these style IDs.")
    args = parser.parse_args()

    cfg = PAPER_CFG if args.full else FAST_CFG
    logger.info(
        "preset: %s, dataset=%s, style_sources=%s",
        "paper" if args.full else "fast",
        DATA_DIR,
        STYLE_SRC,
    )

    if not DATA_DIR.exists() or not any(DATA_DIR.rglob("*.jpg")):
        logger.error(
            "No training images at %s. Run: python scripts/download_coco.py", DATA_DIR
        )
        return 1

    selected = list(STYLE_TITLES.keys())
    if args.only:
        selected = [s for s in selected if s in set(args.only)]
    if args.skip:
        selected = [s for s in selected if s not in set(args.skip)]

    logger.info("training %d styles: %s", len(selected), ", ".join(selected))

    t0 = time.time()
    ok = 0
    failed: list[str] = []
    for i, style_id in enumerate(selected, 1):
        name, artist = STYLE_TITLES[style_id]
        logger.info("=" * 60)
        logger.info("[%d/%d] %s — %s", i, len(selected), style_id, name)
        logger.info("=" * 60)
        if train_one(style_id, name, artist, cfg):
            ok += 1
        else:
            failed.append(style_id)

    total_dt = time.time() - t0
    logger.info(
        "batch complete: %d/%d ok in %.1f min. failed=%s",
        ok,
        len(selected),
        total_dt / 60,
        failed or "none",
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
