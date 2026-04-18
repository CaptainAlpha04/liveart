"""Training REST endpoints: ``/api/training/{start, stop, status, style-sources}``."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ..core.training_engine import TrainingConfig as EngineTrainingConfig
from ..core.training_engine import slugify
from ..schemas import TrainingConfig, TrainingStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])


# Human-readable labels for the curated artwork references. Keyed by the
# ``<style_id>.jpg`` filename stems in ``backend/style_sources/``.
_STYLE_SOURCE_LABELS: dict[str, tuple[str, str]] = {
    "starry_night": ("Starry Night", "Van Gogh"),
    "the_scream": ("The Scream", "Edvard Munch"),
    "great_wave": ("The Great Wave", "Hokusai"),
    "composition_viii": ("Composition VIII", "Kandinsky"),
    "udnie": ("Udnie", "Francis Picabia"),
    "la_muse": ("La Muse / Cubist Muse", "Cézanne / Braque"),
    "mosaic": ("Byzantine Mosaic", ""),
    "candy": ("Candy", "Fast-Style reference"),
    "feathers": ("Decorative Pattern", "Klimt / Morris"),
    "rain_princess": ("Rain Princess", "Afremov / Monet"),
}


@router.get("/style-sources")
async def list_style_sources() -> list[dict]:
    """List curated reference artworks available as training sources.

    Returns one entry per ``<style_id>.jpg`` file present in the
    ``backend/style_sources/`` directory. The frontend uses this to populate
    the "Choose a predefined style" picker on the training page.
    """
    from ..main import style_sources_dir  # type: ignore[attr-defined]

    if not style_sources_dir.exists():
        return []

    results: list[dict] = []
    for path in sorted(style_sources_dir.glob("*.jpg")):
        style_id = path.stem
        name, artist = _STYLE_SOURCE_LABELS.get(
            style_id, (style_id.replace("_", " ").title(), "")
        )
        results.append(
            {
                "id": style_id,
                "name": name,
                "artist": artist,
                "image_url": f"/api/training/style-sources/{style_id}/image",
            }
        )
    return results


@router.get("/style-sources/{source_id}/image")
async def style_source_image(source_id: str):
    """Serve a style-source reference image (used by the training UI gallery)."""
    from fastapi.responses import FileResponse

    from ..main import style_sources_dir  # type: ignore[attr-defined]

    candidate = style_sources_dir / f"{source_id}.jpg"
    if not candidate.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No style source: {source_id}",
        )
    return FileResponse(candidate, media_type="image/jpeg")


@router.post("/start")
async def training_start(
    config: str = Form(...),
    style_image: Optional[UploadFile] = File(None),
    style_source_id: Optional[str] = Form(None),
) -> dict:
    """Start a training run.

    Exactly one of ``style_image`` (user upload) or ``style_source_id``
    (curated reference in ``backend/style_sources/``) must be provided.
    """
    from ..main import (  # type: ignore[attr-defined]
        training_engine,
        training_broadcaster,
        thumbnails_dir,
        uploads_dir,
        style_sources_dir,
    )

    if (style_image is None) == (style_source_id is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide exactly one of 'style_image' (upload) or "
            "'style_source_id' (predefined reference).",
        )

    try:
        cfg_dict = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid config JSON: {exc}",
        ) from exc

    try:
        cfg = TrainingConfig(**cfg_dict)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid config: {exc}",
        ) from exc

    if training_engine.is_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training already in progress",
        )

    style_id = slugify(cfg.style_name)

    uploads_dir_path: Path = uploads_dir
    thumbnails_dir_path: Path = thumbnails_dir
    uploads_dir_path.mkdir(parents=True, exist_ok=True)
    thumbnails_dir_path.mkdir(parents=True, exist_ok=True)

    if style_source_id is not None:
        # Use the curated artwork from backend/style_sources/.
        source_path = style_sources_dir / f"{style_source_id}.jpg"
        if not source_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown style_source_id: {style_source_id}",
            )
        suffix = ".jpg"
        style_image_path = uploads_dir_path / f"style_{style_id}{suffix}"
        shutil.copyfile(source_path, style_image_path)
    else:
        assert style_image is not None  # narrowed by the XOR check above
        suffix = Path(style_image.filename or "").suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png"):
            suffix = ".jpg"
        style_image_path = uploads_dir_path / f"style_{style_id}{suffix}"
        with style_image_path.open("wb") as f:
            shutil.copyfileobj(style_image.file, f)

    thumbnail_path = thumbnails_dir_path / f"{style_id}{suffix}"
    try:
        shutil.copyfile(style_image_path, thumbnail_path)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to copy thumbnail for %s", style_id)

    engine_cfg = EngineTrainingConfig(
        style_name=cfg.style_name,
        style_weight=cfg.style_weight,
        content_weight=cfg.content_weight,
        tv_weight=cfg.tv_weight,
        learning_rate=cfg.learning_rate,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
    )

    def on_event(event: dict) -> None:
        training_broadcaster.emit_threadsafe(event)

    try:
        job_id = training_engine.start(engine_cfg, style_image_path, on_event=on_event)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return {"job_id": job_id}


@router.post("/stop")
async def training_stop() -> dict:
    from ..main import training_engine  # type: ignore[attr-defined]

    training_engine.request_stop()
    return {"status": "stopped"}


@router.get("/status", response_model=TrainingStatus)
async def training_status() -> TrainingStatus:
    from ..main import training_engine  # type: ignore[attr-defined]

    state = training_engine.get_status()
    return TrainingStatus(
        state=state.get("state", "idle"),
        style_name=state.get("style_name", ""),
        style_id=state.get("style_id", ""),
        epoch=int(state.get("epoch", 0)),
        batch=int(state.get("batch", 0)),
        total_batches=int(state.get("total_batches", 0)),
        progress=float(state.get("progress", 0.0)),
        error=state.get("error"),
    )
