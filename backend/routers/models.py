"""Styles / models REST endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from ..schemas import StyleInfo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


def _list_styles() -> List[StyleInfo]:
    from ..main import model_manager  # type: ignore[attr-defined]

    return [StyleInfo(**s) for s in model_manager.list_styles()]


@router.get("/api/styles", response_model=List[StyleInfo])
async def list_styles() -> List[StyleInfo]:
    return _list_styles()


@router.get("/api/models", response_model=List[StyleInfo])
async def list_models() -> List[StyleInfo]:
    return _list_styles()


@router.get("/api/styles/{style_id}/thumbnail")
async def get_thumbnail(style_id: str) -> FileResponse:
    from ..main import model_manager, thumbnails_dir  # type: ignore[attr-defined]

    if not model_manager.has(style_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown style: {style_id}",
        )
    thumbnails_dir_path: Path = thumbnails_dir
    for ext, media in ((".jpg", "image/jpeg"), (".jpeg", "image/jpeg"), (".png", "image/png")):
        candidate = thumbnails_dir_path / f"{style_id}{ext}"
        if candidate.exists():
            return FileResponse(path=str(candidate), media_type=media)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No thumbnail for style: {style_id}",
    )


@router.post("/api/models/refresh", response_model=List[StyleInfo])
async def refresh_models() -> List[StyleInfo]:
    """Rescan the models directories and reload every `.pth` from disk.

    Useful after dropping new models into ``backend/models/`` (e.g. after
    running ``scripts/train_all_styles.py``) without restarting the server,
    and for flushing stale in-memory entries whose files have been removed.
    """
    from ..main import model_manager  # type: ignore[attr-defined]

    model_manager.load_all()
    return _list_styles()


@router.delete("/api/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(model_id: str) -> None:
    from ..main import model_manager  # type: ignore[attr-defined]

    try:
        model_manager.delete(model_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown model: {model_id}",
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    return None
