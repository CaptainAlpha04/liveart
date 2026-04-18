"""Video processing REST endpoints.

- ``POST /api/video/stylize`` (multipart, 500MB cap)
- ``GET  /api/video/status/{job_id}``
- ``GET  /api/video/download/{job_id}``
- ``DELETE /api/video/{job_id}``
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from ..schemas import VideoJobResponse, VideoJobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["video"])

MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500MB


@router.post("/stylize", response_model=VideoJobResponse)
async def video_stylize(
    file: UploadFile = File(...),
    style_id: str = Form(...),
) -> VideoJobResponse:
    from ..main import (  # type: ignore[attr-defined]
        model_manager,
        uploads_dir,
        video_processor,
    )

    if not model_manager.has(style_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown style: {style_id}",
        )

    uploads_dir_path: Path = uploads_dir
    uploads_dir_path.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex[:12]
    in_suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    input_path = uploads_dir_path / f"in_{token}{in_suffix}"
    output_path = uploads_dir_path / f"out_{token}.mp4"

    bytes_written = 0
    try:
        with input_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_VIDEO_BYTES:
                    out.close()
                    try:
                        input_path.unlink()
                    except Exception:  # noqa: BLE001
                        pass
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds {MAX_VIDEO_BYTES // (1024 * 1024)}MB cap",
                    )
                out.write(chunk)
    finally:
        await file.close()

    job_id = video_processor.submit(input_path, output_path, style_id)
    return VideoJobResponse(job_id=job_id)


@router.get("/status/{job_id}", response_model=VideoJobStatus)
async def video_status(job_id: str) -> VideoJobStatus:
    from ..main import video_registry  # type: ignore[attr-defined]

    job = video_registry.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    return VideoJobStatus(**job.to_dict())


@router.get("/download/{job_id}")
async def video_download(job_id: str) -> FileResponse:
    from ..main import video_registry  # type: ignore[attr-defined]

    job = video_registry.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    if job.status != "done":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job not complete (status={job.status})",
        )
    if not job.output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Output file missing",
        )
    return FileResponse(
        path=str(job.output_path),
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
    )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def video_delete(job_id: str) -> None:
    from ..main import video_registry  # type: ignore[attr-defined]

    job = video_registry.delete(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    for path in (job.input_path, job.output_path):
        try:
            if path.exists():
                path.unlink()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to delete job file: %s", path)
    return None
