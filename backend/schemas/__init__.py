"""Pydantic schemas for REST + WebSocket payloads.

Field shapes and defaults follow spec A4 exactly.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class StyleInfo(BaseModel):
    id: str
    name: str
    artist: str
    is_pretrained: bool
    thumbnail_url: Optional[str] = None


class TrainingConfig(BaseModel):
    style_name: str
    style_weight: float = Field(default=1e10)
    content_weight: float = Field(default=1e5)
    tv_weight: float = Field(default=1e-6)
    learning_rate: float = Field(default=1e-3)
    epochs: int = Field(default=2, ge=1)
    batch_size: int = Field(default=4, ge=1)
    image_size: int = Field(default=256, ge=64)


TrainingState = Literal["idle", "running", "done", "error"]


class TrainingStatus(BaseModel):
    state: TrainingState
    style_name: str = ""
    style_id: str = ""
    epoch: int = 0
    batch: int = 0
    total_batches: int = 0
    progress: float = 0.0
    error: Optional[str] = None


class VideoJobResponse(BaseModel):
    job_id: str


VideoJobState = Literal["queued", "processing", "done", "error"]


class VideoJobStatus(BaseModel):
    job_id: str
    status: VideoJobState
    progress: float = 0.0
    total_frames: int = 0
    processed_frames: int = 0
    elapsed_s: int = 0
    error: Optional[str] = None


class HealthStatus(BaseModel):
    status: str = "ok"
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    models_loaded: int = 0
    uptime_s: int = 0


__all__ = [
    "StyleInfo",
    "TrainingConfig",
    "TrainingStatus",
    "VideoJobResponse",
    "VideoJobStatus",
    "HealthStatus",
]
