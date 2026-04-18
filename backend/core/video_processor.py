"""OpenCV-based video stylization pipeline.

Decodes an input video with ``cv2.VideoCapture``, runs inference at native
resolution via the inference engine, and re-encodes with ``cv2.VideoWriter``
using the ``mp4v`` fourcc. One worker daemon thread drains a bounded queue so
multiple uploads are processed serially but without blocking the REST handler.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from .inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class VideoJob:
    job_id: str
    input_path: Path
    output_path: Path
    style_id: str
    status: str = "queued"  # queued | processing | done | error
    total_frames: int = 0
    processed_frames: int = 0
    started_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0
    error: Optional[str] = None

    def progress(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(1.0, self.processed_frames / self.total_frames)

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress(),
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "elapsed_s": int(self.elapsed_s),
            "error": self.error,
        }


class VideoJobRegistry:
    """Thread-safe registry of video jobs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: Dict[str, VideoJob] = {}

    def put(self, job: VideoJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Optional[VideoJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> Optional[VideoJob]:
        with self._lock:
            return self._jobs.pop(job_id, None)

    def all(self) -> Dict[str, VideoJob]:
        with self._lock:
            return dict(self._jobs)


class VideoProcessor:
    """Daemon-threaded video stylization worker."""

    def __init__(
        self,
        engine: InferenceEngine,
        registry: VideoJobRegistry,
        device: torch.device,
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.device = device
        self._queue: "queue.Queue[VideoJob]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop, name="video-processor", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, input_path: Path, output_path: Path, style_id: str) -> str:
        """Queue a new job and return its id."""
        job_id = uuid.uuid4().hex[:12]
        job = VideoJob(
            job_id=job_id,
            input_path=Path(input_path),
            output_path=Path(output_path),
            style_id=style_id,
        )
        self.registry.put(job)
        self._queue.put(job)
        return job_id

    def shutdown(self) -> None:
        self._stop.set()
        # Best-effort: push a sentinel so the thread wakes up.
        try:
            self._queue.put_nowait(None)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                self._process(job)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Video job %s failed", job.job_id)
                job.status = "error"
                job.error = str(exc)

    def _process(self, job: VideoJob) -> None:
        job.status = "processing"
        job.started_at = time.time()

        cap = cv2.VideoCapture(str(job.input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {job.input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        job.total_frames = max(total, 0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(job.output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open output writer: {job.output_path}")

        processed = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                # BGR -> RGB, HWC uint8 -> BCHW float32 in [-1, 1]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(frame_rgb.astype(np.float32))
                tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
                tensor = tensor / 127.5 - 1.0
                tensor = tensor.to(self.device, non_blocking=True)

                out_tensor = self.engine.stylize_tensor_sync(job.style_id, tensor)
                out_tensor = out_tensor.detach().clamp(-1.0, 1.0)
                out_tensor = (out_tensor + 1.0) * 127.5
                arr = out_tensor.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()

                out_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                writer.write(out_bgr)

                processed += 1
                if processed % 10 == 0:
                    job.processed_frames = processed
                    job.elapsed_s = time.time() - job.started_at
        finally:
            cap.release()
            writer.release()

        job.processed_frames = processed
        job.elapsed_s = time.time() - job.started_at
        job.status = "done"
