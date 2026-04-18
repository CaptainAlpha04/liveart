"""Async wrapper around the model manager.

A single-worker ``ThreadPoolExecutor`` serializes all GPU work so forward
passes never collide. ``stylize_b64`` decodes a base64 JPEG, runs the model,
re-encodes as JPEG (quality 85), and returns the result along with wall-clock
inference time in milliseconds. ``stylize_tensor`` is the hot path for the
video pipeline which already has a decoded tensor.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from .model_manager import ModelManager


def _b64_to_array(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8)


def _array_to_b64_jpeg(arr: np.ndarray, quality: int = 85) -> str:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _array_to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    # HWC uint8 -> BCHW float32 in [-1, 1]
    t = torch.from_numpy(arr.astype(np.float32))
    t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
    t = t / 127.5 - 1.0
    return t.to(device, non_blocking=True)


def _tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    # BCHW float32 in [-1, 1] -> HWC uint8
    t = tensor.detach().clamp(-1.0, 1.0)
    t = (t + 1.0) * 127.5
    t = t.squeeze(0).permute(1, 2, 0)
    return t.to(torch.uint8).cpu().numpy()


class InferenceEngine:
    """Serialized, async-friendly inference wrapper."""

    def __init__(self, model_manager: ModelManager, device: torch.device) -> None:
        self.model_manager = model_manager
        self.device = device
        # max_workers=1 keeps GPU access deterministic and avoids interleaving
        # forward passes that could otherwise trash each other's CUDA context.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

    # ------------------------------------------------------------------
    # Synchronous primitives (run on executor thread)
    # ------------------------------------------------------------------

    def _stylize_tensor_sync(self, style_id: str, tensor: torch.Tensor) -> torch.Tensor:
        if not self.model_manager.has(style_id):
            raise KeyError(style_id)
        model = self.model_manager.get_model(style_id)
        with torch.inference_mode():
            out = model(tensor)
        return out

    def _stylize_b64_sync(self, style_id: str, b64_jpeg: str) -> Tuple[str, float]:
        if not self.model_manager.has(style_id):
            raise KeyError(style_id)
        start = time.perf_counter()
        arr = _b64_to_array(b64_jpeg)
        tensor = _array_to_tensor(arr, self.device)
        out_tensor = self._stylize_tensor_sync(style_id, tensor)
        out_arr = _tensor_to_array(out_tensor)
        b64_out = _array_to_b64_jpeg(out_arr, quality=85)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return b64_out, elapsed_ms

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def stylize_b64(self, style_id: str, b64_jpeg: str) -> Tuple[str, float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, self._stylize_b64_sync, style_id, b64_jpeg
        )

    async def stylize_tensor(self, style_id: str, tensor: torch.Tensor) -> torch.Tensor:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, self._stylize_tensor_sync, style_id, tensor
        )

    # Sync variant for the video pipeline, which already runs on a worker thread.
    def stylize_tensor_sync(self, style_id: str, tensor: torch.Tensor) -> torch.Tensor:
        return self._stylize_tensor_sync(style_id, tensor)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
