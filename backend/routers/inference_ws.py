"""WebSocket inference endpoint at ``/ws/inference``.

Per-connection frame dropping: while a frame is being processed we skip any
incoming frames (don't queue, don't await). This keeps perceived latency near
the inference time itself rather than queue depth * inference time.

FPS is computed from a rolling 1-second window of frame-completion timestamps.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/inference")
async def inference_ws(websocket: WebSocket) -> None:
    # Lazy import -- main.py imports routers, so top-level import would cycle.
    from ..main import engine  # type: ignore[attr-defined]

    await websocket.accept()
    processing = False
    fps_window: Deque[float] = deque()

    try:
        while True:
            message = await websocket.receive_json()
            style_id = message.get("style")
            frame_b64 = message.get("frame")

            if not style_id or not frame_b64:
                await websocket.send_json(
                    {"error": "bad_request", "detail": "style and frame are required"}
                )
                continue

            # Frame-drop: if we're still working on the previous frame, discard
            # this one entirely so we always process the freshest frame next.
            if processing:
                continue

            processing = True
            try:
                try:
                    out_b64, elapsed_ms = await engine.stylize_b64(style_id, frame_b64)
                except KeyError:
                    await websocket.send_json(
                        {
                            "error": "style_not_found",
                            "detail": f"No model loaded for style: {style_id}",
                        }
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Inference failed")
                    await websocket.send_json(
                        {"error": "inference_failed", "detail": str(exc)}
                    )
                    continue

                # Rolling 1-second FPS window
                now = time.perf_counter()
                fps_window.append(now)
                while fps_window and now - fps_window[0] > 1.0:
                    fps_window.popleft()
                fps = float(len(fps_window))

                await websocket.send_json(
                    {
                        "frame": out_b64,
                        "inference_ms": int(round(elapsed_ms)),
                        "fps": fps,
                        "style": style_id,
                    }
                )
            finally:
                processing = False
    except WebSocketDisconnect:
        logger.info("Inference WS disconnected")
    except Exception:  # noqa: BLE001
        logger.exception("Inference WS error")
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass
