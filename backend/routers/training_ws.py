"""Training WebSocket endpoint at ``/ws/training`` + event broadcaster.

``TrainingEventBroadcaster`` bridges the training thread (sync) to the asyncio
event loop (async) by capturing the loop at FastAPI startup and using
``asyncio.run_coroutine_threadsafe`` inside ``emit_threadsafe``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class TrainingEventBroadcaster:
    """Fan-out of training events to all connected WebSocket clients."""

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the running event loop -- called once from lifespan."""
        self._loop = loop

    async def register(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def unregister(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def _broadcast(self, event: Dict[str, Any]) -> None:
        # Snapshot under lock, then send without holding it.
        async with self._lock:
            clients = list(self._clients)
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(event)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to send training event; dropping client")
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    def emit_threadsafe(self, event: Dict[str, Any]) -> None:
        """Schedule a broadcast from any (non-async) thread."""
        loop = self._loop
        if loop is None:
            logger.warning("TrainingEventBroadcaster has no loop; dropping event")
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(event), loop)
        except Exception:  # noqa: BLE001
            logger.exception("emit_threadsafe failed to schedule coroutine")


@router.websocket("/ws/training")
async def training_ws(websocket: WebSocket) -> None:
    from ..main import training_broadcaster  # type: ignore[attr-defined]

    await websocket.accept()
    await training_broadcaster.register(websocket)
    try:
        while True:
            # We don't expect messages from the client, but keeping the
            # connection alive requires receiving. Any received message is
            # ignored; a disconnect raises WebSocketDisconnect.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.exception("Training WS error")
    finally:
        await training_broadcaster.unregister(websocket)
