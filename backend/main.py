"""FastAPI app entry point.

Creates singletons inside a ``@asynccontextmanager`` lifespan, captures the
running asyncio event loop for the training broadcaster, registers routers and
CORS middleware, and exposes ``/health``. Singletons are module-level so
routers can ``from ..main import engine`` lazily from inside handlers (the
top-level imports would cycle because this module imports the routers).
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.inference_engine import InferenceEngine
from .core.model_manager import ModelManager
from .core.training_engine import TrainingArtifacts, TrainingEngine
from .core.video_processor import VideoJobRegistry, VideoProcessor
from .routers import inference_ws as inference_ws_router
from .routers import models as models_router
from .routers import training as training_router
from .routers import training_ws as training_ws_router
from .routers import video as video_router
from .routers.training_ws import TrainingEventBroadcaster
from .schemas import HealthStatus

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
PRETRAINED_DIR = BACKEND_DIR / "models"
TRAINED_DIR = BACKEND_DIR / "trained_models"
THUMBNAILS_DIR = BACKEND_DIR / "thumbnails"
UPLOADS_DIR = BACKEND_DIR / "uploads"
STYLE_SOURCES_DIR = BACKEND_DIR / "style_sources"

# --------------------------------------------------------------------------
# Module-level singletons (populated in lifespan).
#
# Routers import these lazily inside handler functions to avoid an import
# cycle: this module imports the routers, so they cannot import us at module
# load time.
# --------------------------------------------------------------------------

device: torch.device = torch.device("cpu")
model_manager: Optional[ModelManager] = None  # type: ignore[assignment]
engine: Optional[InferenceEngine] = None  # type: ignore[assignment]
training_engine: Optional[TrainingEngine] = None  # type: ignore[assignment]
video_registry: Optional[VideoJobRegistry] = None  # type: ignore[assignment]
video_processor: Optional[VideoProcessor] = None  # type: ignore[assignment]
training_broadcaster: TrainingEventBroadcaster = TrainingEventBroadcaster()

thumbnails_dir: Path = THUMBNAILS_DIR
uploads_dir: Path = UPLOADS_DIR
trained_dir: Path = TRAINED_DIR
pretrained_dir: Path = PRETRAINED_DIR
style_sources_dir: Path = STYLE_SOURCES_DIR

_started_at: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global device, model_manager, engine, training_engine
    global video_registry, video_processor, _started_at

    _started_at = time.time()

    # Ensure directories exist before anything tries to scan them.
    for d in (PRETRAINED_DIR, TRAINED_DIR, THUMBNAILS_DIR, UPLOADS_DIR, STYLE_SOURCES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Device autodetection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available; running on CPU")

    # Singletons
    model_manager = ModelManager(
        pretrained_dir=PRETRAINED_DIR,
        trained_dir=TRAINED_DIR,
        thumbnails_dir=THUMBNAILS_DIR,
        device=device,
    )
    model_manager.load_all()

    engine = InferenceEngine(model_manager=model_manager, device=device)

    training_engine = TrainingEngine(
        model_manager=model_manager,
        device=device,
        artifacts=TrainingArtifacts(
            trained_dir=TRAINED_DIR,
            thumbnails_dir=THUMBNAILS_DIR,
            project_root=PROJECT_ROOT,
        ),
    )

    video_registry = VideoJobRegistry()
    video_processor = VideoProcessor(
        engine=engine,
        registry=video_registry,
        device=device,
    )

    # Capture the running event loop so training threads can schedule coroutines.
    training_broadcaster.set_loop(asyncio.get_running_loop())

    logger.info(
        "LiveArt ready: device=%s models=%d", device, len(model_manager.list_styles())
    )

    try:
        yield
    finally:
        logger.info("Shutting down LiveArt")
        try:
            if engine is not None:
                engine.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("Error shutting down inference engine")
        try:
            if video_processor is not None:
                video_processor.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("Error shutting down video processor")


app = FastAPI(title="LiveArt", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(inference_ws_router.router)
app.include_router(training_ws_router.router)
app.include_router(training_router.router)
app.include_router(video_router.router)
app.include_router(models_router.router)


@app.get("/health", response_model=HealthStatus)
async def health() -> HealthStatus:
    global model_manager, _started_at
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    models_loaded = len(model_manager.list_styles()) if model_manager is not None else 0
    uptime = int(time.time() - _started_at) if _started_at else 0
    return HealthStatus(
        status="ok",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        models_loaded=models_loaded,
        uptime_s=uptime,
    )
