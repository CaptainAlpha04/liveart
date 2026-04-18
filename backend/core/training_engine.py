"""Full Johnson et al. training loop for one style at a time.

Single-job state machine: only one training run active at any moment. Events
are emitted every 50 batches via an optional callback with the shape required
by the ``/ws/training`` WebSocket (see spec A4.1). A ``threading.Event`` allows
graceful early termination.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .model_manager import ModelManager
from .transform_net import TransformNet
from .vgg import VGGFeatures, gram_matrix

logger = logging.getLogger(__name__)

EventCallback = Callable[[Dict], None]


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class ImageFolderDataset(Dataset):
    """Simple recursive image dataset for arbitrary content corpora (e.g. COCO).

    Transforms: resize short-side -> center crop to ``image_size`` -> ToTensor
    -> scale to [-1, 1] to match the TransformNet input/output range.
    """

    def __init__(self, root: str, image_size: int = 256) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        self.paths: List[Path] = []
        for ext in IMAGE_EXTS:
            self.paths.extend(self.root.rglob(f"*{ext}"))
        self.paths.sort()
        if not self.paths:
            raise RuntimeError(f"No images (.jpg/.jpeg/.png) found under {self.root}")
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),  # -> [0, 1]
                transforms.Lambda(lambda t: t * 2.0 - 1.0),  # -> [-1, 1]
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)


def _load_style_tensor(style_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 2.0 - 1.0),
        ]
    )
    with Image.open(style_path) as img:
        img = img.convert("RGB")
        return tf(img).unsqueeze(0).to(device)


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    """Total variation loss: sum of squared differences between adjacent pixels."""
    diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
    return torch.sum(diff_h ** 2) + torch.sum(diff_w ** 2)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "custom_style"


@dataclass
class TrainingConfig:
    style_name: str
    style_weight: float = 1e10
    content_weight: float = 1e5
    tv_weight: float = 1e-6
    learning_rate: float = 1e-3
    epochs: int = 2
    batch_size: int = 4
    image_size: int = 256


@dataclass
class TrainingState:
    state: str = "idle"  # idle | running | done | error
    style_name: str = ""
    style_id: str = ""
    epoch: int = 0
    batch: int = 0
    total_batches: int = 0
    started_at: float = 0.0
    error: Optional[str] = None

    def progress(self) -> float:
        if self.total_batches <= 0:
            return 0.0
        return min(1.0, self.batch / self.total_batches)

    def to_dict(self) -> Dict:
        return {
            "state": self.state,
            "style_name": self.style_name,
            "style_id": self.style_id,
            "epoch": self.epoch,
            "batch": self.batch,
            "total_batches": self.total_batches,
            "progress": self.progress(),
            "error": self.error,
        }


@dataclass
class TrainingArtifacts:
    trained_dir: Path
    thumbnails_dir: Path
    # Optional override for the dataset root
    dataset_root: Optional[str] = None
    project_root: Optional[Path] = field(default=None)


class TrainingEngine:
    """Single-job training engine.

    ``start()`` spawns a daemon worker thread. ``request_stop()`` sets a stop
    event that the worker checks on every batch. ``is_running()`` reports
    whether the state machine is currently busy. ``get_status()`` returns a
    ``TrainingStatus``-shaped dict.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        device: torch.device,
        artifacts: TrainingArtifacts,
    ) -> None:
        self.model_manager = model_manager
        self.device = device
        self.artifacts = artifacts
        self._lock = threading.Lock()
        self._state = TrainingState()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        with self._lock:
            return self._state.state == "running"

    def get_status(self) -> Dict:
        with self._lock:
            return self._state.to_dict()

    def request_stop(self) -> None:
        self._stop_event.set()

    def start(
        self,
        config: TrainingConfig,
        style_image_path: Path,
        on_event: Optional[EventCallback] = None,
    ) -> str:
        """Kick off a new training job. Returns the derived ``style_id``."""
        with self._lock:
            if self._state.state == "running":
                raise RuntimeError("Training already in progress")
            style_id = slugify(config.style_name)
            self._state = TrainingState(
                state="running",
                style_name=config.style_name,
                style_id=style_id,
                epoch=0,
                batch=0,
                total_batches=0,
                started_at=time.time(),
                error=None,
            )
            self._stop_event.clear()

        self._worker = threading.Thread(
            target=self._run,
            args=(config, Path(style_image_path), on_event, style_id),
            name=f"train-{style_id}",
            daemon=True,
        )
        self._worker.start()
        return style_id

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(
        self,
        config: TrainingConfig,
        style_image_path: Path,
        on_event: Optional[EventCallback],
        style_id: str,
    ) -> None:
        try:
            self._run_inner(config, style_image_path, on_event, style_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training failed for %s", style_id)
            with self._lock:
                self._state.state = "error"
                self._state.error = str(exc)
            if on_event is not None:
                try:
                    on_event(
                        {
                            "status": "error",
                            "error": str(exc),
                            "style_id": style_id,
                        }
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("on_event failed during error notification")

    def _run_inner(
        self,
        config: TrainingConfig,
        style_image_path: Path,
        on_event: Optional[EventCallback],
        style_id: str,
    ) -> None:
        device = self.device
        started = time.time()

        # --- Dataset -----------------------------------------------------
        dataset_root = self.artifacts.dataset_root or os.environ.get("COCO_TRAIN_DIR")
        if not dataset_root:
            root = self.artifacts.project_root or Path.cwd()
            dataset_root = str(root / "data" / "coco_train")
        dataset = ImageFolderDataset(dataset_root, image_size=config.image_size)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )
        batches_per_epoch = len(loader)
        total_batches = batches_per_epoch * config.epochs

        with self._lock:
            self._state.total_batches = total_batches

        # --- Models ------------------------------------------------------
        transformer = TransformNet().to(device)
        transformer.train()
        vgg = VGGFeatures().to(device)
        vgg.eval()

        # --- Style features (fixed throughout training) ------------------
        style_tensor = _load_style_tensor(style_image_path, config.image_size, device)
        with torch.no_grad():
            style_features = vgg(style_tensor)
        # Pre-compute gram matrices of the style image once per-layer.
        style_grams = [gram_matrix(f).detach() for f in style_features]

        mse = nn.MSELoss()
        optimizer = optim.Adam(transformer.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        global_batch = 0
        for epoch in range(1, config.epochs + 1):
            with self._lock:
                self._state.epoch = epoch

            for batch_idx, batch in enumerate(loader, start=1):
                if self._stop_event.is_set():
                    logger.info("Training stop requested at epoch=%d batch=%d", epoch, batch_idx)
                    with self._lock:
                        self._state.state = "done"
                    if on_event is not None:
                        try:
                            on_event(
                                {
                                    "status": "stopped",
                                    "style_id": style_id,
                                    "elapsed_s": int(time.time() - started),
                                }
                            )
                        except Exception:  # noqa: BLE001
                            logger.exception("on_event failed during stop notification")
                    return

                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                output = transformer(batch)

                output_features = vgg(output)
                content_features = vgg(batch)

                # Content loss uses relu3_3 -> index 2 in the VGGOutputs tuple
                # (layers are [relu1_2, relu2_2, relu3_3, relu4_3]).
                content_loss = config.content_weight * mse(
                    output_features[2], content_features[2].detach()
                )

                style_loss_val = torch.zeros((), device=device)
                for out_feat, style_gram in zip(output_features, style_grams):
                    out_gram = gram_matrix(out_feat)
                    # Broadcast style gram over batch if needed
                    if out_gram.shape[0] != style_gram.shape[0]:
                        style_gram_bc = style_gram.expand(out_gram.shape[0], -1, -1)
                    else:
                        style_gram_bc = style_gram
                    style_loss_val = style_loss_val + mse(out_gram, style_gram_bc)
                style_loss = config.style_weight * style_loss_val

                tv_loss = config.tv_weight * _total_variation(output)

                total_loss = content_loss + style_loss + tv_loss
                total_loss.backward()
                optimizer.step()

                global_batch += 1
                with self._lock:
                    self._state.batch = global_batch

                if on_event is not None and (global_batch % 50 == 0 or global_batch == 1):
                    elapsed = time.time() - started
                    eta = 0.0
                    if global_batch > 0 and total_batches > 0:
                        per_batch = elapsed / global_batch
                        eta = max(0.0, per_batch * (total_batches - global_batch))
                    event = {
                        "epoch": epoch,
                        "batch": global_batch,
                        "total_batches": total_batches,
                        "content_loss": float(content_loss.item()),
                        "style_loss": float(style_loss.item()),
                        "tv_loss": float(tv_loss.item()),
                        "total_loss": float(total_loss.item()),
                        "elapsed_s": int(elapsed),
                        "eta_s": int(eta),
                        "status": "running",
                    }
                    try:
                        on_event(event)
                    except Exception:  # noqa: BLE001
                        logger.exception("on_event failed during progress notification")

            # End of epoch -- decay LR
            scheduler.step()

        # --- Save model + metadata --------------------------------------
        transformer.eval()
        self.artifacts.trained_dir.mkdir(parents=True, exist_ok=True)
        weights_path = self.artifacts.trained_dir / f"{style_id}.pth"
        torch.save(transformer.state_dict(), weights_path)

        meta_path = weights_path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(
                {
                    "name": config.style_name,
                    "artist": "User",
                    "style_id": style_id,
                    "trained_at": time.time(),
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "style_weight": config.style_weight,
                    "content_weight": config.content_weight,
                    "tv_weight": config.tv_weight,
                    "image_size": config.image_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Register with manager so it becomes immediately available for inference.
        self.model_manager.register_trained(
            style_id,
            weights_path,
            name=config.style_name,
            artist="User",
        )

        elapsed = int(time.time() - started)
        with self._lock:
            self._state.state = "done"
        if on_event is not None:
            try:
                on_event(
                    {
                        "status": "done",
                        "model_id": style_id,
                        "style_id": style_id,
                        "elapsed_s": elapsed,
                    }
                )
            except Exception:  # noqa: BLE001
                logger.exception("on_event failed during completion notification")
