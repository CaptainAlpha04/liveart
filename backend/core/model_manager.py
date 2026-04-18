"""Thread-safe registry of TransformNet models.

Loads ``.pth`` files from two directories at startup:
- ``backend/models/``          -> pretrained / immutable styles
- ``backend/trained_models/``  -> user-trained / deletable styles

Optional ``<style_id>.json`` metadata files (``name``, ``artist``) may live
alongside the weights. Missing metadata falls back to a humanized filename and
``"Unknown"`` artist.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .transform_net import TransformNet

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    style_id: str
    name: str
    artist: str
    is_pretrained: bool
    path: Path
    model: TransformNet


def _humanize(style_id: str) -> str:
    return " ".join(word.capitalize() for word in style_id.replace("-", "_").split("_"))


def _thumbnail_filename(style_id: str, thumbnails_dir: Path) -> Optional[str]:
    for ext in (".jpg", ".jpeg", ".png"):
        if (thumbnails_dir / f"{style_id}{ext}").exists():
            return f"{style_id}{ext}"
    return None


class ModelManager:
    """Thread-safe model registry."""

    def __init__(
        self,
        pretrained_dir: Path,
        trained_dir: Path,
        thumbnails_dir: Path,
        device: torch.device,
    ) -> None:
        self.pretrained_dir = Path(pretrained_dir)
        self.trained_dir = Path(trained_dir)
        self.thumbnails_dir = Path(thumbnails_dir)
        self.device = device
        self._lock = threading.RLock()
        self._entries: Dict[str, ModelEntry] = {}

    # ------------------------------------------------------------------
    # Startup / loading
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Scan both directories and load every ``.pth`` found."""
        with self._lock:
            self._entries.clear()
            for directory, is_pretrained in (
                (self.pretrained_dir, True),
                (self.trained_dir, False),
            ):
                directory.mkdir(parents=True, exist_ok=True)
                for pth_path in sorted(directory.glob("*.pth")):
                    try:
                        self._load_one(pth_path, is_pretrained=is_pretrained)
                    except Exception:  # noqa: BLE001
                        logger.exception("Failed to load model %s", pth_path)
        logger.info("ModelManager loaded %d models", len(self._entries))

    def _load_one(self, pth_path: Path, *, is_pretrained: bool) -> None:
        style_id = pth_path.stem
        model = TransformNet()
        state_dict = torch.load(str(pth_path), map_location=self.device, weights_only=True)
        # Some community weights wrap their state dict in {"state_dict": ...}
        if isinstance(state_dict, dict) and "state_dict" in state_dict and all(
            not hasattr(v, "shape") for v in state_dict.values() if not isinstance(v, dict)
        ):
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()

        meta_path = pth_path.with_suffix(".json")
        name = _humanize(style_id)
        artist = "Unknown"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                name = meta.get("name", name)
                artist = meta.get("artist", artist)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to read metadata at %s", meta_path)

        self._entries[style_id] = ModelEntry(
            style_id=style_id,
            name=name,
            artist=artist,
            is_pretrained=is_pretrained,
            path=pth_path,
            model=model,
        )
        logger.info("Loaded model %s (pretrained=%s)", style_id, is_pretrained)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has(self, style_id: str) -> bool:
        with self._lock:
            return style_id in self._entries

    def get_model(self, style_id: str) -> TransformNet:
        with self._lock:
            entry = self._entries.get(style_id)
            if entry is None:
                raise KeyError(style_id)
            return entry.model

    def get_entry(self, style_id: str) -> ModelEntry:
        with self._lock:
            entry = self._entries.get(style_id)
            if entry is None:
                raise KeyError(style_id)
            return entry

    def list_styles(self) -> List[dict]:
        """Return a serializable description of all styles currently loaded."""
        with self._lock:
            result: List[dict] = []
            for style_id, entry in sorted(self._entries.items()):
                thumb = _thumbnail_filename(style_id, self.thumbnails_dir)
                result.append(
                    {
                        "id": style_id,
                        "name": entry.name,
                        "artist": entry.artist,
                        "is_pretrained": entry.is_pretrained,
                        "thumbnail_url": f"/api/styles/{style_id}/thumbnail"
                        if thumb is not None
                        else None,
                    }
                )
            return result

    def register_trained(
        self,
        style_id: str,
        weights_path: Path,
        *,
        name: Optional[str] = None,
        artist: Optional[str] = None,
    ) -> ModelEntry:
        """Load a freshly trained model into the registry."""
        weights_path = Path(weights_path)
        with self._lock:
            model = TransformNet()
            state_dict = torch.load(
                str(weights_path), map_location=self.device, weights_only=True
            )
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()

            resolved_name = name or _humanize(style_id)
            resolved_artist = artist or "User"

            # Write/refresh metadata JSON next to the weights so it survives restarts.
            meta_path = weights_path.with_suffix(".json")
            try:
                meta_path.write_text(
                    json.dumps({"name": resolved_name, "artist": resolved_artist}, indent=2),
                    encoding="utf-8",
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to persist metadata for %s", style_id)

            entry = ModelEntry(
                style_id=style_id,
                name=resolved_name,
                artist=resolved_artist,
                is_pretrained=False,
                path=weights_path,
                model=model,
            )
            self._entries[style_id] = entry
            return entry

    def delete(self, style_id: str) -> None:
        """Delete a user-trained model. Pretrained models cannot be deleted."""
        with self._lock:
            entry = self._entries.get(style_id)
            if entry is None:
                raise KeyError(style_id)
            if entry.is_pretrained:
                raise PermissionError(f"Cannot delete pretrained model: {style_id}")
            try:
                if entry.path.exists():
                    entry.path.unlink()
                meta = entry.path.with_suffix(".json")
                if meta.exists():
                    meta.unlink()
            except Exception:  # noqa: BLE001
                logger.exception("Error deleting model files for %s", style_id)
            self._entries.pop(style_id, None)
