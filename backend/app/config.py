"""Reglages de l'API, lus dans les variables d'environnement (12-factor)."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CLASSES = [
    "call",
    "dislike",
    "fist",
    "four",
    "like",
    "mute",
    "ok",
    "one",
    "palm",
    "peace",
    "peace_inverted",
    "rock",
    "stop",
    "stop_inverted",
    "three",
    "three2",
    "two_up",
    "two_up_inverted",
    "no_gesture",
]


@dataclass
class Settings:
    model_path: Path
    metadata_path: Path
    conf_threshold: float
    iou_threshold: float
    max_upload_mb: float
    cors_origins: list = field(default_factory=list)


def get_settings():
    return Settings(
        model_path=Path(os.getenv("MODEL_PATH", "models/touchfree.onnx")),
        metadata_path=Path(os.getenv("METADATA_PATH", "models/metadata.json")),
        conf_threshold=float(os.getenv("CONF_THRESHOLD", "0.35")),
        iou_threshold=float(os.getenv("IOU_THRESHOLD", "0.5")),
        max_upload_mb=float(os.getenv("MAX_UPLOAD_MB", "8")),
        cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    )


def load_metadata(path):
    """metadata.json est ecrit par scripts/promote.py a cote du modele."""
    base = {"names": DEFAULT_CLASSES, "imgsz": 512, "variant": "inconnu"}
    if Path(path).exists():
        base.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return base
