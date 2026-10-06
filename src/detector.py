"""Detection engine: runs the YOLO models and returns person / helmet / no_helmet detections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import Config
from src.utils import Box

PERSON = "person"
HELMET = "helmet"
NO_HELMET = "no_helmet"


@dataclass
class Detection:
    box: Box
    label: str          # person | helmet | no_helmet
    confidence: float
    class_name: str     # Original class name from the model


class ModelLoadError(Exception):
    """A required model file is missing or cannot be loaded."""


def map_class(class_name: str, config: Config) -> str | None:
    """Translate a model's class name into person / helmet / no_helmet (or None to ignore)."""
    name = class_name.strip().lower()
    if name in (item.lower() for item in config.person_classes):
        return PERSON
    if name in (item.lower() for item in config.helmet_classes):
        return HELMET
    if name in (item.lower() for item in config.no_helmet_classes):
        return NO_HELMET
    return None


def threshold_for(label: str, config: Config) -> float:
    return {
        PERSON: config.person_confidence,
        HELMET: config.helmet_confidence,
        NO_HELMET: config.no_helmet_confidence,
    }[label]


def parse_detections(boxes, confidences, class_ids, names: dict, config: Config,
                     allowed: set[str]) -> list[Detection]:
    """Turn raw model output into filtered Detection objects.

    Kept separate from the model so it can be unit-tested without YOLO.
    """
    detections = []

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        class_name = str(names.get(int(class_id), class_id))
        label = map_class(class_name, config)

        if label is None or label not in allowed:
            continue
        if float(confidence) < threshold_for(label, config):
            continue

        x1, y1, x2, y2 = (float(value) for value in box)
        detections.append(Detection((x1, y1, x2, y2), label, float(confidence), class_name))

    return detections


def model_labels(names: dict, config: Config) -> set[str]:
    return {label for label in (map_class(str(name), config) for name in names.values()) if label}


class Detector:
    """Loads the helmet model (and a person model if needed) and runs inference on frames."""

    def __init__(self, config: Config):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ModelLoadError("The 'ultralytics' package is not installed. Run: pip install -r requirements.txt") from error

        self.config = config
        self.warnings: list[str] = []
        self.helmet_model = None
        self.helmet_labels: set[str] = set()
        self.person_model = None

        helmet_path = Path(config.helmet_model)
        if helmet_path.exists():
            try:
                self.helmet_model = YOLO(str(helmet_path))
            except Exception as error:  # noqa: BLE001 - any loading failure is reported to the user
                raise ModelLoadError(f"Cannot load helmet model {helmet_path}: {error}") from error

            self.helmet_labels = model_labels(self.helmet_model.names, config)
            if not self.helmet_labels & {HELMET, NO_HELMET}:
                raise ModelLoadError(
                    f"Helmet model classes {list(self.helmet_model.names.values())} do not match "
                    "helmet_classes / no_helmet_classes in config.yaml."
                )
        else:
            self.warnings.append(
                f"Helmet model not found ({helmet_path}). Showing people only; every status is UNKNOWN. "
                "Run: python scripts/download_model.py"
            )

        # A separate person model is only needed if the helmet model cannot detect people itself.
        if PERSON not in self.helmet_labels:
            try:
                self.person_model = YOLO(config.person_model)
            except Exception as error:  # noqa: BLE001
                raise ModelLoadError(f"Cannot load person model {config.person_model}: {error}") from error

    def _run(self, model, frame: np.ndarray, allowed: set[str]) -> list[Detection]:
        lowest = min(threshold_for(label, self.config) for label in allowed)
        result = model.predict(
            frame,
            conf=lowest,
            imgsz=self.config.image_size,
            device=self.config.device or None,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        return parse_detections(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            result.names,
            self.config,
            allowed,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []

        if self.helmet_model is not None:
            detections += self._run(self.helmet_model, frame, self.helmet_labels)

        if self.person_model is not None:
            detections += self._run(self.person_model, frame, {PERSON})

        return detections
