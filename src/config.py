"""Configuration manager: loads, validates and saves config.yaml."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class Config:
    # Input
    source: str = "0"
    camera_id: str = "camera-1"
    capture_width: int = 1280
    capture_height: int = 720
    reconnect_attempts: int = 5

    # Models
    person_model: str = "yolo11n.pt"
    helmet_model: str = "models/helmet_model.pt"
    device: str = ""
    image_size: int = 640
    person_classes: list[str] = field(default_factory=lambda: ["person"])
    helmet_classes: list[str] = field(default_factory=lambda: ["helmet", "hardhat", "hard-hat"])
    no_helmet_classes: list[str] = field(
        default_factory=lambda: ["no_helmet", "no-helmet", "no-hardhat", "head"]
    )

    # Thresholds
    person_confidence: float = 0.5
    helmet_confidence: float = 0.45
    no_helmet_confidence: float = 0.5
    head_region_ratio: float = 0.35
    association_min_overlap: float = 0.5
    association_margin: float = 0.15
    derive_no_helmet: bool = False

    # Temporal stability
    smoothing_window: int = 15
    status_min_ratio: float = 0.6
    max_missed_frames: int = 30
    tracker_iou: float = 0.3

    # Alerts
    alerts_enabled: bool = True
    alert_sound: bool = True
    alert_stable_frames: int = 10
    alert_cooldown_seconds: float = 10.0

    # Logging and snapshots
    logging_enabled: bool = True
    log_dir: str = "logs"
    log_retention_days: int = 30
    snapshots_enabled: bool = False
    snapshot_dir: str = "snapshots"
    snapshot_retention_days: int = 7

    # Video recording
    recording_enabled: bool = False
    recording_dir: str = "recordings"
    recording_retention_days: int = 7
    alert_recording_enabled: bool = True
    alert_record_pre_seconds: float = 3.0
    alert_record_post_seconds: float = 5.0
    alert_record_max_seconds: float = 60.0

    # Web dashboard
    web_port: int = 8000
    phone_port: int = 8443
    saved_cameras: list[str] = field(default_factory=list)

    # Display
    display_width: int = 1280

    def validate(self) -> list[str]:
        """Return a list of problems; an empty list means the configuration is usable."""
        problems = []

        for name in ("person_confidence", "helmet_confidence", "no_helmet_confidence",
                     "head_region_ratio", "association_min_overlap", "status_min_ratio", "tracker_iou"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                problems.append(f"{name} must be between 0 and 1 (got {value}).")

        for name in ("smoothing_window", "alert_stable_frames", "image_size", "display_width"):
            if getattr(self, name) < 1:
                problems.append(f"{name} must be at least 1.")

        for name in ("max_missed_frames", "reconnect_attempts", "log_retention_days",
                     "snapshot_retention_days", "recording_retention_days", "alert_cooldown_seconds",
                     "association_margin", "alert_record_pre_seconds", "alert_record_post_seconds"):
            if getattr(self, name) < 0:
                problems.append(f"{name} cannot be negative.")

        for name in ("web_port", "phone_port"):
            if not 1 <= getattr(self, name) <= 65535:
                problems.append(f"{name} must be between 1 and 65535.")
        if self.alert_record_max_seconds < 1:
            problems.append("alert_record_max_seconds must be at least 1.")

        return problems

    def to_dict(self) -> dict:
        return asdict(self)


def convert_value(current, text: str):
    """Convert user/YAML input to the same type as the current value."""
    if isinstance(current, bool):
        lowered = str(text).strip().lower()
        if lowered in ("true", "yes", "y", "1", "on"):
            return True
        if lowered in ("false", "no", "n", "0", "off"):
            return False
        raise ValueError("Enter true or false.")
    if isinstance(current, int):
        return int(text)
    if isinstance(current, float):
        return float(text)
    if isinstance(current, list):
        if isinstance(text, list):
            return [str(item) for item in text]
        return [item.strip() for item in str(text).split(",") if item.strip()]
    return str(text)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load settings from YAML. Missing keys use defaults; unknown keys are reported and ignored."""
    config = Config()
    path = Path(path)

    if not path.exists():
        return config

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain key: value settings.")

    known = {item.name for item in fields(Config)}

    for key, value in data.items():
        if key not in known:
            print(f"[config] Ignoring unknown setting '{key}' in {path}.")
            continue
        try:
            setattr(config, key, convert_value(getattr(config, key), value))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid value for '{key}' in {path}: {error}") from error

    problems = config.validate()
    if problems:
        raise ValueError("Invalid settings in config.yaml:\n  - " + "\n  - ".join(problems))

    return config


def save_config(config: Config, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.to_dict(), file, sort_keys=False, allow_unicode=True)
