"""Event logger: violation events to daily CSV files, plus optional snapshots.

Privacy: only time, event type, anonymous track number, confidence and camera name are stored.
No names, faces or continuous video are saved.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.utils import cleanup_old_files, ensure_dir

FIELDS = ["timestamp", "event_type", "track_id", "status", "confidence", "camera_id", "snapshot"]


class EventLogger:
    def __init__(self, log_dir: str = "logs", camera_id: str = "camera-1", enabled: bool = True,
                 retention_days: int = 30, snapshots_enabled: bool = False,
                 snapshot_dir: str = "snapshots", snapshot_retention_days: int = 7):
        self.log_dir = Path(log_dir)
        self.camera_id = camera_id
        self.enabled = enabled
        self.retention_days = retention_days
        self.snapshots_enabled = snapshots_enabled
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_retention_days = snapshot_retention_days

        if self.enabled:
            ensure_dir(self.log_dir)
            cleanup_old_files(self.log_dir, retention_days, "events_*.csv")
        if self.snapshots_enabled:
            ensure_dir(self.snapshot_dir)
            cleanup_old_files(self.snapshot_dir, snapshot_retention_days, "*.jpg")

    def _log_file(self, when: datetime) -> Path:
        return self.log_dir / f"events_{when:%Y-%m-%d}.csv"

    def log(self, event_type: str, track_id: int | str = "", status: str = "",
            confidence: float | None = None, snapshot: str = "") -> None:
        if not self.enabled:
            return

        now = datetime.now()
        path = self._log_file(now)
        new_file = not path.exists()

        try:
            with path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": now.isoformat(timespec="seconds"),
                    "event_type": event_type,
                    "track_id": track_id,
                    "status": status,
                    "confidence": "" if confidence is None else f"{confidence:.2f}",
                    "camera_id": self.camera_id,
                    "snapshot": snapshot,
                })
        except OSError as error:
            print(f"[log] Could not write event log: {error}")

    def save_snapshot(self, frame: np.ndarray, reason: str = "violation") -> str:
        """Save a frame as JPEG. Returns the file path, or "" when snapshots are off or saving failed."""
        if frame is None:
            return ""

        ensure_dir(self.snapshot_dir)
        path = self.snapshot_dir / f"{reason}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"

        if cv2.imwrite(str(path), frame):
            return str(path)

        print(f"[log] Could not save snapshot to {path}")
        return ""


def read_events(log_dir: str = "logs") -> list[dict]:
    """Read all logged events, oldest first."""
    events = []
    for path in sorted(Path(log_dir).glob("events_*.csv")):
        try:
            with path.open("r", newline="", encoding="utf-8") as file:
                events.extend(csv.DictReader(file))
        except OSError as error:
            print(f"[log] Could not read {path}: {error}")
    return events
