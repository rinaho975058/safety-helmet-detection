"""Video recorder: saves the monitoring view (with boxes and labels) to MP4 files.

Privacy: recordings show people's faces. Recording is off until the user presses V or sets
recording_enabled, the screen shows REC while it runs, and old files are deleted automatically.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.utils import cleanup_old_files, ensure_dir

DEFAULT_FPS = 15.0


class VideoRecorder:
    def __init__(self, recording_dir: str = "recordings", camera_id: str = "camera-1",
                 retention_days: int = 7):
        self.recording_dir = Path(recording_dir)
        self.camera_id = camera_id
        self.retention_days = retention_days
        self.writer: cv2.VideoWriter | None = None
        self.path: Path | None = None
        self.frames = 0
        cleanup_old_files(self.recording_dir, retention_days, "*.mp4")

    @property
    def recording(self) -> bool:
        return self.writer is not None

    def start(self, frame: np.ndarray, fps: float) -> Path:
        """Start a new file sized to `frame`. `fps` should match how fast frames are written."""
        self.stop()
        ensure_dir(self.recording_dir)
        fps = fps if fps and fps > 0 else DEFAULT_FPS
        fps = min(max(fps, 1.0), 60.0)

        safe_camera = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in self.camera_id)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = self.recording_dir / f"{safe_camera}_{stamp}.mp4"

        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            raise OSError(f"Cannot create video file {path}.")

        self.writer, self.path, self.frames = writer, path, 0
        self._size = (width, height)
        return path

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            return
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)
        self.writer.write(frame)
        self.frames += 1

    def stop(self) -> Path | None:
        """Finish the current file. Returns its path, or None if nothing was recorded."""
        if self.writer is None:
            return None
        self.writer.release()
        path, frames = self.path, self.frames
        self.writer, self.path, self.frames = None, None, 0
        if frames == 0 and path is not None:
            path.unlink(missing_ok=True)
            return None
        return path
