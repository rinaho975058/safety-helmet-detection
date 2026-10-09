"""Video recorders: save the monitoring view (with boxes and labels) to MP4 files.

VideoRecorder records while the user asks for it (V key, --record, the dashboard REC button).
AlertClipRecorder records a short clip around every no-helmet alert: a few seconds before the
alert (kept in memory) until a few seconds after the last person without a helmet.

Privacy: recordings show people's faces. The screen shows REC while recording, and old files
are deleted automatically after recording_retention_days.
"""

from __future__ import annotations

import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.utils import cleanup_old_files, ensure_dir

DEFAULT_FPS = 15.0


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """MP4 writer. Uses H.264 where available (plays in web browsers), otherwise MPEG-4."""
    candidates = []
    if sys.platform == "win32":
        candidates.append((cv2.CAP_MSMF, "H264"))   # Windows Media Foundation
    candidates.append((cv2.CAP_ANY, "mp4v"))

    for api, codec in candidates:
        writer = cv2.VideoWriter(str(path), api, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise OSError(f"Cannot create video file {path}.")


def _even_size(frame: np.ndarray) -> tuple[int, int]:
    """H.264 needs even width and height."""
    height, width = frame.shape[:2]
    return width - width % 2, height - height % 2


def _clamp_fps(fps: float | None) -> float:
    fps = fps if fps and fps > 0 else DEFAULT_FPS
    return min(max(fps, 1.0), 60.0)


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

    def start(self, frame: np.ndarray, fps: float, prefix: str = "") -> Path:
        """Start a new file sized to `frame`. `fps` should match how fast frames are written."""
        self.stop()
        ensure_dir(self.recording_dir)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = self.recording_dir / f"{prefix}{_safe_name(self.camera_id)}_{stamp}.mp4"
        number = 2
        while path.exists():
            path = path.with_name(f"{prefix}{_safe_name(self.camera_id)}_{stamp}_{number}.mp4")
            number += 1

        self._size = _even_size(frame)
        self.writer = open_writer(path, _clamp_fps(fps), self._size)
        self.path, self.frames = path, 0
        return path

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            return
        if (frame.shape[1], frame.shape[0]) != self._size:
            if frame.shape[1] - self._size[0] in (0, 1) and frame.shape[0] - self._size[1] in (0, 1):
                frame = frame[:self._size[1], :self._size[0]]
            else:
                frame = cv2.resize(frame, self._size)
        self.writer.write(np.ascontiguousarray(frame))
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


class AlertClipRecorder:
    """Records a clip around violations: `pre_seconds` before the first alert, until
    `post_seconds` after the last frame with someone without a helmet (at most `max_seconds`)."""

    def __init__(self, recording_dir: str = "recordings", camera_id: str = "camera-1",
                 pre_seconds: float = 3.0, post_seconds: float = 5.0, max_seconds: float = 60.0,
                 enabled: bool = True):
        self.recorder = VideoRecorder(recording_dir, camera_id, retention_days=0)
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.max_seconds = max_seconds
        self.enabled = enabled
        self.buffer: deque[tuple[float, np.ndarray]] = deque()
        self.started_at = 0.0
        self.end_at = 0.0

    @property
    def recording(self) -> bool:
        return self.recorder.recording

    @property
    def path(self) -> Path | None:
        return self.recorder.path

    def trigger(self, now: float, fps: float) -> Path | None:
        """A violation alert fired. Starts a clip (or extends the current one); returns its path."""
        if not self.enabled:
            return None
        if not self.recording and self.buffer:
            self.recorder.start(self.buffer[0][1], fps, prefix="alert_")
            for _, frame in self.buffer:
                self.recorder.write(frame)
            self.buffer.clear()
            self.started_at = now
        self.extend(now)
        return self.recorder.path

    def extend(self, now: float) -> None:
        """Someone is still without a helmet: keep recording."""
        if self.recording:
            self.end_at = now + self.post_seconds

    def push(self, frame: np.ndarray, now: float) -> Path | None:
        """Add the newest frame. Returns the clip's path when a clip has just finished."""
        if self.recording:
            self.recorder.write(frame)
            if now >= self.end_at or now - self.started_at >= self.max_seconds:
                return self.recorder.stop()
            return None

        self.buffer.append((now, frame))
        while self.buffer and now - self.buffer[0][0] > self.pre_seconds:
            self.buffer.popleft()
        return None

    def stop(self) -> Path | None:
        self.buffer.clear()
        return self.recorder.stop()
