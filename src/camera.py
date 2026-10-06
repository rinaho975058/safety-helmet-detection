"""Input manager: webcam, video file or RTSP/CCTV stream."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class SourceError(Exception):
    """The input source cannot be opened or has stopped working."""


def parse_source(source: str | int) -> int | str:
    """'0' -> webcam 0; anything else is a file path or stream URL."""
    text = str(source).strip()
    return int(text) if text.isdigit() else text


def is_stream_url(source: int | str) -> bool:
    return isinstance(source, str) and "://" in source


def describe_source(source: str | int) -> str:
    parsed = parse_source(source)
    if isinstance(parsed, int):
        return f"webcam {parsed}"
    if is_stream_url(parsed):
        return "network stream"
    return f"video file {parsed}"


class VideoSource:
    def __init__(self, source: str | int, width: int = 1280, height: int = 720,
                 reconnect_attempts: int = 5):
        self.source = parse_source(source)
        self.width = width
        self.height = height
        self.reconnect_attempts = reconnect_attempts
        self.capture: cv2.VideoCapture | None = None
        self.ended = False

    @property
    def is_live(self) -> bool:
        """Webcams and streams are live; video files are not."""
        return isinstance(self.source, int) or is_stream_url(self.source)

    @property
    def fps(self) -> float:
        if self.capture is None:
            return 0.0
        value = self.capture.get(cv2.CAP_PROP_FPS)
        return value if value and value > 0 else 0.0

    def open(self) -> None:
        if isinstance(self.source, str) and not self.is_live and not Path(self.source).exists():
            raise SourceError(f"Video file not found: {self.source}")

        self.release()
        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            self.release()
            if isinstance(self.source, int):
                raise SourceError(
                    f"Cannot open webcam {self.source}. Check that it is connected and not used by another app."
                )
            if is_stream_url(self.source):
                raise SourceError("Cannot connect to the network stream. Check the URL, username and password.")
            raise SourceError(f"Cannot read video file {self.source}. It may be corrupted or in an unsupported format.")

        if isinstance(self.source, int):
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.ended = False

    def read(self) -> np.ndarray | None:
        """Return the next frame, or None when a video file has ended.

        Live sources are reconnected automatically; SourceError is raised if that fails.
        """
        if self.capture is None:
            raise SourceError("Input source is not open.")

        ok, frame = self.capture.read()
        if ok and frame is not None and frame.size > 0:
            return frame

        if not self.is_live:
            self.ended = True
            return None

        return self._reconnect()

    def _reconnect(self) -> np.ndarray:
        for attempt in range(1, self.reconnect_attempts + 1):
            print(f"[camera] Lost signal. Reconnecting ({attempt}/{self.reconnect_attempts})...")
            time.sleep(1.0)
            try:
                self.open()
            except SourceError:
                continue
            ok, frame = self.capture.read()
            if ok and frame is not None:
                print("[camera] Reconnected.")
                return frame

        raise SourceError("The camera was disconnected and could not be reconnected.")

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.release()


def check_source(source: str | int) -> str | None:
    """Try to open the source and read one frame. Returns an error message or None."""
    video = VideoSource(source, reconnect_attempts=0)
    try:
        video.open()
        ok, frame = video.capture.read()
        if not ok or frame is None:
            return f"Opened {describe_source(source)} but could not read a frame."
        return None
    except SourceError as error:
        return str(error)
    finally:
        video.release()
