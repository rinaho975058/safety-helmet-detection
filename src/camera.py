"""Input manager: webcam, video file or RTSP/CCTV stream."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


# Network cameras: use TCP (UDP drops packets and gives grey, smeared frames).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

STREAM_TIMEOUT_MSEC = 10000
READ_TIMEOUT_SECONDS = 10.0


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


def webcam_backends() -> list[int]:
    """Capture backends to try for webcams, best first for this platform."""
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_webcam(index: int) -> cv2.VideoCapture | None:
    """Open a webcam with the first backend that works, or return None."""
    for backend in webcam_backends():
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            return capture
        capture.release()
    return None


def open_stream(url: str) -> cv2.VideoCapture:
    """Open an rtsp:// or http:// stream with timeouts so a dead camera cannot hang the app."""
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MSEC,
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MSEC]
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)


def find_webcam(max_index: int = 4) -> int | None:
    """Return the number of the first webcam that delivers a frame, or None."""
    for index in range(max_index + 1):
        capture = open_webcam(index)
        if capture is None:
            continue
        ok, frame = capture.read()
        capture.release()
        if ok and frame is not None:
            return index
    return None


class VideoSource:
    def __init__(self, source: str | int, width: int = 1280, height: int = 720,
                 reconnect_attempts: int = 5, threaded: bool = True):
        self.source = parse_source(source)
        self.width = width
        self.height = height
        self.reconnect_attempts = reconnect_attempts
        # Live sources are read on a background thread so the newest frame is always used;
        # otherwise frames queue up while detection runs and the picture falls behind.
        self.threaded = threaded
        self.capture: cv2.VideoCapture | None = None
        self.ended = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._new_frame = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._failed = False

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
        if isinstance(self.source, int):
            self.capture = open_webcam(self.source)
        elif is_stream_url(self.source):
            self.capture = open_stream(self.source)
        else:
            self.capture = cv2.VideoCapture(self.source)

        if self.capture is None or not self.capture.isOpened():
            self.release()
            if isinstance(self.source, int):
                raise SourceError(
                    f"Cannot open webcam {self.source}. Check that it is connected and not used by another app."
                )
            if is_stream_url(self.source):
                raise SourceError("Cannot connect to the network stream. Check the URL, username and password.")
            raise SourceError(f"Cannot read video file {self.source}. It may be corrupted or in an unsupported format.")

        if isinstance(self.source, int):
            # MJPG lets most USB webcams deliver 720p at full frame rate instead of ~5 FPS.
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.is_live:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ended = False

    def read(self) -> np.ndarray | None:
        """Return the next frame, or None when a video file has ended.

        Live sources are reconnected automatically; SourceError is raised if that fails.
        """
        if self.capture is None:
            raise SourceError("Input source is not open.")

        if self.is_live and self.threaded:
            return self._read_latest()

        ok, frame = self.capture.read()
        if ok and frame is not None and frame.size > 0:
            return frame

        if not self.is_live:
            self.ended = True
            return None

        return self._reconnect()

    def _read_latest(self) -> np.ndarray:
        if self._thread is None:
            self._start_reader()

        if self._new_frame.wait(READ_TIMEOUT_SECONDS):
            with self._lock:
                self._new_frame.clear()
                frame, failed = self._latest, self._failed
            if not failed and frame is not None:
                return frame

        frame = self._reconnect()
        self._start_reader()
        return frame

    def _start_reader(self) -> None:
        self._stop.clear()
        self._new_frame.clear()
        self._failed = False
        self._latest = None
        self._thread = threading.Thread(target=self._reader, args=(self.capture,), daemon=True)
        self._thread.start()

    def _reader(self, capture: cv2.VideoCapture) -> None:
        while not self._stop.is_set():
            ok, frame = capture.read()
            with self._lock:
                if not ok or frame is None or frame.size == 0:
                    self._failed = True
                    self._new_frame.set()
                    return
                self._latest = frame
                self._new_frame.set()

    def _stop_reader(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None

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
        self._stop_reader()
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
    video = VideoSource(source, reconnect_attempts=0, threaded=False)
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
