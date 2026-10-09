"""Input manager: webcam, video file, RTSP/CCTV stream or this computer's screen."""

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
SCREEN_MAX_FPS = 15.0  # Screen grabs are cheap to repeat; no need to use a whole CPU core


class SourceError(Exception):
    """The input source cannot be opened or has stopped working."""


def parse_source(source: str | int) -> int | str:
    """'0' -> webcam 0; anything else is a file path or stream URL."""
    text = str(source).strip()
    return int(text) if text.isdigit() else text


def is_stream_url(source: int | str) -> bool:
    return isinstance(source, str) and "://" in source


def is_screen(source: int | str) -> bool:
    """'screen', 'screen:2' (monitor 2) or 'screen:left,top,width,height' (an area)."""
    return isinstance(source, str) and (source.lower() == "screen" or source.lower().startswith("screen:"))


def is_phone(source: int | str) -> bool:
    """'phone' = a phone (or any device) sending its camera from the browser page of the dashboard."""
    return isinstance(source, str) and source.strip().lower() == "phone"


def parse_screen(source: str) -> tuple[int, tuple[int, int, int, int] | None]:
    """Return (monitor number, area or None). Monitor 1 is the main screen."""
    spec = source.split(":", 1)[1].strip() if ":" in source else ""
    if not spec:
        return 1, None
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) == 1 and parts[0].isdigit():
        return int(parts[0]), None
    if len(parts) == 4 and all(part.lstrip("-").isdigit() for part in parts):
        left, top, width, height = (int(part) for part in parts)
        if width > 0 and height > 0:
            return 0, (left, top, width, height)
    raise ValueError(f"Invalid screen source '{source}'. Use screen, screen:2 or screen:left,top,width,height.")


def stream_host(url: str) -> str:
    """Host part of a stream URL, without the username and password."""
    rest = url.split("://", 1)[1]
    host = rest.split("/", 1)[0]
    return host.rsplit("@", 1)[-1]


def describe_source(source: str | int) -> str:
    parsed = parse_source(source)
    if isinstance(parsed, int):
        return f"webcam {parsed}"
    if is_stream_url(parsed):
        return f"network camera {stream_host(parsed)}"
    if is_phone(parsed):
        return "phone camera (browser)"
    if is_screen(parsed):
        try:
            monitor, area = parse_screen(parsed)
        except ValueError:
            return "screen (invalid setting)"
        if area:
            return f"screen area {area[2]}x{area[3]} at ({area[0]}, {area[1]})"
        return "screen" if monitor == 1 else f"screen {monitor}"
    return f"video file {parsed}"


def webcam_backends() -> list[int]:
    """Capture backends to try for webcams, best first for this platform."""
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
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


def list_webcams(max_index: int = 4) -> list[int]:
    """Numbers of the webcams on this computer that deliver a frame."""
    found = []
    for index in range(max_index + 1):
        capture = open_webcam(index)
        if capture is None:
            continue
        ok, frame = capture.read()
        capture.release()
        if ok and frame is not None:
            found.append(index)
    return found


def _new_mss():
    import mss

    return mss.MSS() if hasattr(mss, "MSS") else mss.mss()


def list_screens() -> list[dict]:
    """Monitors of this computer: [{'number', 'left', 'top', 'width', 'height'}], main screen first."""
    try:
        with _new_mss() as grabber:
            monitors = grabber.monitors[1:]
    except Exception:  # noqa: BLE001 - no display, or mss not installed
        return []
    return [{"number": number, **{key: monitor[key] for key in ("left", "top", "width", "height")}}
            for number, monitor in enumerate(monitors, start=1)]


class BrowserFeed:
    """Latest camera picture sent by a browser (phone page of the dashboard)."""

    def __init__(self):
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._count = 0
        self.last_time = 0.0
        self.sender = ""

    def push_jpeg(self, data: bytes, sender: str = "") -> bool:
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self._condition:
            self._frame = frame
            self._count += 1
            self.last_time = time.time()
            self.sender = sender
            self._condition.notify_all()
        return True

    def wait_frame(self, after: int, timeout: float) -> tuple[int, np.ndarray | None]:
        """Wait for a frame newer than number `after`. Returns (number, frame or None on timeout)."""
        with self._condition:
            if self._count <= after:
                self._condition.wait(timeout)
            if self._count <= after:
                return after, None
            return self._count, self._frame

    @property
    def connected(self) -> bool:
        return time.time() - self.last_time < 3.0


class ScreenCapture:
    """Grabs the screen like a camera. Has the parts of the cv2.VideoCapture API that VideoSource uses."""

    def __init__(self, source: str, max_fps: float = SCREEN_MAX_FPS):
        self.monitor_number, self.area = parse_screen(source)
        self.min_interval = 1.0 / max_fps
        self.region: dict | None = None
        self._last = 0.0
        self._local = threading.local()  # mss handles only work on the thread that made them
        self._grabbers: list = []
        try:
            grabber = self._grabber()
        except ImportError:
            return
        if self.area:
            left, top, width, height = self.area
            self.region = {"left": left, "top": top, "width": width, "height": height}
        elif 1 <= self.monitor_number < len(grabber.monitors):
            self.region = dict(grabber.monitors[self.monitor_number])

    def _grabber(self):
        if getattr(self._local, "grabber", None) is None:
            self._local.grabber = _new_mss()
            self._grabbers.append(self._local.grabber)
        return self._local.grabber

    def isOpened(self) -> bool:
        return self.region is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.region is None:
            return False, None
        wait = self._last + self.min_interval - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        self._last = time.perf_counter()
        try:
            shot = self._grabber().grab(self.region)
        except Exception:  # noqa: BLE001 - e.g. screen locked or display changed
            return False, None
        return True, np.ascontiguousarray(np.asarray(shot)[:, :, :3])

    def set(self, *args) -> bool:
        return False

    def get(self, prop: int) -> float:
        return 1.0 / self.min_interval if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self) -> None:
        for grabber in self._grabbers:
            try:
                grabber.close()
            except Exception:  # noqa: BLE001
                pass
        self._grabbers.clear()
        self.region = None


def find_webcam(max_index: int = 4) -> int | None:
    """Return the number of the first webcam that delivers a frame, or None."""
    found = list_webcams(max_index)
    return found[0] if found else None


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
        """Webcams, streams and the screen are live; video files are not."""
        return isinstance(self.source, int) or is_stream_url(self.source) or is_screen(self.source)

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
        elif is_screen(self.source):
            try:
                self.capture = ScreenCapture(self.source)
            except ValueError as error:
                raise SourceError(str(error)) from error
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
            if is_screen(self.source):
                raise SourceError(
                    f"Cannot capture {describe_source(self.source)}. Check that the monitor exists "
                    "and that 'mss' is installed (pip install -r requirements.txt)."
                )
            raise SourceError(f"Cannot read video file {self.source}. It may be corrupted or in an unsupported format.")

        if isinstance(self.source, int):
            # MJPG lets most USB webcams deliver 720p at full frame rate instead of ~5 FPS.
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.is_live and not is_screen(self.source):
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
