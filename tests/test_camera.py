"""Live-source reading with a fake camera (no real webcam needed)."""

import threading

import numpy as np
import pytest

from src import camera
from src.camera import SourceError, VideoSource


class FakeCapture:
    """Delivers numbered frames, then reports a lost signal."""

    def __init__(self, frames: int):
        self.remaining = frames
        self.count = 0
        self.released = False
        self.gate = threading.Semaphore(0)

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def get(self, *args):
        return 30.0

    def read(self):
        self.gate.acquire(timeout=2)
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        self.count += 1
        return True, np.full((4, 4, 3), self.count, dtype=np.uint8)

    def release(self):
        self.released = True


def test_live_source_returns_newest_frame(monkeypatch):
    fake = FakeCapture(frames=5)
    monkeypatch.setattr(camera, "open_webcam", lambda index: fake)

    video = VideoSource(0, reconnect_attempts=0)
    video.open()
    for _ in range(3):
        fake.gate.release()
    first = video.read()
    assert first is not None
    video.release()
    assert fake.released


def test_lost_live_source_raises_after_reconnect_fails(monkeypatch):
    fake = FakeCapture(frames=0)
    monkeypatch.setattr(camera, "open_webcam", lambda index: fake)

    video = VideoSource(0, reconnect_attempts=0)
    video.open()
    fake.gate.release()
    with pytest.raises(SourceError):
        video.read()
    video.release()


def test_webcam_not_found_message(monkeypatch):
    monkeypatch.setattr(camera, "open_webcam", lambda index: None)
    assert "Cannot open webcam 3" in camera.check_source("3")
