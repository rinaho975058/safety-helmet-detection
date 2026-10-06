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


def test_network_camera_description_hides_password():
    text = camera.describe_source("rtsp://admin:secret@192.168.1.10:554/stream1")
    assert text == "network camera 192.168.1.10:554"
    assert camera.describe_source("http://192.168.1.20:8080/video") == "network camera 192.168.1.20:8080"


def test_list_webcams_returns_working_cameras(monkeypatch):
    working = {1: FakeCapture(frames=1), 3: FakeCapture(frames=1)}
    for fake in working.values():
        fake.gate.release()
    monkeypatch.setattr(camera, "open_webcam", lambda index: working.get(index))
    assert camera.list_webcams() == [1, 3]
