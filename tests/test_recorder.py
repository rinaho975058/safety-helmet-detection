"""Video recorder: files are created, playable, and empty recordings are removed."""

import cv2
import numpy as np

from src.recorder import VideoRecorder


def test_recording_is_playable(tmp_path):
    recorder = VideoRecorder(str(tmp_path), camera_id="cam 1")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    path = recorder.start(frame, fps=10)
    assert recorder.recording
    for value in range(20):
        frame[:] = value * 10
        recorder.write(frame)
    assert recorder.stop() == path
    assert not recorder.recording

    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    assert round(capture.get(cv2.CAP_PROP_FPS)) == 10
    capture.release()
    assert path.name.startswith("cam_1_")


def test_frames_of_another_size_are_resized(tmp_path):
    recorder = VideoRecorder(str(tmp_path))
    recorder.start(np.zeros((120, 160, 3), dtype=np.uint8), fps=0)
    recorder.write(np.zeros((240, 320, 3), dtype=np.uint8))
    path = recorder.stop()

    capture = cv2.VideoCapture(str(path))
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 160
    capture.release()


def test_empty_recording_is_deleted(tmp_path):
    recorder = VideoRecorder(str(tmp_path))
    recorder.start(np.zeros((120, 160, 3), dtype=np.uint8), fps=15)
    assert recorder.stop() is None
    assert list(tmp_path.glob("*.mp4")) == []
