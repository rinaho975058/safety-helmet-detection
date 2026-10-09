"""Web dashboard, monitoring engine and alert clips, with a fake detector (no model or camera needed)."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import Config
from src.detector import NO_HELMET, PERSON, Detection
from src.engine import MonitorEngine
from src.recorder import AlertClipRecorder


class NoHelmetDetector:
    """Every frame: one person without a helmet."""

    def detect(self, frame):
        return [Detection((200, 60, 420, 470), PERSON, 0.9, "person"),
                Detection((260, 70, 360, 160), NO_HELMET, 0.85, "no_helmet")]


def write_video(path: Path, frames: int = 60, fps: float = 20.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    for number in range(frames):
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        cv2.putText(frame, str(number), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()
    return path


def make_config(tmp_path: Path, source: str) -> Config:
    return Config(source=source, smoothing_window=3, alert_stable_frames=2, alert_sound=False,
                  recording_dir=str(tmp_path / "rec"), log_dir=str(tmp_path / "logs"),
                  snapshot_dir=str(tmp_path / "snaps"), alert_record_pre_seconds=0.5,
                  alert_record_post_seconds=0.5, display_width=640)


def wait_until(condition, seconds=20.0):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


# ---------- Alert clips ----------

def test_alert_clip_has_frames_from_before_the_alert(tmp_path):
    clips = AlertClipRecorder(str(tmp_path), pre_seconds=1.0, post_seconds=0.5)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    now = 100.0
    for _ in range(20):                      # 2 s at 10 FPS; only the last ~1 s is kept
        now += 0.1
        assert clips.push(frame, now) is None
    path = clips.trigger(now, fps=10)
    assert path is not None and path.name.startswith("alert_")

    finished = None
    while finished is None:
        now += 0.1
        finished = clips.push(frame, now)
    capture = cv2.VideoCapture(str(finished))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert 13 <= frames <= 18                 # ~1 s before + 0.5 s after


def test_alert_clip_keeps_recording_while_the_violation_lasts(tmp_path):
    clips = AlertClipRecorder(str(tmp_path), pre_seconds=0.2, post_seconds=0.5, max_seconds=60)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    clips.push(frame, 0.0)
    clips.trigger(0.0, fps=10)
    for step in range(1, 30):                 # 3 s with someone still without a helmet
        clips.extend(step / 10)
        assert clips.push(frame, step / 10) is None
    assert clips.recording


def test_alert_clip_can_be_turned_off(tmp_path):
    clips = AlertClipRecorder(str(tmp_path), enabled=False)
    clips.push(np.zeros((120, 160, 3), dtype=np.uint8), 0.0)
    assert clips.trigger(0.0, fps=10) is None
    assert not clips.recording


# ---------- Engine ----------

def test_engine_alerts_records_and_tracks_people(tmp_path):
    video = write_video(tmp_path / "clip.mp4")
    engine = MonitorEngine(make_config(tmp_path, str(video)), detector_factory=lambda config: NoHelmetDetector())
    engine.start()
    try:
        assert wait_until(lambda: engine.state()["state"] == "ended")
    finally:
        engine.stop()

    state = engine.state()
    assert [card["status"] for card in state["people"]] == ["NO_HELMET"]
    assert engine.thumbnail(state["people"][0]["id"]).startswith(b"\xff\xd8")   # JPEG photo

    assert len(state["alerts"]) == 1
    alert = state["alerts"][0]
    assert alert["message"] == f"Person {alert['track_id']} is not wearing a helmet."
    assert engine.alert_image(alert["id"]).startswith(b"\xff\xd8")
    assert alert["clip"] and (tmp_path / "rec" / alert["clip"]).exists()
    assert not alert["clip_recording"]

    assert engine.dismiss(alert["id"])
    assert engine.state()["alerts"][0]["dismissed"]


def test_engine_reports_a_missing_camera(tmp_path):
    engine = MonitorEngine(make_config(tmp_path, str(tmp_path / "missing.mp4")),
                           detector_factory=lambda config: NoHelmetDetector())
    engine.start()
    try:
        assert wait_until(lambda: engine.state()["state"] == "error")
    finally:
        engine.stop()
    assert "not found" in engine.state()["message"]


def test_engine_uses_pictures_from_the_phone_page(tmp_path):
    engine = MonitorEngine(make_config(tmp_path, "phone"), detector_factory=lambda config: NoHelmetDetector())
    engine.start()
    try:
        jpeg = cv2.imencode(".jpg", np.full((480, 640, 3), 90, dtype=np.uint8))[1].tobytes()
        for _ in range(8):
            assert engine.browser_feed.push_jpeg(jpeg)
            time.sleep(0.05)
        assert wait_until(lambda: engine.state()["alerts"])
        assert engine.state()["phone_connected"]
    finally:
        engine.stop()
    assert not engine.browser_feed.push_jpeg(b"not a picture")


# ---------- Web server ----------

@pytest.fixture()
def client(tmp_path):
    pytest.importorskip("flask")
    from src.dashboard.server import Dashboard

    video = write_video(tmp_path / "clip.mp4")
    config = make_config(tmp_path, str(video))
    engine = MonitorEngine(config, detector_factory=lambda config: NoHelmetDetector())
    dashboard = Dashboard(config, str(tmp_path / "config.yaml"), engine)
    yield dashboard.app.test_client(), engine, tmp_path
    engine.stop()


def test_dashboard_pages_and_state(client):
    http, engine, _ = client
    assert http.get("/").status_code == 200
    assert b"Detection Result" in http.get("/").data
    assert http.get("/static/app.js").status_code == 200
    assert http.get("/api/state").get_json()["state"] == "stopped"

    sources = http.get("/api/sources").get_json()["sources"]
    assert any(item["kind"] == "phone" for item in sources)


def test_dashboard_switch_camera_alert_and_recording_list(client):
    http, engine, tmp_path = client
    response = http.post("/api/source", json={"source": str(tmp_path / "clip.mp4")})
    assert response.status_code == 200
    assert wait_until(lambda: engine.state()["state"] == "ended")

    alert = http.get("/api/state").get_json()["alerts"][0]
    assert http.get(f"/api/alerts/{alert['id']}.jpg").mimetype == "image/jpeg"
    assert http.post(f"/api/alerts/{alert['id']}/dismiss").get_json()["ok"]

    recordings = http.get("/api/recordings").get_json()["recordings"]
    assert recordings and recordings[0]["alert"]
    clip = http.get(recordings[0]["url"])
    assert clip.status_code == 200 and clip.mimetype == "video/mp4"
    assert (tmp_path / "config.yaml").exists()   # The chosen camera is saved


def test_dashboard_rejects_bad_cameras(client):
    http, _, tmp_path = client
    response = http.post("/api/source", json={"source": str(tmp_path / "nothing.mp4")})
    assert response.status_code == 400 and "not found" in response.get_json()["error"]
    response = http.post("/api/cameras", json={"name": "Gate", "address": "192.168.1.10"})
    assert response.status_code == 400 and "rtsp://" in response.get_json()["error"]
    assert http.get("/media/recordings/../config.yaml").status_code == 404
