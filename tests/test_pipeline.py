"""End-to-end pipeline test with a fake detector (no model or camera needed)."""

import numpy as np
import yaml

from src.alert_manager import AlertManager
from src.config import Config, load_config, save_config
from src.detector import HELMET, NO_HELMET, PERSON, Detection
from src.event_logger import EventLogger, read_events
from src.helmet_classifier import Status
from src.pipeline import SafetyPipeline
from src.renderer import Overlay, render

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


class ScriptedDetector:
    """Returns pre-written detections, one list per frame."""

    def __init__(self, frames):
        self.frames = list(frames)

    def detect(self, frame):
        return self.frames.pop(0) if self.frames else []


def person(x, confidence=0.9):
    return Detection((x, 50, x + 100, 400), PERSON, confidence, "person")


def head(x, label, confidence=0.9):
    return Detection((x + 20, 55, x + 80, 110), label, confidence, label)


def make_pipeline(frames, tmp_path, **overrides):
    config = Config(smoothing_window=5, status_min_ratio=0.6, alert_stable_frames=3,
                    alert_sound=False, **overrides)
    logger = EventLogger(str(tmp_path / "logs"), "test-cam")
    alerts = AlertManager(config.alert_stable_frames, config.alert_cooldown_seconds,
                          True, sound=False)
    return SafetyPipeline(config, ScriptedDetector(frames), alert_manager=alerts, logger=logger)


def test_two_people_one_violation_one_alert(tmp_path):
    frame_detections = [[person(0), head(0, HELMET), person(300), head(300, NO_HELMET)]] * 30
    pipeline = make_pipeline(frame_detections, tmp_path)

    total_alerts = 0
    for _ in range(30):
        result = pipeline.process(FRAME)
        total_alerts += len(result.alerts)

    statuses = sorted(p.status for p in result.people)
    assert statuses == sorted([Status.HELMET, Status.NO_HELMET])
    assert total_alerts == 1
    assert result.counts.people == 2

    events = read_events(str(tmp_path / "logs"))
    assert [e["event_type"] for e in events] == ["NO_HELMET_ALERT"]
    assert events[0]["camera_id"] == "test-cam"


def test_flickering_low_evidence_never_alerts(tmp_path):
    # No-helmet evidence only every third frame: not stable, no alert.
    frames = [[person(0), head(0, NO_HELMET)] if i % 3 == 0 else [person(0)] for i in range(60)]
    pipeline = make_pipeline(frames, tmp_path)
    assert sum(len(pipeline.process(FRAME).alerts) for _ in range(60)) == 0


def test_bad_frame_from_detector_is_isolated(tmp_path):
    class Broken:
        def detect(self, frame):
            raise RuntimeError("bad frame")

    pipeline = make_pipeline([], tmp_path)
    pipeline.detector = Broken()
    try:
        pipeline.process(FRAME)
    except RuntimeError:
        pass  # monitor.py catches this per frame
    pipeline.detector = ScriptedDetector([[person(0)]])
    assert pipeline.process(FRAME).counts.people == 1


def test_renderer_draws_without_errors(tmp_path):
    pipeline = make_pipeline([[person(0), head(0, HELMET)]], tmp_path)
    result = pipeline.process(FRAME)
    image = render(FRAME, Overlay(result.people, result.counts, pipeline.statistics,
                                  messages=["test"]), 800)
    assert image.shape[1] == 800


def test_config_roundtrip_and_validation(tmp_path):
    path = tmp_path / "config.yaml"
    config = Config(source="video.mp4", helmet_confidence=0.7)
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.source == "video.mp4"
    assert loaded.helmet_confidence == 0.7

    path.write_text(yaml.safe_dump({"helmet_confidence": 3}), encoding="utf-8")
    try:
        load_config(path)
        assert False, "invalid threshold should be rejected"
    except ValueError:
        pass
