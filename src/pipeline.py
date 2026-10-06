"""Detection pipeline: one frame in, people with stable safety status out.

detect -> filter -> track people -> associate helmets -> per-frame status
-> temporal smoothing -> alerts -> statistics -> event log
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.alert_manager import Alert, AlertManager
from src.association import associate
from src.config import Config
from src.detector import HELMET, NO_HELMET, PERSON, Detection
from src.event_logger import EventLogger
from src.helmet_classifier import Status, StatusSmoother, frame_status
from src.renderer import PersonView
from src.statistics import FrameCounts, SafetyStatistics
from src.tracker import IoUTracker


class DetectorLike(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


@dataclass
class FrameResult:
    people: list[PersonView]
    counts: FrameCounts
    alerts: list[Alert] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)


class SafetyPipeline:
    def __init__(self, config: Config, detector: DetectorLike,
                 alert_manager: AlertManager | None = None,
                 logger: EventLogger | None = None,
                 statistics: SafetyStatistics | None = None):
        self.config = config
        self.detector = detector
        self.tracker = IoUTracker(config.tracker_iou, config.max_missed_frames)
        self.smoother = StatusSmoother(config.smoothing_window, config.status_min_ratio)
        self.alerts = alert_manager or AlertManager(
            config.alert_stable_frames, config.alert_cooldown_seconds,
            config.alerts_enabled, config.alert_sound,
        )
        self.logger = logger
        self.statistics = statistics or SafetyStatistics()

    def process(self, frame: np.ndarray) -> FrameResult:
        detections = self.detector.detect(frame)

        persons = [det for det in detections if det.label == PERSON]
        heads = [det for det in detections if det.label in (HELMET, NO_HELMET)]

        tracks = self.tracker.update([det.box for det in persons], [det.confidence for det in persons])

        for track_id in self.tracker.removed:   # Person left the scene: reset their state.
            self.smoother.remove(track_id)
            self.alerts.remove(track_id)

        evidence = associate(
            [track.box for track in tracks], heads,
            self.config.head_region_ratio, self.config.association_min_overlap,
            self.config.association_margin,
        )

        people: list[PersonView] = []
        statuses: dict[int, Status] = {}
        new_alerts: list[Alert] = []

        for track, head in zip(tracks, evidence):
            current = frame_status(head, track.confidence, self.config)
            stable, stable_frames = self.smoother.update(track.track_id, current)
            statuses[track.track_id] = stable

            people.append(PersonView(track.track_id, track.box, stable, track.confidence,
                                     head.box, head.confidence))

            alert = self.alerts.update(track.track_id, stable, stable_frames, head.confidence)
            if alert:
                new_alerts.append(alert)
                self.statistics.record_alert()

        counts = self.statistics.update(statuses)

        if new_alerts and self.logger:
            snapshot = self.logger.save_snapshot(frame) if self.logger.snapshots_enabled else ""
            for alert in new_alerts:
                self.logger.log("NO_HELMET_ALERT", alert.track_id, Status.NO_HELMET.value,
                                alert.confidence or None, snapshot)

        return FrameResult(people, counts, new_alerts, detections)

    def reset(self) -> None:
        self.tracker.reset()
        self.smoother.reset()
        self.alerts.reset()
        self.statistics.reset()
