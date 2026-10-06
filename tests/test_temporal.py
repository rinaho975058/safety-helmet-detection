"""Temporal stability, tracking and status decision tests."""

from src.association import HeadEvidence
from src.config import Config
from src.detector import HELMET, NO_HELMET
from src.helmet_classifier import Status, StatusSmoother, frame_status
from src.tracker import IoUTracker

# ---------- per-frame decision ----------


def test_frame_status_uses_thresholds():
    config = Config(helmet_confidence=0.5, no_helmet_confidence=0.6)
    assert frame_status(HeadEvidence(HELMET, 0.7), 0.9, config) == Status.HELMET
    assert frame_status(HeadEvidence(HELMET, 0.4), 0.9, config) == Status.UNKNOWN
    assert frame_status(HeadEvidence(NO_HELMET, 0.7), 0.9, config) == Status.NO_HELMET
    assert frame_status(HeadEvidence(NO_HELMET, 0.55), 0.9, config) == Status.UNKNOWN


def test_ambiguous_or_missing_evidence_is_unknown():
    config = Config()
    assert frame_status(HeadEvidence(NO_HELMET, 0.99, ambiguous=True), 0.9, config) == Status.UNKNOWN
    assert frame_status(HeadEvidence(), 0.99, config) == Status.UNKNOWN


def test_derived_no_helmet_mode():
    config = Config(derive_no_helmet=True)
    assert frame_status(HeadEvidence(), 0.9, config) == Status.NO_HELMET
    assert frame_status(HeadEvidence(), 0.3, config) == Status.UNKNOWN


# ---------- smoothing ----------


def test_single_odd_frame_does_not_flip_status():
    smoother = StatusSmoother(window=10, min_ratio=0.6)
    for _ in range(10):
        status, _ = smoother.update(1, Status.HELMET)
    assert status == Status.HELMET

    status, _ = smoother.update(1, Status.NO_HELMET)
    assert status == Status.HELMET


def test_status_changes_after_consistent_frames():
    smoother = StatusSmoother(window=10, min_ratio=0.6)
    for _ in range(10):
        smoother.update(1, Status.HELMET)

    results = [smoother.update(1, Status.NO_HELMET)[0] for _ in range(10)]
    assert results[0] == Status.HELMET
    assert results[-1] == Status.NO_HELMET


def test_starts_unknown_until_enough_evidence():
    smoother = StatusSmoother(window=10, min_ratio=0.6)
    first, _ = smoother.update(1, Status.NO_HELMET)
    assert first == Status.UNKNOWN


def test_occlusion_frames_keep_previous_status():
    smoother = StatusSmoother(window=10, min_ratio=0.6)
    for _ in range(10):
        smoother.update(1, Status.NO_HELMET)
    for _ in range(5):
        status, _ = smoother.update(1, Status.UNKNOWN)
    assert status == Status.NO_HELMET


def test_stable_frame_counter():
    smoother = StatusSmoother(window=4, min_ratio=0.5)
    counts = [smoother.update(1, Status.HELMET)[1] for _ in range(5)]
    assert counts[-1] > counts[-2] > 0


# ---------- tracking ----------


def test_tracker_keeps_id_for_moving_person():
    tracker = IoUTracker(iou_threshold=0.3, max_missed=5)
    first = tracker.update([(0, 0, 100, 200)], [0.9])[0].track_id
    second = tracker.update([(10, 0, 110, 200)], [0.9])[0].track_id
    assert first == second


def test_tracker_gives_new_people_new_ids_and_drops_lost_people():
    tracker = IoUTracker(iou_threshold=0.3, max_missed=2)
    ids = {track.track_id for track in tracker.update([(0, 0, 100, 200), (300, 0, 400, 200)], [0.9, 0.9])}
    assert len(ids) == 2

    removed = []
    for _ in range(4):
        tracker.update([(0, 0, 100, 200)], [0.9])
        removed += tracker.removed
    assert len(removed) == 1
    assert len(tracker.tracks) == 1
