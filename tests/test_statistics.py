import pytest

from src.helmet_classifier import Status
from src.statistics import SafetyStatistics


def test_current_frame_counts():
    stats = SafetyStatistics()
    counts = stats.update({1: Status.HELMET, 2: Status.NO_HELMET, 3: Status.UNKNOWN, 4: Status.HELMET})
    assert (counts.people, counts.helmet, counts.no_helmet, counts.unknown) == (4, 2, 1, 1)


def test_people_are_counted_once_across_frames():
    stats = SafetyStatistics()
    for _ in range(100):
        stats.update({1: Status.HELMET, 2: Status.HELMET})
    assert stats.total_people == 2
    assert stats.total(Status.HELMET) == 2


def test_person_who_was_ever_in_violation_counts_as_violation():
    stats = SafetyStatistics()
    stats.update({1: Status.UNKNOWN})
    stats.update({1: Status.NO_HELMET})
    stats.update({1: Status.HELMET})
    assert stats.total(Status.NO_HELMET) == 1
    assert stats.total(Status.HELMET) == 0


def test_violation_rate_ignores_unknown_people():
    stats = SafetyStatistics()
    stats.update({1: Status.HELMET, 2: Status.HELMET, 3: Status.HELMET,
                  4: Status.NO_HELMET, 5: Status.UNKNOWN})
    assert stats.violation_rate == pytest.approx(0.25)


def test_violation_rate_is_zero_without_data():
    assert SafetyStatistics().violation_rate == 0.0


def test_reset_and_summary():
    stats = SafetyStatistics()
    stats.update({1: Status.NO_HELMET})
    stats.record_alert()
    assert stats.summary()["alerts"] == 1
    stats.reset()
    assert stats.summary()["total_people"] == 0
    assert stats.summary()["alerts"] == 0
