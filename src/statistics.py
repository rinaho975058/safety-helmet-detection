"""Statistics manager: current-frame counts and session totals."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.helmet_classifier import Status


@dataclass
class FrameCounts:
    people: int = 0
    helmet: int = 0
    no_helmet: int = 0
    unknown: int = 0


class SafetyStatistics:
    """Session totals are counted per tracked person, not per frame, so one person is counted once.

    A person who was ever confirmed NO_HELMET counts as a violation person; otherwise one who was
    ever confirmed HELMET counts as compliant; otherwise UNKNOWN.
    """

    def __init__(self, clock=time.time):
        self.clock = clock
        self.reset()

    def reset(self) -> None:
        self.started_at = self.clock()
        self.current = FrameCounts()
        self.person_status: dict[int, Status] = {}
        self.alerts = 0
        self.frames = 0

    def update(self, statuses: dict[int, Status]) -> FrameCounts:
        """`statuses` maps track ID -> stable status for every person visible in this frame."""
        self.frames += 1
        counts = FrameCounts(people=len(statuses))

        for track_id, status in statuses.items():
            if status == Status.HELMET:
                counts.helmet += 1
            elif status == Status.NO_HELMET:
                counts.no_helmet += 1
            else:
                counts.unknown += 1

            previous = self.person_status.get(track_id, Status.UNKNOWN)
            if status == Status.NO_HELMET or previous == Status.NO_HELMET:
                self.person_status[track_id] = Status.NO_HELMET
            elif status == Status.HELMET or previous == Status.HELMET:
                self.person_status[track_id] = Status.HELMET
            else:
                self.person_status[track_id] = Status.UNKNOWN

        self.current = counts
        return counts

    def record_alert(self) -> None:
        self.alerts += 1

    @property
    def total_people(self) -> int:
        return len(self.person_status)

    def total(self, status: Status) -> int:
        return sum(1 for value in self.person_status.values() if value == status)

    @property
    def violation_rate(self) -> float:
        """Share of people with a decided status who were not wearing a helmet (0..1)."""
        helmet = self.total(Status.HELMET)
        no_helmet = self.total(Status.NO_HELMET)
        decided = helmet + no_helmet
        return no_helmet / decided if decided else 0.0

    @property
    def elapsed_seconds(self) -> float:
        return self.clock() - self.started_at

    def summary(self) -> dict:
        return {
            "monitoring_seconds": round(self.elapsed_seconds, 1),
            "total_people": self.total_people,
            "helmet": self.total(Status.HELMET),
            "no_helmet": self.total(Status.NO_HELMET),
            "unknown": self.total(Status.UNKNOWN),
            "violation_rate": round(self.violation_rate, 3),
            "alerts": self.alerts,
        }
