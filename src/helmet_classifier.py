"""Safety decision engine: HELMET / NO_HELMET / UNKNOWN per person, with temporal smoothing."""

from __future__ import annotations

from collections import Counter, deque
from enum import Enum

from src.association import HeadEvidence
from src.config import Config
from src.detector import HELMET, NO_HELMET


class Status(str, Enum):
    HELMET = "HELMET"
    NO_HELMET = "NO_HELMET"
    UNKNOWN = "UNKNOWN"

    @property
    def text(self) -> str:
        return {"HELMET": "Helmet", "NO_HELMET": "No Helmet", "UNKNOWN": "Unknown"}[self.value]


def frame_status(evidence: HeadEvidence, person_confidence: float, config: Config) -> Status:
    """Decision for a single frame. Uncertain cases are UNKNOWN, never NO_HELMET."""
    if evidence.ambiguous:
        return Status.UNKNOWN

    if evidence.label == HELMET and evidence.confidence >= config.helmet_confidence:
        return Status.HELMET

    if evidence.label == NO_HELMET and evidence.confidence >= config.no_helmet_confidence:
        return Status.NO_HELMET

    # Optional design: a clearly visible person with no helmet evidence at all counts as No Helmet.
    if (config.derive_no_helmet and evidence.label is None
            and person_confidence >= max(config.person_confidence, config.no_helmet_confidence)):
        return Status.NO_HELMET

    return Status.UNKNOWN


class StatusSmoother:
    """Keeps a short history per person so one odd frame cannot flip the status.

    The displayed status only changes when a new status fills at least `min_ratio` of the
    last `window` frames (hysteresis). UNKNOWN frames never push a person into NO_HELMET.
    """

    def __init__(self, window: int = 15, min_ratio: float = 0.6):
        self.window = window
        self.min_ratio = min_ratio
        self.history: dict[int, deque[Status]] = {}
        self.current: dict[int, Status] = {}
        self.stable_frames: dict[int, int] = {}

    def update(self, track_id: int, status: Status) -> tuple[Status, int]:
        """Add this frame's status. Returns (stable status, frames it has been stable)."""
        history = self.history.setdefault(track_id, deque(maxlen=self.window))
        history.append(status)

        current = self.current.get(track_id, Status.UNKNOWN)
        needed = max(1, round(self.window * self.min_ratio))

        candidate, count = Counter(history).most_common(1)[0]
        if candidate != current and count >= needed:
            current = candidate
            self.stable_frames[track_id] = 0

        self.current[track_id] = current
        self.stable_frames[track_id] = self.stable_frames.get(track_id, 0) + 1
        return current, self.stable_frames[track_id]

    def remove(self, track_id: int) -> None:
        self.history.pop(track_id, None)
        self.current.pop(track_id, None)
        self.stable_frames.pop(track_id, None)

    def reset(self) -> None:
        self.history.clear()
        self.current.clear()
        self.stable_frames.clear()
