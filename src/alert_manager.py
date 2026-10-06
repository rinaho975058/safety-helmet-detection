"""Alert manager: one alert per stable violation, with cooldown and optional sound."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

from src.helmet_classifier import Status


@dataclass
class Alert:
    track_id: int
    confidence: float
    timestamp: float  # time.time()


def play_alert_sound() -> None:
    """Short beep without blocking the video loop."""
    def beep():
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1000, 300)
                winsound.Beep(1000, 300)
            else:
                print("\a", end="", flush=True)
        except Exception:  # noqa: BLE001 - sound is optional
            pass

    threading.Thread(target=beep, daemon=True).start()


class AlertManager:
    def __init__(self, stable_frames: int = 10, cooldown_seconds: float = 10.0,
                 enabled: bool = True, sound: bool = True,
                 clock: Callable[[], float] = time.monotonic,
                 sound_player: Callable[[], None] = play_alert_sound):
        self.stable_frames = stable_frames
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self.sound = sound
        self.clock = clock
        self.sound_player = sound_player

        self.active: set[int] = set()            # People currently in an already-handled violation
        self.last_alert: dict[int, float] = {}   # Per person, for the cooldown
        self.last_alert_time: float | None = None
        self.total_alerts = 0

    def update(self, track_id: int, status: Status, stable_frames: int,
               confidence: float) -> Alert | None:
        """Call once per person per frame. Returns an Alert when a new violation should be raised."""
        if status != Status.NO_HELMET:
            self.active.discard(track_id)   # Violation ended; a new one may alert again (after cooldown).
            return None

        if track_id in self.active or stable_frames < self.stable_frames:
            return None

        # The event is now handled, even if alerts are off or still cooling down,
        # so it never fires later for the same continuous violation.
        self.active.add(track_id)

        if not self.enabled:
            return None

        now = self.clock()
        last = self.last_alert.get(track_id)
        if last is not None and now - last < self.cooldown_seconds:
            return None

        self.last_alert[track_id] = now
        self.last_alert_time = now
        self.total_alerts += 1

        if self.sound:
            self.sound_player()

        return Alert(track_id, confidence, time.time())

    def recently_alerted(self, seconds: float = 3.0) -> bool:
        return self.last_alert_time is not None and self.clock() - self.last_alert_time < seconds

    def remove(self, track_id: int) -> None:
        self.active.discard(track_id)

    def reset(self) -> None:
        self.active.clear()
        self.last_alert.clear()
        self.last_alert_time = None
        self.total_alerts = 0
