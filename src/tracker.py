"""Lightweight IoU tracker that gives each person a stable ID across frames.

Keeps people from being counted again every frame and lets status smoothing work per person.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils import Box, iou


@dataclass
class Track:
    track_id: int
    box: Box
    confidence: float
    missed: int = 0      # Consecutive frames without a match
    hits: int = 1        # Frames this person has been seen


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 30):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: dict[int, Track] = {}
        self.removed: list[int] = []   # Track IDs dropped during the last update
        self._next_id = 1

    def update(self, boxes: list[Box], confidences: list[float]) -> list[Track]:
        """Match this frame's person boxes to existing tracks.

        Returns the tracks seen in this frame (same order is not guaranteed).
        """
        self.removed = []

        # Greedy matching: best IoU pairs first.
        pairs = sorted(
            (
                (iou(track.box, box), track_id, index)
                for track_id, track in self.tracks.items()
                for index, box in enumerate(boxes)
            ),
            reverse=True,
        )

        matched_tracks: set[int] = set()
        matched_boxes: set[int] = set()
        visible: list[Track] = []

        for score, track_id, index in pairs:
            if score < self.iou_threshold:
                break
            if track_id in matched_tracks or index in matched_boxes:
                continue

            track = self.tracks[track_id]
            track.box = boxes[index]
            track.confidence = confidences[index]
            track.missed = 0
            track.hits += 1

            matched_tracks.add(track_id)
            matched_boxes.add(index)
            visible.append(track)

        # Unmatched boxes become new people.
        for index, box in enumerate(boxes):
            if index in matched_boxes:
                continue
            track = Track(self._next_id, box, confidences[index])
            self.tracks[track.track_id] = track
            self._next_id += 1
            visible.append(track)

        # Tracks that were not seen get older and are eventually dropped.
        visible_ids = {track.track_id for track in visible}
        for track_id in list(self.tracks):
            if track_id in visible_ids:
                continue
            self.tracks[track_id].missed += 1
            if self.tracks[track_id].missed > self.max_missed:
                del self.tracks[track_id]
                self.removed.append(track_id)

        return visible

    def reset(self) -> None:
        self.tracks.clear()
        self.removed = []
        self._next_id = 1
