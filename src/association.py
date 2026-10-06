"""Person-helmet association: decides which helmet / no-helmet detection belongs to which person.

Method: a helmet box belongs to a person when most of it lies inside the top part of the
person's box (the head region). If a detection fits two people almost equally well, or a person
has conflicting helmet and no-helmet evidence of similar strength, the result is marked
ambiguous so the person is shown as UNKNOWN instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.detector import HELMET, NO_HELMET, Detection
from src.utils import Box, overlap_ratio


@dataclass
class HeadEvidence:
    label: str | None = None          # helmet | no_helmet | None (no evidence)
    confidence: float = 0.0
    box: Box | None = None            # The matched helmet / no-helmet box
    ambiguous: bool = False
    candidates: list[Detection] = field(default_factory=list)


def head_region(person_box: Box, ratio: float) -> Box:
    x1, y1, x2, y2 = person_box
    return (x1, y1, x2, y1 + (y2 - y1) * ratio)


def associate(person_boxes: list[Box], head_detections: list[Detection],
              head_ratio: float = 0.35, min_overlap: float = 0.5,
              margin: float = 0.15) -> list[HeadEvidence]:
    """Return one HeadEvidence per person box (same order as `person_boxes`)."""
    evidence = [HeadEvidence() for _ in person_boxes]
    regions = [head_region(box, head_ratio) for box in person_boxes]

    for detection in head_detections:
        if detection.label not in (HELMET, NO_HELMET):
            continue

        scores = sorted(
            ((overlap_ratio(detection.box, region), index) for index, region in enumerate(regions)),
            reverse=True,
        )
        scores = [(score, index) for score, index in scores if score >= min_overlap]

        if not scores:
            continue  # Helmet not on anyone's head (for example lying on a table).

        best_score, best_index = scores[0]

        # Fits two people almost equally (overlapping people): do not guess.
        if len(scores) > 1 and best_score - scores[1][0] < margin:
            for _, index in scores[:2]:
                evidence[index].ambiguous = True
            continue

        evidence[best_index].candidates.append(detection)

    for item in evidence:
        if not item.candidates:
            continue

        best = max(item.candidates, key=lambda det: det.confidence)
        item.label, item.confidence, item.box = best.label, best.confidence, best.box

        # Both helmet and no-helmet found on one head with similar confidence: uncertain.
        opposite = [det for det in item.candidates if det.label != best.label]
        if opposite and best.confidence - max(det.confidence for det in opposite) < margin:
            item.ambiguous = True

    return evidence
