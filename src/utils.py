"""Small shared helpers: box geometry and file housekeeping."""

from __future__ import annotations

import time
from pathlib import Path

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(inner: Box, outer: Box) -> float:
    """Share of `inner` that lies inside `outer` (0..1)."""
    area = box_area(inner)
    return intersection_area(inner, outer) / area if area > 0 else 0.0


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_old_files(folder: Path | str, retention_days: float, pattern: str = "*") -> int:
    """Delete files older than `retention_days`. Returns how many files were removed."""
    folder = Path(folder)
    if retention_days <= 0 or not folder.exists():
        return 0

    cutoff = time.time() - retention_days * 86400
    removed = 0

    for file in folder.glob(pattern):
        if file.is_file() and file.stat().st_mtime < cutoff:
            try:
                file.unlink()
                removed += 1
            except OSError:
                pass

    return removed
