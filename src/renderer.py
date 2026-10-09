"""Visualization: draws boxes, labels, safety banner, statistics and controls on each frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.helmet_classifier import Status
from src.statistics import FrameCounts, SafetyStatistics
from src.utils import Box

# Colours are BGR.
GREEN = (60, 180, 75)
RED = (40, 40, 230)
AMBER = (0, 190, 255)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)
GREY = (170, 170, 170)

STATUS_COLOURS = {Status.HELMET: GREEN, Status.NO_HELMET: RED, Status.UNKNOWN: AMBER}
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class PersonView:
    track_id: int
    box: Box
    status: Status
    person_confidence: float
    head_box: Box | None = None
    head_confidence: float = 0.0


@dataclass
class Overlay:
    people: list[PersonView]
    counts: FrameCounts
    stats: SafetyStatistics
    fps: float = 0.0
    paused: bool = False
    alerts_enabled: bool = True
    snapshots_enabled: bool = False
    recording: bool = False
    alert_flash: bool = False
    messages: list[str] = field(default_factory=list)


def _label(image, text, origin, colour, scale=0.55, thickness=1):
    """Text on a filled background so it stays readable on any scene."""
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(origin[0]), int(origin[1])
    y = max(y, height + baseline + 2)
    cv2.rectangle(image, (x, y - height - baseline - 4), (x + width + 6, y), colour, -1)
    cv2.putText(image, text, (x + 3, y - baseline - 2), FONT, scale, WHITE, thickness, cv2.LINE_AA)


def _panel(image, top_left, size, alpha=0.6):
    x, y = top_left
    w, h = size
    region = image[y:y + h, x:x + w]
    if region.size:
        image[y:y + h, x:x + w] = cv2.addWeighted(region, 1 - alpha, np.full_like(region, DARK), alpha, 0)


def resize_for_display(frame: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    height, current_width = frame.shape[:2]
    if current_width == width:
        return frame.copy(), 1.0
    scale = width / current_width
    return cv2.resize(frame, (width, int(height * scale))), scale


def render(frame: np.ndarray, overlay: Overlay, display_width: int = 1280) -> np.ndarray:
    image, scale = resize_for_display(frame, display_width)
    height, width = image.shape[:2]

    def scaled(box: Box) -> tuple[int, int, int, int]:
        return tuple(int(value * scale) for value in box)

    # People and their head evidence.
    for person in overlay.people:
        colour = STATUS_COLOURS[person.status]
        x1, y1, x2, y2 = scaled(person.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

        text = f"#{person.track_id} {person.status.text}"
        if person.head_box is not None and person.status != Status.UNKNOWN:
            text += f" {person.head_confidence:.2f}"
        _label(image, text, (x1, y1), colour)

        if person.head_box is not None:
            hx1, hy1, hx2, hy2 = scaled(person.head_box)
            cv2.rectangle(image, (hx1, hy1), (hx2, hy2), colour, 1)

    # Top banner: SAFE / ATTENTION.
    attention = overlay.counts.no_helmet > 0
    banner_colour = RED if attention else GREEN
    cv2.rectangle(image, (0, 0), (width, 44), banner_colour, -1)
    banner = "ATTENTION: person without helmet" if attention else "SAFE"
    cv2.putText(image, banner, (14, 31), FONT, 0.9, WHITE, 2, cv2.LINE_AA)

    flags = []
    if overlay.paused:
        flags.append("PAUSED")
    if not overlay.alerts_enabled:
        flags.append("ALERTS OFF")
    if overlay.snapshots_enabled:
        flags.append("SNAPSHOTS ON")
    if overlay.recording:
        flags.append("REC")
    if overlay.fps > 0:
        flags.append(f"{overlay.fps:.1f} FPS")
    flag_text = "  |  ".join(flags)
    (flag_width, _), _ = cv2.getTextSize(flag_text, FONT, 0.55, 1)
    cv2.putText(image, flag_text, (width - flag_width - 14, 28), FONT, 0.55, WHITE, 1, cv2.LINE_AA)

    # Statistics panel.
    counts, stats = overlay.counts, overlay.stats
    lines = [
        ("NOW", None),
        (f"People: {counts.people}", WHITE),
        (f"Helmet: {counts.helmet}", GREEN),
        (f"No Helmet: {counts.no_helmet}", RED),
        (f"Unknown: {counts.unknown}", AMBER),
        ("SESSION", None),
        (f"People seen: {stats.total_people}", WHITE),
        (f"Violation rate: {stats.violation_rate * 100:.0f}%", WHITE),
        (f"Alerts: {stats.alerts}", WHITE),
    ]
    panel_height = 14 + 24 * len(lines)
    _panel(image, (10, 54), (230, panel_height))
    y = 76
    for text, colour in lines:
        if colour is None:
            cv2.putText(image, text, (20, y), FONT, 0.45, GREY, 1, cv2.LINE_AA)
        else:
            cv2.putText(image, text, (20, y), FONT, 0.55, colour, 1, cv2.LINE_AA)
        y += 24

    # Warnings / errors.
    y = height - 44
    for message in overlay.messages[-3:]:
        _label(image, message[:120], (10, y), AMBER, 0.5)
        y -= 26

    # Controls.
    controls = "[Space] Pause   [S] Snapshot   [A] Alerts on/off   [R] Reset stats   [V] Record   [Q] Stop"
    _panel(image, (0, height - 30), (width, 30), 0.7)
    cv2.putText(image, controls, (12, height - 10), FONT, 0.5, WHITE, 1, cv2.LINE_AA)

    # Red frame while a fresh violation alert is active.
    if overlay.alert_flash:
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), RED, 8)

    return image


# ---------- Dashboard style: head box + rounded label, as in the web dashboard ----------

PILL_TEXT = {Status.HELMET: "Helmet Detected", Status.NO_HELMET: "No Helmet", Status.UNKNOWN: "Checking..."}
PILL_RGB = {Status.HELMET: (22, 163, 74), Status.NO_HELMET: (220, 38, 38), Status.UNKNOWN: (217, 119, 6)}
_FONT_CACHE: dict = {}


def _font(size: int, bold: bool = True):
    from PIL import ImageFont

    key = (size, bold)
    if key not in _FONT_CACHE:
        names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
                 else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
        font = None
        for name in names:
            try:
                font = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        _FONT_CACHE[key] = font or ImageFont.load_default()
    return _FONT_CACHE[key]


def head_area(person: PersonView, head_ratio: float = 0.35) -> Box:
    """The box drawn around a person's head: the helmet/head detection, or the top of the person."""
    if person.head_box is not None:
        x1, y1, x2, y2 = person.head_box
        pad_x, pad_y = (x2 - x1) * 0.25, (y2 - y1) * 0.2
        return (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y * 3)
    x1, y1, x2, y2 = person.box
    return (x1, y1, x2, y1 + (y2 - y1) * head_ratio)


def render_dashboard(frame: np.ndarray, people: list[PersonView], max_width: int = 1280,
                     stamp: str = "") -> tuple[np.ndarray, float]:
    """Draw the dashboard style labels. Returns (image, scale used from frame to image)."""
    from PIL import Image, ImageDraw

    image, scale = resize_for_display(frame, min(max_width, frame.shape[1]))
    height, width = image.shape[:2]
    line = max(2, round(height / 240))
    max_font = max(13, round(height / 32))

    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)

    for person in people:
        colour = PILL_RGB[person.status]
        x1, y1, x2, y2 = (value * scale for value in head_area(person))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=line)

        # Rounded label centred under the box, with a white circle icon. Sized to the person, so
        # labels of small (distant) people stay small; very small people get the icon only.
        person_width = (person.box[2] - person.box[0]) * scale
        font_size = int(min(max_font, max(11, person_width * 0.11)))
        font = _font(font_size)
        text = PILL_TEXT[person.status]
        pill_h = round(font_size * 1.9)
        icon = round(pill_h * 0.62)
        pad = round(pill_h * 0.3)
        if person_width < font_size * 6:
            text = ""
            pill_w = pad * 2 + icon
        else:
            pill_w = pad + icon + pad * 0.8 + draw.textlength(text, font=font) + pad * 1.3
        cx = (x1 + x2) / 2
        px1 = min(max(0, cx - pill_w / 2), width - pill_w)
        py1 = min(y2 - pill_h * 0.35, height - pill_h)
        draw.rounded_rectangle((px1, py1, px1 + pill_w, py1 + pill_h), radius=pill_h // 2, fill=colour)

        ix1, iy1 = px1 + pad, py1 + (pill_h - icon) / 2
        draw.ellipse((ix1, iy1, ix1 + icon, iy1 + icon), fill=(255, 255, 255))
        mid_x, mid_y = ix1 + icon / 2, iy1 + icon / 2
        mark = max(2, round(icon / 7))
        if person.status == Status.HELMET:
            draw.line([(ix1 + icon * 0.27, mid_y), (ix1 + icon * 0.44, iy1 + icon * 0.68),
                       (ix1 + icon * 0.74, iy1 + icon * 0.33)], fill=colour, width=mark, joint="curve")
        elif person.status == Status.NO_HELMET:
            draw.line([(mid_x, iy1 + icon * 0.22), (mid_x, iy1 + icon * 0.58)], fill=colour, width=mark)
            draw.ellipse((mid_x - mark * 0.6, iy1 + icon * 0.7, mid_x + mark * 0.6, iy1 + icon * 0.7 + mark * 1.2),
                         fill=colour)
        else:
            draw.text((mid_x, mid_y), "?", font=_font(round(icon * 0.75)), fill=colour, anchor="mm")

        if text:
            draw.text((ix1 + icon + pad * 0.8, py1 + pill_h / 2), text, font=font,
                      fill=(255, 255, 255), anchor="lm")

    if stamp:
        small = _font(max(11, round(height / 50)), bold=False)
        text_width = draw.textlength(stamp, font=small)
        box_h = round(small.size * 1.7)
        bx1 = width - text_width - box_h
        draw.rounded_rectangle((bx1 - 8, 10, width - 10, 10 + box_h), radius=6, fill=(15, 23, 42))
        draw.text((bx1 + (text_width + box_h - 8) / 2 - 4, 10 + box_h / 2), stamp, font=small,
                  fill=(255, 255, 255), anchor="mm")

    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR), scale
