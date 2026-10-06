"""Live monitoring loop: reads frames, runs the pipeline, shows the window and handles keys."""

from __future__ import annotations

import time
import traceback

import cv2

from src.camera import SourceError, VideoSource, describe_source
from src.config import Config
from src.detector import Detector, ModelLoadError
from src.event_logger import EventLogger
from src.pipeline import SafetyPipeline
from src.renderer import Overlay, render

WINDOW_NAME = "Safety Helmet Detection"
MAX_CONSECUTIVE_ERRORS = 30


def run_monitoring(config: Config) -> dict | None:
    """Run until the user stops it or the source ends. Returns the session summary."""
    print(f"Loading models (the first run downloads {config.person_model})...")
    try:
        detector = Detector(config)
    except ModelLoadError as error:
        print(f"\n[error] {error}\n")
        return None

    for warning in detector.warnings:
        print(f"[warning] {warning}")

    logger = EventLogger(
        config.log_dir, config.camera_id, config.logging_enabled, config.log_retention_days,
        config.snapshots_enabled, config.snapshot_dir, config.snapshot_retention_days,
    )
    pipeline = SafetyPipeline(config, detector, logger=logger)

    video = VideoSource(config.source, config.capture_width, config.capture_height,
                        config.reconnect_attempts)
    try:
        video.open()
    except SourceError as error:
        print(f"\n[error] {error}\n")
        return None

    print(f"Monitoring {describe_source(config.source)}. Press Q in the video window to stop.")
    logger.log("SESSION_START", status=describe_source(config.source))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    paused = False
    frame = None
    result = None
    errors = 0
    fps = 0.0
    last_time = time.perf_counter()
    messages = list(detector.warnings)

    try:
        while True:
            if not paused:
                try:
                    frame = video.read()
                except SourceError as error:
                    print(f"\n[error] {error}")
                    messages.append(str(error))
                    break

                if frame is None:
                    print("Video finished.")
                    break

                try:
                    result = pipeline.process(frame)
                    errors = 0
                except Exception as error:  # noqa: BLE001 - one bad frame must not stop monitoring
                    errors += 1
                    print(f"[warning] Skipped a frame: {error}")
                    if errors >= MAX_CONSECUTIVE_ERRORS:
                        traceback.print_exc()
                        print("[error] Too many failed frames in a row. Stopping.")
                        break
                    continue

                for alert in result.alerts:
                    print(f"[ALERT] Person #{alert.track_id} without helmet "
                          f"({time.strftime('%H:%M:%S', time.localtime(alert.timestamp))})")

                now = time.perf_counter()
                instant = 1.0 / max(now - last_time, 1e-6)
                fps = instant if fps == 0 else fps * 0.9 + instant * 0.1
                last_time = now

            if frame is not None and result is not None:
                overlay = Overlay(
                    people=result.people,
                    counts=result.counts,
                    stats=pipeline.statistics,
                    fps=fps,
                    paused=paused,
                    alerts_enabled=pipeline.alerts.enabled,
                    snapshots_enabled=logger.snapshots_enabled,
                    alert_flash=pipeline.alerts.recently_alerted(),
                    messages=messages,
                )
                cv2.imshow(WINDOW_NAME, render(frame, overlay, config.display_width))

            key = cv2.waitKey(1 if not paused else 50) & 0xFF

            # Window closed with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                last_time = time.perf_counter()
            elif key in (ord("s"), ord("S")) and frame is not None:
                path = logger.save_snapshot(frame, "manual")
                if path:
                    print(f"Snapshot saved: {path}")
                    logger.log("MANUAL_SNAPSHOT", snapshot=path)
            elif key in (ord("a"), ord("A")):
                pipeline.alerts.enabled = not pipeline.alerts.enabled
                print("Alerts", "enabled" if pipeline.alerts.enabled else "disabled")
            elif key in (ord("r"), ord("R")):
                pipeline.reset()
                print("Statistics reset.")
    finally:
        video.release()
        cv2.destroyAllWindows()

    summary = pipeline.statistics.summary()
    logger.log("SESSION_END", status=str(summary))
    return summary


def analyze_image(config: Config, image_path: str, output_path: str | None = None,
                  show: bool = True) -> dict | None:
    """Detect helmets on one image. No temporal smoothing: each person is decided from this image."""
    from pathlib import Path

    from src.association import associate
    from src.detector import HELMET, NO_HELMET, PERSON
    from src.helmet_classifier import frame_status
    from src.renderer import PersonView
    from src.statistics import SafetyStatistics

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[error] Cannot read image: {image_path}")
        return None

    try:
        detector = Detector(config)
    except ModelLoadError as error:
        print(f"[error] {error}")
        return None

    detections = detector.detect(frame)
    persons = [det for det in detections if det.label == PERSON]
    heads = [det for det in detections if det.label in (HELMET, NO_HELMET)]
    evidence = associate([det.box for det in persons], heads, config.head_region_ratio,
                         config.association_min_overlap, config.association_margin)

    people = []
    statuses = {}
    for number, (person, head) in enumerate(zip(persons, evidence), start=1):
        status = frame_status(head, person.confidence, config)
        statuses[number] = status
        people.append(PersonView(number, person.box, status, person.confidence, head.box, head.confidence))

    stats = SafetyStatistics()
    counts = stats.update(statuses)
    image = render(frame, Overlay(people, counts, stats, messages=detector.warnings), config.display_width)

    output = Path(output_path) if output_path else Path(image_path).with_name(Path(image_path).stem + "_result.jpg")
    cv2.imwrite(str(output), image)
    print(f"People: {counts.people}  Helmet: {counts.helmet}  No Helmet: {counts.no_helmet}  "
          f"Unknown: {counts.unknown}")
    print(f"Result saved to {output}")

    if show:
        cv2.imshow(WINDOW_NAME, image)
        print("Press any key in the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {"people": counts.people, "helmet": counts.helmet,
            "no_helmet": counts.no_helmet, "unknown": counts.unknown, "output": str(output)}
