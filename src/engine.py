"""Monitoring engine for the web dashboard.

Runs detection on a background thread and keeps what the dashboard shows: the latest picture
(as JPEG), the people seen with a small photo and their status, alerts with their picture and
video clip, statistics and recording state.
"""

from __future__ import annotations

import itertools
import threading
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.camera import BrowserFeed, SourceError, VideoSource, describe_source, is_phone
from src.config import Config
from src.event_logger import EventLogger
from src.helmet_classifier import Status
from src.recorder import AlertClipRecorder, VideoRecorder
from src.renderer import PersonView, head_area, render_dashboard

MAX_PEOPLE_CARDS = 30
MAX_ALERTS = 50
THUMBNAIL_EVERY_SECONDS = 1.5
FPS_WARMUP_FRAMES = 10


def _jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return data.tobytes() if ok else b""


def _thumbnail(frame: np.ndarray, person: PersonView, head_ratio: float) -> bytes:
    """Square photo around the person's head and shoulders."""
    x1, y1, x2, y2 = head_area(person, head_ratio)
    size = max(x2 - x1, y2 - y1) * 1.6
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + size * 0.12
    height, width = frame.shape[:2]
    left, top = int(max(0, cx - size / 2)), int(max(0, cy - size / 2))
    right, bottom = int(min(width, cx + size / 2)), int(min(height, cy + size / 2))
    if right - left < 4 or bottom - top < 4:
        return b""
    crop = cv2.resize(frame[top:bottom, left:right], (160, 160))
    return _jpeg(crop, 85)


class MonitorEngine:
    def __init__(self, config: Config, detector_factory=None):
        self.config = config
        self.browser_feed = BrowserFeed()
        self._detector_factory = detector_factory
        self.detector = None

        self._lock = threading.Lock()
        self._frame_ready = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ids = itertools.count(1)

        self.jpeg: bytes = b""
        self.frame_number = 0
        self.want_recording = config.recording_enabled
        self._reset_session()

    # ---------- Control (called from the web server) ----------

    def start(self, source: str | None = None) -> None:
        self.stop()
        if source is not None:
            self.config.source = str(source)
        self._stop.clear()
        self._reset_session()
        self._set(state="starting", message=f"Opening {describe_source(self.config.source)}...")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=15)
            self._thread = None

    def dismiss(self, alert_id: int) -> bool:
        with self._lock:
            for alert in self.alerts:
                if alert["id"] == alert_id:
                    alert["dismissed"] = True
                    return True
        return False

    def set_recording(self, on: bool) -> None:
        self.want_recording = on

    def set_alerts_enabled(self, on: bool) -> None:
        self.config.alerts_enabled = on
        if self._pipeline is not None:
            self._pipeline.alerts.enabled = on

    def snapshot(self) -> str | None:
        """Save the current view to the snapshot folder. Returns the file name."""
        with self._lock:
            data = self.jpeg
        if not data:
            return None
        folder = Path(self.config.snapshot_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"snapshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')[:-3]}.jpg"
        path.write_bytes(data)
        return path.name

    def wait_frame(self, after: int, timeout: float = 2.0) -> tuple[int, bytes]:
        with self._frame_ready:
            if self.frame_number <= after:
                self._frame_ready.wait(timeout)
            return self.frame_number, self.jpeg

    def state(self) -> dict:
        with self._lock:
            people = sorted(self.people.values(), key=lambda card: card["first_seen_ts"], reverse=True)
            return {
                **self.status,
                "source": self.config.source,
                "source_label": describe_source(self.config.source),
                "camera_id": self.config.camera_id,
                "alerts_enabled": self.config.alerts_enabled,
                "recording": self.want_recording,
                "recording_file": self._recording_file,
                "alert_recording": self.config.alert_recording_enabled,
                "people": [{k: v for k, v in card.items() if k != "first_seen_ts"} for card in people],
                "alerts": list(reversed(self.alerts)),
                "phone_connected": self.browser_feed.connected,
            }

    def thumbnail(self, track_id: int) -> bytes:
        with self._lock:
            return self.thumbs.get(track_id, b"")

    def alert_image(self, alert_id: int) -> bytes:
        with self._lock:
            return self.alert_images.get(alert_id, b"")

    # ---------- Internal ----------

    def _reset_session(self) -> None:
        with self._lock:
            self.people: dict[int, dict] = {}
            self.thumbs: dict[int, bytes] = {}
            self._thumb_time: dict[int, float] = {}
            self.alerts: list[dict] = []
            self.alert_images: dict[int, bytes] = {}
            self._pipeline = None
            self._recording_file = ""
            self.status = {
                "state": "stopped", "message": "", "fps": 0.0, "live": True, "started_at": None,
                "counts": {"people": 0, "helmet": 0, "no_helmet": 0, "unknown": 0},
                "summary": None, "clip_recording": False,
            }

    def _set(self, **values) -> None:
        with self._lock:
            self.status.update(values)

    def _load_detector(self):
        if self.detector is None:
            self._set(state="loading", message="Loading the detection models...")
            if self._detector_factory is not None:
                self.detector = self._detector_factory(self.config)
            else:
                from src.detector import Detector

                self.detector = Detector(self.config)
        return self.detector

    def _run(self) -> None:
        video = None
        try:
            detector = self._load_detector()
        except Exception as error:  # noqa: BLE001 - shown on the dashboard
            self._set(state="error", message=str(error))
            return

        from src.pipeline import SafetyPipeline

        config = replace(self.config)
        logger = EventLogger(
            config.log_dir, config.camera_id, config.logging_enabled, config.log_retention_days,
            False, config.snapshot_dir, config.snapshot_retention_days,
        )
        pipeline = SafetyPipeline(config, detector, logger=logger)
        pipeline.alerts.enabled = self.config.alerts_enabled
        self._pipeline = pipeline

        recorder = VideoRecorder(config.recording_dir, config.camera_id, config.recording_retention_days)
        clips = AlertClipRecorder(config.recording_dir, config.camera_id, config.alert_record_pre_seconds,
                                  config.alert_record_post_seconds, config.alert_record_max_seconds,
                                  config.alert_recording_enabled)
        pending_clip_alerts: list[int] = []

        phone = is_phone(config.source)
        try:
            if not phone:
                video = VideoSource(config.source, config.capture_width, config.capture_height,
                                    config.reconnect_attempts)
                video.open()
        except SourceError as error:
            self._set(state="error", message=str(error))
            return

        live = phone or video.is_live
        file_fps = 0.0 if live else video.fps
        self._set(state="running", live=live, started_at=time.time(),
                  message=f"Monitoring {describe_source(config.source)}")
        logger.log("SESSION_START", status=describe_source(config.source))

        fps, last_time, processed, phone_frame = 0.0, time.perf_counter(), 0, 0
        try:
            while not self._stop.is_set():
                # ----- Next frame -----
                if phone:
                    phone_frame, frame = self.browser_feed.wait_frame(phone_frame, timeout=0.5)
                    if frame is None:
                        self._set(message="Waiting for the phone... open the phone page and press Start.")
                        continue
                else:
                    frame = video.read()
                    if frame is None:
                        self._set(state="ended", message="Video finished.")
                        break

                try:
                    result = pipeline.process(frame)
                except Exception as error:  # noqa: BLE001 - one bad frame must not stop monitoring
                    print(f"[warning] Skipped a frame: {error}")
                    continue
                processed += 1

                now_perf = time.perf_counter()
                instant = 1.0 / max(now_perf - last_time, 1e-6)
                fps = instant if fps == 0 else fps * 0.9 + instant * 0.1
                last_time = now_perf
                # Frames are recorded as fast as they are processed, so recordings play at real speed.
                record_fps = fps if processed >= FPS_WARMUP_FRAMES or not file_fps else file_fps
                now = time.time()

                stamp = f"{config.camera_id}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                view, _ = render_dashboard(frame, result.people, config.display_width, stamp)
                view_jpeg = _jpeg(view)

                # ----- People cards and thumbnails -----
                self._update_people(frame, result.people, now)

                # ----- Alerts and alert clips -----
                for alert in result.alerts:
                    alert_id = next(self._ids)
                    clip_path = clips.trigger(now, record_fps) if processed >= 2 else None
                    with self._lock:
                        self.alert_images[alert_id] = view_jpeg
                        self.alerts.append({
                            "id": alert_id,
                            "track_id": alert.track_id,
                            "time": datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S"),
                            "date": datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d"),
                            "confidence": round(alert.confidence, 2) if alert.confidence else None,
                            "message": f"Person {alert.track_id} is not wearing a helmet.",
                            "dismissed": False,
                            "clip": None,
                            "clip_recording": clip_path is not None,
                        })
                        if len(self.alerts) > MAX_ALERTS:
                            removed = self.alerts.pop(0)
                            self.alert_images.pop(removed["id"], None)
                    if clip_path is not None:
                        pending_clip_alerts.append(alert_id)
                if result.counts.no_helmet > 0:
                    clips.extend(now)
                finished = clips.push(view, now)
                if finished is not None or (not clips.recording and pending_clip_alerts):
                    self._attach_clip(pending_clip_alerts, finished)
                    if finished is not None:
                        logger.log("ALERT_CLIP", snapshot=str(finished))
                    pending_clip_alerts = []

                # ----- Manual recording -----
                if self.want_recording and not recorder.recording and processed >= FPS_WARMUP_FRAMES:
                    try:
                        path = recorder.start(view, record_fps)
                        logger.log("RECORDING_START", snapshot=str(path))
                        with self._lock:
                            self._recording_file = path.name
                    except OSError as error:
                        self.want_recording = False
                        self._set(message=str(error))
                elif not self.want_recording and recorder.recording:
                    path = recorder.stop()
                    logger.log("RECORDING_STOP", snapshot=str(path or ""))
                    with self._lock:
                        self._recording_file = ""
                recorder.write(view)

                # ----- Publish -----
                counts = result.counts
                with self._lock:
                    self.status.update(
                        fps=round(fps, 1),
                        counts={"people": counts.people, "helmet": counts.helmet,
                                "no_helmet": counts.no_helmet, "unknown": counts.unknown},
                        summary=pipeline.statistics.summary(),
                        clip_recording=clips.recording,
                    )
                with self._frame_ready:
                    self.jpeg = view_jpeg
                    self.frame_number += 1
                    self._frame_ready.notify_all()

                if not live and file_fps > 0:
                    # Video files: play at their own speed instead of as fast as possible.
                    delay = 1.0 / file_fps - (time.perf_counter() - now_perf)
                    if delay > 0:
                        time.sleep(delay)
        except SourceError as error:
            self._set(state="error", message=str(error))
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            self._set(state="error", message=f"Monitoring stopped: {error}")
        finally:
            finished = clips.stop()
            if finished is not None or pending_clip_alerts:
                self._attach_clip(pending_clip_alerts, finished)
            path = recorder.stop()
            if path:
                logger.log("RECORDING_STOP", snapshot=str(path))
            if video is not None:
                video.release()
            summary = pipeline.statistics.summary()
            logger.log("SESSION_END", status=str(summary))
            with self._lock:
                self._recording_file = ""
                self.status["summary"] = summary
                self.status["clip_recording"] = False
                if self.status["state"] in ("running", "starting"):
                    self.status.update(state="stopped", message="Monitoring stopped.")

    def _attach_clip(self, alert_ids: list[int], path: Path | None) -> None:
        with self._lock:
            for alert in self.alerts:
                if alert["id"] in alert_ids:
                    alert["clip"] = path.name if path else None
                    alert["clip_recording"] = False

    def _update_people(self, frame: np.ndarray, people: list[PersonView], now: float) -> None:
        clock = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        with self._lock:
            visible = {person.track_id for person in people}
            for card in self.people.values():
                card["in_view"] = card["id"] in visible

            for person in people:
                card = self.people.get(person.track_id)
                if card is None:
                    card = {"id": person.track_id, "first_seen": clock, "first_seen_ts": now,
                            "status": None, "ever_no_helmet": False}
                    self.people[person.track_id] = card
                changed = card["status"] != person.status.value
                card.update(status=person.status.value, status_text=person.status.text,
                            last_seen=clock, in_view=True)
                if person.status == Status.NO_HELMET:
                    card["ever_no_helmet"] = True

                last_thumb = self._thumb_time.get(person.track_id, 0.0)
                if changed or now - last_thumb >= THUMBNAIL_EVERY_SECONDS:
                    data = _thumbnail(frame, person, self.config.head_region_ratio)
                    if data:
                        self.thumbs[person.track_id] = data
                        self._thumb_time[person.track_id] = now
                card["has_photo"] = person.track_id in self.thumbs

            if len(self.people) > MAX_PEOPLE_CARDS:
                oldest = sorted((card for card in self.people.values() if not card["in_view"]),
                                key=lambda card: card["first_seen_ts"])
                for card in oldest[:len(self.people) - MAX_PEOPLE_CARDS]:
                    self.people.pop(card["id"], None)
                    self.thumbs.pop(card["id"], None)
                    self._thumb_time.pop(card["id"], None)
