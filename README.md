# Safety Helmet Detection

A computer-vision app that watches a webcam, CCTV stream, video file or image, finds people, and shows whether each one is wearing a safety helmet. It uses three statuses: **Helmet**, **No Helmet** or **Unknown**. It raises an alert when a No Helmet case stays stable, keeps statistics, and logs violation events.

Built with **Python + OpenCV + YOLO (Ultralytics)**.

> ⚠️ **This is a demonstration and decision-support tool.** It is not 100% accurate and must not be the only way safety is enforced, or be used to punish anyone based on one prediction. Validate it on the real camera and site before any operational use, and always have a person review alerts.

---

## 1. Installation (Windows)

1. **Install Python 3.11 or newer** from <https://www.python.org/downloads/>. On the first installer screen, tick **"Add python.exe to PATH"**.
2. Open a terminal in this folder (in VS Code: **Terminal → New Terminal**) and run:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Optional: use the NVIDIA GPU (much faster).** The default install runs on the CPU. For a GTX 1650 or other NVIDIA card, run:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
   python -c "import torch; print(torch.cuda.is_available())"   # should print True
   ```

4. **Get a helmet model.** Either download a ready-trained one (good for a first demo):

   ```bash
   python scripts/download_model.py
   ```

   or train your own (see section 7). Without a helmet model, the app still runs, but it only finds people and shows every status as Unknown.

5. **Check that everything works:** `pytest` should report all tests passed, and `python main.py --start --source samples/demo.mp4` should play the demo video with boxes.

**Setup problems**
- *PowerShell says running scripts is disabled* when you activate: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activating and use `.venv\Scripts\python.exe` in place of `python`.
- *pip fails with a "Long Path" hint*: clone the project into a short folder (for example `C:\projects\helmet`), or enable long paths in Windows.

## 2. Running

```bash
python main.py                                  # menu: Start, Choose camera, Analyse image, Settings, Statistics, Exit
python main.py --start                          # start monitoring the webcam directly
python main.py --start --source videos/site.mp4 # test with a video file
python main.py --start --source "rtsp://user:pass@192.168.1.10:554/stream1"
python main.py --start --source screen          # detect on this computer's screen
python main.py --image photo.jpg                # analyse one image -> photo_result.jpg
```

**Choosing a camera** (menu option 2). The camera is tested before it is saved to `config.yaml`.

| Camera | How to connect | Example input |
|---|---|---|
| Webcam on this computer (USB or built-in) | Plug it in; the app lists the webcams it finds | `0` |
| Phone as a webcam (DroidCam app + DroidCam Client on Windows) | Appears as a normal webcam | `0` or `1` |
| Phone over Wi-Fi ("IP Webcam" app on Android) | Same Wi-Fi as this computer | `http://192.168.1.20:8080/video` |
| CCTV / IP camera | Same network as this computer; RTSP address is in the camera's manual or web page | `rtsp://user:password@192.168.1.10:554/stream1` |
| Video file | Any MP4, AVI, MOV or MKV | `videos/site.mp4` |
| This computer's screen | Shows whatever is on screen: a CCTV viewer program, a video call, a phone mirrored to the PC. Select only the area with the camera picture (the app's own window must not be inside it) | `screen`, `screen:2`, or an area `screen:100,80,1280,720` |

If an address does not work in the app, open it in VLC (Media > Open Network Stream) to check the address, username and password.

The person model (`yolo11n.pt`) downloads automatically the first time.

**Keys in the video window**

| Key | Action |
|---|---|
| `Space` | Pause / resume |
| `S` | Save a snapshot |
| `V` | Start / stop recording video to `recordings/` |
| `A` | Turn alerts on / off |
| `R` | Reset statistics |
| `Q` / `Esc` | Stop (the session summary is printed) |

**On screen**
- **Boxes:** green = Helmet, red = No Helmet, amber = Unknown. Each person is labelled `#id status confidence`.
- **Top banner:** **SAFE**, or **ATTENTION** when someone currently has No Helmet.
- **Left panel:** current counts (people, helmet, no helmet, unknown), plus session totals, violation rate and alert count.
- **Alerts:** a red frame around the video, a beep (if enabled), a console message and a log entry.

## 3. Settings

All settings are in [`config.yaml`](config.yaml). Change them there, or use **Settings** in the menu.

| Setting | Meaning |
|---|---|
| `source` | `0` = webcam, a video path, or an `rtsp://` URL |
| `helmet_model`, `person_model` | Model files |
| `person_confidence`, `helmet_confidence`, `no_helmet_confidence` | Minimum confidence for each class |
| `head_region_ratio`, `association_min_overlap`, `association_margin` | How helmets are matched to people |
| `smoothing_window`, `status_min_ratio` | How stable a status must be before it changes |
| `alerts_enabled`, `alert_sound`, `alert_stable_frames`, `alert_cooldown_seconds` | Alert behaviour |
| `logging_enabled`, `log_retention_days` | Event log in `logs/` |
| `snapshots_enabled`, `snapshot_retention_days` | Save a frame on each alert to `snapshots/` (off by default) |
| `recording_enabled`, `recording_dir`, `recording_retention_days` | Record the monitoring view to MP4 in `recordings/` (off by default; `V` key or `--record` to start) |
| `derive_no_helmet` | Alternative design: a person with no helmet evidence counts as No Helmet |
| `display_width` | Window resolution |

## 4. How it works

### System architecture

```mermaid
flowchart LR
    CFG[Configuration Manager<br/>config.py] -.-> IN & DET & DEC & AL & LOG
    IN[Input Manager<br/>camera.py] --> DET[Detection Engine<br/>detector.py]
    DET --> TR[Tracking<br/>tracker.py]
    TR --> AS[Association<br/>association.py]
    DET --> AS
    AS --> DEC[Safety Decision Engine<br/>helmet_classifier.py]
    DEC --> AL[Alert Manager<br/>alert_manager.py]
    DEC --> ST[Statistics Manager<br/>statistics.py]
    AL --> LOG[Event Logger<br/>event_logger.py]
    DEC & AL & ST --> UI[Visualization<br/>renderer.py + monitor.py]
```

### Detection pipeline (per frame, [`src/pipeline.py`](src/pipeline.py))

```mermaid
flowchart TD
    A[1. Capture frame] --> B[2-3. YOLO inference<br/>helmet model + person model]
    B --> C[4. Drop detections below confidence thresholds]
    C --> D[5. People -> IoU tracker gives stable IDs]
    C --> E[6. Helmet / no_helmet detections]
    D & E --> F[7. Associate: helmet inside a person's head region]
    F --> G[Per-frame status: Helmet / No Helmet / Unknown]
    G --> H[8-9. Temporal smoothing -> stable status]
    H --> I[10. Draw boxes, labels, banner, stats]
    H --> J[11. Alert if No Helmet stable for N frames]
    H --> K[12. Update statistics]
    J --> L[Log event + optional snapshot]
```

### Safety status logic

| Status | When | Action |
|---|---|---|
| **HELMET** | Helmet evidence on the person's head is at least `helmet_confidence` | None |
| **NO HELMET** | `no_helmet` evidence is at least `no_helmet_confidence`, **and** stays stable across frames | Alert after `alert_stable_frames` |
| **UNKNOWN** | Low confidence, no head evidence, helmet between two overlapping people, or conflicting evidence | Never an automatic violation |

- **Association:** a helmet box belongs to a person when at least `association_min_overlap` of it lies inside the top `head_region_ratio` of the person's box. If it fits two people almost equally, both are set to Unknown rather than guessing.
- **Temporal stability:** each person keeps the last `smoothing_window` frame decisions. Their status only changes when a new status fills at least `status_min_ratio` of that window (hysteresis), so one odd frame cannot cause a violation. When a person leaves the scene, their state is reset.
- **Alerts:** one alert per continuous violation. There is a per-person `alert_cooldown_seconds`, so someone quickly taking a helmet off and on doesn't cause repeated alerts.

## 5. Privacy

- Video is processed locally, and nothing is uploaded.
- No faces or names are identified. People only get temporary numbers (`#3`) that reset every session.
- Video is only recorded when you ask for it (`V` key, `--record`, or `recording_enabled: true`). While recording, the screen shows `REC`, and MP4 files with the boxes and labels are saved to `recordings/`. Recordings show people's faces, so restrict access to that folder.
- The event log (`logs/events_YYYY-MM-DD.csv`) stores only: time, event type, temporary person number, status, confidence and camera name.
- Snapshots are **off by default**. When on, the screen shows `SNAPSHOTS ON`, and full frames are saved to `snapshots/`. These frames can show people's faces, so restrict access to that folder.
- Old logs, snapshots and recordings are deleted automatically after `log_retention_days`, `snapshot_retention_days` and `recording_retention_days`.

## 6. Tests

```bash
pytest
```

The automated tests cover class mapping and filtering, person-helmet association (including the ambiguous cases), status decisions, temporal smoothing, tracking, alerts (stability, de-duplication and cooldown), statistics, the event log, configuration validation, and a full pipeline run with a fake detector.

The tests that need a camera, the real model and real scenes (lighting, distance, occlusion, FPS) are in [`docs/test_plan.md`](docs/test_plan.md). Record their results there.

## 7. Training your own model

1. Prepare the dataset as described in [`datasets/README.md`](datasets/README.md). It uses three classes: `person`, `helmet` and `no_helmet`.
2. Train:

   ```bash
   python scripts/train.py --epochs 50 --batch 8
   ```

   The best model is copied to `models/helmet_model.pt`. Its settings are saved to `models/helmet_model_info.yaml`.
3. Evaluate. This measures precision, recall, mAP50 and mAP50-95 per class, plus FPS and latency, and writes [`docs/model_evaluation.md`](docs/model_evaluation.md):

   ```bash
   python scripts/evaluate.py --video videos/site.mp4
   ```

## 8. Project structure

```
├── main.py                  Entry point and menu
├── config.yaml              Settings
├── requirements.txt
├── src/
│   ├── camera.py            Input manager (webcam / video / RTSP, reconnect)
│   ├── detector.py          YOLO detection engine and class mapping
│   ├── tracker.py           IoU person tracker
│   ├── association.py       Person-helmet association
│   ├── helmet_classifier.py Status decision and temporal smoothing
│   ├── alert_manager.py     Stable alerts, cooldown, sound
│   ├── statistics.py        Counts and violation rate
│   ├── event_logger.py      CSV event log and snapshots
│   ├── renderer.py          Drawing the overlay
│   ├── pipeline.py          Per-frame pipeline
│   ├── monitor.py           Live loop and image analysis
│   ├── config.py            Configuration manager
│   └── utils.py             Geometry and file helpers
├── scripts/                 download_model.py, train.py, evaluate.py
├── models/                  helmet_model.pt (not in git)
├── datasets/                data.yaml and dataset README
├── tests/                   Automated tests
├── logs/  snapshots/        Created at runtime
└── docs/                    Test plan, evaluation report
```

## 9. Known limitations

- **Pretrained model:** the downloaded model was trained on public construction-site photos. It may miss unusual helmets, very small or distant people, people seen from above or behind, and dark or backlit scenes. Fine-tune it with images from your own camera.
- **Crowds:** when people overlap heavily, the simple IoU tracker can swap IDs. A helmet between two heads is shown as Unknown on purpose.
- **Caps and hoods:** a normal cap can be mistaken for a helmet, and the reverse. Check this on the test set.
- **Speed:** depends on the hardware. Use the GPU, `image_size: 480` or a smaller `capture_width` to raise FPS.

## 10. Future enhancements

Vest, shoe, glove and eye-protection detection; restricted-area detection; multi-camera monitoring; a web dashboard; email or messaging notifications; ByteTrack tracking; ONNX/TensorRT export for edge devices such as NVIDIA Jetson.
