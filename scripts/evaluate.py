"""Model evaluation: precision, recall, mAP per class and speed (FPS / latency).

Usage:
    python scripts/evaluate.py                          # accuracy on the dataset test/val split + speed
    python scripts/evaluate.py --video videos/site.mp4  # also measure speed on a real video
    python scripts/evaluate.py --speed-only --source 0  # speed only, on the webcam

Writes the results to docs/model_evaluation.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def accuracy(model_path: str, data: str, split: str, device: str) -> list[str]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=data, split=split, device=device or None, verbose=False)
    box = metrics.box

    lines = [
        f"Split: `{split}`",
        "",
        "| Class | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|",
    ]
    for index, class_id in enumerate(box.ap_class_index):
        precision, recall, map50, map5095 = box.class_result(index)
        lines.append(f"| {model.names[int(class_id)]} | {precision:.3f} | {recall:.3f} | {map50:.3f} | {map5095:.3f} |")
    lines.append(f"| **all** | {box.mp:.3f} | {box.mr:.3f} | {box.map50:.3f} | {box.map:.3f} |")
    return lines


def speed(source: str, frames: int) -> list[str]:
    import cv2

    from src.camera import VideoSource
    from src.config import load_config
    from src.detector import Detector
    from src.pipeline import SafetyPipeline

    config = load_config(ROOT / "config.yaml")
    config.source = source
    config.logging_enabled = False
    config.alert_sound = False
    pipeline = SafetyPipeline(config, Detector(config))

    times = []
    with VideoSource(source) as video:
        for _ in range(frames):
            frame = video.read()
            if frame is None:
                break
            start = time.perf_counter()
            pipeline.process(frame)
            times.append(time.perf_counter() - start)
    cv2.destroyAllWindows()

    if len(times) <= 5:
        return [f"Not enough frames read from {source} to measure speed."]

    times = times[5:]  # Skip warm-up frames.
    average = sum(times) / len(times)
    worst = sorted(times)[int(len(times) * 0.95) - 1]
    return [
        f"Source: `{source}`, frames measured: {len(times)}",
        "",
        f"- Average latency per frame (whole pipeline): **{average * 1000:.1f} ms**",
        f"- 95th percentile latency: {worst * 1000:.1f} ms",
        f"- Throughput: **{1 / average:.1f} FPS** (target 20-30 FPS)",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the helmet model")
    parser.add_argument("--model", default=str(ROOT / "models" / "helmet_model.pt"))
    parser.add_argument("--data", default=str(ROOT / "datasets" / "data.yaml"))
    parser.add_argument("--split", default="test", help="test or val")
    parser.add_argument("--device", default="")
    parser.add_argument("--video", help="Video file for the speed test")
    parser.add_argument("--source", default=None, help="Webcam number for the speed test")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--speed-only", action="store_true")
    args = parser.parse_args()

    report = [
        "# Model Evaluation Report",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Model: `{args.model}`",
    ]

    info_file = Path(args.model).with_name("helmet_model_info.yaml")
    if info_file.exists():
        info = yaml.safe_load(info_file.read_text(encoding="utf-8"))
        report += ["", "## Model version", "", "```yaml", yaml.safe_dump(info, sort_keys=False).strip(), "```"]

    if not args.speed_only:
        if not Path(args.data).exists():
            print(f"Dataset not found ({args.data}); skipping accuracy. See datasets/README.md.")
        else:
            data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
            split = args.split if args.split in data else "val"
            print(f"Measuring accuracy on the '{split}' split...")
            report += ["", "## Accuracy", ""] + accuracy(args.model, args.data, split, args.device)

    speed_source = args.video or args.source
    if speed_source is not None:
        print(f"Measuring speed on {speed_source}...")
        report += ["", "## Speed", ""] + speed(speed_source, args.frames)

    report += [
        "",
        "## Notes",
        "",
        "- Precision for no_helmet matters most for avoiding false alerts; recall for no_helmet for not missing violations.",
        "- Record the camera position, lighting and hardware used, and any known failure cases, below.",
        "",
    ]

    output = ROOT / "docs" / "model_evaluation.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
