"""Train (fine-tune) a YOLO helmet model on datasets/data.yaml.

Usage:
    python scripts/train.py                       # defaults below
    python scripts/train.py --epochs 100 --model yolo11s.pt --batch 8

The best weights are copied to models/helmet_model.pt and the training settings are written
to models/helmet_model_info.yaml so the model version is documented.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a helmet detection model")
    parser.add_argument("--data", default=str(ROOT / "datasets" / "data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="Starting weights (yolo11n.pt is fast, yolo11s.pt more accurate)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="Use 8 or less on a 4 GB GPU such as a GTX 1650")
    parser.add_argument("--device", default="", help='"" = auto, "cpu", or "0" for the first GPU')
    parser.add_argument("--name", default="helmet")
    args = parser.parse_args()

    data = Path(args.data)
    if not data.exists():
        print(f"Dataset file not found: {data}. See datasets/README.md.")
        return 1

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=False,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        print("Training finished but best.pt was not found.")
        return 1

    target = ROOT / "models" / "helmet_model.pt"
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best, target)

    info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "base_model": args.model,
        "dataset": str(data),
        "epochs": args.epochs,
        "image_size": args.imgsz,
        "batch": args.batch,
        "classes": list(YOLO(str(target)).names.values()),
        "run_folder": str(results.save_dir),
    }
    with (ROOT / "models" / "helmet_model_info.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(info, file, sort_keys=False)

    print(f"\nModel saved to {target}")
    print("Next: python scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
