"""Download a ready-trained hard-hat model so the app works before you train your own.

Model: keremberke/yolov8n-hard-hat-detection (Hugging Face), YOLOv8n,
classes: Hardhat, NO-Hardhat; reported mAP50 0.836 on its validation set.
It was trained on a public construction-site dataset; validate it on your own camera
before relying on it (see docs/model_evaluation.md).

Usage:  python scripts/download_model.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/keremberke/yolov8n-hard-hat-detection/resolve/main/best.pt"
TARGET = Path(__file__).resolve().parent.parent / "models" / "helmet_model.pt"


def main() -> int:
    if TARGET.exists() and "--force" not in sys.argv:
        print(f"{TARGET} already exists. Use --force to download again.")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(".part")
    print(f"Downloading {URL}")

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(100, blocks * block_size * 100 // total)
            print(f"\r  {done}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(URL, temporary, progress)
    except Exception as error:  # noqa: BLE001
        temporary.unlink(missing_ok=True)
        print(f"\nDownload failed: {error}")
        print("Check your internet connection, or train your own model with scripts/train.py.")
        return 1

    temporary.replace(TARGET)
    print(f"\nSaved to {TARGET}")

    try:
        from ultralytics import YOLO

        names = YOLO(str(TARGET)).names
        print(f"Model classes: {list(names.values())}")
    except Exception as error:  # noqa: BLE001
        print(f"Downloaded, but the model could not be loaded for a check: {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
