"""Safety Helmet Detection - application entry point.

Usage:
    python main.py                         Menu (Start, Settings, Statistics, Exit)
    python main.py --start                 Start monitoring straight away
    python main.py --start --source video.mp4
    python main.py --image photo.jpg       Analyse a single image
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import fields

from src.camera import check_source, describe_source
from src.config import DEFAULT_CONFIG_PATH, Config, convert_value, load_config, save_config

DISCLAIMER = (
    "Note: this is a demonstration and decision-support tool. It is not 100% accurate and must not\n"
    "be the only way workplace safety is enforced. Always confirm alerts with a person."
)


def print_summary(summary: dict | None) -> None:
    if not summary:
        return
    print("\n--- Session summary ---")
    print(f"Monitoring time : {summary['monitoring_seconds']} s")
    print(f"People seen     : {summary['total_people']}")
    print(f"Helmet          : {summary['helmet']}")
    print(f"No helmet       : {summary['no_helmet']}")
    print(f"Unknown         : {summary['unknown']}")
    print(f"Violation rate  : {summary['violation_rate'] * 100:.0f}%")
    print(f"Alerts          : {summary['alerts']}\n")


def start(config: Config) -> dict | None:
    from src.monitor import run_monitoring

    print(f"Checking {describe_source(config.source)}...")
    problem = check_source(config.source)
    if problem:
        print(f"\n[error] {problem}")
        print("Change the input source in Settings (or config.yaml) and try again.\n")
        return None

    summary = run_monitoring(config)
    print_summary(summary)
    return summary


def settings_menu(config: Config, config_path: str) -> Config:
    names = [item.name for item in fields(Config)]

    while True:
        print("\n--- Settings ---")
        for number, name in enumerate(names, start=1):
            print(f"{number:2}. {name:26} = {getattr(config, name)}")
        print(" 0. Back (changes are saved to config.yaml)")

        choice = input("\nSetting number to change: ").strip()
        if choice in ("", "0"):
            return config
        if not choice.isdigit() or not 1 <= int(choice) <= len(names):
            print("Please enter a number from the list.")
            continue

        name = names[int(choice) - 1]
        current = getattr(config, name)
        hint = " (comma separated)" if isinstance(current, list) else ""
        text = input(f"New value for {name}{hint} [{current}]: ").strip()
        if not text:
            continue

        try:
            setattr(config, name, convert_value(current, text))
        except ValueError as error:
            print(f"Invalid value: {error}")
            continue

        problems = config.validate()
        if problems:
            setattr(config, name, current)
            print("Not saved: " + " ".join(problems))
            continue

        save_config(config, config_path)
        print(f"Saved: {name} = {getattr(config, name)}")


def statistics_menu(config: Config, last_summary: dict | None) -> None:
    from src.event_logger import read_events

    print_summary(last_summary)

    events = read_events(config.log_dir)
    alerts = [event for event in events if event.get("event_type") == "NO_HELMET_ALERT"]

    print("--- Logged violation alerts ---")
    if not alerts:
        print(f"No violation alerts in {config.log_dir}/ yet.\n")
        return

    per_day = Counter(event["timestamp"][:10] for event in alerts)
    for day, count in sorted(per_day.items()):
        print(f"{day}: {count}")
    print(f"Total: {len(alerts)}")

    print("\nLatest alerts:")
    for event in alerts[-5:]:
        print(f"  {event['timestamp']}  person #{event['track_id']}  "
              f"confidence {event['confidence'] or '-'}  {event['camera_id']}")

    if input("\nShow chart? (y/N): ").strip().lower() == "y":
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed.")
            return
        days = sorted(per_day)
        plt.figure(figsize=(8, 4))
        plt.bar(days, [per_day[day] for day in days], color="#d9534f")
        plt.title("No-helmet alerts per day")
        plt.ylabel("Alerts")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


def menu(config: Config, config_path: str) -> None:
    last_summary = None

    while True:
        print("\n========== Safety Helmet Detection ==========")
        print(f"Input: {describe_source(config.source)}   Model: {config.helmet_model}")
        print("1. Start monitoring")
        print("2. Analyse an image")
        print("3. Settings")
        print("4. Statistics")
        print("5. Exit")
        choice = input("Choose 1-5: ").strip()

        if choice == "1":
            last_summary = start(config) or last_summary
        elif choice == "2":
            from src.monitor import analyze_image

            path = input("Image path: ").strip().strip('"')
            if path:
                analyze_image(config, path)
        elif choice == "3":
            config = settings_menu(config, config_path)
        elif choice == "4":
            statistics_menu(config, last_summary)
        elif choice == "5":
            print("Goodbye.")
            return
        else:
            print("Please choose 1, 2, 3, 4 or 5.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Safety Helmet Detection")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Settings file (default: config.yaml)")
    parser.add_argument("--source", help="Webcam number, video file or rtsp:// URL (overrides config)")
    parser.add_argument("--model", help="Helmet model path (overrides config)")
    parser.add_argument("--start", action="store_true", help="Start monitoring without the menu")
    parser.add_argument("--image", help="Analyse one image and save the result")
    parser.add_argument("--output", help="Where to save the analysed image")
    parser.add_argument("--no-show", action="store_true", help="With --image: save the result without opening a window")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (ValueError, OSError) as error:
        print(f"[error] {error}")
        return 1

    if args.source is not None:
        config.source = args.source
    if args.model:
        config.helmet_model = args.model

    print(DISCLAIMER)

    try:
        if args.image:
            from src.monitor import analyze_image
            return 0 if analyze_image(config, args.image, args.output, show=not args.no_show) else 1
        if args.start:
            return 0 if start(config) else 1
        menu(config, args.config)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
