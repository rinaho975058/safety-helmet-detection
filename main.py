"""Safety Helmet Detection - application entry point.

Usage:
    python main.py                         Menu (Start, Choose camera, Settings, Statistics, Exit)
    python main.py --start                 Start monitoring straight away
    python main.py --start --source video.mp4
    python main.py --start --record        Start monitoring and record the view to recordings/
    python main.py --image photo.jpg       Analyse a single image
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

from src.camera import check_source, describe_source, find_webcam, list_screens, list_webcams
from src.config import DEFAULT_CONFIG_PATH, Config, convert_value, load_config, save_config

DISCLAIMER = (
    "Note: this is a demonstration and decision-support tool. It is not 100% accurate and must not\n"
    "be the only way workplace safety is enforced. Always confirm alerts with a person."
)

DEMO_VIDEO = Path(__file__).resolve().parent / "samples" / "demo.mp4"


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


def start(config: Config, interactive: bool = False) -> dict | None:
    from src.monitor import run_monitoring

    print(f"Checking {describe_source(config.source)}...")
    problem = check_source(config.source)
    is_webcam = str(config.source).strip().isdigit()
    if problem and is_webcam:
        print(f"{problem}\nLooking for another webcam...")
        found = find_webcam()
        if found is not None:
            print(f"Found webcam {found}. Using it for this session "
                  f"(set source: \"{found}\" in Settings to keep it).")
            config = replace(config, source=str(found))
            problem = None
    if problem:
        print(f"\n[error] {problem}")
        if is_webcam:
            print("No working webcam was found. Plug in a USB webcam, or use a phone/IP camera "
                  "by setting source to its http:// or rtsp:// address.")
        if is_webcam and interactive and DEMO_VIDEO.exists():
            answer = input(f"Use the demo video {DEMO_VIDEO.as_posix()} instead? (Y/n): ").strip().lower()
            if answer in ("", "y", "yes"):
                demo_config = replace(config, source=str(DEMO_VIDEO))
                summary = run_monitoring(demo_config)
                print_summary(summary)
                return summary
        print("Change the input source in Settings (or config.yaml) and try again.")
        if is_webcam and DEMO_VIDEO.exists():
            print(f"To try the demo video: python main.py --start --source {DEMO_VIDEO.as_posix()}")
        print()
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


CAMERA_URL_EXAMPLES = (
    "  CCTV / IP camera : rtsp://user:password@192.168.1.10:554/stream1\n"
    "  Phone (IP Webcam): http://192.168.1.20:8080/video\n"
    "  Phone (DroidCam) : http://192.168.1.20:4747/video"
)


def camera_menu(config: Config, config_path: str) -> Config:
    """Pick the input: a webcam on this computer, a network camera, or a video file."""
    print("\n--- Choose camera ---")
    print(f"Current input: {describe_source(config.source)}")
    print("1. Webcam on this computer (USB or built-in, also DroidCam on Windows)")
    print("2. IP / CCTV / phone camera over the network")
    print("3. Video file")
    print("4. This computer's screen (e.g. a CCTV viewer program, video call or mirrored phone)")
    print("0. Back")
    choice = input("Choose 0-4: ").strip()

    if choice == "1":
        print("Looking for webcams...")
        webcams = list_webcams()
        if not webcams:
            print("No webcam found on this computer. Plug one in (or start DroidCam) and try again,\n"
                  "or choose option 2 to use a network camera.")
            return config
        if len(webcams) == 1:
            source = str(webcams[0])
            print(f"Found webcam {source}.")
        else:
            print("Found webcams: " + ", ".join(str(index) for index in webcams))
            source = input(f"Webcam number [{webcams[0]}]: ").strip() or str(webcams[0])
            if not source.isdigit() or int(source) not in webcams:
                print("That webcam was not found.")
                return config
    elif choice == "2":
        print("Enter the camera address. Examples:")
        print(CAMERA_URL_EXAMPLES)
        source = input("Address: ").strip().strip('"')
        if not source:
            return config
        if "://" not in source:
            print("The address must start with rtsp://, http:// or https://.")
            return config
    elif choice == "3":
        source = input("Video file path: ").strip().strip('"')
        if not source:
            return config
    elif choice == "4":
        source = choose_screen()
        if not source:
            return config
    else:
        return config

    print(f"Testing {describe_source(source)}...")
    problem = check_source(source)
    if problem:
        print(f"[error] {problem}\nThe input was not changed.")
        if choice == "2":
            print("Check that the camera is on, on the same network as this computer, and that the\n"
                  "address, username and password are right. You can open the same address in VLC to test it.")
        return config

    config.source = source
    if choice == "2":
        camera_id = input(f"Camera name for the log [{config.camera_id}]: ").strip()
        if camera_id:
            config.camera_id = camera_id
    save_config(config, config_path)
    print(f"OK. Input is now {describe_source(source)} (saved to {config_path}).")
    return config


def choose_screen() -> str | None:
    """Ask for a monitor and, optionally, an area of it. Returns a screen source or None."""
    screens = list_screens()
    if not screens:
        print("Cannot capture the screen. Install the requirements (pip install -r requirements.txt).")
        return None

    for screen in screens:
        main_text = " (main)" if screen["number"] == 1 else ""
        print(f"  Screen {screen['number']}: {screen['width']}x{screen['height']}{main_text}")
    number = 1
    if len(screens) > 1:
        text = input("Screen number [1]: ").strip() or "1"
        if not text.isdigit() or not 1 <= int(text) <= len(screens):
            print("That screen was not found.")
            return None
        number = int(text)

    print("Tip: the monitoring window must not be inside the captured area, or it will film itself.\n"
          "Select only the part that shows the camera picture, or put the app window on another screen.")
    if input("Select an area with the mouse? (Y/n): ").strip().lower() in ("n", "no"):
        return "screen" if number == 1 else f"screen:{number}"

    area = select_screen_area(screens[number - 1])
    if area is None:
        print("No area selected.")
        return None
    return "screen:" + ",".join(str(value) for value in area)


def select_screen_area(screen: dict) -> tuple[int, int, int, int] | None:
    """Show a screenshot and let the user drag a rectangle. Returns (left, top, width, height)."""
    import cv2

    from src.camera import ScreenCapture

    capture = ScreenCapture(f"screen:{screen['number']}")
    ok, shot = capture.read()
    capture.release()
    if not ok:
        return None

    scale = min(1.0, 1280 / shot.shape[1], 720 / shot.shape[0])
    preview = cv2.resize(shot, None, fx=scale, fy=scale) if scale < 1 else shot
    print("Drag a rectangle over the camera picture, then press Enter (C to cancel).")
    title = "Select area - drag, then press Enter"
    x, y, w, h = cv2.selectROI(title, preview, showCrosshair=False)
    cv2.destroyWindow(title)
    if w < 10 or h < 10:
        return None
    return (screen["left"] + round(x / scale), screen["top"] + round(y / scale),
            round(w / scale), round(h / scale))


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
        print("2. Choose camera")
        print("3. Analyse an image")
        print("4. Settings")
        print("5. Statistics")
        print("6. Exit")
        choice = input("Choose 1-6: ").strip()

        if choice == "1":
            last_summary = start(config, interactive=True) or last_summary
        elif choice == "2":
            config = camera_menu(config, config_path)
        elif choice == "3":
            from src.monitor import analyze_image

            path = input("Image path: ").strip().strip('"')
            if path:
                analyze_image(config, path)
        elif choice == "4":
            config = settings_menu(config, config_path)
        elif choice == "5":
            statistics_menu(config, last_summary)
        elif choice == "6":
            print("Goodbye.")
            return
        else:
            print("Please choose a number from 1 to 6.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Safety Helmet Detection")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Settings file (default: config.yaml)")
    parser.add_argument("--source", help="Webcam number, video file, rtsp:// URL or 'screen' (overrides config)")
    parser.add_argument("--model", help="Helmet model path (overrides config)")
    parser.add_argument("--start", action="store_true", help="Start monitoring without the menu")
    parser.add_argument("--record", action="store_true", help="Record the monitoring view to MP4 (overrides config)")
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
    if args.record:
        config.recording_enabled = True

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
