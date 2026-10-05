from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

from obs_clock_detector import crop_roi
from window_capture import WindowCapture


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python save_template_from_roi.py <number>")
        raise SystemExit(2)

    label = sys.argv[1]
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config.json"

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    capture = WindowCapture(str(config["window_title_contains"]))
    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        raise RuntimeError("Kein Frame vom OBS-Projektor empfangen.")

    roi_frame = crop_roi(frame, config["roi"])
    output_path = base_dir / config["templates_dir"] / f"stage_{label}.png"
    cv2.imwrite(str(output_path), roi_frame)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
