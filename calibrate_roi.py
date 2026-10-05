from __future__ import annotations

import json
from pathlib import Path

import cv2

from window_capture import WindowCapture


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config.json"

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    capture = WindowCapture(str(config["window_title_contains"]))

    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError("Kein Frame vom OBS-Projektor empfangen.")

    x, y, width, height = cv2.selectROI("Uhr-Region auswaehlen", frame, showCrosshair=True)
    cv2.destroyAllWindows()

    if width == 0 or height == 0:
        print("Keine Region gespeichert.")
        return

    frame_height, frame_width = frame.shape[:2]
    config["roi"] = {
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
        "reference_width": int(frame_width),
        "reference_height": int(frame_height),
    }

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")

    print(
        f"ROI gespeichert: x={x}, y={y}, width={width}, height={height}, "
        f"Referenz={frame_width}x{frame_height}"
    )


if __name__ == "__main__":
    main()
