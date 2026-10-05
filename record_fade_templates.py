from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from obs_clock_detector import crop_roi
from window_capture import WindowCapture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="Template number, for example 2")
    parser.add_argument("--seconds", type=float, default=0.75, help="Burst duration")
    parser.add_argument("--samples", type=int, default=8, help="Frames saved from each burst")
    return parser.parse_args()


def normalize_roi_size(roi_frame, roi: dict):
    size = (int(roi["width"]), int(roi["height"]))
    return cv2.resize(roi_frame, size, interpolation=cv2.INTER_AREA)


def save_frames(frames: list, templates_dir: Path, label: str, sample_count: int) -> list[Path]:
    if not frames:
        return []

    sample_count = max(1, min(sample_count, len(frames)))
    indices = (
        [len(frames) // 2]
        if sample_count == 1
        else [
            round(index * (len(frames) - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths: list[Path] = []
    for variant_index, frame_index in enumerate(indices, start=1):
        output_path = templates_dir / f"{label}_fade_{timestamp}_{variant_index:02d}.png"
        cv2.imwrite(str(output_path), frames[frame_index])
        saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    with (base_dir / "config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)

    roi = config["roi"]
    templates_dir = base_dir / config["templates_dir"]
    templates_dir.mkdir(parents=True, exist_ok=True)
    capture = WindowCapture(str(config["window_title_contains"]))

    print("R: record fade burst | S: save one frame | Q: quit")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("No frame received from the OBS projector.")

            roi_frame = normalize_roi_size(crop_roi(frame, roi), roi)
            preview = cv2.resize(roi_frame, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
            cv2.putText(
                preview,
                f"Label {args.label} | R burst | S single | Q quit",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Fade Template Recorder", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = templates_dir / f"{args.label}_single_{timestamp}.png"
                cv2.imwrite(str(output_path), roi_frame)
                print(f"Saved {output_path.name}")

            if key == ord("r"):
                recorded_frames = []
                started_at = time.monotonic()
                while time.monotonic() - started_at < args.seconds:
                    ok, burst_frame = capture.read()
                    if not ok or burst_frame is None:
                        break
                    burst_roi = normalize_roi_size(crop_roi(burst_frame, roi), roi)
                    recorded_frames.append(burst_roi.copy())

                    burst_preview = cv2.resize(
                        burst_roi,
                        None,
                        fx=5,
                        fy=5,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    cv2.putText(
                        burst_preview,
                        "RECORDING",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Fade Template Recorder", burst_preview)
                    cv2.waitKey(1)

                saved = save_frames(recorded_frames, templates_dir, args.label, args.samples)
                print(f"Saved {len(saved)} fade variants for label {args.label}.")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
