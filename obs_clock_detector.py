from __future__ import annotations

import json
import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from window_capture import WindowCapture


@dataclass(frozen=True)
class Template:
    label: str
    image: np.ndarray
    hand_descriptor: np.ndarray
    hand_geometry_feature: np.ndarray
    display_image: np.ndarray
    source_size: tuple[int, int]


def numeric_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\d+", path.stem)
    return (int(match.group(0)) if match else 999999, path.name)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def label_from_filename(path: Path) -> str:
    match = re.search(r"\d+", path.stem)
    return match.group(0) if match else path.stem


def preprocess_for_matching(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    return cv2.equalizeHist(blurred)


def extract_hand_descriptor(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    low = float(np.percentile(gray, 35))
    high = float(np.percentile(gray, 98))
    normalized = np.clip((gray - low) / max(1.0, high - low), 0.0, 1.0)

    width, height = size
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    max_radius = min(width, height) * 0.34
    min_radius = min(width, height) * 0.05
    angle_count = 180
    radius_count = 28

    angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False, dtype=np.float32)
    radii = np.linspace(min_radius, max_radius, radius_count, dtype=np.float32)
    map_x = center_x + np.cos(angles)[:, None] * radii[None, :]
    map_y = center_y + np.sin(angles)[:, None] * radii[None, :]
    polar = cv2.remap(
        normalized,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    polar = np.maximum(polar - 0.35, 0.0)
    radial_weights = np.linspace(0.65, 1.35, radius_count, dtype=np.float32)
    return (polar * radial_weights[None, :]).astype(np.float32)


def hand_geometry_feature(descriptor: np.ndarray) -> np.ndarray:
    angular_profile = descriptor.mean(axis=1)
    radius_weights = np.linspace(0.5, 1.5, descriptor.shape[1], dtype=np.float32)
    radial_profile = (descriptor * radius_weights[None, :]).mean(axis=0)

    angular_profile = (angular_profile - np.mean(angular_profile)) / (
        np.std(angular_profile) + 1e-6
    )
    radial_profile = (radial_profile - np.mean(radial_profile)) / (
        np.std(radial_profile) + 1e-6
    )
    return np.concatenate((angular_profile, radial_profile * 0.35)).astype(np.float32)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-6:
        return -1.0
    return float(np.dot(first, second) / denominator)


def geometry_scores_with_alignment(
    roi_frame: np.ndarray,
    size: tuple[int, int],
    templates: list[Template],
    search_radius: int,
) -> list[tuple[str, float]]:
    offsets = (0,) if search_radius <= 0 else (-search_radius, 0, search_radius)
    best_scores = {template.label: -1.0 for template in templates}
    resized_roi = cv2.resize(roi_frame, size, interpolation=cv2.INTER_CUBIC)

    for offset_y in offsets:
        for offset_x in offsets:
            transform = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            aligned = cv2.warpAffine(
                resized_roi,
                transform,
                size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            descriptor = extract_hand_descriptor(aligned, size)
            feature = hand_geometry_feature(descriptor)

            for template in templates:
                score = cosine_similarity(feature, template.hand_geometry_feature)
                if score > best_scores[template.label]:
                    best_scores[template.label] = score

    return sorted(best_scores.items(), key=lambda item: item[1], reverse=True)


def make_discriminative_weight_map(templates: list[Template], strength: float) -> np.ndarray:
    grouped_images: dict[str, list[np.ndarray]] = {}
    for template in templates:
        grouped_images.setdefault(template.label, []).append(template.image.astype(np.float32))

    class_images = [
        np.mean(np.stack(images, axis=0), axis=0)
        for images in grouped_images.values()
    ]
    stack = np.stack(class_images, axis=0)
    weights = np.std(stack, axis=0)

    if float(np.max(weights)) <= 0:
        return np.ones_like(weights, dtype=np.float32)

    weights = weights / float(np.max(weights))
    weights = np.power(weights, strength)
    weights = 0.15 + (0.85 * weights)
    return weights.astype(np.float32)


def load_templates(
    templates_dir: Path,
    size: tuple[int, int],
    source_size: tuple[int, int] | None = None,
    warn_on_size_mismatch: bool = True,
) -> list[Template]:
    image_paths = sorted(
        [
            *templates_dir.glob("*.png"),
            *templates_dir.glob("*.jpg"),
            *templates_dir.glob("*.jpeg"),
            *templates_dir.glob("*.bmp"),
        ],
        key=numeric_sort_key,
    )

    if not image_paths:
        raise FileNotFoundError(f"Keine Vergleichsbilder in {templates_dir} gefunden.")

    templates: list[Template] = []
    for image_path in image_paths:
        display_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warnung: {image_path.name} konnte nicht gelesen werden.")
            continue

        source_height, source_width = image.shape[:2]
        expected_source_size = source_size or size
        if warn_on_size_mismatch and (source_width, source_height) != expected_source_size:
            print(
                "Warnung: "
                f"{image_path.name} ist {source_width}x{source_height}, "
                f"ROI ist {expected_source_size[0]}x{expected_source_size[1]}. "
                "Besser neu aus der ROI speichern."
            )

        descriptor = extract_hand_descriptor(image, size)
        templates.append(
            Template(
                label=label_from_filename(image_path),
                image=preprocess_for_matching(image, size),
                hand_descriptor=descriptor,
                hand_geometry_feature=hand_geometry_feature(descriptor),
                display_image=display_image,
                source_size=(source_width, source_height),
            )
        )

    if not templates:
        raise FileNotFoundError("Keine lesbaren Vergleichsbilder gefunden.")

    return templates


def scaled_roi(frame: np.ndarray, roi: dict) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame.shape[:2]
    reference_width = int(roi.get("reference_width", frame_width))
    reference_height = int(roi.get("reference_height", frame_height))

    scale_x = frame_width / max(1, reference_width)
    scale_y = frame_height / max(1, reference_height)

    x = int(round(int(roi["x"]) * scale_x))
    y = int(round(int(roi["y"]) * scale_y))
    width = max(1, int(round(int(roi["width"]) * scale_x)))
    height = max(1, int(round(int(roi["height"]) * scale_y)))

    x = min(max(0, x), frame_width - 1)
    y = min(max(0, y), frame_height - 1)
    width = min(width, frame_width - x)
    height = min(height, frame_height - y)
    return x, y, width, height


def crop_roi(frame: np.ndarray, roi: dict) -> np.ndarray:
    x, y, width, height = scaled_roi(frame, roi)
    cropped = frame[y : y + height, x : x + width]
    if cropped.size == 0:
        raise RuntimeError("Die skalierte ROI liegt ausserhalb des OBS-Projektorfensters.")
    return cropped


def weighted_correlation(first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> float:
    first_float = first.astype(np.float32)
    second_float = second.astype(np.float32)

    weight_sum = float(np.sum(weights))
    first_mean = float(np.sum(weights * first_float) / weight_sum)
    second_mean = float(np.sum(weights * second_float) / weight_sum)

    first_centered = first_float - first_mean
    second_centered = second_float - second_mean
    numerator = float(np.sum(weights * first_centered * second_centered))
    denominator = float(
        np.sqrt(np.sum(weights * first_centered * first_centered) * np.sum(weights * second_centered * second_centered))
    )

    if denominator <= 1e-6:
        return -1.0

    return numerator / denominator


def labels_are_neighbors(first: str, second: str, stage_count: int | None) -> bool:
    try:
        first_number = int(first)
        second_number = int(second)
    except ValueError:
        return False

    if abs(first_number - second_number) == 1:
        return True

    if stage_count is None:
        return False

    return {first_number, second_number} == {0, stage_count - 1}


def later_stage(first: str, second: str, stage_count: int | None) -> str:
    first_number = int(first)
    second_number = int(second)

    if stage_count is not None and {first_number, second_number} == {0, stage_count - 1}:
        return "0"

    return str(max(first_number, second_number))


def apply_later_stage_rule(
    scores: list[tuple[str, float]],
    prefer_later_stage_when_close: bool,
    later_stage_close_margin: float,
    stage_count: int | None,
) -> tuple[str, float]:
    best_label, best_score = scores[0]

    if not prefer_later_stage_when_close or len(scores) < 2:
        return best_label, best_score

    second_label, second_score = scores[1]
    if not labels_are_neighbors(best_label, second_label, stage_count):
        return best_label, best_score

    if best_score - second_score > later_stage_close_margin:
        return best_label, best_score

    try:
        preferred_label = later_stage(best_label, second_label, stage_count)
    except ValueError:
        return best_label, best_score

    preferred_score = next(score for label, score in scores if label == preferred_label)
    return preferred_label, preferred_score


def refine_close_pair(
    roi_hand_descriptor: np.ndarray,
    templates: list[Template],
    scores: list[tuple[str, float]],
    close_margin: float,
    hand_weight: float,
    allowed_pairs: set[frozenset[str]],
) -> list[tuple[str, float]]:
    if len(scores) < 2:
        return scores

    template_by_label = {template.label: template for template in templates}
    first_label, first_score = scores[0]
    candidate = next(
        (
            (label, score)
            for label, score in scores[1:5]
            if frozenset((first_label, label)) in allowed_pairs
            and first_score - score <= close_margin
        ),
        None,
    )
    if candidate is None:
        return scores

    second_label, second_score = candidate
    first_descriptor = template_by_label[first_label].hand_descriptor
    second_descriptor = template_by_label[second_label].hand_descriptor

    def dominant_angle(descriptor: np.ndarray) -> float:
        profile = cv2.GaussianBlur(
            descriptor.mean(axis=1).reshape(-1, 1),
            (1, 7),
            0,
        ).ravel()
        peak = int(np.argmax(profile))
        indices = np.array([(peak + offset) % len(profile) for offset in range(-3, 4)])
        values = profile[indices]
        angles = indices * (2.0 * np.pi / len(profile))
        x = float(np.sum(np.cos(angles) * values))
        y = float(np.sum(np.sin(angles) * values))
        angle = np.arctan2(y, x)
        return float(angle if angle >= 0 else angle + 2.0 * np.pi)

    def angle_similarity(first_angle: float, second_angle: float) -> float:
        difference = abs(first_angle - second_angle)
        circular_difference = min(difference, (2.0 * np.pi) - difference)
        return 1.0 - min(1.0, circular_difference / (np.pi / 2.0))

    roi_angle = dominant_angle(roi_hand_descriptor)
    first_angle_similarity = angle_similarity(roi_angle, dominant_angle(first_descriptor))
    second_angle_similarity = angle_similarity(roi_angle, dominant_angle(second_descriptor))
    base_weight = 1.0 - hand_weight
    refined = {
        first_label: (base_weight * first_score) + (hand_weight * first_angle_similarity),
        second_label: (base_weight * second_score) + (hand_weight * second_angle_similarity),
    }

    updated = [(label, refined.get(label, score)) for label, score in scores]
    updated.sort(key=lambda item: item[1], reverse=True)
    return updated


def compare_to_templates(
    roi_frame: np.ndarray,
    templates: list[Template],
    weight_map: np.ndarray,
    prefer_later_stage_when_close: bool,
    later_stage_close_margin: float,
    stage_count: int | None,
    pairwise_refinement_enabled: bool = True,
    pairwise_close_margin: float = 0.2,
    pairwise_hand_weight: float = 0.75,
    pairwise_refinement_pairs: tuple[tuple[str, str], ...] = (
        ("1", "7"),
        ("2", "7"),
        ("2", "3"),
    ),
    classification_method: str = "hand_geometry",
    geometry_min_score: float = 0.35,
    geometry_min_margin: float = 0.02,
    geometry_alignment_radius: int = 2,
) -> tuple[str, float, float, list[tuple[str, float]]]:
    size = templates[0].image.shape[::-1]
    prepared_roi = preprocess_for_matching(roi_frame, size)
    roi_hand_descriptor = extract_hand_descriptor(roi_frame, size)
    best_scores_by_label: dict[str, float] = {}

    for template in templates:
        correlation = weighted_correlation(prepared_roi, template.image, weight_map)
        diff = prepared_roi.astype(np.float32) - template.image.astype(np.float32)
        weighted_mse = float(np.sum(weight_map * diff * diff) / np.sum(weight_map))
        mse_similarity = 1.0 - min(1.0, weighted_mse / (255.0 * 255.0))
        score = (0.80 * correlation) + (0.20 * mse_similarity)
        if score > best_scores_by_label.get(template.label, -1.0):
            best_scores_by_label[template.label] = score

    scores = list(best_scores_by_label.items())
    scores.sort(key=lambda item: item[1], reverse=True)
    original_scores = dict(scores)
    if classification_method == "hand_geometry":
        geometry_scores = geometry_scores_with_alignment(
            roi_frame,
            size,
            templates,
            geometry_alignment_radius,
        )
        geometry_margin = (
            geometry_scores[0][1] - geometry_scores[1][1]
            if len(geometry_scores) > 1
            else geometry_scores[0][1]
        )
        if geometry_scores[0][1] >= geometry_min_score and geometry_margin >= geometry_min_margin:
            scores = geometry_scores
    elif pairwise_refinement_enabled:
        allowed_pairs = {
            frozenset((first, second))
            for first, second in pairwise_refinement_pairs
        }
        scores = refine_close_pair(
            roi_hand_descriptor,
            templates,
            scores,
            pairwise_close_margin,
            pairwise_hand_weight,
            allowed_pairs,
        )
    best_label, best_score = apply_later_stage_rule(
        scores,
        prefer_later_stage_when_close,
        later_stage_close_margin,
        stage_count,
    )
    confidence_score = original_scores[best_label]
    second_score = max(
        (score for label, score in original_scores.items() if label != best_label),
        default=0.0,
    )
    ranked_original_scores = [(label, original_scores[label]) for label, _ in scores]
    return best_label, confidence_score, confidence_score - second_score, ranked_original_scores[:5]


def draw_preview(frame: np.ndarray, roi: dict, label: str, score: float, last_label: str | None) -> None:
    x, y, width, height = scaled_roi(frame, roi)

    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 255), 2)
    cv2.putText(
        frame,
        f"match: {label or '-'} ({score:.2f}) last: {last_label or '-'}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )


def make_results_board(found_numbers: list[str], template_by_label: dict[str, Template]) -> np.ndarray:
    columns = 4
    tile_width = 180
    tile_height = 160
    image_height = 105
    rows = int(np.ceil(len(found_numbers) / columns))
    board = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)

    for index, number in enumerate(found_numbers):
        row = index // columns
        column = index % columns
        x = column * tile_width
        y = row * tile_height

        cv2.rectangle(board, (x + 8, y + 8), (x + tile_width - 8, y + tile_height - 8), (40, 40, 40), 1)
        cv2.putText(
            board,
            f"number {index + 1}: {number}",
            (x + 18, y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        template = template_by_label.get(number)
        if template is None:
            continue

        image = template.display_image
        height, width = image.shape[:2]
        scale = min((tile_width - 40) / width, image_height / height)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        image_x = x + (tile_width - resized_width) // 2
        image_y = y + 45 + (image_height - resized_height) // 2
        board[image_y : image_y + resized_height, image_x : image_x + resized_width] = resized

    return board


def add_found_number(found_numbers: list[str], number: str, template_by_label: dict[str, Template]) -> bool:
    print(number)
    found_numbers.append(number)

    if len(found_numbers) < 8:
        return True

    print("\nGefundene Zahlen:")
    for index, found_number in enumerate(found_numbers, start=1):
        print(f"number {index}: {found_number}")

    results_board = make_results_board(found_numbers, template_by_label)
    cv2.imshow("Gefundene Templates", results_board)
    cv2.waitKey(1)

    answer = input("\nNeue 8 Zahlen suchen? (j/n): ").strip().lower()
    cv2.destroyWindow("Gefundene Templates")
    if answer not in {"j", "ja", "y", "yes"}:
        return False

    found_numbers.clear()
    print("Suche neue 8 Zahlen...")
    return True


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    config = load_config(base_dir / "config.json")

    roi = config["roi"]
    roi_size = (int(roi["width"]), int(roi["height"]))
    processing_scale = max(1.0, float(config.get("processing_scale", 1.0)))
    template_size = (
        max(1, int(round(roi_size[0] * processing_scale))),
        max(1, int(round(roi_size[1] * processing_scale))),
    )
    templates = load_templates(
        base_dir / config["templates_dir"],
        template_size,
        source_size=roi_size,
    )
    template_by_label = {template.label: template for template in templates}
    weight_strength = float(config.get("discriminative_pixel_weight_strength", 1.6))
    weight_map = make_discriminative_weight_map(templates, weight_strength)
    prefer_later_stage_when_close = bool(config.get("prefer_later_stage_when_close", True))
    later_stage_close_margin = float(config.get("later_stage_close_margin", 0.04))
    configured_stage_count = config.get("stage_count", None)
    stage_count = int(configured_stage_count) if configured_stage_count is not None else None
    capture = WindowCapture(str(config["window_title_contains"]))

    threshold = float(config["match_threshold"])
    first_on_appear_start_threshold = float(config.get("first_on_appear_start_threshold", threshold))
    margin_threshold = float(config.get("match_margin_threshold", 0.0))
    stable_frames_required = int(config.get("stable_frames_required", 2))
    first_on_appear_stable_frames_required = int(
        config.get("first_on_appear_stable_frames_required", stable_frames_required)
    )
    stable_window_size = int(config.get("stable_window_size", 6))
    lost_frames_required = int(config["lost_frames_required"])
    show_preview = bool(config["show_preview"])
    debug_top_matches = bool(config.get("debug_top_matches", False))
    max_fps = float(config.get("max_fps", 20))
    min_frame_seconds = 1.0 / max_fps if max_fps > 0 else 0.0

    last_label: str | None = None
    stable_labels: deque[str] = deque(maxlen=stable_window_size)
    clock_was_visible = False
    number_saved_for_current_clock = False
    lost_frames = 0
    found_numbers: list[str] = []

    print("Detector laeuft. Druecke q im Vorschaufenster zum Beenden.")

    while True:
        frame_started_at = time.monotonic()
        ok, frame = capture.read()
        if not ok or frame is None:
            print("Kein Frame empfangen.")
            break

        roi_frame = crop_roi(frame, roi)
        label, score, margin, top_scores = compare_to_templates(
            roi_frame,
            templates,
            weight_map,
            prefer_later_stage_when_close,
            later_stage_close_margin,
            stage_count,
        )
        margin_ok = margin >= margin_threshold if margin_threshold > 0 else True
        is_waiting_for_first_appearance = not clock_was_visible
        active_threshold = first_on_appear_start_threshold if is_waiting_for_first_appearance else threshold
        active_stable_frames_required = (
            first_on_appear_stable_frames_required if is_waiting_for_first_appearance else stable_frames_required
        )
        clock_visible = score >= active_threshold and margin_ok

        if debug_top_matches:
            top_text = ", ".join(f"{top_label}:{top_score:.3f}" for top_label, top_score in top_scores)
            print(
                f"top matches: {top_text} | margin:{margin:.3f} "
                f"| threshold:{active_threshold:.3f} | visible:{clock_visible}"
            )

        if clock_visible:
            stable_labels.append(label)
            stable_label, stable_count = Counter(stable_labels).most_common(1)[0]

            if stable_count >= active_stable_frames_required and last_label != stable_label:
                last_label = stable_label

            if last_label is not None and not number_saved_for_current_clock:
                if not add_found_number(found_numbers, last_label, template_by_label):
                    break
                number_saved_for_current_clock = True

            clock_was_visible = True
            lost_frames = 0
        elif clock_was_visible:
            lost_frames += 1
            if lost_frames >= lost_frames_required:
                clock_was_visible = False
                number_saved_for_current_clock = False
                last_label = None
                lost_frames = 0
                stable_labels.clear()

        if show_preview:
            draw_preview(frame, roi, label if clock_visible else "", score, last_label)
            cv2.imshow("Gen 7 Clock RNG", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        elapsed = time.monotonic() - frame_started_at
        if elapsed < min_frame_seconds:
            time.sleep(min_frame_seconds - elapsed)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
