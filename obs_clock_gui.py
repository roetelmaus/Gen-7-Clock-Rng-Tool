from __future__ import annotations

import json
import os
import queue
import shutil
import sys
import threading
import time
import tkinter as tk
from collections import Counter, deque
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from obs_clock_detector import (
    load_templates,
    make_discriminative_weight_map,
    scaled_roi,
)
from window_capture import WindowCapture, list_visible_windows
from quality_matcher import QualityMatcher
from screen_tracker import ScreenTracker


class ClockDetectorGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gen 7 Clock RNG V2.1")
        self.root.geometry("1120x760")
        self.root.minsize(900, 640)
        self.screen_lock = threading.Lock()
        self.screen_tracker = None
        self.selecting_screen = False
        self.screen_points = []
        self.screen_selection_frame = None
        self.screen_tracking_lost = False
        self.screen_tracking_loss_since = None
        self.tracking_warning_visible = False

        self.resource_dir = (
            Path(sys._MEIPASS)
            if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parent
        )
        if getattr(sys, "frozen", False):
            local_app_data = Path(
                os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
            )
            self.data_dir = local_app_data / "Gen7ClockRNG_V2"
            self.data_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.data_dir = self.resource_dir

        self.config_path = self.data_dir / "config.json"
        bundled_config = self.resource_dir / "config.json"
        if not self.config_path.exists() and bundled_config != self.config_path:
            shutil.copy2(bundled_config, self.config_path)
        self.config = self.load_config()
        self.roi = dict(self.config["roi"])
        self.roi_lock = threading.Lock()
        self.source_lock = threading.Lock()
        self.selected_window_hwnd: int | None = None
        self.selected_window_title = str(self.config.get("window_title_contains", ""))
        self.window_choices: dict[str, tuple[int, str]] = {}
        self.alignment_lock = threading.Lock()
        self.alignment_matrix, self.alignment_source_size = self.load_saved_alignment()
        self.alignment_applied_event = threading.Event()
        self.alignment_request_lock = threading.Lock()
        self.alignment_generation = 0
        self.calibration_restart_notice_shown = False
        self.pending_alignment_box: tuple[int, int, int, int] | None = None
        self.alignment_frame_queue: queue.Queue = queue.Queue(maxsize=12)
        self.alignment_wait_stop_event = threading.Event()
        self.alignment_wait_worker: threading.Thread | None = None

        templates_dir = self.resource_dir / self.config["templates_dir"]
        template_source_size = self.read_template_source_size(templates_dir)
        self.processing_scale = max(1.0, float(self.config.get("processing_scale", 1.0)))
        template_size = (
            max(1, int(round(template_source_size[0] * self.processing_scale))),
            max(1, int(round(template_source_size[1] * self.processing_scale))),
        )
        self.templates = load_templates(
            templates_dir,
            template_size,
            source_size=template_source_size,
            warn_on_size_mismatch=False,
        )
        self.template_by_label = {}
        for template in self.templates:
            self.template_by_label.setdefault(template.label, template)
        self.weight_map = make_discriminative_weight_map(
            self.templates,
            float(self.config.get("discriminative_pixel_weight_strength", 1.6)),
        )
        self.alignment_refinement_templates = [
            template.display_image
            for template in self.templates
        ]
        self.matcher = QualityMatcher(self.templates, self.weight_map, int(self.config.get("geometry_alignment_radius", 2)), bool(self.config.get("quality_variants", True)))

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.event_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.detect_event = threading.Event()
        self.reset_event = threading.Event()
        self.search_stopped_event = threading.Event()
        self.capture_enabled_event = threading.Event()
        self.capture_enabled_event.set()
        self.roi_preview_event = threading.Event()
        self.worker: threading.Thread | None = None

        self.current_frame = None
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.photo = None
        self.template_photos: list[ImageTk.PhotoImage] = []
        self.last_template_photo: ImageTk.PhotoImage | None = None
        self.selecting_alignment = False
        self.alignment_selection_start: tuple[int, int] | None = None
        self.alignment_selection_end: tuple[int, int] | None = None

        self.last_number_var = tk.StringVar(value="-")
        self.last_score_var = tk.StringVar(value="Score: -")
        self.window_choice_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready. Set the clock box to begin.")
        self.numbers_text_var = tk.StringVar(value="")

        self.build_ui()
        self.start_worker()
        self.root.after(30, self.poll_queues)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def load_config(self) -> dict:
        with self.config_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def read_template_source_size(self, templates_dir: Path) -> tuple[int, int]:
        image_paths = [
            *templates_dir.glob("*.png"),
            *templates_dir.glob("*.jpg"),
            *templates_dir.glob("*.jpeg"),
            *templates_dir.glob("*.bmp"),
        ]
        canonical_paths = [path for path in image_paths if path.stem.isdigit()]
        for image_path in sorted(canonical_paths or image_paths):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                height, width = image.shape[:2]
                return width, height
        raise FileNotFoundError(f"No readable templates found in {templates_dir}.")

    def load_saved_alignment(self):
        saved = self.config.get("obs_content_alignment")
        if not isinstance(saved, dict):
            return None, None

        matrix = np.asarray(saved.get("matrix"), dtype=np.float32)
        source_width = int(saved.get("source_width", 0))
        source_height = int(saved.get("source_height", 0))
        if matrix.shape != (2, 3) or source_width <= 0 or source_height <= 0:
            return None, None
        return matrix, (source_width, source_height)

    def save_config(self) -> None:
        self.config["roi"] = dict(self.roi)
        with self.config_path.open("w", encoding="utf-8") as file:
            json.dump(self.config, file, indent=2)
            file.write("\n")

    def output_label(self, detected_label: str) -> str:
        try:
            number = int(detected_label)
        except ValueError:
            return detected_label

        stage_count = int(self.config.get("stage_count", 17))
        offset = int(self.config.get("first_on_appear_output_offset", 4))
        return str((number + offset) % stage_count)

    def build_ui(self) -> None:
        self.root.configure(bg="#151719")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#151719")
        style.configure("Panel.TFrame", background="#202326")
        style.configure("TLabel", background="#151719", foreground="#f2f2f2", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#151719", foreground="#ffffff", font=("Segoe UI", 18, "bold"))
        style.configure("Value.TLabel", background="#202326", foreground="#ffd84d", font=("Segoe UI", 34, "bold"))
        style.configure("Number.TLabel", background="#202326", foreground="#ffffff", font=("Segoe UI", 16, "bold"))
        style.configure("Status.TLabel", background="#293a40", foreground="#ffffff", font=("Segoe UI", 12, "bold"), padding=(14, 12))
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=(14, 9))
        style.configure("TCombobox", padding=6)

        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=18, pady=(14, 10))
        ttk.Label(header, text="Gen 7 Clock RNG V2.1", style="Title.TLabel").pack(side="left")

        controls = ttk.Frame(header)
        controls.pack(side="right")
        ttk.Button(
            controls,
            text="Set Clock Box",
            command=self.begin_alignment_selection,
        ).pack(side="left", padx=4)
        ttk.Button(
            controls,
            text="Stop Searching",
            command=self.stop_searching,
        ).pack(side="left", padx=4)
        ttk.Button(controls, text="Restart", command=self.restart_detection).pack(side="left", padx=4)

        source_selector = ttk.Frame(header)
        source_selector.pack(side="left", fill="x", expand=True, padx=18)
        self.window_dropdown = ttk.Combobox(
            source_selector,
            textvariable=self.window_choice_var,
            state="readonly",
            width=46,
            postcommand=self.refresh_window_choices,
        )
        self.window_dropdown.pack()
        self.window_dropdown.bind("<<ComboboxSelected>>", self.select_capture_window)
        self.refresh_window_choices()

        body = ttk.Frame(self.root)
        webcam_controls = ttk.Frame(self.root)
        webcam_controls.pack(fill="x", padx=18, pady=(0, 8))
        self.webcam_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(webcam_controls, text="Webcam Mode", variable=self.webcam_mode_var,
                        command=self.change_webcam_mode).pack(side="left")
        ttk.Button(webcam_controls, text="Set Screen Corners", command=self.begin_screen_selection).pack(side="left", padx=8)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 12))

        preview_panel = ttk.Frame(body, style="Panel.TFrame")
        preview_panel.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(preview_panel, bg="#08090a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.start_alignment_drag)
        self.canvas.bind("<B1-Motion>", self.drag_alignment_box)
        self.canvas.bind("<ButtonRelease-1>", self.end_alignment_drag)
        self.canvas.bind("<Configure>", self.resize_preview_warning)

        results = ttk.Frame(body, style="Panel.TFrame", width=360)
        results.pack(side="right", fill="y", padx=(12, 0))
        results.pack_propagate(False)

        ttk.Label(results, text="Last Number", background="#202326", foreground="#d4d9dd", font=("Segoe UI", 11)).pack(
            pady=(20, 2)
        )
        last_value_row = ttk.Frame(results, style="Panel.TFrame")
        last_value_row.pack(pady=(0, 16))
        ttk.Label(last_value_row, textvariable=self.last_number_var, style="Value.TLabel").pack(side="left")
        self.last_template_label = tk.Label(
            last_value_row,
            bg="#202326",
            bd=0,
        )
        self.last_template_label.pack(side="left", padx=(12, 0))
        self.last_score_label = ttk.Label(
            last_value_row,
            textvariable=self.last_score_var,
            background="#202326",
            foreground="#8d969e",
            font=("Consolas", 11),
        )
        if bool(self.config.get("show_score_debug", False)):
            self.last_score_label.pack(side="left", padx=(12, 0))

        separator = ttk.Separator(results)
        separator.pack(fill="x", padx=18, pady=(0, 16))
        ttk.Label(results, text="Detected Numbers (max. 8)", background="#202326", foreground="#d4d9dd").pack(
            pady=(0, 10)
        )

        self.numbers_entry = tk.Entry(
            results,
            textvariable=self.numbers_text_var,
            state="readonly",
            readonlybackground="#151719",
            fg="#ffffff",
            insertbackground="#ffffff",
            selectbackground="#3d77a8",
            selectforeground="#ffffff",
            relief="flat",
            font=("Consolas", 13),
            justify="left",
        )
        self.numbers_entry.pack(fill="x", padx=18, ipady=10)

        ttk.Label(
            results,
            text="Select the line or press Ctrl+C to copy.",
            background="#202326",
            foreground="#8d969e",
            justify="center",
        ).pack(pady=(8, 0))

        ttk.Label(
            results,
            text="Matching Templates",
            background="#202326",
            foreground="#d4d9dd",
        ).pack(pady=(20, 8))

        template_strip = ttk.Frame(results, style="Panel.TFrame")
        template_strip.pack(fill="x", padx=18)
        self.templates_canvas = tk.Canvas(
            template_strip,
            height=46,
            bg="#151719",
            highlightthickness=0,
        )
        self.templates_scrollbar = ttk.Scrollbar(
            template_strip,
            orient="horizontal",
            command=self.templates_canvas.xview,
        )
        self.templates_canvas.configure(xscrollcommand=self.templates_scrollbar.set)
        self.templates_canvas.pack(fill="x")
        self.templates_scrollbar.pack(fill="x", pady=(4, 0))
        self.templates_row = tk.Frame(self.templates_canvas, bg="#151719")
        self.templates_window = self.templates_canvas.create_window(
            (0, 0),
            window=self.templates_row,
            anchor="nw",
        )
        self.templates_row.bind("<Configure>", self.update_template_scrollregion)
        self.templates_canvas.bind("<Configure>", self.resize_template_window)

        ttk.Label(
            results,
            text="Set Clock Box: draw a rough box around the clock.",
            background="#202326",
            foreground="#8d969e",
            justify="center",
        ).pack(side="bottom", pady=20)

        self.status_label = ttk.Label(self.root, textvariable=self.status_var,
                                      style="Status.TLabel", anchor="w", justify="left", wraplength=850)
        self.status_label.pack(fill="x", padx=18, pady=(0, 12))
        self.status_label.bind("<Configure>", lambda event: self.status_label.configure(wraplength=max(100, event.width - 32)))

    def start_worker(self) -> None:
        self.worker = threading.Thread(target=self.capture_loop, daemon=True)
        self.worker.start()

    def refresh_window_choices(self) -> None:
        windows = list_visible_windows()
        title_counts: dict[str, int] = {}
        for window in windows:
            title_counts[window.title] = title_counts.get(window.title, 0) + 1

        choices: dict[str, tuple[int, str]] = {}
        for window in windows:
            display = window.title
            if title_counts[window.title] > 1:
                display = f"{window.title} [{window.hwnd}]"
            choices[display] = (window.hwnd, window.title)

        current_hwnd = self.selected_window_hwnd
        self.window_choices = choices
        self.window_dropdown["values"] = list(choices)

        selected_display = next(
            (
                display
                for display, (hwnd, _title) in choices.items()
                if hwnd == current_hwnd
            ),
            None,
        )
        if selected_display is None and current_hwnd is None and self.selected_window_title:
            selected_display = next(
                (
                    display
                    for display, (_hwnd, title) in choices.items()
                    if self.selected_window_title.lower() in title.lower()
                ),
                None,
            )
        if selected_display is None and current_hwnd is None and choices:
            selected_display = next(iter(choices))

        if selected_display is not None:
            self.window_choice_var.set(selected_display)
            hwnd, title = choices[selected_display]
            with self.source_lock:
                self.selected_window_hwnd = hwnd
                self.selected_window_title = title

    def select_capture_window(self, _event=None) -> None:
        selected = self.window_choices.get(self.window_choice_var.get())
        if selected is None:
            return

        hwnd, title = selected
        with self.source_lock:
            source_changed = hwnd != self.selected_window_hwnd
            self.selected_window_hwnd = hwnd
            self.selected_window_title = title
        if not source_changed:
            return

        self.calibration_restart_notice_shown = False
        self.tracking_warning_visible = False

        with self.screen_lock:
            self.screen_tracker = None
        self.selecting_screen = False
        self.screen_selection_frame = None
        self.screen_tracking_lost = False
        self.screen_tracking_loss_since = None

        self.cancel_alignment()

        self.config["window_title_contains"] = title
        self.config.pop("obs_content_alignment", None)
        self.save_config()
        with self.alignment_lock:
            self.alignment_matrix = None
            self.alignment_source_size = None
        self.roi_preview_event.clear()
        self.detect_event.clear()
        self.reset_event.set()
        self.status_var.set("Window selected. Set Clock Box to begin detection.")

    def change_webcam_mode(self) -> None:
        self.calibration_restart_notice_shown = False
        self.tracking_warning_visible = False
        self.detect_event.clear()
        self.cancel_alignment()
        with self.screen_lock:
            self.screen_tracker = None
        self.selecting_screen = False
        self.screen_selection_frame = None
        self.screen_tracking_lost = False
        self.screen_tracking_loss_since = None
        with self.alignment_lock:
            self.alignment_matrix = None
            self.alignment_source_size = None
        self.config.pop("obs_content_alignment", None)
        self.save_config()
        self.roi_preview_event.clear()
        self.status_var.set("Set the four screen corners." if self.webcam_mode_var.get() else "Set Clock Box to begin detection.")

    def begin_screen_selection(self) -> None:
        if not self.webcam_mode_var.get():
            self.status_var.set("Enable Webcam Mode first.")
            return
        if self.search_stopped_event.is_set():
            self.status_var.set("Press Restart first.")
            return
        self.change_webcam_mode()
        self.screen_points = []
        self.screen_selection_frame = None
        self.selecting_screen = True
        self.canvas.configure(cursor="crosshair")
        self.status_var.set("Click the SCREEN corners: top-left, top-right, bottom-right, bottom-left.")

    def add_screen_corner(self, event) -> None:
        frame = self.screen_selection_frame
        if frame is None or self.current_frame is not frame:
            return
        x = (event.x - self.display_offset_x) / self.display_scale
        y = (event.y - self.display_offset_y) / self.display_scale
        if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
            return
        self.screen_points.append((x, y))
        if len(self.screen_points) < 4:
            self.draw_frame(frame)
            return
        try:
            tracker = ScreenTracker(frame, self.screen_points)
        except ValueError as error:
            self.screen_points = []
            self.status_var.set(str(error))
            return
        with self.screen_lock:
            self.screen_tracker = tracker
        self.selecting_screen = False
        self.screen_selection_frame = None
        self.canvas.configure(cursor="")
        self.status_var.set("Screen corners set. Set Clock Box, then restart the game after calibration.")

    def update_screen_tracking_status(self, lost, now, transitioning=False) -> None:
        if lost:
            if self.screen_tracking_loss_since is None:
                self.screen_tracking_loss_since = now
            # Menu fades briefly hide the reference patches; recognition still waits.
            warning_delay = 10.0 if transitioning else 2.0
            if now - self.screen_tracking_loss_since >= warning_delay and not self.screen_tracking_lost:
                self.screen_tracking_lost = True
                self.event_queue.put(("screen_tracking", True))
        else:
            self.screen_tracking_loss_since = None
            if self.screen_tracking_lost:
                self.screen_tracking_lost = False
                self.event_queue.put(("screen_tracking", False))

    def start_detection(self) -> None:
        if self.webcam_mode_var.get() and self.screen_tracker is None:
            self.status_var.set("Set the four screen corners first.")
            return
        if self.search_stopped_event.is_set():
            return
        self.save_config()
        self.detect_event.set()
        with self.alignment_lock:
            alignment_ready = self.alignment_matrix is not None
        if alignment_ready:
            self.status_var.set("Detection is running with OBS image alignment.")
        else:
            self.status_var.set("Detection is running. Set Clock Box if the OBS image differs.")

    def restart_detection(self) -> None:
        self.detect_event.clear()
        self.search_stopped_event.clear()
        self.capture_enabled_event.set()
        self.reset_event.set()
        self.last_number_var.set("-")
        self.last_score_var.set("Score: -")
        self.render_last_template(None)
        self.numbers_text_var.set("")
        self.render_template_row([])
        self.root.after(100, self.start_detection)

    def stop_searching(self) -> None:
        self.tracking_warning_visible = False
        self.selecting_screen = False
        self.screen_selection_frame = None
        self.detect_event.clear()
        self.search_stopped_event.set()
        self.capture_enabled_event.clear()
        self.cancel_alignment()
        self.current_frame = None
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.draw_black_preview()
        self.status_var.set("Searching stopped. Press Restart to continue.")

    def cancel_alignment(self) -> None:
        with self.alignment_request_lock:
            self.alignment_generation += 1
            self.pending_alignment_box = None
        self.alignment_wait_stop_event.set()
        self.selecting_alignment = False
        self.alignment_selection_start = None
        self.alignment_selection_end = None
        self.canvas.configure(cursor="")

    def begin_alignment_selection(self) -> None:
        if self.webcam_mode_var.get() and (self.screen_tracker is None or self.selecting_screen):
            self.status_var.set("Set the four screen corners first.")
            return
        if self.search_stopped_event.is_set():
            self.status_var.set("Searching is stopped. Press Restart first.")
            return
        if self.current_frame is None:
            self.status_var.set("Waiting for the OBS projector frame.")
            return

        self.detect_event.clear()
        self.cancel_alignment()
        self.roi_preview_event.clear()
        with self.alignment_lock:
            self.alignment_matrix = None
            self.alignment_source_size = None
        self.selecting_alignment = True
        self.alignment_selection_start = None
        self.alignment_selection_end = None
        self.canvas.configure(cursor="crosshair")
        self.status_var.set("Drag a rough box around the visible clock.")

    def finish_alignment_selection(self) -> None:
        start = self.alignment_selection_start
        end = self.alignment_selection_end
        self.selecting_alignment = False
        self.alignment_selection_start = None
        self.alignment_selection_end = None
        self.canvas.configure(cursor="")
        if self.current_frame is None or start is None or end is None:
            return

        left, right = sorted((start[0], end[0]))
        top, bottom = sorted((start[1], end[1]))
        if right - left < 8 or bottom - top < 8:
            self.status_var.set("Alignment selection was too small.")
            return

        frame_height, frame_width = self.current_frame.shape[:2]
        x = int(round((left - self.display_offset_x) / self.display_scale))
        y = int(round((top - self.display_offset_y) / self.display_scale))
        width = int(round((right - left) / self.display_scale))
        height = int(round((bottom - top) / self.display_scale))
        x = min(max(0, x), frame_width - 1)
        y = min(max(0, y), frame_height - 1)
        width = min(max(8, width), frame_width - x)
        height = min(max(8, height), frame_height - y)
        coarse_box = (x, y, width, height)
        with self.alignment_request_lock:
            self.pending_alignment_box = coarse_box
        while not self.alignment_frame_queue.empty():
            try:
                self.alignment_frame_queue.get_nowait()
            except queue.Empty:
                break

        self.alignment_wait_stop_event.clear()
        self.alignment_wait_worker = threading.Thread(
            target=self.wait_for_alignment_clock,
            args=(self.alignment_generation,),
            daemon=True,
        )
        self.alignment_wait_worker.start()
        self.status_var.set("Waiting for the clock. Go back with B, then show it again with A.")

    def wait_for_alignment_clock(self, generation) -> None:
        clock_triggered = False
        match_attempts = 0
        missing_frames = 0
        max_match_attempts = int(
            self.config.get("alignment_max_match_attempts", 12)
        )
        missing_frames_before_failure = int(
            self.config.get("lost_frames_required", 5)
        )

        while not self.stop_event.is_set() and not self.alignment_wait_stop_event.is_set():
            if generation != self.alignment_generation:
                return
            try:
                frame = self.alignment_frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            with self.alignment_request_lock:
                coarse_box = self.pending_alignment_box
            if coarse_box is None:
                return

            try:
                candidate_box, candidate_score = self.find_clock_in_selected_area(
                    frame,
                    coarse_box,
                )
                trigger_score = float(
                    self.config.get("manual_alignment_clock_trigger_score", 0.70)
                )
                if candidate_box is None or candidate_score < trigger_score:
                    if clock_triggered:
                        missing_frames += 1
                        if missing_frames >= missing_frames_before_failure:
                            with self.alignment_request_lock:
                                self.pending_alignment_box = None
                            self.event_queue.put(("alignment_failed",))
                            return
                    continue

                clock_triggered = True
                missing_frames = 0
                match_attempts += 1
                result = self.build_alignment_from_box(frame, candidate_box)
                if generation != self.alignment_generation or self.search_stopped_event.is_set():
                    return
                if result is None:
                    if match_attempts >= max_match_attempts:
                        with self.alignment_request_lock:
                            self.pending_alignment_box = None
                        self.event_queue.put(("alignment_failed",))
                        return
                    continue

                matrix, score, detected_box = result
                with self.alignment_lock:
                    self.alignment_matrix = matrix
                    self.alignment_source_size = (frame.shape[1], frame.shape[0])
                with self.alignment_request_lock:
                    self.pending_alignment_box = None
                self.alignment_applied_event.set()
                self.roi_preview_event.set()
                self.detect_event.set()
                self.event_queue.put(
                    (
                        "alignment_found",
                        matrix.tolist(),
                        score,
                        detected_box,
                        frame.shape[1],
                        frame.shape[0],
                        generation,
                    )
                )
                return
            except Exception as error:
                self.event_queue.put(("error", str(error)))
                return

    def find_clock_in_selected_area(
        self,
        frame,
        coarse_box: tuple[int, int, int, int],
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        frame_height, frame_width = frame.shape[:2]
        coarse_x, coarse_y, coarse_width, coarse_height = coarse_box
        padding_x = max(8, int(round(coarse_width * 0.20)))
        padding_y = max(8, int(round(coarse_height * 0.20)))
        left = max(0, coarse_x - padding_x)
        top = max(0, coarse_y - padding_y)
        right = min(frame_width, coarse_x + coarse_width + padding_x)
        bottom = min(frame_height, coarse_y + coarse_height + padding_y)
        search_gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)

        minimum_width = max(8, int(round(coarse_width * 0.50)))
        maximum_width = max(minimum_width, int(round(coarse_width * 1.50)))
        width_step = max(1, int(round((maximum_width - minimum_width) / 16)))
        best_score = -1.0
        best_box = None
        quick_templates = [
            self.template_by_label[label].display_image
            for label in ("0", "5", "10", "15")
            if label in self.template_by_label
        ]

        for template_image in quick_templates:
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            template_height, template_width = template_gray.shape[:2]
            aspect = template_height / max(1, template_width)
            for width in range(minimum_width, maximum_width + 1, width_step):
                height = max(8, int(round(width * aspect)))
                if width >= search_gray.shape[1] or height >= search_gray.shape[0]:
                    continue
                resized_template = cv2.resize(
                    template_gray,
                    (width, height),
                    interpolation=cv2.INTER_CUBIC if width > template_width else cv2.INTER_AREA,
                )
                score_map = cv2.matchTemplate(
                    search_gray,
                    resized_template,
                    cv2.TM_CCOEFF_NORMED,
                )
                _, score, _, location = cv2.minMaxLoc(score_map)
                if score > best_score:
                    best_score = float(score)
                    best_box = (
                        left + int(location[0]),
                        top + int(location[1]),
                        width,
                        height,
                    )

        return best_box, best_score

    def build_alignment_from_box(
        self,
        frame,
        coarse_box: tuple[int, int, int, int],
    ) -> tuple[object, float, tuple[int, int, int, int]] | None:
        best_box, best_score = self.refine_content_box(frame, coarse_box)
        minimum_match_score = float(
            self.config.get("obs_content_alignment_threshold", 0.93)
        )
        required_match_score = minimum_match_score
        if best_score < float(self.config.get("manual_alignment_clock_trigger_score", .7)):
            return None

        source_x, source_y, source_width, source_height = best_box
        with self.roi_lock:
            target_x = int(self.roi["x"])
            target_y = int(self.roi["y"])
            target_width = int(self.roi["width"])
            target_height = int(self.roi["height"])

        scale_x = target_width / max(1, source_width)
        scale_y = target_height / max(1, source_height)
        matrix = np.array(
            [
                [scale_x, 0.0, target_x - (source_x * scale_x)],
                [0.0, scale_y, target_y - (source_y * scale_y)],
            ],
            dtype=np.float32,
        )
        aligned_frame = self.apply_content_alignment(frame, matrix)
        aligned_roi = self.crop_current_roi(aligned_frame, self.roi)
        _, matcher_score, matcher_margin, _ = self.matcher.compare(
            aligned_roi,
        )
        if matcher_score < required_match_score or matcher_margin < float(self.config.get("match_margin_threshold", .02)):
            return None
        return matrix, matcher_score, best_box

    def capture_loop(self) -> None:
        with self.source_lock:
            source_hwnd = self.selected_window_hwnd
            source_title = self.selected_window_title
        capture = WindowCapture(source_title, source_hwnd)
        threshold = float(self.config["match_threshold"])
        start_threshold = float(self.config.get("first_on_appear_start_threshold", threshold))
        presence_threshold = float(
            self.config.get("invalid_clock_presence_threshold", min(threshold, 0.65))
        )
        margin_threshold = float(self.config.get("match_margin_threshold", 0.0))
        stable_frames_required = int(self.config.get("stable_frames_required", 2))
        first_stable_required = int(
            self.config.get("first_on_appear_stable_frames_required", stable_frames_required)
        )
        stable_window_size = int(self.config.get("stable_window_size", 6))
        lost_frames_required = int(self.config["lost_frames_required"])
        max_fps = float(self.config.get("max_fps", 20))
        frame_delay = 1.0 / max_fps if max_fps > 0 else 0.0
        preview_delay = 1.0 / max(1.0, float(self.config.get("preview_fps", 10)))
        last_preview_time = float('-inf')
        idle_after_seconds = max(
            120.0,
            float(self.config.get("idle_after_seconds", 120)),
        )
        idle_scan_fps = max(0.1, float(self.config.get("idle_scan_fps", 1)))
        idle_frame_delay = 1.0 / idle_scan_fps
        last_label: str | None = None
        last_label_score = 0.0
        stable_labels: deque[str] = deque(maxlen=stable_window_size)
        clock_was_visible = False
        number_saved = False
        lost_frames = 0
        possible_clock_seen = False
        possible_clock_lost_frames = 0
        waited_below_threshold = False
        found_numbers: list[str] = []
        found_template_labels: list[str] = []
        last_capture_error: str | None = None
        last_clock_activity = time.monotonic()
        idle_mode = False

        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                if self.search_stopped_event.is_set():
                    self.capture_enabled_event.wait()
                    continue
                with self.alignment_request_lock:
                    alignment_active = self.pending_alignment_box is not None
                target_frame_delay = (
                    frame_delay
                    if alignment_active
                    or (self.detect_event.is_set() and not idle_mode)
                    else idle_frame_delay
                )
                with self.source_lock:
                    selected_hwnd = self.selected_window_hwnd
                    selected_title = self.selected_window_title
                if selected_hwnd is not None and (
                    capture.hwnd != selected_hwnd
                    or capture.title_contains != selected_title
                ):
                    capture.set_window(selected_hwnd, selected_title)
                    last_capture_error = None
                try:
                    ok, frame = capture.read()
                    last_capture_error = None
                except Exception as error:
                    error_message = str(error)
                    if error_message != last_capture_error:
                        self.event_queue.put(("capture_error", error_message))
                        last_capture_error = error_message
                    time.sleep(0.2)
                    continue
                if not ok or frame is None:
                    raise RuntimeError("No frame received from the selected window.")

                if self.selecting_screen:
                    if self.screen_selection_frame is None:
                        self.screen_selection_frame = frame.copy()
                    selection_frame = self.screen_selection_frame
                    if selection_frame is not None:
                        self.put_latest_frame(selection_frame)
                    time.sleep(0.05)
                    continue
                with self.screen_lock:
                    tracker = self.screen_tracker
                    corrected = None if tracker is None else tracker.rectify(frame, started)
                if tracker is not None:
                    lost = corrected is None
                    self.update_screen_tracking_status(lost, started, tracker.black_screen)
                    if tracker.dark_frames >= 2:
                        # A fade separates clocks even if contour tracking is unavailable.
                        stable_labels.clear()
                        clock_was_visible = False
                        number_saved = False
                        lost_frames = 0
                        possible_clock_seen = False
                        possible_clock_lost_frames = 0
                        waited_below_threshold = False
                        last_label = None
                        last_label_score = 0.0
                    if lost or tracker.black_screen:
                        elapsed = time.monotonic() - started
                        if elapsed < target_frame_delay:
                            time.sleep(target_frame_delay - elapsed)
                        continue
                    frame = corrected

                with self.alignment_request_lock:
                    alignment_waiting = self.pending_alignment_box is not None
                if alignment_waiting:
                    try:
                        self.alignment_frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        try:
                            self.alignment_frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.alignment_frame_queue.put_nowait(frame.copy())

                with self.alignment_lock:
                    if (
                        self.alignment_matrix is not None
                        and self.alignment_source_size is not None
                        and self.alignment_source_size != (frame.shape[1], frame.shape[0])
                    ):
                        self.alignment_matrix = None
                        self.alignment_source_size = None
                        self.event_queue.put(("alignment_size_changed",))
                    alignment_matrix = (
                        None
                        if self.alignment_matrix is None
                        else self.alignment_matrix.copy()
                    )
                with self.roi_lock:
                    roi = dict(self.roi)
                aligned_roi = None
                if alignment_matrix is not None:
                    aligned_roi = self.apply_roi_alignment(frame, alignment_matrix, roi)
                if not self.search_stopped_event.is_set() and started - last_preview_time >= preview_delay:
                    if aligned_roi is not None and self.roi_preview_event.is_set():
                        display_frame = aligned_roi
                    elif alignment_matrix is not None:
                        display_frame = self.apply_content_alignment(frame, alignment_matrix)
                    else:
                        display_frame = frame
                    self.put_latest_frame(display_frame)
                    last_preview_time = started

                if self.reset_event.is_set():
                    last_label = None
                    last_label_score = 0.0
                    stable_labels.clear()
                    clock_was_visible = False
                    number_saved = False
                    lost_frames = 0
                    possible_clock_seen = False
                    possible_clock_lost_frames = 0
                    waited_below_threshold = False
                    found_numbers.clear()
                    found_template_labels.clear()
                    last_clock_activity = time.monotonic()
                    idle_mode = False
                    self.reset_event.clear()

                if self.alignment_applied_event.is_set():
                    stable_labels.clear()
                    clock_was_visible = True
                    number_saved = True
                    lost_frames = 0
                    last_label = None
                    last_label_score = 0.0
                    waited_below_threshold = False
                    last_clock_activity = time.monotonic()
                    idle_mode = False
                    self.alignment_applied_event.clear()

                if self.detect_event.is_set():
                    if alignment_waiting:
                        elapsed = time.monotonic() - started
                        if elapsed < frame_delay:
                            time.sleep(frame_delay - elapsed)
                        continue

                    now = time.monotonic()
                    if (
                        not idle_mode
                        and idle_after_seconds > 0
                        and now - last_clock_activity >= idle_after_seconds
                    ):
                        idle_mode = True
                        target_frame_delay = idle_frame_delay
                        self.detect_event.clear()
                        self.event_queue.put(("idle",))
                        continue

                    roi_frame = aligned_roi if aligned_roi is not None else self.crop_current_roi(frame, roi)
                    label, score, margin, top_scores = self.matcher.compare(
                        roi_frame,
                    )

                    margin_ok = margin >= margin_threshold if margin_threshold > 0 else True
                    waiting_first = not clock_was_visible
                    active_threshold = start_threshold if waiting_first else threshold
                    active_stable_required = (
                        first_stable_required if waiting_first else stable_frames_required
                    )
                    clock_visible = score >= active_threshold and margin_ok
                    clock_maybe_present = score >= presence_threshold

                    if clock_maybe_present:
                        last_clock_activity = now
                        if idle_mode:
                            idle_mode = False
                            target_frame_delay = frame_delay
                            self.event_queue.put(("idle_resumed",))

                    if number_saved:
                        possible_clock_seen = False
                        possible_clock_lost_frames = 0
                    elif clock_maybe_present:
                        possible_clock_seen = True
                        possible_clock_lost_frames = 0
                        if not clock_visible:
                            waited_below_threshold = True
                    elif possible_clock_seen:
                        possible_clock_lost_frames += 1

                    if clock_visible:
                        stable_labels.append(label)
                        stable_label, stable_count = Counter(stable_labels).most_common(1)[0]

                        if stable_count >= active_stable_required and last_label != stable_label:
                            matched_label = stable_label
                            last_label = matched_label
                            last_label_score = next(
                                (
                                    candidate_score
                                    for candidate_label, candidate_score in top_scores
                                    if candidate_label == matched_label
                                ),
                                score,
                            )
                        if last_label is not None and not number_saved:
                            displayed_label = self.output_label(last_label)
                            row_became_full = False
                            if len(found_numbers) < 8:
                                found_numbers.append(displayed_label)
                                found_template_labels.append(last_label)
                                row_became_full = len(found_numbers) == 8
                            self.event_queue.put(
                                (
                                    "number",
                                    displayed_label,
                                    list(found_numbers),
                                    last_label_score,
                                    list(found_template_labels),
                                    last_label,
                                )
                            )
                            number_saved = True
                            possible_clock_seen = False
                            possible_clock_lost_frames = 0
                            waited_below_threshold = False
                            if row_became_full:
                                self.event_queue.put(("row_full",))

                        clock_was_visible = True
                        lost_frames = 0
                    elif clock_was_visible and not clock_maybe_present:
                        lost_frames += 1
                        if lost_frames >= lost_frames_required:
                            clock_was_visible = False
                            number_saved = False
                            last_label = None
                            last_label_score = 0.0
                            lost_frames = 0
                            stable_labels.clear()
                            waited_below_threshold = False
                    elif clock_was_visible:
                        lost_frames = 0

                    if (
                        possible_clock_seen
                        and not number_saved
                        and possible_clock_lost_frames >= lost_frames_required
                    ):
                        last_label = None
                        last_label_score = 0.0
                        stable_labels.clear()
                        clock_was_visible = False
                        number_saved = False
                        lost_frames = 0
                        possible_clock_seen = False
                        possible_clock_lost_frames = 0
                        waited_below_threshold = False
                        found_numbers.clear()
                        found_template_labels.clear()
                        self.event_queue.put(("invalid_clock_restart",))

                elapsed = time.monotonic() - started
                if elapsed < target_frame_delay:
                    time.sleep(target_frame_delay - elapsed)
        except Exception as error:
            self.event_queue.put(("error", str(error)))
        finally:
            capture.release()

    def refine_content_box(
        self,
        frame,
        coarse_box: tuple[int, int, int, int],
    ) -> tuple[tuple[int, int, int, int], float]:
        frame_height, frame_width = frame.shape[:2]
        coarse_x, coarse_y, coarse_width, coarse_height = coarse_box
        refinement_range = max(
            0.10,
            float(self.config.get("obs_content_refinement_range", 0.18)),
        )
        padding_x = max(10, int(round(coarse_width * (refinement_range + 0.10))))
        padding_y = max(10, int(round(coarse_height * (refinement_range + 0.10))))
        left = max(0, coarse_x - padding_x)
        top = max(0, coarse_y - padding_y)
        right = min(frame_width, coarse_x + coarse_width + padding_x)
        bottom = min(frame_height, coarse_y + coarse_height + padding_y)
        search_gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)

        best_score = -1.0
        best_box = coarse_box
        minimum_width = max(8, int(round(coarse_width * (1.0 - refinement_range))))
        maximum_width = max(
            minimum_width,
            int(round(coarse_width * (1.0 + refinement_range))),
        )
        coarse_width_step = max(1, int(round(coarse_width / 100)))

        for template_image in self.alignment_refinement_templates:
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            template_height, template_width = template_gray.shape[:2]
            aspect = template_height / max(1, template_width)
            for width in range(minimum_width, maximum_width + 1, coarse_width_step):
                height = max(8, int(round(width * aspect)))
                if width >= search_gray.shape[1] or height >= search_gray.shape[0]:
                    continue
                resized_template = cv2.resize(
                    template_gray,
                    (width, height),
                    interpolation=(
                        cv2.INTER_CUBIC
                        if width > template_width
                        else cv2.INTER_AREA
                    ),
                )
                score_map = cv2.matchTemplate(
                    search_gray,
                    resized_template,
                    cv2.TM_CCOEFF_NORMED,
                )
                _, score, _, location = cv2.minMaxLoc(score_map)
                if score > best_score:
                    best_score = float(score)
                    best_box = (
                        left + int(location[0]),
                        top + int(location[1]),
                        width,
                        height,
                    )

        fine_center_width = best_box[2]
        fine_minimum_width = max(minimum_width, fine_center_width - coarse_width_step)
        fine_maximum_width = min(maximum_width, fine_center_width + coarse_width_step)
        for template_image in self.alignment_refinement_templates:
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            template_height, template_width = template_gray.shape[:2]
            aspect = template_height / max(1, template_width)
            for width in range(fine_minimum_width, fine_maximum_width + 1):
                height = max(8, int(round(width * aspect)))
                if width >= search_gray.shape[1] or height >= search_gray.shape[0]:
                    continue
                resized_template = cv2.resize(
                    template_gray,
                    (width, height),
                    interpolation=(
                        cv2.INTER_CUBIC
                        if width > template_width
                        else cv2.INTER_AREA
                    ),
                )
                score_map = cv2.matchTemplate(
                    search_gray,
                    resized_template,
                    cv2.TM_CCOEFF_NORMED,
                )
                _, score, _, location = cv2.minMaxLoc(score_map)
                if score > best_score:
                    best_score = float(score)
                    best_box = (
                        left + int(location[0]),
                        top + int(location[1]),
                        width,
                        height,
                    )
        return best_box, best_score

    def apply_content_alignment(self, frame, matrix):
        with self.roi_lock:
            reference_width = int(self.roi.get("reference_width", frame.shape[1]))
            reference_height = int(self.roi.get("reference_height", frame.shape[0]))
        return cv2.warpAffine(
            frame,
            matrix,
            (reference_width, reference_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def apply_roi_alignment(self, frame, matrix, roi):
        key = (matrix.tobytes(), roi["x"], roi["y"], roi["width"], roi["height"])
        cached = getattr(self, "_roi_maps", None)
        if cached is None or cached[0] != key:
            inverse = cv2.invertAffineTransform(matrix.astype(np.float64))
            x = np.arange(roi["x"], roi["x"] + roi["width"], dtype=np.float64)[None, :]
            y = np.arange(roi["y"], roi["y"] + roi["height"], dtype=np.float64)[:, None]
            # Match warpAffine's 10-bit accumulation and 5-bit interpolation grid.
            sx = (np.rint(inverse[0, 0] * x * 1024).astype(np.int64) + np.rint((inverse[0, 1] * y + inverse[0, 2]) * 1024).astype(np.int64) + 16) >> 5
            sy = (np.rint(inverse[1, 0] * x * 1024).astype(np.int64) + np.rint((inverse[1, 1] * y + inverse[1, 2]) * 1024).astype(np.int64) + 16) >> 5
            map_xy = np.stack((sx >> 5, sy >> 5), axis=-1).clip(-32768, 32767).astype(np.int16)
            map_fraction = (((sy & 31) << 5) + (sx & 31)).astype(np.uint16)
            cached = (key, map_xy, map_fraction)
            self._roi_maps = cached
        return cv2.remap(
            frame, cached[1], cached[2], interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def crop_current_roi(self, frame, roi: dict):
        x, y, width, height = scaled_roi(frame, roi)
        cropped = frame[y : y + height, x : x + width]
        if cropped.size == 0:
            raise RuntimeError("The ROI is outside the projector window.")
        return cropped

    def put_latest_frame(self, frame) -> None:
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(frame)

    def poll_queues(self) -> None:
        if self.search_stopped_event.is_set():
            self.current_frame = None
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
        else:
            new_frame = False
            try:
                while True:
                    self.current_frame = self.frame_queue.get_nowait()
                    new_frame = True
            except queue.Empty:
                pass

            if new_frame and self.current_frame is not None and not self.screen_tracking_lost:
                self.draw_frame(self.current_frame)

        try:
            while True:
                event = self.event_queue.get_nowait()
                if self.search_stopped_event.is_set():
                    continue
                if event[0] == "alignment_found" and event[6] != self.alignment_generation:
                    continue
                if event[0] == "screen_tracking":
                    self.tracking_warning_visible = event[1]
                    self.status_var.set(
                        "Screen tracking lost. Please bring all four calibration points back into view."
                        if event[1] else "Screen tracking restored. Detection can continue."
                    )
                    if event[1]:
                        self.draw_tracking_warning()
                    elif self.current_frame is not None:
                        self.draw_frame(self.current_frame)
                elif event[0] == "number":
                    number = event[1]
                    numbers = event[2]
                    score = event[3]
                    template_labels = event[4]
                    current_template_label = event[5]
                    self.last_number_var.set(number)
                    self.last_score_var.set(f"Score: {score:.3f}")
                    self.render_last_template(current_template_label)
                    self.numbers_text_var.set(",".join(numbers[:8]))
                    self.render_template_row(template_labels[:8])
                elif event[0] == "row_full":
                    self.status_var.set("The 8-number row is full. Last Number will keep updating.")
                elif event[0] == "idle":
                    self.status_var.set(
                        "Idle: detection paused. Press Restart before showing the next clock."
                    )
                elif event[0] == "idle_resumed":
                    self.status_var.set(
                        "Clock activity detected. Full-speed detection resumed."
                    )
                elif event[0] == "invalid_clock_restart":
                    self.last_number_var.set("-")
                    self.last_score_var.set("Score: -")
                    self.render_last_template(None)
                    self.numbers_text_var.set("")
                    self.render_template_row([])
                    self.status_var.set(
                        "No valid number detected. The sequence was restarted automatically."
                    )
                elif event[0] == "alignment_found":
                    matrix = event[1]
                    score = event[2]
                    detected_box = event[3]
                    source_width = event[4]
                    source_height = event[5]
                    self.config["obs_content_alignment"] = {
                        "matrix": matrix,
                        "source_width": source_width,
                        "source_height": source_height,
                        "detected_box": list(detected_box),
                        "score": score,
                    }
                    self.save_config()
                    if min(detected_box[2], detected_box[3]) < int(
                        self.config.get("obs_content_small_clock_pixel_limit", 35)
                    ):
                        self.status_var.set(
                            "Clock calibrated at low resolution. Restart the game, then press Restart."
                        )
                    else:
                        self.status_var.set(
                            "Clock calibrated. Restart the game, then press Restart before the first clock."
                        )
                    if not self.calibration_restart_notice_shown:
                        self.calibration_restart_notice_shown = True
                        capture_guidance = (
                            "Keep the camera and console in the same position. "
                            if self.webcam_mode_var.get() else
                            "Keep the capture window and image scale unchanged. "
                        )
                        messagebox.showinfo(
                            "Restart the Game After Calibration",
                            "Calibration is complete.\n\n"
                            "Restart the game before collecting your clock sequence. "
                            "The clock used for calibration is not included in the sequence.\n\n"
                            + capture_guidance
                            + "After restarting the game, click \"Restart\" in this tool "
                            "before showing the first clock.",
                            parent=self.root,
                        )
                elif event[0] == "alignment_failed":
                    self.status_var.set(
                        "Alignment score stayed below 0.900. Resize the OBS image and try again."
                    )
                    messagebox.showwarning(
                        "Clock Alignment Failed",
                        "No clock alignment reached a score above 0.900.\n\n"
                        "Please make the image in OBS larger or smaller, then click "
                        "\"Set Clock Box\" and try again.",
                    )
                elif event[0] == "alignment_size_changed":
                    self.status_var.set(
                        "Selected window size changed. Set Clock Box again."
                    )
                elif event[0] == "capture_error":
                    self.detect_event.clear()
                    self.status_var.set(
                        f"Window capture unavailable: {event[1]} Select another window."
                    )
                elif event[0] == "error":
                    self.detect_event.clear()
                    self.status_var.set("Capture error.")
                    messagebox.showerror("Gen 7 Clock RNG", event[1])
        except queue.Empty:
            pass

        if not self.stop_event.is_set():
            self.root.after(30, self.poll_queues)

    def render_template_row(self, numbers: list[str]) -> None:
        for child in self.templates_row.winfo_children():
            child.destroy()
        self.template_photos.clear()

        for number in numbers:
            template = self.template_by_label.get(number)
            if template is None:
                continue

            rgb = cv2.cvtColor(template.display_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((32, 32), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.template_photos.append(photo)

            tk.Label(
                self.templates_row,
                image=photo,
                bg="#151719",
                bd=0,
                padx=4,
                pady=4,
            ).pack(side="left", padx=3)

        self.templates_row.update_idletasks()
        self.update_template_scrollregion()
        self.templates_canvas.xview_moveto(1.0)

    def render_last_template(self, number: str | None) -> None:
        template = self.template_by_label.get(number) if number is not None else None
        if template is None:
            self.last_template_photo = None
            self.last_template_label.configure(image="")
            return

        rgb = cv2.cvtColor(template.display_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((48, 48), Image.Resampling.LANCZOS)
        self.last_template_photo = ImageTk.PhotoImage(image)
        self.last_template_label.configure(image=self.last_template_photo)

    def update_template_scrollregion(self, _event=None) -> None:
        self.templates_canvas.configure(scrollregion=self.templates_canvas.bbox("all"))

    def resize_template_window(self, event) -> None:
        requested_width = self.templates_row.winfo_reqwidth()
        self.templates_canvas.itemconfigure(self.templates_window, width=max(event.width, requested_width))

    def draw_frame(self, frame) -> None:
        if self.tracking_warning_visible:
            self.draw_tracking_warning()
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        frame_height, frame_width = frame.shape[:2]
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        display_width = max(1, int(frame_width * scale))
        display_height = max(1, int(frame_height * scale))
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2

        resized = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.photo)
        if self.selecting_screen:
            for index, (x, y) in enumerate(self.screen_points, 1):
                px, py = offset_x + x * scale, offset_y + y * scale
                self.canvas.create_oval(px-5, py-5, px+5, py+5, outline="#4fc3f7", width=2)
                self.canvas.create_text(px+12, py, text=str(index), fill="#4fc3f7")

        if (
            self.selecting_alignment
            and self.alignment_selection_start is not None
            and self.alignment_selection_end is not None
        ):
            start_x, start_y = self.alignment_selection_start
            end_x, end_y = self.alignment_selection_end
            self.canvas.create_rectangle(
                start_x,
                start_y,
                end_x,
                end_y,
                outline="#4fc3f7",
                width=3,
                dash=(6, 4),
                tags="alignment_selection",
            )

        self.display_scale = scale
        self.display_offset_x = offset_x
        self.display_offset_y = offset_y

    def resize_preview_warning(self, _event=None) -> None:
        if self.tracking_warning_visible and not self.search_stopped_event.is_set():
            self.draw_tracking_warning()

    def draw_tracking_warning(self) -> None:
        width = max(200, self.canvas.winfo_width())
        height = max(200, self.canvas.winfo_height())
        self.draw_black_preview()
        self.canvas.create_rectangle(0, 0, width, height, fill="#260b0d", outline="", tags="tracking_warning")
        self.canvas.create_text(
            width / 2, height / 2 - 65, text="SCREEN TRACKING LOST",
            font=("Segoe UI", 24, "bold"), fill="#ff7979", width=width - 40,
            justify="center", anchor="s", tags="tracking_warning",
        )
        self.canvas.create_text(
            width / 2, height / 2 - 30,
            text="Please bring all four calibration points back into view.\n\nRecognition is paused.",
            font=("Segoe UI", 13, "bold"), fill="#ffffff", width=width - 48,
            justify="center", anchor="n", tags="tracking_warning",
        )

    def draw_black_preview(self) -> None:
        self.photo = None
        self.canvas.delete("all")
        self.canvas.configure(bg="#000000")

    def start_alignment_drag(self, event) -> None:
        if self.selecting_screen:
            self.add_screen_corner(event)
            return
        if not self.selecting_alignment:
            return
        point = self.clamp_canvas_point_to_frame(event.x, event.y)
        self.alignment_selection_start = point
        self.alignment_selection_end = point

    def drag_alignment_box(self, event) -> None:
        if self.selecting_alignment and self.alignment_selection_start is not None:
            self.alignment_selection_end = self.clamp_canvas_point_to_frame(
                event.x,
                event.y,
            )

    def end_alignment_drag(self, event) -> None:
        if self.selecting_alignment and self.alignment_selection_start is not None:
            self.alignment_selection_end = self.clamp_canvas_point_to_frame(
                event.x,
                event.y,
            )
            self.finish_alignment_selection()

    def clamp_canvas_point_to_frame(
        self,
        canvas_x: int,
        canvas_y: int,
    ) -> tuple[int, int]:
        if self.current_frame is None:
            return canvas_x, canvas_y

        frame_height, frame_width = self.current_frame.shape[:2]
        right = int(round(self.display_offset_x + frame_width * self.display_scale))
        bottom = int(round(self.display_offset_y + frame_height * self.display_scale))
        return (
            min(max(canvas_x, self.display_offset_x), right),
            min(max(canvas_y, self.display_offset_y), bottom),
        )

    def close(self) -> None:
        self.detect_event.clear()
        self.stop_event.set()
        self.capture_enabled_event.set()
        self.alignment_wait_stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=1.5)
        if self.alignment_wait_worker is not None:
            self.alignment_wait_worker.join(timeout=1.0)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ClockDetectorGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
