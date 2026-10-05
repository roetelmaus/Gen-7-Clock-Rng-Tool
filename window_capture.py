from __future__ import annotations

import ctypes
from dataclasses import dataclass

import cv2
import numpy as np
import win32gui
import win32ui


PW_CLIENTONLY = 0x00000001
PW_RENDERFULLCONTENT = 0x00000002


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


def list_visible_windows() -> list[WindowInfo]:
    windows: list[WindowInfo] = []

    def add_window(hwnd: int, _: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return

        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return

        windows.append(
            WindowInfo(
                hwnd=hwnd,
                title=title,
                left=left,
                top=top,
                width=width,
                height=height,
            )
        )

    win32gui.EnumWindows(add_window, None)
    return windows


def find_window(title_contains: str) -> WindowInfo:
    needle = title_contains.lower()
    visible_windows = list_visible_windows()
    matches = [window for window in visible_windows if needle in window.title.lower()]

    if not matches:
        available = "\n".join(f"- {window.title}" for window in visible_windows)
        raise RuntimeError(
            f"Kein Fenster gefunden, dessen Titel '{title_contains}' enthaelt.\n"
            f"Sichtbare Fenster:\n{available}"
        )

    return max(matches, key=lambda window: window.width * window.height)


def capture_window_client(hwnd: int) -> np.ndarray:
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise RuntimeError("The selected window has no capturable client area.")

    window_dc = win32gui.GetWindowDC(hwnd)
    source_dc = win32ui.CreateDCFromHandle(window_dc)
    memory_dc = source_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(source_dc, width, height)
    memory_dc.SelectObject(bitmap)

    try:
        result = ctypes.windll.user32.PrintWindow(
            hwnd,
            memory_dc.GetSafeHdc(),
            PW_CLIENTONLY | PW_RENDERFULLCONTENT,
        )
        if result != 1:
            raise RuntimeError("Windows could not capture the selected window.")

        bitmap_bytes = bitmap.GetBitmapBits(True)
        image = np.frombuffer(bitmap_bytes, dtype=np.uint8).reshape((height, width, 4))
        frame = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(frame)
    finally:
        win32gui.DeleteObject(bitmap.GetHandle())
        memory_dc.DeleteDC()
        source_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, window_dc)


class WindowCapture:
    def __init__(self, title_contains: str, hwnd: int | None = None):
        self.title_contains = title_contains
        self.hwnd = hwnd

    def set_window(self, hwnd: int, title: str) -> None:
        self.hwnd = hwnd
        self.title_contains = title

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.hwnd is not None:
            if not win32gui.IsWindow(self.hwnd):
                raise RuntimeError("The selected window is no longer available.")
            hwnd = self.hwnd
        else:
            hwnd = find_window(self.title_contains).hwnd

        if win32gui.IsIconic(hwnd):
            raise RuntimeError("The selected window must not be minimized.")

        frame = capture_window_client(hwnd)
        return True, frame

    def release(self) -> None:
        pass
