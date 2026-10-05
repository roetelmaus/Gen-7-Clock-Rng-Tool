from __future__ import annotations

from window_capture import list_visible_windows


def main() -> None:
    for window in list_visible_windows():
        print(f"{window.title}  [{window.left},{window.top} {window.width}x{window.height}]")


if __name__ == "__main__":
    main()
