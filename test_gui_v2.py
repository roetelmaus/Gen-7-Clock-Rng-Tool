"""GUI smoke checks with capture disabled; no game interaction."""
import copy
import tkinter as tk
import numpy as np
from unittest.mock import patch
from obs_clock_gui import ClockDetectorGui


root = tk.Tk()
root.withdraw()
with patch.object(ClockDetectorGui, 'start_worker'), patch.object(ClockDetectorGui, 'save_config'):
    app = ClockDetectorGui(root)
    roi = copy.deepcopy(app.roi)
    app.alignment_matrix = np.float32([[1, 0, 0], [0, 1, 0]])
    app.roi_preview_event.set()
    app.detect_event.set()
    app.stop_searching()
    assert app.search_stopped_event.is_set()
    assert not app.detect_event.is_set()
    assert not app.canvas.find_all()
    assert app.roi == roi and app.alignment_matrix is not None
    app.restart_detection()
    root.after(150, root.quit)
    root.mainloop()
    assert not app.search_stopped_event.is_set() and app.detect_event.is_set()
    assert app.roi_preview_event.is_set()
    assert app.roi == roi
    app.current_frame = np.zeros((240, 320, 3), np.uint8)
    app.frame_queue.put(app.current_frame)
    with patch.object(app, 'draw_frame') as draw:
        app.poll_queues()
        app.poll_queues()
        assert draw.call_count == 1
    app.stop_searching()
    app.start_detection()
    assert not app.detect_event.is_set()
    app.close()
print('GUI: stop/restart preserves ROI, no repeated redraw, stopped state respected')
