"""GUI smoke checks with capture disabled; no game interaction."""
import copy
import tkinter as tk
import numpy as np
from unittest.mock import patch
from types import SimpleNamespace
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
    alignment_event = (
        'alignment_found', [[1, 0, 0], [0, 1, 0]], 0.95,
        (10, 10, 60, 59), 320, 240, app.alignment_generation,
    )
    with patch('obs_clock_gui.messagebox.showinfo') as notice:
        app.event_queue.put(alignment_event)
        app.poll_queues()
        app.event_queue.put(alignment_event)
        app.poll_queues()
        assert notice.call_count == 1
        assert 'Restart the game' in notice.call_args.args[1]
    app.current_frame = np.zeros((240, 320, 3), np.uint8)
    app.frame_queue.put(app.current_frame)
    with patch.object(app, 'draw_frame') as draw:
        app.poll_queues()
        app.poll_queues()
        assert draw.call_count == 1
    app.stop_searching()
    app.start_detection()
    assert not app.detect_event.is_set()
    app.restart_detection()
    app.webcam_mode_var.set(True)
    app.begin_screen_selection()
    frame = np.full((360, 640, 3), 210, dtype=np.uint8)
    frame[70:311, 100:501] = 10
    app.screen_selection_frame = frame
    app.current_frame = frame
    app.draw_frame(frame)
    for x, y in [(100, 70), (500, 70), (500, 310), (100, 310)]:
        app.add_screen_corner(SimpleNamespace(
            x=app.display_offset_x + x * app.display_scale,
            y=app.display_offset_y + y * app.display_scale,
        ))
    assert app.screen_tracker is not None and not app.selecting_screen
    app.update_screen_tracking_status(True, 10.0)
    app.update_screen_tracking_status(True, 11.0)
    assert not app.screen_tracking_lost
    app.update_screen_tracking_status(False, 11.5)
    assert app.screen_tracking_loss_since is None
    app.update_screen_tracking_status(True, 20.0)
    app.update_screen_tracking_status(True, 21.0)
    assert not app.screen_tracking_lost
    app.update_screen_tracking_status(True, 22.1)
    assert app.screen_tracking_lost
    app.update_screen_tracking_status(False, 22.2)
    assert not app.screen_tracking_lost
    app.poll_queues()
    app.numbers_text_var.set('1,2,3')
    app.update_screen_tracking_status(True, 40.0, transitioning=True)
    app.update_screen_tracking_status(True, 46.0, transitioning=True)
    assert not app.screen_tracking_lost
    app.update_screen_tracking_status(False, 46.1)
    assert app.numbers_text_var.get() == '1,2,3'
    app.event_queue.put(('screen_tracking', True))
    app.poll_queues()
    assert 'calibration points' in app.status_var.get()
    assert app.canvas.find_withtag('tracking_warning')
    app.draw_frame(frame)
    assert app.canvas.find_withtag('tracking_warning')
    app.event_queue.put(('screen_tracking', False))
    app.poll_queues()
    assert not app.canvas.find_withtag('tracking_warning')
    assert app.numbers_text_var.get() == '1,2,3'
    app.webcam_mode_var.set(False)
    app.change_webcam_mode()
    assert app.screen_tracker is None and app.alignment_matrix is None
    with patch('obs_clock_gui.messagebox.showinfo') as notice:
        app.event_queue.put((*alignment_event[:6], app.alignment_generation))
        app.poll_queues()
        assert notice.call_count == 1
        assert 'capture window' in notice.call_args.args[1]
        assert 'before showing the first clock' in notice.call_args.args[1]
    app.close()
print('GUI: stop/restart preserves ROI, no repeated redraw, stopped state respected')
