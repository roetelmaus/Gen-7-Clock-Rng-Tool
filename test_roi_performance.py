import threading
import time
import cv2
import numpy as np
from obs_clock_gui import ClockDetectorGui

app = object.__new__(ClockDetectorGui)
app.roi_lock = threading.Lock()
app.roi = dict(x=802, y=84, width=75, height=74, reference_width=2000, reference_height=1177)
frame = np.random.default_rng(42).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
for scale in (0.65, 1.0, 1.1384615898, 1.5):
    matrix = np.float32([[scale, 0, 31.8636], [0, scale, -11.846]])
    full = app.crop_current_roi(app.apply_content_alignment(frame, matrix), app.roi)
    small = app.apply_roi_alignment(frame, matrix, app.roi)
    difference = np.abs(full.astype(float)-small.astype(float))
    print('scale', scale, 'max pixel difference', difference.max(), 'mean', difference.mean())
    assert np.array_equal(full, small)
matrix = np.float32([[1.1384615898, 0, 31.8636], [0, 1.1384615898, -11.846]])
for name, operation in [('full', lambda: app.apply_content_alignment(frame, matrix)), ('roi', lambda: app.apply_roi_alignment(frame, matrix, app.roi))]:
    operation()
    start = time.perf_counter()
    for _ in range(100):
        operation()
    print(name, 'ms/frame', round((time.perf_counter()-start)*10, 3))
