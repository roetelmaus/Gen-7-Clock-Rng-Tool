"""Synthetic screen movement, occlusion, and recovery checks."""
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from screen_tracker import ScreenTracker


class ScreenTrackerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.frame = np.full((360, 640, 3), 210, dtype=np.uint8)
        self.frame[70:311, 100:501] = 10
        self.frame[76:305, 106:495] = rng.integers(0, 256, (229, 389, 3), dtype=np.uint8)
        self.points = [(100, 70), (500, 70), (500, 310), (100, 310)]

    def test_translation_and_recovery(self):
        tracker = ScreenTracker(self.frame, self.points)
        original = tracker.rectify(self.frame, 1.0)
        moved = cv2.warpAffine(self.frame, np.float32([[1, 0, 18], [0, 1, -12]]), (640, 360))
        result = tracker.rectify(moved, 1.3)
        self.assertIsNotNone(result)
        self.assertEqual(original.shape, result.shape)
        expected = np.float32(self.points) + [18, -12]
        # Polygon approximation on the noisy synthetic display is accurate to 3 px.
        self.assertLess(np.linalg.norm(tracker.points - expected, axis=1).max(), 3)
        self.assertIsNone(tracker.rectify(np.zeros_like(moved), 1.6))
        self.assertIsNotNone(tracker.rectify(moved, 2.2))

    def test_invalid_selection_and_window_resize(self):
        with self.assertRaises(ValueError):
            ScreenTracker(self.frame, [(100, 70)] * 4)
        tracker = ScreenTracker(self.frame, self.points)
        self.assertIsNone(tracker.rectify(self.frame[:200], 1.0))

    def test_black_screen_movement_and_menu_change(self):
        tracker = ScreenTracker(self.frame, self.points)
        tracker.rectify(self.frame, 1.0)
        black = self.frame.copy()
        black[70:311, 100:501] = 0
        moved = cv2.warpAffine(black, np.float32([[1, 0, 16], [0, 1, 9]]), (640, 360), borderValue=(210, 210, 210))
        self.assertIsNotNone(tracker.rectify(moved, 1.3))
        expected = np.float32(self.points) + [16, 9]
        self.assertLess(np.linalg.norm(tracker.points - expected, axis=1).max(), 3)
        moved[85:314, 122:511] = (180, 90, 220)
        self.assertIsNotNone(tracker.rectify(moved, 1.6))

    def test_large_translation_and_rotation(self):
        tracker = ScreenTracker(self.frame, self.points)
        matrix = cv2.getRotationMatrix2D((300, 190), 8, 1.05)
        matrix[:, 2] += [55, -10]
        moved = cv2.warpAffine(self.frame, matrix, (640, 360), borderValue=(210, 210, 210))
        self.assertIsNotNone(tracker.rectify(moved, 1.0))
        expected = cv2.transform(np.float32(self.points)[None], matrix)[0]
        self.assertLess(np.linalg.norm(tracker.points - expected, axis=1).max(), 4)

    def test_reacquisition_does_not_wait_half_a_second(self):
        tracker = ScreenTracker(self.frame, self.points)
        blank = self.frame.copy()
        blank[70:311, 100:501] = 0
        with patch.object(tracker, "find_border", return_value=None) as find:
            self.assertIsNone(tracker.rectify(blank, 1.0))
            self.assertIsNone(tracker.rectify(blank, 1.025))
            self.assertEqual(find.call_count, 2)
        self.assertTrue(tracker.black_screen)
        self.assertEqual(tracker.dark_frames, 2)
        self.assertIsNotNone(tracker.rectify(self.frame, 1.05))
        self.assertFalse(tracker.black_screen)
        self.assertEqual(tracker.dark_frames, 0)

    def test_real_menu_does_not_replace_display_bottom(self):
        path = Path(__file__).parent / "camo_diagnostics" / "initial.png"
        if not path.exists():
            self.skipTest("Local Camo recording is not distributed")
        frame = cv2.imread(str(path))
        points = [(548, 278), (972, 290), (989, 548), (543, 570)]
        tracker = ScreenTracker(frame, points)
        self.assertIsNotNone(tracker.rectify(frame, 1.0))
        self.assertLess(np.linalg.norm(tracker.points / tracker.scale - points, axis=1).max(), 1)
        # Menu changes are simulated without changing the physical screen border.
        black = frame.copy()
        cv2.fillConvexPoly(black, np.int32([(554, 285), (965, 296), (981, 543), (549, 562)]), (0, 0, 0))
        self.assertIsNotNone(tracker.rectify(black, 1.3))
        self.assertGreater(float((tracker.points / tracker.scale)[2:, 1].min()), 530)


if __name__ == "__main__":
    unittest.main()
