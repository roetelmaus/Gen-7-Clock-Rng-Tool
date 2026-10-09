"""Track complete display contours instead of independent menu edges."""
import cv2
import numpy as np


class ScreenTracker:
    def __init__(self, frame, points):
        self.scale = min(1.0, 640.0 / frame.shape[1])
        self.points = np.float32(points) * self.scale
        contour = self.points.reshape(-1, 1, 2)
        if (not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 600
                or self.points[1, 0] <= self.points[0, 0]
                or self.points[3, 1] <= self.points[0, 1]):
            raise ValueError("Select screen corners clockwise from top-left.")
        self.area = cv2.contourArea(contour)
        self.reference_lengths = np.linalg.norm(np.roll(self.points, -1, axis=0) - self.points, axis=1)
        initial_outline = self.find_border(self.gray(frame))
        if initial_outline is None:
            raise ValueError("Complete display border not found. Keep all four edges visible.")
        self.outline = initial_outline
        self.reference_outline = initial_outline.copy()
        self.reference_points = self.points.copy()
        self.source_shape = frame.shape[:2]
        self.last_check = 0.0
        self.valid = True
        self.black_screen = False
        self.dark_frames = 0
        self.update_matrix()

    def gray(self, frame):
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def update_matrix(self):
        self.matrix = cv2.getPerspectiveTransform(
            self.points / self.scale,
            np.float32([[0, 0], [399, 0], [399, 239], [0, 239]]),
        )

    def find_border(self, gray):
        anchor = getattr(self, "outline", self.points)
        smooth = cv2.GaussianBlur(gray, (3, 3), 0.6)
        edges = cv2.Canny(smooth, 35, 100)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            if not 0.65 < abs(cv2.contourArea(contour)) / self.area < 1.5:
                continue
            quad = cv2.approxPolyDP(contour, 0.015 * cv2.arcLength(contour, True), True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            points = quad[:, 0].astype(np.float32)
            if cv2.contourArea(points, oriented=True) < 0:
                points = points[::-1]
            points = min((np.roll(points, i, axis=0) for i in range(4)),
                         key=lambda p: np.linalg.norm(p - anchor, axis=1).mean())
            lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
            ratios = lengths / self.reference_lengths
            # Reject isolated side changes, such as a menu panel replacing the bottom edge.
            if np.max(ratios) / np.min(ratios) > 1.18 or not np.all((ratios > 0.75) & (ratios < 1.3)):
                continue
            movement = np.linalg.norm(points - anchor, axis=1)
            if movement.max() > max(self.reference_lengths) * 0.6:
                continue
            candidates.append((movement.mean() + 50 * np.std(ratios), points))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def rectify(self, frame, now):
        if frame.shape[:2] != self.source_shape:
            self.valid = False
            return None
        # A tiny screen-only sample detects fades without matching clock templates.
        sample_matrix = np.diag([32 / 400, 20 / 240, 1.0]) @ self.matrix
        sample = cv2.warpPerspective(frame, sample_matrix, (32, 20), flags=cv2.INTER_AREA)
        sample_gray = cv2.cvtColor(sample[3:-3, 3:-3], cv2.COLOR_BGR2GRAY)
        was_black = self.black_screen
        self.black_screen = bool(np.percentile(sample_gray, 90) < 65)
        self.dark_frames = self.dark_frames + 1 if self.black_screen else 0
        # Reacquire on the very next captured frame, especially when a fade ends.
        if not self.valid or was_black != self.black_screen or now - self.last_check >= 0.2:
            self.last_check = now
            points = self.find_border(self.gray(frame))
            self.valid = points is not None
            if not self.valid:
                return None
            # Hold the transform steady for small contour jitter.
            if np.max(np.linalg.norm(points - self.outline, axis=1)) > 1.5:
                motion = cv2.getPerspectiveTransform(self.reference_outline, points)
                self.points = cv2.perspectiveTransform(self.reference_points[None], motion)[0]
                self.outline = points
                self.update_matrix()
        if not self.valid:
            return None
        return cv2.warpPerspective(frame, self.matrix, (400, 240), flags=cv2.INTER_LINEAR)
