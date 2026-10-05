"""One aligned pixel metric for selection, confidence and class margin."""
import cv2
import numpy as np

from obs_clock_detector import preprocess_for_matching


class QualityMatcher:
    def __init__(self, templates, weight_map, radius=2, quality_variants=True):
        self.size = templates[0].image.shape[::-1]
        self.radius = max(0, min(int(radius), 4))
        self.labels = []
        images = []
        for template in templates:
            source = cv2.resize(template.display_image, self.size, interpolation=cv2.INTER_AREA)
            variants = [source]
            if quality_variants:
                variants.extend(cv2.GaussianBlur(source, (5, 5), sigma) for sigma in (0.7, 1.2))
                small = cv2.resize(source, (max(8, round(self.size[0] * .65)), max(8, round(self.size[1] * .65))), interpolation=cv2.INTER_AREA)
                variants.append(cv2.resize(small, self.size, interpolation=cv2.INTER_CUBIC))
            for variant in variants:
                self.labels.append(template.label)
                images.append(self.inner(preprocess_for_matching(variant, self.size)).ravel())
        self.weights = self.inner(weight_map).ravel().astype(np.float32)
        self.weights /= self.weights.sum()
        self.bank = np.asarray(images, dtype=np.float32)
        means = self.bank @ self.weights
        centered = self.bank - means[:, None]
        norms = np.sqrt(np.sum(centered * centered * self.weights, axis=1))
        self.normalized = centered * self.weights / np.maximum(norms[:, None], 1e-6)
        self.bank_square = np.sum(self.bank * self.bank * self.weights, axis=1)
        self.weighted_bank = self.bank * self.weights
        self.classes = sorted(set(self.labels), key=int)
        self.class_indices = [np.flatnonzero(np.array(self.labels) == label) for label in self.classes]

    def inner(self, image):
        r = self.radius
        return image[r:-r, r:-r] if r else image

    def compare(self, frame):
        resized = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        prepared = preprocess_for_matching(resized, self.size)
        rows = []
        # Compare all integer translations. Ignore artificial border pixels.
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                shifted = cv2.warpAffine(prepared, np.float32([[1, 0, dx], [0, 1, dy]]), self.size, borderMode=cv2.BORDER_REFLECT)
                rows.append(self.inner(shifted).ravel())
        rows = np.asarray(rows, dtype=np.float32)
        means = rows @ self.weights
        centered = rows - means[:, None]
        norms = np.sqrt(np.sum(centered * centered * self.weights, axis=1))
        correlations = np.einsum('ik,jk->ij', centered, self.normalized, optimize=False) / np.maximum(norms[:, None], 1e-6)
        mse = np.sum(rows * rows * self.weights, axis=1)[:, None] + self.bank_square[None, :] - 2 * np.einsum('ik,jk->ij', rows, self.weighted_bank, optimize=False)
        scores = .8 * correlations + .2 * (1 - np.clip(mse / 65025, 0, 1))
        scores[norms < 1e-6, :] = -1
        best = np.max(scores, axis=0)
        ranked = sorted(((label, float(np.max(best[indices]))) for label, indices in zip(self.classes, self.class_indices)), key=lambda item: item[1], reverse=True)
        label, score = ranked[0]
        margin = score - ranked[1][1] if len(ranked) > 1 else score
        return label, score, margin, ranked[:5]
