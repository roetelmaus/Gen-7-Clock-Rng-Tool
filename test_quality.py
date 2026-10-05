import json
import time
from pathlib import Path

import cv2
import numpy as np

from obs_clock_detector import load_templates, make_discriminative_weight_map, compare_to_templates
from quality_matcher import QualityMatcher


def run():
    base = Path(__file__).parent
    templates = load_templates(base / 'templates', (75, 74), warn_on_size_mismatch=False)
    weights = make_discriminative_weight_map(templates, 1.6)
    matcher = QualityMatcher(templates, weights)
    cases = {
        'original': lambda im: im,
        'shift1': lambda im: cv2.warpAffine(im, np.float32([[1, 0, 1], [0, 1, 1]]), (75, 74), borderMode=cv2.BORDER_REFLECT),
        'small40': lambda im: cv2.resize(cv2.resize(im, (40, 40), interpolation=cv2.INTER_AREA), (75, 74), interpolation=cv2.INTER_CUBIC),
        'blur1.2': lambda im: cv2.GaussianBlur(im, (5, 5), 1.2),
        'blur0.95_shift1': lambda im: cv2.warpAffine(cv2.GaussianBlur(im, (5, 5), .95), np.float32([[1, 0, -1], [0, 1, 1]]), (75, 74), borderMode=cv2.BORDER_REFLECT),
    }
    report = {}
    for mode, transform in cases.items():
        report[mode] = {}
        for version in ('v1', 'v2'):
            wrong, accepted_wrong, rejected = [], [], 0
            start = time.perf_counter()
            for template in templates:
                im = transform(cv2.resize(template.display_image, (75, 74)))
                result = matcher.compare(im) if version == 'v2' else compare_to_templates(im, templates, weights, False, .04, 17)
                label, score, margin, _ = result
                accepted = score >= .9 and (version == 'v1' or margin >= .02)
                rejected += not accepted
                if label != template.label:
                    wrong.append([template.label, label, round(score, 4)])
                    if accepted:
                        accepted_wrong.append(wrong[-1])
            report[mode][version] = dict(total=len(templates), wrong=wrong, accepted_wrong=accepted_wrong, rejected=rejected, ms_per_frame=round((time.perf_counter()-start)*1000/len(templates), 2))
    for value in (0, 127, 255):
        assert matcher.compare(np.full((74, 75, 3), value, np.uint8))[1] < .9
    print(json.dumps(report, indent=2))
    assert not report['original']['v2']['wrong']
    assert all(not results['v2']['accepted_wrong'] for results in report.values())
    return report


if __name__ == '__main__':
    run()
