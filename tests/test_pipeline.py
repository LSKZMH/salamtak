"""Shared pipeline: the 'never silently safe' guarantees, downscaling and the size guard."""
from __future__ import annotations

from datetime import datetime

import pytest
from PIL import Image

from safesite import detect
from safesite.pipeline import MAX_SIDE, ImageTooLargeError, analyze_image, prepare_image
from safesite.types import Detection, SceneContext

WINTER = SceneContext(taken_at=datetime(2026, 11, 6, 9, 0))


class FixedDetector:
    is_mock, supports_home, name = False, False, "fixed"

    def __init__(self, dets):
        self._dets = dets

    def __call__(self, image, mode="site"):
        return list(self._dets)


def _img(w=4000, h=3000):
    return Image.new("RGB", (w, h), (120, 120, 120))


def test_empty_detection_is_not_assessed_rather_than_safe():
    _, result = analyze_image(FixedDetector([]), _img(), WINTER)
    assert result.assessed is False and result.hazards == ()
    assert any("can't be assessed" in n for n in result.notes)


def test_home_mode_with_detector_lacking_home_support_is_not_assessed():
    _, result = analyze_image(FixedDetector([]), _img(), SceneContext(taken_at=WINTER.taken_at, mode="home"))
    assert result.assessed is False


def test_mock_site_photo_is_assessed_with_expected_score():
    _, result = analyze_image(detect.MockDetector(), _img(1280, 853), WINTER)
    assert result.assessed and result.score == 40
    assert sorted(h.id for h in result.hazards) == ["machinery_zone", "no_helmet", "no_vest_near_vehicle"]


def test_low_confidence_detections_are_dropped_and_reported():
    dets = [Detection("person", 0.9, (0.1, 0.1, 0.3, 0.9)), Detection("helmet", 0.9, (0.12, 0.09, 0.28, 0.25)),
            Detection("vehicle", 0.1, (0.3, 0.1, 0.9, 0.9))]
    _, result = analyze_image(FixedDetector(dets), _img(), WINTER)
    assert all(d.confidence >= 0.35 for d in result.detections)
    assert any("1 low-confidence" in n for n in result.notes)


def test_high_wbgt_without_detected_workers_is_still_mentioned():
    _, result = analyze_image(FixedDetector([]), _img(), SceneContext(taken_at=WINTER.taken_at, wbgt_c=35.0))
    assert any("stop-work limit" in n for n in result.notes)


def test_prepare_image_downscales_and_keeps_input_untouched():
    src = _img(4000, 3000)
    out = prepare_image(src)
    assert max(out.size) == MAX_SIDE and src.size == (4000, 3000)


def test_prepare_image_rejects_huge_images():
    with pytest.raises(ImageTooLargeError):
        prepare_image(Image.new("L", (8000, 7000)))
