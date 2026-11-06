"""Detector contract. YoloDetector tests need ultralytics + stock weights and are skipped otherwise:
    SAFESITE_TEST_WEIGHTS=/path/to/yolo26n.pt pytest tests/test_detect.py
"""
from __future__ import annotations

import importlib.util
import os

import pytest
from PIL import Image

from safesite import detect
from safesite.types import UNIFIED_CLASSES

WEIGHTS = os.environ.get("SAFESITE_TEST_WEIGHTS", "")
needs_yolo = pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None or not os.path.isfile(WEIGHTS),
    reason="needs ultralytics and SAFESITE_TEST_WEIGHTS pointing at a .pt file",
)


def test_check_labels_accepts_exact_unified_classes():
    detect.check_labels(set(UNIFIED_CLASSES))


@pytest.mark.parametrize("names", [{"hardhat", "no-hardhat", "person"}, set(UNIFIED_CLASSES) | {"dog"}])
def test_check_labels_rejects_mismatched_classes(names):
    with pytest.raises(detect.LabelMismatchError):
        detect.check_labels(names)


def test_mock_boxes_are_normalized():
    for mode in ("site", "home"):
        for d in detect.MockDetector()(Image.new("RGB", (64, 64)), mode):
            x1, y1, x2, y2 = d.box
            assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1


@needs_yolo
def test_yolo_detector_refuses_weights_with_wrong_classes():
    with pytest.raises(detect.LabelMismatchError):
        detect.YoloDetector(weights_path=WEIGHTS)  # stock COCO weights


@needs_yolo
def test_yolo_detector_returns_normalized_detections():
    model = detect.YoloDetector(weights_path=WEIGHTS, strict_labels=False, threads=2)
    assert model.name.startswith("YOLO26") and not model.is_mock
    for d in model(Image.new("RGB", (640, 480), (90, 90, 90))):
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1
        assert isinstance(d.label, str) and d.source == "detector"
