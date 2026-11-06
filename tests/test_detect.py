"""Detector contract. YoloDetector tests need ultralytics + stock weights and are skipped otherwise:
    SAFESITE_TEST_WEIGHTS=/path/to/yolo26n.pt pytest tests/test_detect.py
"""
from __future__ import annotations

import importlib.util
import os

import pytest
from PIL import Image

from safesite import detect
from safesite.types import EXTENDED_CLASSES, MODEL_CLASSES_V2, UNIFIED_CLASSES, Detection

WEIGHTS = os.environ.get("SAFESITE_TEST_WEIGHTS", "")
needs_yolo = pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None or not os.path.isfile(WEIGHTS),
    reason="needs ultralytics and SAFESITE_TEST_WEIGHTS pointing at a .pt file",
)


@pytest.mark.parametrize("value, expected", [(None, "dflt"), ("", "dflt"), ("   ", "dflt"), (" abc123 ", "abc123")])
def test_env_treats_blank_space_variables_as_unset(monkeypatch, value, expected):
    # A Space variable can exist but be empty; "" as a revision builds .../resolve//best.pt (404).
    if value is None:
        monkeypatch.delenv("SAFESITE_TEST_VAR", raising=False)
    else:
        monkeypatch.setenv("SAFESITE_TEST_VAR", value)
    assert detect._env("SAFESITE_TEST_VAR", "dflt") == expected


@pytest.mark.parametrize("classes", [UNIFIED_CLASSES, MODEL_CLASSES_V2], ids=["v1", "v2"])
def test_check_labels_accepts_v1_and_v2_class_lists(classes):
    assert detect.check_labels(list(classes)) == tuple(classes)


def test_check_labels_accepts_ultralytics_names_dict():
    # model.names is {id: name}; order comes from the ids, not dict insertion order
    names = {i: n for i, n in reversed(list(enumerate(MODEL_CLASSES_V2)))}
    assert detect.check_labels(names) == MODEL_CLASSES_V2


@pytest.mark.parametrize(
    "names",
    [
        ["hardhat", "no-hardhat", "person"],                      # stock / foreign classes
        [*UNIFIED_CLASSES, "dog"],                                # unknown extra class
        [*MODEL_CLASSES_V2, "dog"],                               # unknown extra after v2
        [UNIFIED_CLASSES[1], UNIFIED_CLASSES[0], *UNIFIED_CLASSES[2:]],  # right names, wrong order
        [*UNIFIED_CLASSES, *EXTENDED_CLASSES[:2]],                # partial v2
    ],
    ids=["foreign", "v1+extra", "v2+extra", "wrong-order", "partial-v2"],
)
def test_check_labels_rejects_mismatched_classes(names):
    with pytest.raises(detect.LabelMismatchError):
        detect.check_labels(names)


class _Fake:
    """Detector stub returning one fixed detection, recording the modes it was called with."""

    def __init__(self, label, name, is_mock=False, supports_home=False):
        self.label, self.name, self.is_mock, self.supports_home = label, name, is_mock, supports_home
        self.modes: list[str] = []

    def __call__(self, image, mode="site"):
        self.modes.append(mode)
        return [Detection(self.label, 0.9, (0.1, 0.1, 0.2, 0.2), "test")]


def test_combined_site_mode_concatenates_both_detectors():
    primary, finder = _Fake("person", "P"), _Fake("scaffold", "F", supports_home=True)
    found = detect.CombinedDetector(primary, finder)(Image.new("RGB", (8, 8)), "site")
    assert [d.label for d in found] == ["person", "scaffold"]


def test_combined_home_mode_uses_only_the_open_vocab_finder():
    primary, finder = _Fake("helmet", "P", supports_home=True), _Fake("knife", "F", supports_home=True)
    found = detect.CombinedDetector(primary, finder)(Image.new("RGB", (8, 8)), "home")
    assert [d.label for d in found] == ["knife"]
    assert primary.modes == []  # trained classes carry no home hazards, so it is not run


def test_combined_reports_name_mock_and_home_support():
    combined = detect.CombinedDetector(_Fake("person", "MOCK", is_mock=True), _Fake("cord", "YOLOE"))
    assert combined.name == "MOCK + YOLOE"
    assert combined.is_mock and combined.supports_home


def test_combined_tolerates_finder_without_contract_attributes():
    class Bare:
        name = "bare"

        def __call__(self, image, mode="site"):
            return []

    combined = detect.CombinedDetector(_Fake("person", "P"), Bare())
    assert combined.is_mock is False and combined.supports_home


def test_combined_detector_labels_are_the_union_when_both_parts_declare_them():
    primary, finder = _Fake("person", "P"), _Fake("harness", "F")
    primary.labels, finder.labels = frozenset(UNIFIED_CLASSES), frozenset({"harness", "rebar"})
    assert detect.CombinedDetector(primary, finder).labels == frozenset(UNIFIED_CLASSES) | {"harness", "rebar"}


def test_combined_detector_has_no_labels_when_a_part_does_not_declare_them():
    primary = _Fake("person", "P")
    primary.labels = frozenset(UNIFIED_CLASSES)
    assert getattr(detect.CombinedDetector(primary, _Fake("harness", "F")), "labels", None) is None


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
