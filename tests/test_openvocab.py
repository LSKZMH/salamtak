"""Tests for safesite.openvocab. A fake YOLOE model stands in, so no ultralytics or weights needed."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from safesite import openvocab, rules
from safesite.openvocab import (
    HOME_PROMPTS,
    SITE_PROMPTS,
    OpenVocabDetector,
    prompts_for,
    to_detections,
)
from safesite.types import Detection


# ---------- prompt sets ----------
def test_site_prompts_cover_the_requested_objects():
    assert set(SITE_PROMPTS) == {
        "safety harness", "scaffolding", "ladder", "shade tent",
        "water cooler", "extension cord", "cable on the floor", "hose",
    }


def test_home_prompts_cover_the_requested_objects():
    assert set(HOME_PROMPTS) == {
        "extension cord", "cable on the floor", "wet floor", "knife", "cleaning chemicals",
    }


@pytest.mark.parametrize("prompts", [SITE_PROMPTS, HOME_PROMPTS], ids=["site", "home"])
def test_cable_on_the_floor_counts_as_a_cord(prompts):
    # "extension cord" alone missed every real cord photo we tried; this wording finds them.
    assert prompts["cable on the floor"] == prompts["extension cord"] == "cord"


def test_every_prompt_maps_to_a_label_rules_py_understands():
    labels = set(SITE_PROMPTS.values()) | set(HOME_PROMPTS.values())
    assert labels <= rules.OPEN_VOCAB_LABELS  # so rules.py gives them its open-vocab cut-off


def test_detector_meets_the_app_detector_contract():
    assert OpenVocabDetector.is_mock is False
    assert OpenVocabDetector.supports_home is True


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("site", SITE_PROMPTS), ("home", HOME_PROMPTS), ("HOME", HOME_PROMPTS),
     (SimpleNamespace(value="home"), HOME_PROMPTS)],
)
def test_prompts_for_accepts_strings_and_enum_like_modes(mode, expected):
    assert prompts_for(mode) is expected


def test_prompts_for_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        prompts_for("office")


# ---------- raw output -> Detection ----------
def test_to_detections_maps_class_ids_to_rule_labels():
    prompts = ("safety harness", "scaffolding")
    dets = to_detections([[0.1, 0.2, 0.3, 0.4]], [0.8], [1], prompts, SITE_PROMPTS)
    assert len(dets) == 1
    assert isinstance(dets[0], Detection)
    assert dets[0].label == "scaffold"
    assert dets[0].confidence == pytest.approx(0.8)
    assert dets[0].box == (0.1, 0.2, 0.3, 0.4)


def test_to_detections_clamps_boxes_into_the_unit_square():
    dets = to_detections([[-0.05, 0.5, 1.2, 1.01]], [0.5], [0], ("hose",), SITE_PROMPTS)
    assert dets[0].box == (0.0, 0.5, 1.0, 1.0)


def test_to_detections_drops_boxes_with_no_area():
    dets = to_detections([[0.5, 0.5, 0.5, 0.9], [1.1, 0.1, 1.3, 0.2]], [0.9, 0.9], [0, 0],
                         ("ladder",), SITE_PROMPTS)
    assert dets == []


def test_to_detections_rounds_coordinates():
    dets = to_detections([[0.123456, 0.2, 0.3, 0.4]], [0.5], [0], ("ladder",), SITE_PROMPTS)
    assert dets[0].box[0] == 0.1235


@pytest.mark.skipif(
    "source" not in getattr(Detection, "__dataclass_fields__", {}),
    reason="this safesite/types.py Detection has no source field",
)
def test_to_detections_tags_source_when_types_supports_it():
    det = to_detections([[0.1, 0.1, 0.2, 0.2]], [0.5], [0], ("knife",), HOME_PROMPTS)[0]
    assert det.source == "yoloe"


# ---------- detector with a fake YOLOE model ----------
class FakeTensor:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class FakeYOLOE:
    """Records set_classes / get_text_pe calls; predict returns one box for class 0."""

    def __init__(self):
        self.text_pe_calls: list[tuple[str, ...]] = []
        self.set_classes_calls: list[tuple[str, ...]] = []
        self.predict_kwargs: dict = {}

    def get_text_pe(self, names):
        self.text_pe_calls.append(tuple(names))
        return f"pe:{'|'.join(names)}"

    def set_classes(self, names, embeddings=None):
        assert embeddings == f"pe:{'|'.join(names)}"
        self.set_classes_calls.append(tuple(names))

    def predict(self, image, **kwargs):
        self.predict_kwargs = kwargs
        boxes = SimpleNamespace(
            xyxyn=FakeTensor([[0.1, 0.7, 0.9, 0.95]]), conf=FakeTensor([0.6]), cls=FakeTensor([0.0])
        )
        return [SimpleNamespace(boxes=boxes)]


@pytest.fixture
def image():
    return Image.new("RGB", (64, 48))


def test_detector_encodes_each_prompt_set_once_at_start(image):
    fake = FakeYOLOE()
    det = OpenVocabDetector(model=fake)
    for mode in ("site", "home", "site", "home"):
        det(image, mode)
    assert sorted(fake.text_pe_calls) == sorted([tuple(SITE_PROMPTS), tuple(HOME_PROMPTS)])


def test_detector_only_switches_classes_when_mode_changes(image):
    fake = FakeYOLOE()
    det = OpenVocabDetector(model=fake)
    calls_after_init = len(fake.set_classes_calls)
    det(image, "site")
    det(image, "site")
    det(image, "home")
    det(image, "home")
    switches = fake.set_classes_calls[calls_after_init:]
    assert switches == [tuple(HOME_PROMPTS)]  # init leaves SITE active


def test_detector_returns_mode_specific_labels(image):
    det = OpenVocabDetector(model=FakeYOLOE())
    first_site = list(SITE_PROMPTS.values())[0]
    first_home = list(HOME_PROMPTS.values())[0]
    assert [d.label for d in det(image, "site")] == [first_site]
    assert [d.label for d in det(image, "home")] == [first_home]


def test_detector_runs_on_cpu_with_its_threshold(image):
    fake = FakeYOLOE()
    det = OpenVocabDetector(model=fake, conf=0.2, imgsz=480)
    det(image, "site")
    assert fake.predict_kwargs["device"] == "cpu"
    assert fake.predict_kwargs["conf"] == 0.2
    assert fake.predict_kwargs["imgsz"] == 480


def test_default_weights_are_the_smallest_yoloe26():
    assert openvocab.DEFAULT_WEIGHTS == "yoloe-26n-seg.pt"
