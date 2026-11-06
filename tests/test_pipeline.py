"""Shared pipeline: the 'never silently safe' guarantees, downscaling and the size guard."""
from __future__ import annotations

from datetime import datetime

import pytest
from PIL import Image

from safesite import detect
from safesite.pipeline import (
    HAZARD_INPUTS,
    MAX_SIDE,
    ImageTooLargeError,
    analyze_image,
    prepare_image,
    unchecked_hazards,
)
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


@pytest.mark.parametrize("label", ["crane_hook", "utility_pole"])
def test_v2_equipment_alone_makes_a_site_photo_assessable(label):
    # A photo with only a crane hook or a utility pole is a real site photo, not "can't assess".
    _, result = analyze_image(FixedDetector([Detection(label, 0.9, (0.4, 0.1, 0.6, 0.5))]), _img(), WINTER)
    assert result.assessed is True
    assert not any("can't be assessed" in n for n in result.notes)


PERSON = [Detection("person", 0.9, (0.1, 0.1, 0.3, 0.9)), Detection("helmet", 0.9, (0.12, 0.09, 0.28, 0.25))]


def test_unchecked_hazards_lists_hazards_whose_inputs_no_model_produces():
    assert unchecked_hazards({"person", "crane_hook", "fire"}) == (
        "power_line_contact", "no_fire_extinguisher", "exposed_rebar", "fall_into_opening", "fall_no_harness")


def test_power_line_contact_needs_machinery_and_fall_rule_needs_a_harness_detector():
    assert "power_line_contact" in unchecked_hazards({"utility_pole"})  # nothing to touch the pole
    assert "power_line_contact" not in unchecked_hazards({"utility_pole", "machinery"})
    assert "fall_no_harness" in unchecked_hazards({"scaffold"})  # "no harness seen" means nothing
    assert "fall_no_harness" not in unchecked_hazards({"scaffold", "harness"})


# One layout where every object sits on/next to the worker, so any rule whose labels are present fires.
_LAYOUT = {
    "person": (0.40, 0.30, 0.60, 0.90), "machinery": (0.55, 0.40, 0.95, 0.90),
    "crane_hook": (0.45, 0.10, 0.55, 0.28), "utility_pole": (0.62, 0.00, 0.70, 0.90),
    "power_line": (0.00, 0.30, 1.00, 0.45), "fire": (0.30, 0.50, 0.50, 0.80),
    "welding": (0.35, 0.50, 0.55, 0.80), "gas_cylinder": (0.50, 0.55, 0.58, 0.85),
    "rebar": (0.20, 0.70, 0.80, 0.95), "trench": (0.30, 0.85, 0.80, 1.00),
    "floor_opening": (0.30, 0.85, 0.80, 1.00), "scaffold": (0.30, 0.10, 0.70, 0.95),
    "ladder": (0.35, 0.30, 0.65, 0.95),
}
_ABSENCE_LABELS = {"fire_extinguisher", "harness"}  # the rule fires when these are NOT seen


@pytest.mark.parametrize(
    ("hazard", "option"),
    [(h, o) for h, options in HAZARD_INPUTS.items() for o in options],
    ids=lambda v: v if isinstance(v, str) else "+".join(sorted(v)),
)
def test_hazard_inputs_match_what_rules_py_needs_to_fire(hazard, option):
    from safesite import rules

    trained = {"person", "machinery"}
    present = {"person"} | (option - _ABSENCE_LABELS)
    dets = [Detection(lab, 0.95, _LAYOUT[lab], "detector" if lab in trained else "yoloe") for lab in present]
    assert hazard in {h.id for h in rules.evaluate(dets, WINTER)}
    every_input = set().union(*(s for options in HAZARD_INPUTS.values() for s in options))
    assert unchecked_hazards(every_input) == ()


def test_hot_work_needs_fire_or_welding_with_cylinder_and_extinguisher_check_needs_extinguisher():
    assert "hot_work_fire" in unchecked_hazards({"welding"})  # welding alone can't trigger it
    assert "hot_work_fire" not in unchecked_hazards({"welding", "gas_cylinder"})
    assert "no_fire_extinguisher" in unchecked_hazards({"fire"})  # can't judge "no extinguisher"
    assert "no_fire_extinguisher" not in unchecked_hazards({"welding", "fire_extinguisher"})


def test_note_names_hazards_the_detector_cannot_see():
    detector = FixedDetector(PERSON)
    detector.labels = frozenset({"person", "helmet", "rebar", "scaffold"})
    _, result = analyze_image(detector, _img(), WINTER)
    note = next(n for n in result.notes if n.startswith("Not checked with the current models"))
    assert "power line contact" in note and "exposed rebar" not in note


def test_no_unchecked_note_when_detector_does_not_declare_labels_or_in_home_mode():
    _, site = analyze_image(FixedDetector(PERSON), _img(), WINTER)
    home_detector = FixedDetector([])
    home_detector.labels, home_detector.supports_home = frozenset({"cord"}), True
    _, home = analyze_image(home_detector, _img(), SceneContext(taken_at=WINTER.taken_at, mode="home"))
    assert not any("Not checked" in n for n in (*site.notes, *home.notes))


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
