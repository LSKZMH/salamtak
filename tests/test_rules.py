"""Tests for rules.py, heat.py and score.py, using fake detections only (no models)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from safesite import heat, rules
from safesite.score import dedupe, safety_score
from safesite.types import Detection, Hazard, Mode, SceneContext, Severity

pytestmark = pytest.mark.unit

# A standing worker: height 0.4, head region (top quarter) is y 0.30-0.40.
PERSON = (0.40, 0.30, 0.50, 0.70)
HEAD = (0.42, 0.29, 0.48, 0.36)
TORSO = (0.41, 0.42, 0.49, 0.55)
INDOOR = SceneContext(outdoor=False)


def det(label: str, box: tuple[float, float, float, float], conf: float = 0.9) -> Detection:
    return Detection(label, conf, box)


def person(box: tuple[float, float, float, float] = PERSON) -> Detection:
    return det("person", box)


def ids(hazards: list[Hazard]) -> list[str]:
    return [h.id for h in hazards]


# --- no_helmet -------------------------------------------------------------------
def test_no_helmet_flags_person_with_bare_head_box():
    hazards = rules.no_helmet([person(), det("no_helmet", HEAD, 0.88)])
    assert len(hazards) == 1
    assert hazards[0].subject == "person:0"
    assert hazards[0].severity is Severity.HIGH
    assert "bare head" in hazards[0].reason
    assert "0.88" in hazards[0].reason


def test_no_helmet_flags_person_with_no_helmet_box_on_head():
    hazards = rules.no_helmet([person()])
    assert ids(hazards) == ["no_helmet"]
    assert "no helmet seen" in hazards[0].reason


def test_no_helmet_flags_helmet_carried_below_the_head():
    held = (0.42, 0.55, 0.48, 0.60)
    assert ids(rules.no_helmet([person(), det("helmet", held)])) == ["no_helmet"]


def test_no_helmet_clear_when_helmet_on_head():
    assert rules.no_helmet([person(), det("helmet", HEAD)]) == []


def test_no_helmet_flags_lone_bare_head_without_person_box():
    hazards = rules.no_helmet([det("no_helmet", (0.10, 0.10, 0.15, 0.16))])
    assert [h.subject for h in hazards] == ["head:0"]


def test_no_helmet_ignores_low_confidence_bare_head():
    scene = [person(), det("helmet", HEAD), det("no_helmet", HEAD, conf=0.20)]
    assert "no_helmet" not in ids(rules.evaluate(scene, INDOOR))


def test_no_helmet_never_drops_a_bare_head_box_off_the_head_region():
    # Bare head overlaps the person's body but nobody's head region (e.g. crouching).
    scene = [person(), det("helmet", HEAD), det("no_helmet", (0.42, 0.45, 0.48, 0.52))]
    assert [h.subject for h in rules.no_helmet(scene)] == ["person:0"]


def test_no_helmet_bare_head_counts_for_the_owner_of_the_head_region():
    neighbour = person((0.30, 0.42, 0.44, 0.90))  # body overlaps person 1's head box
    scene = [person(), neighbour, det("no_helmet", HEAD), det("helmet", (0.32, 0.41, 0.40, 0.48))]
    assert [h.subject for h in rules.no_helmet(scene)] == ["person:0"]


def test_no_helmet_counts_once_per_person_with_two_bare_head_boxes():
    scene = [person(), det("no_helmet", HEAD), det("no_helmet", (0.43, 0.30, 0.47, 0.35))]
    hazards = rules.no_helmet(scene)
    assert len(hazards) == 1
    assert len(hazards[0].boxes) == 3


# --- machinery_zone ----------------------------------------------------------------
def test_machinery_zone_flags_person_overlapping_machine():
    hazards = rules.machinery_zone([person(), det("machinery", (0.45, 0.50, 0.80, 0.90))])
    assert ids(hazards) == ["machinery_zone"]
    assert "overlaps a machine" in hazards[0].reason


def test_machinery_zone_flags_vehicle_within_half_person_height():
    vehicle = det("vehicle", (0.65, 0.40, 0.90, 0.70))  # gap 0.15 <= 0.5 * 0.4
    hazards = rules.machinery_zone([person(), vehicle])
    assert ids(hazards) == ["machinery_zone"]
    assert "0.38x their height from a vehicle" in hazards[0].reason


def test_machinery_zone_clear_beyond_half_person_height():
    far = det("machinery", (0.75, 0.40, 0.95, 0.70))  # gap 0.25 > 0.2
    assert rules.machinery_zone([person(), far]) == []


def test_machinery_zone_uses_aspect_ratio_for_horizontal_gaps():
    near_on_square = det("machinery", (0.65, 0.40, 0.90, 0.70))
    assert rules.machinery_zone([person(), near_on_square], aspect_ratio=16 / 9) == []


def test_machinery_zone_counts_once_per_person_with_two_machines():
    machines = [
        det("machinery", (0.45, 0.50, 0.80, 0.90)),
        det("vehicle", (0.55, 0.30, 0.70, 0.70)),
    ]
    hazards = rules.machinery_zone([person(), *machines])
    assert len(hazards) == 1
    assert len(hazards[0].boxes) == 3


# --- fall_no_harness -------------------------------------------------------------
def test_fall_flags_person_on_ladder_without_harness():
    hazards = rules.fall_no_harness([person(), det("ladder", (0.42, 0.40, 0.52, 0.95))])
    assert ids(hazards) == ["fall_no_harness"]
    assert "on a ladder" in hazards[0].reason


def test_fall_flags_person_near_scaffold_without_harness():
    scaffold = det("scaffold", (0.55, 0.10, 0.90, 0.90))  # gap 0.05 <= 0.25 * 0.4
    assert "near a scaffold" in rules.fall_no_harness([person(), scaffold])[0].reason


def test_fall_clear_when_harness_overlaps_person():
    scene = [person(), det("ladder", (0.42, 0.40, 0.52, 0.95)), det("harness", TORSO)]
    assert rules.fall_no_harness(scene) == []


def test_fall_clear_when_ladder_is_far_away():
    assert rules.fall_no_harness([person(), det("ladder", (0.80, 0.40, 0.90, 0.95))]) == []


# --- no_vest_near_vehicle --------------------------------------------------------
def test_no_vest_flags_person_near_vehicle():
    scene = [person(), det("no_vest", TORSO, 0.77), det("vehicle", (0.55, 0.40, 0.90, 0.70))]
    hazards = rules.no_vest_near_vehicle(scene)
    assert ids(hazards) == ["no_vest_near_vehicle"]
    assert hazards[0].severity is Severity.MEDIUM
    assert "0.77" in hazards[0].reason


def test_no_vest_clear_when_vehicle_is_far():
    scene = [person(), det("no_vest", TORSO), det("vehicle", (0.80, 0.40, 0.95, 0.70))]
    assert rules.no_vest_near_vehicle(scene) == []


def test_no_vest_clear_when_wearing_vest_near_vehicle():
    scene = [person(), det("vest", TORSO), det("vehicle", (0.55, 0.40, 0.90, 0.70))]
    assert rules.no_vest_near_vehicle(scene) == []


def test_no_vest_rule_only_counts_vehicles_not_machinery():
    scene = [person(), det("no_vest", TORSO), det("machinery", (0.55, 0.40, 0.90, 0.70))]
    assert rules.no_vest_near_vehicle(scene) == []


# --- trip_cord -------------------------------------------------------------------
@pytest.mark.parametrize("label", ["cord", "hose", "extension cord"])
def test_trip_cord_flags_cord_or_hose_in_lower_third(label):
    hazards = rules.trip_cord([det(label, (0.10, 0.80, 0.90, 0.86), 0.4)])
    assert ids(hazards) == ["trip_cord"]
    assert hazards[0].subject is None


def test_trip_cord_clear_when_cord_is_overhead():
    assert rules.trip_cord([det("cord", (0.10, 0.10, 0.90, 0.15))]) == []


# --- midday_ban (Qatar 10:00-15:30, 1 Jun-15 Sep) ----------------------------------
@pytest.mark.parametrize(
    ("moment", "banned"),
    [
        (datetime(2026, 7, 15, 9, 59, 59), False),
        (datetime(2026, 7, 15, 10, 0), True),
        (datetime(2026, 7, 15, 15, 29, 59), True),
        (datetime(2026, 7, 15, 15, 30), False),
        (datetime(2026, 5, 31, 12, 0), False),
        (datetime(2026, 6, 1, 10, 0), True),
        (datetime(2026, 6, 1, 12, 0), True),
        (datetime(2026, 9, 15, 15, 29), True),
        (datetime(2026, 9, 16, 12, 0), False),
    ],
    ids=[
        "09:59:59-allowed",
        "10:00-banned",
        "15:29:59-banned",
        "15:30-allowed",
        "31-May-allowed",
        "1-Jun-10:00-banned",
        "1-Jun-noon-banned",
        "15-Sep-15:29-banned",
        "16-Sep-allowed",
    ],
)
def test_midday_ban_boundaries(moment, banned):
    assert (heat.midday_ban(moment, outdoor=True) is not None) is banned


def test_midday_ban_reason_names_time_and_window():
    hazard = heat.midday_ban(datetime(2026, 9, 15, 12, 5), outdoor=True)
    assert hazard is not None
    assert hazard.severity is Severity.HIGH
    assert "15 Sep 12:05" in hazard.reason
    assert "10:00-15:30" in hazard.reason


def test_midday_ban_off_when_indoor_toggle_off():
    assert heat.midday_ban(datetime(2026, 7, 15, 12, 0), outdoor=False) is None


def test_midday_ban_off_without_a_time():
    assert heat.midday_ban(None, outdoor=True) is None


def test_midday_ban_converts_aware_times_to_doha():
    assert heat.midday_ban(datetime(2026, 7, 15, 7, 0, tzinfo=UTC), True) is not None
    assert heat.midday_ban(datetime(2026, 7, 15, 6, 59, tzinfo=UTC), True) is None


# --- wbgt_stop (> 32.1 C with workers) ---------------------------------------------
def test_wbgt_exactly_32_1_is_not_a_stop():
    assert heat.wbgt_stop(32.1, worker_count=1) is None


def test_wbgt_32_2_is_a_stop():
    hazard = heat.wbgt_stop(32.2, worker_count=2)
    assert hazard is not None
    assert "32.2" in hazard.reason
    assert "2 workers" in hazard.reason


def test_wbgt_stop_needs_workers_present():
    assert heat.wbgt_stop(35.0, worker_count=0) is None


def test_wbgt_stop_not_raised_without_a_reading():
    assert heat.wbgt_stop(None, worker_count=3) is None


def test_wbgt_counts_lone_helmet_head_as_worker():
    scene = [det("helmet", (0.10, 0.10, 0.15, 0.16))]
    assert "wbgt_stop" in ids(rules.evaluate(scene, SceneContext(wbgt_c=33, outdoor=False)))


def test_wbgt_rejects_non_finite_input():
    with pytest.raises(ValueError, match="WBGT must be a finite number"):
        SceneContext(wbgt_c=math.nan)
    with pytest.raises(ValueError, match="WBGT must be a finite number"):
        heat.wbgt_stop(math.nan, worker_count=1)


# --- heat_welfare_gap -------------------------------------------------------------
SUMMER_MORNING = SceneContext(taken_at=datetime(2026, 7, 15, 8, 0))  # in season, before the ban


def test_welfare_gap_flags_outdoor_workers_without_shade_or_water_in_summer():
    hazards = rules.heat_hazards([person()], SUMMER_MORNING)
    assert ids(hazards) == ["heat_welfare_gap"]
    assert "summer heat season" in hazards[0].reason


@pytest.mark.parametrize("label", ["shade tent", "water cooler"])
def test_welfare_gap_clear_when_shade_or_water_seen(label):
    scene = [person(), det(label, (0.70, 0.10, 0.95, 0.40), 0.3)]
    assert rules.heat_hazards(scene, SUMMER_MORNING) == []


def test_welfare_gap_clear_without_workers():
    assert rules.heat_hazards([], SUMMER_MORNING) == []


@pytest.mark.parametrize(
    "moment",
    [datetime(2026, 11, 6, 9, 0), datetime(2026, 5, 31, 8, 0), datetime(2026, 9, 16, 8, 0)],
    ids=["november", "31-May", "16-Sep"],
)
def test_welfare_gap_not_raised_outside_summer(moment):
    assert rules.heat_hazards([person()], SceneContext(taken_at=moment)) == []


def test_welfare_gap_not_judged_without_a_time():
    assert rules.heat_hazards([person()], SceneContext()) == []


# --- indoor toggle off: only wbgt_stop survives --------------------------------------
def test_indoor_toggle_disables_midday_ban_and_welfare_but_not_wbgt():
    ctx = SceneContext(taken_at=datetime(2026, 7, 15, 12, 0), wbgt_c=32.2, outdoor=False)
    found = ids(rules.heat_hazards([person()], ctx))
    assert found == ["wbgt_stop"]


def test_outdoor_summer_noon_raises_all_heat_hazards():
    ctx = SceneContext(taken_at=datetime(2026, 7, 15, 12, 0), wbgt_c=32.2)
    assert ids(rules.heat_hazards([person()], ctx)) == [
        "midday_ban",
        "wbgt_stop",
        "heat_welfare_gap",
    ]


# --- home mode ---------------------------------------------------------------------
HOME = SceneContext(mode=Mode.HOME)


@pytest.mark.parametrize("label", ["cord", "extension cord"])
def test_home_trip_flags_cord_on_floor_and_not_site_trip_cord(label):
    assert ids(rules.evaluate([det(label, (0.20, 0.85, 0.70, 0.90))], HOME)) == ["home_trip"]


def test_home_trip_ignores_hoses():
    assert rules.evaluate([det("hose", (0.20, 0.85, 0.70, 0.90))], HOME) == []


def test_home_trip_clear_when_cord_is_high():
    assert rules.evaluate([det("cord", (0.20, 0.20, 0.70, 0.25))], HOME) == []


def test_home_slip_flags_wet_floor():
    hazards = rules.evaluate([det("wet floor", (0.0, 0.70, 1.0, 1.0), 0.71)], HOME)
    assert ids(hazards) == ["home_slip"]
    assert "0.71" in hazards[0].reason


def test_home_slip_clear_without_wet_floor():
    assert rules.home_slip([det("cord", (0.20, 0.20, 0.70, 0.25))]) == []


@pytest.mark.parametrize("label", ["knife", "chemicals", "cleaning chemicals"])
def test_home_child_reach_flags_low_items(label):
    assert ids(rules.evaluate([det(label, (0.40, 0.70, 0.50, 0.80))], HOME)) == ["home_child_reach"]


def test_home_child_reach_clear_when_item_is_high():
    assert rules.evaluate([det("knife", (0.40, 0.10, 0.50, 0.20))], HOME) == []


# --- reason wording for several objects --------------------------------------------
def _floor(label: str, x: float) -> Detection:
    return det(label, (x, 0.80, x + 0.1, 0.90))


def test_reason_counts_several_cords_by_name():
    hazards = rules.home_trip([_floor("cord", 0.05 + 0.2 * i) for i in range(4)])
    assert hazards[0].reason == "Trip risk: 4 cords on the floor (lower third of the frame)."


def test_reason_lists_mixed_objects_with_articles():
    hazards = rules.trip_cord([_floor("cord", 0.05), _floor("cord", 0.3), _floor("hose", 0.55)])
    assert "2 cords and a hose in the walkway" in hazards[0].reason


def test_reason_uses_an_and_irregular_plurals():
    reach = rules.home_child_reach(
        [_floor("knife", 0.1), _floor("knife", 0.4), _floor("chemicals", 0.7)]
    )
    assert "chemicals and 2 knives" in reach[0].reason
    assert "an extension cord" in rules.home_trip([_floor("extension cord", 0.1)])[0].reason


def test_home_mode_skips_site_and_heat_rules():
    ctx = SceneContext(taken_at=datetime(2026, 7, 15, 12, 0), wbgt_c=40, mode=Mode.HOME)
    assert rules.evaluate([person(), det("no_helmet", HEAD)], ctx) == []


def test_site_mode_skips_home_rules():
    assert "home_slip" not in ids(rules.evaluate([det("wet floor", (0.0, 0.7, 1.0, 1.0))]))


# --- evaluate ---------------------------------------------------------------------
def test_evaluate_lists_high_before_medium():
    scene = [person(), det("cord", (0.10, 0.80, 0.90, 0.86))]
    severities = [h.severity for h in rules.evaluate(scene, SceneContext())]
    assert severities == sorted(severities, key=lambda s: s is not Severity.HIGH)
    assert severities[0] is Severity.HIGH


# --- score -------------------------------------------------------------------------
def _hazard(hazard_id: str, subject: str | None = None) -> Hazard:
    return Hazard.of(hazard_id, "test reason", subject=subject)


def test_score_is_100_with_no_hazards():
    assert safety_score([]) == 100


def test_score_subtracts_25_per_high_and_10_per_medium():
    assert safety_score([_hazard("no_helmet", "person:0"), _hazard("trip_cord")]) == 65


def test_score_floors_at_zero():
    hazards = [_hazard("no_helmet", f"person:{i}") for i in range(5)]  # 100 - 125
    assert safety_score(hazards) == 0


def test_score_counts_each_hazard_once_per_person():
    same_person = [_hazard("no_helmet", "person:0"), _hazard("no_helmet", "person:0")]
    assert safety_score(same_person) == 75
    assert len(dedupe(same_person)) == 1


def test_score_counts_same_hazard_on_different_people():
    assert safety_score([_hazard("no_helmet", "person:0"), _hazard("no_helmet", "person:1")]) == 50


def test_score_counts_different_hazards_on_same_person():
    assert (
        safety_score([_hazard("no_helmet", "person:0"), _hazard("machinery_zone", "person:0")])
        == 50
    )


def test_score_counts_scene_hazard_once():
    assert safety_score([_hazard("trip_cord"), _hazard("trip_cord")]) == 90


def test_score_of_crowded_unsafe_scene_floors_at_zero():
    people = [person((0.05 + 0.18 * i, 0.30, 0.15 + 0.18 * i, 0.70)) for i in range(5)]
    assert safety_score(rules.evaluate(people, INDOOR)) == 0


# --- types --------------------------------------------------------------------------
def test_detection_normalizes_label():
    assert det("Shade-Tent", (0.1, 0.1, 0.2, 0.2)).label == "shade_tent"


@pytest.mark.parametrize(
    ("box", "message"),
    [
        ((0.5, 0.1, 0.4, 0.2), "box must be normalized"),
        ((0.1, 0.1, 1.2, 0.2), "box must be normalized"),
        ((0.1, 0.1, 0.2), "box must have 4 values"),
    ],
    ids=["x1-after-x2", "outside-image", "three-values"],
)
def test_detection_rejects_unnormalized_boxes(box, message):
    with pytest.raises(ValueError, match=message):
        det("person", box)


def test_from_model_clamps_overshoot():
    clamped = Detection.from_model("person", 1.0000001, (-0.01, 0.5, 1.0000001, 0.6))
    assert clamped is not None
    assert clamped.box == (0.0, 0.5, 1.0, 0.6)
    assert clamped.confidence == 1.0


def test_from_model_drops_zero_area_boxes():
    assert Detection.from_model("person", 0.9, (0.5, 0.5, 0.5, 0.6)) is None


def test_detection_rejects_confidence_above_one():
    with pytest.raises(ValueError, match="confidence must be in"):
        det("person", PERSON, conf=1.5)


def test_hazard_rejects_wrong_severity():
    with pytest.raises(ValueError, match="no_helmet must be High"):
        Hazard("no_helmet", Severity.MEDIUM, "x")


def test_hazard_rejects_multiline_reason():
    with pytest.raises(ValueError, match="one non-empty line"):
        Hazard.of("trip_cord", "line one\nline two")


def test_severity_prints_as_its_label():
    assert str(Severity.HIGH) == "High"


# --- v2 contract ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("hazard_id", "severity"),
    [
        ("worker_down", Severity.HIGH),
        ("under_suspended_load", Severity.HIGH),
        ("power_line_contact", Severity.HIGH),
        ("hot_work_fire", Severity.HIGH),
        ("fall_into_opening", Severity.HIGH),
        ("no_fire_extinguisher", Severity.MEDIUM),
        ("exposed_rebar", Severity.MEDIUM),
    ],
)
def test_v2_hazard_ids_have_their_severity(hazard_id, severity):
    assert Hazard.of(hazard_id, "test reason").severity is severity


def test_v2_class_contract_extends_v1_without_reordering():
    from safesite.types import EXTENDED_CLASSES, MODEL_CLASSES_V2, UNIFIED_CLASSES

    assert UNIFIED_CLASSES == (
        "person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle"
    )
    assert EXTENDED_CLASSES == ("harness", "scaffold", "crane_hook", "utility_pole")
    assert MODEL_CLASSES_V2 == UNIFIED_CLASSES + EXTENDED_CLASSES
    assert MODEL_CLASSES_V2.index("utility_pole") == 10


# --- v2 labels and confidence cut-offs ---------------------------------------------------
def src(label: str, conf: float, source: str, box=(0.1, 0.1, 0.2, 0.2)) -> Detection:
    return Detection(label, conf, box, source)


V2_OPEN_VOCAB = (
    "power_line", "welding", "fire", "gas_cylinder", "fire_extinguisher", "rebar",
    "floor_opening", "trench", "crane_hook", "utility_pole",
)


@pytest.mark.parametrize("label", V2_OPEN_VOCAB)
def test_v2_open_vocab_labels_are_registered(label):
    assert label in rules.OPEN_VOCAB_LABELS


@pytest.mark.parametrize(
    ("source", "conf", "kept"),
    [
        ("yoloe", 0.30, True),  # open-vocab cut-off 0.25
        ("yoloe", 0.20, False),
        ("detector", 0.30, False),  # trained class: MIN_CONFIDENCE 0.35
        ("detector", 0.36, True),
        ("mock", 0.30, True),  # mock: fall back by label (harness is open-vocab)
    ],
)
def test_harness_cutoff_follows_detection_source(source, conf, kept):
    assert bool(rules.confident([src("harness", conf, source)])) is kept


def test_open_vocab_only_label_from_default_source_keeps_open_vocab_cutoff():
    # A "cord" can never come from the trained detector, so the label decides.
    assert rules.confident([src("cord", 0.30, "detector")])


def test_open_vocab_cutoffs_override_per_label(monkeypatch):
    monkeypatch.setattr(rules, "OPEN_VOCAB_CUTOFFS", {"fire": 0.5})
    assert not rules.confident([src("fire", 0.45, "yoloe")])
    assert rules.confident([src("fire", 0.55, "yoloe")])
    assert rules.confident([src("rebar", 0.30, "yoloe")])  # unlisted: default 0.25


def test_open_vocab_cutoffs_are_never_used_for_trained_detector_output(monkeypatch):
    monkeypatch.setattr(rules, "OPEN_VOCAB_CUTOFFS", {"scaffold": 0.9})
    assert rules.confident([src("scaffold", 0.40, "detector")])


@pytest.mark.parametrize(
    ("label", "name"),
    [
        ("crane_hook", "crane hook"),
        ("utility_pole", "power pole"),
        ("power_line", "power line"),
        ("gas_cylinder", "gas cylinder"),
        ("fire_extinguisher", "fire extinguisher"),
        ("floor_opening", "floor opening"),
    ],
)
def test_v2_display_names(label, name):
    assert rules.DISPLAY_NAME[label] == name


# --- worker_down -------------------------------------------------------------------------
LYING = (0.30, 0.70, 0.60, 0.85)  # 0.30 wide x 0.15 tall: ratio 2.0


def test_worker_down_flags_person_lying_on_the_ground():
    hazards = rules.worker_down([person(LYING)])
    assert ids(hazards) == ["worker_down"]
    assert hazards[0].severity is Severity.HIGH
    assert hazards[0].subject == "person:0"
    assert "lying on the ground in the middle" in hazards[0].reason
    assert "possible heat illness or fall" in hazards[0].reason


@pytest.mark.parametrize(
    "box",
    [PERSON, (0.40, 0.50, 0.49, 0.60)],  # standing; crouching (ratio 0.9)
    ids=["standing", "crouching"],
)
def test_worker_down_clear_for_upright_people(box):
    assert rules.worker_down([person(box)]) == []


@pytest.mark.parametrize(
    ("box", "down"),
    [((0.40, 0.60, 0.539, 0.70), False), ((0.40, 0.60, 0.541, 0.70), True)],
    ids=["ratio-1.39", "ratio-1.41"],
)
def test_worker_down_ratio_threshold(box, down):
    assert bool(rules.worker_down([person(box)])) is down


@pytest.mark.parametrize(
    ("box", "down"),
    [((0.40, 0.60, 0.50, 0.639), False), ((0.40, 0.60, 0.50, 0.641), True)],
    ids=["height-0.039", "height-0.041"],
)
def test_worker_down_min_height(box, down):
    assert bool(rules.worker_down([person(box)])) is down


@pytest.mark.parametrize(("conf", "down"), [(0.49, False), (0.50, True)])
def test_worker_down_needs_a_confident_person(conf, down):
    assert bool(rules.worker_down([det("person", LYING, conf)])) is down


def test_worker_down_uses_real_pixels_on_wide_and_tall_images():
    square = (0.40, 0.60, 0.50, 0.70)  # 1.0 on a square image, 1.78 at 16:9
    assert rules.worker_down([person(square)]) == []
    assert ids(rules.worker_down([person(square)], aspect_ratio=16 / 9)) == ["worker_down"]
    assert rules.worker_down([person(LYING)], aspect_ratio=9 / 16) == []  # 2.0 -> 1.125


@pytest.mark.parametrize(
    "box",
    [(0.0, 0.70, 0.30, 0.85), (0.005, 0.70, 0.30, 0.85), (0.30, 0.85, 0.60, 1.0),
     (0.70, 0.70, 0.995, 0.85)],
    ids=["left-edge", "within-margin", "bottom-edge", "right-margin"],
)
def test_worker_down_ignores_boxes_cut_off_by_the_border(box):
    assert rules.worker_down([person(box)]) == []


def test_worker_down_flags_box_just_inside_the_border_margin():
    assert ids(rules.worker_down([person((0.02, 0.70, 0.32, 0.85))])) == ["worker_down"]


@pytest.mark.parametrize(
    ("vehicle", "down"),
    [
        ((0.20, 0.60, 0.80, 0.95), False),  # fully inside: a driver
        ((0.10, 0.50, 0.51, 0.95), False),  # 70% inside
        ((0.10, 0.50, 0.42, 0.95), True),  # 40% inside
    ],
    ids=["inside", "70pct", "40pct"],
)
def test_worker_down_ignores_people_inside_a_vehicle(vehicle, down):
    assert bool(rules.worker_down([person(LYING), det("vehicle", vehicle)])) is down


def test_worker_down_ignores_people_inside_machinery():
    assert rules.worker_down([person(LYING), det("machinery", (0.20, 0.60, 0.80, 0.95))]) == []


def test_worker_down_once_per_person_and_names_the_side():
    left, right = (0.05, 0.70, 0.25, 0.80), (0.70, 0.70, 0.95, 0.80)
    hazards = rules.worker_down([person(left), person(PERSON), person(right)])
    assert [h.subject for h in hazards] == ["person:0", "person:2"]
    assert "on the left" in hazards[0].reason
    assert "on the right" in hazards[1].reason


def test_worker_down_runs_in_site_mode_only():
    assert "worker_down" in ids(rules.evaluate([person(LYING)], INDOOR))
    assert rules.evaluate([person(LYING)], SceneContext(mode=Mode.HOME)) == []
