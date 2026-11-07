"""Fast, network-free tests for the v2 dataset builder and the pseudo-labeller."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_data_v2 as p2  # noqa: E402
import pseudo_label as pl  # noqa: E402

# Source class names as published by each source.
LISTED = {
    "chd": ["Person", "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest", "machinery", "vehicle",
            "utility pole", "Mask", "NO-Mask", "Safety Cone"],
    "ppe": ["helmet", "gloves", "vest", "boots", "goggles", "Person", "no_helmet", "no_goggle",
            "no_gloves", "no_boots", "forklift", "no_vest"],
    "mocs": ["Worker", "Excavator", "Bulldozer", "Loader", "Roller", "Crane", "Static crane",
             "Concrete mixer", "Pump truck", "Pile driving", "Truck", "Other vehicle", "Hanging head"],
    "hrn": ["head", "helmet", "safety-harness"],
    "scf": ["scaffold", "diagonal-brace", "ledger", "standard"],
}


def test_class_contract():
    assert p2.CLASSES == pl.CLASSES
    assert p2.CLASSES[:7] == ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle"]
    assert p2.CLASSES[7:] == ["harness", "scaffold", "crane_hook", "utility_pole"]


@pytest.mark.parametrize("alias", sorted(LISTED))
def test_mapping_covers_every_listed_class(alias):
    cmap = p2.class_map_for(alias, LISTED[alias])
    assert set(cmap) == set(range(len(LISTED[alias])))
    assert all(v is None or 0 <= v < len(p2.CLASSES) for v in cmap.values())


def test_mapping_targets_case_insensitive():
    cmap = p2.class_map_for("chd", {0: "  utility POLE ", 1: "Safety Cone", 2: "NO-Safety Vest"})
    assert cmap == {0: p2.CLS_IDX["utility_pole"], 1: None, 2: p2.CLS_IDX["no_vest"]}
    assert p2.class_map_for("mocs", ["Hanging head"]) == {0: p2.CLS_IDX["crane_hook"]}


def test_v1_sources_keep_v1_mapping():
    cmap = p2.class_map_for("hh", ["hardhat", "no-hardhat"])
    assert cmap == {0: 1, 1: 2}


def test_hrn_head_is_no_helmet():
    assert p2.class_map_for("hrn", ["head", "helmet", "safety-harness"]) == {
        0: p2.CLS_IDX["no_helmet"], 1: p2.CLS_IDX["helmet"], 2: p2.CLS_IDX["harness"]}


def test_unknown_class_exits():
    with pytest.raises(SystemExit):
        p2.class_map_for("scf", ["scaffold", "plank"])


def test_teacher_map_matches_chd_mapping():
    chd = {p2.norm(k): v for k, v in p2.V2_MAPPING["chd"].items()}
    assert dict(pl.YIHONG_MAP) == chd


@pytest.mark.parametrize("name,alias", [("chd_valid_000012.jpg", "chd"), ("hh_train_000000.txt", "hh"),
                                        ("mocs_test_001234.jpg", "mocs"), ("C:/x/scf_train_000001.jpg", "scf")])
def test_alias_prefix(name, alias):
    assert p2.alias_of(name) == alias
    assert pl.alias_of(name) == alias


def test_alias_prefix_unknown():
    with pytest.raises(ValueError):
        p2.alias_of("source-0123abc.jpg")


def test_split_routing():
    assert [p2.route_split(s) for s in ("train", "valid", "Validation", "val", "test", "valid-mini")] == \
        ["train", "val", "val", "val", "train", None]


def test_carve_val_is_seeded_ten_percent():
    keys = [f"train_{i:06d}" for i in range(200)]
    a, b = p2.carve_val(keys), p2.carve_val(list(reversed(keys)))
    assert len(a) == 20 and a == b and a <= set(keys)


def test_remap_lines_drops_and_converts_polygons():
    cmap = {0: 8, 1: None}
    out = p2.remap_lines(["0 0.5 0.5 0.2 0.4", "1 0.1 0.1 0.1 0.1", "0 0.1 0.1 0.3 0.1 0.3 0.5", ""], cmap)
    assert out[0] == "8 0.500000 0.500000 0.200000 0.400000"
    assert out[1] == "8 0.200000 0.300000 0.200000 0.400000"
    assert len(out) == 2


def test_cap_keeps_priority_class_then_rare():
    hook, exc, truck = p2.CLS_IDX["crane_hook"], 5, 6
    items = [(f"h{i}", frozenset({hook})) for i in range(3)]
    items += [(f"e{i}", frozenset({exc})) for i in range(10)]
    items += [(f"t{i}", frozenset({truck})) for i in range(2)]
    keep = p2.cap_items(items, 6, hook)
    assert len(keep) == 6
    assert {"h0", "h1", "h2", "t0", "t1"} <= set(keep)


def test_dedupe_drops_train_matching_val_and_keeps_richer_source():
    train = {"hh_train_000001": 0b1111, "cs_train_000001": 0b1110, "fr_train_000002": 0xFFFF_0000,
             "hh_train_000009": 0xF0F0_F0F0_0000}
    val = {"hh_valid_000001": 0xFFFF_0001}
    prio = lambda k: len(p2.LABELLED[p2.alias_of(k)])  # noqa: E731
    vs_val, dupes = p2.dedupe(train, val, prio)
    assert vs_val == {"fr_train_000002"}
    assert dupes == {"hh_train_000001"}  # cs labels more classes than hh, so cs survives


# ----------------------------- pseudo-labeller ----------------------------- #

def test_iou():
    a = (0, 0.5, 0.5, 0.2, 0.2)
    assert pl.iou(a, a) == pytest.approx(1.0)
    assert pl.iou(a, (0, 0.9, 0.9, 0.1, 0.1)) == 0.0


def test_missing_classes():
    producible = {0, 1, 2, 3, 4, 5, 6, 10}
    assert pl.missing_classes(p2.LABELLED["hh"], producible) == {0, 3, 4, 5, 6, 10}
    assert pl.missing_classes(p2.LABELLED["chd"], producible) == set()


def test_conflict_rule_skips_opposite_of_ground_truth():
    helmet, no_helmet, person = 1, 2, 0
    gt = [f"{helmet} 0.5 0.2 0.1 0.1"]
    teacher = [((no_helmet, 0.5, 0.2, 0.1, 0.1), 0.9),   # overlaps GT helmet -> skip
               ((no_helmet, 0.8, 0.2, 0.1, 0.1), 0.9),   # elsewhere -> keep
               ((person, 0.5, 0.5, 0.2, 0.6), 0.4),      # below conf -> skip
               ((person, 0.2, 0.5, 0.2, 0.6), 0.7)]      # keep
    lines, added = pl.merge_pseudo(gt, teacher, {person, no_helmet}, 0.5)
    assert lines[0] == gt[0]
    assert added == Counter({no_helmet: 1, person: 1})
    assert len(lines) == 3


def test_conflict_rule_between_teacher_boxes_keeps_most_confident():
    vest, no_vest = 3, 4
    teacher = [((vest, 0.5, 0.5, 0.2, 0.3), 0.6), ((no_vest, 0.5, 0.5, 0.2, 0.3), 0.8)]
    _, added = pl.merge_pseudo([], teacher, {vest, no_vest}, 0.5)
    assert added == Counter({no_vest: 1})


def test_only_missing_classes_added_and_gt_untouched():
    gt = ["1 0.5 0.2 0.1 0.1", "2 0.7 0.2 0.1 0.1"]
    teacher = [((1, 0.1, 0.1, 0.1, 0.1), 0.99), ((5, 0.5, 0.7, 0.4, 0.4), 0.9)]
    lines, added = pl.merge_pseudo(gt, teacher, {5}, 0.5)
    assert lines[:2] == gt and added == Counter({5: 1})


def test_val_labels_never_modified():
    gt = ["1 0.5 0.2 0.1 0.1"]
    teacher = [((0, 0.5, 0.5, 0.3, 0.6), 0.99)]
    lines, added = pl.update_labels("val", gt, teacher, {0}, 0.5)
    assert lines == gt and not added
    lines, added = pl.update_labels("train", gt, teacher, {0}, 0.5)
    assert len(lines) == 2 and added == Counter({0: 1})


def test_pick_teacher_file_largest_pt():
    files = {"models/yolo26/pt/yolo26x.pt": 118, "models/yolo26/pt/yolo26n.pt": 5,
             "models/yolo11/pt/yolo11x.pt": 114, "models/yolo26/onnx/yolo26x.onnx": 999}
    assert pl.pick_teacher_file(files) == "models/yolo26/pt/yolo26x.pt"


def test_teacher_class_map_requires_names():
    names = {0: "Hardhat", 1: "Mask", 2: "NO-Hardhat", 3: "NO-Mask", 4: "NO-Safety Vest", 5: "Person",
             6: "Safety Cone", 7: "Safety Vest", 8: "machinery", 9: "utility pole", 10: "vehicle"}
    cmap = pl.teacher_class_map(names)
    assert cmap[9] == p2.CLS_IDX["utility_pole"] and 1 not in cmap
    with pytest.raises(SystemExit):
        pl.teacher_class_map({0: "Person"})
