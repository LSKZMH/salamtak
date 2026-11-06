"""Tests for the pure helpers in scripts/train.py (no GPU, no network, no Ultralytics)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train", SCRIPT)
tr = importlib.util.module_from_spec(_spec)
sys.modules["train"] = tr
_spec.loader.exec_module(tr)

V2_NAMES = ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle",
            "harness", "scaffold", "crane_hook", "utility_pole"]

SOURCES = {
    "classes": V2_NAMES,
    "sources": {
        "hh": {"labelled_classes": ["helmet", "no_helmet"]},
        "cs": {"labelled_classes": ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle"]},
        "fr": {"labelled_classes": ["person", "helmet", "no_helmet", "vest", "no_vest"]},
        "chd": {"labelled_classes": ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle",
                                     "utility_pole"]},
        "hrn": {"labelled_classes": ["harness", "helmet"]},
    },
}


# ---------- class names ----------

def test_parse_names_accepts_list_and_dict_in_id_order():
    assert tr.parse_names(["a", "b"]) == ["a", "b"]
    assert tr.parse_names({1: "b", 0: "a"}) == ["a", "b"]
    assert tr.parse_names({"1": "b", "0": "a"}) == ["a", "b"]


def test_load_class_names_reads_data_yaml(tmp_path):
    (tmp_path / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(V2_NAMES)), encoding="utf-8")
    assert tr.load_class_names([tmp_path]) == V2_NAMES


def test_load_class_names_falls_back_to_v1_without_yaml(tmp_path):
    assert tr.load_class_names([tmp_path]) == tr.CLASS_NAMES
    assert len(tr.CLASS_NAMES) == 7


def test_load_class_names_ignores_yaml_without_names(tmp_path):
    (tmp_path / "data.yaml").write_text("train: x\n", encoding="utf-8")
    assert tr.load_class_names([tmp_path]) == tr.CLASS_NAMES


def test_check_class_contract_accepts_v1_and_v2():
    tr.check_class_contract(tr.CLASS_NAMES)
    tr.check_class_contract(V2_NAMES)


def test_check_class_contract_rejects_reordered_ids():
    with pytest.raises(SystemExit):
        tr.check_class_contract(["helmet", "person", *V2_NAMES[2:]])


def test_find_sources_json(tmp_path):
    nested = tmp_path / "a"
    nested.mkdir()
    assert tr.find_sources_json([tmp_path, nested]) is None
    (nested / "sources.json").write_text(json.dumps(SOURCES), encoding="utf-8")
    assert tr.find_sources_json([tmp_path, nested]) == nested / "sources.json"


def test_download_patterns_include_metadata():
    for pattern in ("*.zip", "*.yaml", "*.json", "README.md"):
        assert pattern in tr.DOWNLOAD_PATTERNS


# ---------- alias filtering / list files ----------

def test_images_for_alias_matches_prefix_only():
    paths = [Path(p) for p in ("hh_valid_000001.jpg", "hhx_valid_000001.jpg", "chd_val_000002.jpg",
                               "cs_valid_000003.jpg", "c_val_000004.jpg")]
    assert [p.name for p in tr.images_for_alias(paths, "hh")] == ["hh_valid_000001.jpg"]
    assert [p.name for p in tr.images_for_alias(paths, "c")] == ["c_val_000004.jpg"]
    assert tr.images_for_alias(paths, "mocs") == []


def test_images_for_aliases_pools_several_sources():
    paths = [Path(p) for p in ("hh_1.jpg", "cs_2.jpg", "fr_3.jpg", "chd_4.jpg")]
    assert [p.name for p in tr.images_for_aliases(paths, ["hh", "fr"])] == ["hh_1.jpg", "fr_3.jpg"]


def test_write_list_file_uses_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "lists" / "hh.txt"
    tr.write_list_file([Path("images/val/hh_1.jpg"), Path("images/val/hh_2.jpg")], out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(Path(line).is_absolute() for line in lines)
    assert lines[0].endswith("hh_1.jpg")


def test_write_eval_yaml_points_train_and_val_at_list(tmp_path):
    lst = tmp_path / "x.txt"
    lst.write_text("", encoding="utf-8")
    out = tr.write_eval_yaml(lst, ["a", "b"], tmp_path / "x.yaml")
    import yaml
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["val"] == str(lst.resolve())
    assert data["train"] == str(lst.resolve())
    assert data["names"] == {0: "a", 1: "b"}


# ---------- labelled classes ----------

def test_labelled_classes_single_source_in_class_order():
    assert tr.labelled_classes(SOURCES, ["hrn"], V2_NAMES) == ["helmet", "harness"]


def test_labelled_classes_union_restricted_to_allowed():
    got = tr.labelled_classes(SOURCES, ["hh", "cs", "fr"], V2_NAMES[:7])
    assert got == V2_NAMES[:7]
    assert tr.labelled_classes(SOURCES, ["hh"], V2_NAMES[:7]) == ["helmet", "no_helmet"]


def test_labelled_classes_unknown_alias_is_empty():
    assert tr.labelled_classes(SOURCES, ["nope"], V2_NAMES) == []


def test_pick_ap_keeps_only_requested_classes():
    ap = {"person": 0.5, "helmet": 0.9, "harness": 0.1}
    assert tr.pick_ap(ap, ["helmet", "harness", "scaffold"]) == {"helmet": 0.9, "harness": 0.1}


def test_pooled_views_match_plan():
    assert tr.POOLED_VIEWS["v1_val"]["aliases"] == ("hh", "cs", "fr")
    assert tr.POOLED_VIEWS["fully_labelled"]["aliases"] == ("cs", "fr", "chd", "ppe")
    assert set(tr.POOLED_VIEWS["fully_labelled"]["classes"]) == {
        "person", "vest", "no_vest", "helmet", "no_helmet", "machinery", "vehicle"}


def test_view_classes_v1_val_is_union_of_labelled_v1_classes():
    view = tr.POOLED_VIEWS["v1_val"]
    assert tr.view_classes(view, SOURCES, V2_NAMES) == V2_NAMES[:7]


def test_view_classes_fixed_list_kept_in_class_order():
    view = tr.POOLED_VIEWS["fully_labelled"]
    assert tr.view_classes(view, SOURCES, V2_NAMES) == V2_NAMES[:7]


# ---------- label filtering for the 7-class v1 comparison ----------

def test_filter_label_text_drops_ids_outside_v1():
    text = "0 0.5 0.5 0.1 0.1\n10 0.2 0.2 0.1 0.1\n6 0.3 0.3 0.1 0.1\n\n7 0.1 0.1 0.1 0.1\n"
    assert tr.filter_label_text(text, 7) == "0 0.5 0.5 0.1 0.1\n6 0.3 0.3 0.1 0.1\n"


def test_filter_label_text_empty():
    assert tr.filter_label_text("", 7) == ""
    assert tr.filter_label_text("9 0.1 0.1 0.1 0.1\n", 7) == ""


def test_label_path_for_image():
    img = Path("/d/images/val/cs_valid_000001.jpg")
    assert tr.label_path_for(img) == Path("/d/labels/val/cs_valid_000001.txt")


def test_build_filtered_view_copies_images_and_filters_labels(tmp_path):
    root = tmp_path / "data"
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    img = root / "images" / "val" / "chd_val_000001.jpg"
    img.write_bytes(b"jpg")
    (root / "labels" / "val" / "chd_val_000001.txt").write_text("0 .5 .5 .1 .1\n10 .2 .2 .1 .1\n", encoding="utf-8")
    img2 = root / "images" / "val" / "hh_valid_000002.jpg"
    img2.write_bytes(b"jpg")  # no label file: must still get an (empty) one

    out_list = tr.build_filtered_view([img, img2], tmp_path / "view", 7)

    listed = [Path(p) for p in out_list.read_text(encoding="utf-8").splitlines()]
    assert [p.name for p in listed] == ["chd_val_000001.jpg", "hh_valid_000002.jpg"]
    assert all(p.exists() for p in listed)
    assert all("images" in p.parts for p in listed)
    assert tr.label_path_for(listed[0]).read_text(encoding="utf-8") == "0 .5 .5 .1 .1\n"
    assert tr.label_path_for(listed[1]).read_text(encoding="utf-8") == ""
    # originals untouched
    assert "10 " in (root / "labels" / "val" / "chd_val_000001.txt").read_text(encoding="utf-8")


# ---------- args ----------

def test_parse_args_defaults_keep_v1_behaviour():
    args = tr.parse_args([])
    assert args.cache == "off"
    assert args.per_source_val is None and args.compare_v1 is None
    assert args.dataset_repo == tr.DEFAULT_DATASET_REPO


def test_parse_args_new_flags():
    args = tr.parse_args(["--cache", "ram", "--no-per-source-val", "--compare-v1"])
    assert args.cache == "ram" and args.per_source_val is False and args.compare_v1 is True


@pytest.mark.parametrize("value,expected", [("off", False), ("ram", "ram"), ("disk", "disk")])
def test_cache_value(value, expected):
    assert tr.cache_value(value) == expected


@pytest.mark.parametrize("flag,has_sources,expected", [(None, True, True), (None, False, False),
                                                       (False, True, False), (True, False, False)])
def test_resolve_auto(flag, has_sources, expected):
    assert tr.resolve_auto(flag, has_sources) is expected
