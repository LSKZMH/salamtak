"""Tests for the pure parts of scripts/bench_constructionsite.py (fake records, no download)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


cs = _load("bench_constructionsite")
ev = _load("evaluate")


def _rec(image_id: str, **rules) -> dict:
    """Fake record shaped like the dataset card: rule_N_violation is None or {bounding_box, reason}."""
    record = {"image_id": image_id}
    for n in (1, 2, 3, 4):
        reason = rules.get(f"r{n}")
        record[f"rule_{n}_violation"] = (
            None if reason is None else {"bounding_box": [[0.1, 0.1, 0.2, 0.3]], "reason": reason})
    return record


# ---------- violations ----------
@pytest.mark.parametrize(("value", "expected"), [
    (None, False),
    ({"bounding_box": [], "reason": ""}, False),
    ({"bounding_box": None, "reason": None}, False),
    ({"bounding_box": [[0.1, 0.2, 0.3, 0.4]], "reason": ""}, True),
    ({"bounding_box": [], "reason": "A worker is not wearing a hard hat."}, True),
])
def test_violated(value, expected):
    assert cs.violated(value) is expected


@pytest.mark.parametrize(("reason", "about_helmet"), [
    ("The worker with a white sweatshirt on the left is not wearing a hard hat.", True),
    ("Two workers without helmets are standing near the truck.", True),
    ("The worker has no hardhat.", True),
    ("The worker is wearing neither a hard hat nor a vest.", True),
    ("The worker isn't wearing a safety helmet.", True),
    ("The worker is wearing a hard hat but not a reflective vest at night.", False),
    ("The worker is wearing shorts that do not cover his legs.", False),
    ("The worker is grinding without safety glasses.", False),
    ("", False),
])
def test_rule1_reason_is_about_helmets(reason, about_helmet):
    assert cs.is_helmet_reason(reason) is about_helmet


def test_image_labels_map_rules_to_our_hazards():
    rec = _rec("1", r1="Worker is not wearing a hard hat.", r2="No harness at height.",
               r3="Pit edge has no guardrail.", r4="Worker inside the excavator radius.")
    labels = cs.image_labels(rec, rules=(1, 2, 3, 4))
    assert labels == cs.ImageLabels("1", frozenset(
        {"no_helmet", "fall_no_harness", "fall_into_opening", "machinery_zone"}), False, True)


def test_rule3_is_not_scored_unless_asked_but_still_makes_the_photo_unclean():
    labels = cs.image_labels(_rec("1", r3="Open pit edge."), rules=(1, 2, 4))
    assert labels.hazards == frozenset() and labels.any_violation is True


def test_rule1_without_helmet_mention_is_ambiguous_not_no_helmet():
    labels = cs.image_labels(_rec("1", r1="Worker in shorts.", r4="Near excavator."), rules=(1, 2, 4))
    assert labels.hazards == {"machinery_zone"} and labels.rule1_non_helmet is True


def test_clean_record():
    labels = cs.image_labels(_rec("9"), rules=(1, 2, 4))
    assert labels == cs.ImageLabels("9", frozenset(), False, False)


def test_rules_parse_and_reject_unknown():
    assert cs.parse_rules("1,2,4") == (1, 2, 4)
    with pytest.raises(SystemExit, match="5"):
        cs.parse_rules("1,5")


# ---------- sampling ----------
def _population() -> list:
    labels = []
    for i in range(40):
        labels.append(cs.ImageLabels(f"h{i:02d}", frozenset({"no_helmet"}), False, True))
    for i in range(5):
        labels.append(cs.ImageLabels(f"m{i:02d}", frozenset({"machinery_zone"}), False, True))
    labels.append(cs.ImageLabels("both", frozenset({"no_helmet", "machinery_zone"}), False, True))
    labels.append(cs.ImageLabels("amb", frozenset(), True, True))  # rule 1, not about helmets
    labels.append(cs.ImageLabels("r3only", frozenset(), False, True))  # violation we don't score
    for i in range(30):
        labels.append(cs.ImageLabels(f"c{i:02d}", frozenset(), False, False))
    return labels


def test_stratified_sample_caps_each_rule_and_clean_and_is_deterministic():
    pop = _population()
    first = cs.stratified_sample(pop, ("no_helmet", "machinery_zone"), per_rule=10, clean=8, seed=0)
    again = cs.stratified_sample(list(reversed(pop)), ("no_helmet", "machinery_zone"),
                                 per_rule=10, clean=8, seed=0)
    assert [x.image_id for x in first] == [x.image_id for x in again]  # input order doesn't matter
    ids = [x.image_id for x in first]
    assert len(ids) == len(set(ids))
    assert sum("no_helmet" in x.hazards for x in first) >= 10
    assert {x.image_id for x in first if "machinery_zone" in x.hazards} >= {f"m{i:02d}" for i in range(5)}
    assert sum(not x.any_violation for x in first) == 8
    assert "amb" not in ids and "r3only" not in ids


def test_stratified_sample_differs_with_another_seed():
    pop = _population()
    a = cs.stratified_sample(pop, ("no_helmet",), per_rule=10, clean=8, seed=0)
    b = cs.stratified_sample(pop, ("no_helmet",), per_rule=10, clean=8, seed=1)
    assert {x.image_id for x in a} != {x.image_id for x in b}


def test_pool_counts_report_every_bucket():
    counts = cs.pool_counts(_population(), ("no_helmet", "machinery_zone"))
    assert counts == {"no_helmet": 41, "machinery_zone": 6, "clean": 30,
                      "rule1_not_helmet_excluded": 1}


# ---------- labels.csv ----------
def test_labels_csv_is_valid_evaluate_input():
    sample = [cs.ImageLabels("0000424", frozenset({"no_helmet", "machinery_zone"}), False, True),
              cs.ImageLabels("0000007", frozenset(), False, False)]
    text = cs.labels_csv(sample)
    photos = ev.parse_labels(text.splitlines(), ev.ALL_HAZARD_IDS)
    assert [p.filename for p in photos] == ["images/0000007.jpg", "images/0000424.jpg"]
    assert photos[1].hazards == {"no_helmet", "machinery_zone"}
    assert all(p.taken_at is None and p.outdoor and p.mode == "site" for p in photos)


def test_image_id_must_be_a_plain_name():
    with pytest.raises(ValueError):
        cs.image_filename("../evil")
    assert cs.image_filename("0000424") == "images/0000424.jpg"


def test_evaluate_args_score_only_the_benchmark_rules(tmp_path: Path):
    args = cs.evaluate_args(tmp_path, ("no_helmet", "machinery_zone"), "yolo", None, 8, tmp_path / "o.json")
    assert args[args.index("--only-hazards") + 1] == "no_helmet,machinery_zone"
    assert args[args.index("--labels") + 1] == str(tmp_path / "subset" / "labels.csv")
    assert "--weights" not in args and args[args.index("--threads") + 1] == "8"
