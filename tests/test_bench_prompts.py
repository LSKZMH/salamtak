"""Tests for scripts/bench_prompts.py: bench loading, scoring and the ship rule (no model)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from safesite.types import Detection

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bench_prompts", ROOT / "scripts" / "bench_prompts.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench_prompts"] = bench
_spec.loader.exec_module(bench)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.fixture
def bench_dir(tmp_path: Path) -> Path:
    for name in ("fire/pos/a.jpg", "fire/pos/b.jpg", "rebar/pos/c.jpg", "neg/d.jpg", "neg/e.png"):
        _touch(tmp_path / name)
    _touch(tmp_path / "neg" / "notes.txt")  # ignored: not an image
    return tmp_path


def test_load_truth_uses_folder_labels_and_ignores_non_images(bench_dir: Path) -> None:
    truth = bench.load_truth(bench_dir)
    assert truth == {
        "fire/pos/a.jpg": {"fire"}, "fire/pos/b.jpg": {"fire"}, "rebar/pos/c.jpg": {"rebar"},
        "neg/d.jpg": frozenset(), "neg/e.png": frozenset(),
    }


def test_load_truth_adds_extra_labels_from_labels_csv(bench_dir: Path) -> None:
    (bench_dir / "labels.csv").write_text("file,labels\nrebar/pos/c.jpg,fire; scaffold\n", encoding="utf-8")
    assert bench.load_truth(bench_dir)["rebar/pos/c.jpg"] == {"rebar", "fire", "scaffold"}


def test_load_truth_rejects_unknown_file_in_labels_csv(bench_dir: Path) -> None:
    (bench_dir / "labels.csv").write_text("file,labels\nnope.jpg,fire\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nope.jpg"):
        bench.load_truth(bench_dir)


def test_max_conf_by_prompt_keeps_best_per_prompt_and_zero_for_missing() -> None:
    prompts = ("power lines", "fire")
    dets = [Detection("power lines", 0.3, (0, 0, 0.5, 0.5)),
            Detection("power lines", 0.7, (0.1, 0.1, 0.4, 0.4))]
    assert bench.max_conf_by_prompt(dets, prompts) == {"power lines": 0.7, "fire": 0.0}


def test_candidates_are_singles_then_pairs() -> None:
    assert bench.candidates(("a", "b", "c")) == [("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c")]


TRUTH = {
    "p1": frozenset({"fire"}), "p2": frozenset({"fire"}), "p3": frozenset({"fire"}),
    "o1": frozenset({"welding"}), "n1": frozenset(), "n2": frozenset(),
    "neg/neg_site_0.jpg": frozenset(),  # a plain site photo, so the site safeguard is real
}


def test_operating_point_counts_hits_and_false_positives_at_threshold() -> None:
    scores = {"p1": {"fire": 0.5}, "p2": {"fire": 0.3}, "p3": {"fire": 0.1},
              "o1": {"fire": 0.4}, "n1": {"fire": 0.0}, "n2": {"fire": 0.2},
              "neg/neg_site_0.jpg": {"fire": 0.0}}
    point = bench.operating_point("fire", ("fire",), 0.25, scores, TRUTH)
    assert point.hit_rate == pytest.approx(2 / 3)
    assert point.fp_rate == pytest.approx(1 / 4)  # o1 fires; n1, n2 and the site photo do not
    assert point.fp_rate_neg == 0.0
    assert (point.n_pos, point.n_other, point.n_neg, point.n_site) == (3, 4, 3, 1)
    assert not point.ships  # FP rate too high


def test_union_fires_when_either_phrasing_fires() -> None:
    scores = {k: {"fire": 0.0, "flames": 0.0} for k in TRUTH}
    scores["p1"]["fire"] = 0.5
    scores["p2"]["flames"] = 0.5
    assert bench.operating_point("fire", ("fire", "flames"), 0.3, scores, TRUTH).hit_rate == pytest.approx(2 / 3)


def test_sweep_picks_safest_shipping_threshold() -> None:
    # Positives score 0.5; the only negative hit scores 0.2 -> thresholds 0.25..0.50 ship.
    scores = {k: {"fire": 0.5 if k.startswith("p") else 0.0} for k in TRUTH}
    scores["n2"]["fire"] = 0.2
    best = bench.best_point(list(bench.sweep("fire", ("fire",), scores, TRUTH).values()))
    assert best.ships
    assert best.hit_rate == 1.0 and best.fp_rate == 0.0
    assert best.threshold == 0.50  # highest (safest) threshold with the same rates


def _point(hit: float, fp: float) -> object:
    return bench.OperatingPoint("welding", ("a",), 0.3, hit, fp, fp, 10, 100, 50, 0.0, 20)


def test_shipping_points_rank_by_hit_rate_minus_fp_rate() -> None:
    # a big hit-rate gain beats a small FP saving ...
    assert bench.best_point([_point(0.90, 0.05), _point(0.60, 0.03)]).hit_rate == 0.90
    # ... and a big FP saving beats a small hit-rate gain
    assert bench.best_point([_point(0.70, 0.09), _point(0.65, 0.0)]).fp_rate == 0.0
    # a non-shipping point never beats a shipping one
    assert bench.best_point([_point(0.95, 0.20), _point(0.60, 0.10)]).hit_rate == 0.60


def test_threshold_at_the_yoloe_floor_never_ships() -> None:
    at_floor = bench.OperatingPoint("fire", ("a",), bench.YOLOE_FLOOR, 0.9, 0.0, 0.0, 10, 100, 50, 0.0, 20)
    assert not at_floor.ships


def test_false_alarms_on_plain_site_photos_block_shipping() -> None:
    truth = {**TRUTH, "neg/neg_site_1.jpg": frozenset(), "neg/neg_site_2.jpg": frozenset()}
    scores = {k: {"trench": 0.5 if k.startswith("p") else 0.0} for k in truth}
    scores["neg/neg_site_1.jpg"]["trench"] = 0.5  # one plain site photo fires
    point = bench.operating_point("trench", ("trench",), 0.3,
                                  scores, {**truth, "p1": {"trench"}, "p2": {"trench"}, "p3": {"trench"}})
    assert point.fp_rate_site == pytest.approx(1 / 3) and not point.ships  # 1 of 3 site photos


def test_held_back_label_is_reported_but_not_recommended() -> None:
    truth = {**TRUTH, "w1": frozenset({"welding"}), "w2": frozenset({"welding"})}
    scores = {k: {p: 0.0 for p in bench.all_prompts()} for k in truth}
    scores["w1"]["person welding"] = scores["w2"]["person welding"] = 0.5
    text = bench.report(scores, truth, "yoloe-26n-seg.pt", 640)
    assert "### `welding` — passes, **held back**" in text
    assert "| `welding` |" not in text


def test_label_with_no_positives_never_ships() -> None:
    scores = {k: {"rebar": 0.0} for k in TRUTH}
    point = bench.operating_point("rebar", ("rebar",), 0.3, scores, TRUTH)
    assert point.n_pos == 0 and not point.ships


def test_report_lists_shipped_label_in_cutoffs_table() -> None:
    scores = {k: {p: 0.0 for p in bench.all_prompts()} for k in TRUTH}
    for k in ("p1", "p2", "p3"):
        scores[k]["fire"] = 0.55
    text = bench.report(scores, TRUTH, "yoloe-26n-seg.pt", 640)
    assert "### `fire` — **shipped**" in text
    assert "| `fire` | 'fire' | 0.55 |" in text
    assert "### `rebar` — not benched (no positive photos)" in text  # TRUTH has no rebar photos
    # existing prompts are re-checked at the cut-off rules.py really uses (default 0.25)
    assert "### `harness` — already shipped; at its deployed cut-off 0.25: FP rate 0.00" in text


def test_report_marks_benched_label_that_fails_as_not_shipped() -> None:
    truth = {**TRUTH, "w1": frozenset({"welding"})}
    scores = {k: {p: 0.0 for p in bench.all_prompts()} for k in truth}
    scores["n1"]["welding sparks"] = scores["n2"]["welding sparks"] = 0.9  # fires on negatives too
    scores["o1"]["welding sparks"] = scores["w1"]["welding sparks"] = 0.9
    text = bench.report(scores, truth, "yoloe-26n-seg.pt", 640)
    assert "### `welding` — not shipped" in text
    assert "| `welding` |" not in text


SITE_TRUTH = {**TRUTH, "neg/neg_site_1.jpg": frozenset(), "r1": frozenset({"rebar"}),
              "u1": frozenset({"utility_pole"})}


def test_production_section_shows_deployed_and_recommended_cutoffs(monkeypatch) -> None:
    from safesite import rules

    monkeypatch.setattr(rules, "OPEN_VOCAB_CUTOFFS", {})  # rules.py not updated yet -> default 0.25
    scores = {k: {} for k in SITE_TRUTH}
    scores["r1"]["rebar"] = 0.22  # a hit at the recommended 0.20, a miss at the deployed 0.25
    text = "\n".join(bench.production_section(scores, SITE_TRUTH))
    assert "| `rebar` | 0.25 | 0.00 | 0.00 | 0.00 | 0.20 | 1.00 | 0.00 |" in text
    assert "| `harness` | 0.25 |" in text


def test_production_section_flags_missing_cutoffs_in_rules(monkeypatch) -> None:
    from safesite import rules

    monkeypatch.setattr(rules, "OPEN_VOCAB_CUTOFFS", {})
    text = "\n".join(bench.production_section({k: {} for k in SITE_TRUTH}, SITE_TRUTH))
    assert "not yet in `rules.OPEN_VOCAB_CUTOFFS`" in text
    monkeypatch.setattr(rules, "OPEN_VOCAB_CUTOFFS", dict(bench.RECOMMENDED_CUTOFFS))
    text = "\n".join(bench.production_section({k: {} for k in SITE_TRUTH}, SITE_TRUTH))
    assert "not yet in" not in text


def test_production_section_refuses_stale_scores() -> None:
    with pytest.raises(ValueError, match="re-run with --production"):
        bench.production_section({"p1": {}}, SITE_TRUTH)


def test_recommended_cutoffs_match_the_shipped_site_prompts() -> None:
    assert set(bench.RECOMMENDED_CUTOFFS) <= set(bench.openvocab.SITE_PROMPTS.values())


def test_no_site_negatives_means_nothing_ships() -> None:
    # the site-photo safeguard must not pass vacuously
    truth = {k: v for k, v in TRUTH.items() if not k.startswith("neg/neg_site")}
    scores = {k: {"fire": 0.5 if k.startswith("p") else 0.0} for k in truth}
    assert not bench.operating_point("fire", ("fire",), 0.3, scores, truth).ships


def test_all_prompts_has_no_duplicates_and_includes_existing() -> None:
    prompts = bench.all_prompts()
    assert len(prompts) == len(set(prompts))
    assert "safety harness" in prompts and "power lines" in prompts
