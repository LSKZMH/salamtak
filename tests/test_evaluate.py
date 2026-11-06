"""Tests for scripts/evaluate.py: label parsing, metrics, COSTS.md parsing, end-to-end run."""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"
_spec = importlib.util.spec_from_file_location("evaluate", SCRIPT)
ev = importlib.util.module_from_spec(_spec)
sys.modules["evaluate"] = ev  # dataclasses need the module registered
_spec.loader.exec_module(ev)

IDS = ev.CLAUDE_MD_HAZARD_IDS
HEADER = "filename,hazards,taken_at,wbgt_c,outdoor,mode,notes"


def _parse(*rows: str) -> list:
    return ev.parse_labels([HEADER, *rows], IDS)


def _result(name: str, expected: set[str], predicted: set[str], wall: float = 1.0, mode: str = "site"):
    return ev.PhotoResult(name, frozenset(expected), frozenset(predicted), 100, wall, wall, mode)


# ---------- labels ----------
def test_parses_full_row():
    (photo,) = _parse("a.jpg,no_helmet; machinery_zone ,2026-08-12 11:30,33.4,false,site,x")
    assert photo == ev.LabelledPhoto(
        "a.jpg", frozenset({"no_helmet", "machinery_zone"}),
        datetime(2026, 8, 12, 11, 30), 33.4, False, "site",
    )


def test_blank_optional_columns_use_defaults_and_blank_hazards_mean_clean():
    (photo,) = _parse("b.jpg,,,,,,")
    assert photo == ev.LabelledPhoto("b.jpg", frozenset())


def test_only_required_columns_are_enough():
    (photo,) = ev.parse_labels(["filename,hazards", "c.jpg,no_helmet"], IDS)
    assert photo.hazards == {"no_helmet"} and photo.mode == "site"


def test_shipped_labels_parse_and_every_photo_exists():
    folder = SCRIPT.parent.parent / "examples" / "qatar_test"
    photos = ev.load_labels(folder / "labels.csv", IDS)
    assert photos and any(not p.hazards for p in photos) and any(p.hazards for p in photos)
    assert [p.filename for p in photos if not (folder / p.filename).is_file()] == []


def test_skips_blank_lines():
    assert len(_parse("a.jpg,,,,,,", ",,,,,,", "b.jpg,,,,,,")) == 2


def test_missing_required_column_is_rejected():
    with pytest.raises(ev.LabelError, match="hazards"):
        ev.parse_labels(["filename,notes", "a.jpg,x"], IDS)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("a.jpg,no_hat,,,,,", "unknown hazard"),
        ("a.jpg,home_trip,,,,site,", "can't occur in site mode"),
        ("a.jpg,no_helmet,,,,home,", "can't occur in home mode"),
        ("a.jpg,,yesterday,,,,", "taken_at"),
        ("a.jpg,,,hot,,,", "not a number"),
        ("a.jpg,,,75,,,", "outside"),
        ("a.jpg,,,,maybe,,", "true or false"),
        ("a.jpg,,,,,office,", "site or home"),
        (",no_helmet,,,,,", "filename is empty"),
        ("../secret.jpg,,,,,,", "relative"),
        ("C:/photos/a.jpg,,,,,,", "relative"),
        ("a.jpg,wbgt_stop,,,,,", "wbgt_stop needs wbgt_c"),
    ],
)
def test_bad_rows_are_rejected(row: str, message: str):
    with pytest.raises(ev.LabelError, match=message):
        _parse(row)


def test_reports_every_bad_row_and_duplicates_at_once():
    with pytest.raises(ev.LabelError) as err:
        _parse("a.jpg,bogus,,,,,", "b.jpg,,,,,,", "b.jpg,,,,,,", "c.jpg,,,hot,,,")
    text = str(err.value)
    assert "line 2" in text and "duplicate filename b.jpg" in text and "line 5" in text


# ---------- metrics ----------
def test_hazard_stats_counts_catches_and_false_alarms():
    results = [
        _result("1", {"no_helmet"}, {"no_helmet"}),
        _result("2", {"no_helmet"}, set()),
        _result("3", set(), {"no_helmet"}),
        _result("4", set(), set()),
    ]
    (s,) = ev.hazard_stats(results, ["no_helmet"])
    assert (s.labelled, s.caught, s.negatives, s.false_alarms) == (2, 1, 2, 1)
    assert s.catch_rate == 0.5 and s.false_alarm_rate == 0.5


def test_negatives_only_count_photos_where_the_hazard_can_occur():
    results = [
        _result("s1", {"no_helmet"}, {"no_helmet"}),
        _result("s2", set(), {"no_helmet"}),
        _result("h1", set(), set(), mode="home"),
        _result("h2", {"home_trip"}, set(), mode="home"),
    ]
    helmet, trip = ev.hazard_stats(results, ["no_helmet", "home_trip"])
    assert (helmet.negatives, helmet.false_alarms, helmet.false_alarm_rate) == (1, 1, 1.0)
    assert (trip.labelled, trip.negatives) == (1, 1)


def test_rates_are_none_without_data():
    (s,) = ev.hazard_stats([_result("1", {"wbgt_stop"}, {"wbgt_stop"})], ["wbgt_stop"])
    assert s.catch_rate == 1.0 and s.false_alarm_rate is None
    (empty,) = ev.hazard_stats([], ["wbgt_stop"])
    assert empty.catch_rate is None


def test_ids_to_report_keeps_table_order_and_appends_unknown():
    results = [_result("1", {"trip_cord"}, {"no_helmet", "custom_rule"})]
    assert ev.ids_to_report(results, IDS) == ["no_helmet", "trip_cord", "custom_rule"]
    assert ev.ids_to_report([], IDS) == []


def test_summarize_headline_numbers():
    results = [
        _result("1", {"no_helmet", "midday_ban"}, {"no_helmet"}, wall=1.0),
        _result("2", set(), {"trip_cord"}, wall=3.0),
        _result("3", set(), set(), wall=2.0),
    ]
    summary = ev.summarize(results, ev.hazard_stats(results, ev.ids_to_report(results, IDS)))
    assert summary["catch_rate"] == 0.5
    assert summary["false_alarms"] == 1
    assert (summary["clean_photos"], summary["clean_photos_flagged"]) == (2, 1)
    assert summary["exact_match_photos"] == 1
    assert (summary["wall_s_mean"], summary["wall_s_median"], summary["wall_s_max"]) == (2.0, 2.0, 3.0)


def test_summarize_empty():
    summary = ev.summarize([], [])
    assert summary["photos"] == 0 and summary["catch_rate"] is None and summary["wall_s_mean"] is None


# ---------- costs ----------
COSTS_MD = """# Credits log

| date | job / item | flavor | runtime | est. cost |
|---|---|---|---|---|
| 2026-11-06 | Space demo | cpu-basic | — | $0.00 |
| 2026-11-06 | smoke test | l4x1 | 6 min | $0.08 |
| 2026-11-06 | full train | a10g-large | 1h 40m | $ 2.50 |

**Total: $2.58**

## Space latency

| date | detector | s / photo | notes |
|---|---|---|---|
| 2026-11-06 | YOLO26s | 1.4 | costs $0 on free CPU |
"""


def test_parse_costs_sums_cost_column_only():
    costs = ev.parse_costs(COSTS_MD)
    assert costs == ev.CostSummary(stated_total=2.58, row_sum=2.58, rows=3)


def test_parse_costs_ignores_total_rows_and_uses_the_last_total_line():
    text = (
        "**Total: $9.99**\n\n"  # stale subtotal earlier in the file
        "| item | est. cost |\n|---|---|\n| a | $1.00 |\n| b | $0.50 |\n| **Total** | **$1.50** |\n\n"
        "**Total: $1.50**\n"
    )
    assert ev.parse_costs(text) == ev.CostSummary(stated_total=1.5, row_sum=1.5, rows=2)


def test_cost_lines_flag_a_stale_total():
    costs = ev.parse_costs(COSTS_MD.replace("$2.58", "$1.00"))
    assert any("disagree" in line for line in ev.cost_lines(costs, Path("COSTS.md")))


def test_cost_lines_without_total_or_file():
    assert "no '**Total" in ev.cost_lines(ev.parse_costs("nothing"), Path("C"))[0]
    assert "not found" in ev.cost_lines(None, Path("C"))[0]


# ---------- adapter ----------
def test_context_kwargs_match_either_scene_context():
    photo = ev.LabelledPhoto("a.jpg", frozenset(), wbgt_c=30.0, outdoor=False, mode="home")
    when = datetime(2026, 1, 1, 8, 0)
    chat3 = ev.context_kwargs({"taken_at", "wbgt_c", "outdoor", "mode", "aspect_ratio"}, photo, when, 1.5)
    chat4 = ev.context_kwargs({"taken_at", "outdoor", "wbgt", "mode"}, photo, when, 1.5)
    assert chat3 == {"taken_at": when, "wbgt_c": 30.0, "outdoor": False, "mode": "home", "aspect_ratio": 1.5}
    assert chat4 == {"taken_at": when, "wbgt": 30.0, "outdoor": False, "mode": "home"}


def test_load_detector_rejects_bad_spec():
    with pytest.raises(SystemExit, match="module:Name"):
        ev.load_detector("not-a-spec", None)


def test_module_spec_detector_is_flagged_mock_by_class_name():
    _, is_mock = ev.load_detector("evaluate:BuiltinMockDetector", None)
    _, is_real = ev.load_detector("collections:OrderedDict", None)
    assert is_mock is True and is_real is False


def test_input_warnings_for_rules_that_cannot_fire():
    ban = ev.LabelledPhoto("a.jpg", frozenset({"midday_ban"}))
    wbgt = ev.LabelledPhoto("b.jpg", frozenset(), wbgt_c=33.0)
    no_wbgt_field = {"taken_at", "outdoor", "mode"}
    assert any("midday_ban" in w for w in ev.input_warnings(ban, False, no_wbgt_field))
    assert ev.input_warnings(ban, True, no_wbgt_field) == []
    assert any("WBGT" in w for w in ev.input_warnings(wbgt, True, no_wbgt_field))
    assert ev.input_warnings(wbgt, True, {"taken_at", "wbgt_c"}) == []


# ---------- end to end ----------
FAKE_TYPES = '''
from dataclasses import dataclass
from datetime import datetime
HAZARD_IDS = ("no_helmet", "midday_ban", "home_trip")
@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple
@dataclass(frozen=True)
class Hazard:
    id: str
    severity: str
@dataclass(frozen=True)
class SceneContext:
    taken_at: datetime
    outdoor: bool = True
    wbgt: float | None = None
    mode: str = "site"
'''
FAKE_RULES = '''
from safesite.types import Hazard
def find_hazards(detections, context):
    out = [Hazard("no_helmet", "High") for d in detections if d.label == "no_helmet"]
    if context.taken_at.month == 8 and context.outdoor:
        out.append(Hazard("midday_ban", "High"))
    return out
'''
FAKE_SCORE = "def safety_score(h):\n    return max(0, 100 - 25 * len(list(h)))\n"


@pytest.fixture
def fake_repo(tmp_path: Path):
    pkg = tmp_path / "safesite"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "types.py").write_text(FAKE_TYPES)
    (pkg / "rules.py").write_text(FAKE_RULES)
    (pkg / "score.py").write_text(FAKE_SCORE)  # no detect.py: exercises the built-in mock
    (tmp_path / "COSTS.md").write_text("| item | cost |\n|---|---|\n| a | $1.25 |\n\n**Total: $1.25**\n")
    photos = tmp_path / "examples" / "qatar_test"
    photos.mkdir(parents=True)
    from PIL import Image

    for name in ("hot.jpg", "clean.jpg"):
        Image.new("RGB", (64, 48), "gray").save(photos / name)
    (photos / "labels.csv").write_text(
        "filename,hazards,taken_at\n"
        "hot.jpg,no_helmet;midday_ban,2026-08-12 11:30\n"
        "clean.jpg,,2026-11-05 16:00\n"
        "missing.jpg,no_helmet,\n"
    )
    saved = dict(sys.modules)
    # In the merged repo other tests import the real safesite first; drop it so the fake one loads.
    for name in [m for m in sys.modules if m == "safesite" or m.startswith("safesite.")]:
        del sys.modules[name]
    yield tmp_path
    for name in [m for m in sys.modules if m == "safesite" or m.startswith("safesite.")]:
        del sys.modules[name]
    sys.modules.update({k: v for k, v in saved.items() if k.startswith("safesite")})
    sys.path[:] = [p for p in sys.path if p != str(tmp_path)]


def test_main_fails_by_default_when_a_photo_is_skipped(fake_repo: Path, capsys):
    assert ev.main(["--repo-root", str(fake_repo)]) == 3
    report = capsys.readouterr().out
    assert "WARNING: 1 photo(s) skipped" in report


def test_main_end_to_end_with_builtin_mock(fake_repo: Path, capsys):
    out_json = fake_repo / "eval.json"
    code = ev.main(["--repo-root", str(fake_repo), "--json", str(out_json), "--allow-skipped"])
    report = capsys.readouterr().out
    assert code == 0
    assert "MOCK DETECTOR" in report and "built-in mock" in report
    assert "Photos evaluated: 2 (1 skipped)" in report and "missing.jpg" in report
    assert "WARNING: 1 photo(s) skipped" in report
    assert "| `no_helmet` | 1 | 1 | 100% | 1 / 1 | 100% |" in report  # mock flags every photo
    assert "| `midday_ban` | 1 | 1 | 100% | 0 / 1 | 0% |" in report
    assert "Credits spent (COSTS.md total): $1.25" in report
    data = json.loads(out_json.read_text())
    assert data["mock"] is True and data["summary"]["catch_rate"] == 1.0
    assert {p["filename"]: p["predicted"] for p in data["photos"]}["clean.jpg"] == ["no_helmet"]


def test_main_returns_2_on_bad_labels(fake_repo: Path, capsys):
    (fake_repo / "examples" / "qatar_test" / "labels.csv").write_text("filename,hazards\na.jpg,bogus\n")
    assert ev.main(["--repo-root", str(fake_repo)]) == 2
    assert "unknown hazard" in capsys.readouterr().err


def test_main_returns_1_when_no_photo_could_be_read(fake_repo: Path):
    (fake_repo / "examples" / "qatar_test" / "labels.csv").write_text("filename,hazards\nnope.jpg,\n")
    assert ev.main(["--repo-root", str(fake_repo)]) == 1


# ---------- v2: new hazard ids, --only-hazards, combined detector ----------
V2_IDS = ("worker_down", "under_suspended_load", "power_line_contact", "hot_work_fire",
          "no_fire_extinguisher", "exposed_rebar", "fall_into_opening")


def test_fallback_list_has_the_v2_hazard_ids():
    assert set(V2_IDS) <= set(ev.CLAUDE_MD_HAZARD_IDS)
    assert all(ev.hazard_mode(i) == "site" for i in V2_IDS)


def test_labelable_ids_keep_rules_order_and_add_fallback_ids():
    ids = ev.labelable_ids(("no_helmet", "custom_rule"))
    assert ids[:2] == ("no_helmet", "custom_rule")
    assert set(V2_IDS) <= set(ids) and len(ids) == len(set(ids))


def test_restrict_drops_out_of_scope_hazards_from_both_sides():
    results = [_result("a.jpg", {"no_helmet", "trip_cord"}, {"trip_cord", "midday_ban"}),
               _result("b.jpg", {"machinery_zone"}, {"no_helmet"})]
    a, b = ev.restrict(results, frozenset({"no_helmet", "machinery_zone"}))
    assert (a.expected, a.predicted) == ({"no_helmet"}, frozenset())
    assert (b.expected, b.predicted) == ({"machinery_zone"}, {"no_helmet"})
    assert results[0].predicted == {"trip_cord", "midday_ban"}  # originals untouched


def test_restrict_with_no_filter_returns_results_unchanged():
    results = [_result("a.jpg", {"no_helmet"}, {"trip_cord"})]
    assert ev.restrict(results, None) == results


def test_only_hazards_make_out_of_scope_predictions_not_false_alarms():
    results = ev.restrict([_result("a.jpg", set(), {"trip_cord", "heat_welfare_gap"})],
                          frozenset({"no_helmet"}))
    summary = ev.summarize(results, ev.hazard_stats(results, ["no_helmet"]))
    assert summary["false_alarms"] == 0 and summary["clean_photos_flagged"] == 0


def test_parse_only_hazards_validates_ids():
    assert ev.parse_only_hazards("no_helmet, machinery_zone", IDS) == {"no_helmet", "machinery_zone"}
    assert ev.parse_only_hazards(None, IDS) is None
    with pytest.raises(SystemExit, match="bogus"):
        ev.parse_only_hazards("no_helmet,bogus", IDS)
    with pytest.raises(SystemExit, match="empty"):
        ev.parse_only_hazards(" , ", IDS)


def test_report_ids_list_every_scored_hazard_even_if_never_seen():
    results = [_result("a.jpg", set(), set())]
    only = frozenset({"no_helmet", "fall_no_harness"})
    assert ev.report_ids(results, IDS, only) == ["no_helmet", "fall_no_harness"]
    assert ev.report_ids(results, IDS, None) == []


def test_unproducible_label_warnings_name_ids_the_rules_lack():
    photos = [ev.LabelledPhoto("a.jpg", frozenset({"no_helmet", "worker_down"}))]
    warnings = ev.unproducible_warnings(photos, ("no_helmet",))
    assert len(warnings) == 1 and "worker_down" in warnings[0]


def test_parse_costs_accepts_total_spent_with_approx_sign():
    costs = ev.parse_costs("| a | cost |\n|---|---|\n| x | $2.01 |\n\n**Total spent: ≈ $2.01** (Jobs ...)\n")
    assert costs.stated_total == 2.01


def test_combined_detector_spec_builds_trained_plus_open_vocab(monkeypatch):
    import types as pytypes

    made = []

    class FakeYolo:
        name = "yolo"

        def __init__(self, **kwargs):
            made.append(kwargs)

    class FakeOV:
        name = "yoloe"

    class FakeCombined:
        def __init__(self, primary, open_vocab):
            self.name = f"{primary.name} + {open_vocab.name}"

    detect = pytypes.SimpleNamespace(YoloDetector=FakeYolo, CombinedDetector=FakeCombined)
    openvocab = pytypes.SimpleNamespace(OpenVocabDetector=FakeOV)
    monkeypatch.setattr(ev, "_import_safesite", lambda name: {"detect": detect, "openvocab": openvocab}[name])
    detector, is_mock = ev.load_detector("combined", "w.pt")
    assert detector.name == "yolo + yoloe" and is_mock is False and made == [{"weights_path": "w.pt"}]


def test_main_only_hazards_end_to_end(fake_repo: Path, capsys):
    out_json = fake_repo / "eval.json"
    code = ev.main(["--repo-root", str(fake_repo), "--allow-skipped", "--only-hazards", "midday_ban",
                    "--json", str(out_json)])
    report = capsys.readouterr().out
    assert code == 0 and "Scored hazards only: midday_ban" in report
    assert "| `midday_ban` | 1 | 1 | 100% | 0 / 1 | 0% |" in report
    assert "`no_helmet`" not in report  # mock's no_helmet on the clean photo is ignored
    data = json.loads(out_json.read_text())
    assert data["only_hazards"] == ["midday_ban"] and data["summary"]["false_alarms"] == 0
