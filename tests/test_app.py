"""App-level tests: Gradio handler, time resolution, alert/score rendering, annotation.
Rule-by-rule tests live in tests/test_rules.py."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr
import pytest
from PIL import Image

import app
from safesite import alerts
from safesite.annotate import SEVERITY_COLORS, annotate
from safesite.pipeline import Analysis
from safesite.types import HAZARD_SEVERITY, Hazard

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SUMMER_NOON = datetime(2026, 7, 15, 12, 30)


def _blank(w: int = 640, h: int = 480) -> Image.Image:
    return Image.new("RGB", (w, h), (200, 200, 200))


def _exif_jpeg(tmp_path: Path, stamp: str, tag: int = app.EXIF_DATETIME_ORIGINAL) -> Image.Image:
    exif = Image.Exif()
    if tag == app.EXIF_DATETIME_ORIGINAL:
        exif.get_ifd(app.EXIF_IFD)[tag] = stamp
    else:
        exif[tag] = stamp
    path = tmp_path / "p.jpg"
    _blank().save(path, exif=exif)
    return Image.open(path)


@pytest.fixture
def fake_alerts() -> alerts.AlertTable:
    """A small parsed table, passed via table= so tests never depend on alerts.json contents."""
    return alerts.parse_alerts({
        "no_helmet": {
            "en": {"text": "Wear <b>helmet</b>", "verified": True},
            "ar": {"text": "ارتدِ الخوذة", "verified": False},
        },
        "machinery_zone": {"en": {"text": "Step away", "verified": True}},
    })


# ---------- handler ----------
@pytest.mark.parametrize("row", app.EXAMPLES, ids=lambda row: Path(row[0]).name)
def test_analyze_runs_end_to_end_on_examples(row):
    """Each example row exactly as a judge's click passes it (photo, time, WBGT, outdoor, lang, mode)."""
    path, *inputs = row
    score_html, alert_html, annotated, table = app.analyze(Image.open(path), *inputs)
    assert any(w in score_html for w in ("Unsafe", "Needs attention", "Safe")) and "/100" not in score_html
    assert max(annotated.size) <= 1280
    assert table and all(len(row) == 3 for row in table)
    assert "<li" in alert_html


def test_summer_noon_example_reads_exif_and_floors_score(tmp_path):
    score_html, *_ = app.analyze(Image.open(EXAMPLES / "site_summer_noon.jpg"), "", 33.5, True, "en", "Site")
    assert "Unsafe" in score_html and "report to your supervisor" in score_html
    assert "photo EXIF" in score_html and "2026-07-15 12:30" in score_html


def test_analyze_without_image_raises():
    with pytest.raises(gr.Error):
        app.analyze(None, "", None, True, "en", "Site")


@pytest.mark.parametrize("wbgt", [-1, 61, float("nan")])
def test_out_of_range_wbgt_raises(wbgt):
    with pytest.raises(gr.Error):
        app.analyze(_blank(), "", wbgt, True, "en", "Site")


def test_detector_failure_becomes_friendly_error(monkeypatch):
    def boom(image, mode):
        raise RuntimeError("internal detail")

    boom.name, boom.is_mock, boom.supports_home = "boom", False, False
    monkeypatch.setattr(app, "DETECTOR", boom)
    with pytest.raises(gr.Error) as info:
        app.analyze(_blank(), "", None, True, "en", "Site")
    assert "internal detail" not in str(info.value) and isinstance(info.value.__cause__, RuntimeError)


def test_unknown_language_falls_back_to_english():
    _, alert_html, *_ = app.analyze(_blank(), "", None, True, "xx", "Site")
    assert "lang='en'" in alert_html


# ---------- time ----------
def test_resolve_time_prefers_override_then_exif_then_now(tmp_path):
    img = _exif_jpeg(tmp_path, "2026:07:15 12:30:00")
    assert app.resolve_time("2026-01-02 03:04", img) == (datetime(2026, 1, 2, 3, 4), "manual override")
    assert app.resolve_time("2026-01-02T03:04:05", img)[0] == datetime(2026, 1, 2, 3, 4, 5)
    assert app.resolve_time("  ", img) == (SUMMER_NOON, "photo EXIF")
    now, source = app.resolve_time(None, _blank())
    doha_now = datetime.now(app.DOHA_TZ).replace(tzinfo=None)
    assert source.startswith("now (Doha time)") and abs((doha_now - now).total_seconds()) < 5


def test_exif_falls_back_to_base_datetime_tag(tmp_path):
    assert app.exif_time(_exif_jpeg(tmp_path, "2026:08:01 11:00:00", app.EXIF_DATETIME)) == datetime(2026, 8, 1, 11)


def test_malformed_exif_time_is_ignored(tmp_path):
    assert app.exif_time(_exif_jpeg(tmp_path, "garbage")) is None


def test_bad_override_raises_friendly_error():
    with pytest.raises(gr.Error):
        app.resolve_time("next tuesday", _blank())


# ---------- alerts rendering ----------
def test_alerts_are_escaped_and_tagged_with_language(fake_alerts):
    hazards = [Hazard.of("no_helmet", "r"), Hazard.of("no_helmet", "r2")]
    html_out = app._alerts_html(hazards, "en", table=fake_alerts)
    assert "&lt;b&gt;helmet&lt;/b&gt;" in html_out and "<b>" not in html_out
    assert html_out.count("<li") == 1  # same id twice -> one alert


def test_rtl_translation_shows_without_unverified_label(fake_alerts):
    # Workers see the alert alone; the "verified" flag stays in alerts.json for the team, not in the UI.
    html_out = app._alerts_html([Hazard.of("no_helmet", "r")], "ar", table=fake_alerts)
    assert "lang='ar' dir='rtl'" in html_out and "unverified" not in html_out


def test_missing_translation_falls_back_to_english_with_note(fake_alerts):
    # A missing hazard id can't happen: alerts.load_alerts() guarantees every id at boot.
    html_out = app._alerts_html([Hazard.of("machinery_zone", "r")], "hi", table=fake_alerts)
    assert "lang='en' dir='ltr'" in html_out and "English: no translation yet" in html_out


def test_real_alerts_json_covers_every_rendered_hazard():
    html_out = app._alerts_html([Hazard.of("no_helmet", "r"), Hazard.of("heat_welfare_gap", "r")], "ar")
    assert html_out.count("<li") == 2


def test_no_hazards_message():
    assert "No hazards found" in app._alerts_html([], "en")


# ---------- score rendering ----------
def _analysis(score: int, assessed: bool = True, hazards: tuple = ()) -> Analysis:
    return Analysis((), hazards, score, assessed, ("a <note>",))


def _hazard(hazard_id: str) -> Hazard:
    return Hazard(id=hazard_id, severity=HAZARD_SEVERITY[hazard_id], reason="seen")


@pytest.mark.parametrize("hazards, css, word, action", [
    ((), "good", "Safe", ""),
    ((_hazard("trip_cord"),), "warn", "Needs attention", "Report to your supervisor."),
    ((_hazard("no_helmet"), _hazard("trip_cord")), "bad", "Unsafe", "Stop work and report to your supervisor now."),
])
def test_score_card_shows_verdict_and_action_not_a_number(hazards, css, word, action):
    out = app._score_html(_analysis(50, hazards=hazards), SUMMER_NOON, "photo EXIF", 0.1)
    assert f"ss-card ss-{css}" in out and f">{word}<" in out and "a &lt;note&gt;" in out
    assert "/100" not in out and "50" not in out.split("Photo time")[0]
    assert (action in out) if action else ("ss-action" not in out)


def test_home_mode_action_does_not_mention_a_site_officer():
    out = app._score_html(_analysis(90, hazards=(_hazard("trip_cord"),)), SUMMER_NOON, "x", 0.1, "home")
    assert "Fix this soon." in out and "supervisor" not in out


def test_unassessed_photo_never_shows_a_verdict():
    out = app._score_html(_analysis(100, assessed=False), SUMMER_NOON, "x", 0.1)
    assert "Not assessed" in out and ">Safe<" not in out and "ss-action" not in out


# ---------- UI + annotation ----------
# ---------- detector switch (SAFESITE_DETECTOR) ----------
def _must_not_build():
    raise AssertionError("the trained model must not be loaded in mock mode")


@pytest.mark.parametrize("kind", ["mock", "", "  MOCK "])
def test_detector_switch_defaults_to_mock(kind):
    primary, warning = app.build_primary(kind, factory=_must_not_build)
    assert primary.is_mock and warning is None


def test_detector_switch_yolo_builds_the_trained_model():
    trained = object()
    primary, warning = app.build_primary("yolo", factory=lambda: trained)
    assert primary is trained and warning is None


def test_trained_model_failure_falls_back_to_mock_with_visible_warning():
    def broken():
        raise RuntimeError("401 Unauthorized: hf_secret_should_not_leak")

    primary, warning = app.build_primary("yolo", factory=broken)
    assert primary.is_mock
    assert "failed to load" in warning and "RuntimeError" in warning
    assert "hf_secret_should_not_leak" not in warning  # details go to the logs, not the page


def test_unknown_detector_value_falls_back_with_warning():
    primary, warning = app.build_primary("yolov8", factory=_must_not_build)
    assert primary.is_mock and "yolov8" in warning


def test_default_app_boot_uses_mock_without_warning():
    assert app.PRIMARY.is_mock and app.LOAD_WARNING is None


def test_ui_builds_and_examples_exist():
    assert isinstance(app.build_ui(), gr.Blocks)
    assert all(Path(row[0]).is_file() for row in app.EXAMPLES)


def test_every_real_example_photo_is_credited():
    """CC BY-SA requires attribution: each example we didn't draw must appear in a CREDITS.md next to it."""
    synthetic = {"site_summer_noon.jpg", "site_winter_morning.jpg", "home_kitchen.jpg"}  # scripts/make_placeholders.py
    for row in app.EXAMPLES:
        photo = Path(row[0])
        if photo.name not in synthetic:
            credits = photo.parent / "CREDITS.md"
            assert credits.is_file() and f"`{photo.name}`" in credits.read_text(encoding="utf-8"), photo.name


def test_annotate_draws_high_box_red_and_keeps_input():
    img = _blank()
    out = annotate(img, [], [Hazard.of("no_helmet", "x", ((0.1, 0.3, 0.5, 0.7),))])
    assert out.size == img.size and img.getpixel((64, 200)) == (200, 200, 200)
    assert out.getpixel((65, 200)) == SEVERITY_COLORS["High"]  # inside the thick left edge


def test_annotate_frame_level_hazard_draws_border():
    out = annotate(_blank(), [], [Hazard.of("midday_ban", "x")])
    assert out.getpixel((1, 240)) == SEVERITY_COLORS["High"]
