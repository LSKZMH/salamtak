"""Live webcam mode: frame analysis, hazard debouncing and the Live tab."""
from __future__ import annotations

import gradio as gr
import numpy as np
import pytest
from PIL import Image

from safesite import detect, live
from safesite.types import Hazard, Severity


def _frame(width: int = 1280, height: int = 720) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


def _no_alerts(hazards, lang: str) -> str:
    return "<p>alerts</p>"


# ---------- debouncing ----------
def test_hazard_seen_once_is_not_shown():
    history = live.update_history((), frozenset({"no_helmet"}))
    assert live.stable_ids(history) == frozenset()


def test_hazard_seen_in_two_of_three_frames_is_shown():
    history: tuple[frozenset[str], ...] = ()
    for seen in ({"no_helmet"}, set(), {"no_helmet"}):
        history = live.update_history(history, frozenset(seen))
    assert live.stable_ids(history) == frozenset({"no_helmet"})


def test_history_keeps_only_the_last_window_of_frames():
    history: tuple[frozenset[str], ...] = ()
    for _ in range(10):
        history = live.update_history(history, frozenset({"x"}))
    assert len(history) == live.STABLE_WINDOW


def test_old_hazard_drops_out_after_it_leaves_the_window():
    history: tuple[frozenset[str], ...] = ()
    for seen in ({"no_helmet"}, {"no_helmet"}, set(), set()):
        history = live.update_history(history, frozenset(seen))
    assert live.stable_ids(history) == frozenset()


def test_update_history_does_not_mutate_its_input():
    before = (frozenset({"a"}),)
    live.update_history(before, frozenset({"b"}))
    assert before == (frozenset({"a"}),)


# ---------- frame conversion ----------
def test_numpy_frame_becomes_small_rgb_image():
    image = live.to_image(_frame(1920, 1080))
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert max(image.size) == live.LIVE_MAX_SIDE


def test_missing_frame_gives_none():
    assert live.to_image(None) is None


@pytest.mark.parametrize("value, expected", [(None, None), (31.0, 31.0), (-5.0, None), (99.0, None)])
def test_out_of_range_wbgt_is_ignored(value, expected):
    assert live.clean_wbgt(value) == expected


# ---------- frame analysis ----------
def test_first_frame_shows_no_hazards_until_they_repeat():
    first = live.analyze_frame(detect.MockDetector(), _frame(), live.live_context(None, True), ())
    assert first.analysis.hazards, "mock scene should contain hazards"
    assert first.shown == ()
    second = live.analyze_frame(detect.MockDetector(), _frame(), live.live_context(None, True), first.history)
    assert {h.id for h in second.shown} == {h.id for h in second.analysis.hazards}
    assert isinstance(second.annotated, Image.Image)


def test_score_reflects_only_shown_hazards():
    first = live.analyze_frame(detect.MockDetector(), _frame(), live.live_context(None, True), ())
    assert first.score == 100


def test_live_context_uses_site_mode_and_now():
    ctx = live.live_context(33.0, False)
    assert ctx.mode == "site" and ctx.outdoor is False and ctx.wbgt_c == 33.0
    assert ctx.taken_at is not None


# ---------- stream handler ----------
def test_handler_skips_empty_frames_and_keeps_history():
    handler = live.make_handler(detect.MockDetector(), _no_alerts)
    history = (frozenset({"no_helmet"}),)
    image, status, new_history = handler(None, history, "en", None, True)
    assert new_history == history


def test_handler_returns_frame_status_and_history():
    handler = live.make_handler(detect.MockDetector(), _no_alerts)
    image, status, history = handler(_frame(), (), "en", None, True)
    assert isinstance(image, Image.Image)
    assert "Live" in status and "alerts" not in status  # first frame: nothing stable yet
    assert len(history) == 1
    _, status, history = handler(_frame(), history, "en", None, True)
    assert "alerts" in status and len(history) == 2


def test_handler_survives_a_detector_crash():
    def broken(image, mode):
        raise RuntimeError("boom")

    handler = live.make_handler(broken, _no_alerts)
    image, status, history = handler(_frame(), (), "en", None, True)
    assert "paused" in status.lower()
    assert history == ()


def test_status_escapes_hazard_text():
    hazard = Hazard(id="no_helmet", severity=Severity.HIGH, reason="<script>x</script>")
    html_out = live.status_html(100, (hazard,), "Mock", 0.1, "<p>a</p>")
    assert "<script>" not in html_out


# ---------- UI ----------
def test_live_tab_builds_a_streaming_webcam():
    with gr.Blocks() as demo:
        live.build_live_tab(detect.MockDetector(), {"en": "English"}, _no_alerts)
    cams = [b for b in demo.blocks.values() if isinstance(b, gr.Image) and getattr(b, "streaming", False)]
    assert len(cams) == 1
    assert cams[0].sources == ["webcam"]


def test_app_has_photo_and_live_tabs():
    import app

    labels = {b.label for b in app.demo.blocks.values() if isinstance(b, gr.Tab)}
    assert {"Photo", "Live"} <= labels
