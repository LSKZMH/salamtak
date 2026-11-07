"""Live webcam mode: about one frame per second through the trained detector + rules.

No YOLOE here (too slow for live video), so scaffold / harness / cord / Home checks stay in the
Photo tab. A hazard is shown only after it appears in STABLE_MIN_HITS of the last STABLE_WINDOW
frames, so one noisy frame can't flash a false alarm. app.py mounts build_live_tab() as "Live".
"""
from __future__ import annotations

import html
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

from safesite import score
from safesite.annotate import annotate
from safesite.pipeline import Analysis, analyze_image
from safesite.types import Hazard, SceneContext
from safesite.verdict import verdict

log = logging.getLogger(__name__)

LIVE_INTERVAL_S = 1.0  # one frame per second; the trained YOLO26s takes ~0.2 s per frame on the Space CPU
LIVE_TIME_LIMIT_S = 180  # each viewer's stream stops after 3 min so one open tab can't hog the shared CPU
LIVE_CONCURRENCY = 1  # live frames analysed at once (the Live tab has its own model copy)
LIVE_MAX_SIDE = 640  # the detector works at 640 px anyway; smaller frames are faster end to end
STABLE_WINDOW = 3  # frames remembered per viewer
STABLE_MIN_HITS = 2  # a hazard must appear in this many of them before it is shown
WBGT_MIN, WBGT_MAX = 0.0, 60.0
DOHA_TZ = timezone(timedelta(hours=3))  # Qatar: UTC+3, no DST
PAUSED_HTML = ("<p role='status'>Live analysis paused: this frame could not be analysed. "
               "It will try again on the next frame.</p>")

History = tuple[frozenset[str], ...]
AlertsHtml = Callable[[Sequence[Hazard], str], str]


@dataclass(frozen=True)
class LiveFrame:
    annotated: Image.Image
    analysis: Analysis  # everything the rules found in this frame
    shown: tuple[Hazard, ...]  # only the hazards that are stable across frames
    history: History
    score: int  # score of the shown hazards


# ---------- debouncing (pure) ----------
def update_history(history: History, seen: frozenset[str], window: int = STABLE_WINDOW) -> History:
    """New history with this frame's hazard ids appended, keeping the last ``window`` frames."""
    return (*history, seen)[-window:]


def stable_ids(history: History, min_hits: int = STABLE_MIN_HITS) -> frozenset[str]:
    """Hazard ids seen in at least ``min_hits`` frames of the history."""
    counts: dict[str, int] = {}
    for frame_ids in history:
        for hazard_id in frame_ids:
            counts[hazard_id] = counts.get(hazard_id, 0) + 1
    return frozenset(h for h, n in counts.items() if n >= min_hits)


# ---------- inputs ----------
def to_image(frame: Any) -> Image.Image | None:
    """Webcam frame (numpy RGB array or PIL image) -> RGB PIL image, at most LIVE_MAX_SIDE px."""
    if frame is None:
        return None
    image = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame, dtype=np.uint8))
    small = image.convert("RGB")  # always a new image
    small.thumbnail((LIVE_MAX_SIDE, LIVE_MAX_SIDE), Image.Resampling.BILINEAR)
    return small


def clean_wbgt(value: float | None) -> float | None:
    """WBGT in range, else None (the stream must never stop on a typo)."""
    if value is None or not (WBGT_MIN <= float(value) <= WBGT_MAX):
        return None
    return float(value)


def live_context(wbgt: float | None, outdoor: bool) -> SceneContext:
    """Site mode, time = now in Doha (so the midday ban applies live)."""
    now = datetime.now(DOHA_TZ).replace(tzinfo=None, microsecond=0)
    return SceneContext(taken_at=now, outdoor=bool(outdoor), wbgt_c=clean_wbgt(wbgt), mode="site")


# ---------- analysis ----------
def analyze_frame(detector: Any, frame: Any, context: SceneContext, history: History) -> LiveFrame:
    image = to_image(frame)
    if image is None:
        raise ValueError("no frame")
    prepared, analysis = analyze_image(detector, image, context)
    new_history = update_history(history, frozenset(h.id for h in analysis.hazards))
    keep = stable_ids(new_history)
    shown = tuple(h for h in analysis.hazards if h.id in keep)
    annotated = annotate(prepared, analysis.detections, shown)
    return LiveFrame(annotated, analysis, shown, new_history, score.safety_score(shown))


# ---------- presentation ----------
def status_html(score_value: int, shown: Sequence[Hazard], detector_name: str, secs: float,
                alerts: str, note: str = "") -> str:
    """Safe/Unsafe line, what to do, stable hazards and alerts. ``alerts`` is trusted HTML from
    app._alerts_html. ``score_value`` is kept for callers but no longer shown (verdict only)."""
    action = ""
    if note:
        css, headline = "na", f"Live · {html.escape(note)}"
    else:
        v = verdict(shown)
        css, headline = v.css, f"Live · {v.word}"
        if v.action:
            action = f"<p class='ss-action'><span aria-hidden='true'>⚠ </span>{html.escape(v.action)}</p>"
    rows = "".join(f"<li><strong>{html.escape(h.severity)}</strong> {html.escape(h.id)}: "
                   f"{html.escape(h.reason)}</li>" for h in shown)
    return (
        f"<div role='status' aria-live='polite' class='ss-card ss-{css}'><div class='ss-score ss-{css}'>{headline}</div>"
        f"{action}{f'<ul>{rows}</ul>' if rows else ''}{alerts}"
        f"<p><small>Detector: {html.escape(detector_name)} · {secs:.2f} s/frame · a hazard shows after "
        f"{STABLE_MIN_HITS} of {STABLE_WINDOW} frames</small></p></div>"
    )


def make_handler(detector: Any, alerts_html: AlertsHtml) -> Callable[..., tuple[Any, Any, History]]:
    """Gradio stream handler: (frame, history, lang, wbgt, outdoor) -> (annotated, status, history)."""
    name = str(getattr(detector, "name", "detector"))

    def handle(frame: Any, history: History | None, lang: str, wbgt: float | None,
               outdoor: bool) -> tuple[Any, Any, History]:
        past: History = tuple(history or ())
        if frame is None:
            return gr.skip(), gr.skip(), past
        started = time.perf_counter()
        try:
            result = analyze_frame(detector, frame, live_context(wbgt, outdoor), past)
        except Exception:  # one bad frame must not end the stream
            log.exception("live frame failed")
            return gr.skip(), PAUSED_HTML, past
        secs = time.perf_counter() - started
        note = "" if result.analysis.assessed else "no people or equipment in view"
        alerts = alerts_html(result.shown, lang) if result.shown else ""
        return result.annotated, status_html(result.score, result.shown, name, secs, alerts, note), result.history

    return handle


# ---------- UI ----------
def build_live_tab(detector: Any, languages: dict[str, str], alerts_html: AlertsHtml) -> None:
    """Add the Live webcam UI to the enclosing gr.Blocks / gr.Tab."""
    gr.Markdown(
        f"Point the camera at the site and press record. About one frame per second is checked with the trained "
        f"detector and the safety rules; a hazard appears once it is seen in {STABLE_MIN_HITS} of "
        f"{STABLE_WINDOW} frames. Scaffold, harness, cord and Home checks are in the Photo tab. "
        f"The stream stops after {LIVE_TIME_LIMIT_S // 60} minutes; press record again to continue."
    )
    with gr.Row():
        with gr.Column(scale=1):
            cam = gr.Image(label="Webcam", sources=["webcam"], streaming=True, type="numpy")
            lang = gr.Dropdown(label="Alert language", choices=[(n, c) for c, n in languages.items()],
                               value=next(iter(languages)))
            wbgt = gr.Number(label="WBGT °C (optional)", value=None, minimum=WBGT_MIN, maximum=WBGT_MAX)
            outdoor = gr.Checkbox(label="Outdoor", value=True)
        with gr.Column(scale=1):
            annotated = gr.Image(label="Live hazards (HIGH = red, MED = orange)", type="pil", format="jpeg")
            status = gr.HTML()
    history = gr.State(())
    cam.stream(
        make_handler(detector, alerts_html),
        inputs=[cam, history, lang, wbgt, outdoor],
        outputs=[annotated, status, history],
        stream_every=LIVE_INTERVAL_S,
        time_limit=LIVE_TIME_LIMIT_S,
        concurrency_limit=LIVE_CONCURRENCY,
        show_progress="hidden",
    )
