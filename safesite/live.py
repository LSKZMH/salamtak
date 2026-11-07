"""Live webcam mode: about one frame per second through the trained detector + rules.

No YOLOE here (too slow for live video), so scaffold / harness / cord / Home checks stay in the
Photo tab. A hazard is shown only after it appears in STABLE_MIN_HITS of the last STABLE_WINDOW
frames, so one noisy frame can't flash a false alarm. app.py mounts build_live_tab() as "Live".
"""
from __future__ import annotations

import base64
import html
import io
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
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

# Demo site camera: an 11 s CCTV-style clip (CC BY-SA 4.0, credited in examples/demo_video/CREDITS.md),
# analysed frame by frame like a live feed. The rules use the time burnt into its overlay.
DEMO_CLIP = Path(__file__).resolve().parent.parent / "examples" / "demo_video" / "site_cam_03.mp4"
DEMO_CLIP_START = datetime(2026, 11, 5, 8, 0, 0)
DEMO_FPS = 10  # the clip's frame rate; every frame is analysed and played back in real time
DEMO_JPEG_QUALITY = 72  # frames are sent inline as JPEG data URIs (~40 KB at 640 px), no per-frame download
LIVE_CONCURRENCY_ID = "live-detector"  # webcam and demo share one model copy, so one runs at a time
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


# ---------- demo site camera ----------
def clip_frames(path: Path) -> Iterator[np.ndarray]:
    """RGB frames of a video file, in order."""
    import cv2  # deferred: only the demo player needs it (installed with ultralytics)

    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def frame_html(image: Image.Image, quality: int = DEMO_JPEG_QUALITY) -> str:
    """One annotated frame as an inline JPEG <img>. Sent inside the event itself, so the browser shows it at once
    instead of downloading a separate file per frame (that is what made the feed stutter and flash)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return (f"<img class='ss-feed' src='data:image/jpeg;base64,{data}' "
            f"alt='CAM 03 demo frame with hazard boxes' style='width:100%;display:block'>")


FEED_PLACEHOLDER = ("<div class='ss-feed-idle' style='aspect-ratio:16/9;display:flex;align-items:center;"
                    "justify-content:center;background:#111;color:#bbb'>CAM 03 · press play</div>")


def make_demo_player(detector: Any, alerts_html: AlertsHtml, clip: Path = DEMO_CLIP, fps: float = DEMO_FPS,
                     sleep: Callable[[float], None] = time.sleep) -> Callable[[str], Iterator[tuple[Any, Any]]]:
    """Gradio generator: plays the demo clip through the detector + rules, yielding (frame html, status).

    The status is only re-sent when the verdict or the shown hazards change (gr.skip() otherwise), so the
    panel under the feed does not redraw ten times a second."""
    name = str(getattr(detector, "name", "detector"))
    context = SceneContext(taken_at=DEMO_CLIP_START, outdoor=True, wbgt_c=None, mode="site")

    def play(lang: str) -> Iterator[tuple[Any, Any]]:
        if not clip.is_file():
            yield FEED_PLACEHOLDER, "<p role='status'>Demo clip not found.</p>"
            return
        history: History = ()
        last_key: tuple | None = None
        last_drawn = float("-inf")
        for index, frame in enumerate(clip_frames(clip)):
            started = time.perf_counter()
            try:
                result = analyze_frame(detector, frame, context, history)
            except Exception:  # one bad frame must not end the demo
                log.exception("demo frame failed")
                continue
            history, secs = result.history, time.perf_counter() - started
            note = "" if result.analysis.assessed else "no people or equipment in view"
            # Redraw at once when the verdict changes; otherwise refresh the hazard list at most once a second.
            key = (note, verdict(result.shown).word)
            status: Any = gr.skip()
            if key != last_key or index - last_drawn >= fps:
                alerts = alerts_html(result.shown, lang) if result.shown else ""
                status = status_html(result.score, result.shown, name, secs, alerts, note)
                last_key, last_drawn = key, index
            yield frame_html(result.annotated), status
            sleep(max(0.0, 1.0 / fps - (time.perf_counter() - started)))

    return play


def build_demo_camera(detector: Any, lang: Any, alerts_html: AlertsHtml) -> None:
    """'Play demo site camera' button, its feed and status, inside the Live tab."""
    gr.Markdown("### Demo site camera\nNo webcam? Play an 11-second clip from a site camera. Every frame is "
                "checked live by the same detector and rules.")
    with gr.Row():
        with gr.Column(scale=1):
            play_btn = gr.Button("▶ Play demo site camera", variant="primary", elem_classes="ss-run")
            gr.Markdown("<small>Clip: *Building construction Excavator Moira Close Broadwater Farm Haringey 2025 01*"
                        " by Acabashi, CC BY-SA 4.0, via Wikimedia Commons; cut to 11 s, recoloured and given a "
                        "camera overlay (see `examples/demo_video/CREDITS.md`).</small>")
        with gr.Column(scale=1):
            gr.Markdown("<small>CAM 03: live hazards (HIGH = red, MED = orange)</small>")
            feed = gr.HTML(FEED_PLACEHOLDER)
            feed_status = gr.HTML()
    play_btn.click(make_demo_player(detector, alerts_html), inputs=[lang], outputs=[feed, feed_status],
                   concurrency_limit=LIVE_CONCURRENCY, concurrency_id=LIVE_CONCURRENCY_ID, show_progress="hidden")


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
        concurrency_id=LIVE_CONCURRENCY_ID,
        show_progress="hidden",
    )
    build_demo_camera(detector, lang, alerts_html)
