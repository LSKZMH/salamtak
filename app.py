"""Salamtak سلامتك — Gradio Space entry point.

Photo -> detector -> rules engine -> verdict + annotated image + alerts. No LLM anywhere.
All scoring goes through safesite.pipeline.analyze_image; scripts/evaluate.py should call it too
so the pitch numbers measure exactly this app (it currently calls rules directly).
"""
from __future__ import annotations

import html
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from safesite import alerts, detect, live, openvocab
from safesite.annotate import annotate
from safesite.pipeline import Analysis, ImageTooLargeError, analyze_image
from safesite.types import Hazard, SceneContext
from safesite.verdict import verdict

# code -> display name (native script first so workers recognise it); codes match alerts.LANGUAGES
LANGUAGES: dict[str, str] = {
    "en": "English",
    "ar": "العربية (Arabic)",
    "hi": "हिन्दी (Hindi)",
    "bn": "বাংলা (Bengali)",
    "ne": "नेपाली (Nepali)",
    "ur": "اردو (Urdu)",
    "ml": "മലയാളം (Malayalam)",
}
RTL_LANGUAGES = frozenset({"ar", "ur"})

log = logging.getLogger("safesite")
logging.basicConfig(level=logging.INFO)
MAX_ECHO_KIND = 20  # chars of an unknown SAFESITE_DETECTOR value shown in the banner


def build_primary(kind: str, factory: Callable[[], Any] = detect.YoloDetector) -> tuple[Any, str | None]:
    """The trained-detector slot: "mock" (default) or "yolo" (Chat 2's YOLO26 from the Hub,
    configured by SAFESITE_MODEL_REPO / SAFESITE_WEIGHTS_FILE / SAFESITE_MODEL_REVISION).

    Never crashes the Space: an unknown value or a model that fails to load falls back to the
    mock and returns a warning the UI shows as a banner. Full error details go only to the logs.
    """
    choice = (kind or "mock").strip().lower()
    if choice == "mock":
        return detect.MockDetector(), None
    if choice != "yolo":
        log.error("unknown SAFESITE_DETECTOR value %r; using the mock", kind)
        shown = html.escape(choice[:MAX_ECHO_KIND])
        return detect.MockDetector(), f"Unknown SAFESITE_DETECTOR value '{shown}' (use mock or yolo); showing the mock detector."
    try:
        return factory(), None
    except Exception as exc:  # any load failure: keep the demo up, say so loudly
        log.exception("trained detector failed to load; using the mock")
        return detect.MockDetector(), (
            f"Trained model failed to load ({type(exc).__name__}); showing the mock detector. See the Space logs."
        )


# Trained detector: set the Space variable SAFESITE_DETECTOR=yolo once Chat 2's best.pt is on the Hub
# (and pin SAFESITE_MODEL_REVISION); no code change needed.
PRIMARY, LOAD_WARNING = build_primary(os.environ.get("SAFESITE_DETECTOR", "mock"))
# YOLOE-26 open-vocab finder (harness, scaffold, cords, ...; all of Home mode). Set the Space
# variable SAFESITE_OPEN_VOCAB=0 to switch it off (tests do, so they never load the model).
OPEN_VOCAB = os.environ.get("SAFESITE_OPEN_VOCAB", "1") != "0"
DETECTOR = detect.CombinedDetector(PRIMARY, openvocab.OpenVocabDetector()) if OPEN_VOCAB else PRIMARY
# Live tab: its own copy of the trained model (Ultralytics models are not thread-safe, and live frames
# run alongside photo checks). Trained detector + rules only; YOLOE is too slow for live video.
LIVE_DETECTOR, _ = build_primary(os.environ.get("SAFESITE_DETECTOR", "mock"))

DOHA_TZ = timezone(timedelta(hours=3))  # Qatar: UTC+3, no DST
EXIF_DATETIME_ORIGINAL = 0x9003
EXIF_DATETIME = 0x0132
EXIF_IFD = 0x8769
TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M")
WBGT_MIN, WBGT_MAX = 0.0, 60.0
MAX_ECHO = 40  # chars of user input echoed back in error messages
MAX_FILE_SIZE = "15mb"
QUEUE_SIZE = 16
CONCURRENCY = 1  # one photo at a time: torch already uses every core
EXAMPLES_DIR = Path(__file__).parent / "examples"
TABLE_HEADERS = ["Hazard", "Severity", "Reason"]
DISCLAIMER = "Advisory only: no alert does not mean no hazard. Always follow your site safety officer."
# Construction-site look: asphalt + safety yellow, hazard-stripe band, condensed signage type.
SAFETY_YELLOW = "#FFC20E"
ASPHALT = "#1B1C1E"
CSS = f"""
:root{{--ss-yellow:{SAFETY_YELLOW};--ss-asphalt:{ASPHALT};--ss-stripe:repeating-linear-gradient(-45deg,
  {SAFETY_YELLOW} 0 14px,{ASPHALT} 14px 28px)}}
.ss-header{{background:var(--ss-asphalt);color:#f5f5f4;border-radius:4px;overflow:hidden;margin-bottom:4px}}
.dark .ss-header{{border:1px solid #3f3f46}}
.ss-header-body{{padding:18px 22px 14px;display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 18px}}
.ss-brand{{font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:2.6rem;letter-spacing:.5px;
  color:var(--ss-yellow)!important;line-height:1;text-transform:uppercase}}
.ss-brand *{{color:var(--ss-yellow)!important}}.ss-tagline{{color:#e7e5e4!important}}
.ss-brand span{{display:inline-block;font-family:'Noto Sans Arabic',sans-serif;font-weight:700;
  text-transform:none;margin-left:14px;font-size:.85em}}
.ss-tagline{{flex:1 1 320px;color:#e7e5e4;font-size:1rem;line-height:1.45}}
.ss-badges{{display:flex;flex-wrap:wrap;gap:6px;padding:0 22px 14px}}
.ss-badge{{border:1px solid #57534e;color:#d6d3d1;font-size:.78rem;padding:2px 8px;border-radius:2px;
  text-transform:uppercase;letter-spacing:.6px;font-family:'Barlow Condensed',sans-serif;font-weight:600}}
.ss-stripe{{height:12px;background:var(--ss-stripe)}}
.ss-disclaimer{{font-size:.85rem;color:#57534e;margin:6px 2px 0}}
.dark .ss-disclaimer{{color:#a8a29e}}
.ss-card{{border:1px solid #d6d3d1;border-inline-start:10px solid #78716c;background:#fff;padding:14px 18px;
  border-radius:2px}}
.dark .ss-card{{background:#232325;border-color:#3f3f46}}
.ss-card.ss-good{{border-inline-start-color:#15803d}}.ss-card.ss-warn{{border-inline-start-color:#d97706}}
.ss-card.ss-bad{{border-inline-start-color:#b91c1c}}.ss-card.ss-na{{border-inline-start-color:#78716c}}
.ss-score{{font-family:'Barlow Condensed',sans-serif;font-size:3rem;font-weight:800;line-height:1;
  text-transform:uppercase;letter-spacing:.5px}}
.ss-score.ss-good{{color:#166534}}.ss-score.ss-warn{{color:#92400e}}
.ss-score.ss-bad{{color:#b91c1c}}.ss-score.ss-na{{color:#4b5563}}
.dark .ss-score.ss-good{{color:#4ade80}}.dark .ss-score.ss-warn{{color:#fbbf24}}
.dark .ss-score.ss-bad{{color:#f87171}}.dark .ss-score.ss-na{{color:#9ca3af}}
.ss-card .ss-action{{background:var(--ss-asphalt);color:var(--ss-yellow);font-family:'Barlow Condensed',sans-serif;
  font-weight:700;text-transform:uppercase;letter-spacing:.6px;font-size:1.15rem;padding:8px 12px;margin:10px 0 4px;
  border-radius:2px}}
.ss-card p{{margin:.5em 0 0;font-size:.92rem}}.ss-card ul{{margin:.4em 0 0;font-size:.88rem}}
.ss-alerts{{list-style:none;padding:0;margin:0;display:grid;gap:8px;font-size:1.2rem;line-height:1.5}}
.ss-alerts li{{text-align:start;background:var(--ss-yellow);color:#111;font-weight:700;padding:10px 14px;
  border-inline-start:8px solid var(--ss-asphalt);border-radius:2px}}
.ss-alerts li small{{font-weight:400}}
.ss-section-title{{font-family:'Barlow Condensed',sans-serif;font-weight:700;text-transform:uppercase;
  letter-spacing:.8px;font-size:1.05rem;margin:0;border-bottom:3px solid var(--ss-yellow);display:inline-block}}
.ss-run{{font-family:'Barlow Condensed',sans-serif!important;text-transform:uppercase;letter-spacing:1px;
  font-size:1.2rem!important;font-weight:700!important}}
[role=tab]{{font-family:'Barlow Condensed',sans-serif;text-transform:uppercase;letter-spacing:.8px;
  font-weight:600;font-size:1rem}}
input[type=checkbox],input[type=radio]{{min-width:24px;min-height:24px}}
"""
HEADER_HTML = (
    "<header class='ss-header'>"
    "<div class='ss-header-body'>"
    "<h1 class='ss-brand' style='margin:0'>Salamtak<span lang='ar' dir='rtl'>سلامتك</span></h1>"
    "<p class='ss-tagline' style='margin:0'>One photo of a site → hazards, a Safe / Needs attention / Unsafe verdict and short alerts "
    "in workers' languages. Built for Qatar: midday summer work ban and WBGT stop-work rule.</p></div>"
    "<div class='ss-badges' aria-label='About this tool'>"
    "<span class='ss-badge'>Computer vision</span><span class='ss-badge'>Rules engine</span>"
    "<span class='ss-badge'>No LLM</span><span class='ss-badge'>7 languages</span></div>"
    "<div class='ss-stripe' aria-hidden='true'></div></header>"
)
# Condensed signage font for headings (the body font comes from the theme).
HEAD_HTML = ("<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
             "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800"
             "&display=swap'>")


def build_theme() -> gr.themes.Base:
    """Gradio theme: stone neutrals, safety-yellow primary button with black text, square corners."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.yellow,
        neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_sm,
        font=[gr.themes.GoogleFont("Barlow"), gr.themes.GoogleFont("Noto Sans Arabic"), "sans-serif"],
    ).set(
        body_background_fill="#F3F1EC",
        body_background_fill_dark="#141414",
        button_primary_background_fill=SAFETY_YELLOW,
        button_primary_background_fill_hover="#FFD24D",
        button_primary_text_color="#111111",
        button_primary_background_fill_dark=SAFETY_YELLOW,
        button_primary_background_fill_hover_dark="#FFD24D",
        button_primary_text_color_dark="#111111",
        block_title_text_weight="600",
        block_label_text_weight="600",
    )


# ---------- time ----------
def exif_time(image: Image.Image) -> datetime | None:
    """Photo time from EXIF (camera local time, assumed Doha). None if missing or unreadable."""
    try:
        exif = image.getexif()
        raw = exif.get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
        return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S") if raw else None
    except Exception:  # malformed EXIF must never break the analysis
        log.warning("unreadable EXIF time; falling back", exc_info=True)
        return None


def _parse_override(value: str | None) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise gr.Error(f"Couldn't read the date/time '{text[:MAX_ECHO]}'. Use YYYY-MM-DD HH:MM, e.g. 2026-11-06 14:30.")


def resolve_time(override: str | None, image: Image.Image) -> tuple[datetime, str]:
    """Override > EXIF > now (Doha). Returns (naive Doha-local datetime, source label)."""
    manual = _parse_override(override)
    if manual:
        return manual, "manual override"
    from_exif = exif_time(image)
    if from_exif:
        return from_exif, "photo EXIF"
    return datetime.now(DOHA_TZ).replace(tzinfo=None, microsecond=0), "now (Doha time): no photo time found"


# ---------- presentation ----------
def action_html(text: str) -> str:
    """The 'alert the safety officer' strip under the verdict; empty when there is nothing to do."""
    if not text:
        return ""
    return f"<p class='ss-action'><span aria-hidden='true'>⚠ </span>{html.escape(text)}</p>"


def _score_html(result: Analysis, taken_at: datetime, source: str, secs: float, mode: str = "site") -> str:
    if result.assessed:
        v = verdict(result.hazards, mode)
        css, headline, aria, action = v.css, v.word, f"Safety verdict: {v.word}", action_html(v.action)
    else:
        css, headline, aria, action = "na", "Not assessed", "Safety verdict: not assessed", ""
    highs = sum(h.severity == "High" for h in result.hazards)
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in result.notes)
    return (
        f"<div role='status' aria-label='{aria}' class='ss-card ss-{css}'><div class='ss-score ss-{css}'>{headline}</div>"
        f"{action}"
        f"<p><strong>{highs} High · {len(result.hazards) - highs} Medium</strong> hazards<br>"
        f"Photo time: {taken_at:%Y-%m-%d %H:%M} ({html.escape(source)})<br>"
        f"Detector: {html.escape(DETECTOR.name)} · {secs:.2f} s</p>"
        f"{f'<ul>{notes}</ul>' if notes else ''}</div>"
    )


def _alert_item(a: alerts.Alert, requested: str) -> str:
    direction = "rtl" if a.language in RTL_LANGUAGES else "ltr"
    # Only the English fallback is labelled; translations show without an "unverified" tag
    # (alerts.json still records which ones a native speaker has checked).
    note = "English: no translation yet" if a.language != requested else ""
    tail = f" <small lang='en' dir='ltr'>({note})</small>" if note else ""
    return (f"<li lang='{a.language}' dir='{direction}'><span aria-hidden='true'>⚠️</span> "
            f"{html.escape(a.text)}{tail}</li>")


def _alerts_html(
    hazards: tuple[Hazard, ...] | list[Hazard], lang: str, table: alerts.AlertTable | None = None
) -> str:
    """One alert per distinct hazard id. ``table`` defaults to the validated alerts.json."""
    if not hazards:
        return "<p>No hazards found.</p>"
    items = "".join(_alert_item(a, lang) for a in alerts.alerts_for(hazards, lang, table))
    return f"<section aria-label='Safety alerts' aria-live='polite'><ul class='ss-alerts'>{items}</ul></section>"


# ---------- request handling ----------
def _validate(image: Image.Image | None, wbgt: float | None, lang: str, mode: str) -> tuple[str, str]:
    if image is None:
        raise gr.Error("Upload a photo or take one with the webcam first.")
    if wbgt is not None and not (WBGT_MIN <= wbgt <= WBGT_MAX):
        raise gr.Error(f"WBGT should be between {WBGT_MIN:.0f} and {WBGT_MAX:.0f} °C.")
    lang_key = lang if lang in LANGUAGES else alerts.DEFAULT_LANGUAGE
    mode_key = "home" if str(mode).lower().startswith("home") else "site"
    return lang_key, mode_key


def analyze(
    image: Image.Image | None, time_override: str | None, wbgt: float | None,
    outdoor: bool, lang: str, mode: str,
) -> tuple[str, str, Image.Image, list[list[str]]]:
    """Gradio handler: returns (score html, alerts html, annotated image, hazard table rows)."""
    lang_key, mode_key = _validate(image, wbgt, lang, mode)
    started = time.perf_counter()
    taken_at, source = resolve_time(time_override, image)  # EXIF is read before any resizing
    context = SceneContext(taken_at=taken_at, outdoor=bool(outdoor), wbgt_c=wbgt, mode=mode_key)
    try:
        prepared, result = analyze_image(DETECTOR, image, context)
        annotated = annotate(prepared, result.detections, result.hazards)
    except ImageTooLargeError as exc:
        raise gr.Error("That photo is too large. Please use one under 50 megapixels.") from exc
    except Exception as exc:
        log.exception("analysis failed")
        raise gr.Error("Something went wrong analysing this photo. Try another image.") from exc
    secs = time.perf_counter() - started
    log.info("analyzed: %d detections, %d hazards, score %d, assessed=%s, %.2fs",
             len(result.detections), len(result.hazards), result.score, result.assessed, secs)
    table = [[h.id, h.severity, h.reason] for h in result.hazards]
    return (_score_html(result, taken_at, source, secs, mode_key), _alerts_html(result.hazards, lang_key),
            annotated, table)


# ---------- UI ----------
EXAMPLES = [
    # Real Wikimedia photos (CC BY-SA, credited in examples/qatar_test/CREDITS.md); faces covered or turned away.
    # Doha crew at a summer noon with high WBGT: no hard hats + Qatar midday ban + WBGT stop-work.
    [str(EXAMPLES_DIR / "qatar_test" / "doha_roadside_crew_meeting.jpg"), "2026-07-15 12:30", 33.5, True, "ar", "Site"],
    # Workers walking away from camera on a winter morning: the helmet check on its own.
    [str(EXAMPLES_DIR / "qatar_test" / "dubai_workers_walking.jpg"), "2026-11-05 08:00", None, True, "en", "Site"],
    # Held-out CC BY 4.0 photos showing v2 classes, credited in examples/v2_classes/CREDITS.md.
    # Excavators working under overhead lines: power_line_contact.
    [str(EXAMPLES_DIR / "v2_classes" / "earthworks_dump_trucks.jpg"), "2026-11-05 08:00", None, True, "en", "Site"],
    # Mixer truck passing a pole, nobody on foot: machinery, vehicle and pole are found, verdict Safe.
    [str(EXAMPLES_DIR / "v2_classes" / "mixer_truck_utility_pole.jpg"), "2026-11-05 08:00", None, True, "en", "Site"],
    [str(EXAMPLES_DIR / "home_cords_floor.jpg"), "", None, False, "hi", "Home"],  # CC BY-SA, see examples/CREDITS.md
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Salamtak سلامتك") as demo:
        gr.HTML(HEADER_HTML + f"<p class='ss-disclaimer'>{html.escape(DISCLAIMER)}</p>")
        if LOAD_WARNING:
            gr.Markdown(f"> ⚠️ **{LOAD_WARNING}**")
        if PRIMARY.is_mock and OPEN_VOCAB:
            gr.Markdown("> **Demo mode:** people, helmets, vests and vehicles are fixed fake boxes from a mock "
                        "detector. Harness, scaffold, ladder, shade, water, cord and all Home-mode finds are real.")
        elif DETECTOR.is_mock:
            gr.Markdown("> **Demo mode:** the detector is a mock that returns fixed fake boxes for every photo.")
        with gr.Tab("Photo"):
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(label="Photo", type="pil", sources=["upload", "webcam"])
                    when = gr.Textbox(
                        label="Date/time override (optional, Doha time)",
                        info="Format YYYY-MM-DD HH:MM. Leave blank to use the photo's own time, else now.",
                        placeholder="2026-11-06 14:30", lines=1, max_lines=1,
                    )
                    wbgt = gr.Number(label="WBGT °C (optional)", value=None, minimum=WBGT_MIN, maximum=WBGT_MAX)
                    outdoor = gr.Checkbox(label="Outdoor", value=True)
                    lang = gr.Dropdown(
                        label="Alert language",
                        choices=[(name, code) for code, name in LANGUAGES.items()],
                        value="en",
                    )
                    mode = gr.Radio(label="Mode", choices=["Site", "Home"], value="Site")
                    run = gr.Button("Check safety", variant="primary", size="lg", elem_classes="ss-run")
                with gr.Column(scale=1):
                    gr.HTML("<h2 class='ss-section-title'>Site report</h2>")
                    score_box = gr.HTML()
                    alert_box = gr.HTML()
                    annotated = gr.Image(label="Annotated photo (HIGH = red, MED = orange); hazards listed below",
                                         type="pil", format="jpeg")
                    table = gr.Dataframe(headers=TABLE_HEADERS, label="Hazards", wrap=True, interactive=False)

            inputs = [image, when, wbgt, outdoor, lang, mode]
            outputs = [score_box, alert_box, annotated, table]
            run.click(analyze, inputs, outputs, concurrency_limit=CONCURRENCY)
            gr.Examples(EXAMPLES, inputs, outputs, fn=analyze, cache_examples=False, run_on_click=True)
        with gr.Tab("Live"):
            live.build_live_tab(LIVE_DETECTOR, LANGUAGES, _alerts_html)
    return demo.queue(max_size=QUEUE_SIZE, default_concurrency_limit=CONCURRENCY)


alerts.load_alerts()  # fail at boot, not mid-demo, if alerts.json is broken
demo = build_ui()

if __name__ == "__main__":
    demo.launch(css=CSS, theme=build_theme(), head=HEAD_HTML, max_file_size=MAX_FILE_SIZE)
