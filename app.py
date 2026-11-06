"""SafeSite Qatar — Gradio Space entry point.

Photo -> detector -> rules engine -> score + annotated image + alerts. No LLM anywhere.
All scoring goes through safesite.pipeline.analyze_image; scripts/evaluate.py should call it too
so the pitch numbers measure exactly this app (it currently calls rules directly).
"""
from __future__ import annotations

import html
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gradio as gr
from PIL import Image

from safesite import alerts, detect
from safesite.annotate import annotate
from safesite.pipeline import Analysis, ImageTooLargeError, analyze_image
from safesite.types import Hazard, SceneContext

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

# ---- The one line to change when the trained model is on the Hub: ----
DETECTOR = detect.MockDetector()  # -> detect.YoloDetector()

log = logging.getLogger("safesite")
logging.basicConfig(level=logging.INFO)

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
# (min score, css class, status word)
SCORE_BANDS = ((80, "good", "Good"), (50, "warn", "Needs attention"), (0, "bad", "Unsafe"))
CSS = """
.ss-score{font-size:2.6rem;font-weight:700;line-height:1.1}
.ss-good{color:#166534}.ss-warn{color:#92400e}.ss-bad{color:#b91c1c}.ss-na{color:#4b5563}
.dark .ss-good{color:#4ade80}.dark .ss-warn{color:#fbbf24}.dark .ss-bad{color:#f87171}.dark .ss-na{color:#9ca3af}
.ss-alerts{font-size:1.25rem;line-height:1.8}.ss-alerts li{text-align:start}
input[type=checkbox],input[type=radio]{min-width:24px;min-height:24px}
"""


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
def _band(value: int) -> tuple[str, str]:
    return next((css, word) for floor, css, word in SCORE_BANDS if value >= floor)


def _score_html(result: Analysis, taken_at: datetime, source: str, secs: float) -> str:
    if result.assessed:
        css, word = _band(result.score)
        headline = f"{result.score}/100 · {word}"
        aria = f"Safety score {result.score} out of 100, {word}"
    else:
        css, headline, aria = "na", "Not assessed", "Safety score not assessed"
    highs = sum(h.severity == "High" for h in result.hazards)
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in result.notes)
    return (
        f"<div role='status' aria-label='{aria}'><div class='ss-score ss-{css}'>{headline}</div>"
        f"<p><strong>{highs} High · {len(result.hazards) - highs} Medium</strong> hazards<br>"
        f"Photo time: {taken_at:%Y-%m-%d %H:%M} ({html.escape(source)})<br>"
        f"Detector: {html.escape(DETECTOR.name)} · {secs:.2f} s</p>"
        f"{f'<ul>{notes}</ul>' if notes else ''}</div>"
    )


def _alert_item(a: alerts.Alert, requested: str) -> str:
    direction = "rtl" if a.language in RTL_LANGUAGES else "ltr"
    note = ""
    if a.language != requested:
        note = "English: no translation yet"
    elif not a.verified:
        note = "unverified translation"
    tail = f" <small lang='en' dir='ltr'>({note})</small>" if note else ""
    return (f"<li lang='{a.language}' dir='{direction}'><span aria-hidden='true'>⚠️</span> "
            f"{html.escape(a.text)}{tail}</li>")


def _alerts_html(hazards: tuple[Hazard, ...] | list[Hazard], lang: str) -> str:
    if not hazards:
        return "<p>No hazards found.</p>"
    unique: dict[str, alerts.Alert] = {}
    for h in hazards:
        unique.setdefault(h.id, alerts.alert_for(h.id, lang))
    items = "".join(_alert_item(a, lang) for a in unique.values())
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
    return _score_html(result, taken_at, source, secs), _alerts_html(result.hazards, lang_key), annotated, table


# ---------- UI ----------
EXAMPLES = [
    [str(EXAMPLES_DIR / "site_summer_noon.jpg"), "", 33.5, True, "ar", "Site"],
    [str(EXAMPLES_DIR / "site_winter_morning.jpg"), "", None, True, "en", "Site"],
    [str(EXAMPLES_DIR / "home_kitchen.jpg"), "", None, False, "hi", "Home"],
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SafeSite Qatar") as demo:
        gr.Markdown(
            "# SafeSite Qatar\n"
            "One photo of a site → hazards, a 0–100 safety score, and short alerts in workers' languages. "
            "Includes Qatar's midday summer work ban and the WBGT stop-work rule. No LLM in the loop.  \n"
            f"*{DISCLAIMER}*"
        )
        if DETECTOR.is_mock:
            gr.Markdown("> **Demo mode:** the detector is a mock that returns fixed fake boxes for every photo.")
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
                run = gr.Button("Check safety", variant="primary")
            with gr.Column(scale=1):
                score_box = gr.HTML()
                alert_box = gr.HTML()
                annotated = gr.Image(label="Annotated photo (HIGH = red, MED = orange); hazards listed below",
                                     type="pil", format="jpeg")
                table = gr.Dataframe(headers=TABLE_HEADERS, label="Hazards", wrap=True, interactive=False)

        inputs = [image, when, wbgt, outdoor, lang, mode]
        outputs = [score_box, alert_box, annotated, table]
        run.click(analyze, inputs, outputs, concurrency_limit=CONCURRENCY)
        gr.Examples(EXAMPLES, inputs, outputs, fn=analyze, cache_examples=False, run_on_click=True)
    return demo.queue(max_size=QUEUE_SIZE, default_concurrency_limit=CONCURRENCY)


alerts.load_alerts()  # fail at boot, not mid-demo, if alerts.json is broken
demo = build_ui()

if __name__ == "__main__":
    demo.launch(css=CSS, max_file_size=MAX_FILE_SIZE)
