"""One photo -> Analysis. Shared by app.py and scripts/evaluate.py so both score photos identically.

No Gradio imports here: callers turn exceptions into UI messages.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PIL import Image, ImageOps

from safesite import rules, score
from safesite.heat import WBGT_STOP_C
from safesite.types import EQUIPMENT_LABELS, MIN_CONFIDENCE, WORKER_LABELS, Detection, Hazard, Mode, SceneContext

MAX_SIDE = 1280  # downscale before detect/annotate: ~10 s -> ~0.2 s per 12 MP photo on CPU
MAX_INPUT_PIXELS = 50_000_000  # refuse anything bigger (decompression-bomb guard)
Image.MAX_IMAGE_PIXELS = MAX_INPUT_PIXELS

Detector = Callable[[Image.Image, Mode], list[Detection]]
log = logging.getLogger(__name__)


class ImageTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class Analysis:
    detections: tuple[Detection, ...]  # what the rules saw (confidence >= MIN_CONFIDENCE)
    hazards: tuple[Hazard, ...]
    score: int
    assessed: bool  # False = the photo could not be judged; never show it as "safe"
    notes: tuple[str, ...]  # plain-English caveats to show next to the score


def prepare_image(image: Image.Image) -> Image.Image:
    """Upright, RGB, at most MAX_SIDE px on the long side. Returns a new image."""
    if image.width * image.height > MAX_INPUT_PIXELS:
        raise ImageTooLargeError(f"{image.width}x{image.height} is too large")
    upright = ImageOps.exif_transpose(image).convert("RGB")  # no-op if Gradio already rotated it
    upright.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.BILINEAR, reducing_gap=2.0)
    return upright


def _notes(dets: Sequence[Detection], ignored: int, context: SceneContext, detector: object) -> tuple[bool, list[str]]:
    labels = {d.label for d in dets}
    notes: list[str] = []
    assessed = True
    if context.mode == "home" and not getattr(detector, "supports_home", False):
        assessed = False
        notes.append("Home mode needs the open-vocabulary detector, which isn't enabled yet.")
    elif context.mode == "site" and not labels & (WORKER_LABELS | EQUIPMENT_LABELS):
        assessed = False
        notes.append("No people or equipment detected, so this photo can't be assessed.")
    elif context.mode == "home" and not dets:
        notes.append("Nothing hazardous recognised; the home checks cover cords, wet floors, knives and chemicals.")
    if context.wbgt is not None and context.wbgt > WBGT_STOP_C and not labels & WORKER_LABELS:
        notes.append(f"WBGT {context.wbgt:.1f} °C is above the {WBGT_STOP_C} °C stop-work limit.")
    if ignored:
        notes.append(f"{ignored} low-confidence detection(s) ignored.")
    pending = getattr(rules, "NOT_IMPLEMENTED", ())
    if context.mode == "site" and pending:
        notes.append("Not checked yet: " + ", ".join(p.replace("_", " ") for p in pending) + ".")
    return assessed, notes


def analyze_image(detector: Detector, image: Image.Image, context: SceneContext) -> tuple[Image.Image, Analysis]:
    """Run detector + rules + score on one photo. Returns (prepared image, analysis)."""
    prepared = prepare_image(image)
    context = dataclasses.replace(context, aspect_ratio=prepared.width / prepared.height)
    raw = detector(prepared, context.mode)
    kept = tuple(d for d in raw if d.confidence >= MIN_CONFIDENCE)
    hazards = tuple(rules.find_hazards(kept, context))
    assessed, notes = _notes(kept, len(raw) - len(kept), context, detector)
    if not assessed:
        log.warning("photo not assessed (%d raw detections, mode=%s)", len(raw), context.mode)
    return prepared, Analysis(kept, hazards, score.safety_score(hazards), assessed, tuple(notes))
