"""Draw detections and hazards on the photo.

Severity is never colour-only: every hazard label is prefixed HIGH / MED, High boxes are thicker,
and frame-level hazards (heat rules, no boxes) get a border plus a text banner.
Red = High, orange = Medium, grey = plain detection.
"""
from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont

from safesite.types import Box, Detection, Hazard

Color = tuple[int, int, int]
SEVERITY_COLORS: dict[str, Color] = {"High": (220, 38, 38), "Medium": (245, 140, 0)}
LABEL_TEXT_COLORS: dict[str, Color] = {"High": (255, 255, 255), "Medium": (0, 0, 0)}  # >= 4.5:1 contrast
SEVERITY_TAGS = {"High": "HIGH", "Medium": "MED"}
DETECTION_COLOR: Color = (150, 150, 150)
FRAME_BORDER_FRACTION = 0.012


def _px(box: Box, w: int, h: int) -> tuple[int, int, int, int]:
    return (round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h))


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, severity: str,
           font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
    """Draw a filled label whose bottom-left corner is xy. Returns the label height."""
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    height = (bottom - top) + 4
    y = max(0, y - height)
    draw.rectangle((x, y, x + (right - left) + 6, y + height), fill=SEVERITY_COLORS[severity])
    draw.text((x + 3, y + 1), text, fill=LABEL_TEXT_COLORS[severity], font=font)
    return height


def annotate(image: Image.Image, detections: Sequence[Detection], hazards: Sequence[Hazard]) -> Image.Image:
    """Return a new RGB image with boxes and labels; the input image is not modified."""
    out = image.convert("RGB")  # convert always returns a new image
    w, h = out.size
    draw = ImageDraw.Draw(out)
    thin = max(1, w // 400)
    widths = {"High": max(4, w // 140), "Medium": max(2, w // 260)}
    font = ImageFont.load_default(size=max(12, w // 55))

    for d in detections:
        draw.rectangle(_px(d.box, w, h), outline=DETECTION_COLOR, width=thin)

    frame_level = [hz for hz in hazards if not hz.boxes]
    boxed = [z for z in hazards if z.boxes]
    # Boxes: Medium first so High outlines end up on top.
    for hz in sorted(boxed, key=lambda z: z.severity == "High"):
        for box in hz.boxes:
            draw.rectangle(_px(box, w, h), outline=SEVERITY_COLORS[hz.severity], width=widths[hz.severity])
    # Labels: High first; hazards sharing an anchor box stack downwards instead of overlapping.
    stacked: dict[tuple[int, int], int] = {}
    for hz in sorted(boxed, key=lambda z: z.severity != "High"):
        anchor = _px(hz.boxes[0], w, h)[:2]
        offset = stacked.get(anchor, 0)
        stacked[anchor] = offset + _label(draw, (anchor[0], anchor[1] + offset), f"{SEVERITY_TAGS[hz.severity]}: {hz.id}",
                                          hz.severity, font)

    if frame_level:
        worst = "High" if any(z.severity == "High" for z in frame_level) else "Medium"
        border = max(4, round(min(w, h) * FRAME_BORDER_FRACTION))
        draw.rectangle((0, 0, w - 1, h - 1), outline=SEVERITY_COLORS[worst], width=border)
        banner = "  ".join(f"{SEVERITY_TAGS[z.severity]}: {z.id}" for z in frame_level)
        _label(draw, (border + 2, h - border - 2), banner, worst, font)
    return out
