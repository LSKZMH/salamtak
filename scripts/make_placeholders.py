"""Draw 3 synthetic placeholder example photos (no real people) that line up with the mock boxes.

Replace with our own consented photos later. Usage:  python scripts/make_placeholders.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safesite.detect import MOCK_HOME, MOCK_SITE  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "examples"
W, H = 1280, 853
EXIF_IFD, DATETIME, DATETIME_ORIGINAL = 0x8769, 0x0132, 0x9003


def _px(box):
    return (box[0] * W, box[1] * H, box[2] * W, box[3] * H)


def _box(label: str, mock, nth: int = 0):
    return _px([d for d in mock if d.label == label][nth].box)


def _worker(draw: ImageDraw.ImageDraw, body, helmet: bool, vest: bool) -> None:
    x1, y1, x2, y2 = body
    cx, bw, bh = (x1 + x2) / 2, x2 - x1, y2 - y1
    head_r = bw * 0.28
    head_cy = y1 + head_r + bh * 0.02
    draw.ellipse((cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r), fill=(120, 90, 70))
    if helmet:
        draw.pieslice((cx - head_r * 1.2, head_cy - head_r * 1.4, cx + head_r * 1.2, head_cy + head_r * 0.6),
                      180, 360, fill=(250, 204, 21))
    torso = (x1 + bw * 0.15, y1 + bh * 0.2, x2 - bw * 0.15, y1 + bh * 0.55)
    draw.rectangle(torso, fill=(249, 115, 22) if vest else (59, 130, 246))
    draw.rectangle((x1 + bw * 0.2, y1 + bh * 0.55, cx - 3, y2), fill=(55, 65, 81))
    draw.rectangle((cx + 3, y1 + bh * 0.55, x2 - bw * 0.2, y2), fill=(55, 65, 81))


def _stamp(draw: ImageDraw.ImageDraw, text: str) -> None:
    font = ImageFont.load_default(size=34)
    draw.text((24, 20), "PLACEHOLDER - synthetic image", fill=(255, 255, 255), font=font,
              stroke_width=3, stroke_fill=(0, 0, 0))
    draw.text((24, 64), text, fill=(255, 255, 255), font=ImageFont.load_default(size=24),
              stroke_width=2, stroke_fill=(0, 0, 0))


def site(sky: tuple[int, int, int], caption: str) -> Image.Image:
    img = Image.new("RGB", (W, H), sky)
    d = ImageDraw.Draw(img)
    d.rectangle((0, H * 0.6, W, H), fill=(214, 186, 140))  # sand
    d.rectangle((W * 0.25, H * 0.15, W * 0.45, H * 0.6), outline=(120, 120, 120), width=6)  # unfinished frame
    tx1, ty1, tx2, ty2 = _box("vehicle", MOCK_SITE)
    d.rectangle((tx1, ty1 + (ty2 - ty1) * 0.15, tx2, ty2 - (ty2 - ty1) * 0.2), fill=(234, 179, 8))
    d.rectangle((tx1, ty1, tx1 + (tx2 - tx1) * 0.35, ty2 - (ty2 - ty1) * 0.2), fill=(202, 138, 4))
    for cx in (tx1 + (tx2 - tx1) * 0.2, tx2 - (tx2 - tx1) * 0.2):
        r = (ty2 - ty1) * 0.14
        d.ellipse((cx - r, ty2 - 2 * r, cx + r, ty2), fill=(31, 41, 55))
    _worker(d, _box("person", MOCK_SITE, 0), helmet=False, vest=True)
    _worker(d, _box("person", MOCK_SITE, 1), helmet=True, vest=False)
    _stamp(d, caption)
    return img


def home() -> Image.Image:
    img = Image.new("RGB", (W, H), (241, 236, 228))
    d = ImageDraw.Draw(img)
    d.rectangle((0, H * 0.7, W, H), fill=(180, 150, 120))  # floor
    d.rectangle((W * 0.5, H * 0.62, W * 0.95, H * 0.7), fill=(120, 85, 60))  # low table
    kx1, ky1, kx2, ky2 = _box("knife", MOCK_HOME)
    d.polygon([(kx1, ky1 + 6), (kx2, (ky1 + ky2) / 2), (kx1, ky2 - 2)], fill=(200, 200, 210))
    cx1, cy1, cx2, cy2 = _box("extension cord", MOCK_HOME)
    pts = [(cx1 + (cx2 - cx1) * t / 10, cy1 + (cy2 - cy1) * (0.5 + 0.4 * (-1) ** t)) for t in range(11)]
    d.line(pts, fill=(20, 20, 20), width=8)
    _stamp(d, "Home: cord across floor, knife on low table. No EXIF time.")
    return img


def _exif(timestamp: str) -> Image.Exif:
    exif = Image.Exif()
    exif[DATETIME] = timestamp
    exif.get_ifd(EXIF_IFD)[DATETIME_ORIGINAL] = timestamp
    return exif


def main() -> None:
    OUT.mkdir(exist_ok=True)
    site((135, 196, 235), "Site, EXIF 2026-07-15 12:30 (summer midday ban)").save(
        OUT / "site_summer_noon.jpg", quality=88, exif=_exif("2026:07:15 12:30:00"))
    site((186, 214, 235), "Site, EXIF 2026-11-06 08:15 (no heat ban)").save(
        OUT / "site_winter_morning.jpg", quality=88, exif=_exif("2026:11:06 08:15:00"))
    home().save(OUT / "home_kitchen.jpg", quality=88)
    print(f"wrote 3 placeholders to {OUT}")


if __name__ == "__main__":
    main()
