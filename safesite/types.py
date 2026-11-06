"""Shared dataclasses and constants for the SafeSite pipeline.

STUB written by the app chat (Chat 4) so the UI runs end to end.
Chat 3 owns this file: replace freely, but keep these names/fields or update app.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

# Normalized (x1, y1, x2, y2), each in [0, 1], origin top-left.
Box = tuple[float, float, float, float]
Severity = Literal["High", "Medium"]
Mode = Literal["site", "home"]

# Single source of truth for the detector -> rules contract (detect.py, rules.py, evaluate.py).
UNIFIED_CLASSES: tuple[str, ...] = ("person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle")
WORKER_LABELS = frozenset({"person", "helmet", "no_helmet", "vest", "no_vest"})
EQUIPMENT_LABELS = frozenset({"machinery", "vehicle"})
MIN_CONFIDENCE = 0.35  # detections below this are ignored by the rules and not drawn


@dataclass(frozen=True)
class Detection:
    label: str  # unified class name, e.g. "person", "no_helmet", or a YOLOE prompt
    confidence: float
    box: Box
    source: str = "detector"  # "detector" (YOLO26) | "yoloe" | "mock"


@dataclass(frozen=True)
class Hazard:
    id: str  # hazard id from the CLAUDE.md table, e.g. "no_helmet"
    severity: Severity
    reason: str  # one plain-English line referencing what was seen
    boxes: tuple[Box, ...] = ()  # empty for frame-level hazards (heat rules)


@dataclass(frozen=True)
class SceneContext:
    taken_at: datetime  # local Doha time of the photo
    outdoor: bool = True
    wbgt: float | None = None  # degrees C, optional
    mode: Mode = "site"
    aspect_ratio: float = 1.0  # image width / height, so distances can be measured in true proportions
