"""Core data types shared by detection, the rules engine, scoring and the UI.

Boxes are normalized to the image: ``(x1, y1, x2, y2)`` with
``0 <= x1 < x2 <= 1`` and ``0 <= y1 < y2 <= 1``; y grows downwards.
No model or I/O imports here, so everything that uses these types tests instantly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

Box = tuple[float, float, float, float]


class Severity(StrEnum):
    HIGH = "High"
    MEDIUM = "Medium"


class Mode(StrEnum):
    SITE = "site"
    HOME = "home"


# Hazard id -> severity. Mirrors the "Hazard rules" table in CLAUDE.md.
HAZARD_SEVERITY: dict[str, Severity] = {
    "no_helmet": Severity.HIGH,
    "machinery_zone": Severity.HIGH,
    "fall_no_harness": Severity.HIGH,
    "midday_ban": Severity.HIGH,
    "wbgt_stop": Severity.HIGH,
    "no_vest_near_vehicle": Severity.MEDIUM,
    "heat_welfare_gap": Severity.MEDIUM,
    "trip_cord": Severity.MEDIUM,
    "home_trip": Severity.MEDIUM,
    "home_slip": Severity.MEDIUM,
    "home_child_reach": Severity.MEDIUM,
    # v2 (PLAN-v2.md)
    "worker_down": Severity.HIGH,
    "under_suspended_load": Severity.HIGH,
    "power_line_contact": Severity.HIGH,
    "hot_work_fire": Severity.HIGH,
    "fall_into_opening": Severity.HIGH,
    "no_fire_extinguisher": Severity.MEDIUM,
    "exposed_rebar": Severity.MEDIUM,
}
HAZARD_IDS: tuple[str, ...] = tuple(HAZARD_SEVERITY)

# Trained detector classes (YOLO ids 0-6), from the Data section of CLAUDE.md.
UNIFIED_CLASSES: tuple[str, ...] = ("person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle")
# v2 detector adds ids 7-10. A model is valid if its class names equal
# UNIFIED_CLASSES (v1 weights) or MODEL_CLASSES_V2 (v2 weights).
EXTENDED_CLASSES: tuple[str, ...] = ("harness", "scaffold", "crane_hook", "utility_pole")
MODEL_CLASSES_V2: tuple[str, ...] = UNIFIED_CLASSES + EXTENDED_CLASSES


def normalize_label(label: str) -> str:
    """'Shade Tent' / 'shade-tent' / ' shade_tent ' -> 'shade_tent'."""
    return "_".join(label.strip().lower().replace("-", " ").replace("_", " ").split())


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def validate_box(box: Box) -> Box:
    """Return ``box`` as a float 4-tuple, or raise ValueError if it is not normalized."""
    if len(box) != 4:
        raise ValueError(f"box must have 4 values (x1, y1, x2, y2), got {box!r}")
    x1, y1, x2, y2 = (_finite(v, "box coordinate") for v in box)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"box must be normalized with x1<x2, y1<y2 in [0, 1], got {box!r}")
    return (x1, y1, x2, y2)


@dataclass(frozen=True)
class Detection:
    """One object found by a detector. ``label`` is normalized on construction."""

    label: str
    confidence: float
    box: Box
    source: str = "detector"  # "detector" (YOLO26) | "yoloe" | "mock"; not used by the rules

    def __post_init__(self) -> None:
        label = normalize_label(self.label)
        if not label:
            raise ValueError("detection label must not be empty")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "box", validate_box(self.box))

    @classmethod
    def from_model(cls, label: str, confidence: float, box: Box) -> Detection | None:
        """For raw detector output: clamp the box into [0, 1] and confidence into
        [0, 1]; return None for boxes with no area left, instead of raising."""
        x1, y1, x2, y2 = (min(1.0, max(0.0, _finite(v, "box coordinate"))) for v in box)
        if x2 <= x1 or y2 <= y1:
            return None
        conf = min(1.0, max(0.0, _finite(confidence, "confidence")))
        return cls(label, conf, (x1, y1, x2, y2))


@dataclass(frozen=True)
class Hazard:
    """A rule hit. ``subject`` names who it is about ("person:0", "head:1"),
    or is None for scene-wide hazards; scoring counts each (id, subject) once."""

    id: str
    severity: Severity
    reason: str
    boxes: tuple[Box, ...] = ()
    subject: str | None = None

    def __post_init__(self) -> None:
        expected = HAZARD_SEVERITY.get(self.id)
        if expected is None:
            raise ValueError(f"unknown hazard id {self.id!r}")
        if self.severity is not expected:
            raise ValueError(f"{self.id} must be {expected.value}, got {self.severity!r}")
        if not self.reason.strip() or "\n" in self.reason:
            raise ValueError("hazard reason must be one non-empty line")
        object.__setattr__(self, "boxes", tuple(validate_box(b) for b in self.boxes))

    @classmethod
    def of(
        cls,
        hazard_id: str,
        reason: str,
        boxes: tuple[Box, ...] = (),
        subject: str | None = None,
    ) -> Hazard:
        """Build a hazard, looking its severity up in HAZARD_SEVERITY."""
        if hazard_id not in HAZARD_SEVERITY:
            raise ValueError(f"unknown hazard id {hazard_id!r}")
        return cls(hazard_id, HAZARD_SEVERITY[hazard_id], reason, boxes, subject)


@dataclass(frozen=True)
class SceneContext:
    """Everything about the photo that is not a detection.

    ``taken_at``: naive datetimes are Doha local time (EXIF style); aware ones are converted.
    ``aspect_ratio``: image width / height, so distances are measured fairly on wide photos.
    """

    taken_at: datetime | None = None
    wbgt_c: float | None = None
    outdoor: bool = True
    mode: Mode = Mode.SITE
    aspect_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.wbgt_c is not None:
            object.__setattr__(self, "wbgt_c", _finite(self.wbgt_c, "WBGT"))
        ratio = _finite(self.aspect_ratio, "aspect_ratio")
        if ratio <= 0:
            raise ValueError(f"aspect_ratio must be positive, got {self.aspect_ratio!r}")
        object.__setattr__(self, "aspect_ratio", ratio)
        object.__setattr__(self, "mode", Mode(self.mode))
