"""Hazard rules (pure functions, no model or I/O imports).

STUB written by the app chat (Chat 4) so the UI runs end to end. Chat 3 owns this file.
Implemented here: no_helmet, machinery_zone, no_vest_near_vehicle, midday_ban, wbgt_stop,
and simple home rules. Not yet: see NOT_IMPLEMENTED (they need YOLOE detections).

Public entry point used by the pipeline:  find_hazards(detections, context) -> list[Hazard]

Geometry: boxes are normalized, so x is scaled by the image aspect ratio (context.aspect_ratio)
and every distance is expressed in units of image height. Gaps are Euclidean.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import hypot

from safesite.heat import heat_hazards
from safesite.types import (
    MIN_CONFIDENCE,
    WORKER_LABELS,
    Box,
    Detection,
    Hazard,
    SceneContext,
)

HEAD_FRACTION = 0.25  # top quarter of a person box is the head region
ZONE_FACTOR = 0.5  # "near" = gap no more than 0.5 x person height
TORSO_TO_PERSON = 3.0  # a lone no_vest (torso) box is ~1/3 of a person's height
DUPLICATE_IOU = 0.5  # same-label boxes overlapping this much are one object
LOWER_THIRD_Y = 2 / 3
CHILD_REACH_Y = 0.5  # object centre below mid-frame = within a child's reach

NOT_IMPLEMENTED: tuple[str, ...] = ("fall_no_harness", "heat_welfare_gap", "trip_cord")
HOME_TRIP_LABELS = frozenset({"extension cord", "cord", "hose"})
HOME_SLIP_LABELS = frozenset({"wet floor"})
HOME_REACH_LABELS = frozenset({"knife", "cleaning chemicals", "chemicals"})


# ---------- geometry helpers ----------
def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: Box, b: Box) -> float:
    inter = _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _centre_inside(outer: Box, inner: Box) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _gap(a: Box, b: Box, aspect: float) -> float:
    """Euclidean gap between two boxes in image-height units (0 if they overlap)."""
    dx = max(0.0, b[0] - a[2], a[0] - b[2]) * aspect
    dy = max(0.0, b[1] - a[3], a[1] - b[3])
    return hypot(dx, dy)


def _height(box: Box) -> float:
    return box[3] - box[1]


def _head(person: Box) -> Box:
    return (person[0], person[1], person[2], person[1] + HEAD_FRACTION * _height(person))


def _near(box: Box, other: Box, ref_height: float, aspect: float) -> bool:
    return _gap(box, other, aspect) <= ZONE_FACTOR * ref_height


def _of(dets: Sequence[Detection], labels: frozenset[str] | set[str]) -> list[Detection]:
    return [d for d in dets if d.label in labels]


def _dedupe(dets: Sequence[Detection]) -> list[Detection]:
    """Drop same-label duplicates (IoU above DUPLICATE_IOU), keeping the most confident."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: -x.confidence):
        if not any(k.label == d.label and _iou(k.box, d.box) > DUPLICATE_IOU for k in kept):
            kept.append(d)
    return kept


def _owner(part: Detection, persons: Sequence[Detection]) -> Detection | None:
    return next((p for p in persons if _centre_inside(p.box, part.box)), None)


# ---------- site rules ----------
def _no_helmet(persons: list[Detection], dets: Sequence[Detection]) -> list[Hazard]:
    """At most one hazard per worker. A bare head inside a person box belongs to that person;
    a bare head with no person box (datasets label heads far more than persons) counts on its own;
    a person with no helmet over their head region also counts."""
    bare_heads = _dedupe(_of(dets, {"no_helmet"}))
    helmets = _of(dets, {"helmet"})
    hazards: list[Hazard] = []
    claimed: set[int] = set()
    for p in persons:
        own = [i for i, h in enumerate(bare_heads) if i not in claimed and _centre_inside(p.box, h.box)]
        if own:
            claimed.update(own)
            head = bare_heads[own[0]]
            hazards.append(Hazard("no_helmet", "High",
                                  f"Worker without a helmet (conf {head.confidence:.2f}).", (p.box, head.box)))
        elif not any(_overlaps(_head(p.box), h.box) for h in helmets):
            hazards.append(Hazard("no_helmet", "High", "Person with no helmet visible on their head.", (p.box,)))
    hazards.extend(
        Hazard("no_helmet", "High", f"Worker without a helmet (conf {h.confidence:.2f}).", (h.box,))
        for i, h in enumerate(bare_heads) if i not in claimed
    )
    return hazards


def _machinery_zone(persons: list[Detection], dets: Sequence[Detection], aspect: float) -> list[Hazard]:
    machines = _of(dets, {"machinery", "vehicle"})
    hazards = []
    for p in persons:
        close = next((m for m in machines if _near(p.box, m.box, _height(p.box), aspect)), None)
        if close:
            hazards.append(Hazard("machinery_zone", "High",
                                  f"Person inside the danger zone of a {close.label}.", (p.box, close.box)))
    return hazards


def _no_vest_near_vehicle(persons: list[Detection], dets: Sequence[Detection], aspect: float) -> list[Hazard]:
    """Measured from the worker's person box (same zone as machinery_zone), one hazard per worker."""
    vehicles = _of(dets, {"vehicle"})
    hazards = []
    seen_owners: set[int] = set()
    for nv in _dedupe(_of(dets, {"no_vest"})):
        owner = _owner(nv, persons)
        if owner is not None:
            if id(owner) in seen_owners:
                continue
            seen_owners.add(id(owner))
        ref_box = owner.box if owner else nv.box
        ref_height = _height(owner.box) if owner else _height(nv.box) * TORSO_TO_PERSON
        close = next((v for v in vehicles if _near(ref_box, v.box, ref_height, aspect)), None)
        if close:
            hazards.append(Hazard("no_vest_near_vehicle", "Medium",
                                  "Worker without a hi-vis vest close to a vehicle.", (ref_box, close.box)))
    return hazards


# ---------- home rules ----------
def _home_rules(dets: Sequence[Detection]) -> list[Hazard]:
    hazards = []
    for d in _of(dets, HOME_TRIP_LABELS):
        if d.box[3] >= LOWER_THIRD_Y:
            hazards.append(Hazard("home_trip", "Medium", f"{d.label.capitalize()} lying across the floor.", (d.box,)))
    for d in _of(dets, HOME_SLIP_LABELS):
        hazards.append(Hazard("home_slip", "Medium", "Wet floor: slip risk.", (d.box,)))
    for d in _of(dets, HOME_REACH_LABELS):
        if (d.box[1] + d.box[3]) / 2 >= CHILD_REACH_Y:
            hazards.append(Hazard("home_child_reach", "Medium", f"{d.label.capitalize()} at child height.", (d.box,)))
    return hazards


# ---------- entry point ----------
def find_hazards(detections: Sequence[Detection], context: SceneContext) -> list[Hazard]:
    """Apply the CLAUDE.md hazard table to one photo's detections. Pure and deterministic."""
    dets = [d for d in detections if d.confidence >= MIN_CONFIDENCE]
    if context.mode == "home":
        return _home_rules(dets)
    persons = _dedupe(_of(dets, {"person"}))
    workers_present = any(d.label in WORKER_LABELS for d in dets)
    return [
        *_no_helmet(persons, dets),
        *_machinery_zone(persons, dets, context.aspect_ratio),
        *_no_vest_near_vehicle(persons, dets, context.aspect_ratio),
        *heat_hazards(context.taken_at, context.outdoor, context.wbgt, workers_present),
    ]
