"""Hazard rules: normalized detections + scene context -> hazards.

Pure functions, no model or I/O imports. The "Hazard rules" table in CLAUDE.md is
the source of truth; keep this module, tests/test_rules.py and that table in sync.
Every hazard carries a one-line plain-English ``reason`` naming what was seen.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Sequence

from safesite import heat
from safesite.types import MODEL_CLASSES_V2, Box, Detection, Hazard, Mode, SceneContext, Severity

# --- Thresholds ---------------------------------------------------------------
MIN_CONFIDENCE = 0.35  # cut-off for the trained detector (YOLO26s)
OPEN_VOCAB_MIN_CONFIDENCE = 0.25  # cut-off for the open-vocabulary finder (YOLOE-26)
# Per-label YOLOE cut-offs (unlisted labels use OPEN_VOCAB_MIN_CONFIDENCE).
# Copy recommended values here from Chat 9's docs/prompt_bench.md.
OPEN_VOCAB_CUTOFFS: dict[str, float] = {}
HEAD_REGION_FRACTION = 0.25  # top quarter of a person box is their head region
MACHINERY_ZONE_FACTOR = 0.5  # danger zone: within 0.5 x person height of machinery/vehicle
FALL_ZONE_FACTOR = 0.25  # "near" a scaffold/ladder: within 0.25 x person height
WALKWAY_TOP_Y = 2 / 3  # lower third of the frame (y > 2/3) is where people walk
CHILD_REACH_MIN_Y = 0.5  # objects centred in the lower half of the frame are in a child's reach
# v2 (starting values from chat-8-rules-v2.md; tune only with evidence)
WORKER_DOWN_MIN_CONF = 0.5  # only a confident person box may count as lying down
WORKER_DOWN_RATIO = 1.4  # real width / height at or above this = lying down
WORKER_DOWN_MIN_HEIGHT = 0.04  # ignore tiny far-away boxes (fraction of image height)
BORDER_MARGIN = 0.01  # a box this close to an edge is cut off (a cut-off torso looks wide)
WORKER_DOWN_MAX_INSIDE = 0.6  # a person box >= 60% inside a machine/vehicle is its driver
SUSPENDED_LOAD_FACTOR = 1.0  # crane hook within 1.0 x person height to the side
SUSPENDED_LOAD_TOP_FRACTION = 0.1  # hook bottom may reach 0.1 x person height below their top
POWER_LINE_FACTOR = 0.5  # machinery within 0.5 x its own height of a power pole / line
HOT_WORK_FACTOR = 2.0  # welding within 2.0 x gas-cylinder height of the cylinder

# --- Labels (normalized: lowercase, underscores) ------------------------------
PERSON, HELMET, NO_HELMET = "person", "helmet", "no_helmet"
VEST, NO_VEST, MACHINERY, VEHICLE = "vest", "no_vest", "machinery", "vehicle"
HARNESS, SCAFFOLD, LADDER = "harness", "scaffold", "ladder"
SHADE_TENT, WATER_COOLER, CORD, HOSE = "shade_tent", "water_cooler", "cord", "hose"
WET_FLOOR, KNIFE, CHEMICALS = "wet_floor", "knife", "chemicals"
# Raw YOLOE prompt names (e.g. the mock's "extension cord") normalize to these, so accept both.
EXTENSION_CORD, CLEANING_CHEMICALS = "extension_cord", "cleaning_chemicals"
CORD_LABELS = frozenset({CORD, EXTENSION_CORD})
CHILD_REACH_LABELS = frozenset({KNIFE, CHEMICALS, CLEANING_CHEMICALS})
# v2 labels: crane_hook and utility_pole come from the trained detector and YOLOE;
# the rest only from YOLOE prompts (see the label contract in PLAN-v2.md).
CRANE_HOOK, UTILITY_POLE, POWER_LINE = "crane_hook", "utility_pole", "power_line"
WELDING, FIRE, GAS_CYLINDER = "welding", "fire", "gas_cylinder"
FIRE_EXTINGUISHER, REBAR = "fire_extinguisher", "rebar"
FLOOR_OPENING, TRENCH = "floor_opening", "trench"

OPEN_VOCAB_LABELS = frozenset(
    {HARNESS, SCAFFOLD, LADDER, SHADE_TENT, WATER_COOLER, HOSE, WET_FLOOR}
    | CORD_LABELS
    | CHILD_REACH_LABELS
    | {CRANE_HOOK, UTILITY_POLE, POWER_LINE, WELDING, FIRE, GAS_CYLINDER}
    | {FIRE_EXTINGUISHER, REBAR, FLOOR_OPENING, TRENCH}
)
TRAINED_LABELS = frozenset(MODEL_CLASSES_V2)  # what the trained detector (v1 or v2) can output
YOLOE_SOURCE, DETECTOR_SOURCE = "yoloe", "detector"  # Detection.source values
DISPLAY_NAME = {
    MACHINERY: "machine",
    NO_HELMET: "bare head",
    WET_FLOOR: "wet floor",
    CRANE_HOOK: "crane hook",
    UTILITY_POLE: "power pole",
    POWER_LINE: "power line",
    GAS_CYLINDER: "gas cylinder",
    FIRE_EXTINGUISHER: "fire extinguisher",
    FLOOR_OPENING: "floor opening",
}
IRREGULAR_PLURALS = {"knife": "knives"}  # display name -> plural, for reason strings


# --- Geometry and filtering helpers -------------------------------------------
def _open_vocab_cutoff(label: str) -> float:
    return OPEN_VOCAB_CUTOFFS.get(label, OPEN_VOCAB_MIN_CONFIDENCE)


def confidence_cutoff(detection: Detection) -> float:
    """YOLOE output -> its per-label open-vocab cut-off; trained-detector output of a
    trained class -> MIN_CONFIDENCE; anything else (mocks, hand-built detections) by label."""
    label = detection.label
    if detection.source == YOLOE_SOURCE:
        return _open_vocab_cutoff(label)
    if detection.source == DETECTOR_SOURCE and label in TRAINED_LABELS:
        return MIN_CONFIDENCE
    return _open_vocab_cutoff(label) if label in OPEN_VOCAB_LABELS else MIN_CONFIDENCE


def confident(detections: Sequence[Detection]) -> list[Detection]:
    return [d for d in detections if d.confidence >= confidence_cutoff(d)]


def with_labels(detections: Sequence[Detection], labels: Collection[str]) -> list[Detection]:
    return [d for d in detections if d.label in labels]


def overlaps(a: Box, b: Box) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def gap(a: Box, b: Box, aspect_ratio: float = 1.0) -> float:
    """Shortest distance between two boxes in image heights (0 when they touch or overlap)."""
    dx = max(0.0, b[0] - a[2], a[0] - b[2]) * aspect_ratio
    dy = max(0.0, b[1] - a[3], a[1] - b[3])
    return math.hypot(dx, dy)


def height(box: Box) -> float:
    return box[3] - box[1]


def head_region(box: Box) -> Box:
    return (box[0], box[1], box[2], box[1] + HEAD_REGION_FRACTION * height(box))


def centre_y(box: Box) -> float:
    return (box[1] + box[3]) / 2


def width_to_height(box: Box, aspect_ratio: float = 1.0) -> float:
    """Box width / height in real pixels (same aspect_ratio conversion as ``gap``)."""
    return (box[2] - box[0]) * aspect_ratio / height(box)


def horizontal_gap(a: Box, b: Box, aspect_ratio: float = 1.0) -> float:
    """Side-to-side distance between two boxes in image heights (0 when they overlap in x)."""
    return max(0.0, b[0] - a[2], a[0] - b[2]) * aspect_ratio


def touches_border(box: Box, margin: float = BORDER_MARGIN) -> bool:
    return box[0] <= margin or box[1] <= margin or box[2] >= 1 - margin or box[3] >= 1 - margin


def inside_fraction(inner: Box, outer: Box) -> float:
    """Share of ``inner``'s area that lies inside ``outer`` (0-1)."""
    w = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    h = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return (w * h) / ((inner[2] - inner[0]) * height(inner))


def _side(box: Box) -> str:
    centre_x = (box[0] + box[2]) / 2
    if centre_x < 1 / 3:
        return "on the left"
    return "on the right" if centre_x > 2 / 3 else "in the middle"


def within_zone(
    person: Detection, others: Sequence[Detection], factor: float, aspect_ratio: float
) -> list[tuple[Detection, float]]:
    """Objects within ``factor`` x person height of the person, nearest first, with their gap."""
    limit = factor * height(person.box)
    hits = [(o, gap(person.box, o.box, aspect_ratio)) for o in others]
    return sorted([(o, g) for o, g in hits if g <= limit], key=lambda hit: hit[1])


def _name(label: str) -> str:
    return DISPLAY_NAME.get(label, label.replace("_", " "))


def _where(person: Detection, nearest: tuple[Detection, float]) -> str:
    obj, distance = nearest
    if distance == 0:
        return f"overlaps a {_name(obj.label)}"
    return f"is {distance / height(person.box):.2f}x their height from a {_name(obj.label)}"


def _subject(index: int) -> str:
    return f"person:{index}"


def count_workers(detections: Sequence[Detection]) -> int:
    """Person boxes plus helmet / bare-head boxes that sit outside every person box."""
    people = with_labels(detections, {PERSON})
    heads = with_labels(detections, {HELMET, NO_HELMET})
    lone_heads = [h for h in heads if not any(overlaps(h.box, p.box) for p in people)]
    return len(people) + len(lone_heads)


# --- Site rules ----------------------------------------------------------------
def _bare_heads_of(
    person: Detection, people: Sequence[Detection], bare: Sequence[Detection]
) -> list[Detection]:
    """no_helmet boxes on this person's head region, plus boxes that overlap this
    person's body but nobody's head region (so no bare-head box is ever dropped)."""
    head = head_region(person.box)
    heads = [head_region(p.box) for p in people]
    return [
        b
        for b in bare
        if overlaps(b.box, head)
        or (overlaps(b.box, person.box) and not any(overlaps(b.box, h) for h in heads))
    ]


def _person_no_helmet(
    index: int, person: Detection, helmets: Sequence[Detection], bare_on_head: Sequence[Detection]
) -> Hazard | None:
    head = head_region(person.box)
    if bare_on_head:
        best = max(b.confidence for b in bare_on_head)
        return Hazard.of(
            "no_helmet",
            f"Person {index + 1}: bare head detected (no_helmet, confidence {best:.2f}).",
            (person.box, *(b.box for b in bare_on_head)),
            _subject(index),
        )
    if any(overlaps(h.box, head) for h in helmets):
        return None
    return Hazard.of(
        "no_helmet",
        f"Person {index + 1}: no helmet seen on their head.",
        (person.box,),
        _subject(index),
    )


def no_helmet(detections: Sequence[Detection]) -> list[Hazard]:
    """Person with a no_helmet box, or no helmet box overlapping their head region.

    A no_helmet box outside every person box is itself a worker's head (the main
    hard-hat dataset has no person class), so it raises one hazard of its own.
    """
    people = with_labels(detections, {PERSON})
    helmets = with_labels(detections, {HELMET})
    bare = with_labels(detections, {NO_HELMET})
    found = [
        _person_no_helmet(i, p, helmets, _bare_heads_of(p, people, bare))
        for i, p in enumerate(people)
    ]
    hazards = [h for h in found if h is not None]
    for k, head in enumerate(bare):
        if any(overlaps(head.box, p.box) for p in people):
            continue
        hazards.append(
            Hazard.of(
                "no_helmet",
                f"Bare head detected with no helmet (no_helmet, confidence {head.confidence:.2f}).",
                (head.box,),
                f"head:{k}",
            )
        )
    return hazards


def machinery_zone(detections: Sequence[Detection], aspect_ratio: float = 1.0) -> list[Hazard]:
    """Person overlaps or is within 0.5 x their height of a machinery/vehicle box."""
    machines = with_labels(detections, {MACHINERY, VEHICLE})
    hazards = []
    for i, person in enumerate(with_labels(detections, {PERSON})):
        close = within_zone(person, machines, MACHINERY_ZONE_FACTOR, aspect_ratio)
        if close:
            hazards.append(
                Hazard.of(
                    "machinery_zone",
                    f"Person {i + 1} {_where(person, close[0])} "
                    f"(danger zone {MACHINERY_ZONE_FACTOR}x person height).",
                    (person.box, *(m.box for m, _ in close)),
                    _subject(i),
                )
            )
    return hazards


def fall_no_harness(detections: Sequence[Detection], aspect_ratio: float = 1.0) -> list[Hazard]:
    """Person on/near a scaffold or ladder with no harness overlapping them."""
    climbables = with_labels(detections, {SCAFFOLD, LADDER})
    harnesses = with_labels(detections, {HARNESS})
    hazards = []
    for i, person in enumerate(with_labels(detections, {PERSON})):
        close = within_zone(person, climbables, FALL_ZONE_FACTOR, aspect_ratio)
        if not close or any(overlaps(h.box, person.box) for h in harnesses):
            continue
        nearest, distance = close[0]
        where = "on" if distance == 0 else "near"
        hazards.append(
            Hazard.of(
                "fall_no_harness",
                f"Person {i + 1} is {where} a {_name(nearest.label)} with no harness seen on them.",
                (person.box, *(c.box for c, _ in close)),
                _subject(i),
            )
        )
    return hazards


def no_vest_near_vehicle(
    detections: Sequence[Detection], aspect_ratio: float = 1.0
) -> list[Hazard]:
    """Person with a no_vest box within the machinery-zone distance of a vehicle."""
    vehicles = with_labels(detections, {VEHICLE})
    no_vests = with_labels(detections, {NO_VEST})
    hazards = []
    for i, person in enumerate(with_labels(detections, {PERSON})):
        bare = [v for v in no_vests if overlaps(v.box, person.box)]
        close = within_zone(person, vehicles, MACHINERY_ZONE_FACTOR, aspect_ratio)
        if not bare or not close:
            continue
        best = max(v.confidence for v in bare)
        hazards.append(
            Hazard.of(
                "no_vest_near_vehicle",
                f"Person {i + 1} has no safety vest (no_vest, confidence {best:.2f}) "
                f"and {_where(person, close[0])}.",
                (person.box, *(v.box for v in bare), *(c.box for c, _ in close)),
                _subject(i),
            )
        )
    return hazards


def _cords_in_walkway(
    detections: Sequence[Detection], labels: Collection[str] = CORD_LABELS | {HOSE}
) -> list[Detection]:
    return [c for c in with_labels(detections, labels) if c.box[3] > WALKWAY_TOP_Y]


def _count_phrase(name: str, count: int) -> str:
    """'a hose', 'an extension cord', 'chemicals', '4 cords', '2 knives'."""
    if name.endswith("s"):  # already plural, e.g. "chemicals"
        return name if count == 1 else f"{count} {name}"
    if count == 1:
        return f"{'an' if name[0] in 'aeiou' else 'a'} {name}"
    return f"{count} {IRREGULAR_PLURALS.get(name, name + 's')}"


def _describe(found: Sequence[Detection]) -> str:
    """'4 cords', 'a knife and chemicals', '2 cords, a hose and an extension cord'."""
    counts = Counter(_name(d.label) for d in found)
    parts = [_count_phrase(name, n) for name, n in sorted(counts.items())]
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def trip_cord(detections: Sequence[Detection]) -> list[Hazard]:
    """Cord or hose crossing the lower third of the frame, where people walk."""
    cords = _cords_in_walkway(detections)
    if not cords:
        return []
    return [
        Hazard.of(
            "trip_cord",
            f"Trip risk: {_describe(cords)} in the walkway (lower third of the frame).",
            tuple(c.box for c in cords),
        )
    ]


def heat_hazards(detections: Sequence[Detection], context: SceneContext) -> list[Hazard]:
    """midday_ban, wbgt_stop and heat_welfare_gap (see heat.py)."""
    workers = count_workers(detections)
    found = (
        heat.midday_ban(context.taken_at, context.outdoor),
        heat.wbgt_stop(context.wbgt_c, workers),
        heat.heat_welfare_gap(
            context.outdoor,
            workers,
            shade_seen=bool(with_labels(detections, {SHADE_TENT})),
            water_seen=bool(with_labels(detections, {WATER_COOLER})),
            taken_at=context.taken_at,
        ),
    )
    return [h for h in found if h is not None]


# --- Site rules v2 -------------------------------------------------------------------
def _is_lying(person: Detection, machines: Sequence[Detection], aspect_ratio: float) -> bool:
    box = person.box
    return (
        person.confidence >= WORKER_DOWN_MIN_CONF
        and height(box) >= WORKER_DOWN_MIN_HEIGHT
        and width_to_height(box, aspect_ratio) >= WORKER_DOWN_RATIO
        and not touches_border(box)
        and not any(inside_fraction(box, m.box) >= WORKER_DOWN_MAX_INSIDE for m in machines)
    )


def worker_down(detections: Sequence[Detection], aspect_ratio: float = 1.0) -> list[Hazard]:
    """Person box clearly wider than tall in real pixels: lying down (heat illness or fall)."""
    machines = with_labels(detections, {MACHINERY, VEHICLE})
    hazards = []
    for i, person in enumerate(with_labels(detections, {PERSON})):
        if not _is_lying(person, machines, aspect_ratio):
            continue
        ratio = width_to_height(person.box, aspect_ratio)
        hazards.append(
            Hazard.of(
                "worker_down",
                f"Person {i + 1} is lying on the ground {_side(person.box)} "
                f"(box {ratio:.1f}x wider than tall): possible heat illness or fall.",
                (person.box,),
                _subject(i),
            )
        )
    return hazards


def _hooks_over(
    person: Detection, hooks: Sequence[Detection], aspect_ratio: float
) -> list[tuple[Detection, float]]:
    """Crane hooks above this person and close to the side of them, nearest first."""
    tall = height(person.box)
    lowest_bottom = person.box[1] + SUSPENDED_LOAD_TOP_FRACTION * tall
    hits = [
        (h, horizontal_gap(person.box, h.box, aspect_ratio))
        for h in hooks
        if h.box[3] <= lowest_bottom
    ]
    close = [(h, g) for h, g in hits if g <= SUSPENDED_LOAD_FACTOR * tall]
    return sorted(close, key=lambda hit: hit[1])


def under_suspended_load(
    detections: Sequence[Detection], aspect_ratio: float = 1.0
) -> list[Hazard]:
    """Person below a crane hook (suspended load) and horizontally close to it."""
    hooks = with_labels(detections, {CRANE_HOOK})
    hazards = []
    for i, person in enumerate(with_labels(detections, {PERSON})):
        close = _hooks_over(person, hooks, aspect_ratio)
        if not close:
            continue
        side = close[0][1] / height(person.box)
        where = "directly under" if side == 0 else f"{side:.2f}x their height to the side of"
        hazards.append(
            Hazard.of(
                "under_suspended_load",
                f"Person {i + 1} is {where} a crane hook (suspended load overhead).",
                (person.box, *(h.box for h, _ in close)),
                _subject(i),
            )
        )
    return hazards


def _unique_boxes(detections: Sequence[Detection]) -> tuple[Box, ...]:
    return tuple(dict.fromkeys(d.box for d in detections))


def power_line_contact(
    detections: Sequence[Detection], aspect_ratio: float = 1.0
) -> list[Hazard]:
    """Machinery overlapping or within 0.5 x its height of a utility pole or power line."""
    lines = with_labels(detections, {UTILITY_POLE, POWER_LINE})
    pairs = [
        (m, line, gap(m.box, line.box, aspect_ratio) / height(m.box))
        for m in with_labels(detections, {MACHINERY})
        for line in lines
    ]
    close = [p for p in pairs if p[2] <= POWER_LINE_FACTOR]
    if not close:
        return []
    machine, line, distance = min(close, key=lambda p: p[2])
    where = "overlaps" if distance == 0 else f"is {distance:.2f}x its height from"
    involved = [d for m, ln, _ in close for d in (m, ln)]
    return [
        Hazard.of(
            "power_line_contact",
            f"A machine {where} a {_name(line.label)} "
            f"(danger zone {POWER_LINE_FACTOR}x machine height).",
            _unique_boxes(involved),
        )
    ]


def _welding_near_cylinders(
    detections: Sequence[Detection], aspect_ratio: float
) -> list[tuple[Detection, Detection, float]]:
    """(welding, cylinder, gap in cylinder heights) for pairs within HOT_WORK_FACTOR."""
    cylinders = with_labels(detections, {GAS_CYLINDER})
    pairs = [
        (w, c, gap(w.box, c.box, aspect_ratio) / height(c.box))
        for w in with_labels(detections, {WELDING})
        for c in cylinders
    ]
    return [p for p in pairs if p[2] <= HOT_WORK_FACTOR]


def hot_work_fire(detections: Sequence[Detection], aspect_ratio: float = 1.0) -> list[Hazard]:
    """Fire anywhere, or welding within 2.0 x gas-cylinder height of a gas cylinder."""
    fires = with_labels(detections, {FIRE})
    close = _welding_near_cylinders(detections, aspect_ratio)
    if not fires and not close:
        return []
    causes = []
    if fires:
        causes.append(f"fire seen (confidence {max(f.confidence for f in fires):.2f})")
    if close:
        distance = min(p[2] for p in close)
        where = "next to" if distance == 0 else f"{distance:.2f}x cylinder height from"
        causes.append(f"welding {where} a gas cylinder")
    involved = [*fires, *(d for w, c, _ in close for d in (w, c))]
    return [
        Hazard.of(
            "hot_work_fire",
            f"Hot work fire risk: {' and '.join(causes)}.",
            _unique_boxes(involved),
        )
    ]


# --- Home rules ------------------------------------------------------------------
def home_trip(detections: Sequence[Detection]) -> list[Hazard]:
    """Cord on the floor (lower third of the frame)."""
    cords = _cords_in_walkway(detections, CORD_LABELS)
    if not cords:
        return []
    return [
        Hazard.of(
            "home_trip",
            f"Trip risk: {_describe(cords)} on the floor (lower third of the frame).",
            tuple(c.box for c in cords),
        )
    ]


def home_slip(detections: Sequence[Detection]) -> list[Hazard]:
    """Wet floor detected."""
    wet = with_labels(detections, {WET_FLOOR})
    if not wet:
        return []
    best = max(w.confidence for w in wet)
    return [
        Hazard.of(
            "home_slip",
            f"Slip risk: wet floor detected (confidence {best:.2f}).",
            tuple(w.box for w in wet),
        )
    ]


def home_child_reach(detections: Sequence[Detection]) -> list[Hazard]:
    """Knife or chemicals at low height (centred in the lower half of the frame)."""
    reachable = with_labels(detections, CHILD_REACH_LABELS)
    items = [d for d in reachable if centre_y(d.box) >= CHILD_REACH_MIN_Y]
    if not items:
        return []
    return [
        Hazard.of(
            "home_child_reach",
            f"Within a child's reach: {_describe(items)} in the lower half of the frame.",
            tuple(d.box for d in items),
        )
    ]


# --- Entry point -------------------------------------------------------------------
def evaluate(detections: Sequence[Detection], context: SceneContext | None = None) -> list[Hazard]:
    """Run every rule for the scene's mode. High-severity hazards come first."""
    ctx = context if context is not None else SceneContext()
    dets = confident(detections)
    if ctx.mode is Mode.HOME:
        return [*home_trip(dets), *home_slip(dets), *home_child_reach(dets)]
    ratio = ctx.aspect_ratio
    hazards = [
        *no_helmet(dets),
        *machinery_zone(dets, ratio),
        *fall_no_harness(dets, ratio),
        *heat_hazards(dets, ctx),
        *no_vest_near_vehicle(dets, ratio),
        *trip_cord(dets),
        *worker_down(dets, ratio),
        *under_suspended_load(dets, ratio),
        *power_line_contact(dets, ratio),
        *hot_work_fire(dets, ratio),
    ]
    return sorted(hazards, key=lambda h: h.severity is not Severity.HIGH)
