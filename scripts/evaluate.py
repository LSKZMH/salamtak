"""Evaluate the full SafeSite pipeline on our own Qatar test photos.

Produces pitch numbers 2-4 from CLAUDE.md: per-hazard catch rate and false alarms,
seconds per photo on CPU, and the credits total from COSTS.md. These photos are
never used for training.

labels.csv (UTF-8, one row per photo; only `filename` and `hazards` are required):
  filename  photo file, relative to the labels file's folder (or --images)
  hazards   hazard ids present, separated by ";"  (blank = clean photo, no hazards)
  taken_at  "YYYY-MM-DD HH:MM" Doha time; blank = photo EXIF time, else a neutral
            time outside the summer ban (so no time-based hazard fires by accident)
  wbgt_c    WBGT in degrees C; blank = not measured (required if wbgt_stop is labelled)
  outdoor   true/false (default true)
  mode      site/home (default site)
  notes     free text, ignored

Usage (from the repo root):
  python scripts/evaluate.py                                # mock detector
  python scripts/evaluate.py --detector yolo                # fine-tuned YOLO26 from the Hub
  python scripts/evaluate.py --detector yolo --weights best.pt
  python scripts/evaluate.py --detector combined           # trained YOLO26 + YOLOE (what the Space runs)
  python scripts/evaluate.py --detector package.module:Name # any detector class
  python scripts/evaluate.py --threads 8 --json eval.json   # 8 threads = cpu-upgrade Space
  python scripts/evaluate.py --space plitip/team-qh-091-safesite-app   # the live Space, timed on its CPU
  python scripts/evaluate.py --labels other/labels.csv --only-hazards no_helmet,machinery_zone
      score only these ids; other predicted hazards are ignored, not counted as false alarms
      (for outside benchmarks that don't label e.g. trip_cord or the heat rules)

Exit codes: 0 ok, 1 no photo could be read, 2 bad labels.csv, 3 some photos skipped
(their hazards are excluded from every number; pass --allow-skipped to accept that).

Works with either rules entry point (`rules.evaluate` or `rules.find_hazards`) and
fills SceneContext with whichever fields it declares.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import html
import importlib
import json
import os
import re
import statistics
import sys
import time
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = Path("examples/qatar_test/labels.csv")
DEFAULT_COSTS = Path("COSTS.md")

REQUIRED_COLUMNS = ("filename", "hazards")
HAZARD_SEPARATOR = ";"
HOME_PREFIX = "home_"
TIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S")
NEUTRAL_TIME = datetime(2026, 1, 15, 8, 0)  # winter morning: outside the midday ban
WBGT_MIN_C, WBGT_MAX_C = 0.0, 60.0  # same bounds as app.py
TRUE_WORDS = frozenset({"true", "yes", "y", "1"})
FALSE_WORDS = frozenset({"false", "no", "n", "0"})
MODES = ("site", "home")
EXIF_IFD, EXIF_DATETIME_ORIGINAL, EXIF_DATETIME = 0x8769, 0x9003, 0x0132
DEFAULT_WARMUP_RUNS = 1  # first inference is slow (lazy init); keep it out of the timing
TIME_HAZARD, WBGT_HAZARD = "midday_ban", "wbgt_stop"  # rules that need an input, not just pixels
WBGT_CONTEXT_FIELDS = ("wbgt_c", "wbgt")
EXIT_OK, EXIT_NO_PHOTOS, EXIT_BAD_LABELS, EXIT_SKIPPED = 0, 1, 2, 3

# Fallback when safesite.types has no HAZARD_IDS. Mirrors the CLAUDE.md hazard table + PLAN-v2.
CLAUDE_MD_HAZARD_IDS = (
    "no_helmet", "machinery_zone", "fall_no_harness", "midday_ban", "wbgt_stop",
    "no_vest_near_vehicle", "heat_welfare_gap", "trip_cord",
    "worker_down", "under_suspended_load", "power_line_contact", "hot_work_fire",
    "no_fire_extinguisher", "exposed_rebar", "fall_into_opening",
    "home_trip", "home_slip", "home_child_reach",
)


class LabelError(ValueError):
    """labels.csv is malformed. The message lists every bad row."""


# ---------- data ----------
@dataclass(frozen=True)
class LabelledPhoto:
    filename: str
    hazards: frozenset[str]
    taken_at: datetime | None = None
    wbgt_c: float | None = None
    outdoor: bool = True
    mode: str = "site"


@dataclass(frozen=True)
class PhotoResult:
    filename: str
    expected: frozenset[str]
    predicted: frozenset[str]
    score: int
    wall_s: float
    cpu_s: float
    mode: str = "site"


@dataclass(frozen=True)
class RunOutcome:
    results: list[PhotoResult]
    skipped: list[str]  # photos that could not be read: excluded from every metric
    warnings: list[str]  # labelled rules that cannot fire with the inputs given


@dataclass(frozen=True)
class HazardStats:
    hazard_id: str
    labelled: int  # photos labelled with this hazard
    caught: int  # ...that the pipeline also flagged
    negatives: int  # photos not labelled with this hazard
    false_alarms: int  # ...that the pipeline flagged anyway

    @property
    def catch_rate(self) -> float | None:
        return self.caught / self.labelled if self.labelled else None

    @property
    def false_alarm_rate(self) -> float | None:
        return self.false_alarms / self.negatives if self.negatives else None


@dataclass(frozen=True)
class CostSummary:
    stated_total: float | None  # the "**Total: $X**" line
    row_sum: float  # sum of the cost column
    rows: int


@dataclass(frozen=True)
class Pipeline:
    detect: Callable[[Any, str], list]
    find_hazards: Callable[[list, Any], list]
    safety_score: Callable[[list], int]
    context_type: type
    detector_name: str
    is_mock: bool


# ---------- labels ----------
def _cell(row: dict[str, str | None], key: str) -> str:
    return (row.get(key) or "").strip()


def parse_time(text: str) -> datetime | None:
    if not text:
        return None
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise LabelError(f"taken_at {text!r} is not YYYY-MM-DD HH:MM")


def parse_wbgt(text: str) -> float | None:
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise LabelError(f"wbgt_c {text!r} is not a number") from None
    if not WBGT_MIN_C <= value <= WBGT_MAX_C:
        raise LabelError(f"wbgt_c {value} is outside {WBGT_MIN_C:g}-{WBGT_MAX_C:g} C")
    return value


def parse_bool(text: str, default: bool) -> bool:
    word = text.lower()
    if not word:
        return default
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise LabelError(f"outdoor {text!r} must be true or false")


def hazard_mode(hazard_id: str) -> str:
    return "home" if hazard_id.startswith(HOME_PREFIX) else "site"


def parse_hazards(text: str, mode: str, known_ids: Collection[str]) -> frozenset[str]:
    ids = frozenset(part.strip() for part in text.split(HAZARD_SEPARATOR) if part.strip())
    unknown = sorted(ids - set(known_ids))
    if unknown:
        raise LabelError(f"unknown hazard id(s) {', '.join(unknown)}")
    wrong_mode = sorted(i for i in ids if hazard_mode(i) != mode)
    if wrong_mode:
        raise LabelError(f"{', '.join(wrong_mode)} can't occur in {mode} mode")
    return ids


def parse_filename(text: str) -> str:
    if not text:
        raise LabelError("filename is empty")
    path = PurePath(text.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in text:
        raise LabelError(f"filename {text!r} must be relative to the labels folder")
    return path.as_posix()


def parse_row(row: dict[str, str | None], known_ids: Collection[str]) -> LabelledPhoto:
    mode = _cell(row, "mode").lower() or "site"
    if mode not in MODES:
        raise LabelError(f"mode {mode!r} must be site or home")
    photo = LabelledPhoto(
        filename=parse_filename(_cell(row, "filename")),
        hazards=parse_hazards(_cell(row, "hazards"), mode, known_ids),
        taken_at=parse_time(_cell(row, "taken_at")),
        wbgt_c=parse_wbgt(_cell(row, "wbgt_c")),
        outdoor=parse_bool(_cell(row, "outdoor"), default=True),
        mode=mode,
    )
    if WBGT_HAZARD in photo.hazards and photo.wbgt_c is None:
        raise LabelError(f"{WBGT_HAZARD} needs wbgt_c: the rule only fires on a measured WBGT")
    return photo


def parse_labels(lines: Iterable[str], known_ids: Collection[str]) -> list[LabelledPhoto]:
    """Parse labels.csv content. Collects every row error before raising."""
    reader = csv.DictReader(lines)
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or ())]
    if missing:
        raise LabelError(f"missing column(s): {', '.join(missing)}")
    photos: list[LabelledPhoto] = []
    errors: list[str] = []
    seen: set[str] = set()
    for row in reader:
        if not any(_cell(row, k) for k in row if k):
            continue  # blank line
        try:
            photo = parse_row(row, known_ids)
        except LabelError as exc:
            errors.append(f"line {reader.line_num}: {exc}")
            continue
        if photo.filename in seen:
            errors.append(f"line {reader.line_num}: duplicate filename {photo.filename}")
        seen.add(photo.filename)
        photos.append(photo)
    if errors:
        raise LabelError("\n".join(errors))
    return photos


def load_labels(path: Path, known_ids: Collection[str]) -> list[LabelledPhoto]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return parse_labels(handle, known_ids)


# ---------- costs ----------
_DOLLARS = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
# "**Total: $X**" or "**Total spent: ≈ $X**"
_TOTAL = re.compile(r"\*\*\s*total\b[^$*\n]*\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*\*\*", re.IGNORECASE)


def _dollars(text: str) -> float | None:
    match = _DOLLARS.search(text)
    return float(match.group(1).replace(",", "")) if match else None


def _table_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_total_row(cells: Sequence[str]) -> bool:
    return bool(cells) and cells[0].strip("* ").lower().startswith("total")


def parse_costs(text: str) -> CostSummary:
    """Sum the cost column of every table whose header has a 'cost' column.
    The stated total is the last '**Total: $X**' line; table 'Total' rows are not summed."""
    totals = _TOTAL.findall(text)
    stated = float(totals[-1].replace(",", "")) if totals else None
    cost_col: int | None = None
    row_sum, rows = 0.0, 0
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            cost_col = None  # a table ends at the first non-table line
            continue
        cells = _table_cells(line)
        if cost_col is None:
            cost_col = next((i for i, c in enumerate(cells) if "cost" in c.lower()), -1)
            continue  # header row
        if cost_col < 0 or cost_col >= len(cells) or _is_total_row(cells):
            continue
        amount = _dollars(cells[cost_col])
        if amount is not None:
            row_sum, rows = row_sum + amount, rows + 1
    return CostSummary(stated, round(row_sum, 2), rows)


# ---------- metrics ----------
def hazard_stats(results: Sequence[PhotoResult], hazard_ids: Iterable[str]) -> list[HazardStats]:
    """Photo-level counts. Only photos in the hazard's own mode count as negatives,
    so home photos don't dilute a site hazard's false-alarm rate (and vice versa)."""
    stats = []
    for hid in hazard_ids:
        applicable = [r for r in results if r.mode == hazard_mode(hid)]
        labelled = [r for r in applicable if hid in r.expected]
        stats.append(
            HazardStats(
                hazard_id=hid,
                labelled=len(labelled),
                caught=sum(hid in r.predicted for r in labelled),
                negatives=len(applicable) - len(labelled),
                false_alarms=sum(hid in r.predicted and hid not in r.expected for r in applicable),
            )
        )
    return stats


def ids_to_report(results: Sequence[PhotoResult], known_ids: Sequence[str]) -> list[str]:
    """Known ids that were labelled or predicted at least once, in table order."""
    seen = set().union(*(r.expected | r.predicted for r in results)) if results else set()
    return [i for i in known_ids if i in seen] + sorted(seen - set(known_ids))


def parse_only_hazards(text: str | None, known_ids: Collection[str]) -> frozenset[str] | None:
    """--only-hazards value -> id set (None = score every hazard)."""
    if text is None:
        return None
    ids = frozenset(part.strip() for part in text.split(",") if part.strip())
    if not ids:
        raise SystemExit("--only-hazards is empty")
    unknown = sorted(ids - set(known_ids))
    if unknown:
        raise SystemExit(f"--only-hazards: unknown hazard id(s) {', '.join(unknown)}")
    return ids


def restrict(results: Sequence[PhotoResult], only: frozenset[str] | None) -> list[PhotoResult]:
    """Keep only the scored ids on both sides, so an unscored prediction is neither a catch nor
    a false alarm. Returns new results; the inputs are not changed."""
    if only is None:
        return list(results)
    return [dataclasses.replace(r, expected=r.expected & only, predicted=r.predicted & only)
            for r in results]


def report_ids(results: Sequence[PhotoResult], known_ids: Sequence[str],
               only: frozenset[str] | None) -> list[str]:
    """Rows for the per-hazard table: every scored id when filtering (a zero row is information),
    else the ids that were labelled or predicted."""
    if only is not None:
        return [i for i in known_ids if i in only] + sorted(only - set(known_ids))
    return ids_to_report(results, known_ids)


def summarize(results: Sequence[PhotoResult], stats: Sequence[HazardStats]) -> dict[str, Any]:
    labelled = sum(s.labelled for s in stats)
    caught = sum(s.caught for s in stats)
    false_alarms = sum(s.false_alarms for s in stats)
    clean = [r for r in results if not r.expected]
    wall = [r.wall_s for r in results]
    cpu = [r.cpu_s for r in results]
    return {
        "photos": len(results),
        "hazards_labelled": labelled,
        "hazards_caught": caught,
        "catch_rate": caught / labelled if labelled else None,
        "false_alarms": false_alarms,
        "false_alarms_per_photo": false_alarms / len(results) if results else None,
        "clean_photos": len(clean),
        "clean_photos_flagged": sum(bool(r.predicted) for r in clean),
        "exact_match_photos": sum(r.expected == r.predicted for r in results),
        "wall_s_mean": statistics.fmean(wall) if wall else None,
        "wall_s_median": statistics.median(wall) if wall else None,
        "wall_s_max": max(wall) if wall else None,
        "cpu_s_mean": statistics.fmean(cpu) if cpu else None,
    }


# ---------- pipeline adapter ----------
def context_kwargs(field_names: Collection[str], photo: LabelledPhoto, taken_at: datetime,
                   aspect_ratio: float) -> dict[str, Any]:
    """SceneContext kwargs for whichever fields this version of types.py declares."""
    values = {
        "taken_at": taken_at,
        "outdoor": photo.outdoor,
        "mode": photo.mode,
        "wbgt_c": photo.wbgt_c,
        "wbgt": photo.wbgt_c,
        "aspect_ratio": aspect_ratio,
    }
    return {k: v for k, v in values.items() if k in field_names}


class BuiltinMockDetector:
    """Used only when safesite.detect.MockDetector can't be imported. Same idea: fixed boxes."""

    name = "built-in mock (fixed fake detections)"
    _SITE = (
        ("person", 0.91, (0.10, 0.35, 0.22, 0.88)),
        ("no_helmet", 0.84, (0.125, 0.35, 0.195, 0.45)),
        ("person", 0.89, (0.50, 0.40, 0.60, 0.88)),
        ("helmet", 0.87, (0.52, 0.39, 0.58, 0.47)),
        ("no_vest", 0.72, (0.505, 0.49, 0.595, 0.65)),
        ("vehicle", 0.93, (0.66, 0.42, 0.97, 0.86)),
    )
    _HOME = (("cord", 0.78, (0.15, 0.82, 0.75, 0.90)), ("knife", 0.70, (0.62, 0.58, 0.72, 0.62)))

    def __call__(self, image: Any, mode: str = "site") -> list:
        from safesite.types import Detection

        rows = self._HOME if str(getattr(mode, "value", mode)) == "home" else self._SITE
        return [Detection(label, conf, box) for label, conf, box in rows]


def _mock_detector() -> Any:
    try:
        from safesite import detect

        return detect.MockDetector()
    except (ImportError, AttributeError, TypeError) as exc:
        print(f"note: safesite.detect.MockDetector unavailable ({exc}); using built-in mock",
              file=sys.stderr)
        return BuiltinMockDetector()


def _import_safesite(name: str) -> Any:
    return importlib.import_module(f"safesite.{name}")


def _yolo(weights: str | None) -> Any:
    detect = _import_safesite("detect")
    return detect.YoloDetector(weights_path=weights) if weights else detect.YoloDetector()


def load_detector(spec: str, weights: str | None) -> tuple[Any, bool]:
    """'mock' | 'yolo' | 'combined' | 'package.module:Name'. Returns (detector, is_mock).
    'combined' = trained YOLO26 + YOLOE open-vocab finder, as the Space runs by default."""
    if spec == "mock":
        return _mock_detector(), True
    if spec == "yolo":
        return _yolo(weights), False
    if spec == "combined":
        detect, openvocab = _import_safesite("detect"), _import_safesite("openvocab")
        return detect.CombinedDetector(_yolo(weights), openvocab.OpenVocabDetector()), False
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"--detector must be mock, yolo, combined or module:Name, got {spec!r}")
    target = getattr(importlib.import_module(module_name), attr)
    detector = target() if isinstance(target, type) else target
    return detector, "mock" in type(detector).__name__.lower()


def load_pipeline(detector_spec: str, weights: str | None) -> Pipeline:
    from safesite import rules, score, types

    find = getattr(rules, "evaluate", None) or getattr(rules, "find_hazards", None)
    if find is None:
        raise SystemExit("safesite.rules has neither evaluate() nor find_hazards()")
    detector, is_mock = load_detector(detector_spec, weights)
    return Pipeline(
        detect=detector,
        find_hazards=find,
        safety_score=score.safety_score,
        context_type=types.SceneContext,
        detector_name=str(getattr(detector, "name", type(detector).__name__)),
        is_mock=is_mock,
    )


def known_hazard_ids() -> tuple[str, ...]:
    """Ids the loaded rules can produce."""
    try:
        from safesite.types import HAZARD_IDS

        return tuple(HAZARD_IDS)
    except ImportError:
        return CLAUDE_MD_HAZARD_IDS


def labelable_ids(rule_ids: Sequence[str]) -> tuple[str, ...]:
    """Ids allowed in labels.csv: the rules' ids plus the fallback list, so photos can be labelled
    with a new hazard before (or without) the rules that detect it. Such labels count as misses."""
    return tuple(rule_ids) + tuple(i for i in CLAUDE_MD_HAZARD_IDS if i not in rule_ids)


def unproducible_warnings(photos: Sequence[LabelledPhoto], rule_ids: Collection[str]) -> list[str]:
    labelled = set().union(*(p.hazards for p in photos)) if photos else set()
    return [f"labelled {i} but these rules can't produce it (every such label is a miss)"
            for i in sorted(labelled - set(rule_ids))]


# ---------- running ----------
def exif_time(image: Any) -> datetime | None:
    try:
        exif = image.getexif()
        raw = exif.get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
        return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S") if raw else None
    except (ValueError, TypeError, AttributeError, KeyError):
        return None


def open_photo(path: Path) -> tuple[Any, datetime | None]:
    """Decode like the Space does (Gradio applies EXIF rotation). Returns (RGB image, EXIF time)."""
    from PIL import Image, ImageOps

    with Image.open(path) as raw:
        taken = exif_time(raw)
        image = ImageOps.exif_transpose(raw).convert("RGB")
    return image, taken


def input_warnings(photo: LabelledPhoto, time_known: bool,
                   context_fields: Collection[str]) -> list[str]:
    """Labelled rules that cannot fire because an input is missing (counted as misses)."""
    warnings = []
    if TIME_HAZARD in photo.hazards and not time_known:
        warnings.append(f"{photo.filename}: {TIME_HAZARD} labelled but no taken_at and no EXIF time")
    if photo.wbgt_c is not None and not any(f in context_fields for f in WBGT_CONTEXT_FIELDS):
        warnings.append(f"{photo.filename}: WBGT given but SceneContext has no WBGT field")
    return warnings


def run_photo(pipeline: Pipeline, photo: LabelledPhoto, image: Any,
              exif_taken: datetime | None) -> PhotoResult:
    """Time detect + rules + score on an already-decoded image (as in app.analyze)."""
    taken_at = photo.taken_at or exif_taken or NEUTRAL_TIME
    names = context_fields(pipeline)
    context = pipeline.context_type(
        **context_kwargs(names, photo, taken_at, image.width / image.height)
    )
    wall0, cpu0 = time.perf_counter(), time.process_time()
    detections = pipeline.detect(image, photo.mode)
    hazards = pipeline.find_hazards(detections, context)
    value = pipeline.safety_score(hazards)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    return PhotoResult(photo.filename, photo.hazards, frozenset(h.id for h in hazards),
                       int(value), wall, cpu, photo.mode)


def context_fields(pipeline: Pipeline) -> set[str]:
    return {f.name for f in dataclasses.fields(pipeline.context_type)}


def run_all(pipeline: Pipeline, photos: Sequence[LabelledPhoto], image_dir: Path,
            warmup: int) -> RunOutcome:
    outcome = RunOutcome([], [], [])
    for photo in photos:
        try:
            image, exif_taken = open_photo(image_dir / photo.filename)
        except (OSError, ValueError) as exc:  # missing or unreadable: report, don't hide
            outcome.skipped.append(f"{photo.filename} ({type(exc).__name__}: {exc})")
            continue
        for _ in range(warmup if not outcome.results else 0):
            pipeline.detect(image, photo.mode)
        time_known = (photo.taken_at or exif_taken) is not None
        outcome.warnings.extend(input_warnings(photo, time_known, context_fields(pipeline)))
        result = run_photo(pipeline, photo, image, exif_taken)
        outcome.results.append(result)
        print(f"  {photo.filename}: score {result.score}, {result.wall_s:.2f} s", file=sys.stderr)
    return outcome


# ---------- live Space (--space) ----------
SPACE_API = "/analyze"
SPACE_TIME_FORMAT = "%Y-%m-%d %H:%M"
SPACE_LANG = "en"
_SPACE_STATUS = re.compile(r"Detector:\s*(.*?)\s*·\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*</p>", re.DOTALL)


def parse_space_status(score_html: str) -> tuple[str, float]:
    """(detector name, server-side seconds) from the Space's score HTML (app._score_html)."""
    match = _SPACE_STATUS.search(score_html or "")
    if not match:
        raise ValueError(f"Space score HTML has no 'Detector: ... · N s' line: {score_html[:200]!r}")
    return html.unescape(match.group(1)), float(match.group(2))


def space_hazard_ids(table: Any) -> frozenset[str]:
    """Hazard ids from the Space's Dataframe output ({headers, data} or a list of rows)."""
    rows = table.get("data", []) if isinstance(table, dict) else (table or [])
    return frozenset(str(row[0]) for row in rows if row and row[0])


def space_inputs(photo: LabelledPhoto, exif_taken: datetime | None) -> dict[str, Any]:
    """Always pass a time: with a blank override the Space would use EXIF or *now*."""
    taken_at = photo.taken_at or exif_taken or NEUTRAL_TIME
    return {"time_override": taken_at.strftime(SPACE_TIME_FORMAT), "wbgt": photo.wbgt_c,
            "outdoor": photo.outdoor, "lang": SPACE_LANG, "mode": photo.mode.capitalize()}


def run_space(space: str, photos: Sequence[LabelledPhoto], image_dir: Path,
              warmup: int) -> tuple[Pipeline, RunOutcome]:
    """Send each photo to the live Space. wall_s/cpu_s = the Space's own processing time
    (excludes upload and queueing); the client round trip is printed to stderr only."""
    from gradio_client import Client, handle_file

    client = Client(space, verbose=False)
    outcome = RunOutcome([], [], [])
    name = space
    for photo in photos:
        path = image_dir / photo.filename
        try:
            _, exif_taken = open_photo(path)
        except (OSError, ValueError) as exc:
            outcome.skipped.append(f"{photo.filename} ({type(exc).__name__}: {exc})")
            continue
        inputs = space_inputs(photo, exif_taken)
        for _ in range(warmup if not outcome.results else 0):
            client.predict(handle_file(str(path)), **inputs, api_name=SPACE_API)  # wake + lazy init
        started = time.perf_counter()
        score_html, _alerts, _image, table = client.predict(handle_file(str(path)), **inputs,
                                                            api_name=SPACE_API)
        round_trip = time.perf_counter() - started
        name, secs = parse_space_status(score_html)
        hazards = space_hazard_ids(table)
        outcome.results.append(PhotoResult(photo.filename, photo.hazards, hazards,
                                           100, secs, secs, photo.mode))
        print(f"  {photo.filename}: {sorted(hazards)}, Space {secs:.2f} s, "
              f"round trip {round_trip:.2f} s", file=sys.stderr)
    pipeline = Pipeline(None, None, None, object, f"live Space {space}: {name}",
                        is_mock="mock" in name.lower())
    return pipeline, outcome


# ---------- report ----------
def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def _secs(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def hazard_table(stats: Sequence[HazardStats]) -> list[str]:
    if not stats:
        return ["(no hazards labelled or predicted)"]
    lines = ["| hazard | labelled | caught | catch rate | false alarms | false-alarm rate |",
             "|---|---|---|---|---|---|"]
    for s in stats:
        lines.append(f"| `{s.hazard_id}` | {s.labelled} | {s.caught} | {_pct(s.catch_rate)} "
                     f"| {s.false_alarms} / {s.negatives} | {_pct(s.false_alarm_rate)} |")
    return lines


def mismatch_lines(results: Sequence[PhotoResult]) -> list[str]:
    lines = []
    for r in results:
        missed, extra = sorted(r.expected - r.predicted), sorted(r.predicted - r.expected)
        if missed or extra:
            lines.append(f"- `{r.filename}`: missed {', '.join(missed) or '-'}; "
                         f"false alarm {', '.join(extra) or '-'}")
    return lines


def cost_lines(costs: CostSummary | None, path: Path) -> list[str]:
    if costs is None:
        return [f"- Credits spent: n/a ({path} not found)"]
    lines = [f"- Credits spent (COSTS.md total): ${costs.stated_total:.2f}"
             if costs.stated_total is not None else "- Credits spent: no '**Total: $X**' line"]
    lines.append(f"  - cost column sums to ${costs.row_sum:.2f} over {costs.rows} row(s)")
    if costs.stated_total is not None and abs(costs.stated_total - costs.row_sum) > 0.005:
        lines.append("  - WARNING: stated total and cost column disagree; fix COSTS.md")
    return lines


def headline_lines(summary: dict[str, Any]) -> list[str]:
    return [
        f"- Catch rate: {_pct(summary['catch_rate'])} "
        f"({summary['hazards_caught']}/{summary['hazards_labelled']} labelled hazards)",
        f"- False alarms: {summary['false_alarms']} ({_secs(summary['false_alarms_per_photo'])} "
        f"per photo); clean photos flagged: {summary['clean_photos_flagged']}/{summary['clean_photos']}",
        f"- Exact match (all hazards right): {summary['exact_match_photos']}/{summary['photos']}",
        f"- Seconds per photo on CPU (wall, detect+rules+score): mean {_secs(summary['wall_s_mean'])}, "
        f"median {_secs(summary['wall_s_median'])}, max {_secs(summary['wall_s_max'])}; "
        f"process CPU time mean {_secs(summary['cpu_s_mean'])}",
    ]


def render_report(pipeline: Pipeline, stats: Sequence[HazardStats], summary: dict[str, Any],
                  outcome: RunOutcome, costs_block: Sequence[str],
                  only: frozenset[str] | None = None, title: str = "Qatar test photos") -> str:
    skipped = outcome.skipped
    lines = [f"# SafeSite evaluation: {title}", ""]
    if pipeline.is_mock:
        lines += ["> **MOCK DETECTOR: these numbers are not real. Do not put them on a slide.**", ""]
    skipped_note = f" ({len(skipped)} skipped)" if skipped else ""
    lines += [f"Detector: {pipeline.detector_name}  ",
              f"Photos evaluated: {summary['photos']}{skipped_note}"]
    if only is not None:
        lines += ["", f"Scored hazards only: {', '.join(sorted(only))} "
                      "(other predicted hazards are ignored, not counted as false alarms)"]
    lines += ["", "## Per hazard", ""]
    lines += hazard_table(stats)
    lines += ["", "## Headline", ""]
    if skipped:
        lines.append(f"- **WARNING: {len(skipped)} photo(s) skipped; every number below excludes them.**")
    lines += [*headline_lines(summary), *costs_block]
    if mismatches := mismatch_lines(outcome.results):
        lines += ["", "## Photos with mistakes", "", *mismatches]
    if outcome.warnings:
        lines += ["", "## Input warnings (these rules could not fire)", "",
                  *(f"- {w}" for w in outcome.warnings)]
    if skipped:
        lines += ["", "## Skipped (not counted above)", "", *(f"- {s}" for s in skipped)]
    return "\n".join(lines) + "\n"


def write_json(path: Path, pipeline: Pipeline, stats: Sequence[HazardStats],
               summary: dict[str, Any], outcome: RunOutcome, costs: CostSummary | None,
               only: frozenset[str] | None = None, threads: int | None = None) -> None:
    payload = {
        "detector": pipeline.detector_name,
        "mock": pipeline.is_mock,
        "only_hazards": sorted(only) if only is not None else None,
        "threads": threads,
        "summary": summary,
        "per_hazard": [{**dataclasses.asdict(s), "catch_rate": s.catch_rate,
                        "false_alarm_rate": s.false_alarm_rate} for s in stats],
        "photos": [{**dataclasses.asdict(r), "expected": sorted(r.expected),
                    "predicted": sorted(r.predicted)} for r in outcome.results],
        "skipped": outcome.skipped,
        "input_warnings": outcome.warnings,
        "costs": dataclasses.asdict(costs) if costs else None,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------- CLI ----------
def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                        help="checkout whose safesite package and COSTS.md to use")
    parser.add_argument("--labels", type=Path, help=f"default: <repo>/{DEFAULT_LABELS.as_posix()}")
    parser.add_argument("--images", type=Path, help="photo folder (default: the labels file's folder)")
    parser.add_argument("--costs", type=Path, help=f"default: <repo>/{DEFAULT_COSTS}")
    parser.add_argument("--detector", default="mock",
                        help="mock | yolo | combined | package.module:Name")
    parser.add_argument("--weights", help="local .pt for --detector yolo (default: download from the Hub)")
    parser.add_argument("--space", metavar="OWNER/NAME",
                        help="score the live Space instead of a local detector; seconds per photo "
                             "are the Space's own processing time (its true CPU)")
    parser.add_argument("--threads", type=int,
                        help="limit CPU threads to the Space's vCPUs (cpu-basic 2, cpu-upgrade 8)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_RUNS,
                        help="untimed detector runs before the first timed photo")
    parser.add_argument("--only-hazards", metavar="ID,ID",
                        help="score only these hazard ids; others are ignored, not false alarms")
    parser.add_argument("--title", default="Qatar test photos", help="report heading")
    parser.add_argument("--json", type=Path, help="also write full results here")
    parser.add_argument("--allow-skipped", action="store_true",
                        help=f"exit {EXIT_OK} even if some photos could not be read "
                             f"(default: exit {EXIT_SKIPPED})")
    return parser.parse_args(argv)


def limit_threads(threads: int | None) -> None:
    if threads is None:
        return
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "SAFESITE_THREADS"):
        os.environ[var] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass  # mock detector: torch not installed, env vars are enough


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))
    limit_threads(args.threads)
    labels_path = args.labels or repo / DEFAULT_LABELS
    costs_path = args.costs or repo / DEFAULT_COSTS
    rule_ids = known_hazard_ids()
    label_ids = labelable_ids(rule_ids)
    only = parse_only_hazards(args.only_hazards, label_ids)
    try:
        photos = load_labels(labels_path, label_ids)
    except (OSError, LabelError) as exc:
        print(f"error: can't use {labels_path}:\n{exc}", file=sys.stderr)
        return EXIT_BAD_LABELS
    image_dir = args.images or labels_path.parent
    if args.space:  # the live Space decides; rule ids are still read from this checkout
        pipeline, raw = run_space(args.space, photos, image_dir, args.warmup)
    else:
        pipeline = load_pipeline(args.detector, args.weights)
        raw = run_all(pipeline, photos, image_dir, args.warmup)
    scored = photos if only is None else [
        dataclasses.replace(p, hazards=p.hazards & only) for p in photos]
    outcome = RunOutcome(restrict(raw.results, only), raw.skipped,
                         unproducible_warnings(scored, rule_ids) + raw.warnings)
    stats = hazard_stats(outcome.results, report_ids(outcome.results, label_ids, only))
    summary = summarize(outcome.results, stats)
    costs = parse_costs(costs_path.read_text(encoding="utf-8")) if costs_path.is_file() else None
    report = render_report(pipeline, stats, summary, outcome, cost_lines(costs, costs_path),
                           only, args.title)
    print(report)
    if args.json:
        write_json(args.json, pipeline, stats, summary, outcome, costs, only, args.threads)
    if not outcome.results:
        return EXIT_NO_PHOTOS
    return EXIT_SKIPPED if outcome.skipped and not args.allow_skipped else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
