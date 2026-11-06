"""Bench YOLOE-26 text prompts on an image-level test set and decide which ones ship.

    python scripts/bench_prompts.py C:/sqbench                 # run YOLOE, then report
    python scripts/bench_prompts.py C:/sqbench --from-scores    # re-report from saved scores
    python scripts/bench_prompts.py C:/sqbench --out docs/prompt_bench.md

Bench layout (images are never committed)::

    <bench>/<label>/pos/*.jpg   photos that clearly contain <label>
    <bench>/neg/*.jpg           ordinary / hard-negative site photos
    <bench>/labels.csv          optional: file,labels  e.g. "power_line/pos/a.jpg,utility_pole;ladder"
                                (extra labels also present in an image, so they are not false alarms)

One YOLOE pass per image with every phrasing as a class (YOLOE scores classes
independently), keeping the max confidence per phrasing; every threshold and union
is then computed from those scores, so the sweep costs one pass.

Metrics are image-level, per phrasing (or union of two phrasings for one label):
  hit rate = share of images containing the label with >= 1 detection at the threshold
  FP rate  = share of images NOT containing it (neg + other labels' pos) with >= 1 detection
Ship rule: hit rate >= 0.6 and FP rate <= 0.10 at one threshold.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safesite import openvocab  # noqa: E402
from safesite.types import normalize_label  # noqa: E402

# label -> phrasings to try (from chat-9-detect-app-v2.md)
NEW_PHRASINGS: Mapping[str, tuple[str, ...]] = {
    "power_line": ("power lines", "overhead power cable", "electric wires"),
    "utility_pole": ("electricity pole", "utility pole"),
    "crane_hook": ("crane hook", "load hanging from a crane", "suspended load"),
    "welding": ("welding sparks", "person welding"),
    "fire": ("fire", "flames"),
    "gas_cylinder": ("gas cylinder", "oxygen and acetylene cylinders"),
    "fire_extinguisher": ("fire extinguisher",),
    "rebar": ("exposed rebar", "steel reinforcement bars sticking up"),
    "floor_opening": ("hole in the floor", "floor opening"),
    "trench": ("open trench", "excavation pit"),
}
# Prompts already shipped in SITE_PROMPTS that we re-check for false alarms.
EXISTING_PHRASINGS: Mapping[str, tuple[str, ...]] = {
    "harness": ("safety harness",),
    "scaffold": ("scaffolding",),
    "ladder": ("ladder",),
}
THRESHOLDS: tuple[float, ...] = tuple(round(0.15 + 0.05 * i, 2) for i in range(10))  # 0.15..0.60
SCORE_FLOOR = 0.10  # collect everything above this once; thresholds are applied afterwards
MIN_HIT_RATE = 0.6
MAX_FP_RATE = 0.10
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SCORES_FILE = "scores.json"

Scores = Mapping[str, Mapping[str, float]]  # image -> phrasing -> max confidence


@dataclass(frozen=True)
class OperatingPoint:
    label: str
    phrasings: tuple[str, ...]
    threshold: float
    hit_rate: float
    fp_rate: float
    fp_rate_neg: float
    n_pos: int
    n_other: int
    n_neg: int

    @property
    def ships(self) -> bool:
        return self.n_pos > 0 and self.hit_rate >= MIN_HIT_RATE and self.fp_rate <= MAX_FP_RATE


# ---------- bench set ----------

def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def load_truth(bench: Path) -> dict[str, frozenset[str]]:
    """Relative image path -> set of labels it contains (folder label + labels.csv extras)."""
    truth: dict[str, set[str]] = {}
    for label_dir in sorted(p for p in bench.iterdir() if p.is_dir()):
        if label_dir.name == "neg":
            for img in _images(label_dir):
                truth[img.relative_to(bench).as_posix()] = set()
        for img in _images(label_dir / "pos"):
            truth[img.relative_to(bench).as_posix()] = {label_dir.name}
    extras = bench / "labels.csv"
    if extras.is_file():
        with extras.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = row["file"].strip().replace("\\", "/")
                if key not in truth:
                    raise ValueError(f"labels.csv names {key!r}, which is not in the bench set")
                truth[key] |= {x.strip() for x in row["labels"].split(";") if x.strip()}
    return {k: frozenset(v) for k, v in truth.items()}


# ---------- scoring ----------

def max_conf_by_prompt(detections: Sequence, prompts: Sequence[str]) -> dict[str, float]:
    """Detections labelled with their (normalized) prompt -> {prompt: max confidence, 0 if none}."""
    by_label = {normalize_label(p): p for p in prompts}
    best = {p: 0.0 for p in prompts}
    for det in detections:
        prompt = by_label.get(det.label)
        if prompt is not None:
            best[prompt] = max(best[prompt], round(det.confidence, 4))
    return best


def run_yoloe(bench: Path, images: Sequence[str], prompts: Sequence[str], weights: str,
              imgsz: int) -> dict[str, dict[str, float]]:
    from PIL import Image

    model = openvocab._load_yoloe(weights)
    model.set_classes(list(prompts), model.get_text_pe(list(prompts)))
    identity = {p: p for p in prompts}
    scores: dict[str, dict[str, float]] = {}
    start = time.perf_counter()
    for i, rel in enumerate(images, 1):
        with Image.open(bench / rel) as img:
            rgb = img.convert("RGB")
        result = model.predict(rgb, conf=SCORE_FLOOR, imgsz=imgsz, device="cpu", verbose=False)[0]
        boxes = result.boxes
        dets = openvocab.to_detections(boxes.xyxyn.tolist(), boxes.conf.tolist(),
                                       boxes.cls.tolist(), tuple(prompts), identity)
        scores[rel] = max_conf_by_prompt(dets, prompts)
        if i % 25 == 0:
            print(f"  {i}/{len(images)} images ({time.perf_counter() - start:.0f} s)", flush=True)
    return scores


# ---------- metrics ----------

def fires(image_scores: Mapping[str, float], phrasings: Sequence[str], threshold: float) -> bool:
    return any(image_scores.get(p, 0.0) >= threshold for p in phrasings)


def operating_point(label: str, phrasings: Sequence[str], threshold: float, scores: Scores,
                    truth: Mapping[str, frozenset[str]]) -> OperatingPoint:
    pos = [k for k, labels in truth.items() if label in labels]
    others = [k for k, labels in truth.items() if label not in labels]
    negs = [k for k in others if not truth[k]]

    def rate(keys: list[str]) -> float:
        return sum(fires(scores[k], phrasings, threshold) for k in keys) / len(keys) if keys else 0.0

    return OperatingPoint(label, tuple(phrasings), threshold, rate(pos), rate(others), rate(negs),
                          len(pos), len(others), len(negs))


def candidates(phrasings: Sequence[str]) -> list[tuple[str, ...]]:
    """Each phrasing alone, plus every union of two."""
    return [(p,) for p in phrasings] + list(itertools.combinations(phrasings, 2))


def best_point(points: Sequence[OperatingPoint]) -> OperatingPoint:
    """Shipping points beat non-shipping ones; then higher hit rate (minus FP when not
    shipping), lower FP, fewer phrasings (cheaper), higher threshold (safer)."""
    return max(points, key=lambda p: (p.ships, p.hit_rate - (0 if p.ships else p.fp_rate),
                                      -p.fp_rate, -len(p.phrasings), p.threshold))


def sweep(label: str, phrasings: Sequence[str], scores: Scores,
          truth: Mapping[str, frozenset[str]]) -> dict[tuple[str, ...], OperatingPoint]:
    """Best threshold for each candidate (single phrasing or union of two)."""
    return {
        cand: best_point([operating_point(label, cand, t, scores, truth) for t in THRESHOLDS])
        for cand in candidates(phrasings)
    }


# ---------- report ----------

def _table(label: str, points: Mapping[tuple[str, ...], OperatingPoint], best: OperatingPoint) -> list[str]:
    lines = [f"### `{label}` — {'**shipped**' if best.ships else 'not shipped'}", "",
             "| phrasing | best thr | hit rate | FP rate | FP neg | n pos / other | ships |",
             "|---|---|---|---|---|---|---|"]
    for cand, p in points.items():
        lines.append(f"| {' + '.join(cand)} | {p.threshold:.2f} | {p.hit_rate:.2f} | {p.fp_rate:.2f} | "
                     f"{p.fp_rate_neg:.2f} | {p.n_pos} / {p.n_other} | {'yes' if p.ships else 'no'} |")
    return lines + [""]


def report(scores: Scores, truth: Mapping[str, frozenset[str]], weights: str, imgsz: int) -> str:
    n_neg = sum(1 for v in truth.values() if not v)
    lines = [
        "# YOLOE prompt bench",
        "",
        f"Generated by `scripts/bench_prompts.py` with `{weights}`, imgsz {imgsz}, on CPU: "
        f"{len(truth)} images ({n_neg} pure negatives). Images stay outside the repo "
        "(URLs and licences in the bench folder's `sources.csv`).",
        "",
        f"Image-level metrics. **Ship rule:** hit rate ≥ {MIN_HIT_RATE} and false-positive "
        f"rate ≤ {MAX_FP_RATE} at one threshold (swept {THRESHOLDS[0]}–{THRESHOLDS[-1]}). "
        "*FP rate* counts every image that does not contain the label (negatives and other "
        "labels' positives); *FP neg* counts pure negatives only.",
        "",
    ]
    shipped: dict[str, OperatingPoint] = {}
    for title, table in (("New labels", NEW_PHRASINGS), ("Existing prompts (re-check)", EXISTING_PHRASINGS)):
        lines += [f"## {title}", ""]
        for label, phrasings in table.items():
            points = sweep(label, phrasings, scores, truth)
            best = best_point(list(points.values()))
            if best.ships:
                shipped[label] = best
            lines += _table(label, points, best)
    lines += ["## Recommended `OPEN_VOCAB_CUTOFFS` (for Chat 8's `rules.py`)", "",
              "| label | prompts | cut-off | hit rate | FP rate |", "|---|---|---|---|---|"]
    for label, p in shipped.items():
        lines.append(f"| `{label}` | {', '.join(repr(x) for x in p.phrasings)} | {p.threshold:.2f} | "
                     f"{p.hit_rate:.2f} | {p.fp_rate:.2f} |")
    if not shipped:
        lines.append("| — | nothing passed | — | — | — |")
    return "\n".join(lines + [""])


def all_prompts() -> list[str]:
    seen: dict[str, None] = {}
    for table in (NEW_PHRASINGS, EXISTING_PHRASINGS):
        for phrasings in table.values():
            seen.update(dict.fromkeys(phrasings))
    return list(seen)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bench", type=Path, help="bench folder (see module docstring)")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "prompt_bench.md")
    parser.add_argument("--from-scores", action="store_true",
                        help=f"reuse <bench>/{SCORES_FILE} instead of running YOLOE")
    parser.add_argument("--weights", default=openvocab.WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=openvocab.IMG_SIZE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    truth = load_truth(args.bench)
    if not truth:
        print(f"No bench images under {args.bench}", file=sys.stderr)
        return 1
    scores_path = args.bench / SCORES_FILE
    if args.from_scores:
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        missing = set(truth) - set(scores)
        if missing:
            print(f"{len(missing)} bench images have no saved score; rerun without --from-scores",
                  file=sys.stderr)
            return 1
    else:
        print(f"Scoring {len(truth)} images with {len(all_prompts())} prompts ...", flush=True)
        scores = run_yoloe(args.bench, sorted(truth), all_prompts(), args.weights, args.imgsz)
        scores_path.write_text(json.dumps(scores, indent=1), encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report(scores, truth, args.weights, args.imgsz), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
