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
Ship rule: at one threshold above YOLOE's own floor (openvocab.CONF_THRESHOLD), hit rate >= 0.6,
FP rate <= 0.10 and FP rate on plain site photos (neg/neg_site_*, at least one needed) <= 0.10.
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
# Round 1 = the brief's phrasings; round 2 (after "+") added where round 1 came close or failed.
NEW_PHRASINGS: Mapping[str, tuple[str, ...]] = {
    "power_line": ("power lines", "overhead power cable", "electric wires",
                   "power line", "overhead wires"),
    "utility_pole": ("electricity pole", "utility pole", "power pole", "wooden pole"),
    "crane_hook": ("crane hook", "load hanging from a crane", "suspended load", "hook block", "lifting hook"),
    "welding": ("welding sparks", "person welding"),
    "fire": ("fire", "flames", "open flame", "burning fire"),
    "gas_cylinder": ("gas cylinder", "oxygen and acetylene cylinders", "gas bottle",
                     "compressed gas cylinders"),
    "fire_extinguisher": ("fire extinguisher", "red fire extinguisher", "extinguisher"),
    "rebar": ("exposed rebar", "steel reinforcement bars sticking up"),
    "floor_opening": ("hole in the floor", "floor opening"),
    "trench": ("open trench", "excavation pit", "trench", "deep trench in the ground"),
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
# Extra ship conditions (see docs/prompt_bench.md):
#  - the app sees construction-site photos, so plain site photos must not false-alarm either;
#  - a cut-off at YOLOE's own reporting floor (openvocab.CONF_THRESHOLD) rests on scores the
#    model barely reports and that move when the prompt set changes, so it must sit above it.
SITE_NEG_PREFIX = "neg/neg_site_"
YOLOE_FLOOR = openvocab.CONF_THRESHOLD
# Labels that pass the bench but stay out, because every rule they feed is unjudgeable today.
HELD_BACK: Mapping[str, str] = {
    "welding": "its only reachable rule is no_fire_extinguisher (hot_work_fire also needs a "
               "gas cylinder, which did not pass) and that rule cannot be judged without a "
               "working fire-extinguisher prompt, so it would flag every welding photo",
}
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
    fp_rate_site: float = 0.0  # plain construction-site negatives only
    n_site: int = 0  # how many of those there are; 0 = the site safeguard cannot be checked

    @property
    def ships(self) -> bool:
        return (self.n_pos > 0 and self.hit_rate >= MIN_HIT_RATE and self.fp_rate <= MAX_FP_RATE
                and self.n_site > 0 and self.fp_rate_site <= MAX_FP_RATE
                and self.threshold > YOLOE_FLOOR)


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
    site = [k for k in negs if k.startswith(SITE_NEG_PREFIX)]

    def rate(keys: list[str]) -> float:
        return sum(fires(scores[k], phrasings, threshold) for k in keys) / len(keys) if keys else 0.0

    return OperatingPoint(label, tuple(phrasings), threshold, rate(pos), rate(others), rate(negs),
                          len(pos), len(others), len(negs), rate(site), len(site))


def candidates(phrasings: Sequence[str]) -> list[tuple[str, ...]]:
    """Each phrasing alone, plus every union of two."""
    return [(p,) for p in phrasings] + list(itertools.combinations(phrasings, 2))


def best_point(points: Sequence[OperatingPoint]) -> OperatingPoint:
    """Shipping points beat non-shipping ones; then hit rate minus FP rate (Youden's J, so
    neither a few extra hits nor a tiny FP saving dominates), then lower FP, fewer phrasings
    (cheaper), higher threshold (safer)."""
    return max(points, key=lambda p: (p.ships, round(p.hit_rate - p.fp_rate, 6), -p.fp_rate,
                                      -len(p.phrasings), p.threshold))


def deployed_cutoff(label: str) -> float:
    """The cut-off rules.py applies to a YOLOE label today."""
    from safesite import rules

    return rules.OPEN_VOCAB_CUTOFFS.get(label, rules.OPEN_VOCAB_MIN_CONFIDENCE)


def sweep(label: str, phrasings: Sequence[str], scores: Scores,
          truth: Mapping[str, frozenset[str]]) -> dict[tuple[str, ...], OperatingPoint]:
    """Best threshold for each candidate (single phrasing or union of two)."""
    return {
        cand: best_point([operating_point(label, cand, t, scores, truth) for t in THRESHOLDS])
        for cand in candidates(phrasings)
    }


# ---------- report ----------

def _verdict(best: OperatingPoint, existing: bool) -> str:
    if existing:  # already in SITE_PROMPTS: a false-alarm re-check at the deployed cut-off
        return (f"already shipped; at its deployed cut-off {best.threshold:.2f}: FP rate "
                f"{best.fp_rate:.2f}, FP neg {best.fp_rate_neg:.2f}")
    if best.n_pos == 0:
        return "not benched (no positive photos)"
    if best.ships and best.label in HELD_BACK:
        return f"passes, **held back**: {HELD_BACK[best.label]}"
    return "**shipped**" if best.ships else "not shipped"


def _table(label: str, points: Mapping[tuple[str, ...], OperatingPoint], best: OperatingPoint,
           existing: bool = False) -> list[str]:
    lines = [f"### `{label}` — {_verdict(best, existing)}", "",
             "| phrasing | best thr | hit rate | FP rate | FP neg | FP site | n pos / other | ships |",
             "|---|---|---|---|---|---|---|---|"]
    for cand, p in points.items():
        lines.append(f"| {' + '.join(cand)} | {p.threshold:.2f} | {p.hit_rate:.2f} | {p.fp_rate:.2f} | "
                     f"{p.fp_rate_neg:.2f} | {p.fp_rate_site:.2f} | {p.n_pos} / {p.n_other} | "
                     f"{'yes' if p.ships else 'no'} |")
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
        f"Image-level metrics. **Ship rule:** at one threshold (swept {THRESHOLDS[0]}–{THRESHOLDS[-1]}), "
        f"hit rate ≥ {MIN_HIT_RATE}, FP rate ≤ {MAX_FP_RATE} **and** FP site ≤ {MAX_FP_RATE}, with "
        f"the threshold above YOLOE's own reporting floor ({YOLOE_FLOOR}). *FP rate* counts every "
        "image that does not contain the label (negatives and other labels' positives); *FP neg* "
        "pure negatives only; *FP site* the plain construction-site photos among them (what the "
        "app actually sees). The best threshold per row is the one with the highest hit rate − "
        "FP rate among passing thresholds. Labels that pass but feed only rules that cannot be "
        "judged yet are held back.",
        "",
        "Caveat: small sets (9–29 positives per label), and phrasings were picked on this same "
        "set, so treat rates as ±0.1. Weights: YOLOE-26n (the Space's model).",
        "",
    ]
    shipped: dict[str, OperatingPoint] = {}
    for title, table, existing in (("New labels", NEW_PHRASINGS, False),
                                   ("Existing prompts (false-alarm re-check)", EXISTING_PHRASINGS, True)):
        lines += [f"## {title}", ""]
        for label, phrasings in table.items():
            if existing:  # measure at the cut-off actually used, not the best one
                cut = deployed_cutoff(label)
                points = {c: operating_point(label, c, cut, scores, truth) for c in candidates(phrasings)}
            else:
                points = sweep(label, phrasings, scores, truth)
            best = best_point(list(points.values()))
            if best.ships and not existing and label not in HELD_BACK:
                shipped[label] = best
            lines += _table(label, points, best, existing)
    lines += ["## Recommended `OPEN_VOCAB_CUTOFFS` (for Chat 8's `rules.py`)", "",
              "| label | prompts | cut-off | hit rate | FP rate |", "|---|---|---|---|---|"]
    for label, p in shipped.items():
        lines.append(f"| `{label}` | {', '.join(repr(x) for x in p.phrasings)} | {p.threshold:.2f} | "
                     f"{p.hit_rate:.2f} | {p.fp_rate:.2f} |")
    if not shipped:
        lines.append("| — | nothing passed | — | — | — |")
    return "\n".join(lines + [""])


# What we recommend for rules.OPEN_VOCAB_CUTOFFS (from the sweep above, prompts as shipped).
RECOMMENDED_CUTOFFS: Mapping[str, float] = {"utility_pole": 0.20, "rebar": 0.20}


def production_scores(bench: Path, images: Sequence[str], weights: str) -> dict[str, dict[str, float]]:
    """Run the app's own OpenVocabDetector (Site prompts as shipped) -> image -> label -> max conf."""
    from PIL import Image

    finder = openvocab.OpenVocabDetector(weights=weights)
    out: dict[str, dict[str, float]] = {}
    for rel in images:
        with Image.open(bench / rel) as img:
            dets = finder(img.convert("RGB"), "site")
        best: dict[str, float] = {}
        for d in dets:
            best[d.label] = max(best.get(d.label, 0.0), round(d.confidence, 4))
        out[rel] = best
    return out


def _rates(label: str, cut: float, scores: Scores, truth: Mapping[str, frozenset[str]]) -> str:
    p = operating_point(label, (label,), cut, scores, truth)
    return f"{cut:.2f} | {p.hit_rate:.2f} | {p.fp_rate:.2f} | {p.fp_rate_site:.2f}"


def production_section(scores: Scores, truth: Mapping[str, frozenset[str]]) -> list[str]:
    """Shipped prompt set as the app runs it: rates at the cut-off rules.py applies today
    (deployed) and at the recommended one, so the doc can never describe a different app."""
    missing = set(truth) - set(scores)
    if missing:
        raise ValueError(f"{len(missing)} bench images have no production score; re-run with --production")
    lines = ["## Production check (shipped Site prompt set, app's `OpenVocabDetector`)", "",
             "Scores move a little when the prompt set changes, so the shipped set is re-run as the "
             "app runs it. *Deployed* = the cut-off `rules.py` applies today.", "",
             "| label | deployed cut-off | hit | FP | FP site | recommended cut-off | hit | FP | FP site |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label in [*RECOMMENDED_CUTOFFS, *EXISTING_PHRASINGS]:
        rec = RECOMMENDED_CUTOFFS.get(label, deployed_cutoff(label))
        lines.append(f"| `{label}` | {_rates(label, deployed_cutoff(label), scores, truth)} | "
                     f"{_rates(label, rec, scores, truth)} |")
    pending = [lab for lab, cut in RECOMMENDED_CUTOFFS.items() if deployed_cutoff(lab) != cut]
    if pending:
        lines += ["", f"Recommended cut-offs for {', '.join(f'`{x}`' for x in pending)} are not yet in "
                  "`rules.OPEN_VOCAB_CUTOFFS` (Chat 8 copies them)."]
    return lines + [""]


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
    parser.add_argument("--production", action="store_true",
                        help="also re-run the shipped Site prompt set through OpenVocabDetector")
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
    text = report(scores, truth, args.weights, args.imgsz)
    prod_path = args.bench / "scores_production.json"
    if args.production:
        print("Re-running the shipped Site prompt set ...", flush=True)
        prod_path.write_text(json.dumps(production_scores(args.bench, sorted(truth), args.weights),
                                        indent=1), encoding="utf-8")
    if prod_path.is_file():
        text += "\n".join(production_section(json.loads(prod_path.read_text(encoding="utf-8")), truth))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
