"""Outside benchmark: score the SafeSite pipeline on a stratified subset of ConstructionSite 10k.

LouisChen15/ConstructionSite is gated (card license CC BY-NC 4.0). EVALUATION ONLY: never train on
it, never commit its images or derived labels, never upload it. Everything lives under --cs-dir,
outside the repo. Its images come from MOCS, so a detector trained on MOCS may have seen them.

Image-level ground truth per benchmark rule -> our hazard id:
  Rule 1 basic PPE           -> no_helmet  (only when the reason is about a hard hat; Rule 1 images
                                            whose reason is about other PPE are left out entirely)
  Rule 2 harness at height   -> fall_no_harness
  Rule 3 edge protection     -> fall_into_opening  (only with --rules including 3)
  Rule 4 near excavator      -> machinery_zone

Usage (from the repo root; C:\\sqcs is the default --cs-dir):
  python scripts/bench_constructionsite.py prepare             # download test split, write subset
  python scripts/bench_constructionsite.py run --detector yolo --threads 8 --json C:\\sqcs\\cs_before.json
  python scripts/bench_constructionsite.py run --detector yolo --weights v2_best.pt ...
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import random
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET = "LouisChen15/ConstructionSite"
SPLIT_FILE = "test.parquet"
DEFAULT_CS_DIR = Path("C:/sqcs")
RULE_HAZARD = {1: "no_helmet", 2: "fall_no_harness", 3: "fall_into_opening", 4: "machinery_zone"}
DEFAULT_RULES = (1, 2, 4)  # rule 3 only once the floor_opening/trench prompts ship
DEFAULT_PER_RULE, DEFAULT_CLEAN, DEFAULT_SEED = 150, 300, 0
EXAMPLE_REASONS = 10
_NEGATION = r"(?:\bnot\b|\bno\b|\bwithout\b|\bneither\b|\bnor\b|n't\b)"
_HELMET = r"\b(?:hard[\s-]?hats?|helmets?)\b"
HELMET_REASON = re.compile(_NEGATION + r"[^.;]{0,40}?" + _HELMET, re.IGNORECASE)
IMAGE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
CSV_HEADER = ("filename", "hazards", "taken_at", "wbgt_c", "outdoor", "mode", "notes")


@dataclass(frozen=True)
class ImageLabels:
    image_id: str
    hazards: frozenset[str]  # our hazard ids that are present and scored
    rule1_non_helmet: bool  # Rule 1 violated for a reason other than a hard hat: left out
    any_violation: bool  # any of the four rules violated (scored or not)


# ---------- pure parts ----------
def violated(value: Any) -> bool:
    """A rule_N_violation cell: None, or {bounding_box, reason}; empty both = no violation."""
    if not value:
        return False
    return bool(value.get("reason") or value.get("bounding_box"))


def is_helmet_reason(reason: str | None) -> bool:
    return bool(reason) and HELMET_REASON.search(reason) is not None


def parse_rules(text: str) -> tuple[int, ...]:
    try:
        rules = tuple(sorted({int(part) for part in text.split(",") if part.strip()}))
    except ValueError:
        raise SystemExit(f"--rules must be numbers like 1,2,4, got {text!r}") from None
    bad = [r for r in rules if r not in RULE_HAZARD]
    if bad or not rules:
        raise SystemExit(f"--rules: unknown rule(s) {bad or text!r}; use 1-4")
    return rules


def image_labels(record: dict[str, Any], rules: Sequence[int]) -> ImageLabels:
    found = {n: record.get(f"rule_{n}_violation") for n in RULE_HAZARD}
    hit = {n for n, v in found.items() if violated(v)}
    rule1_other = 1 in hit and not is_helmet_reason((found[1] or {}).get("reason"))
    hazards = frozenset(RULE_HAZARD[n] for n in hit if n in rules and not (n == 1 and rule1_other))
    return ImageLabels(str(record["image_id"]), hazards, rule1_other, bool(hit))


def _eligible(labels: Iterable[ImageLabels]) -> list[ImageLabels]:
    return sorted((x for x in labels if not x.rule1_non_helmet), key=lambda x: x.image_id)


def stratified_sample(labels: Iterable[ImageLabels], hazards: Sequence[str], per_rule: int,
                      clean: int, seed: int) -> list[ImageLabels]:
    """Up to per_rule positives for each hazard plus `clean` violation-free images, drawn with a
    fixed seed from id-sorted pools (so the input order doesn't matter). Sorted by image_id."""
    pool = _eligible(labels)
    rng = random.Random(seed)
    chosen: dict[str, ImageLabels] = {}
    for hazard in hazards:
        positives = [x for x in pool if hazard in x.hazards]
        chosen.update((x.image_id, x) for x in rng.sample(positives, min(per_rule, len(positives))))
    no_violation = [x for x in pool if not x.any_violation]
    chosen.update((x.image_id, x) for x in rng.sample(no_violation, min(clean, len(no_violation))))
    return sorted(chosen.values(), key=lambda x: x.image_id)


def pool_counts(labels: Iterable[ImageLabels], hazards: Sequence[str]) -> dict[str, int]:
    labels = list(labels)
    pool = _eligible(labels)
    counts = {h: sum(h in x.hazards for x in pool) for h in hazards}
    counts["clean"] = sum(not x.any_violation for x in pool)
    counts["rule1_not_helmet_excluded"] = sum(x.rule1_non_helmet for x in labels)
    return counts


def image_filename(image_id: str) -> str:
    if not IMAGE_ID.match(image_id):
        raise ValueError(f"unexpected image_id {image_id!r}")
    return f"images/{image_id}.jpg"


def labels_csv(sample: Iterable[ImageLabels]) -> str:
    """evaluate.py labels: taken_at blank (no heat rule), outdoor, site mode."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    for x in sorted(sample, key=lambda x: x.image_id):
        writer.writerow((image_filename(x.image_id), ";".join(sorted(x.hazards)), "", "", "true",
                         "site", "ConstructionSite test"))
    return out.getvalue()


def evaluate_args(cs_dir: Path, hazards: Sequence[str], detector: str, weights: str | None,
                  threads: int | None, json_path: Path | None) -> list[str]:
    args = ["--labels", str(cs_dir / "subset" / "labels.csv"), "--only-hazards", ",".join(hazards),
            "--detector", detector, "--title", "ConstructionSite test subset (outside benchmark)"]
    if weights:
        args += ["--weights", weights]
    if threads:
        args += ["--threads", str(threads)]
    if json_path:
        args += ["--json", str(json_path)]
    return args


def rule1_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    reasons = [(r.get("rule_1_violation") or {}).get("reason") or "" for r in records
               if violated(r.get("rule_1_violation"))]
    helmet = [x for x in reasons if is_helmet_reason(x)]
    other = [x for x in reasons if not is_helmet_reason(x)]
    return {"rule1_images": len(reasons), "about_hard_hat": len(helmet), "other_ppe": len(other),
            "examples_hard_hat": helmet[:EXAMPLE_REASONS], "examples_other": other[:EXAMPLE_REASONS]}


# ---------- I/O ----------
def download(cs_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    try:
        return Path(hf_hub_download(DATASET, SPLIT_FILE, repo_type="dataset",
                                    local_dir=str(cs_dir / "raw")))
    except GatedRepoError:
        raise SystemExit(f"{DATASET} is gated and this token has no access: the captain must log in "
                         f"as plitip and accept the conditions at https://huggingface.co/datasets/"
                         f"{DATASET}") from None


def read_annotations(parquet: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.ParquetFile(parquet)
    columns = [c for c in table.schema_arrow.names if c != "image"]
    return table.read(columns=columns).to_pylist()


def extract_images(parquet: Path, ids: set[str], out_dir: Path) -> int:
    """Write the chosen images' original JPEG bytes. Returns how many files exist afterwards."""
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    for batch in pq.ParquetFile(parquet).iter_batches(columns=["image_id", "image"], batch_size=64):
        for row in batch.to_pylist():
            image_id = str(row["image_id"])
            if image_id not in ids:
                continue
            target = out_dir.parent / image_filename(image_id)
            if not target.exists():
                target.write_bytes(row["image"]["bytes"])
    return sum((out_dir.parent / image_filename(i)).is_file() for i in ids)


def prepare(args: argparse.Namespace) -> int:
    rules = parse_rules(args.rules)
    hazards = [RULE_HAZARD[n] for n in rules]
    parquet = download(args.cs_dir)
    records = read_annotations(parquet)
    example = {k: v for k, v in records[0].items() if k != "image"}
    print("fields:", ", ".join(records[0]), "\nexample record:", json.dumps(example, indent=1)[:1500])
    labels = [image_labels(r, rules) for r in records]
    sample = stratified_sample(labels, hazards, args.per_rule, args.clean, args.seed)
    subset = args.cs_dir / "subset"
    written = extract_images(parquet, {x.image_id for x in sample}, subset / "images")
    (subset / "labels.csv").write_text(labels_csv(sample), encoding="utf-8")
    summary = {
        "dataset": DATASET, "split": SPLIT_FILE, "records": len(records), "rules": list(rules),
        "seed": args.seed, "per_rule": args.per_rule, "clean": args.clean,
        "pool": pool_counts(labels, hazards),
        "sample": {**{h: sum(h in x.hazards for x in sample) for h in hazards},
                   "clean": sum(not x.any_violation for x in sample), "images": len(sample)},
        "images_written": written, "rule1": rule1_summary(records),
    }
    (subset / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rule1"}, indent=2))
    r1 = summary["rule1"]
    print(f"Rule 1: {r1['rule1_images']} images, {r1['about_hard_hat']} about hard hats, "
          f"{r1['other_ppe']} about other PPE (left out). Other-PPE examples:")
    print("\n".join(f"  - {x}" for x in r1["examples_other"]))
    return 0 if written == len(sample) else 1


def _evaluate_module() -> Any:
    spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).with_name("evaluate.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = module
    spec.loader.exec_module(module)
    return module


def run(args: argparse.Namespace) -> int:
    hazards = [RULE_HAZARD[n] for n in parse_rules(args.rules)]
    if not (args.cs_dir / "subset" / "labels.csv").is_file():
        raise SystemExit("no subset yet: run `prepare` first")
    return _evaluate_module().main(
        evaluate_args(args.cs_dir, hazards, args.detector, args.weights, args.threads, args.json))


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--cs-dir", type=Path, default=DEFAULT_CS_DIR, help="outside the repo")
    parser.add_argument("--rules", default=",".join(map(str, DEFAULT_RULES)),
                        help="benchmark rules to score (add 3 once floor_opening/trench ship)")
    parser.add_argument("--per-rule", type=int, default=DEFAULT_PER_RULE)
    parser.add_argument("--clean", type=int, default=DEFAULT_CLEAN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--detector", default="yolo", help="passed to evaluate.py")
    parser.add_argument("--weights", help="local .pt, passed to evaluate.py")
    parser.add_argument("--threads", type=int, help="the Space's vCPU count")
    parser.add_argument("--json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if Path(args.cs_dir).resolve().is_relative_to(Path(__file__).resolve().parent.parent):
        raise SystemExit("--cs-dir must be outside the repo (the benchmark is never committed)")
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    sys.exit(main())
