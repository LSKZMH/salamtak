"""Run the YOLOE-26 open-vocab finder on example photos and report CPU latency.

    python scripts/bench_openvocab.py                      # examples/*.jpg, mode from file name
    python scripts/bench_openvocab.py my.jpg --mode site --runs 10

Mode "auto" uses Home prompts for files whose name starts with "home", else Site.
Latency is wall-clock time of one detector call (pre-process + inference + NMS) on CPU,
after warm-up runs. Model load and the one-off text encoding are reported separately.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from safesite import openvocab  # noqa: E402

LFS_POINTER_PREFIX = b"version https://git-lfs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="*", type=Path, help="photos (default: examples/*.jpg)")
    parser.add_argument("--mode", choices=("auto", "site", "home"), default="auto")
    parser.add_argument("--runs", type=int, default=5, help="timed runs per photo")
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs per photo")
    parser.add_argument("--weights", default=openvocab.WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=openvocab.IMG_SIZE)
    parser.add_argument("--conf", type=float, default=openvocab.CONF_THRESHOLD)
    args = parser.parse_args(argv)
    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be >= 1 and --warmup >= 0")
    return args


def mode_for(path: Path, mode: str) -> str:
    if mode != "auto":
        return mode
    return "home" if path.name.lower().startswith("home") else "site"


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def time_calls(detector: openvocab.OpenVocabDetector, image: Image.Image, mode: str,
               runs: int, warmup: int) -> tuple[list[float], list]:
    for _ in range(warmup):
        detector(image, mode)
    timings_ms: list[float] = []
    detections: list = []
    for _ in range(runs):
        start = time.perf_counter()
        detections = detector(image, mode)
        timings_ms.append((time.perf_counter() - start) * 1000)
    return timings_ms, detections


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = args.images or sorted((ROOT / "examples").glob("*.jpg"))
    if not paths:
        print("No images found. Pass photo paths or add examples/*.jpg.", file=sys.stderr)
        return 1

    import torch  # deferred so --help works without torch

    print(f"CPU: {platform.processor() or platform.machine()} | torch threads: "
          f"{torch.get_num_threads()} | torch {torch.__version__}")
    start = time.perf_counter()
    detector = openvocab.OpenVocabDetector(weights=args.weights, conf=args.conf, imgsz=args.imgsz)
    print(f"Loaded {detector.name} + encoded prompts in {time.perf_counter() - start:.1f} s "
          f"(imgsz={args.imgsz}, conf>={args.conf})\n")

    medians: list[float] = []
    for path in paths:
        if not path.is_file():
            print(f"skip {path}: not a file")
            continue
        if is_lfs_pointer(path):
            print(f"skip {path}: Git LFS pointer, run `git lfs pull` or pass the real photo")
            continue
        mode = mode_for(path, args.mode)
        with Image.open(path) as img:
            image = img.convert("RGB")
        timings, detections = time_calls(detector, image, mode, args.runs, args.warmup)
        median = statistics.median(timings)
        medians.append(median)
        found = ", ".join(f"{d.label} {d.confidence:.2f}" for d in detections) or "nothing"
        print(f"{path.name} [{mode}] {image.width}x{image.height}: median {median:.0f} ms "
              f"(min {min(timings):.0f}, max {max(timings):.0f}, n={len(timings)})")
        print(f"    found: {found}")

    if not medians:
        print("No photos were timed.", file=sys.stderr)
        return 1
    print(f"\nOverall median over {len(medians)} photo(s): {statistics.median(medians):.0f} ms "
          f"per call on CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
