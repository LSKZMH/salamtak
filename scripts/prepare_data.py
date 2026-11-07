"""Download datasets, inspect them, convert to YOLO format, push a private dataset.

Pipeline (README.md "Model and data"). Runs locally on CPU — no HF credits.

Phases (run in order; conversion waits for a confirmed mapping):

    python scripts/prepare_data.py inspect [--work-dir DIR]
        Download each source, print real class names + per-split image/annotation
        counts + license. Tries datasets.load_dataset first, falls back to
        snapshot_download + raw COCO parsing.

    python scripts/prepare_data.py convert [--work-dir DIR]
        Map source classes -> unified classes (MAPPING below, filled in after the
        mapping is signed off), convert to YOLO with train/val splits + data.yaml.

    python scripts/prepare_data.py push --repo CAPTAIN/team-NN-safesite-data [--work-dir DIR]
        Create the PRIVATE dataset repo and upload the YOLO dataset + card.

--work-dir holds downloads and the YOLO output (default "data", i.e. repo-local
and gitignored). On Windows keep it SHORT (e.g. C:\\sqdata) to stay under the
260-char path limit, since the Roboflow filenames are long.

Sources:
  - keremberke/hard-hat-detection                   (COCO zips; main set, ~19.7k imgs)
  - keremberke/construction-safety-object-detection (COCO zips; vests/machinery)
  - Francesco/construction-safety-gsnvb             (HF parquet; small)

The datasets ship an old loading .py script. It is never executed here: the COCO
sources are parsed from their raw _annotations.coco.json, and the parquet source
is read with datasets.load_dataset.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

SOURCES: list[dict] = [
    {"repo": "keremberke/hard-hat-detection", "kind": "coco-zip"},
    {"repo": "keremberke/construction-safety-object-detection", "kind": "coco-zip"},
    {"repo": "Francesco/construction-safety-gsnvb", "kind": "hf-parquet"},
]

# Short, filesystem-safe alias per source (keeps output paths well under MAX_PATH).
ALIAS = {
    "keremberke/hard-hat-detection": "hh",
    "keremberke/construction-safety-object-detection": "cs",
    "Francesco/construction-safety-gsnvb": "fr",
}

# COCO-zip source splits -> file in the repo. "valid-mini" is skipped.
COCO_SPLIT_FILES = {"train": "data/train.zip", "valid": "data/valid.zip", "test": "data/test.zip"}

# Unified class list + fixed indices. Order == YOLO class id.
UNIFIED = ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle"]
UNIFIED_IDX = {name: i for i, name in enumerate(UNIFIED)}

# Per-source mapping: source-class-name -> unified-class-name, or None to DROP.
# Left empty on purpose: filled in only after `inspect` output is reviewed and the
# mapping is approved. `convert` refuses to run until every source is mapped.
MAPPING: dict[str, dict[str, str | None]] = {
    "keremberke/hard-hat-detection": {
        "hardhat": "helmet",
        "no-hardhat": "no_helmet",
    },
    "keremberke/construction-safety-object-detection": {
        "hardhat": "helmet",
        "no-hardhat": "no_helmet",
        "safety vest": "vest",
        "no-safety vest": "no_vest",
        "person": "person",
        "excavators": "machinery",
        "wheel loader": "machinery",
        "dump truck": "vehicle",
        "mini-van": "vehicle",
        "truck": "vehicle",
        # Dropped (not in the unified set):
        "barricade": None,
        "dumpster": None,
        "gloves": None,
        "mask": None,
        "no-mask": None,
        "safety net": None,
        "safety shoes": None,
    },
    "Francesco/construction-safety-gsnvb": {
        "construction-safety": None,  # Roboflow supercategory placeholder (0 anns)
        "helmet": "helmet",
        "no-helmet": "no_helmet",
        "no-vest": "no_vest",
        "person": "person",
        "vest": "vest",
    },
}

# Which source splits feed which YOLO split.
SPLIT_ROUTING = {"train": "train", "valid": "val", "validation": "val", "test": "train"}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Set from --work-dir in main().
CACHE = Path("data/_sources")   # downloaded + extracted sources
OUT = Path("data/yolo")         # YOLO dataset output


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _alias(repo: str) -> str:
    return ALIAS.get(repo, repo.replace("/", "__"))


def _license_of(repo: str) -> str:
    """License from card metadata, falling back to the Roboflow README line."""
    try:
        info = HfApi().dataset_info(repo)
        cd = info.card_data or {}
        lic = cd.get("license") if hasattr(cd, "get") else getattr(cd, "license", None)
        if lic:
            return str(lic)
    except Exception:  # noqa: BLE001
        pass
    for fn in ("README.dataset.txt", "README.roboflow.txt"):
        try:
            p = hf_hub_download(repo, fn, repo_type="dataset")
            for line in Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("license:"):
                    return line.split(":", 1)[1].strip()
        except Exception:  # noqa: BLE001
            continue
    return "(not declared)"


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract guarding against zip-slip (absolute / .. member paths)."""
    dest = dest.resolve()
    for member in zf.namelist():
        target = (dest / member).resolve()
        if not str(target).startswith(str(dest)):
            raise ValueError(f"unsafe path in archive: {member!r}")
    zf.extractall(dest)


def _download_and_extract_coco(repo: str) -> dict[str, Path]:
    """Download (shared HF cache) + extract each COCO split zip into CACHE/alias."""
    base = CACHE / _alias(repo)
    dirs: dict[str, Path] = {}
    for split, rel in COCO_SPLIT_FILES.items():
        out = base / split
        marker = out / ".extracted"
        if marker.exists():
            dirs[split] = out
            continue
        try:
            zpath = hf_hub_download(repo, rel, repo_type="dataset")  # cached, not copied
        except Exception as e:  # noqa: BLE001
            print(f"    [{split}] download failed: {type(e).__name__}: {e}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            _safe_extract(zf, out)
        marker.write_text("ok", encoding="utf-8")
        dirs[split] = out
    return dirs


def _find_coco_json(root: Path) -> Path | None:
    hits = list(root.rglob("_annotations.coco.json"))
    if hits:
        return hits[0]
    hits = [p for p in root.rglob("*.json") if "coco" in p.name.lower() or "annotation" in p.name.lower()]
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# Inspect
# --------------------------------------------------------------------------- #

def _parquet_category_names(repo: str, d=None) -> list[str]:
    """ClassLabel names for the parquet source.

    Prefer names on the loaded feature; fall back to parsing dataset_info.json
    (some datasets store category as a plain int with names only in metadata).
    """
    if d is not None:
        feat = d.features.get("objects")
        inner = getattr(feat, "feature", None)
        cat = inner.get("category") if isinstance(inner, dict) else None
        names = getattr(cat, "names", None)
        if names:
            return list(names)
    try:
        p = hf_hub_download(repo, "dataset_info.json", repo_type="dataset")
        di = json.loads(Path(p).read_text(encoding="utf-8"))
        found: list = []

        def walk(o: object) -> None:
            if isinstance(o, dict):
                if o.get("_type") == "ClassLabel" and o.get("names"):
                    found.append(o["names"])
                for v in o.values():
                    walk(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v)

        walk(di.get("features", {}))
        if found:
            return list(found[0])
    except Exception:  # noqa: BLE001
        pass
    return []


def inspect_coco(repo: str) -> dict:
    info: dict = {"method": "coco-zip", "categories": {}, "splits": {}}
    dirs = _download_and_extract_coco(repo)
    for split, root in dirs.items():
        cj = _find_coco_json(root)
        if cj is None:
            print(f"    [{split}] no COCO json found under {root}")
            continue
        data = json.loads(cj.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        info["categories"] = cats
        per: dict[str, int] = defaultdict(int)
        for a in data.get("annotations", []):
            per[cats.get(a["category_id"], f"id={a['category_id']}")] += 1
        info["splits"][split] = {"images": len(data.get("images", [])), "per_class": dict(per)}
    return info


def inspect_parquet(repo: str) -> dict:
    from datasets import load_dataset

    ds = load_dataset(repo)
    first = next(iter(ds.values()))
    names = _parquet_category_names(repo, first)
    info: dict = {"method": "load_dataset", "categories": {i: n for i, n in enumerate(names)}, "splits": {}}
    for split, d in ds.items():
        per: dict[str, int] = defaultdict(int)
        for ex in d.select_columns(["objects"]):
            for c in ex["objects"]["category"]:
                per[names[c] if names else f"id={c}"] += 1
        info["splits"][split] = {"images": d.num_rows, "per_class": dict(per)}
    return info


def inspect_source(src: dict) -> dict:
    """Try load_dataset first, fall back to snapshot + COCO parsing."""
    repo = src["repo"]
    if src["kind"] == "hf-parquet":
        try:
            return inspect_parquet(repo)
        except Exception as e:  # noqa: BLE001
            print(f"    load_dataset failed ({type(e).__name__}: {e}); falling back to COCO zips")
            return inspect_coco(repo)
    try:
        from datasets import load_dataset

        load_dataset(repo)  # expected to fail on datasets>=3
        print("    note: load_dataset unexpectedly succeeded; still using COCO parse for exact counts")
    except Exception as e:  # noqa: BLE001
        print(f"    load_dataset not usable ({type(e).__name__}); using snapshot + raw COCO parsing")
    return inspect_coco(repo)


def cmd_inspect(_args: argparse.Namespace) -> None:
    for src in SOURCES:
        repo = src["repo"]
        print("=" * 78)
        print(f"SOURCE: {repo}  (kind={src['kind']})")
        print(f"  license: {_license_of(repo)}")
        info = inspect_source(src)
        print(f"  method: {info['method']}")
        print(f"  class names (source id -> name): {info['categories']}")
        totals: dict[str, int] = defaultdict(int)
        for split, s in info["splits"].items():
            print(f"  [{split}] images={s['images']}  annotations_per_class={s['per_class']}")
            for k, v in s["per_class"].items():
                totals[k] += v
        print(f"  TOTAL annotations per class: {dict(totals)}")
    print("=" * 78)
    print("UNIFIED target classes:", {i: n for i, n in enumerate(UNIFIED)})
    print("Next: fill MAPPING, then `python scripts/prepare_data.py convert`.")


# --------------------------------------------------------------------------- #
# Convert
# --------------------------------------------------------------------------- #

def _check_mapping() -> None:
    missing = [s["repo"] for s in SOURCES if s["repo"] not in MAPPING]
    if missing:
        raise SystemExit(
            "MAPPING is not filled in for: " + ", ".join(missing)
            + "\nRun `inspect`, agree the mapping, then edit MAPPING before converting."
        )
    for repo, m in MAPPING.items():
        for src_name, uni in m.items():
            if uni is not None and uni not in UNIFIED_IDX:
                raise SystemExit(f"{repo}: maps {src_name!r} -> unknown unified class {uni!r}")


def _write_label(label_path: Path, lines: list[str]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _yolo_line(cls_id: int, x: float, y: float, w: float, h: float, iw: int, ih: int) -> str | None:
    # COCO bbox [x_min, y_min, w, h] (absolute) -> YOLO normalized cx, cy, w, h.
    if iw <= 0 or ih <= 0 or w <= 0 or h <= 0:
        return None
    cx = min(max((x + w / 2) / iw, 0.0), 1.0)
    cy = min(max((y + h / 2) / ih, 0.0), 1.0)
    nw, nh = min(w / iw, 1.0), min(h / ih, 1.0)
    return f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def convert_coco(repo: str, counts: dict) -> None:
    import shutil

    m = MAPPING[repo]
    alias = _alias(repo)
    dirs = _download_and_extract_coco(repo)
    for split, root in dirs.items():
        yolo_split = SPLIT_ROUTING.get(split)
        if yolo_split is None:
            continue
        cj = _find_coco_json(root)
        if cj is None:
            continue
        data = json.loads(cj.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        images = {im["id"]: im for im in data.get("images", [])}
        anns_by_img: dict[int, list] = defaultdict(list)
        for a in data.get("annotations", []):
            anns_by_img[a["image_id"]].append(a)
        img_root = cj.parent
        for i, (im_id, im) in enumerate(images.items()):
            lines: list[str] = []
            for a in anns_by_img.get(im_id, []):
                uni = m.get(cats.get(a["category_id"]))
                if uni is None:
                    continue
                bx, by, bw, bh = a["bbox"]
                line = _yolo_line(UNIFIED_IDX[uni], bx, by, bw, bh, im["width"], im["height"])
                if line:
                    lines.append(line)
                    counts[yolo_split][uni] += 1
            src_img = img_root / im["file_name"]
            if not src_img.exists():
                continue
            stem = f"{alias}_{split}_{i:06d}"
            dst_img = OUT / "images" / yolo_split / f"{stem}{src_img.suffix.lower()}"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_img, dst_img)
            _write_label(OUT / "labels" / yolo_split / f"{stem}.txt", lines)


def convert_parquet(repo: str, counts: dict) -> None:
    from datasets import load_dataset

    m = MAPPING[repo]
    alias = _alias(repo)
    ds = load_dataset(repo)
    first = next(iter(ds.values()))
    names = _parquet_category_names(repo, first)
    if not names:
        raise SystemExit(f"{repo}: could not resolve category names")
    for split, d in ds.items():
        yolo_split = SPLIT_ROUTING.get(split)
        if yolo_split is None:
            continue
        for idx, ex in enumerate(d):
            iw, ih = ex.get("width"), ex.get("height")
            img = ex["image"]
            if not iw or not ih:
                iw, ih = img.width, img.height
            lines: list[str] = []
            for cat, bbox in zip(ex["objects"]["category"], ex["objects"]["bbox"]):
                uni = m.get(names[cat])
                if uni is None:
                    continue
                bx, by, bw, bh = bbox
                line = _yolo_line(UNIFIED_IDX[uni], bx, by, bw, bh, iw, ih)
                if line:
                    lines.append(line)
                    counts[yolo_split][uni] += 1
            stem = f"{alias}_{split}_{idx:06d}"
            dst_img = OUT / "images" / yolo_split / f"{stem}.jpg"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            img.convert("RGB").save(dst_img, "JPEG", quality=92)
            _write_label(OUT / "labels" / yolo_split / f"{stem}.txt", lines)


def cmd_convert(_args: argparse.Namespace) -> None:
    import yaml

    _check_mapping()
    counts: dict[str, dict[str, int]] = {"train": defaultdict(int), "val": defaultdict(int)}
    for src in SOURCES:
        print(f"converting {src['repo']} ...")
        if src["kind"] == "coco-zip":
            convert_coco(src["repo"], counts)
        else:
            convert_parquet(src["repo"], counts)
    OUT.mkdir(parents=True, exist_ok=True)
    data_yaml = {
        "path": str(OUT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(UNIFIED)},
    }
    (OUT / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print("\nFINAL class counts per split:")
    for split in ("train", "val"):
        imgs = len(list((OUT / "images" / split).glob("*"))) if (OUT / "images" / split).exists() else 0
        print(f"  {split}: images={imgs}  boxes_per_class={dict(counts[split])}")
    print("\nSource licenses:")
    for src in SOURCES:
        print(f"  {src['repo']}: {_license_of(src['repo'])}")
    print(f"\ndata.yaml written to {OUT / 'data.yaml'}")


# --------------------------------------------------------------------------- #
# Push
# --------------------------------------------------------------------------- #

def cmd_push(args: argparse.Namespace) -> None:
    if not args.repo:
        raise SystemExit("--repo CAPTAIN/team-NN-safesite-data is required")
    if not (OUT / "data.yaml").exists():
        raise SystemExit("run `convert` first (no data.yaml under the work dir)")
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    lines = ["---", "license: agpl-3.0", "tags:", "  - qai-hf-hackathon-2026", "---", "",
             "# SafeSite Qatar — unified detection dataset", "",
             "YOLO-format merge of the sources below. Unified classes: " + ", ".join(UNIFIED) + ".",
             "", "## Sources & licenses"]
    for src in SOURCES:
        lines.append(f"- `{src['repo']}` — {_license_of(src['repo'])}")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(OUT),
                      commit_message="Add unified YOLO dataset")
    print(f"pushed to https://huggingface.co/datasets/{args.repo}")


# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="SafeSite Qatar dataset prep")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "convert", "push"):
        sp = sub.add_parser(name)
        sp.add_argument("--work-dir", default="data", help="downloads + YOLO output (keep SHORT on Windows)")
        if name == "push":
            sp.add_argument("--repo", help="CAPTAIN/team-NN-safesite-data")
        sp.set_defaults(func={"inspect": cmd_inspect, "convert": cmd_convert, "push": cmd_push}[name])
    args = p.parse_args()

    global CACHE, OUT
    work = Path(args.work_dir)
    CACHE = work / "_sources"
    OUT = work / "yolo"
    args.func(args)


if __name__ == "__main__":
    main()
