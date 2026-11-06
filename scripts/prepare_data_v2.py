"""Build the v2 (11-class) YOLO dataset and push it as a private Hub dataset.

Runs locally on CPU (no HF credits). Chat 6 owns this file; v1 stays in prepare_data.py.

    python scripts/prepare_data_v2.py download  [--work-dir C:\\sqdata2]
    python scripts/prepare_data_v2.py inspect   [--work-dir ...]   # class names, counts, licenses
    python scripts/prepare_data_v2.py preview --alias hrn [--n 4]   # draw boxes to _preview/
    python scripts/prepare_data_v2.py convert   [--work-dir ...]   # yolo_v2_gt/ + dedupe + sources.json
    python scripts/prepare_data_v2.py verify-v1 [--work-dir ...]   # v1 val images unchanged?
    python scripts/prepare_data_v2.py push      [--repo plitip/team-qh-091-safesite-data-v2-gt]

Roboflow key: env ROBOFLOW_API_KEY, else one line in <work-dir>/.rfkey. Never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_data as v1  # noqa: E402  (reuse v1 code paths; never edit v1)

# --------------------------------------------------------------------------- #
# Contracts (PLAN-v2.md)
# --------------------------------------------------------------------------- #

CLASSES = ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle",
           "harness", "scaffold", "crane_hook", "utility_pole"]
CLS_IDX = {n: i for i, n in enumerate(CLASSES)}
assert CLASSES[: len(v1.UNIFIED)] == v1.UNIFIED, "ids 0-6 must match v1"

ALIASES = ("hh", "cs", "fr", "chd", "ppe", "mocs", "hrn", "scf")
HF_REPO = {"hh": "keremberke/hard-hat-detection",
           "cs": "keremberke/construction-safety-object-detection",
           "fr": "Francesco/construction-safety-gsnvb"}
ROBOFLOW = {  # alias -> [(workspace, project), fallback...]
    "chd": [("object-detection-qn97p", "construction-hazard-detection"),
            ("side-projects", "construction-hazard-detection")],
    "mocs": [("yisong-li", "mocs")],
    "hrn": [("project-xksuq", "helmet-harness-detection")],
    "scf": [("testscaff", "hybrid-dataset-for-scaffold-detection")],
}
# Version pins (default: newest). mocs v1 is the only version keeping all 13 classes (v3/v4 omit
# hanging head etc.); scf v1 is the full 6,250-image set (later versions are 20-280 image experiments).
ROBOFLOW_VERSION = {"mocs": 1, "scf": 1}
# Sources left out on purpose (reason goes into sources.json and the card).
SKIP_REASONS = {
    "mocs": "Roboflow yisong-li/mocs is a ~200-image sample, not MOCS; every version either omits "
            "'hanging head' (v3, v4) or has none in its export (v1: 8 train images). crane_hook is left to YOLOE.",
}
PPE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"

# Source class name (case-insensitive, stripped) -> v2 class, None = drop.
# hh/cs/fr come from v1's MAPPING unchanged.
V2_MAPPING: dict[str, dict[str, str | None]] = {
    "chd": {"person": "person", "hardhat": "helmet", "no-hardhat": "no_helmet",
            "safety vest": "vest", "no-safety vest": "no_vest", "machinery": "machinery",
            "vehicle": "vehicle", "utility pole": "utility_pole",
            "mask": None, "no-mask": None, "safety cone": None},
    "ppe": {"person": "person", "helmet": "helmet", "no_helmet": "no_helmet", "vest": "vest",
            "no_vest": "no_vest", "forklift": "vehicle", "gloves": None, "boots": None,
            "goggles": None, "no_goggle": None, "no_gloves": None, "no_boots": None},
    "mocs": {"worker": "person", "excavator": "machinery", "bulldozer": "machinery",
             "loader": "machinery", "roller": "machinery", "crane": "machinery",
             "static crane": "machinery", "concrete mixer": "machinery",
             "pump truck": "machinery", "pile driving": "machinery",
             "truck": "vehicle", "other vehicle": "vehicle", "hanging head": "crane_hook"},
    # hrn v14 classes are head / helmet / safety-harness (no "head helmet"). Checked on drawn boxes:
    # "helmet" = head wearing a helmet, "head" = bare head -> no_helmet.
    "hrn": {"safety-harness": "harness", "helmet": "helmet", "head": "no_helmet"},
    "scf": {"scaffold": "scaffold", "diagonal-brace": None, "ledger": None, "standard": None},
}
for _a in HF_REPO:
    V2_MAPPING[_a] = dict(v1.MAPPING[HF_REPO[_a]])

LABELLED: dict[str, list[str]] = {
    "hh": ["helmet", "no_helmet"],
    "cs": ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle"],
    "fr": ["person", "helmet", "no_helmet", "vest", "no_vest"],
    "chd": ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle", "utility_pole"],
    "ppe": ["person", "helmet", "no_helmet", "vest", "no_vest", "vehicle"],
    "mocs": ["person", "machinery", "vehicle", "crane_hook"],
    "hrn": ["helmet", "no_helmet", "harness"],
    "scf": ["scaffold"],
}
ORIGIN = {**{a: f"hf:{r}" for a, r in HF_REPO.items()},
          "ppe": "ultralytics:construction-ppe.yaml",
          **{a: f"roboflow:{ws}/{p}" for a, ((ws, p), *_) in ROBOFLOW.items()}}
LICENSE = {"hh": "CC BY 4.0", "cs": "CC BY 4.0", "fr": "CC (Roboflow 100)", "chd": "CC BY 4.0",
           "ppe": "CC BY 4.0", "mocs": "CC BY 4.0", "hrn": "CC BY 4.0", "scf": "CC BY 4.0"}

SPLIT_ROUTING = {"train": "train", "valid": "val", "validation": "val", "val": "val", "test": "train"}
VAL_CARVE_FRACTION = 0.10
SEED = 0
MOCS_TRAIN_CAP = 6000
CHD_TRAIN_CAP = 15000
TOTAL_TRAIN_LIMIT = 50000
PHASH_MAX_DIST = 4
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

WORK = Path(r"C:\sqdata2")


def src_dir(alias: str) -> Path:
    return WORK / "_sources" / alias


def out_dir() -> Path:
    return WORK / "yolo_v2_gt"


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #

def norm(name: str) -> str:
    return str(name).strip().lower()


def class_map_for(alias: str, names: dict[int, str] | list[str]) -> dict[int, int | None]:
    """Source class id -> v2 id (None = drop). SystemExit on any unlisted class name."""
    mapping = {norm(k): v for k, v in V2_MAPPING[alias].items()}
    items = names.items() if isinstance(names, dict) else enumerate(names)
    out: dict[int, int | None] = {}
    unknown = []
    for sid, name in items:
        if norm(name) not in mapping:
            unknown.append(name)
            continue
        target = mapping[norm(name)]
        out[int(sid)] = None if target is None else CLS_IDX[target]
    if unknown:
        raise SystemExit(f"{alias}: source classes not in V2_MAPPING: {unknown}")
    return out


def alias_of(filename: str) -> str:
    """'chd_valid_000012.jpg' -> 'chd'. ValueError for anything not in ALIASES."""
    alias = Path(filename).name.split("_", 1)[0]
    if alias not in ALIASES:
        raise ValueError(f"no known alias prefix in {filename!r}")
    return alias


def route_split(split: str) -> str | None:
    return SPLIT_ROUTING.get(norm(split))


def carve_val(keys: list[str], fraction: float = VAL_CARVE_FRACTION, seed: int = SEED) -> set[str]:
    """Random `fraction` of keys (seeded) to move from train to val."""
    k = round(len(keys) * fraction)
    return set(random.Random(seed).sample(sorted(keys), k))


def _to_box(vals: list[float]) -> tuple[float, float, float, float] | None:
    """YOLO bbox (4 values) or polygon (2n values) -> normalized cx, cy, w, h."""
    if len(vals) == 4:
        return tuple(vals)  # type: ignore[return-value]
    if len(vals) < 6 or len(vals) % 2:
        return None
    xs, ys = vals[0::2], vals[1::2]
    x0, x1, y0, y1 = max(min(xs), 0.0), min(max(xs), 1.0), max(min(ys), 0.0), min(max(ys), 1.0)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def remap_lines(lines: Iterable[str], cmap: dict[int, int | None]) -> list[str]:
    """Rewrite YOLO label lines with v2 ids; drop mapped-to-None and degenerate boxes."""
    out = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        sid = int(float(parts[0]))
        if sid not in cmap:
            raise SystemExit(f"label class id {sid} not in source names")
        new = cmap[sid]
        if new is None:
            continue
        box = _to_box([float(v) for v in parts[1:]])
        if box is None or box[2] <= 0 or box[3] <= 0:
            continue
        out.append(f"{new} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
    return out


def cap_items(items: list[tuple[str, frozenset[int]]], cap: int, keep_cls: int | None = None,
              seed: int = SEED) -> list[str]:
    """Pick at most `cap` keys: all with `keep_cls` first, then favour rarer classes."""
    if len(items) <= cap:
        return [k for k, _ in items]
    rng = random.Random(seed)
    must = [k for k, c in items if keep_cls is not None and keep_cls in c]
    if len(must) >= cap:
        return sorted(rng.sample(must, cap))
    rest = [(k, c) for k, c in items if not (keep_cls is not None and keep_cls in c)]
    freq = Counter(cls for _, c in rest for cls in c)
    big = max(freq.values(), default=0) + 1
    scored = [(min((freq[x] for x in c), default=big), rng.random(), k) for k, c in rest]
    scored.sort()
    return must + [k for _, _, k in scored[: cap - len(must)]]


def _blocks(h: int) -> list[tuple[int, int]]:
    """Split a 64-bit hash in 5 blocks; Hamming <= 4 implies one block matches exactly."""
    widths, out, shift = (13, 13, 13, 13, 12), [], 0
    for i, w in enumerate(widths):
        out.append((i, (h >> shift) & ((1 << w) - 1)))
        shift += w
    return out


def near_pairs(a: dict[str, int], b: dict[str, int] | None = None,
               max_dist: int = PHASH_MAX_DIST) -> set[tuple[str, str]]:
    """Pairs (ka, kb) with Hamming(a[ka], b[kb]) <= max_dist. b=None: pairs inside a."""
    assert max_dist <= 4, "block index assumes max_dist <= 4"
    target = a if b is None else b
    index: dict[tuple[int, int], list[str]] = defaultdict(list)
    for k, h in target.items():
        for blk in _blocks(h):
            index[blk].append(k)
    pairs: set[tuple[str, str]] = set()
    for ka, ha in a.items():
        for blk in _blocks(ha):
            for kb in index.get(blk, ()):
                if b is None and kb <= ka:
                    continue
                if bin(ha ^ target[kb]).count("1") <= max_dist:
                    pairs.add((ka, kb))
    return pairs


def dedupe(train: dict[str, int], val: dict[str, int], priority: Callable[[str], int],
           max_dist: int = PHASH_MAX_DIST) -> tuple[set[str], set[str]]:
    """(train keys matching any val image, train keys dropped as near-duplicates in train).

    Inside a duplicate cluster the key with the highest priority survives (ties: smallest key).
    Val keys are never dropped.
    """
    vs_val = {ka for ka, _ in near_pairs(train, val, max_dist)}
    rest = {k: h for k, h in train.items() if k not in vs_val}
    parent = {k: k for k in rest}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ka, kb in near_pairs(rest, None, max_dist):
        parent[find(ka)] = find(kb)
    groups: dict[str, list[str]] = defaultdict(list)
    for k in rest:
        groups[find(k)].append(k)
    dupes: set[str] = set()
    for members in groups.values():
        if len(members) > 1:
            keep = min(members, key=lambda k: (-priority(k), k))
            dupes.update(m for m in members if m != keep)
    return vs_val, dupes


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def roboflow_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key and (WORK / ".rfkey").exists():
        key = (WORK / ".rfkey").read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit(f"ROBOFLOW_API_KEY not set and {WORK / '.rfkey'} missing")
    return key


def _newest_version(proj) -> int:
    nums = []
    for v in proj.get_version_information():
        vid = str(v.get("id", "")).rstrip("/").split("/")[-1]
        if vid.isdigit():
            nums.append(int(vid))
    if not nums:
        raise RuntimeError("no versions listed")
    return max(nums)


def download_roboflow(alias: str) -> bool:
    from roboflow import Roboflow

    dest = src_dir(alias)
    if (dest / ".done").exists():
        print(f"[{alias}] already downloaded")
        return True
    rf = Roboflow(api_key=roboflow_key())
    for ws, project in ROBOFLOW[alias]:
        for attempt in (1, 2):
            try:
                proj = rf.workspace(ws).project(project)
                ver = ROBOFLOW_VERSION.get(alias) or _newest_version(proj)
                print(f"[{alias}] {ws}/{project} v{ver} (attempt {attempt})", flush=True)
                proj.version(ver).download("yolov8", location=str(dest), overwrite=True)
                (dest / ".done").write_text(json.dumps({"workspace": ws, "project": project,
                                                        "version": ver}), encoding="utf-8")
                return True
            except Exception as e:  # noqa: BLE001  (report and try again / fall back)
                print(f"[{alias}] {ws}/{project} failed: {type(e).__name__}: {e}", flush=True)
    print(f"[{alias}] SKIPPED after two failures per project")
    return False


def download_ppe() -> None:
    dest = src_dir("ppe")
    if (dest / "construction-ppe" / "data.yaml").exists():
        print("[ppe] already downloaded")
        return
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "construction-ppe.zip"
    if not zpath.exists():
        import urllib.request
        urllib.request.urlretrieve(PPE_URL, zpath)  # noqa: S310 (fixed https URL)
    with zipfile.ZipFile(zpath) as zf:
        v1._safe_extract(zf, dest)


def cmd_download(_args: argparse.Namespace) -> None:
    v1.CACHE = WORK / "_sources"
    for a in ("hh", "cs"):
        v1._download_and_extract_coco(HF_REPO[a])
    download_ppe()
    for alias in ROBOFLOW:
        download_roboflow(alias)


# --------------------------------------------------------------------------- #
# YOLO-format sources (Roboflow yolov8 export, Ultralytics construction-ppe)
# --------------------------------------------------------------------------- #

def yolo_root(alias: str) -> Path | None:
    base = src_dir(alias)
    hits = sorted(base.rglob("data.yaml"))
    return hits[0].parent if hits else None


def yolo_names(root: Path) -> dict[int, str]:
    import yaml

    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    return {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))


def yolo_splits(root: Path) -> dict[str, tuple[Path, Path]]:
    """split -> (images dir, labels dir) for both 'train/images' and 'images/train' layouts."""
    out = {}
    for split in ("train", "valid", "val", "validation", "test"):
        for img, lab in ((root / split / "images", root / split / "labels"),
                         (root / "images" / split, root / "labels" / split)):
            if img.is_dir():
                out[split] = (img, lab)
                break
    return out


def read_source(alias: str) -> tuple[dict[int, str], list[dict]]:
    """Every image of a YOLO-format source as {split, idx, img, lines(v2)}, sorted per split."""
    root = yolo_root(alias)
    if root is None:
        return {}, []
    names = yolo_names(root)
    cmap = class_map_for(alias, names)
    items = []
    for split, (img_dir, lab_dir) in yolo_splits(root).items():
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        for i, img in enumerate(imgs):
            lab = lab_dir / f"{img.stem}.txt"
            raw = lab.read_text(encoding="utf-8").splitlines() if lab.exists() else []
            items.append({"split": split, "idx": i, "img": img, "lines": remap_lines(raw, cmap)})
    return names, items


def plan_source(alias: str, items: list[dict], cap: int | None) -> list[tuple[dict, str]]:
    """Assign each item a YOLO split (routing, 10% carve if no val split, train cap)."""
    routed = [(it, route_split(it["split"])) for it in items]
    routed = [(it, s) for it, s in routed if s is not None]
    key = lambda it: f"{it['split']}_{it['idx']:06d}"  # noqa: E731
    if not any(s == "val" for _, s in routed):
        moved = carve_val([key(it) for it, s in routed if s == "train"])
        routed = [(it, "val" if key(it) in moved else s) for it, s in routed]
    if cap is not None:
        train = [(key(it), frozenset(int(ln.split()[0]) for ln in it["lines"]))
                 for it, s in routed if s == "train"]
        keep = set(cap_items(train, cap, CLS_IDX["crane_hook"] if alias == "mocs" else None))
        routed = [(it, s) for it, s in routed if s == "val" or key(it) in keep]
    return routed


def write_item(alias: str, it: dict, split: str, out: Path) -> None:
    stem = f"{alias}_{it['split']}_{it['idx']:06d}"
    dst = out / "images" / split / f"{stem}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if it["img"].suffix.lower() in (".jpg", ".jpeg"):
        shutil.copyfile(it["img"], dst)
    else:
        from PIL import Image
        with Image.open(it["img"]) as im:
            im.convert("RGB").save(dst, "JPEG", quality=92)
    v1._write_label(out / "labels" / split / f"{stem}.txt", it["lines"])


# --------------------------------------------------------------------------- #
# Inspect / preview
# --------------------------------------------------------------------------- #

def cmd_inspect(_args: argparse.Namespace) -> None:
    v1.CACHE = WORK / "_sources"
    for a in ("hh", "cs"):
        info = v1.inspect_coco(HF_REPO[a])
        print(f"== {a}: {HF_REPO[a]}  license={LICENSE[a]}  classes={list(info['categories'].values())}")
        for s, d in info["splits"].items():
            print(f"   [{s}] images={d['images']} boxes={d['per_class']}")
    info = v1.inspect_parquet(HF_REPO["fr"])
    print(f"== fr: {HF_REPO['fr']}  classes={list(info['categories'].values())}")
    for s, d in info["splits"].items():
        print(f"   [{s}] images={d['images']} boxes={d['per_class']}")
    for alias in ("chd", "ppe", "mocs", "hrn", "scf"):
        root = yolo_root(alias)
        if root is None:
            print(f"== {alias}: NOT DOWNLOADED")
            continue
        names = yolo_names(root)
        meta = src_dir(alias) / ".done"
        print(f"== {alias}: {ORIGIN[alias]} {meta.read_text() if meta.exists() else ''}")
        print(f"   classes={list(names.values())}")
        for split, (img_dir, lab_dir) in yolo_splits(root).items():
            per: Counter = Counter()
            n = 0
            for lab in lab_dir.glob("*.txt"):
                for ln in lab.read_text(encoding="utf-8").splitlines():
                    if ln.split():
                        per[names.get(int(float(ln.split()[0])), "?")] += 1
            n = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            print(f"   [{split}] images={n} boxes={dict(per)}")
        for rd in sorted(root.glob("README*")):
            for line in rd.read_text(encoding="utf-8", errors="replace").splitlines():
                if "licen" in line.lower():
                    print(f"   {rd.name}: {line.strip()}")


def cmd_preview(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw

    root = yolo_root(args.alias)
    names = yolo_names(root)
    img_dir, lab_dir = yolo_splits(root)["train"]
    labs = sorted(p for p in lab_dir.glob("*.txt") if p.read_text(encoding="utf-8").strip())
    want = {norm(c) for c in args.cls} if args.cls else None
    if want:
        labs = [p for p in labs if any(norm(names[int(float(l.split()[0]))]) in want
                                       for l in p.read_text(encoding="utf-8").splitlines() if l.split())]
    out = WORK / "_preview"
    out.mkdir(exist_ok=True)
    for lab in random.Random(SEED).sample(labs, min(args.n, len(labs))):
        img = next(p for p in img_dir.iterdir() if p.stem == lab.stem)
        im = Image.open(img).convert("RGB")
        im.thumbnail((900, 900))
        d, (w, h) = ImageDraw.Draw(im), im.size
        for ln in lab.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            box = _to_box([float(v) for v in p[1:]])
            if not box:
                continue
            cx, cy, bw, bh = box
            d.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                        outline="red", width=3)
            d.text(((cx - bw / 2) * w + 3, (cy - bh / 2) * h + 3), names[int(float(p[0]))], fill="yellow")
        dst = out / f"{args.alias}_{lab.stem[:40]}.jpg"
        im.save(dst)
        print(dst)


# --------------------------------------------------------------------------- #
# Convert + dedupe + sources.json
# --------------------------------------------------------------------------- #

def _phash(path: str) -> tuple[str, int]:
    import imagehash
    from PIL import Image

    with Image.open(path) as im:
        return Path(path).stem, int(str(imagehash.phash(im)), 16)


def hash_folder(folder: Path, cache: Path) -> dict[str, int]:
    from multiprocessing import Pool

    known: dict[str, int] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [str(p) for p in sorted(folder.glob("*.jpg")) if p.stem not in known]
    if todo:
        with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
            for stem, h in pool.imap_unordered(_phash, todo, chunksize=64):
                known[stem] = h
        cache.write_text(json.dumps(known))
    present = {p.stem for p in folder.glob("*.jpg")}
    return {k: v for k, v in known.items() if k in present}


def count_labels(out: Path) -> dict:
    """{split: {alias: {"images": n, "boxes": {class: n}}}} from the written label files."""
    res: dict = {}
    for split in ("train", "val"):
        per: dict = defaultdict(lambda: {"images": 0, "boxes": Counter()})
        for lab in (out / "labels" / split).glob("*.txt"):
            a = alias_of(lab.name)
            per[a]["images"] += 1
            for ln in lab.read_text(encoding="utf-8").splitlines():
                if ln.split():
                    per[a]["boxes"][CLASSES[int(ln.split()[0])]] += 1
        res[split] = {a: {"images": v["images"], "boxes": dict(v["boxes"])} for a, v in sorted(per.items())}
    return res


def cmd_convert(args: argparse.Namespace) -> None:
    import yaml

    out = out_dir()
    if out.exists() and not args.keep:
        shutil.rmtree(out)
    v1.CACHE, v1.OUT = WORK / "_sources", out
    skipped = []
    dummy: dict = {"train": defaultdict(int), "val": defaultdict(int)}
    print("converting hh, cs (v1 code path) ...", flush=True)
    for a in ("hh", "cs"):
        v1.convert_coco(HF_REPO[a], dummy)
    print("converting fr (v1 code path) ...", flush=True)
    v1.convert_parquet(HF_REPO["fr"], dummy)

    plans: dict[str, list[tuple[dict, str]]] = {}
    for alias in ("chd", "ppe", "mocs", "hrn", "scf"):
        if alias in SKIP_REASONS:
            print(f"[{alias}] skipped: {SKIP_REASONS[alias]}")
            skipped.append(alias)
            continue
        _, items = read_source(alias)
        if not items:
            print(f"[{alias}] not downloaded -> skipped")
            skipped.append(alias)
            continue
        plans[alias] = plan_source(alias, items, MOCS_TRAIN_CAP if alias == "mocs" else None)
    v1_train = sum(1 for _ in (out / "images" / "train").glob("*.jpg"))
    total = v1_train + sum(1 for p in plans.values() for _, s in p if s == "train")
    if total > TOTAL_TRAIN_LIMIT and "chd" in plans:
        _, items = read_source("chd")
        plans["chd"] = plan_source("chd", items, CHD_TRAIN_CAP)
        print(f"train total {total} > {TOTAL_TRAIN_LIMIT}: capped chd train at {CHD_TRAIN_CAP}")
    for alias, plan in plans.items():
        print(f"writing {alias}: {len(plan)} images ...", flush=True)
        for it, split in plan:
            write_item(alias, it, split, out)

    print("perceptual hashing ...", flush=True)
    train_h = hash_folder(out / "images" / "train", WORK / "phash_train.json")
    val_h = hash_folder(out / "images" / "val", WORK / "phash_val.json")
    prio = lambda k: len(LABELLED[alias_of(k)])  # noqa: E731
    vs_val, dupes = dedupe(train_h, val_h, prio)
    for stem in vs_val | dupes:
        (out / "images" / "train" / f"{stem}.jpg").unlink(missing_ok=True)
        (out / "labels" / "train" / f"{stem}.txt").unlink(missing_ok=True)
    dd = {"phash_max_hamming": PHASH_MAX_DIST,
          "train_dropped_matching_val": len(vs_val),
          "train_dropped_near_duplicates": len(dupes),
          "train_dropped_matching_val_by_source": dict(Counter(alias_of(k) for k in vs_val)),
          "train_dropped_near_duplicates_by_source": dict(Counter(alias_of(k) for k in dupes))}
    print("dedupe:", dd)

    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": ".", "train": "images/train", "val": "images/val",
         "names": dict(enumerate(CLASSES))}, sort_keys=False), encoding="utf-8")
    counts = count_labels(out)
    write_sources_json(out, counts, dd, skipped)
    print_counts(counts)


def write_sources_json(out: Path, counts: dict, dd: dict, skipped: list[str]) -> None:
    sources = {}
    for a in ALIASES:
        if a in skipped:
            continue
        meta = src_dir(a) / ".done"
        sources[a] = {"origin": ORIGIN[a], "license": LICENSE[a],
                      "labelled_classes": LABELLED[a], "pseudo_labelled_classes": [],
                      "train_images": counts["train"].get(a, {}).get("images", 0),
                      "val_images": counts["val"].get(a, {}).get("images", 0)}
        if meta.exists():
            sources[a]["roboflow_version"] = json.loads(meta.read_text())
    doc = {"classes": CLASSES, "sources": sources,
           "skipped_sources": {a: SKIP_REASONS.get(a, "download failed") for a in skipped},
           "teacher": None, "dedupe": dd}
    (out / "sources.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


def print_counts(counts: dict) -> None:
    for split in ("train", "val"):
        tot: Counter = Counter()
        print(f"\n[{split}]  {'alias':<5} {'images':>7}  boxes")
        for a, v in counts[split].items():
            tot.update(v["boxes"])
            print(f"        {a:<5} {v['images']:>7}  {v['boxes']}")
        print(f"        TOTAL {sum(v['images'] for v in counts[split].values()):>7}  "
              f"{ {c: tot[c] for c in CLASSES} }")


# --------------------------------------------------------------------------- #
# v1 val check + push
# --------------------------------------------------------------------------- #

def cmd_verify_v1(args: argparse.Namespace) -> None:
    """Compare hh/cs/fr val images+labels with the v1 dataset repo (byte hashes)."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    files = HfApi().list_repo_files(args.v1_repo, repo_type="dataset")
    ref = WORK / "_v1ref"
    if "val.zip" in files:
        with zipfile.ZipFile(hf_hub_download(args.v1_repo, "val.zip", repo_type="dataset")) as zf:
            v1._safe_extract(zf, ref)
    else:
        snapshot_download(args.v1_repo, repo_type="dataset", local_dir=ref,
                          allow_patterns=["images/val/*", "labels/val/*"])
    root = next(p.parent for p in ref.rglob("images") if (p / "val").is_dir())
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()  # noqa: E731,S324
    old = {p.name: md5(p) for d in ("images", "labels") for p in (root / d / "val").iterdir()}
    new = {p.name: md5(p) for d in ("images", "labels") for p in (out_dir() / d / "val").iterdir()
           if alias_of(p.name) in ("hh", "cs", "fr")}
    same = sum(1 for k in old if new.get(k) == old[k])
    print(f"v1 val files: {len(old)}  v2 hh/cs/fr val files: {len(new)}  identical: {same}")
    diff = sorted(set(old) ^ set(new))[:10] + [k for k in old if k in new and new[k] != old[k]][:10]
    if diff:
        print("differences (first 20):", diff)
    else:
        print("OK: v1 val images and labels are byte-identical in v2")


def _zip_split(out: Path, split: str, dst: Path) -> None:
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zf:
        for kind in ("images", "labels"):
            for p in sorted((out / kind / split).iterdir()):
                zf.write(p, f"{kind}/{split}/{p.name}")


def dataset_card(doc: dict, counts: dict, title: str) -> str:
    lines = ["---", "license: cc-by-4.0", "tags:", "  - qai-hf-hackathon-2026", "  - object-detection",
             "  - yolo", "---", "", f"# {title}", "",
             "YOLO-format merge for Salamtak (team QH-091). Split layout: `train.zip` -> `images/train`, "
             "`labels/train`; `val.zip` -> `images/val`, `labels/val`. Image names start with the source "
             "alias (`{alias}_{split}_{i:06d}.jpg`).", "",
             "## Classes", "", ", ".join(f"{i} {c}" for i, c in enumerate(CLASSES)), "",
             "## Sources", "", "| alias | origin | license | labelled classes | train | val |",
             "|---|---|---|---|---|---|"]
    for a, s in doc["sources"].items():
        lines.append(f"| {a} | `{s['origin']}` | {s['license']} | {', '.join(s['labelled_classes'])} "
                     f"| {s['train_images']} | {s['val_images']} |")
    if doc.get("skipped_sources"):
        lines += ["", "Skipped sources:"] + [f"- {a}: {r}" for a, r in doc["skipped_sources"].items()]
    lines += ["", "## De-duplication", "", "```json", json.dumps(doc["dedupe"], indent=2), "```",
              "", "## Box counts (ground truth)", ""]
    for split in ("train", "val"):
        tot: Counter = Counter()
        for v in counts[split].values():
            tot.update(v["boxes"])
        lines.append(f"- **{split}**: " + ", ".join(f"{c} {tot[c]}" for c in CLASSES))
    lines += ["", "Validation labels are ground truth only. The v1 validation images (hh, cs, fr) are "
              "unchanged so v1 and v2 detectors can be compared on them."]
    return "\n".join(lines) + "\n"


def cmd_push(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    out = out_dir()
    doc = json.loads((out / "sources.json").read_text(encoding="utf-8"))
    counts = count_labels(out)
    stage = WORK / "_push_gt"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    for split in ("train", "val"):
        print(f"zipping {split} ...", flush=True)
        _zip_split(out, split, stage / f"{split}.zip")
    shutil.copyfile(out / "data.yaml", stage / "data.yaml")
    shutil.copyfile(out / "sources.json", stage / "sources.json")
    (stage / "README.md").write_text(dataset_card(doc, counts, "Salamtak v2 dataset (ground truth only)"),
                                     encoding="utf-8")
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(stage),
                      commit_message="v2 ground-truth dataset (11 classes)")
    print(f"pushed https://huggingface.co/datasets/{args.repo}")


# --------------------------------------------------------------------------- #

def main() -> None:
    global WORK
    p = argparse.ArgumentParser(description="Salamtak v2 dataset prep")
    p.add_argument("--work-dir", default=str(WORK))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(func=cmd_download)
    sub.add_parser("inspect").set_defaults(func=cmd_inspect)
    sp = sub.add_parser("preview")
    sp.add_argument("--alias", required=True)
    sp.add_argument("--cls", nargs="*", help="only images containing these source classes")
    sp.add_argument("--n", type=int, default=4)
    sp.set_defaults(func=cmd_preview)
    sp = sub.add_parser("convert")
    sp.add_argument("--keep", action="store_true", help="don't wipe yolo_v2_gt first")
    sp.set_defaults(func=cmd_convert)
    sp = sub.add_parser("verify-v1")
    sp.add_argument("--v1-repo", default="plitip/team-qh-091-safesite-data")
    sp.set_defaults(func=cmd_verify_v1)
    sp = sub.add_parser("push")
    sp.add_argument("--repo", default="plitip/team-qh-091-safesite-data-v2-gt")
    sp.set_defaults(func=cmd_push)
    args = p.parse_args()
    WORK = Path(args.work_dir)
    args.func(args)


if __name__ == "__main__":
    main()
