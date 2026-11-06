# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ultralytics>=8.4.0",
#     "opencv-python-headless>=4.10",
#     "huggingface_hub>=0.34",
#     "pyyaml",
# ]
#
# [tool.uv]
# # Ultralytics pulls in opencv-python, which needs libGL.so.1 (missing on Jobs).
# override-dependencies = ["opencv-python; sys_platform == 'never'"]
# ///
"""Fill missing labels on v2 TRAIN images with a teacher detector (HF Jobs).

Downloads plitip/team-qh-091-safesite-data-v2-gt, runs the teacher on train images, adds
teacher boxes only for classes that source does not label, re-zips and uploads the result
to plitip/team-qh-091-safesite-data-v2 (private). Val labels are never touched: val.zip is
re-uploaded byte-for-byte.

    hf jobs uv run --flavor t4-small --timeout 20m --secrets HF_TOKEN scripts/pseudo_label.py --limit 200
    hf jobs uv run --flavor l4x1 --timeout 1h --secrets HF_TOKEN scripts/pseudo_label.py

Rules: teacher conf >= --conf; only classes missing from that source's ground truth; a teacher
helmet/no_helmet (vest/no_vest) box that overlaps (IoU >= 0.3) a ground-truth box of the
opposite class is skipped, and between two opposite teacher boxes only the more confident one
is kept; ground-truth boxes are never deleted or moved.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle",
           "harness", "scaffold", "crane_hook", "utility_pole"]
CLS_IDX = {n: i for i, n in enumerate(CLASSES)}
OPPOSITE = {CLS_IDX["helmet"]: CLS_IDX["no_helmet"], CLS_IDX["no_helmet"]: CLS_IDX["helmet"],
            CLS_IDX["vest"]: CLS_IDX["no_vest"], CLS_IDX["no_vest"]: CLS_IDX["vest"]}
CONFLICT_IOU = 0.3

YIHONG_REPO = "yihong1120/Construction-Hazard-Detection"
YIHONG_LICENSE = "AGPL-3.0"
# Teacher class name (lower, stripped) -> v2 class; same as the chd mapping in prepare_data_v2.py.
YIHONG_MAP: dict[str, str | None] = {
    "person": "person", "hardhat": "helmet", "no-hardhat": "no_helmet", "safety vest": "vest",
    "no-safety vest": "no_vest", "machinery": "machinery", "vehicle": "vehicle",
    "utility pole": "utility_pole", "mask": None, "no-mask": None, "safety cone": None,
}
YIHONG_REQUIRED = ("person", "hardhat", "no-hardhat", "safety vest", "no-safety vest",
                   "machinery", "vehicle", "utility pole")
FALLBACK_V1_REPO = "plitip/team-qh-091-safesite"
FALLBACK_V1_FILE = "best.pt"
FALLBACK_V1_CLASSES = ("helmet", "no_helmet", "machinery", "vehicle")
FALLBACK_COCO = "yolo26x.pt"
BATCH = 32
IMGSZ = 640

Box = tuple[int, float, float, float, float]  # cls, cx, cy, w, h (normalized)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #

def alias_of(name: str) -> str:
    return Path(name).name.split("_", 1)[0]


def parse_lines(lines: list[str]) -> list[Box]:
    out = []
    for ln in lines:
        p = ln.split()
        if len(p) >= 5:
            out.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def fmt(box: Box) -> str:
    c, x, y, w, h = box
    return f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx0, by0, bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def missing_classes(labelled: list[str], producible: set[int]) -> set[int]:
    """Classes the teacher may add for a source: not labelled there and producible by the teacher."""
    return {i for i, c in enumerate(CLASSES) if c not in labelled} & producible


def merge_pseudo(gt_lines: list[str], teacher: list[tuple[Box, float]], missing: set[int],
                 conf: float) -> tuple[list[str], Counter]:
    """GT lines (unchanged, first) + accepted teacher boxes. Returns (lines, added per class id)."""
    gt = parse_lines(gt_lines)
    cand = sorted(((b, s) for b, s in teacher if s >= conf and b[0] in missing),
                  key=lambda bs: -bs[1])
    kept: list[Box] = []
    for box, _ in cand:
        opp = OPPOSITE.get(box[0])
        if opp is not None and any(g[0] == opp and iou(box, g) >= CONFLICT_IOU for g in gt):
            continue
        if opp is not None and any(k[0] == opp and iou(box, k) >= CONFLICT_IOU for k in kept):
            continue
        kept.append(box)
    added = Counter(b[0] for b in kept)
    return [ln for ln in gt_lines if ln.split()] + [fmt(b) for b in kept], added


def update_labels(split: str, gt_lines: list[str], teacher: list[tuple[Box, float]],
                  missing: set[int], conf: float) -> tuple[list[str], Counter]:
    """Val labels are returned untouched; train labels get merge_pseudo."""
    if split != "train":
        return list(gt_lines), Counter()
    return merge_pseudo(gt_lines, teacher, missing, conf)


def pick_teacher_file(files: dict[str, int]) -> str:
    """Largest YOLO .pt under a models/.../pt/ folder."""
    pts = {f: s for f, s in files.items() if f.endswith(".pt") and f.startswith("models/") and "/pt/" in f}
    if not pts:
        raise SystemExit("no models/**/pt/*.pt in teacher repo")
    return max(pts, key=lambda f: (pts[f], f))


def teacher_class_map(names: dict[int, str]) -> dict[int, int]:
    """yihong names -> v2 ids (dropped classes omitted). SystemExit if a required name is absent."""
    low = {i: str(n).strip().lower() for i, n in names.items()}
    missing = [n for n in YIHONG_REQUIRED if n not in low.values()]
    if missing:
        raise SystemExit(f"teacher names lack {missing}: {names}")
    out = {}
    for i, n in low.items():
        if n not in YIHONG_MAP:
            raise SystemExit(f"teacher class {n!r} not in YIHONG_MAP")
        if YIHONG_MAP[n] is not None:
            out[i] = CLS_IDX[YIHONG_MAP[n]]
    return out


# --------------------------------------------------------------------------- #
# Job
# --------------------------------------------------------------------------- #

def load_teachers(kind: str, token: str | None) -> tuple[list[tuple[object, dict[int, int]]], dict]:
    """[(YOLO model, teacher id -> v2 id)], metadata for sources.json."""
    from huggingface_hub import HfApi, hf_hub_download
    from ultralytics import YOLO

    if kind == "yihong":
        files = {f.path: f.size or 0 for f in HfApi(token=token).list_repo_tree(YIHONG_REPO, recursive=True)
                 if hasattr(f, "size")}
        fname = pick_teacher_file(files)
        model = YOLO(hf_hub_download(YIHONG_REPO, fname, token=token))
        cmap = teacher_class_map(dict(model.names))
        meta = {"name": "yihong", "repo": YIHONG_REPO, "file": fname, "license": YIHONG_LICENSE,
                "teacher_names": dict(model.names)}
        return [(model, cmap)], meta
    v1 = YOLO(hf_hub_download(FALLBACK_V1_REPO, FALLBACK_V1_FILE, token=token))
    v1_map = {i: CLS_IDX[n] for i, n in v1.names.items() if n in FALLBACK_V1_CLASSES}
    coco = YOLO(FALLBACK_COCO)
    coco_map = {i: CLS_IDX["person"] for i, n in coco.names.items() if n == "person"}
    meta = {"name": "fallback", "repo": f"{FALLBACK_V1_REPO} ({FALLBACK_V1_FILE}) + ultralytics {FALLBACK_COCO}",
            "file": f"{FALLBACK_V1_FILE}, {FALLBACK_COCO}", "license": "AGPL-3.0"}
    return [(v1, v1_map), (coco, coco_map)], meta


def predict(teachers, paths: list[Path], conf: float) -> dict[str, list[tuple[Box, float]]]:
    out: dict[str, list[tuple[Box, float]]] = defaultdict(list)
    for t, (model, cmap) in enumerate(teachers):
        for start in range(0, len(paths), BATCH):
            chunk = [str(p) for p in paths[start:start + BATCH]]
            for p, r in zip(chunk, model.predict(chunk, conf=conf, imgsz=IMGSZ, half=True, verbose=False)):
                xywhn = r.boxes.xywhn.tolist()
                for c, s, (x, y, w, h) in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), xywhn):
                    if int(c) in cmap:
                        out[Path(p).stem].append(((cmap[int(c)], x, y, w, h), float(s)))
            if start % (BATCH * 50) == 0:
                print(f"  teacher {t}: {start}/{len(paths)}", flush=True)
    return out


def zip_split(root: Path, split: str, stems: list[str], dst: Path) -> None:
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED) as zf:
        for s in stems:
            zf.write(root / "images" / split / f"{s}.jpg", f"images/{split}/{s}.jpg")
            zf.write(root / "labels" / split / f"{s}.txt", f"labels/{split}/{s}.txt")


def box_counts(label_dir: Path, stems: list[str]) -> Counter:
    c: Counter = Counter()
    for s in stems:
        for b in parse_lines((label_dir / f"{s}.txt").read_text(encoding="utf-8").splitlines()):
            c[CLASSES[b[0]]] += 1
    return c


def _link(origin: str) -> str:
    if origin.startswith("hf:"):
        return f"[{origin[3:]}](https://huggingface.co/datasets/{origin[3:]})"
    if origin.startswith("roboflow:"):
        return f"[{origin[9:]}](https://universe.roboflow.com/{origin[9:]})"
    return "[Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/)"


def card(doc: dict, gt: Counter, pseudo: Counter, val: Counter, limit: int) -> str:
    t = doc["teacher"]
    lines = ["---", "license: cc-by-4.0", "tags:", "  - qai-hf-hackathon-2026", "  - object-detection",
             "  - yolo", "---", "", "# Salamtak v2 detection dataset (team QH-091)", "",
             "YOLO-format, 11 classes. `train.zip` -> `images/train`, `labels/train`; `val.zip` -> "
             "`images/val`, `labels/val`. Image names start with the source alias.", "",
             "## Classes", "", ", ".join(f"{i} {c}" for i, c in enumerate(CLASSES)), "",
             "## Sources", "", "| alias | origin | license | ground-truth classes | pseudo-labelled classes "
             "| train | val |", "|---|---|---|---|---|---|---|"]
    for a, s in doc["sources"].items():
        lines.append(f"| {a} | {_link(s['origin'])} | {s['license']} | {', '.join(s['labelled_classes'])} | "
                     f"{', '.join(s['pseudo_labelled_classes']) or '-'} | {s['train_images']} | {s['val_images']} |")
    lines += ["", "## Teacher (pseudo-labels)", "",
              f"- Model: `{t['repo']}`, file `{t['file']}` ({t['license']}). Used only to label training "
              "images; it is not part of the product.",
              f"- Rule: on **train images only**, add teacher boxes with confidence >= {t['conf']} for classes "
              "that source does not label. A teacher helmet/no_helmet or vest/no_vest box overlapping "
              f"(IoU >= {t['conflict_iou']}) a ground-truth box of the opposite class is skipped; of two opposite "
              "teacher boxes only the more confident is kept. Ground-truth boxes are never deleted or moved. "
              "Validation labels are ground truth only.", "",
              "## De-duplication", "", "```json", json.dumps(doc.get("dedupe", {}), indent=2), "```", "",
              "## Box counts", "", "| class | train GT | train pseudo | val GT |", "|---|---|---|---|"]
    lines += [f"| {c} | {gt[c]} | {pseudo[c]} | {val[c]} |" for c in CLASSES]
    if limit:
        lines += ["", f"**Smoke subset**: {limit} train images only."]
    return "\n".join(lines) + "\n"


def label_train(ds: Path, stems: list[str], preds: dict, missing: dict, conf: float):
    gt_counts, pseudo_counts = Counter(), Counter()
    per_source: dict[str, Counter] = defaultdict(Counter)
    for s in stems:
        lab = ds / "labels" / "train" / f"{s}.txt"
        gt_lines = lab.read_text(encoding="utf-8").splitlines()
        gt_counts.update(CLASSES[b[0]] for b in parse_lines(gt_lines))
        lines, added = update_labels("train", gt_lines, preds.get(s, []), missing.get(alias_of(s), set()), conf)
        if added:
            lab.write_text("\n".join(lines) + "\n", encoding="utf-8")
            for cid, n in added.items():
                per_source[alias_of(s)][CLASSES[cid]] += n
                pseudo_counts[CLASSES[cid]] += n
    return gt_counts, pseudo_counts, per_source


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-repo", default="plitip/team-qh-091-safesite-data-v2-gt")
    p.add_argument("--target-repo", default="plitip/team-qh-091-safesite-data-v2")
    p.add_argument("--teacher", choices=("yihong", "fallback"), default="yihong")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0, help="smoke test: N train images, uploads to <target>-smoke")
    p.add_argument("--work", default="/tmp/pl")
    args = p.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    target = args.target_repo + ("-smoke" if args.limit else "")
    work = Path(args.work)
    src = Path(snapshot_download(args.source_repo, repo_type="dataset", local_dir=work / "src", token=token))
    ds = work / "ds"
    for z in ("train.zip", "val.zip"):
        with zipfile.ZipFile(src / z) as zf:
            zf.extractall(ds)
    doc = json.loads((src / "sources.json").read_text(encoding="utf-8"))

    teachers, meta = load_teachers(args.teacher, token)
    producible = set().union(*(set(m.values()) for _, m in teachers))
    missing = {a: missing_classes(s["labelled_classes"], producible) for a, s in doc["sources"].items()}
    print("teacher:", meta["repo"], meta["file"], "| fills:",
          {a: sorted(CLASSES[i] for i in m) for a, m in missing.items()}, flush=True)

    stems = sorted(p.stem for p in (ds / "images" / "train").glob("*.jpg"))
    if args.limit:
        stems = sorted(random.Random(0).sample(stems, min(args.limit, len(stems))))
    todo = [ds / "images" / "train" / f"{s}.jpg" for s in stems if missing.get(alias_of(s))]
    print(f"train images: {len(stems)}  to label: {len(todo)}", flush=True)
    preds = predict(teachers, todo, args.conf)
    gt_counts, pseudo_counts, per_source = label_train(ds, stems, preds, missing, args.conf)

    for a, s in doc["sources"].items():
        s["pseudo_labelled_classes"] = sorted(per_source[a], key=CLASSES.index)
        s["pseudo_boxes"] = dict(per_source[a])
        if args.limit:
            s["train_images"] = sum(1 for st in stems if alias_of(st) == a)
    doc["teacher"] = {**meta, "conf": args.conf, "conflict_iou": CONFLICT_IOU, "imgsz": IMGSZ,
                      "used_for": "labelling training images only"}

    out = work / "out"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    zip_split(ds, "train", stems, out / "train.zip")
    shutil.copyfile(src / "val.zip", out / "val.zip")  # byte-identical: val is never modified
    shutil.copyfile(src / "data.yaml", out / "data.yaml")
    val_counts = box_counts(ds / "labels" / "val", sorted(p.stem for p in (ds / "labels" / "val").glob("*.txt")))
    doc["box_counts"] = {"train_gt": dict(gt_counts), "train_pseudo": dict(pseudo_counts),
                         "val_gt": dict(val_counts)}
    (out / "sources.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    (out / "README.md").write_text(card(doc, gt_counts, pseudo_counts, val_counts, args.limit), encoding="utf-8")
    print("GT:", dict(gt_counts), "\nPSEUDO:", dict(pseudo_counts), flush=True)

    api.create_repo(target, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=target, repo_type="dataset", folder_path=str(out),
                      commit_message=f"v2 dataset with pseudo-labels ({meta['name']} teacher, conf {args.conf})")
    print(f"uploaded https://huggingface.co/datasets/{target}", flush=True)


if __name__ == "__main__":
    main()
