# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ultralytics>=8.4.0",
#     "opencv-python-headless>=4.10",
#     "huggingface_hub>=0.34",
# ]
#
# [tool.uv]
# # Ultralytics pulls in opencv-python, which needs libGL.so.1 (missing on Jobs).
# # Drop it so only the headless build provides cv2.
# override-dependencies = ["opencv-python; sys_platform == 'never'"]
# ///
"""Fine-tune YOLO26 on the SafeSite dataset on Hugging Face Jobs.

Downloads the private YOLO-format dataset, trains from pretrained weights,
then uploads best.pt, results.csv and metrics.json to the model repo.
Partial checkpoints are uploaded during training so a timeout loses little.

Works with the v1 dataset (7 classes, no sources.json) and the v2 dataset
(class names from its data.yaml, per-source validation from its sources.json).
With sources.json present it also validates best.pt per source / pooled view and
compares v1 best.pt on the same images; results go into metrics.json.

Launch:
    hf jobs uv run --flavor t4-small --timeout 30m --secrets HF_TOKEN \
        scripts/train.py --epochs 1 --fraction 0.05 --path-in-repo smoke

Local data-prep check (no GPU, no upload):
    python scripts/train.py --data-dir <yolo dir> --prepare-only --fraction 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

CLASS_NAMES = ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle"]  # v1 contract, ids 0-6
DEFAULT_DATASET_REPO = "plitip/team-qh-091-safesite-data"
DEFAULT_MODEL_REPO = "plitip/team-qh-091-safesite"
V1_WEIGHTS_FILE = "best.pt"  # v1 weights at the model repo root; never overwritten
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUBSET_SEED = 0
MODEL_SIZES = ("n", "s", "m", "l", "x")
CACHE_MODES = ("off", "ram", "disk")
EVAL_BATCH_FALLBACK = 16
DOWNLOAD_WORKERS = 32  # dataset is ~43k small files; default of 8 is slow
DOWNLOAD_PATTERNS = ["*.zip", "*.yaml", "*.json", "README.md"]  # zips + data.yaml + sources.json + card

# Pooled validation views (PLAN-v2 Gate 3). classes=None: union of the sources' labelled v1 classes.
POOLED_VIEWS: dict[str, dict] = {
    "v1_val": {"aliases": ("hh", "cs", "fr"), "classes": None},
    "fully_labelled": {"aliases": ("cs", "fr", "chd", "ppe"),
                       "classes": ("person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle")},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--fraction", type=float, default=1.0, help="Fraction of train AND val images to use (0-1].")
    p.add_argument("--model-size", choices=MODEL_SIZES, default="s", help="YOLO26 size: n/s/m/l/x.")
    p.add_argument("--batch", type=int, default=16, help="Batch size; -1 = Ultralytics AutoBatch.")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument("--patience", type=int, default=20, help="Early-stopping patience in epochs.")
    p.add_argument("--max-hours", type=float, default=None,
                   help="Ultralytics time budget in hours; training ends cleanly before it. Keep below --timeout.")
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="Upload last.pt + results.csv every N epochs (0 = off).")
    p.add_argument("--cache", choices=CACHE_MODES, default="off", help="Ultralytics image cache.")
    p.add_argument("--per-source-val", action=argparse.BooleanOptionalAction, default=None,
                   help="Validate best.pt per source + pooled views (default: on when sources.json exists).")
    p.add_argument("--compare-v1", action=argparse.BooleanOptionalAction, default=None,
                   help="Validate v1 best.pt on the same pooled images (default: on when sources.json exists).")
    p.add_argument("--v1-revision", default=None, help="Model repo revision for the v1 weights (default: main).")
    p.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    p.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    p.add_argument("--path-in-repo", default="", help="Subfolder in the model repo, e.g. 'smoke'. Empty = repo root.")
    p.add_argument("--public", action="store_true", help="Create the model repo public (default: private).")
    p.add_argument("--work-dir", default=str(Path(tempfile.gettempdir()) / "safesite"))
    p.add_argument("--data-dir", default=None, help="Use a local YOLO dataset instead of downloading.")
    p.add_argument("--prepare-only", action="store_true", help="Prepare data.yaml and exit (no training).")
    args = p.parse_args(argv)
    if not 0 < args.fraction <= 1:
        p.error("--fraction must be in (0, 1]")
    if args.epochs < 1 or args.imgsz < 32:
        p.error("--epochs must be >= 1 and --imgsz >= 32")
    return args


def cache_value(mode: str) -> bool | str:
    return False if mode == "off" else mode


def resolve_auto(flag: bool | None, has_sources: bool) -> bool:
    """Auto flags run only when sources.json exists; an explicit --no-... always wins."""
    return bool(has_sources) and flag is not False


# ---------------------------------------------------------------- data

def download_dataset(repo_id: str, dest: Path, token: str | None) -> Path:
    # With zips, fetch only zips + metadata (skips any stale loose image files from earlier uploads).
    has_zips = any(f.endswith(".zip") for f in HfApi(token=token).list_repo_files(repo_id, repo_type="dataset"))
    print(f"Downloading dataset {repo_id} ({'zips + metadata' if has_zips else 'all files'}) ...", flush=True)
    local = Path(snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dest, token=token,
                                   max_workers=DOWNLOAD_WORKERS,
                                   allow_patterns=DOWNLOAD_PATTERNS if has_zips else None))
    for archive in sorted(local.rglob("*.zip")):
        print(f"Extracting {archive.name} ...", flush=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(archive.parent)
        archive.unlink()  # free disk; Jobs storage is limited
    return local


def find_data_root(base: Path) -> Path:
    """Return the folder that contains images/{train,val} and labels/{train,val}."""
    candidates = [base, *(p.parent for p in sorted(base.rglob("images")) if p.is_dir())]
    for root in candidates:
        if all((root / kind / split).is_dir() for kind in ("images", "labels") for split in ("train", "val")):
            return root
    raise FileNotFoundError(f"No images/{{train,val}} + labels/{{train,val}} found under {base}")


def parse_names(names: list | dict) -> list[str]:
    """Ultralytics `names` may be a list or an {id: name} mapping; return names in id order."""
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names]


def load_class_names(search_dirs: list[Path]) -> list[str]:
    """Class names from the dataset's data.yaml (first dir that has one with names), else the v1 list."""
    import yaml

    for folder in search_dirs:
        path = Path(folder) / "data.yaml"
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("names"):
            return parse_names(data["names"])
    return list(CLASS_NAMES)


def check_class_contract(names: list[str]) -> None:
    """ids 0-6 must stay the v1 classes so v1 weights and the app keep working."""
    if names[: len(CLASS_NAMES)] != CLASS_NAMES:
        raise SystemExit(f"Class names break the contract: ids 0-6 must be {CLASS_NAMES}, got {names}")


def find_sources_json(search_dirs: list[Path]) -> Path | None:
    for folder in search_dirs:
        path = Path(folder) / "sources.json"
        if path.is_file():
            return path
    return None


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def take_subset(paths: list[Path], fraction: float, seed: int = SUBSET_SEED) -> list[Path]:
    if fraction >= 1:
        return list(paths)
    k = max(1, round(len(paths) * fraction))
    return sorted(random.Random(seed).sample(paths, k))


def write_list_file(paths: list[Path], out: Path) -> Path:
    """Ultralytics image-list file: one absolute path per line."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{Path(p).resolve()}\n" for p in paths), encoding="utf-8")
    return out


def _yaml_lines(train: str, val: str, names: list[str], path: str | None = None) -> list[str]:
    head = [f"path: {json.dumps(path)}"] if path else []
    return head + [f"train: {json.dumps(train)}", f"val: {json.dumps(val)}", "names:"] + [
        f"  {i}: {json.dumps(name)}" for i, name in enumerate(names)]


def write_eval_yaml(list_file: Path, names: list[str], out: Path) -> Path:
    """Data yaml for validating on one image list (Ultralytics requires a train key too)."""
    target = str(Path(list_file).absolute())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(_yaml_lines(target, target, names)) + "\n", encoding="utf-8")
    return out


def write_data_yaml(root: Path, fraction: float, out_dir: Path,
                    names: list[str] | None = None) -> tuple[Path, dict[str, int], list[Path]]:
    """Write an Ultralytics data.yaml; for fraction < 1, point it at subset list files.

    Returns (data.yaml, image counts, the val images used) so per-source validation
    later runs on exactly the same val subset.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, str] = {}
    counts: dict[str, int] = {}
    val_images: list[Path] = []
    for split in ("train", "val"):
        images = take_subset(list_images(root / "images" / split), fraction)
        counts[split] = len(images)
        if split == "val":
            val_images = images
        if fraction >= 1:
            entries[split] = str((root / "images" / split).resolve())
        else:
            entries[split] = str(write_list_file(images, out_dir / f"{split}.txt").resolve())
    lines = _yaml_lines(entries["train"], entries["val"], names or CLASS_NAMES, path=str(root.resolve()))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml, counts, val_images


# ---------------------------------------------------------------- per-source helpers (pure)

def images_for_alias(paths: list[Path], alias: str) -> list[Path]:
    """Images named `{alias}_...` (prepare_data_v2.py naming)."""
    prefix = f"{alias}_"
    return [p for p in paths if Path(p).name.startswith(prefix)]


def images_for_aliases(paths: list[Path], aliases: list[str] | tuple[str, ...]) -> list[Path]:
    prefixes = tuple(f"{a}_" for a in aliases)
    return [p for p in paths if Path(p).name.startswith(prefixes)]


def labelled_classes(sources: dict, aliases: list[str] | tuple[str, ...], allowed: list[str]) -> list[str]:
    """Ground-truth classes labelled by any of `aliases`, restricted to `allowed`, in `allowed` order."""
    union: set[str] = set()
    for alias in aliases:
        union.update(sources.get("sources", {}).get(alias, {}).get("labelled_classes", []))
    return [name for name in allowed if name in union]


def view_classes(view: dict, sources: dict, names: list[str]) -> list[str]:
    if view["classes"] is None:
        return labelled_classes(sources, view["aliases"], names[: len(CLASS_NAMES)])
    return [name for name in names if name in view["classes"]]


def pick_ap(per_class: dict[str, float], classes: list[str]) -> dict[str, float]:
    return {c: per_class[c] for c in classes if c in per_class}


def filter_label_text(text: str, n_classes: int) -> str:
    """Keep YOLO label lines whose class id < n_classes (drops v2-only classes for the 7-class v1 model)."""
    kept = [line for line in text.splitlines() if line.split() and int(float(line.split()[0])) < n_classes]
    return "".join(f"{line}\n" for line in kept)


def label_path_for(image: Path) -> Path:
    """Same rule as Ultralytics: last 'images' folder -> 'labels', suffix -> .txt."""
    parts = list(Path(image).parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(Path(src).resolve(), dst)
    except OSError:  # e.g. Windows without symlink rights
        shutil.copy2(src, dst)


def build_filtered_view(images: list[Path], out_dir: Path, n_classes: int) -> Path:
    """Mirror `images` into out_dir/images/val with labels filtered to ids < n_classes; return the list file."""
    img_dir, lbl_dir = out_dir / "images" / "val", out_dir / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    linked = []
    for img in images:
        dst = img_dir / Path(img).name
        _link_or_copy(Path(img), dst)
        src_label = label_path_for(Path(img))
        text = src_label.read_text(encoding="utf-8") if src_label.is_file() else ""
        (lbl_dir / f"{dst.stem}.txt").write_text(filter_label_text(text, n_classes), encoding="utf-8")
        linked.append(dst)
    # absolute but NOT resolved: resolving a symlink would point Ultralytics back at the unfiltered labels
    out = out_dir / "list.txt"
    out.write_text("".join(f"{p.absolute()}\n" for p in linked), encoding="utf-8")
    return out


# ---------------------------------------------------------------- hub

def repo_path(prefix: str, name: str) -> str:
    clean = prefix.strip("/")
    return f"{clean}/{name}" if clean else name


def upload_partial(api: HfApi, repo_id: str, prefix: str, save_dir: Path, message: str) -> None:
    """Best effort: never let a checkpoint upload kill the training run."""
    files = {"last.pt": save_dir / "weights" / "last.pt", "results.csv": save_dir / "results.csv"}
    for name, path in files.items():
        if not path.exists():
            continue
        try:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=repo_path(prefix, f"checkpoints/{name}"),
                            repo_id=repo_id, repo_type="model", commit_message=message)
        except Exception as exc:  # noqa: BLE001 - log and keep training
            print(f"WARNING: checkpoint upload of {name} failed: {exc}", file=sys.stderr, flush=True)
    print(f"Checkpoint step done ({message})", flush=True)


def make_checkpoint_callback(api: HfApi, repo_id: str, prefix: str, every: int):
    def on_fit_epoch_end(trainer) -> None:
        epoch = trainer.epoch + 1
        if every <= 0 or epoch % every:
            return
        upload_partial(api, repo_id, prefix, Path(trainer.save_dir), f"checkpoint after epoch {epoch}")
    return on_fit_epoch_end


def upload_results(api: HfApi, repo_id: str, prefix: str, save_dir: Path, metrics: dict) -> str | None:
    staging = save_dir / "upload"
    staging.mkdir(exist_ok=True)
    shutil.copy2(save_dir / "weights" / "best.pt", staging / "best.pt")
    shutil.copy2(save_dir / "results.csv", staging / "results.csv")
    (staging / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    info = api.upload_folder(folder_path=str(staging), path_in_repo=prefix.strip("/") or ".", repo_id=repo_id,
                             repo_type="model",
                             commit_message=f"Train yolo26{metrics['model_size']}: mAP50={metrics['mAP50']:.3f}")
    uploaded = [repo_path(prefix, f.name) for f in sorted(staging.iterdir())]
    print(f"Uploaded to https://huggingface.co/{repo_id}: {uploaded}", flush=True)
    return getattr(info, "oid", None)


def upload_metrics(api: HfApi, repo_id: str, prefix: str, save_dir: Path, metrics: dict) -> str | None:
    path = save_dir / "upload" / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    info = api.upload_file(path_or_fileobj=str(path), path_in_repo=repo_path(prefix, "metrics.json"),
                           repo_id=repo_id, repo_type="model",
                           commit_message="Add per-source validation and v1 comparison to metrics.json")
    return getattr(info, "oid", None)


# ---------------------------------------------------------------- train / eval

def run_training(args: argparse.Namespace, data_yaml: Path, api: HfApi, work_dir: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(f"yolo26{args.model_size}.pt")  # pretrained COCO weights
    model.add_callback("on_fit_epoch_end",
                       make_checkpoint_callback(api, args.model_repo, args.path_in_repo, args.checkpoint_every))
    train_kwargs = dict(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                        workers=args.workers, patience=args.patience, seed=SUBSET_SEED,
                        cache=cache_value(args.cache),
                        project=str(work_dir / "runs"), name="train", exist_ok=True)
    if args.max_hours:
        train_kwargs["time"] = args.max_hours
    try:
        model.train(**train_kwargs)
    except BaseException:
        save_dir = getattr(model.trainer, "save_dir", None)
        if save_dir:
            upload_partial(api, args.model_repo, args.path_in_repo, Path(save_dir), "partial: training failed")
        raise
    return Path(model.trainer.save_dir)


def evaluate(weights: Path, data_yaml: Path, args: argparse.Namespace, work_dir: Path,
             name: str = "final_val") -> dict:
    from ultralytics import YOLO

    batch = args.batch if args.batch > 0 else EVAL_BATCH_FALLBACK
    res = YOLO(str(weights)).val(data=str(data_yaml), imgsz=args.imgsz, batch=batch, split="val", plots=False,
                                 workers=args.workers, project=str(work_dir / "runs"), name=name, exist_ok=True)
    per_class = {res.names[int(c)]: round(float(res.box.ap50[i]), 4) for i, c in enumerate(res.box.ap_class_index)}
    return {"mAP50": round(float(res.box.map50), 4), "mAP50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4), "recall": round(float(res.box.mr), 4),
            "per_class_mAP50": per_class}


def _scored_view(weights: Path, list_file: Path, names: list[str], classes: list[str], n_images: int,
                 args: argparse.Namespace, work_dir: Path, tag: str) -> dict:
    """Validate `weights` on one image list; keep only the ground-truth `classes`."""
    data_yaml = write_eval_yaml(list_file, names, work_dir / "eval" / f"{tag}.yaml")
    scores = evaluate(weights, data_yaml, args, work_dir, name=f"val_{tag}")
    picked = pick_ap(scores["per_class_mAP50"], classes)
    mean = round(sum(picked.values()) / len(picked), 4) if picked else None
    return {"val_images": n_images, "classes": classes, "per_class_mAP50": picked, "mean_mAP50": mean}


def _safe(fn, *a, **kw) -> dict:
    """Extended evaluation must never cost us the trained weights: record errors instead of raising."""
    try:
        return fn(*a, **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: evaluation step failed: {exc}", file=sys.stderr, flush=True)
        return {"error": f"{type(exc).__name__}: {exc}"}


def _skip(imgs: list[Path], classes: list[str]) -> dict | None:
    if imgs and classes:
        return None
    return {"val_images": len(imgs), "skipped": "no val images" if not imgs else "no labelled classes"}


def per_source_eval(best_pt: Path, names: list[str], sources: dict, val_images: list[Path],
                    args: argparse.Namespace, work_dir: Path) -> dict:
    lists = work_dir / "eval" / "lists"
    per_source: dict[str, dict] = {}
    for alias in sources.get("sources", {}):
        imgs = images_for_alias(val_images, alias)
        classes = labelled_classes(sources, [alias], names)
        if (skipped := _skip(imgs, classes)) is not None:
            per_source[alias] = skipped
            continue
        print(f"Per-source val: {alias} ({len(imgs)} images, classes {classes})", flush=True)
        per_source[alias] = _safe(_scored_view, best_pt, write_list_file(imgs, lists / f"{alias}.txt"), names,
                                  classes, len(imgs), args, work_dir, f"src_{alias}")
    pooled: dict[str, dict] = {}
    for view_name, view in POOLED_VIEWS.items():
        imgs = images_for_aliases(val_images, view["aliases"])
        classes = view_classes(view, sources, names)
        if (skipped := _skip(imgs, classes)) is not None:
            pooled[view_name] = skipped
            continue
        print(f"Pooled val: {view_name} ({len(imgs)} images)", flush=True)
        pooled[view_name] = {"aliases": list(view["aliases"]),
                             **_safe(_scored_view, best_pt, write_list_file(imgs, lists / f"{view_name}.txt"),
                                     names, classes, len(imgs), args, work_dir, f"pool_{view_name}")}
    return {"per_source": per_source, "pooled": pooled}


def v1_same_images_eval(sources: dict, val_images: list[Path], args: argparse.Namespace, work_dir: Path,
                        token: str | None) -> dict:
    """Validate v1 best.pt (7 classes) on the pooled views, labels filtered to ids 0-6 (v2-only boxes dropped)."""
    v1_pt = Path(hf_hub_download(repo_id=args.model_repo, filename=V1_WEIGHTS_FILE, revision=args.v1_revision,
                                 token=token, local_dir=work_dir / "v1"))
    out: dict[str, dict | str] = {"weights": f"{args.model_repo}/{V1_WEIGHTS_FILE}@{args.v1_revision or 'main'}"}
    for view_name, view in POOLED_VIEWS.items():
        imgs = images_for_aliases(val_images, view["aliases"])
        classes = view_classes(view, sources, CLASS_NAMES)
        if (skipped := _skip(imgs, classes)) is not None:
            out[view_name] = skipped
            continue
        print(f"v1 comparison: {view_name} ({len(imgs)} images)", flush=True)
        list_file = build_filtered_view(imgs, work_dir / "eval" / "v1view" / view_name, len(CLASS_NAMES))
        out[view_name] = _safe(_scored_view, v1_pt, list_file, CLASS_NAMES, classes, len(imgs), args, work_dir,
                               f"v1_{view_name}")
    return out


def build_metrics(args: argparse.Namespace, scores: dict, counts: dict, save_dir: Path, started: float,
                  names: list[str]) -> dict:
    import torch
    import ultralytics

    rows = (save_dir / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"model": f"yolo26{args.model_size}", "model_size": args.model_size, **scores,
            "epochs_requested": args.epochs, "epochs_completed": max(len(rows) - 1, 0),
            "imgsz": args.imgsz, "batch": args.batch, "fraction": args.fraction, "cache": args.cache,
            "train_images": counts["train"], "val_images": counts["val"], "classes": names,
            "dataset": args.dataset_repo, "gpu": gpu,
            "runtime_minutes": round((time.time() - started) / 60, 1), "ultralytics": ultralytics.__version__}


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    started = time.time()
    args = parse_args(argv)
    work_dir = Path(args.work_dir)
    token = os.environ.get("HF_TOKEN")  # never print it

    base = Path(args.data_dir) if args.data_dir else download_dataset(args.dataset_repo, work_dir / "data", token)
    root = find_data_root(base)
    names = load_class_names([base, root])
    check_class_contract(names)
    print(f"Classes ({len(names)}): {names}", flush=True)
    sources_path = find_sources_json([base, root])
    sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path else None
    do_per_source = resolve_auto(args.per_source_val, sources is not None)
    do_compare = resolve_auto(args.compare_v1, sources is not None)
    print(f"sources.json: {sources_path or 'not found'} -> per-source val {'on' if do_per_source else 'skipped'}, "
          f"v1 comparison {'on' if do_compare else 'skipped'}", flush=True)

    data_yaml, counts, val_images = write_data_yaml(root, args.fraction, work_dir, names)
    print(f"Data ready: {counts['train']} train / {counts['val']} val images -> {data_yaml}", flush=True)
    if args.prepare_only:
        return 0
    if not token:
        print("ERROR: HF_TOKEN is not set; launch with --secrets HF_TOKEN", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    api.create_repo(args.model_repo, repo_type="model", private=not args.public, exist_ok=True)  # fail fast
    save_dir = run_training(args, data_yaml, api, work_dir)
    best_pt = save_dir / "weights" / "best.pt"
    scores = evaluate(best_pt, data_yaml, args, work_dir)
    metrics = build_metrics(args, scores, counts, save_dir, started, names)
    commit = upload_results(api, args.model_repo, args.path_in_repo, save_dir, metrics)  # weights safe first

    if do_per_source or do_compare:
        if do_per_source:
            metrics.update(per_source_eval(best_pt, names, sources, val_images, args, work_dir))
        if do_compare:
            metrics["v1_same_images"] = _safe(v1_same_images_eval, sources, val_images, args, work_dir, token)
        metrics["runtime_minutes"] = round((time.time() - started) / 60, 1)
        commit = upload_metrics(api, args.model_repo, args.path_in_repo, save_dir, metrics) or commit
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"UPLOAD_COMMIT={commit}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
