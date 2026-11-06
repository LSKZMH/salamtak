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

Launch (see CLAUDE.md > Commands):
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

from huggingface_hub import HfApi, snapshot_download

CLASS_NAMES = ["person", "helmet", "no_helmet", "vest", "no_vest", "machinery", "vehicle"]
DEFAULT_DATASET_REPO = "plitip/team-alloy-safesite-data"
DEFAULT_MODEL_REPO = "plitip/team-alloy-safesite"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUBSET_SEED = 0
MODEL_SIZES = ("n", "s", "m", "l", "x")
EVAL_BATCH_FALLBACK = 16
DOWNLOAD_WORKERS = 32  # dataset is ~43k small files; default of 8 is slow


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


# ---------------------------------------------------------------- data

def download_dataset(repo_id: str, dest: Path, token: str | None) -> Path:
    # If the repo has zip archives, fetch only those (skips any stale loose files from earlier uploads).
    has_zips = any(f.endswith(".zip") for f in HfApi(token=token).list_repo_files(repo_id, repo_type="dataset"))
    print(f"Downloading dataset {repo_id} ({'zip archives' if has_zips else 'all files'}) ...", flush=True)
    local = Path(snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dest, token=token,
                                   max_workers=DOWNLOAD_WORKERS, allow_patterns=["*.zip"] if has_zips else None))
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


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def take_subset(paths: list[Path], fraction: float, seed: int = SUBSET_SEED) -> list[Path]:
    if fraction >= 1:
        return list(paths)
    k = max(1, round(len(paths) * fraction))
    return sorted(random.Random(seed).sample(paths, k))


def write_data_yaml(root: Path, fraction: float, out_dir: Path) -> tuple[Path, dict[str, int]]:
    """Write an Ultralytics data.yaml; for fraction < 1, point it at subset list files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        images = take_subset(list_images(root / "images" / split), fraction)
        counts[split] = len(images)
        if fraction >= 1:
            entries[split] = str((root / "images" / split).resolve())
        else:
            list_file = out_dir / f"{split}.txt"
            list_file.write_text("\n".join(str(p.resolve()) for p in images) + "\n", encoding="utf-8")
            entries[split] = str(list_file.resolve())
    lines = [f"path: {json.dumps(str(root.resolve()))}",
             f"train: {json.dumps(entries['train'])}",
             f"val: {json.dumps(entries['val'])}",
             "names:"] + [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml, counts


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


def upload_results(api: HfApi, repo_id: str, prefix: str, save_dir: Path, metrics: dict) -> None:
    staging = save_dir / "upload"
    staging.mkdir(exist_ok=True)
    shutil.copy2(save_dir / "weights" / "best.pt", staging / "best.pt")
    shutil.copy2(save_dir / "results.csv", staging / "results.csv")
    (staging / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    api.upload_folder(folder_path=str(staging), path_in_repo=prefix.strip("/") or ".", repo_id=repo_id,
                      repo_type="model",
                      commit_message=f"Train yolo26{metrics['model_size']}: mAP50={metrics['mAP50']:.3f}")
    uploaded = [repo_path(prefix, f.name) for f in sorted(staging.iterdir())]
    print(f"Uploaded to https://huggingface.co/{repo_id}: {uploaded}", flush=True)


# ---------------------------------------------------------------- train / eval

def run_training(args: argparse.Namespace, data_yaml: Path, api: HfApi, work_dir: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(f"yolo26{args.model_size}.pt")  # pretrained COCO weights
    model.add_callback("on_fit_epoch_end",
                       make_checkpoint_callback(api, args.model_repo, args.path_in_repo, args.checkpoint_every))
    train_kwargs = dict(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                        workers=args.workers, patience=args.patience, seed=SUBSET_SEED,
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


def evaluate(best_pt: Path, data_yaml: Path, args: argparse.Namespace, work_dir: Path) -> dict:
    from ultralytics import YOLO

    batch = args.batch if args.batch > 0 else EVAL_BATCH_FALLBACK
    res = YOLO(str(best_pt)).val(data=str(data_yaml), imgsz=args.imgsz, batch=batch, split="val", plots=False,
                                 project=str(work_dir / "runs"), name="final_val", exist_ok=True)
    per_class = {res.names[int(c)]: round(float(res.box.ap50[i]), 4) for i, c in enumerate(res.box.ap_class_index)}
    return {"mAP50": round(float(res.box.map50), 4), "mAP50_95": round(float(res.box.map), 4),
            "precision": round(float(res.box.mp), 4), "recall": round(float(res.box.mr), 4),
            "per_class_mAP50": per_class}


def build_metrics(args: argparse.Namespace, scores: dict, counts: dict, save_dir: Path, started: float) -> dict:
    import torch
    import ultralytics

    rows = (save_dir / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"model": f"yolo26{args.model_size}", "model_size": args.model_size, **scores,
            "epochs_requested": args.epochs, "epochs_completed": max(len(rows) - 1, 0),
            "imgsz": args.imgsz, "batch": args.batch, "fraction": args.fraction,
            "train_images": counts["train"], "val_images": counts["val"], "classes": CLASS_NAMES,
            "dataset": args.dataset_repo, "gpu": gpu,
            "runtime_minutes": round((time.time() - started) / 60, 1), "ultralytics": ultralytics.__version__}


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    started = time.time()
    args = parse_args(argv)
    work_dir = Path(args.work_dir)
    token = os.environ.get("HF_TOKEN")  # never print it

    base = Path(args.data_dir) if args.data_dir else download_dataset(args.dataset_repo, work_dir / "data", token)
    data_yaml, counts = write_data_yaml(find_data_root(base), args.fraction, work_dir)
    print(f"Data ready: {counts['train']} train / {counts['val']} val images -> {data_yaml}", flush=True)
    if args.prepare_only:
        return 0
    if not token:
        print("ERROR: HF_TOKEN is not set; launch with --secrets HF_TOKEN", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    api.create_repo(args.model_repo, repo_type="model", private=not args.public, exist_ok=True)  # fail fast
    save_dir = run_training(args, data_yaml, api, work_dir)
    scores = evaluate(save_dir / "weights" / "best.pt", data_yaml, args, work_dir)
    metrics = build_metrics(args, scores, counts, save_dir, started)
    upload_results(api, args.model_repo, args.path_in_repo, save_dir, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
