"""Detectors: image -> list[Detection] with normalized (0-1) xyxy boxes.

Every detector is a callable  detector(image: PIL.Image, mode: "site" | "home") -> list[Detection]
with attributes  name, is_mock, supports_home.  app.py picks one with a single line:

    DETECTOR = detect.MockDetector()     # fixed fake detections, no model
    DETECTOR = detect.YoloDetector()     # fine-tuned YOLO26 from the Hub

Before the demo, pin the weights: set the Space variable SAFESITE_MODEL_REVISION to the model
repo's commit SHA (rollback = set it to the previous SHA). Only load weights from our own repo:
a .pt file is a pickle and runs code when loaded.

YOLOE-26 (open-vocabulary prompts: harness, scaffold, ladder, ...) lives in safesite/openvocab.py;
CombinedDetector concatenates it with the trained detector in Site mode and uses it alone in Home mode.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from PIL import Image

from safesite.types import UNIFIED_CLASSES, Detection, Mode

MODEL_REPO = os.environ.get("SAFESITE_MODEL_REPO", "plitip/team-qh-091-safesite")
WEIGHTS_FILE = os.environ.get("SAFESITE_WEIGHTS_FILE", "best.pt")
MODEL_REVISION = os.environ.get("SAFESITE_MODEL_REVISION")  # commit SHA; None = latest (dev only)
PREDICT_CONF = 0.25  # model-side floor; the shared MIN_CONFIDENCE in types.py decides what counts
IMG_SIZE = 640
MAX_DETECTIONS = 100
log = logging.getLogger(__name__)

# ---------- mock ----------
# Fixed layouts. scripts/make_placeholders.py draws the example images to match these boxes.
MOCK_SITE: tuple[Detection, ...] = (
    Detection("person", 0.91, (0.10, 0.35, 0.22, 0.88), "mock"),
    Detection("no_helmet", 0.84, (0.125, 0.35, 0.195, 0.45), "mock"),
    Detection("vest", 0.80, (0.11, 0.47, 0.21, 0.63), "mock"),
    Detection("person", 0.89, (0.50, 0.40, 0.60, 0.88), "mock"),
    Detection("helmet", 0.87, (0.52, 0.39, 0.58, 0.47), "mock"),
    Detection("no_vest", 0.72, (0.505, 0.49, 0.595, 0.65), "mock"),
    Detection("vehicle", 0.93, (0.66, 0.42, 0.97, 0.86), "mock"),
)
MOCK_HOME: tuple[Detection, ...] = (
    Detection("extension cord", 0.78, (0.15, 0.82, 0.75, 0.90), "mock"),
    Detection("knife", 0.70, (0.62, 0.58, 0.72, 0.62), "mock"),
)


class MockDetector:
    """Returns the same fake detections for any image. Lets the UI run before the model exists."""

    name = "MOCK detector (fixed fake detections)"
    is_mock = True
    supports_home = True

    def __call__(self, image: Image.Image, mode: Mode = "site") -> list[Detection]:
        return list(MOCK_HOME if mode == "home" else MOCK_SITE)


# ---------- real YOLO26 ----------
class LabelMismatchError(RuntimeError):
    """The weights' class names don't match UNIFIED_CLASSES, so every rule would silently miss."""


def check_labels(names: set[str]) -> None:
    missing, extra = set(UNIFIED_CLASSES) - names, names - set(UNIFIED_CLASSES)
    if missing or extra:
        raise LabelMismatchError(
            f"model classes don't match the rules: missing {sorted(missing)}, unexpected {sorted(extra)[:10]}"
        )


class YoloDetector:
    """Fine-tuned YOLO26, downloaded from the Hub and warmed up at construction (Space boot)."""

    is_mock = False
    supports_home = False  # trained classes have no cords/knives; Home mode needs YOLOE (Chat 5)

    def __init__(
        self,
        repo_id: str = MODEL_REPO,
        filename: str = WEIGHTS_FILE,
        revision: str | None = MODEL_REVISION,
        weights_path: str | None = None,  # local .pt override, for tests
        strict_labels: bool = True,  # False only to smoke-test stock COCO weights
        threads: int | None = None,  # torch CPU threads; default = all cores
    ) -> None:
        import numpy as np
        import torch
        from ultralytics import YOLO  # deferred: heavy import, only needed for the real model

        torch.set_num_threads(threads or int(os.environ.get("SAFESITE_THREADS", os.cpu_count() or 2)))
        if weights_path is None:
            from huggingface_hub import hf_hub_download

            weights_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
            source = f"{repo_id}@{(revision or 'latest')[:7]}"
        else:
            source = Path(weights_path).name
        self._model = YOLO(weights_path)
        names = set(self._model.names.values())
        if strict_labels:
            check_labels(names)
        self._known = names
        self.name = f"YOLO26 ({source})"
        self._predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))  # warm-up: first call is slow
        log.info("loaded %s: %d classes, imgsz=%d, threads=%d", self.name, len(names), IMG_SIZE,
                 torch.get_num_threads())

    def _predict(self, source):
        return self._model.predict(source, conf=PREDICT_CONF, imgsz=IMG_SIZE, max_det=MAX_DETECTIONS,
                                   device="cpu", verbose=False)[0]

    def __call__(self, image: Image.Image, mode: Mode = "site") -> list[Detection]:
        result = self._predict(image if image.mode == "RGB" else image.convert("RGB"))
        names, boxes = result.names, result.boxes
        return [
            Detection(
                label=names[int(cls)],
                confidence=float(conf),
                box=(round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)),
                source="detector",
            )
            for (x1, y1, x2, y2), conf, cls in zip(
                boxes.xyxyn.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            )
        ]


# ---------- trained detector + YOLOE open-vocab finder ----------
class CombinedDetector:
    """Site: the trained detector's boxes followed by the open-vocab finder's. Home: the finder only,
    because the trained classes (person, helmet, vest, ...) carry no home hazards."""

    supports_home = True

    def __init__(self, primary: Any, open_vocab: Any) -> None:
        self._primary = primary
        self._open_vocab = open_vocab
        self.name = f"{primary.name} + {open_vocab.name}"
        # getattr: works whether or not the finder declares the contract attributes (chat-5 adds them)
        self.is_mock = bool(getattr(primary, "is_mock", False) or getattr(open_vocab, "is_mock", False))

    def __call__(self, image: Image.Image, mode: Mode = "site") -> list[Detection]:
        found = self._open_vocab(image, mode)
        if str(getattr(mode, "value", mode)).lower() == "home":
            return list(found)
        return [*self._primary(image, mode), *found]
