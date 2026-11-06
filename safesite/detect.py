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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from safesite.types import MODEL_CLASSES_V2, UNIFIED_CLASSES, Detection, Mode

def _env(name: str, default: str | None = None) -> str | None:
    """A Space variable can exist but be blank; treat that as unset (a "" revision would 404)."""
    return (os.environ.get(name) or "").strip() or default


MODEL_REPO = _env("SAFESITE_MODEL_REPO", "plitip/team-qh-091-safesite")
WEIGHTS_FILE = _env("SAFESITE_WEIGHTS_FILE", "best.pt")
MODEL_REVISION = _env("SAFESITE_MODEL_REVISION")  # commit SHA; None = latest (dev only)
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
    """The weights' classes aren't the v1 or v2 contract, so rules would silently miss."""


ACCEPTED_CLASS_LISTS: tuple[tuple[str, ...], ...] = (UNIFIED_CLASSES, MODEL_CLASSES_V2)  # v1, v2


def check_labels(names: Mapping[int, str] | Sequence[str]) -> tuple[str, ...]:
    """Validate a model's classes in id order (``model.names`` dict or a list).

    Accepts exactly UNIFIED_CLASSES (v1, 7 classes) or MODEL_CLASSES_V2 (v2, 11 classes);
    returns them as a tuple. Wrong order, unknown or missing classes raise LabelMismatchError.
    """
    ordered = tuple(names[i] for i in sorted(names)) if isinstance(names, Mapping) else tuple(names)
    if ordered in ACCEPTED_CLASS_LISTS:
        return ordered
    raise LabelMismatchError(
        f"model classes {list(ordered)[:12]} are neither v1 {list(UNIFIED_CLASSES)} nor v2 "
        f"{list(MODEL_CLASSES_V2)}; unexpected {sorted(set(ordered) - set(MODEL_CLASSES_V2))[:10]}"
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

        torch.set_num_threads(threads or int(_env("SAFESITE_THREADS") or os.cpu_count() or 2))
        if weights_path is None:
            from huggingface_hub import hf_hub_download

            weights_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
            source = f"{repo_id}@{(revision or 'latest')[:7]}"
        else:
            source = Path(weights_path).name
        self._model = YOLO(weights_path)
        names = set(self._model.names.values())
        if strict_labels:
            check_labels(self._model.names)  # v1 (7) or v2 (11) classes, in id order
        self._known = names
        self.labels = frozenset(names)  # what this model can produce (pipeline's "not checked" note)
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
        # Labels either part can produce; only declared when both parts declare theirs.
        parts = (getattr(primary, "labels", None), getattr(open_vocab, "labels", None))
        if all(p is not None for p in parts):
            self.labels = frozenset().union(*parts)

    def __call__(self, image: Image.Image, mode: Mode = "site") -> list[Detection]:
        found = self._open_vocab(image, mode)
        if str(getattr(mode, "value", mode)).lower() == "home":
            return list(found)
        return [*self._primary(image, mode), *found]
