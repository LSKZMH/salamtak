"""Open-vocabulary finder: YOLOE-26 with text prompts, no training.

Finds the objects the trained YOLO26 detector has no class for (harness, scaffold,
shade tent, cords, ...) and returns them as ``Detection`` objects whose labels match
the constants in ``safesite/rules.py`` (``harness``, ``scaffold``, ``cord``, ...).

Same callable shape as the detectors in ``safesite/detect.py``::

    finder = OpenVocabDetector()            # downloads yoloe-26n-seg.pt on first use
    detections = finder(pil_image, "site")  # or "home"

Weights: ``yoloe-26n-seg.pt`` is the smallest YOLOE-26 (3.9 M params, 6.1 B FLOPs) and
needs ``ultralytics>=8.4.0``. Only ``-seg`` checkpoints are released for text prompts;
we ignore the masks and keep the boxes. The first text encoding downloads the
MobileCLIP2 text encoder (``mobileclip2_b.ts``, ~254 MB) into the working directory.
We encode both prompt sets once at start-up, so switching modes never re-runs it.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

from PIL import Image

from safesite.types import Box, Detection

DEFAULT_WEIGHTS = "yoloe-26n-seg.pt"  # smallest YOLOE-26; see Ultralytics YOLOE docs
WEIGHTS = os.environ.get("SAFESITE_YOLOE_WEIGHTS", DEFAULT_WEIGHTS)
CONF_THRESHOLD = 0.15  # low on purpose: rules.py applies OPEN_VOCAB_MIN_CONFIDENCE (0.25)
IMG_SIZE = 640
SOURCE = "yoloe"
BOX_DECIMALS = 4

# Text prompt -> label used by rules.py. Prompt wording is what YOLOE "reads";
# the label is what the rules engine matches on. Order fixes YOLOE class ids.
SITE_PROMPTS: Mapping[str, str] = {
    "safety harness": "harness",
    "scaffolding": "scaffold",
    "ladder": "ladder",
    "shade tent": "shade_tent",
    "water cooler": "water_cooler",
    "extension cord": "cord",
    "cable on the floor": "cord",  # finds real floor cords that "extension cord" misses
    "hose": "hose",
    # v2: only prompts that passed docs/prompt_bench.md (cut-offs live in rules.OPEN_VOCAB_CUTOFFS)
    "power pole": "utility_pole",  # -> power_line_contact
    "exposed rebar": "rebar",  # -> exposed_rebar
    "steel reinforcement bars sticking up": "rebar",
}
HOME_PROMPTS: Mapping[str, str] = {
    "extension cord": "cord",
    "cable on the floor": "cord",
    "wet floor": "wet_floor",
    "knife": "knife",
    "cleaning chemicals": "chemicals",
}
PROMPT_SETS: Mapping[str, Mapping[str, str]] = {"site": SITE_PROMPTS, "home": HOME_PROMPTS}

# Chat 4's stub Detection has a ``source`` field; Chat 3's final types.py does not.
_DETECTION_HAS_SOURCE = "source" in {f.name for f in fields(Detection)}


def _mode_key(mode: object) -> str:
    """'site' / 'HOME' / Mode.HOME (str Enum) -> 'site' | 'home'."""
    return str(getattr(mode, "value", mode)).strip().lower()


def prompts_for(mode: object) -> Mapping[str, str]:
    """Prompt -> label mapping for a mode; raises ValueError for unknown modes."""
    key = _mode_key(mode)
    if key not in PROMPT_SETS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(PROMPT_SETS)}")
    return PROMPT_SETS[key]


def _clamped_box(raw: Sequence[float]) -> Box | None:
    """Clamp an xyxyn box into [0, 1] and round it; None if no area is left."""
    x1, y1, x2, y2 = (round(min(1.0, max(0.0, float(v))), BOX_DECIMALS) for v in raw)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _make_detection(label: str, confidence: float, box: Box) -> Detection:
    conf = min(1.0, max(0.0, float(confidence)))
    if _DETECTION_HAS_SOURCE:
        return Detection(label, conf, box, source=SOURCE)  # type: ignore[call-arg]
    return Detection(label, conf, box)


def to_detections(
    boxes_xyxyn: Sequence[Sequence[float]],
    confidences: Sequence[float],
    class_ids: Sequence[float],
    prompts: Sequence[str],
    prompt_labels: Mapping[str, str],
) -> list[Detection]:
    """Raw YOLOE output (normalized xyxy, conf, class id) -> rule-ready Detections.

    ``prompts`` is the class list YOLOE was set with (index = class id);
    ``prompt_labels`` maps each prompt to its rules.py label.
    """
    detections: list[Detection] = []
    for raw_box, conf, cls in zip(boxes_xyxyn, confidences, class_ids):
        box = _clamped_box(raw_box)
        if box is None:
            continue
        label = prompt_labels[prompts[int(cls)]]
        detections.append(_make_detection(label, conf, box))
    return detections


def _load_yoloe(weights: str) -> Any:
    from ultralytics import YOLOE  # deferred: heavy import, only for the real model

    return YOLOE(weights)


class OpenVocabDetector:
    """YOLOE-26 text-prompt finder on CPU. Call with (PIL image, mode) -> list[Detection]."""

    is_mock = False
    supports_home = True

    def __init__(
        self,
        weights: str = WEIGHTS,
        conf: float = CONF_THRESHOLD,
        imgsz: int = IMG_SIZE,
        model: Any | None = None,  # inject a fake in tests
    ) -> None:
        self._model = model if model is not None else _load_yoloe(weights)
        self._conf = conf
        self._imgsz = imgsz
        # Encode every prompt set once; set_classes with cached embeddings is cheap.
        self._embeddings = {
            key: self._model.get_text_pe(list(prompts)) for key, prompts in PROMPT_SETS.items()
        }
        self._active: str | None = None
        self._lock = threading.Lock()
        self._activate("site")
        self.name = f"YOLOE-26 open-vocab ({os.path.basename(weights)})"
        # Site labels this finder can produce (pipeline's "not checked" note is Site-only).
        self.labels = frozenset(SITE_PROMPTS.values())
        # Warm-up at boot: the first real predict is ~5 s slower than the rest on CPU.
        self._model.predict(Image.new("RGB", (imgsz, imgsz)), conf=conf, imgsz=imgsz,
                            device="cpu", verbose=False)

    def _activate(self, key: str) -> None:
        if key == self._active:
            return
        self._model.set_classes(list(PROMPT_SETS[key]), self._embeddings[key])
        self._active = key

    def __call__(self, image: Image.Image, mode: object = "site") -> list[Detection]:
        prompt_labels = prompts_for(mode)
        rgb = image.convert("RGB")
        # set_classes mutates the shared model, so switch + predict must not interleave
        # across concurrent requests, or class ids would index the other prompt set.
        with self._lock:
            self._activate(_mode_key(mode))
            result = self._model.predict(
                rgb, conf=self._conf, imgsz=self._imgsz, device="cpu", verbose=False
            )[0]
        boxes = result.boxes
        return to_detections(
            boxes.xyxyn.tolist(),
            boxes.conf.tolist(),
            boxes.cls.tolist(),
            tuple(prompt_labels),
            prompt_labels,
        )
