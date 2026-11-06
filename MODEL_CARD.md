---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - qai-hf-hackathon-2026
  - object-detection
  - yolo
  - yolo26
  - ultralytics
  - construction-safety
  - ppe-detection
datasets:
  - keremberke/hard-hat-detection
  - keremberke/construction-safety-object-detection
  - Francesco/construction-safety-gsnvb
---

# Salamtak سلامتك — PPE & site-hazard detector (YOLO26s)

Fine-tuned **Ultralytics YOLO26s** that finds people, helmets, missing helmets, vests, missing vests, machinery and vehicles in construction-site photos.
It is the trained detector inside **Salamtak سلامتك** ("your safety"), built at the Qai × Hugging Face Hackathon (Doha, 5–7 Nov 2026). A plain-Python rules engine (no LLM) turns these detections into hazards, a 0–100 safety score and multilingual worker alerts.

- **Demo Space:** [plitip/team-qh-091-safesite-app](https://huggingface.co/spaces/plitip/team-qh-091-safesite-app)
- **Weights:** `best.pt` in this repo · training log `results.csv` · summary `metrics.json`

## Classes

| id | class | meaning |
|---|---|---|
| 0 | `person` | a worker / person |
| 1 | `helmet` | head wearing a hard hat |
| 2 | `no_helmet` | head without a hard hat |
| 3 | `vest` | person wearing a hi-vis vest |
| 4 | `no_vest` | person without a hi-vis vest |
| 5 | `machinery` | excavators, wheel loaders |
| 6 | `vehicle` | dump trucks, trucks, vans |

## Results

Validation split = the source datasets' own `valid` splits only (4,109 images), never used for training.

| metric | value |
|---|---|
| **mAP50** | **0.549** |
| mAP50-95 | 0.355 |
| precision | 0.473 |
| recall | 0.648 |

Per-class mAP50 (small validation box counts mean noisy scores):

| class | mAP50 | val boxes |
|---|---|---|
| helmet | **0.939** | 9,262 |
| no_helmet | **0.928** | 2,255 |
| machinery | 0.876 | 14 |
| vehicle | 0.663 | 8 |
| vest | 0.186 | 135 |
| person | 0.171 | 237 |
| no_vest | 0.079 | 75 |

The headline use case — helmet vs. no helmet — is strong (≈ 0.93). Machinery and vehicle look good but rest on only 14 and 8 validation boxes, so treat those numbers as rough. Person and vest/no_vest are weak (see Limitations).

Training: YOLO26s from COCO-pretrained `yolo26s.pt`, imgsz 640, batch 32, **36 epochs** (1.75 h time budget) on 1× NVIDIA L4 via Hugging Face Jobs — 107 min billed, ≈ $1.43. Script: `scripts/train.py` in the project repo; full curves in `results.csv`.

## Training data and licenses

Merged into one YOLO-format dataset (private: `plitip/team-qh-091-safesite-data`) — **17,240 train / 4,109 val images**.

| source | license | used for |
|---|---|---|
| [keremberke/hard-hat-detection](https://huggingface.co/datasets/keremberke/hard-hat-detection) | CC BY 4.0 | helmet / no_helmet |
| [keremberke/construction-safety-object-detection](https://huggingface.co/datasets/keremberke/construction-safety-object-detection) | CC BY 4.0 | person, helmet, vest, machinery, vehicle (+ "no-" classes) |
| [Francesco/construction-safety-gsnvb](https://huggingface.co/datasets/Francesco/construction-safety-gsnvb) | CC (as listed on the dataset page) | person, helmet, vest (+ "no-" classes) |

Dropped source classes: barricade, dumpster, gloves, mask, no-mask, safety net, safety shoes.

## Intended use

- Flagging likely PPE gaps (missing helmet or vest) and people near machinery/vehicles in **still photos of construction sites**, as input to a safety checklist reviewed by a person.
- Supervisor spot checks, toolbox talks, training material.

**Not intended for:** automated discipline or monitoring of individual workers, identifying people, legal or compliance decisions without human review, or any safety-critical decision on its own.

## Limitations

- **`person` is weak (mAP50 0.17).** The largest source, hard-hat-detection, shows many people but labels only their heads, so most people in training are unlabelled and the model learns to ignore them. Rely on helmet / no_helmet boxes (head-level) for PPE checks; a stock COCO person detector or YOLOE is better for counting people.
- **Vest / no_vest are weak (0.19 / 0.08)** for the same reason — vests appear in hard-hat-detection photos without labels — plus few examples (~1,300 / ~900 boxes).
- **Machinery and vehicle are very sparse** in training data (~190 and ~100 boxes) and in validation (14 and 8 boxes), so their scores are unreliable. The Salamtak app uses open-vocabulary YOLOE for extra objects.
- Small, distant or heavily occluded heads are often missed; "no_helmet" can be confused with caps, scarves or hair.
- Training photos are mostly not from Qatar; lighting, heat haze, clothing (e.g. head coverings under helmets) and site layouts in the Gulf may lower accuracy. Our separate Qatar test photos measure this honestly.
- A detector only sees what is in frame; absence of a detection is not proof of safety.

## License — AGPL-3.0

Built with **Ultralytics YOLO**, which is licensed under **AGPL-3.0**. These fine-tuned weights are released under AGPL-3.0 as well: if you use them in a network service, you must make your full source code available under the same license. Commercial use without open-sourcing requires an [Ultralytics Enterprise License](https://www.ultralytics.com/license).
Training data is CC BY 4.0 / CC — credit the dataset authors above.

## How to use

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download("plitip/team-qh-091-safesite", "best.pt")
model = YOLO(weights)
results = model("site_photo.jpg", conf=0.25)
results[0].show()
```
