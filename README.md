---
title: Salamtak سلامتك
emoji: 🦺
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Site photo to hazards, safe/unsafe verdict, worker alerts
tags:
  - qai-hf-hackathon-2026
  - object-detection
  - yolo
  - construction-safety
  - qatar
models:
  - plitip/team-qh-091-safesite
---

# 🦺 Salamtak سلامتك

One photo of a construction site → detected hazards, a **Safe / Needs attention / Unsafe** verdict, and short alerts
in workers' languages (English, Arabic, Hindi, Bengali, Nepali, Urdu, Malayalam). Built for the Qai × Hugging Face
Hackathon, Doha, November 2026, by team QH-091.

![Annotated result for a Doha roadside crew at a summer noon: four workers flagged for no hard hat, plus the Qatar midday work ban, the WBGT stop-work rule and missing shade or water](docs/demo_doha_crew.jpg)

*The first example in the app: a Doha crew at 12:30 on 15 July with WBGT 33.5 °C. Verdict **Unsafe**: "Stop work and
report to your supervisor now." Red boxes are High hazards; the strip at the bottom lists photo-wide hazards. Photo:
Michael Coghlan, CC BY-SA 2.0, via Wikimedia Commons (see `examples/qatar_test/CREDITS.md`); the annotated version is
shared under the same licence.*

## How it works

**No LLM anywhere in the pipeline.** Two vision models find objects, and a plain-Python rules engine decides what is
a hazard.

```
photo (+ photo time, optional WBGT, language)
   ├─► YOLO26n fine-tuned, 11 classes: person, helmet, no_helmet, vest, no_vest, machinery,
   │                                   vehicle, harness, scaffold, crane_hook, utility_pole
   └─► YOLOE-26 text prompts (no training): ladder, shade tent, water cooler, cord, hose, power pole, rebar, ...
            ▼
   rules engine (pure Python, Qatar heat rules included)  →  hazards with a one-line reason each
            ▼
   verdict + annotated photo + alerts in the chosen language
```

**Verdict:** **Unsafe** if any hazard is High ("Stop work and report to your supervisor now"), **Needs attention** if
all hazards are Medium ("Report to your supervisor"), **Safe** if none. A photo with no people or equipment detected is
**Not assessed**, never Safe. Advisory only: no alert does not mean no hazard.

**Tabs:** *Photo* (upload or webcam snapshot, with date/time, WBGT, outdoor, language and Site / Home mode) and
*Live* (webcam at about one frame per second, trained detector plus rules; a hazard shows once it is seen in 2 of
3 frames).

## What it checks (Site mode)

| hazard | severity | seen by |
|---|---|---|
| no helmet | High | YOLO26 |
| person in a machinery / vehicle zone | High | YOLO26 |
| worker down (lying on the ground: heat illness or fall) | High | YOLO26 |
| person under a suspended load (crane hook above them) | High | YOLO26 (little crane-hook data, so it rarely fires) |
| work at height (scaffold, ladder) without a harness | High | YOLO26 + YOLOE |
| machinery next to a power pole (power-line contact) | High | YOLO26 + YOLOE |
| midday outdoor work ban (1 Jun–15 Sep, 10:00–15:30) | High | photo time |
| WBGT above 32.1 °C with workers present | High | WBGT input |
| no vest near a vehicle | Medium | YOLO26 |
| exposed rebar with workers present (impalement) | Medium | YOLOE |
| no shade or water for outdoor workers | Medium | YOLOE |
| cord or hose across the walkway | Medium | YOLOE |

Home mode checks cords on the floor, wet floors, and knives or chemicals within a child's reach.

YOLOE prompts ship only if they pass our image-level bench (hit rate ≥ 0.6 and false alarms ≤ 10 %, also on plain
site photos): see [docs/prompt_bench.md](docs/prompt_bench.md). Rules whose objects no shipped model can see yet
(fire / hot work, missing extinguisher, floor openings and trenches) are listed under the verdict as
**not checked**, so a photo never looks safer than we can tell.

## Results

| | v1 (YOLO26s, 7 classes) | **shipped: v2n (YOLO26n, 11 classes)** |
|---|---|---|
| Qatar test set (12 of our own photos, never trained on): hazards caught | 25 % (2 of 8) | **62 % (5 of 8)** |
| Qatar test set: false alarms per photo | 0.00 | **0.00** |
| mAP50 on the v1 validation images: helmet / no_helmet | 0.939 / 0.928 | 0.932 / 0.897 |
| mAP50 on fully labelled sources: person / vest / no_vest | 0.497 / 0.564 / 0.218 | 0.761 / 0.827 / 0.587 |
| Seconds per photo on the Space (CPU, detector + YOLOE + rules) | about 0.14 | **0.10–0.19** |

v2n missed our pre-set gate on one number (no_helmet mAP50 0.897 against a floor of 0.908). The captain shipped it
anyway because it catches 2.5× as many hazards on the Qatar photos with no extra false alarms. The whole project
(training, evaluation and the Space) used about **$8.95** of the team's $100 compute credit.

**Known limits:** hard hats on small or distant workers are sometimes missed, so check no_helmet alerts against the
photo. Distance rules (machinery zone, power-line contact) work in 2-D image space and cannot see depth. The Qatar
test set is small (12 photos).

## Model and data

- Model: [plitip/team-qh-091-safesite](https://huggingface.co/plitip/team-qh-091-safesite), file `v2n/best.pt`
  at commit `b3fac75` (YOLO26n fine-tuned, 83 epochs on an L4; the model card lists data and licences). v1
  (`best.pt`) and v2 (`v2/best.pt`, YOLO26s) are in the same repo.
- Training data (merged into our private `plitip/team-qh-091-safesite-data-v2`, 22,922 train / 6,380 val images):
  [keremberke/hard-hat-detection](https://huggingface.co/datasets/keremberke/hard-hat-detection),
  [keremberke/construction-safety-object-detection](https://huggingface.co/datasets/keremberke/construction-safety-object-detection),
  [Francesco/construction-safety-gsnvb](https://huggingface.co/datasets/Francesco/construction-safety-gsnvb),
  Roboflow *Construction Hazard Detection*, *Helmet Harness Detection* and *Hybrid Dataset for Scaffold Detection*,
  and Ultralytics *construction-ppe* (CC BY 4.0, except Francesco's Roboflow 100 CC licence). Missing labels were
  filled in by a larger YOLO26x teacher model (pseudo-labels on 17,102 training images).
- Alerts: English phrases are final. Translations in the other six languages are drafts until a native speaker
  confirms them (`verified` flags in `safesite/alerts.json`).
- Example photos: Wikimedia Commons (CC BY-SA) and held-out Construction Hazard Detection validation photos
  (CC BY 4.0), credited in `examples/**/CREDITS.md`. Faces are covered or turned away.

## Configuration (Space variables)

| variable | default | live value | purpose |
|---|---|---|---|
| `SAFESITE_DETECTOR` | `mock` | `yolo` | `yolo` = the fine-tuned detector; `mock` = fixed fake boxes |
| `SAFESITE_MODEL_REVISION` | latest | `b3fac75…` | **Pin to the model repo commit SHA**; rollback = previous SHA |
| `SAFESITE_WEIGHTS_FILE` | `best.pt` | `v2n/best.pt` | weights file: `best.pt` = v1 (7 classes), `v2/best.pt` or `v2n/best.pt` = v2 (11 classes) |
| `SAFESITE_MODEL_REPO` | `plitip/team-qh-091-safesite` | default | model repo (only point this at our own repo: `.pt` files are pickles) |
| `SAFESITE_OPEN_VOCAB` | `1` | default | `0` turns the YOLOE finder off |
| `SAFESITE_THREADS` | all cores | default | torch CPU threads |

The detector refuses to start unless the weights' classes are exactly v1
(`person, helmet, no_helmet, vest, no_vest, machinery, vehicle`) or v2 (v1 + `harness, scaffold, crane_hook,
utility_pole`), in that order.

Dev: `pip install -r requirements-dev.txt && pytest -q`

License: AGPL-3.0 (Ultralytics).
