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

One photo of a construction site → detected hazards, a **Safe / Needs attention / Unsafe** verdict, and short alerts in workers' languages
(English, Arabic, Hindi, Bengali, Nepali, Urdu, Malayalam). Built for the Qai × Hugging Face Hackathon, Doha, Nov 2026.

**No LLM anywhere in the pipeline.** A fine-tuned YOLO26 detector and the open-vocabulary YOLOE-26 (text prompts,
no training) find objects; a plain-Python rules engine decides what is a hazard, including Qatar's
**summer midday outdoor work ban** (10:00–15:30, 1 Jun–15 Sep) and the **WBGT 32.1 °C stop-work threshold**.

- Model: [plitip/team-qh-091-safesite](https://huggingface.co/plitip/team-qh-091-safesite) (YOLO26 fine-tuned; model card lists data and licences)
- Training data: [keremberke/hard-hat-detection](https://huggingface.co/datasets/keremberke/hard-hat-detection),
  [keremberke/construction-safety-object-detection](https://huggingface.co/datasets/keremberke/construction-safety-object-detection),
  [Francesco/construction-safety-gsnvb](https://huggingface.co/datasets/Francesco/construction-safety-gsnvb)
  (merged into our private `plitip/team-qh-091-safesite-data`)
- Team: QH-091
- Verdict: **Unsafe** if any hazard is High ("Stop work and report to your supervisor now"), **Needs attention**
  if all are Medium ("Report to your supervisor"), **Safe** if none.
- Runs on CPU, about 1 s per photo.

## What it checks (Site mode)

| hazard | severity | seen by |
|---|---|---|
| no helmet | High | YOLO26 |
| person in a machinery / vehicle zone | High | YOLO26 |
| worker down (lying on the ground: heat illness or fall) | High | YOLO26 |
| work at height (scaffold, ladder) without a harness | High | YOLOE |
| machinery next to a power pole (power-line contact) | High | YOLOE |
| midday outdoor work ban (Jun–Sep, 10:00–15:30) | High | photo time |
| WBGT above 32.1 °C with workers present | High | WBGT input |
| no vest near a vehicle | Medium | YOLO26 |
| exposed rebar with workers present (impalement) | Medium | YOLOE |
| no shade or water for outdoor workers | Medium | YOLOE |
| cord or hose across the walkway | Medium | YOLOE |

Home mode checks cords on the floor, wet floors, and knives / chemicals within a child's reach.

YOLOE prompts ship only if they pass our image-level bench (hit rate ≥ 0.6, false alarms ≤ 10 %, also on
plain site photos): see [docs/prompt_bench.md](docs/prompt_bench.md). Rules whose objects no shipped model can see
yet (suspended loads, fire / hot work, missing extinguisher, open trenches) are listed under the verdict as
**not checked**, so a photo never looks safer than we can tell.

## Configuration (Space variables)

| variable | default | purpose |
|---|---|---|
| `SAFESITE_DETECTOR` | `mock` | `yolo` = the fine-tuned detector; `mock` = fixed fake boxes |
| `SAFESITE_MODEL_REVISION` | latest | **Pin to the model repo commit SHA**; rollback = previous SHA |
| `SAFESITE_WEIGHTS_FILE` | `best.pt` | weights file in the model repo (`best.pt` = v1, 7 classes; `v2/best.pt` = v2, 11 classes) |
| `SAFESITE_MODEL_REPO` | `plitip/team-qh-091-safesite` | model repo (only point this at our own repo: `.pt` files are pickles) |
| `SAFESITE_OPEN_VOCAB` | `1` | `0` turns the YOLOE finder off |
| `SAFESITE_THREADS` | all cores | torch CPU threads |

The detector refuses to start unless the weights' classes are exactly v1
(`person, helmet, no_helmet, vest, no_vest, machinery, vehicle`) or v2 (v1 + `harness, scaffold, crane_hook,
utility_pole`), in that order. A photo with no people or equipment detected is shown as **Not assessed**,
never as a safe 100/100.

Dev: `pip install -r requirements-dev.txt && pytest -q`

License: AGPL-3.0 (Ultralytics).
