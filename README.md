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
short_description: Site photo to hazards, safety score and worker alerts
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

One photo of a construction site → detected hazards, a 0–100 safety score, and short alerts in workers' languages
(English, Arabic, Hindi, Bengali, Nepali, Urdu, Malayalam). Built for the Qai × Hugging Face Hackathon, Doha, Nov 2026.

**No LLM anywhere in the pipeline.** A fine-tuned YOLO26 detector finds objects; a plain-Python rules engine decides
what is a hazard, including Qatar's **summer midday outdoor work ban** (10:00–15:30, 1 Jun–15 Sep) and the
**WBGT 32.1 °C stop-work threshold**.

- Model: [plitip/team-qh-091-safesite](https://huggingface.co/plitip/team-qh-091-safesite)
- Team: QH-091
- Score: `max(0, 100 − 25·High − 10·Medium)`
- Runs on free CPU.

> Status: the app currently uses a **mock detector** (fixed fake boxes) while the model trains.
> Example images are synthetic placeholders.

## Configuration (Space variables)

| variable | default | purpose |
|---|---|---|
| `SAFESITE_MODEL_REVISION` | latest | **Pin to the model repo commit SHA before the demo**; rollback = previous SHA |
| `SAFESITE_MODEL_REPO` | `plitip/team-qh-091-safesite` | model repo (only point this at our own repo: `.pt` files are pickles) |
| `SAFESITE_THREADS` | all cores | torch CPU threads |

The detector refuses to start if the weights' class names don't match the rules
(`person, helmet, no_helmet, vest, no_vest, machinery, vehicle`), and a photo with no people or
equipment detected is shown as **Not assessed**, never as a safe 100/100.

Dev: `pip install -r requirements-dev.txt && pytest -q`

License: AGPL-3.0 (Ultralytics).
