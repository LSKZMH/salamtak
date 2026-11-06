# Results v2: Salamtak سلامتك hazard evaluation

Status: **in progress** (Chat 10). "Before" numbers are in; the v1 + new rules and v2 columns fill in as those ship.

## Headline table

| Metric | Before (v1, old rules) | v1 + new rules | v2 + new rules |
|---|---|---|---|
| Qatar set: catch rate | **25%** (2/8) · 33% (2/6) on the original v1 hazard ids | | |
| Qatar set: false alarms per photo | **0.00** (0 on 6 clean photos) | | |
| ConstructionSite subset: catch rate per rule | no_helmet **29%** (46/156) · machinery_zone **9%** (2/23) · fall_no_harness **0%** (0/23) | same (identical predictions on all 490 photos) | |
| ConstructionSite subset: false-alarm rate per rule | no_helmet 12% (40/334) · machinery_zone 1% (4/467) · fall_no_harness 1% (7/467) | same | |
| CPU seconds per photo (Space, server-side) | **0.10 mean · 0.09 median · 0.18 max** | | |
| End-to-end round trip from Doha (upload + queue + result) | 8.5 s median (7.8–13.0 s) | | |

Gate 3 criterion 4 (≤ 3.0 s per photo in the Space): **pass** on v1 (0.18 s worst case).

**Preview, v1 + new rules before deploy** (local, `main` at `fa53518`, `--detector combined`): same 25% (2/8) catch rate and
0 false alarms. The new rules have no inputs yet: `crane_hook`, `rebar`, `utility_pole`, `power_line`, `fire`, `welding`, …
come only from Chat 9's new YOLOE prompts or the v2 weights. No regression: `worker_down` did not fire on the seated
worker or the crowd photos. The official column is measured on the Space after Chat 9 deploys.

## How each number was produced

- **Qatar set:** `examples/qatar_test/labels.csv`, 12 photos (9 Qatar, 3 Dubai; 6 with hazards, 6 clean), never used for training.
  Run as `python scripts/evaluate.py --space plitip/team-qh-091-safesite-app`, which sends each photo to the live Space
  with a pinned November-morning time (no heat rule fires) and reads back the hazard table and the Space's own timing.
  The same photos run locally (`--detector combined`, trained YOLO26 + YOLOE) gave identical hazards.
- **Space CPU:** cpu-upgrade (8 vCPU). Seconds are measured inside the Space's `analyze()` (detect + rules + score + annotate).
  Local laptop timings are not used: five chats share that CPU (100% load during the run), which inflated them 4–10×.
- **Before = current `main`** at `d60cfbb`: v1 weights `plitip/team-qh-091-safesite@835deaa`, old rules, YOLOE-26n.

## Qatar set: per hazard (before)

| hazard | labelled | caught | false alarms |
|---|---|---|---|
| `no_helmet` | 4 | 2 | 0 / 8 |
| `machinery_zone` | 2 | 0 | 0 / 10 |
| `under_suspended_load` | 1 | 0 (rule did not exist yet) | 0 / 11 |
| `exposed_rebar` | 1 | 0 (rule did not exist yet) | 0 / 11 |

Misses: `doha_cable_drums_workers` (cloth head wrap, no hard hat), `westbay_shift_change_crowd` (a few bare heads in a crowd of hundreds),
`doha_alsadd_excavator` and `dubai_night_precast_lift` (machinery zone; v1 detects `person` and `machinery` weakly).

## ConstructionSite outside benchmark

`scripts/bench_constructionsite.py` draws a stratified seed-0 subset of the **test** split (up to 150 positives per rule + 300 with no violation)
and scores only the benchmark's rules (`--only-hazards`), so our other hazards are ignored rather than counted as false alarms.

| Benchmark rule | Our hazard | Scored |
|---|---|---|
| Rule 1 basic PPE | `no_helmet` | Only Rule 1 images whose reason is about a hard hat; Rule 1 images about other PPE (vests, clothing, glasses) are left out |
| Rule 2 harness at height | `fall_no_harness` | yes |
| Rule 3 edge protection | `fall_into_opening` | only if the `floor_opening` / `trench` prompts ship |
| Rule 4 near operating excavator | `machinery_zone` | yes |

**Subset (seed 0, test split of 3,004 images):** 490 photos = 156 `no_helmet` + 23 `fall_no_harness` + 23 `machinery_zone`
positives (every Rule 2 and Rule 4 positive in the test split; they overlap little) + 300 with no violation.
Rule 1: 324 images, 267 with a hard-hat reason (25 of 25 spot-checked correct), 57 about other PPE (vests at night, shorts,
slippers, sleeveless shirts), left out. Rule 3 has 63 positives, not scored: no `floor_opening`/`trench` input ships yet.

| hazard | labelled | caught | catch rate | false alarms | false-alarm rate |
|---|---|---|---|---|---|
| `no_helmet` | 156 | 46 | 29% | 40 / 334 | 12% |
| `machinery_zone` | 23 | 2 | 9% | 4 / 467 | 1% |
| `fall_no_harness` | 23 | 0 | 0% | 7 / 467 | 1% |

Overall 24% (48/202) caught, 0.10 false alarms per photo, 41 of 300 violation-free photos flagged.
Before = `d60cfbb` (old rules); v1 + new rules = `main` at `fa53518`; both trained YOLO26 v1 + YOLOE-26n run locally
(`--detector combined`, the Space's stack). Run: `python scripts/evaluate.py --labels C:/sqcs/subset/labels.csv
--only-hazards no_helmet,fall_no_harness,machinery_zone --detector combined --threads 8` (`--repo-root` an old checkout for "before").

## Caveats (keep these next to any number on a slide)

- **Small Qatar set:** 12 photos and 8 labelled hazards; one photo changes the catch rate by 12 points. Labels were drafted by Claude and
  still need a teammate's check.
- **ConstructionSite is not Gulf sites**, and it is evaluation-only (card license CC BY-NC 4.0, gated): never trained on, never committed.
- **Possible leakage for v2:** ConstructionSite's images come from MOCS, and the v2 training mix includes Roboflow `yisong-li/mocs`.
  v2 may have seen some benchmark images in training, so a v2 gain on ConstructionSite is an upper bound.
- **Rule 1 mapping:** Rule 1 covers all basic PPE; we score helmets only.
- **Rules 2 and 4 have 23 positives each** in the whole test split: one photo moves their catch rate by 4 points.
- **Benchmark "no violation" is the annotators' judgement**, and its photos are mostly MOCS sites in China, often wide shots
  with small, distant workers, harder than our demo photos. Some `no_helmet` "false alarms" may be real bare heads they didn't flag.

## Detector mAP50 (from Chat 7)

_To fill in from `metrics.json`._ v1 on its own val split: mAP50 0.549 (helmet 0.94, no_helmet 0.93).

## Which weights the Space runs

`plitip/team-qh-091-safesite`, `best.pt` @ `835deaa` (v1). _Update if Gate 3 switches to v2._

## Credits

From `COSTS.md`: **$2.01** so far (Jobs $1.44 + Space ≈ $0.57). _Final total at the end of the night._
