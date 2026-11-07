# Results v2: Salamtak سلامتك hazard evaluation

Status: **the Space runs v2n + the helmet fix** (captain's call, deployed by Chat 9 as `0f2541a`). Measured live (Chat 10, 07 Nov).
v2 = `plitip/team-qh-091-safesite` `v2/best.pt` (yolo26s, 61 epochs); **v2n = `v2n/best.pt` @ `b3fac75` (yolo26n, 83 epochs), the shipped model.**

## Pitch numbers (shipped: v2n + new rules + helmet fix, Space at `0f2541a`)

Measured live on the Space after the deploy (`python scripts/evaluate.py --space plitip/team-qh-091-safesite-app`):

| | v1 (before) | **Shipped** |
|---|---|---|
| Qatar set: hazards caught | 25% (2/8) | **62% (5/8)** |
| Qatar set: false alarms per photo | 0.00 | **0.00** (0 of 6 clean photos flagged) |
| CPU seconds per photo in the Space | 0.10 mean · 0.18 max | **0.09 mean · 0.14 max** in this run (Chat 9 saw 0.09–0.21 s) |
| ConstructionSite subset (outside benchmark, local) | 24% caught · 0.10 FA/photo | **61% caught (123/202) · 0.32 FA/photo** |
| Credits spent | | **≈ $8.95** (`COSTS.md`) |

Round trip from Doha in this run was 8.7 s median (7.9–11.6 s), which is mostly upload and queue time.

## Headline table

The v2n column below is v2n with the rules **before** the helmet fix (live Space, 07 Nov ~10:00).

| Metric | Before (v1, old rules) | v1 + new rules | v2 (s) + new rules | **v2n + new rules (shipped)** |
|---|---|---|---|---|
| Qatar set: catch rate | **25%** (2/8) · 33% (2/6) on the original v1 hazard ids | **25%** (2/8), live Space | **62%** (5/8), local | **62%** (5/8), live Space |
| Qatar set: false alarms per photo | **0.00** (0 on 6 clean photos) | **0.00**, live Space | **0.25** (3 false alarms; 1 of 6 clean photos flagged) | **0.08** (1 false alarm; 0 of 6 clean photos flagged) |
| ConstructionSite subset: catch rate per rule | no_helmet **29%** (46/156) · machinery_zone **9%** (2/23) · fall_no_harness **0%** (0/23) | same (identical predictions on all 490 photos) | no_helmet **91%** (142/156) · machinery_zone **78%** (18/23) · fall_no_harness **9%** (2/23); overall **80%** (162/202) | no_helmet **79%** (123/156) · machinery_zone **65%** (15/23) · fall_no_harness **9%** (2/23); overall **69%** (140/202) |
| ConstructionSite subset: false-alarm rate per rule | no_helmet 12% (40/334) · machinery_zone 1% (4/467) · fall_no_harness 1% (7/467) | same | no_helmet 31% (104/334) · machinery_zone 26% (122/467) · fall_no_harness 2% (11/467); 0.48 per photo, 111/300 clean flagged | no_helmet 32% (108/334) · machinery_zone 18% (86/467) · fall_no_harness 2% (9/467); 0.41 per photo, 107/300 clean flagged |
| CPU seconds per photo (Space, server-side) | **0.10 mean · 0.09 median · 0.18 max** | **0.09 mean · 0.08 median · 0.10 max** | ≈ same as v1: trained detector v2/v1 median time ratio **0.98** (alternated on the same photos, local) | **0.08 mean · 0.07 median · 0.17 max** |
| End-to-end round trip from Doha (upload + queue + result) | 8.5 s median (7.8–13.0 s) | 5.9 s median (4.8–7.1 s) | not deployed | 6.5 s median (5.4–7.5 s) |

Gate 3 criterion 4 (≤ 3.0 s per photo in the Space): **pass** on v1 (0.18 s worst case).

## Gate 3 (v2 replaces v1?) — Chat 10's criteria

| # | Criterion | v1 + new rules | v2 + new rules | Result |
|---|---|---|---|---|
| 3 | Qatar catch rate not lower | 25% | 62% | pass |
| 3 | Qatar false alarms per photo not higher by > 0.1 | 0.00 | 0.25 (+0.25) | **FAIL** |
| 4 | CPU s/photo in the Space ≤ 3.0 | 0.09 mean · 0.10 max | ≈ same (ratio 0.98) | pass |

From Chat 7's `metrics.json` (Chat 7 owns 1–2): criterion 1 helmet 0.937 vs floor 0.919 pass, **no_helmet 0.905 vs floor
0.908 fails by 0.003**; criterion 2 (fully labelled cs/fr/chd/ppe) person 0.80 vs 0.50, vest 0.84 vs 0.56, no_vest 0.62 vs 0.22: pass.
**As written, Gate 3 fails for v2 (s): keep v1 weights.**

**Decision: the captain overrode Gate 3 and the Space ships v2n** (`v2n/best.pt` @ `b3fac75`). Measured on the live Space with the
same rules, v2n passes Chat 10's criteria:

| # | Criterion | v1 + new rules | v2n + new rules (live) | Result |
|---|---|---|---|---|
| 3 | Qatar catch rate not lower | 25% | 62% | pass |
| 3 | Qatar false alarms per photo not higher by > 0.1 | 0.00 | 0.08 (+0.08) | pass |
| 4 | CPU s/photo in the Space ≤ 3.0 | 0.09 mean · 0.10 max | 0.08 mean · 0.17 max | pass |

v2n's one false alarm is `no_helmet` on `dubai_night_precast_lift` (night shot). Criterion 1 for v2n (Chat 7's metrics, v1 val):
helmet 0.932 vs floor 0.919 pass, no_helmet 0.897 vs floor 0.908 **misses by 0.011**. Criterion 2: person/vest/no_vest
0.761/0.827/0.587 vs v1 0.50/0.56/0.22, pass.

**Helmet absence fix** (Chat 8, `main` at `4f1b5a1`, merged 10:25 after the freeze; **deployed** to the Space as `0f2541a`, and the live Qatar result matches the local one below):
"no helmet seen" now needs a person with confidence ≥ 0.5 that isn't touching the frame edge, while an explicit bare-head box
still counts. Measured locally with v2n + YOLOE. Local `main` before the fix reproduced the live Space exactly on the Qatar set.

| v2n + YOLOE | Qatar caught | Qatar false alarms | ConstructionSite caught | ConstructionSite false alarms |
|---|---|---|---|---|
| rules before the fix (live Space) | 62% (5/8) | 0.08 per photo (1) | 69% (140/202) | 0.41 per photo, 107/300 clean flagged |
| with the fix | 62% (5/8) | **0.00** (0) | 61% (123/202) | **0.32** per photo, 80/300 clean flagged |

The fix only changes `no_helmet`. On ConstructionSite, catches fall from 79% to 68% (123 to 106 of 156) and false alarms fall
from 32% to 19% (108 to 63 of 334). The 17 lost catches are bare heads on small or edge-cut workers, which are common in
ConstructionSite's wide shots. `machinery_zone` and `fall_no_harness` are unchanged.

**Why v2's false alarms rise, and a possible fix (Chat 8's call):** all 3 Qatar false alarms come from rule geometry, not from v2
seeing a hazard that isn't there. v2 finds far more people, which exposes it:
- `no_helmet` fires on *absence* of a helmet box: a small far worker who wears a helmet (`doha_alsadd_excavator`) and a 0.46-confidence
  "person" cut off at the frame edge (`pearl_qanat_mast_platform`).
- `machinery_zone` measures 2D distance: a tiny background vehicle next to foreground workers (`doha_roadside_crew_meeting`).

Simulation (experiment only, detections filtered before the unchanged rules: drop persons with confidence < 0.5 or touching the frame
edge, drop vehicle/machinery boxes < 5% of image height): Qatar **62% caught, 0.08 false alarms per photo, 0/6 clean flagged**, which
passes criterion 3. On the ConstructionSite subset the same simulation gives **72% caught (146/202), 0.33 false alarms per photo**
(plain v2: 80%, 0.48). The person filter is the good half (`no_helmet` 86% caught, false alarms 31% → 17%); the fixed 5%-height
vehicle filter is too blunt for wide shots (`machinery_zone` caught 78% → 43%), so compare the machine's height with the person's
instead of a fixed cut. In `rules.py` terms: use the "no helmet box on the head" clause only for persons ≥ 0.5 confidence not touching the
border (else require an explicit `no_helmet` box), and skip machinery/vehicle boxes much shorter than the person.

**Preview, v1 + new rules before deploy** (local, `main` at `fa53518`, `--detector combined`): same 25% (2/8) catch rate and
0 false alarms. The new rules have no inputs yet: `crane_hook`, `rebar`, `utility_pole`, `power_line`, `fire`, `welding`, …
come only from Chat 9's new YOLOE prompts or the v2 weights. No regression: `worker_down` did not fire on the seated
worker or the crowd photos. The official column is measured on the Space after Chat 9 deploys.

**Preview, v2 mid-run checkpoint** (`v2/checkpoints/last.pt` at epoch 35, 03:15; local, `--detector combined`, new rules):
Qatar catch rate **62% (5/8)**: all 4 `no_helmet` and 1 of 2 `machinery_zone`. False alarms **0.17 per photo** (2), clean photos
flagged 0/6. Both false alarms come from rule geometry that v2's better person detection now exposes:
`doha_alsadd_excavator` (no_helmet on a small far worker who wears a helmet too small to detect) and
`doha_roadside_crew_meeting` (machinery_zone from a tiny background vehicle next to foreground workers in 2D).
With 12 photos, Gate 3 criterion 3 (false alarms up by at most 0.1 per photo) fails at 2 false alarms and passes at 1.
Superseded by the final v2 numbers above.

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
- **v2's ConstructionSite gain (24% → 80%) is an upper bound** because of the MOCS overlap above, and its false alarms rise too (0.10 → 0.48 per photo);
  some `machinery_zone` "false alarms" may be workers near idle excavators, which Rule 4 (operating excavator, blind spot) does not count (not checked).
- **Rules 2 and 4 have 23 positives each** in the whole test split: one photo moves their catch rate by 4 points.
- **Benchmark "no violation" is the annotators' judgement**, and its photos are mostly MOCS sites in China, often wide shots
  with small, distant workers, harder than our demo photos. Some `no_helmet` "false alarms" may be real bare heads they didn't flag.

## Detector mAP50 (from Chat 7)

Both models on the **same** images (Chat 7's `v2/metrics.json`):

| Val set | Model | person | helmet | no_helmet | vest | no_vest | machinery | vehicle | mean |
|---|---|---|---|---|---|---|---|---|---|
| v1 val (hh, cs, fr; 4,109 imgs) | v1 | 0.17 | 0.94 | 0.93 | 0.19 | 0.08 | 0.86 | 0.67 | 0.547 |
| v1 val | v2 | 0.02 | 0.94 | 0.90 | 0.07 | 0.01 | 0.15 | 0.23 | 0.331 |
| fully labelled (cs, fr, chd, ppe; 1,016 imgs) | v1 | 0.50 | 0.61 | 0.13 | 0.56 | 0.22 | 0.31 | 0.01 | 0.333 |
| fully labelled | v2 | **0.80** | **0.82** | **0.67** | **0.84** | **0.62** | **0.78** | **0.50** | **0.719** |

v2 overall on its own 11-class val (6,380 imgs): mAP50 **0.524** (harness 0.85, scaffold 0.99, utility_pole 0.18).
The v1 val set has unlabelled persons/vests in its biggest source (hh), so its person/vest/machinery rows punish a model that finds them;
the fully labelled set is the fair comparison.

## Which weights the Space runs

**v2n**: `plitip/team-qh-091-safesite`, `v2n/best.pt` @ `b3fac75` (Space variables `SAFESITE_DETECTOR=yolo`,
`SAFESITE_WEIGHTS_FILE=v2n/best.pt`, `SAFESITE_MODEL_REVISION=b3fac75…`), plus YOLOE-26n, on cpu-upgrade. Checked from the Space
variables and from the detector name the Space reports (`YOLO26 (plitip/team-qh-091-safesite@b3fac75)`).
Rollback: set `SAFESITE_WEIGHTS_FILE=best.pt` and `SAFESITE_MODEL_REVISION=835deaa…` (v1).

## Credits

From `COSTS.md`: **$8.95** so far (Jobs $8.38 + Space ≈ $0.57; 10 rows). _Update from the billing page at the end._
