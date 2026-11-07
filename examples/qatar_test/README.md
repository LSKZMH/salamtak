# Qatar test set (never used for training)

Photos hand-labelled with which hazards each one contains.
`python scripts/evaluate.py` scores the pipeline against `labels.csv`.

**Starter set (current):** 12 freely licensed Wikimedia Commons photos, 9 from Qatar and 3 from Dubai;
6 with a hazard and 6 clean. Credits and licences are in [CREDITS.md](CREDITS.md). Labels were drafted
from looking at each photo: **a teammate should check them before the numbers go on a slide.**
`taken_at` is set to a November morning so the heat rules stay off; this set measures the vision hazards.
Our own photos can be added as extra rows at any time.

## Labelling

One row per photo in `labels.csv`; put the photo in this folder.

| column | required | value |
|---|---|---|
| `filename` | yes | photo file name in this folder |
| `hazards` | yes (may be blank) | hazard ids present, separated by `;`. Blank = clean photo. Ids: `no_helmet`, `machinery_zone`, `fall_no_harness`, `midday_ban`, `wbgt_stop`, `no_vest_near_vehicle`, `heat_welfare_gap`, `trip_cord`, `worker_down`, `under_suspended_load`, `power_line_contact`, `hot_work_fire`, `no_fire_extinguisher`, `exposed_rebar`, `fall_into_opening`; Home mode: `home_trip`, `home_slip`, `home_child_reach` |
| `taken_at` | no | `YYYY-MM-DD HH:MM` Doha time. Blank = photo EXIF time, else a winter morning (no heat-ban hazard) |
| `wbgt_c` | no | WBGT in °C, if measured. Required when `wbgt_stop` is labelled |
| `outdoor` | no | `true` / `false` (default `true`) |
| `mode` | no | `site` / `home` (default `site`) |
| `notes` | no | anything; ignored by the script |

- Label what a safety officer would flag, not what you expect the model to see.
- Include clean photos too (blank `hazards`): they measure false alarms.
- `midday_ban` needs a time: fill `taken_at` unless the photo still has its EXIF time (WhatsApp strips it).
- The script exits with code 3 while any listed photo is missing.
- Only photos we took (with consent from anyone identifiable) or freely licensed ones credited in CREDITS.md. Blur faces if unsure.
