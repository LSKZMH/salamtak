"""Qatar heat-stress rules (summer midday outdoor work ban and WBGT stop-work limit).

Pure functions on plain values, no model or I/O imports. ``rules.py`` turns
detections into worker counts / shade flags and calls these.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone

from safesite.types import Hazard

# Qatar is UTC+3 all year (no daylight saving), so a fixed offset is exact and
# avoids needing the tzdata package on Windows.
DOHA_TZ = timezone(timedelta(hours=3), "AST")

MIDDAY_BAN_START = time(10, 0)  # inclusive: work must have stopped at 10:00
MIDDAY_BAN_END = time(15, 30)  # exclusive: outdoor work may resume at 15:30
MIDDAY_BAN_FIRST_DAY = (6, 1)  # 1 June, inclusive
MIDDAY_BAN_LAST_DAY = (9, 15)  # 15 September, inclusive
WBGT_STOP_WORK_C = 32.1  # stop work when WBGT is strictly above this


def to_doha_time(moment: datetime) -> datetime:
    """Naive datetimes are already Doha local time; aware ones are converted."""
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(DOHA_TZ).replace(tzinfo=None)


def in_ban_season(day: date) -> bool:
    return MIDDAY_BAN_FIRST_DAY <= (day.month, day.day) <= MIDDAY_BAN_LAST_DAY


def in_ban_hours(clock: time) -> bool:
    return MIDDAY_BAN_START <= clock < MIDDAY_BAN_END


def is_midday_ban(moment: datetime) -> bool:
    local = to_doha_time(moment)
    return in_ban_season(local.date()) and in_ban_hours(local.time())


def _workers(count: int) -> str:
    return f"{count} worker" if count == 1 else f"{count} workers"


def midday_ban(taken_at: datetime | None, outdoor: bool) -> Hazard | None:
    """Outdoor photo taken 10:00-15:30 between 1 Jun and 15 Sep (Doha time)."""
    if not outdoor or taken_at is None or not is_midday_ban(taken_at):
        return None
    local = to_doha_time(taken_at)
    return Hazard.of(
        "midday_ban",
        f"Outdoor photo taken {local.day} {local:%b %H:%M} Doha time, inside the "
        "10:00-15:30 summer midday work ban (1 Jun-15 Sep).",
    )


def wbgt_stop(wbgt_c: float | None, worker_count: int) -> Hazard | None:
    """WBGT entered and above 32.1 C with workers present (indoors or outdoors)."""
    if wbgt_c is None or worker_count < 1:
        return None
    if not math.isfinite(wbgt_c):
        raise ValueError(f"WBGT must be a finite number, got {wbgt_c!r}")
    if wbgt_c <= WBGT_STOP_WORK_C:
        return None
    return Hazard.of(
        "wbgt_stop",
        f"WBGT {wbgt_c:g} °C is above the {WBGT_STOP_WORK_C:g} °C stop-work limit "
        f"with {_workers(worker_count)} present.",
    )


def heat_welfare_gap(
    outdoor: bool,
    worker_count: int,
    shade_seen: bool,
    water_seen: bool,
    taken_at: datetime | None = None,
) -> Hazard | None:
    """Outdoor workers in the summer heat season (1 Jun-15 Sep, the midday-ban window) and
    neither a shade tent nor a water cooler detected.

    Team decision 6 Nov 2026: outside summer a missing shade tent is not a heat hazard, and
    firing all year would put a false alarm on every outdoor photo. No time = not judged.
    """
    if not outdoor or worker_count < 1 or shade_seen or water_seen or taken_at is None:
        return None
    if not in_ban_season(to_doha_time(taken_at).date()):
        return None
    return Hazard.of(
        "heat_welfare_gap",
        f"{_workers(worker_count)} outdoors in the summer heat season and no shade tent "
        "or water cooler seen.",
    )
