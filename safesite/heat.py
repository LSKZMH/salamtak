"""Qatar heat rules: midday outdoor work ban and WBGT stop-work threshold.

STUB written by the app chat (Chat 4). Chat 3 owns this file and its tests.
Pure functions, no I/O.
"""
from __future__ import annotations

from datetime import date, datetime, time

from safesite.types import Hazard

BAN_START = time(10, 0)
BAN_END = time(15, 30)  # inclusive to the minute: 15:30:59 is still inside the ban
BAN_FIRST_DAY = (6, 1)  # 1 June
BAN_LAST_DAY = (9, 15)  # 15 September
WBGT_STOP_C = 32.1  # stop work when WBGT is strictly above this


def in_midday_ban(taken_at: datetime) -> bool:
    """True for 10:00-15:30 (minute resolution, both ends inclusive) between 1 Jun and 15 Sep."""
    day = taken_at.date()
    season = date(day.year, *BAN_FIRST_DAY) <= day <= date(day.year, *BAN_LAST_DAY)
    minute = taken_at.time().replace(second=0, microsecond=0)
    return season and BAN_START <= minute <= BAN_END


def heat_hazards(
    taken_at: datetime, outdoor: bool, wbgt: float | None, workers_present: bool
) -> list[Hazard]:
    """Frame-level heat hazards (no boxes): midday_ban and wbgt_stop."""
    hazards: list[Hazard] = []
    if outdoor and in_midday_ban(taken_at):
        hazards.append(
            Hazard(
                "midday_ban",
                "High",
                f"Outdoor photo at {taken_at:%H:%M} on {taken_at:%d %b}: "
                "inside Qatar's 10:00-15:30 summer outdoor work ban.",
            )
        )
    if wbgt is not None and wbgt > WBGT_STOP_C and workers_present:
        hazards.append(
            Hazard(
                "wbgt_stop",
                "High",
                f"WBGT {wbgt:.1f} °C is above the {WBGT_STOP_C} °C stop-work threshold "
                "and workers are present.",
            )
        )
    return hazards
