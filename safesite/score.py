"""Safety score: max(0, 100 - 25*#High - 10*#Medium).

STUB written by the app chat (Chat 4). Chat 3 owns this file.
Rules already emit at most one hazard per id per person, so this just counts.
"""
from __future__ import annotations

from collections.abc import Iterable

from safesite.types import Hazard

PENALTY = {"High": 25, "Medium": 10}


def safety_score(hazards: Iterable[Hazard]) -> int:
    return max(0, 100 - sum(PENALTY[h.severity] for h in hazards))
