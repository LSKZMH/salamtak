"""Safety score: max(0, 100 - 25 * #High - 10 * #Medium).

Each hazard id counts at most once per person (``Hazard.subject``); scene-wide
hazards (subject None) count once. Pure functions, no model or I/O imports.
"""

from __future__ import annotations

from collections.abc import Iterable

from safesite.types import Hazard, Severity

MAX_SCORE = 100
MIN_SCORE = 0
PENALTY: dict[Severity, int] = {Severity.HIGH: 25, Severity.MEDIUM: 10}


def dedupe(hazards: Iterable[Hazard]) -> list[Hazard]:
    """Keep the first hazard for each (id, subject) pair, preserving order."""
    seen: set[tuple[str, str | None]] = set()
    unique = []
    for hazard in hazards:
        key = (hazard.id, hazard.subject)
        if key not in seen:
            seen.add(key)
            unique.append(hazard)
    return unique


def safety_score(hazards: Iterable[Hazard]) -> int:
    penalty = sum(PENALTY[h.severity] for h in dedupe(hazards))
    return max(MIN_SCORE, MAX_SCORE - penalty)
