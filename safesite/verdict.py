"""Plain verdict for the report: Safe or Unsafe, plus what to do. No score number is shown.

Any hazard makes the photo Unsafe. The card colour keeps the severity: red if any hazard is High,
orange if all are Medium, green if none. Used by the Photo tab (app.py) and the Live tab (live.py).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from safesite.types import Hazard

SAFE, UNSAFE = "Safe", "Unsafe"
SITE_STOP = "Stop work and alert the site safety officer now."
SITE_REPORT = "Alert the site safety officer."
HOME_STOP = "Fix this now before someone gets hurt."
HOME_REPORT = "Fix this soon."


@dataclass(frozen=True)
class Verdict:
    css: str  # good | warn | bad: card colour
    word: str  # Safe | Unsafe
    action: str  # what to do next; "" when safe


def verdict(hazards: Sequence[Hazard], mode: str = "site") -> Verdict:
    """Verdict for the hazards found in one photo or frame (mode: "site" or "home")."""
    home = str(mode).lower().startswith("home")
    if any(h.severity == "High" for h in hazards):
        return Verdict("bad", UNSAFE, HOME_STOP if home else SITE_STOP)
    if hazards:
        return Verdict("warn", UNSAFE, HOME_REPORT if home else SITE_REPORT)
    return Verdict("good", SAFE, "")
