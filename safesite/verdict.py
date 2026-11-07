"""Plain verdict for the report: Safe, Needs attention or Unsafe, plus what to do. No score is shown.

Unsafe if any hazard is High, Needs attention if all hazards are Medium, Safe if none. Every
non-safe verdict tells the worker to report to their supervisor. Used by the Photo tab (app.py)
and the Live tab (live.py).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from safesite.types import Hazard

SAFE, NEEDS_ATTENTION, UNSAFE = "Safe", "Needs attention", "Unsafe"
SITE_STOP = "Stop work and report to your supervisor now."
SITE_REPORT = "Report to your supervisor."
HOME_STOP = "Fix this now before someone gets hurt."
HOME_REPORT = "Fix this soon."


@dataclass(frozen=True)
class Verdict:
    css: str  # good | warn | bad: card colour
    word: str  # Safe | Needs attention | Unsafe
    action: str  # what to do next; "" when safe


def verdict(hazards: Sequence[Hazard], mode: str = "site") -> Verdict:
    """Verdict for the hazards found in one photo or frame (mode: "site" or "home")."""
    home = str(mode).lower().startswith("home")
    if any(h.severity == "High" for h in hazards):
        return Verdict("bad", UNSAFE, HOME_STOP if home else SITE_STOP)
    if hazards:
        return Verdict("warn", NEEDS_ATTENTION, HOME_REPORT if home else SITE_REPORT)
    return Verdict("good", SAFE, "")
