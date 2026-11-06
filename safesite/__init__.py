"""Salamtak سلامتك: hazard rules, Qatar heat rules, safety score and worker alerts.

Typical use from app.py::

    from safesite import Detection, SceneContext, alerts_for, evaluate, safety_score

    hazards = evaluate(detections, SceneContext(taken_at=when, wbgt_c=wbgt, outdoor=True))
    score = safety_score(hazards)
    phrases = alerts_for(hazards, language="ar")
"""

from safesite.alerts import Alert, alert_for, alerts_for
from safesite.rules import evaluate
from safesite.score import safety_score
from safesite.types import HAZARD_IDS, Detection, Hazard, Mode, SceneContext, Severity

__all__ = [
    "HAZARD_IDS",
    "Alert",
    "Detection",
    "Hazard",
    "Mode",
    "SceneContext",
    "Severity",
    "alert_for",
    "alerts_for",
    "evaluate",
    "safety_score",
]
