"""Alert phrases by hazard id + language, loaded from alerts.json. Falls back to English.

STUB written by the app chat (Chat 4). Chat 3 owns this file and alerts.json.
Entries may be "text" or {"text": ..., "verified": bool}. Only an explicit "verified": true
marks a phrase as confirmed by a native speaker; everything else is shown as unverified.
Returned text is plain text: callers must HTML-escape it before rendering.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

ALERTS_PATH = Path(__file__).with_name("alerts.json")
FALLBACK_LANG = "en"
log = logging.getLogger(__name__)

# code -> display name (native script first so workers recognise it)
LANGUAGES: dict[str, str] = {
    "en": "English",
    "ar": "العربية (Arabic)",
    "hi": "हिन्दी (Hindi)",
    "bn": "বাংলা (Bengali)",
    "ne": "नेपाली (Nepali)",
    "ur": "اردو (Urdu)",
    "ml": "മലയാളം (Malayalam)",
}
RTL_LANGUAGES = frozenset({"ar", "ur"})


@dataclass(frozen=True)
class Alert:
    hazard_id: str
    text: str
    lang: str  # language actually used (may be the fallback)
    verified: bool
    missing: bool = False  # True when no phrase exists at all and a generic one was built


@cache
def _load() -> dict[str, dict[str, Any]]:
    data = json.loads(ALERTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{ALERTS_PATH.name} must be a JSON object of hazard id -> languages")
    return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)}


def validate() -> int:
    """Load alerts.json once at startup so a broken file fails loudly. Returns the hazard count."""
    return len(_load())


def _entry(raw: object) -> tuple[str, bool] | None:
    if isinstance(raw, str) and raw.strip():
        return raw, False
    if isinstance(raw, dict) and isinstance(raw.get("text"), str) and raw["text"].strip():
        return raw["text"], raw.get("verified") is True
    return None


def alert_for(hazard_id: str, lang: str) -> Alert:
    phrases = _load().get(hazard_id, {})
    for code in (lang, FALLBACK_LANG):
        found = _entry(phrases.get(code))
        if found:
            return Alert(hazard_id, found[0], code, found[1])
    log.warning("no alert phrase for hazard %r; using a generic one", hazard_id)
    return Alert(hazard_id, hazard_id.replace("_", " ").capitalize() + ".", FALLBACK_LANG, False, missing=True)
