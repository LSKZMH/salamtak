"""Load alerts.json and return short worker alerts by hazard id and language.

Shape: ``{hazard_id: {lang: {"text": str, "verified": bool}}}`` for
en, ar, hi, bn, ne, ur, ml. Keys starting with "_" are metadata and skipped.

Two layers:
- ``load_alerts()`` is strict: the app calls it at boot so a broken or incomplete file
  fails loudly before the demo, not during it.
- ``alert_for()`` is lenient at runtime: a missing or empty language falls back to English,
  a plain-string phrase counts as unverified, and a hazard with no phrase at all gets a
  flagged generic line (``missing=True``) instead of crashing the UI.
Only an explicit ``"verified": true`` marks a phrase as confirmed by a native speaker;
machine-drafted translations stay ``false``. Returned text is plain text: callers must
HTML-escape it before rendering.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from safesite.types import HAZARD_IDS, Hazard

ALERTS_PATH = Path(__file__).with_name("alerts.json")
DEFAULT_LANGUAGE = "en"
LANGUAGES: tuple[str, ...] = ("en", "ar", "hi", "bn", "ne", "ur", "ml")

AlertTable = Mapping[str, Mapping[str, "Alert"]]
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alert:
    hazard_id: str
    language: str  # language actually used (may be the English fallback)
    text: str
    verified: bool
    missing: bool = False  # True when alerts.json has no phrase for this hazard at all


# ---------- strict validation (boot) ----------
def _parse_phrase(hazard_id: str, language: str, raw: object) -> Alert | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
        raise ValueError(f"alerts.json: {hazard_id}.{language} needs a 'text' string")
    if not isinstance(raw.get("verified"), bool):
        raise ValueError(f"alerts.json: {hazard_id}.{language} needs a boolean 'verified'")
    text = raw["text"].strip()
    return Alert(hazard_id, language, text, raw["verified"]) if text else None


def _parse_entry(hazard_id: str, raw: object) -> Mapping[str, Alert]:
    if not isinstance(raw, dict):
        raise ValueError(f"alerts.json: {hazard_id} must map languages to phrases")
    unknown = set(raw) - set(LANGUAGES)
    if unknown:
        raise ValueError(f"alerts.json: {hazard_id} has unknown languages {sorted(unknown)}")
    phrases = {}
    for language, value in raw.items():
        alert = _parse_phrase(hazard_id, language, value)
        if alert is not None:
            phrases[language] = alert
    if DEFAULT_LANGUAGE not in phrases:
        raise ValueError(f"alerts.json: {hazard_id} must have an English phrase")
    return MappingProxyType(phrases)


def parse_alerts(raw: object) -> AlertTable:
    if not isinstance(raw, dict):
        raise ValueError("alerts.json must be an object keyed by hazard id")
    table = {hid: _parse_entry(hid, entry) for hid, entry in raw.items() if not hid.startswith("_")}
    return MappingProxyType(table)


@cache
def load_alerts(path: Path = ALERTS_PATH) -> AlertTable:
    """Read and validate alerts.json once; raises ValueError on a malformed file
    or one missing a hazard id, so a gap fails at startup rather than mid-demo."""
    table = parse_alerts(json.loads(path.read_text(encoding="utf-8")))
    missing = set(HAZARD_IDS) - set(table)
    if missing:
        raise ValueError(f"alerts.json is missing hazard ids {sorted(missing)}")
    return table


# ---------- lenient lookup (runtime) ----------
@cache
def _load() -> dict[str, Any]:
    """Raw alerts.json, read once. Tests may replace this with fake data."""
    raw = json.loads(ALERTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{ALERTS_PATH.name} must be a JSON object of hazard id -> languages")
    return raw


def _phrase(hazard_id: str, language: str, raw: object) -> Alert | None:
    if isinstance(raw, str):
        text, verified = raw, False
    elif isinstance(raw, dict) and isinstance(raw.get("text"), str):
        text, verified = raw["text"], raw.get("verified") is True
    else:
        return None
    text = text.strip()
    return Alert(hazard_id, language, text, verified) if text else None


def _from_raw(hazard_id: str, language: str) -> Alert | None:
    phrases = _load().get(hazard_id)
    if not isinstance(phrases, dict):
        return None
    for code in (language, DEFAULT_LANGUAGE):
        found = _phrase(hazard_id, code, phrases.get(code))
        if found is not None:
            return found
    return None


def _from_table(hazard_id: str, language: str, table: AlertTable) -> Alert | None:
    phrases = table.get(hazard_id)
    if phrases is None:
        return None
    return phrases.get(language) or phrases.get(DEFAULT_LANGUAGE)


def alert_for(
    hazard_id: str, language: str = DEFAULT_LANGUAGE, table: AlertTable | None = None
) -> Alert:
    """The phrase in ``language``, else English, else a flagged generic line.

    ``table`` (from ``parse_alerts``) overrides the shipped file, e.g. in tests."""
    found = (
        _from_raw(hazard_id, language) if table is None else _from_table(hazard_id, language, table)
    )
    if found is not None:
        return found
    log.warning("no alert phrase for hazard %r; using a generic one", hazard_id)
    generic = f"{hazard_id.replace('_', ' ').capitalize()}: no alert text yet."
    return Alert(hazard_id, DEFAULT_LANGUAGE, generic, verified=False, missing=True)


def alerts_for(
    hazards: Iterable[Hazard], language: str = DEFAULT_LANGUAGE, table: AlertTable | None = None
) -> list[Alert]:
    """One alert per distinct hazard id, in the order the hazards were found."""
    ids = list(dict.fromkeys(h.id for h in hazards))
    return [alert_for(hid, language, table) for hid in ids]
