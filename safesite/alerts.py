"""Load alerts.json and return short worker alerts by hazard id and language.

Shape: ``{hazard_id: {lang: {"text": str, "verified": bool}}}`` for
en, ar, hi, bn, ne, ur, ml. Keys starting with "_" are metadata and skipped.
Unverified (machine-drafted) translations are returned with ``verified=False``
so the UI can mark them; a missing or empty language falls back to English.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

from safesite.types import HAZARD_IDS, Hazard

ALERTS_PATH = Path(__file__).with_name("alerts.json")
DEFAULT_LANGUAGE = "en"
LANGUAGES: tuple[str, ...] = ("en", "ar", "hi", "bn", "ne", "ur", "ml")

AlertTable = Mapping[str, Mapping[str, "Alert"]]


@dataclass(frozen=True)
class Alert:
    hazard_id: str
    language: str
    text: str
    verified: bool


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


def alert_for(
    hazard_id: str, language: str = DEFAULT_LANGUAGE, table: AlertTable | None = None
) -> Alert:
    """The phrase in ``language``, or the English one if that language is missing."""
    phrases = (table if table is not None else load_alerts()).get(hazard_id)
    if phrases is None:
        raise KeyError(f"no alert phrases for hazard id {hazard_id!r}")
    return phrases.get(language) or phrases[DEFAULT_LANGUAGE]


def alerts_for(
    hazards: Iterable[Hazard], language: str = DEFAULT_LANGUAGE, table: AlertTable | None = None
) -> list[Alert]:
    """One alert per distinct hazard id, in the order the hazards were found."""
    ids = list(dict.fromkeys(h.id for h in hazards))
    return [alert_for(hid, language, table) for hid in ids]
