"""Tests for alerts.py and the shipped alerts.json."""

from __future__ import annotations

import json

import pytest

from safesite import alerts
from safesite.alerts import LANGUAGES, alert_for, alerts_for, load_alerts, parse_alerts
from safesite.types import HAZARD_IDS, Hazard

pytestmark = pytest.mark.unit
ENGLISH_ONLY = {"en": {"text": "Wear it.", "verified": True}}


def test_every_hazard_id_has_every_language():
    table = load_alerts()
    assert set(table) == set(HAZARD_IDS)
    for hazard_id in HAZARD_IDS:
        assert set(table[hazard_id]) == set(LANGUAGES), hazard_id


def test_only_english_is_marked_verified():
    for hazard_id, phrases in load_alerts().items():
        for language, alert in phrases.items():
            assert alert.verified is (language == "en"), f"{hazard_id}.{language}"


def test_missing_language_falls_back_to_english():
    alert = alert_for("no_helmet", "fr")
    assert alert.language == "en"
    assert alert.text == "Wear your helmet now."


def test_empty_translation_falls_back_to_english(monkeypatch):
    fake = {"no_helmet": {**ENGLISH_ONLY, "ar": {"text": " ", "verified": False}}}
    monkeypatch.setattr(alerts, "_load", lambda: fake)
    assert alert_for("no_helmet", "ar").language == "en"


def test_plain_string_phrase_counts_as_unverified(monkeypatch):
    monkeypatch.setattr(
        alerts, "_load", lambda: {"no_helmet": {**ENGLISH_ONLY, "ar": "ارتدِ الخوذة"}}
    )
    alert = alert_for("no_helmet", "ar")
    assert alert.language == "ar"
    assert alert.verified is False


def test_returns_unverified_translation_when_present():
    alert = alert_for("wbgt_stop", "ur")
    assert alert.language == "ur"
    assert alert.verified is False
    assert alert.text


def test_unknown_hazard_gets_flagged_generic_phrase_instead_of_crashing():
    alert = alert_for("not_a_hazard", "hi")
    assert alert.missing is True
    assert alert.language == "en"
    assert alert.verified is False
    assert "no alert text yet" in alert.text


def test_malformed_entry_raises():
    with pytest.raises(ValueError, match="boolean 'verified'"):
        parse_alerts({"no_helmet": {"en": {"text": "Wear it."}}})


def test_unknown_language_key_raises():
    with pytest.raises(ValueError, match="unknown languages"):
        parse_alerts({"no_helmet": {**ENGLISH_ONLY, "urdu": {"text": "x", "verified": False}}})


def test_load_rejects_file_missing_a_hazard_id(tmp_path):
    partial = tmp_path / "alerts.json"
    partial.write_text(json.dumps({"no_helmet": ENGLISH_ONLY}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing hazard ids"):
        load_alerts(partial)


def test_alerts_for_gives_one_alert_per_hazard_id_in_order():
    hazards = [
        Hazard.of("no_helmet", "a", subject="person:0"),
        Hazard.of("trip_cord", "b"),
        Hazard.of("no_helmet", "c", subject="person:1"),
    ]
    assert [a.hazard_id for a in alerts_for(hazards, "ml")] == ["no_helmet", "trip_cord"]
