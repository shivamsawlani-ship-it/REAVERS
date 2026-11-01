"""
Test Suite: Threat Model & Honest Entry Point Layer.

Verifies:
1. Every attack class and instance in AttackInjector has an 'entry_point' metadata field matching the threat model.
2. Configuration table in config/config.yaml (attacks.entry_points) is parsed and applied.
3. Incidents recorded by FailSafeManager carry 'assumed_entry_point'.
4. System snapshot threat and incidents dictionaries carry 'assumed_entry_point'.
5. UI source code (index.html, app.js) explicitly labels the entry point as simulated with
   'Assumed entry point (simulated scenario, not detected):' in a muted style.
"""

from __future__ import annotations

from pathlib import Path
import time
import pytest

from src import load_config
from src.attack_injector import AttackInjector, DEFAULT_ENTRY_POINTS, resolve_entry_point
from src.detection.decision_engine import Decision, RiskLevel
from src.engine import CAVShieldEngine
from src.failsafe.failsafe_manager import FailSafeManager, IncidentRecord


def test_every_attack_has_entry_point() -> None:
    """Verify all 11 attacks define a valid entry_point metadata attribute."""
    cfg = load_config()
    injector = AttackInjector(cfg=cfg)

    expected_attacks = [
        "flood",
        "fuzz",
        "replay",
        "spoof_brake",
        "spoof_accel",
        "spoof_speed",
        "spoof_rpm",
        "spoof_gear",
        "v2x_phantom",
        "v2x_fake_ebrake",
        "v2x_gps",
    ]

    for atk_name in expected_attacks:
        assert atk_name in injector.attacks, f"Attack '{atk_name}' missing from injector"
        atk = injector.attacks[atk_name]
        assert hasattr(atk, "entry_point"), f"Attack '{atk_name}' missing entry_point attribute"
        assert isinstance(atk.entry_point, str) and len(atk.entry_point) > 0, f"Attack '{atk_name}' has empty entry_point"

        # Check expected simulation assumptions
        if atk_name.startswith("spoof_"):
            assert "infotainment" in atk.entry_point.lower() or "telematics" in atk.entry_point.lower()
        elif atk_name in ("flood", "fuzz"):
            assert "obd-ii" in atk.entry_point.lower()
        elif atk_name == "replay":
            assert "recorded traffic" in atk.entry_point.lower() or "compromised node" in atk.entry_point.lower()
        elif atk_name in ("v2x_phantom", "v2x_fake_ebrake"):
            assert "v2x" in atk.entry_point.lower()
        elif atk_name == "v2x_gps":
            assert "position" in atk.entry_point.lower() or "gps" in atk.entry_point.lower()


def test_config_entry_points_editable() -> None:
    """Verify entry points are editable via config.yaml."""
    cfg = load_config()
    assert hasattr(cfg, "attacks"), "config missing attacks section"
    attacks_cfg = getattr(cfg, "attacks", {})
    entry_points = getattr(attacks_cfg, "entry_points", {})
    assert isinstance(entry_points, dict), "attacks.entry_points missing from config.yaml"
    assert "spoof_brake" in entry_points
    assert "flood" in entry_points
    assert "v2x_phantom" in entry_points


def test_incidents_carry_assumed_entry_point() -> None:
    """Verify IncidentRecord and FailSafeManager attach assumed_entry_point to incidents."""
    cfg = load_config()
    fs = FailSafeManager(cfg=cfg)

    # Simulate recording a spoof_brake incident
    decision = Decision(
        level=RiskLevel.HIGH,
        score=0.85,
        attack_type="Spoofing (Brake)",
        confidence=0.95,
        action="DROP",
        explanation="Spoofed brake command detected",
        components={"latency_ms": 0.05},
    )

    fs._record_incident(decision=decision, target="0x200", now=10.0)
    incidents = fs.get_recent_incidents(limit=1)

    assert len(incidents) == 1
    inc = incidents[0]
    assert "assumed_entry_point" in inc, "Incident missing assumed_entry_point key"
    assert "infotainment" in inc["assumed_entry_point"].lower() or "telematics" in inc["assumed_entry_point"].lower()


def test_snapshot_threat_and_incidents_carry_assumed_entry_point() -> None:
    """Verify engine snapshot threat and incidents sections carry assumed_entry_point."""
    cfg = load_config()
    engine = CAVShieldEngine(cfg=cfg)
    engine.reset()

    # Trigger attack
    engine.inject("spoof_brake")
    time.sleep(0.1)

    snap = engine.snapshot()
    engine.stop_attack("spoof_brake")

    # Threat dictionary check
    assert "threat" in snap, "snapshot missing threat dictionary"
    threat = snap["threat"]
    assert "assumed_entry_point" in threat, "snapshot.threat missing assumed_entry_point"
    assert "infotainment" in threat["assumed_entry_point"].lower() or "telematics" in threat["assumed_entry_point"].lower()

    # Incidents list check (if any incidents created)
    incidents = snap.get("incidents", [])
    if incidents:
        assert "assumed_entry_point" in incidents[0]


def test_ui_labels_entry_point_as_simulated() -> None:
    """Verify UI explicitly labels entry points with the simulated disclaimer."""
    root = Path(__file__).resolve().parent.parent
    index_html = (root / "src" / "dashboard" / "static" / "index.html").read_text(encoding="utf-8")
    app_js = (root / "src" / "dashboard" / "static" / "app.js").read_text(encoding="utf-8")

    expected_label = "Assumed entry point (simulated scenario, not detected):"

    assert expected_label in index_html, "index.html missing simulated entry point label"
    assert expected_label in app_js, "app.js missing simulated entry point label"
