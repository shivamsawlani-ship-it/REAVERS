"""
Tests for metrics audit, latency integrity, and UI state consistency.

Verifies:
1. Window-level vs frame-level metrics calculation.
2. Precision starts at 100.0% (1.0) on clean traffic with 0 false alarms.
3. Attack Type matches active injector attack key dynamically.
4. Banner headline and threat panel maintain single source of truth.
5. Fail-Safe protection ON maintains Vehicle Safety as SAFE (blocks hostile traffic).
6. Fail-Safe protection OFF forwards hostile traffic and trips Vehicle Safety to UNSAFE.
7. AI backend display badge formatting (IsolationForest / LSTM-AE (ONNX) / LSTM-AE (PyTorch)).
"""

import time
import pytest
from src.engine import CAVShieldEngine, MetricsTracker


def test_clean_traffic_precision_nominal():
    """Verify precision equals 1.0 (100%) when 0 false alarms occur on clean stream."""
    tracker = MetricsTracker()
    tracker.record_window("Normal", is_positive_decision=False)
    tracker.record_window("Normal", is_positive_decision=False)
    metrics = tracker.get_metrics_dict()
    assert metrics["precision"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 0.0


def test_window_metrics_separation_from_frames():
    """Verify window evaluations supersede frame evaluations without count pollution."""
    tracker = MetricsTracker()
    # 64 normal frames before window
    for _ in range(64):
        tracker.record_frame("Normal", is_positive_decision=False, detection_latency_ms=0.03)

    # 1 window evaluated
    tracker.record_window("Normal", is_positive_decision=False)
    m = tracker.get_metrics_dict()
    assert m["unit"] == "window"
    assert m["total_windows"] == 1
    assert m["total_frames"] == 64
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0


def test_live_attack_type_and_banner_dynamic_sync():
    """Verify threat.attack_type and banner sync with active attacks."""
    engine = CAVShieldEngine()
    engine.start()
    try:
        time.sleep(0.5)
        # 1. Clean state
        snap = engine.snapshot()
        assert snap["threat"]["attack_type"] == "Normal"
        assert snap["banner"]["headline"] == "System Secure"
        assert snap["banner"]["level"] == "LOW"

        # 2. Inject Fuzzing
        engine.inject("fuzz", duration=1.0)
        time.sleep(0.3)
        snap = engine.snapshot()
        assert snap["threat"]["attack_type"] == "Fuzzing"
        assert "Fuzzing" in snap["banner"]["headline"]
        engine.stop_attack("fuzz")

        # 3. Inject Spoof Speed
        engine.inject("spoof_speed", duration=1.0)
        time.sleep(0.3)
        snap = engine.snapshot()
        assert snap["threat"]["attack_type"] == "Spoofing (Speed)"
        assert "Spoofing (Speed)" in snap["banner"]["headline"]
        engine.stop_attack("spoof_speed")

        # 4. Inject V2X Phantom
        engine.inject("v2x_phantom", duration=1.0)
        time.sleep(0.3)
        snap = engine.snapshot()
        assert snap["threat"]["attack_type"] == "V2X Phantom Obstacle"
        assert "V2X Phantom Obstacle" in snap["banner"]["headline"]
        engine.stop_attack("v2x_phantom")

    finally:
        engine.stop()


def test_vehicle_safety_protection_on_vs_off():
    """Verify Vehicle Safety remains SAFE with protection ON and turns UNSAFE with protection OFF."""
    engine = CAVShieldEngine()
    engine.start()
    try:
        time.sleep(0.5)

        # Protection ON
        engine.failsafe.set_protection_enabled(True)
        engine.inject("spoof_brake", duration=1.0)
        time.sleep(0.4)
        snap_on = engine.snapshot()
        assert snap_on["twin"]["unsafe"] == "", f"Expected safe twin with protection ON, got {snap_on['twin']['unsafe']}"
        engine.stop_attack("spoof_brake")

        time.sleep(0.5)

        # Protection OFF
        engine.failsafe.set_protection_enabled(False)
        engine.inject("spoof_brake", duration=1.0)
        time.sleep(0.4)
        snap_off = engine.snapshot()
        assert snap_off["twin"]["unsafe"] == "UNINTENDED BRAKING", f"Expected UNINTENDED BRAKING with protection OFF, got {snap_off['twin']['unsafe']}"
        engine.stop_attack("spoof_brake")

    finally:
        engine.stop()


def test_backend_display_badge():
    """Verify AI backend badge reflects loaded backend cleanly."""
    engine = CAVShieldEngine()
    snap = engine.snapshot()
    ai_badge = snap["backend"]["ai"]
    assert ai_badge in ("IsolationForest", "LSTM-AE (ONNX)", "LSTM-AE (PyTorch)")


def test_vehicle_safety_all_attacks_unprotected():
    """Verify all hostile attacks forward to vehicle when protection is OFF and trip UNSAFE."""
    engine = CAVShieldEngine()
    engine.start()
    try:
        time.sleep(0.4)
        engine.failsafe.set_protection_enabled(False)

        # 1. Flooding attack (protocol flood)
        engine.inject("flood", duration=1.0)
        time.sleep(0.4)
        snap = engine.snapshot()
        assert snap["twin"]["unsafe"] != "", "Expected UNSAFE twin under unshielded flood"
        assert "FLOOD" in snap["twin"]["unsafe"]
        engine.stop_attack("flood")

        time.sleep(0.3)

        # 2. Spoof Gear attack
        engine.inject("spoof_gear", duration=1.0)
        time.sleep(0.4)
        snap = engine.snapshot()
        assert snap["twin"]["unsafe"] != "", "Expected UNSAFE twin under unshielded spoof gear"
        engine.stop_attack("spoof_gear")

        time.sleep(0.3)

        # 3. Protection re-enabled -> Vehicle blocks attack and remains SAFE
        engine.failsafe.set_protection_enabled(True)
        engine.inject("flood", duration=1.0)
        time.sleep(0.4)
        snap_safe = engine.snapshot()
        assert snap_safe["twin"]["unsafe"] == "", f"Expected SAFE twin with protection ON, got {snap_safe['twin']['unsafe']}"
        engine.stop_attack("flood")

    finally:
        engine.stop()


def test_realtime_scores_and_plotly_asset():
    """Verify real-time scores stream to dashboard and Plotly asset is served."""
    from pathlib import Path
    plotly_path = Path("src/dashboard/static/plotly.min.js")
    assert plotly_path.exists(), "plotly.min.js bundle must exist in static assets"
    assert plotly_path.stat().st_size > 1_000_000, "plotly.min.js should be the full bundle"

    engine = CAVShieldEngine()
    engine.start()
    try:
        time.sleep(0.8)
        snap = engine.snapshot()
        scores = snap["scores"]
        assert len(scores) > 0, "Expected non-empty real-time score history"
        last = scores[-1]
        assert "t" in last and "anomaly" in last and "attack_prob" in last
        assert 0.0 <= last["anomaly"] <= 1.0
        assert 0.0 <= last["attack_prob"] <= 1.0
    finally:
        engine.stop()


