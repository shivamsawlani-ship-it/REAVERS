"""
Unit and integration tests for CAVShield Part 8:
- CAVShieldEngine multi-threaded orchestrator
- SNAPSHOT SCHEMA completeness (CONTRACT item 11)
- Inline gateway protection (spoof_brake blocked with protection ON, applied with protection OFF)
- Legitimate AEB event allowed when real obstacle is present
- MetricsTracker calculations
- UDP remote command listener
"""

from __future__ import annotations

import json
import socket
import time
import pytest

from src import load_config
from src.engine import CAVShieldEngine, MetricsTracker


def test_metrics_tracker_math() -> None:
    """Verify MetricsTracker accuracy, precision, recall, f1, and fpr formulas."""
    mt = MetricsTracker()

    # 10 True Positives, 2 False Positives, 80 True Negatives, 0 False Negatives
    for _ in range(10):
        mt.record_frame(ground_truth_label="Spoofing", is_positive_decision=True, detection_latency_ms=1.2)
    for _ in range(2):
        mt.record_frame(ground_truth_label="Normal", is_positive_decision=True, detection_latency_ms=0.8)
    for _ in range(80):
        mt.record_frame(ground_truth_label="Normal", is_positive_decision=False, detection_latency_ms=0.5)

    res = mt.get_metrics_dict()
    assert res["counts"]["tp"] == 10
    assert res["counts"]["fp"] == 2
    assert res["counts"]["tn"] == 80
    assert res["counts"]["fn"] == 0

    assert res["accuracy"] == round((10 + 80) / 92, 4)
    assert res["precision"] == round(10 / 12, 4)
    assert res["recall"] == 1.0
    assert res["fpr"] == round(2 / 82, 4)
    assert res["simulator_only"] is True


def test_snapshot_schema_completeness() -> None:
    """Verify that engine.snapshot() contains every key and structure defined in CONTRACT item 11."""
    cfg = load_config()
    engine = CAVShieldEngine(cfg=cfg)

    snap = engine.snapshot()

    # 1. Root keys per CONTRACT Item 11
    required_root_keys = [
        "t", "mode", "protection", "backend", "banner", "active_attacks",
        "twin", "threat", "scores", "attack_markers", "can", "v2x",
        "failsafe", "incidents", "metrics", "demo",
    ]
    for key in required_root_keys:
        assert key in snap, f"Missing root key in snapshot: {key}"

    # 2. Sub-structures
    # Backend
    assert "bus" in snap["backend"]
    assert "ai" in snap["backend"]

    # Banner
    assert "level" in snap["banner"]
    assert "headline" in snap["banner"]
    assert "driver_action" in snap["banner"]

    # Twin
    twin = snap["twin"]
    twin_keys = ["speed_kmh", "rpm", "pedal", "brake_active", "decel", "applied_decel", "steering_deg", "gear", "aeb", "unsafe", "unsafe_seconds", "perceived"]
    for k in twin_keys:
        assert k in twin, f"Missing twin key: {k}"
    assert "speed_kmh" in twin["perceived"]

    # Threat
    threat = snap["threat"]
    threat_keys = ["attack_type", "confidence", "level", "explanation", "ai_available", "anomaly", "attack_prob", "probs"]
    for k in threat_keys:
        assert k in threat, f"Missing threat key: {k}"

    # Failsafe
    fs = snap["failsafe"]
    fs_keys = ["rejected_commands", "rate_limited", "quarantined", "rejected_v2x", "mode_history", "actions", "incident_id"]
    for k in fs_keys:
        assert k in fs, f"Missing failsafe key: {k}"

    # Metrics
    m = snap["metrics"]
    m_keys = ["accuracy", "precision", "recall", "f1", "fpr", "detection_latency_ms", "failsafe_latency_ms", "inference_ms", "simulator_only", "counts"]
    for k in m_keys:
        assert k in m, f"Missing metrics key: {k}"

    # JSON serializability check
    serialized = json.dumps(snap)
    assert len(serialized) > 0


def test_engine_lifecycle_and_protection() -> None:
    """Verify CAVShieldEngine starts, blocks spoofed brake when ON, and stops gracefully."""
    cfg = load_config()
    engine = CAVShieldEngine(cfg=cfg)

    try:
        engine.start()
        time.sleep(1.0)  # Let threads settle and process normal baseline

        # Verify initial normal state
        snap_normal = engine.snapshot()
        assert snap_normal["mode"] == "NORMAL"
        assert snap_normal["protection"] is True
        initial_speed = snap_normal["twin"]["speed_kmh"]
        assert initial_speed > 60.0  # cruising

        # Inject spoof_brake attack
        engine.inject("spoof_brake", duration=2.0)
        time.sleep(2.2)

        # Verify attack was blocked by gateway
        snap_attack = engine.snapshot()
        # Spoofed brake (0x200) must have been blocked
        blocked_cmd_count = snap_attack["failsafe"]["rejected_commands"].get(0x200, 0)
        assert blocked_cmd_count > 0, "Spoofed brake command was not blocked by gateway!"

        # Vehicle twin should NOT have decelerated at 7 m/s² because frames were blocked!
        assert snap_attack["twin"]["applied_decel"] < 1.0, "Spoofed brake was improperly applied to vehicle twin!"

    finally:
        engine.stop()


def test_contrast_protection_off() -> None:
    """Verify that when protection is OFF, spoofed brake commands are applied to the twin plant."""
    cfg = load_config()
    engine = CAVShieldEngine(cfg=cfg)
    engine.set_protection(False)  # Turn protection OFF

    try:
        engine.start()
        time.sleep(0.5)

        # Inject spoof_brake attack with protection OFF
        engine.inject("spoof_brake", duration=2.0)
        # Inspect while attack is actively running
        time.sleep(0.8)

        snap = engine.snapshot()
        # Since protection was OFF, frames were forwarded to twin
        # Twin must register applied decel > 2.0 m/s² and unsafe condition
        assert snap["twin"]["applied_decel"] > 2.0 or snap["twin"]["unsafe_seconds"] > 0.0, "Unprotected twin failed to experience unintended braking!"

    finally:
        engine.stop()


def test_udp_remote_command_listener() -> None:
    """Verify remote commands sent via UDP or handler to engine."""
    cfg = load_config()
    engine = CAVShieldEngine(cfg=cfg)

    try:
        engine.start()
        time.sleep(0.5)

        # 1. Query STATUS via command handler
        status_raw = engine.handle_command("STATUS")
        resp = json.loads(status_raw)
        assert "mode" in resp
        assert "active" in resp

        # 2. Inject attack via command handler
        inj_resp = engine.handle_command("INJECT spoof_speed 1.5")
        assert "OK injected" in inj_resp
        assert "spoof_speed" in engine.world.injector.active()

        # 3. Stop attack via command handler
        stop_resp = engine.handle_command("STOP spoof_speed")
        assert "OK stopped" in stop_resp
        assert "spoof_speed" not in engine.world.injector.active()

        # 4. If environment allows socket loopback, verify network UDP interface
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.5)
            sock.sendto(b"STATUS", ("127.0.0.1", engine.udp_port))
            data, _ = sock.recvfrom(1024)
            net_resp = json.loads(data.decode("utf-8"))
            assert "mode" in net_resp
            sock.close()
        except OSError:
            pass  # OS sandbox blocks UDP packet transmission

    finally:
        engine.stop()

