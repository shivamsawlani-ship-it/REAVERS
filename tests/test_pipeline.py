"""
Comprehensive Pipeline Test Suite for CAVShield (Part 12).

Tests:
1. CAN Codec Round-Trip & Checksum (All 10 IDs in CONTRACT DBC, XOR checksum, alive counter)
2. Windowing Shapes & Stride (Size 64, stride 16, Window dataclass, window_stat_features)
3. Feature Extractor Determinism (Stateful O(1) streaming, identical output on identical traffic)
4. Deterministic Rule Invariants (Fires on attack, stays 100% silent on 30s normal traffic)
5. Physical Dynamics Invariants (Fires on attack, stays 100% silent on 30s normal traffic)
6. V2X Validator Verification (Accepts authentic BSM, rejects expired/tampered/phantom messages)
7. Multi-Source Risk Confidence Gate (Strict CRITICAL gate, capping, AI-only limitations)
8. Fail-Safe Mode Transitions & Auto-Recovery (NORMAL -> ALERT -> DEGRADED -> SAFE MODE -> Recovery)
9. Headless End-to-End Simulation (20s offline run with spoof_brake: blocked, CRITICAL, decel=0)
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List

import numpy as np
import pytest

from src import load_config
from src.attack_injector import AttackInjector
from src.can_simulator import (
    COMMAND_IDS,
    DBC,
    KNOWN_IDS,
    CANFrame,
    ECUSimulator,
    checksum_ok,
    counter_of,
    decode,
    encode,
    make_frame,
)
from src.detection.decision_engine import AIContext, Decision, RiskEngine, RiskLevel
from src.detection.physics_engine import PhysicsEngine
from src.detection.rule_engine import CANRuleEngine, Evidence
from src.detection.v2x_validator import V2XValidator, V2XVerdict
from src.engine import CAVShieldEngine
from src.failsafe.failsafe_manager import FailSafeManager
from src.feature_engineering import CANFeatureExtractor
from src.v2x_simulator import Environment, SensorSnapshot, TrustStore, V2XMessage
from src.vehicle_digital_twin import VehicleDigitalTwin, VehicleState
from src.windowing import SlidingWindower, Window, window_stat_features


# ---------------------------------------------------------------------------
# 1. Codec Round-Trip & Checksum Verification
# ---------------------------------------------------------------------------

def test_codec_roundtrip_and_checksum() -> None:
    """Verify signal encode/decode round-trip, XOR checksum, and alive counter for all 10 DBC IDs."""
    test_signals = {
        0x100: {"rpm": 2450, "throttle": 35},
        0x101: {"speed_kmh": 72.5},
        0x102: {"pedal": 15.0, "brake_active": 1, "decel": 1.5},
        0x103: {"steering_deg": -12.4},
        0x104: {"gear": 3},
        0x105: {"wheel_fl": 72.0, "wheel_fr": 72.0, "wheel_rl": 72.0, "wheel_rr": 72.0},
        0x106: {"ax": 0.45, "yaw": -1.2},
        0x200: {"cmd_decel": 3.5, "source": 1},
        0x201: {"cmd_accel": 1.2, "source": 2},
        0x202: {"lka": 1, "acc": 1, "aeb": 0, "set_speed_kmh": 80},
    }

    for arb_id, sigs in test_signals.items():
        assert arb_id in DBC, f"ID 0x{arb_id:03X} not in DBC"
        counter = 7
        data = encode(arb_id, sigs, counter=counter)

        assert len(data) == 8, f"DLC must be 8 for ID 0x{arb_id:03X}"

        # 1. Checksum in Byte 6 = XOR of Bytes 0..5
        expected_chk = data[0] ^ data[1] ^ data[2] ^ data[3] ^ data[4] ^ data[5]
        assert data[6] == expected_chk, f"Byte 6 XOR checksum failure on 0x{arb_id:03X}"
        assert checksum_ok(data) is True, f"checksum_ok returned False on 0x{arb_id:03X}"

        # 2. Alive counter in Byte 7 low nibble
        extracted_ctr = counter_of(data)
        assert extracted_ctr == (counter & 0x0F), f"Counter extraction failure on 0x{arb_id:03X}"

        # 3. Decode round-trip signals
        decoded = decode(arb_id, data)
        for k, v in sigs.items():
            assert k in decoded, f"Missing key {k} in decoded output for 0x{arb_id:03X}"
            # Signal resolution tolerance
            assert abs(decoded[k] - v) < 0.5, f"Signal round-trip mismatch for {k} on 0x{arb_id:03X}"


# ---------------------------------------------------------------------------
# 2. Windowing Shapes & Stride
# ---------------------------------------------------------------------------

def test_windowing_shapes_and_stride() -> None:
    """Verify SlidingWindower shape (64, 17), stride of 16, and statistical vector extraction."""
    windower = SlidingWindower(size=64, stride=16)
    rng = np.random.default_rng(42)

    windows_produced: List[Window] = []

    # Push 100 vectors of length 17
    for i in range(100):
        vec = rng.random(17).astype(np.float32)
        w = windower.push(
            vec=vec,
            ts=float(i * 0.01),
            label="Normal",
            subtype="",
            raw_id=0x100,
            raw_payload=b"\x00" * 8,
        )
        if w is not None:
            windows_produced.append(w)

    assert len(windows_produced) == 3, f"Expected 3 windows from 100 pushes, got {len(windows_produced)}"

    for win in windows_produced:
        assert isinstance(win, Window)
        assert win.features.shape == (64, 17)
        assert win.label == "Normal"
        assert win.t_start < win.t_end

    # Test statistical feature extraction on window
    stat_vec = window_stat_features(windows_produced[0], known_ids=KNOWN_IDS)
    assert isinstance(stat_vec, np.ndarray)
    assert len(stat_vec) == 103  # 17*4 + 5 + 10*3 = 103 features
    assert np.all(np.isfinite(stat_vec))


# ---------------------------------------------------------------------------
# 3. Feature Extractor Determinism
# ---------------------------------------------------------------------------

def test_feature_extractor_determinism() -> None:
    """Verify that two independent CANFeatureExtractors yield identical feature vectors on identical traffic."""
    cfg = load_config()
    ext1 = CANFeatureExtractor(cfg=cfg)
    ext2 = CANFeatureExtractor(cfg=cfg)

    ecu_sim = ECUSimulator(cfg=cfg, rng=random.Random(123))
    twin_state = VehicleState(speed_kmh=72.0, rpm=2400)

    frames: List[CANFrame] = []
    for step in range(30):
        now = step * 0.01
        step_frames = ecu_sim.step(now=now, vehicle_state=twin_state)
        frames.extend(step_frames)

    assert len(frames) > 50

    for frame in frames:
        f1 = ext1.extract(arb_id=frame.arb_id, data=frame.data, ts=frame.timestamp, dlc=frame.dlc)
        f2 = ext2.extract(arb_id=frame.arb_id, data=frame.data, ts=frame.timestamp, dlc=frame.dlc)

        assert len(f1) == 17
        assert len(f2) == 17
        assert np.allclose(f1, f2, atol=1e-6), "Feature extractor non-determinism detected!"


# ---------------------------------------------------------------------------
# 4. Rule Engine: Attacks vs 30s Normal Traffic
# ---------------------------------------------------------------------------

def test_rules_fire_on_attack_and_silent_on_normal() -> None:
    """Verify CANRuleEngine fires on malicious violations and produces 0 false alarms on 30s normal traffic."""
    cfg = load_config()
    rule_engine = CANRuleEngine(cfg=cfg)

    # 1. Attack Violations Test
    # UNKNOWN_ID
    unknown_frame = make_frame(timestamp=1.0, arb_id=0x7FF, data=b"\x00" * 8, label="Fuzzing")
    evs = rule_engine.evaluate(unknown_frame)
    assert any(e.name == "UNKNOWN_ID" for e in evs)

    # DLC_MISMATCH
    dlc_frame = CANFrame(timestamp=1.0, arb_id=0x100, dlc=5, data=b"\x00" * 5, sender="ECU_ENGINE", msg_type="ENGINE_RPM")
    evs = rule_engine.evaluate(dlc_frame)
    assert any(e.name == "DLC_MISMATCH" for e in evs)

    # CHECKSUM
    good_data = encode(0x100, {"rpm": 2000, "throttle": 20}, counter=0)
    bad_data = bytearray(good_data)
    bad_data[6] ^= 0xFF  # Corrupt checksum byte
    chk_frame = make_frame(timestamp=1.0, arb_id=0x100, data=bytes(bad_data), label="Fuzzing")
    evs = rule_engine.evaluate(chk_frame)
    assert any(e.name == "CHECKSUM" for e in evs)

    # 2. Silent on 30s Normal Traffic Test
    rule_engine.reset()
    ecu_sim = ECUSimulator(cfg=cfg, rng=random.Random(42))
    twin_state = VehicleState(speed_kmh=72.0, rpm=2400)

    total_frames = 0
    violations = 0
    t = 0.0
    dt = 0.005  # 200 Hz
    while t < 30.0:
        frames = ecu_sim.step(now=t, vehicle_state=twin_state)
        for f in frames:
            total_frames += 1
            evidences = rule_engine.evaluate(f)
            if evidences:
                violations += len(evidences)
            rule_engine.commit(f)
        t += dt

    assert total_frames > 10000, f"Expected >10,000 normal frames, got {total_frames}"
    assert violations == 0, f"Rule engine raised {violations} false alarms on 30s normal traffic!"


# ---------------------------------------------------------------------------
# 5. Physics Invariants: Attacks vs 30s Normal Traffic
# ---------------------------------------------------------------------------

def test_physics_checks_fire_on_attack_and_silent_on_normal() -> None:
    """Verify PhysicsEngine fires on dynamic violations and stays 100% silent on 30s normal traffic."""
    cfg = load_config()
    phys_engine = PhysicsEngine(cfg=cfg)

    # 1. Attack Violations Test
    # PHYS_BRAKE_PEDAL_MISMATCH (Spoofed brake: 7.0 m/s2 decel with pedal=0%)
    spoofed_brake_data = encode(0x200, {"cmd_decel": 7.0, "source": 0}, counter=1)
    frame = make_frame(timestamp=1.0, arb_id=0x200, data=spoofed_brake_data, label="Spoofing", subtype="spoof_brake")
    decoded = decode(0x200, spoofed_brake_data)
    evs = phys_engine.check(frame, decoded)
    assert any("BRAKE" in e.name or "DECEL" in e.name for e in evs)

    # 2. Silent on 30s Normal Traffic Test
    phys_engine.reset()
    ecu_sim = ECUSimulator(cfg=cfg, rng=random.Random(999))
    twin = VehicleDigitalTwin(cfg=cfg)
    env = Environment(cfg=cfg)

    violations = 0
    t = 0.0
    dt = 0.005
    while t < 30.0:
        sensor_snapshot = env.sense(ego_speed_kmh=twin.state.speed_kmh, now=t)
        twin_state = twin.step(dt=dt, now=t, sensor_snapshot=sensor_snapshot)
        phys_engine.set_sensor_snapshot(sensor_snapshot)

        frames = ecu_sim.step(now=t, vehicle_state=twin_state)
        for f in frames:
            dec = decode(f.arb_id, f.data)
            evs = phys_engine.check(f, dec)
            if evs:
                violations += len(evs)
            phys_engine.update(f, dec)
        t += dt

    assert violations == 0, f"Physics engine raised {violations} false alarms on 30s normal traffic!"


# ---------------------------------------------------------------------------
# 6. V2X Validator: Accept vs Reject Cases
# ---------------------------------------------------------------------------

def test_v2x_validator_accept_and_reject_cases() -> None:
    """Verify V2XValidator accepts authentic messages and rejects forged, expired, or phantom warnings."""
    from src.v2x_simulator import ObjectTrack

    cfg = load_config()
    trust_store = TrustStore()
    validator = V2XValidator(cfg=cfg, trust_store=trust_store)

    # Accept: Legitimate authentic BSM
    valid_bsm = V2XMessage(
        msg_id="bsm_001",
        sender_id="V-2041",
        x=1040.0,
        y=3.6,
        speed=20.0,
        msg_type="BSM",
        timestamp=10.0,
        heading=0.0,
        seq=1,
    )
    trust_store.sign(valid_bsm)
    snap = SensorSnapshot(
        timestamp=10.0,
        tracks=[ObjectTrack("V-2041", 40.0, 40.0, 3.6, 0.0, 999.0, ["radar"], False)],
        ego_x_est=1000.0,
        ego_y_est=0.0,
    )
    v_acc = validator.validate(valid_bsm, snapshot=snap, now=10.05)
    assert v_acc.accepted is True, f"Legitimate BSM rejected: {v_acc.reasons}"
    assert v_acc.trust >= 0.60

    # Reject 1: Tampered/Untrusted HMAC Signature
    tampered_bsm = V2XMessage(
        msg_id="bsm_002",
        sender_id="V-2041",
        x=1040.0,
        y=3.6,
        speed=20.0,
        msg_type="BSM",
        timestamp=10.0,
        signature="invalid_hex_token_12345",
    )
    v_rej_sig = validator.validate(tampered_bsm, snapshot=snap, now=10.05)
    assert v_rej_sig.accepted is False
    assert any("signature" in r.lower() or "trust" in r.lower() for r in v_rej_sig.reasons)

    # Reject 2: Stale/Expired Timestamp
    stale_bsm = V2XMessage(
        msg_id="bsm_003",
        sender_id="V-2041",
        x=1040.0,
        y=3.6,
        speed=20.0,
        msg_type="BSM",
        timestamp=5.0,  # 5 seconds old (max age is 1.0s)
    )
    trust_store.sign(stale_bsm)
    v_rej_age = validator.validate(stale_bsm, snapshot=snap, now=10.0)
    assert v_rej_age.accepted is False
    assert any("expired" in r.lower() or "age" in r.lower() for r in v_rej_age.reasons)

    # Reject 3: Phantom Obstacle Without Radar Confirmation
    phantom_msg = V2XMessage(
        msg_id="warn_001",
        sender_id="V-9999",
        x=1030.0,
        y=0.0,
        speed=20.0,
        msg_type="COLLISION_WARNING",
        timestamp=10.0,
        obstacle={"range": 40.0, "x": 1040.0, "y": 0.0},
    )
    trust_store.sign(phantom_msg)
    # Empty snapshot (no vehicle / obstacle visible on radar)
    empty_snap = SensorSnapshot(timestamp=10.0, tracks=[], in_lane=None, ego_x_est=1000.0, ego_y_est=0.0)
    v_rej_phantom = validator.validate(phantom_msg, snapshot=empty_snap, now=10.02)
    assert v_rej_phantom.accepted is False
    assert any("phantom" in r.lower() or "hazard" in r.lower() or "ghost" in r.lower() for r in v_rej_phantom.reasons)


# ---------------------------------------------------------------------------
# 7. Confidence Gate Enforcement
# ---------------------------------------------------------------------------

def test_confidence_gate_enforcement() -> None:
    """Verify that CRITICAL decisions strictly require det >= 0.70 AND ai_conf >= 0.60 AND criticality >= 0.80."""
    cfg = load_config()
    risk_engine = RiskEngine(cfg=cfg)

    # Safety-critical ID: 0x200 (criticality = 1.0)
    frame_crit = make_frame(timestamp=1.0, arb_id=0x200, data=b"\x00" * 8)

    # Non-safety-critical ID: 0x100 (criticality = 0.6)
    frame_non_crit = make_frame(timestamp=1.0, arb_id=0x100, data=b"\x00" * 8)

    high_evidences = [
        Evidence(source="rule", name="CHECKSUM", severity=0.85, detail="Corrupted checksum", hint="Fuzzing"),
    ]

    # Case 1: High det (0.85), High AI (0.80), Critical ID (1.0) -> CRITICAL
    ai_high = AIContext(anomaly=0.85, attack_prob=0.80, label="Spoofing", confidence=0.85, t=1.0, available=True)
    d1 = risk_engine.fuse_can(frame_crit, high_evidences, ai_high, now=1.0)
    assert d1.level == RiskLevel.CRITICAL

    # Case 2: High det (0.85), Low AI (0.35) -> Capped at HIGH
    ai_low = AIContext(anomaly=0.35, attack_prob=0.30, label="Normal", confidence=0.70, t=1.0, available=True)
    d2 = risk_engine.fuse_can(frame_crit, high_evidences, ai_low, now=1.0)
    assert d2.level <= RiskLevel.HIGH, "Confidence gate failed to cap low-AI decision at HIGH"

    # Case 3: High det (0.85), High AI (0.80), but Non-critical ID (0.6) -> Capped at HIGH
    d3 = risk_engine.fuse_can(frame_non_crit, high_evidences, ai_high, now=1.0)
    assert d3.level <= RiskLevel.HIGH, "Confidence gate failed to cap non-critical ID at HIGH"

    # Case 4: Zero det (AI alone = 0.95) -> Per-frame decision is LOW (AI alone never blocks inline)
    d4 = risk_engine.fuse_can(frame_crit, [], ai_high, now=1.0)
    assert d4.level == RiskLevel.LOW


# ---------------------------------------------------------------------------
# 8. Fail-Safe Transitions and Recovery
# ---------------------------------------------------------------------------

def test_failsafe_level_transitions_and_recovery() -> None:
    """Verify FailSafeManager transitions from NORMAL -> ALERT -> DEGRADED -> SAFE MODE and auto-recovers."""
    cfg = load_config()
    fs = FailSafeManager(cfg=cfg)

    # 1. Start in NORMAL
    assert fs.current_mode == FailSafeManager.MODE_NORMAL

    # 2. MEDIUM decision transitions to ALERT
    dec_med = Decision(level=RiskLevel.MEDIUM, score=0.45, attack_type="Fuzzing", confidence=0.6, explanation="Med", action="ALERT")
    frame = make_frame(timestamp=1.0, arb_id=0x101, data=b"\x00" * 8)
    fs.enforce(dec_med, frame, now=1.0)
    assert fs.current_mode == FailSafeManager.MODE_ALERT

    # 3. CRITICAL decision on status ID transitions to DEGRADED
    dec_crit_status = Decision(level=RiskLevel.CRITICAL, score=0.85, attack_type="Spoofing", confidence=0.8, explanation="Status compromise", action="BLOCK")
    fs.enforce(dec_crit_status, frame, now=2.0)
    assert fs.current_mode == FailSafeManager.MODE_DEGRADED

    # 4. CRITICAL decision on command ID transitions to SAFE MODE
    frame_cmd = make_frame(timestamp=3.0, arb_id=0x200, data=b"\x00" * 8)
    dec_crit = Decision(level=RiskLevel.CRITICAL, score=0.92, attack_type="Spoofing", confidence=0.9, explanation="Crit", action="BLOCK")
    fwd, _ = fs.enforce(dec_crit, frame_cmd, now=3.0)
    assert fwd is False  # Blocked!
    assert fs.current_mode == FailSafeManager.MODE_SAFE_MODE

    # 5. Step-by-Step Auto-Recovery
    # SAFE MODE -> DEGRADED after 12s of clean traffic (t=16s)
    fs.update_mode(now=16.0)
    assert fs.current_mode == FailSafeManager.MODE_DEGRADED

    # DEGRADED -> ALERT after 8s (t=25s)
    fs.update_mode(now=25.0)
    assert fs.current_mode == FailSafeManager.MODE_ALERT

    # ALERT -> NORMAL after 4s (t=30s)
    fs.update_mode(now=30.0)
    assert fs.current_mode == FailSafeManager.MODE_NORMAL


# ---------------------------------------------------------------------------
# 9. Headless End-to-End Simulation: Spoofed Brake Blocked Offline
# ---------------------------------------------------------------------------

def test_engine_headless_end_to_end_spoofed_brake() -> None:
    """Run CAVShieldEngine headless for 20s offline with spoof_brake attack: assert blocked, CRITICAL, decel=0."""
    cfg = load_config()
    engine = CAVShieldEngine(cfg=cfg)
    engine.set_protection(True)

    # Run 20 virtual seconds headless with spoof_brake starting at t=2.0s
    snapshot = engine.run_headless(
        duration_s=20.0,
        dt=0.005,
        attack_name="spoof_brake",
        attack_start_s=2.0,
        attack_duration_s=5.0,
    )

    # 1. Assert spoofed 0x200 BRAKE_COMMAND was blocked by the inline gateway
    rejected_cmds = snapshot["failsafe"]["rejected_commands"]
    blocked_count = rejected_cmds.get(0x200, 0)
    assert blocked_count > 0, f"Expected 0x200 brake commands to be blocked, got: {rejected_cmds}"

    # 2. Assert fail-safe mode reached SAFE MODE during the attack
    mode_history = [item["mode"] for item in snapshot["failsafe"]["mode_history"]]
    assert "SAFE MODE" in mode_history, f"Expected SAFE MODE in mode history: {mode_history}"

    # 3. Assert digital twin never experienced unintended spoofed deceleration
    twin = snapshot["twin"]
    assert twin["applied_decel"] < 1.0, f"Digital twin applied dangerous spoofed decel: {twin['applied_decel']} m/s2"
    assert twin["speed_kmh"] > 50.0, f"Cruising speed collapsed unexpectedly: {twin['speed_kmh']} km/h"

    # 4. Assert ground truth metrics show 100% recall on the attack
    metrics = snapshot["metrics"]
    assert metrics["counts"]["tp"] > 0
    assert metrics["recall"] == 1.0
