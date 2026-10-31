"""
Unit tests for Part 5:
- src.detection.rule_engine: Evidence, CANRuleEngine rules, timing, counter, flood, commit
- src.detection.physics_engine: PhysicsEngine cross-checks, brake justification, speed/wheel, rpm, gear
- src.detection.v2x_validator: V2XVerdict, V2XValidator signature, freshness, kinematics, ghost, hazard
"""

from pathlib import Path
import random
import sys

import pytest

# Ensure repository root is in python path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import load_config
from src.can_simulator import (
    CANFrame,
    decode,
    encode,
    make_frame,
)
from src.detection.physics_engine import PhysicsEngine
from src.detection.rule_engine import CANRuleEngine, Evidence
from src.detection.v2x_validator import V2XValidator, V2XVerdict
from src.v2x_simulator import ObjectTrack, SensorSnapshot, TrustStore, V2XMessage


# ---------------------------------------------------------------------------
# Rule Engine Tests
# ---------------------------------------------------------------------------

class TestCANRuleEngine:
    def test_unknown_id_and_dlc_mismatch(self):
        engine = CANRuleEngine()
        # Unknown ID
        f_unk = make_frame(1.0, 0x999, b"\x00" * 8)
        evs = engine.evaluate(f_unk)
        assert any(e.name == "UNKNOWN_ID" and e.hint == "Fuzzing" for e in evs)

        # DLC mismatch
        f_bad_dlc = CANFrame(1.0, 0x101, 7, b"\x00" * 7, "ECU_ENGINE", "VEHICLE_SPEED")
        evs_dlc = engine.evaluate(f_bad_dlc)
        assert any(e.name == "DLC_MISMATCH" for e in evs_dlc)

    def test_checksum_violation(self):
        engine = CANRuleEngine()
        # Valid frame data
        data = bytearray(encode(0x101, {"speed_kmh": 70.0}, counter=1))
        # Corrupt checksum byte 6
        data[6] ^= 0xFF
        f_corrupt = make_frame(1.0, 0x101, bytes(data))
        evs = engine.evaluate(f_corrupt)
        assert any(e.name == "CHECKSUM" and e.severity == 0.8 for e in evs)

    def test_counter_continuity_and_replay_detection(self):
        engine = CANRuleEngine()
        data1 = encode(0x101, {"speed_kmh": 70.0}, counter=1)
        f1 = make_frame(1.0, 0x101, data1)
        assert engine.evaluate(f1) == []
        engine.commit(f1)

        # Normal sequential counter 2 -> no evidence
        data2 = encode(0x101, {"speed_kmh": 70.1}, counter=2)
        f2 = make_frame(1.02, 0x101, data2)
        assert engine.evaluate(f2) == []
        engine.commit(f2)

        # Counter jump to 8 -> counter anomaly
        data_jump = encode(0x101, {"speed_kmh": 70.2}, counter=8)
        f_jump = make_frame(1.04, 0x101, data_jump)
        evs_jump = engine.evaluate(f_jump)
        assert any(e.name == "COUNTER" and e.hint == "Spoofing" for e in evs_jump)

        # Replay: exact same payload data1 with counter 1 re-injected
        f_replay = make_frame(1.06, 0x101, data1)
        evs_replay = engine.evaluate(f_replay)
        # Should be flagged with higher severity 0.85 and hint 'Replay'
        assert any(e.name == "COUNTER" and e.hint == "Replay" and e.severity == 0.85 for e in evs_replay)

    def test_timing_inter_arrival_check(self):
        engine = CANRuleEngine()
        # 0x101 has nominal period 20 ms
        f1 = make_frame(10.0, 0x101, encode(0x101, {"speed_kmh": 70.0}, counter=1))
        engine.commit(f1)

        # Injected frame arrives after only 2 ms (< 0.45 * 20ms = 9ms)
        f_fast = make_frame(10.002, 0x101, encode(0x101, {"speed_kmh": 70.0}, counter=2))
        evs = engine.evaluate(f_fast)
        assert any(e.name == "TIMING" for e in evs)

    def test_rate_flood_and_stream_rate_ratio(self):
        engine = CANRuleEngine()
        # Flood 0x101 at high frequency (40 frames in 0.1 s = 400 Hz > 2.5 * 50 Hz = 125 Hz)
        detected_flood = False
        for i in range(40):
            f = make_frame(1.0 + i * 0.002, 0x101, encode(0x101, {"speed_kmh": 70.0}, counter=i % 16))
            evs = engine.evaluate(f)
            if any(e.name == "RATE_FLOOD" and e.hint == "Flooding" for e in evs):
                detected_flood = True

        assert detected_flood is True
        ratio = engine.stream_rate_ratio(0x101)
        assert ratio > 2.5


# ---------------------------------------------------------------------------
# Physics Engine Tests
# ---------------------------------------------------------------------------

class TestPhysicsEngine:
    def test_unjustified_driver_brake_command(self):
        phys = PhysicsEngine()
        # Validated pedal is 0.0%
        phys.validated_pedal = 0.0
        # Command specifies 7.0 m/s2 deceleration from DRIVER (source 0)
        data = encode(0x200, {"cmd_decel": 7.0, "source": 0}, counter=1)
        f = make_frame(1.0, 0x200, data)
        evs = phys.check(f, decode(f.arb_id, f.data))

        assert len(evs) > 0
        assert any(e.name == "UNJUSTIFIED_DRIVER_BRAKE" and "7.00 m/s²" in e.detail for e in evs)

    def test_justified_driver_brake_produces_no_evidence(self):
        phys = PhysicsEngine()
        phys.validated_pedal = 22.0
        # Expected decel: 22 * 0.1 = 2.2 m/s2
        data = encode(0x200, {"cmd_decel": 2.2, "source": 0}, counter=1)
        f = make_frame(1.0, 0x200, data)
        evs = phys.check(f, decode(f.arb_id, f.data))
        assert evs == []

    def test_unjustified_aeb_brake(self):
        phys = PhysicsEngine()
        phys.validated_aeb_flag = True
        # Sensor snapshot has no obstacle ahead (TTC = 999.0)
        phys.set_sensor_snapshot(SensorSnapshot(timestamp=1.0, tracks=[], in_lane=None))

        data = encode(0x200, {"cmd_decel": 5.0, "source": 1}, counter=1)  # AEB source
        f = make_frame(1.0, 0x200, data)
        evs = phys.check(f, decode(f.arb_id, f.data))
        assert any(e.name == "UNJUSTIFIED_AEB_BRAKE" for e in evs)

    def test_accel_command_conflicts_and_limits(self):
        phys = PhysicsEngine()
        # Exceeds max_accel_cmd (3.0 m/s2)
        data_high = encode(0x201, {"cmd_accel": 4.5, "source": 0}, counter=1)
        f_high = make_frame(1.0, 0x201, data_high)
        evs_high = phys.check(f_high, decode(f_high.arb_id, f_high.data))
        assert any(e.name == "ACCEL_LIMIT_EXCEEDED" for e in evs_high)

        # Accel commanded during braking
        phys.validated_pedal = 25.0
        data_conf = encode(0x201, {"cmd_accel": 2.0, "source": 0}, counter=2)
        f_conf = make_frame(1.02, 0x201, data_conf)
        evs_conf = phys.check(f_conf, decode(f_conf.arb_id, f_conf.data))
        assert any(e.name == "CONFLICTING_ACCEL" for e in evs_conf)

    def test_speed_and_wheel_mismatch(self):
        phys = PhysicsEngine()
        phys.validated_wheel_avg = 72.0
        # Reported speed 125 km/h differs by > 4.0 km/h
        data_spd = encode(0x101, {"speed_kmh": 125.0}, counter=1)
        f_spd = make_frame(1.0, 0x101, data_spd)
        evs = phys.check(f_spd, decode(f_spd.arb_id, f_spd.data))
        assert any(e.name == "SPEED_WHEEL_MISMATCH" for e in evs)

    def test_rpm_inconsistency(self):
        phys = PhysicsEngine()
        phys.validated_speed = 72.0
        # Expected RPM is ~2766. Report 7000 RPM
        data_rpm = encode(0x100, {"rpm": 7000.0, "throttle": 0.0}, counter=1)
        f_rpm = make_frame(1.0, 0x100, data_rpm)
        evs = phys.check(f_rpm, decode(f_rpm.arb_id, f_rpm.data))
        assert any(e.name == "RPM_INCONSISTENCY" for e in evs)

    def test_gear_change_speed_interlock(self):
        phys = PhysicsEngine()
        phys.validated_speed = 72.0
        # Gear 1 = Reverse while driving at 72 km/h
        data_gear = encode(0x104, {"gear": 1}, counter=1)
        f_gear = make_frame(1.0, 0x104, data_gear)
        evs = phys.check(f_gear, decode(f_gear.arb_id, f_gear.data))
        assert any(e.name == "IMPLAUSIBLE_GEAR" for e in evs)


# ---------------------------------------------------------------------------
# V2X Validator Tests
# ---------------------------------------------------------------------------

class TestV2XValidator:
    def test_valid_signed_bsm_accepted(self):
        ts = TrustStore()
        val = V2XValidator(trust_store=ts)
        msg = V2XMessage("M-101", "V-2041", 1040.0, 3.6, 20.0, "BSM", 10.0)
        ts.sign(msg)

        snap = SensorSnapshot(
            timestamp=10.0,
            tracks=[ObjectTrack("V-2041", 40.0, 40.0, 3.6, 0.0, 999.0, ["radar"], False)],
            ego_x_est=1000.0,
            ego_y_est=0.0,
        )
        verdict = val.validate(msg, snapshot=snap, now=10.05)
        assert verdict.accepted is True
        assert verdict.trust >= 0.60

    def test_invalid_signature_rejected(self):
        ts = TrustStore()
        val = V2XValidator(trust_store=ts)
        msg = V2XMessage("M-102", "V-2041", 1040.0, 3.6, 20.0, "BSM", 10.0, signature="FAKE_SIG")
        verdict = val.validate(msg, snapshot=None, now=10.05)
        assert verdict.accepted is False
        assert any("signature" in r.lower() for r in verdict.reasons)

    def test_ghost_vehicle_rejected(self):
        ts = TrustStore()
        val = V2XValidator(trust_store=ts)
        # Claims position 30m away (<= 80m radar coverage)
        msg = V2XMessage("M-103", "V-9999", 1030.0, 0.0, 20.0, "BSM", 10.0)
        ts.sign(msg)

        # But empty sensor snapshot (no vehicle visible)
        snap = SensorSnapshot(timestamp=10.0, tracks=[], ego_x_est=1000.0, ego_y_est=0.0)
        verdict = val.validate(msg, snapshot=snap, now=10.02)
        assert verdict.accepted is False
        assert any("ghost" in r.lower() for r in verdict.reasons)

    def test_phantom_hazard_warning_rejected(self):
        ts = TrustStore()
        val = V2XValidator(trust_store=ts)
        # Warning claims obstacle at range 40m in ego lane
        msg = V2XMessage(
            "W-104", "V-9999", 1030.0, 0.0, 20.0, "COLLISION_WARNING", 10.0,
            obstacle={"range": 40.0, "x": 1040.0, "y": 0.0},
        )
        ts.sign(msg)

        # Empty snapshot (no obstacle on radar)
        snap = SensorSnapshot(timestamp=10.0, tracks=[], in_lane=None, ego_x_est=1000.0, ego_y_est=0.0)
        verdict = val.validate(msg, snapshot=snap, now=10.02)
        assert verdict.accepted is False
        assert any("phantom" in r.lower() or "hazard" in r.lower() for r in verdict.reasons)

    def test_real_obstacle_hazard_warning_accepted(self):
        ts = TrustStore()
        val = V2XValidator(trust_store=ts)
        msg = V2XMessage(
            "W-105", "V-2041", 1055.0, 3.6, 20.0, "COLLISION_WARNING", 10.0,
            obstacle={"range": 85.0, "x": 1085.0, "y": 0.0},
        )
        ts.sign(msg)

        # Sensor snapshot confirms real obstacle track at 85m in lane
        snap = SensorSnapshot(
            timestamp=10.0,
            tracks=[
                ObjectTrack("V-2041", 55.0, 55.0, 3.6, 0.0, 999.0, ["radar"], False),
                ObjectTrack("OBS-1", 85.0, 85.0, 0.0, 20.0, 4.25, ["radar", "lidar"], True),
            ],
            in_lane=ObjectTrack("OBS-1", 85.0, 85.0, 0.0, 20.0, 4.25, ["radar", "lidar"], True),
            ego_x_est=1000.0,
            ego_y_est=0.0,
        )
        verdict = val.validate(msg, snapshot=snap, now=10.02)
        assert verdict.accepted is True
        assert verdict.checks["hazard_confirmed"] is True
