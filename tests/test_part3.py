"""
Unit tests for Part 3:
- src.v2x_simulator: V2XMessage, TrustStore, Environment, SensorSnapshot, ObjectTrack, V2XSimulator
- src.attack_injector: Attack base, CAN attacks, V2X attacks, AttackInjector orchestrator, replay buffer, UDP commands
"""

from pathlib import Path
import random
import sys
import time

import pytest

# Ensure repository root is in python path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import load_config
from src.attack_injector import (
    AttackInjector,
    CANFloodAttack,
    CANFuzzAttack,
    CANReplayAttack,
    CANSpoofAccelAttack,
    CANSpoofBrakeAttack,
    CANSpoofGearAttack,
    CANSpoofRpmAttack,
    CANSpoofSpeedAttack,
    V2XFakeEbrakeAttack,
    V2XGpsAttack,
    V2XPhantomAttack,
)
from src.can_simulator import CANFrame, checksum_ok, decode, make_frame
from src.v2x_simulator import (
    Environment,
    ObjectTrack,
    SensorSnapshot,
    TrustStore,
    V2XMessage,
    V2XSimulator,
)


# ---------------------------------------------------------------------------
# V2XMessage & TrustStore Tests
# ---------------------------------------------------------------------------

class TestV2XMessage:
    def test_serialization_and_deserialization(self):
        msg = V2XMessage(
            msg_id="MSG-001",
            sender_id="V-2041",
            x=1050.25,
            y=3.6,
            speed=20.0,
            msg_type="BSM",
            timestamp=12.34,
            heading=0.05,
            signature="abcdef123456",
            seq=42,
            obstacle={"range": 50.0},
            claimed_decel=2.5,
            pos_accuracy_m=0.8,
            label="Normal",
            subtype="",
        )
        json_str = msg.to_json()
        restored = V2XMessage.from_json(json_str)

        assert restored is not None
        assert restored.msg_id == "MSG-001"
        assert restored.sender_id == "V-2041"
        assert restored.x == 1050.25
        assert restored.y == 3.6
        assert restored.speed == 20.0
        assert restored.signature == "abcdef123456"
        assert restored.obstacle == {"range": 50.0}

    def test_defensive_parsing_never_crashes(self):
        assert V2XMessage.from_json("") is None
        assert V2XMessage.from_json("invalid json string {") is None
        assert V2XMessage.from_json(None) is None
        assert V2XMessage.from_dict("not a dict") is None
        assert V2XMessage.from_dict({}) is not None  # fills defaults


class TestTrustStore:
    def test_sign_and_verify_enrolled_vehicle(self):
        ts = TrustStore(secret_seed="test-seed")
        msg = V2XMessage(
            msg_id="B-1",
            sender_id="V-2041",
            x=100.0,
            y=0.0,
            speed=20.0,
            msg_type="BSM",
            timestamp=1.0,
            seq=1,
        )
        sig = ts.sign(msg)
        assert len(sig) == 64
        assert ts.verify(msg) is True

    def test_compromised_vehicle_v9999_has_valid_signature(self):
        ts = TrustStore()
        msg = V2XMessage(
            msg_id="PHANTOM-1",
            sender_id="V-9999",
            x=200.0,
            y=0.0,
            speed=20.0,
            msg_type="COLLISION_WARNING",
            timestamp=2.0,
            seq=1,
        )
        ts.sign(msg)
        assert ts.verify(msg) is True

    def test_verify_fails_on_tampered_payload_or_unenrolled_sender(self):
        ts = TrustStore()
        msg = V2XMessage(
            msg_id="B-2",
            sender_id="V-2041",
            x=100.0,
            y=0.0,
            speed=20.0,
            msg_type="BSM",
            timestamp=1.0,
            seq=1,
        )
        ts.sign(msg)

        # Tamper payload
        msg.x = 999.0
        assert ts.verify(msg) is False

        # Unenrolled sender
        msg.sender_id = "V-UNKNOWN"
        assert ts.verify(msg) is False


# ---------------------------------------------------------------------------
# Environment & Sensor Fusion Tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_environment_stepping_and_complementary_filter(self):
        env = Environment()
        initial_x = env.ego_x

        for step in range(50):
            env.step(dt=0.02, now=step * 0.02, ego_speed_kmh=72.0)

        # 1.0 second at 20 m/s -> ~20 m moved
        assert abs((env.ego_x - initial_x) - 20.0) < 0.5
        # Complementary filter tracks true position within GPS noise bound
        assert abs(env.ego_x_est - env.ego_x) < 3.0

    def test_multi_sensor_fusion_ranges(self):
        env = Environment()
        # Vehicles V-2041 (~55m ahead), V-2042 (~40m ahead), V-2043 (~50m behind)
        snapshot = env.sense(ego_speed_kmh=72.0, now=0.0)

        assert len(snapshot.tracks) == 3
        # V-2042 is at ~40m -> should have radar, lidar, and camera (<= 70m)
        v2042_track = next(t for t in snapshot.tracks if t.obj_id == "V-2042")
        assert "radar" in v2042_track.sources
        assert "lidar" in v2042_track.sources
        assert "camera" in v2042_track.sources

    def test_spawn_real_obstacle_and_in_lane_detection(self):
        env = Environment()
        obs = env.spawn_real_obstacle(distance=50.0, lifetime=5.0, now=0.0)

        snapshot = env.sense(ego_speed_kmh=72.0, now=0.0)
        assert snapshot.in_lane is not None
        assert snapshot.in_lane.obj_id == obs.obs_id
        assert abs(snapshot.in_lane.dx - 50.0) < 1.0


# ---------------------------------------------------------------------------
# V2XSimulator Tests
# ---------------------------------------------------------------------------

class TestV2XSimulator:
    def test_beacon_generation_and_warning_on_obstacle(self):
        env = Environment()
        v2x_sim = V2XSimulator(env=env)

        # Step 0.5s: 3 vehicles * 10 Hz * 0.5s = ~15 BSM messages
        msgs = []
        for step in range(50):
            now = step * 0.01
            msgs.extend(v2x_sim.step(now=now))

        bsm_msgs = [m for m in msgs if m.msg_type == "BSM"]
        assert len(bsm_msgs) >= 12
        for m in bsm_msgs:
            assert v2x_sim.trust_store.verify(m) is True

        # Spawn obstacle ahead of V-2041 -> generates COLLISION_WARNING
        env.spawn_real_obstacle(distance=80.0, lifetime=10.0, now=1.0)
        warning_msgs = []
        for step in range(50, 100):
            now = step * 0.01
            new_msgs = v2x_sim.step(now=now)
            warning_msgs.extend([m for m in new_msgs if m.msg_type == "COLLISION_WARNING"])

        assert len(warning_msgs) > 0
        assert warning_msgs[0].sender_id == "V-2041"
        assert v2x_sim.trust_store.verify(warning_msgs[0]) is True


# ---------------------------------------------------------------------------
# Attack Injector Tests
# ---------------------------------------------------------------------------

class TestAttackInjector:
    def test_attack_lifecycle_and_active_list(self):
        injector = AttackInjector()
        assert injector.active() == []

        injector.start("flood", duration=2.0, now=1.0)
        assert "flood" in injector.active()

        # Tick at t = 2.0 (active)
        can_frames, _ = injector.tick(now=2.0)
        assert len(can_frames) > 0

        # Tick past duration at t = 3.5 (expired)
        injector.tick(now=3.5)
        assert "flood" not in injector.active()

    def test_can_spoof_brake_decoding_and_checksum(self):
        injector = AttackInjector()
        injector.start("spoof_brake", duration=1.0, now=0.0)

        frames, _ = injector.tick(now=0.02)
        assert len(frames) > 0
        frame = frames[0]
        assert frame.arb_id == 0x200
        assert checksum_ok(frame.data) is True

        decoded = decode(frame.arb_id, frame.data)
        assert decoded["cmd_decel"] == 7.0
        assert decoded["source"] == 0

    def test_replay_attack_buffer_and_playback(self):
        injector = AttackInjector()
        # Seed replay buffer with legitimate brake command frame
        fake_legit_data = b"\x01\x2c\x00\x00\x00\x00\x00\x00"  # non-zero decel
        legit_frame = make_frame(timestamp=10.0, arb_id=0x200, data=fake_legit_data)
        injector.observe([legit_frame])

        # Start replay attack
        injector.start("replay", duration=2.0, now=15.0)
        frames, _ = injector.tick(now=15.01)
        assert len(frames) > 0
        assert frames[0].label == "Replay"
        assert frames[0].subtype == "replay_brake"
        assert frames[0].data == fake_legit_data

    def test_v2x_fake_ebrake_has_invalid_signature(self):
        injector = AttackInjector()
        injector.start("v2x_fake_ebrake", duration=1.0, now=0.0)

        _, v2x_msgs = injector.tick(now=0.05)
        assert len(v2x_msgs) > 0
        msg = v2x_msgs[0]
        assert msg.sender_id == "V-2041"
        assert msg.claimed_decel == 9.0
        # Signature must be invalid
        assert injector.trust_store.verify(msg) is False

    def test_v2x_phantom_has_valid_signature_from_v9999(self):
        injector = AttackInjector()
        injector.start("v2x_phantom", duration=1.0, now=0.0)

        _, v2x_msgs = injector.tick(now=0.05)
        assert len(v2x_msgs) > 0
        msg = v2x_msgs[0]
        assert msg.sender_id == "V-9999"
        assert msg.obstacle is not None
        # Cryptographically valid signature
        assert injector.trust_store.verify(msg) is True

    def test_udp_command_handler(self):
        injector = AttackInjector()
        # Test internal command dispatch
        res = injector._handle_command({"cmd": "start", "attack": "spoof_speed", "duration": 5.0})
        assert res["status"] == "ok"
        assert "spoof_speed" in res["active"]

        res2 = injector._handle_command({"cmd": "list"})
        assert "spoof_speed" in res2["active"]
        assert "spoof_speed" in res2["available"]

        res3 = injector._handle_command({"cmd": "stop", "attack": "spoof_speed"})
        assert res3["status"] == "ok"
        assert "spoof_speed" not in res3["active"]

        res4 = injector._handle_command({"cmd": "stop_all"})
        assert res4["active"] == []

