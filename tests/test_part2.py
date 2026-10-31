"""
Unit tests for Part 2:
- src.can_simulator: DBC, codec, checksum, alive counter, CANFrame, LabelRegistry, bus layer, ECUSimulator
- src.vehicle_digital_twin: DrivingMode, VehicleState, VehicleDigitalTwin dynamics, controllers, safety bookkeeping
"""

from pathlib import Path
import random
import sys
import threading
import time

import pytest

# Ensure repository root is in python path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import load_config
from src.can_simulator import (
    COMMAND_IDS,
    DBC,
    KNOWN_IDS,
    SAFETY_CRITICAL_IDS,
    CANFrame,
    ECUSimulator,
    InMemoryBus,
    LabelRegistry,
    MessageDef,
    checksum_ok,
    counter_of,
    create_bus,
    criticality,
    decode,
    encode,
    frame_to_message,
    is_safety_critical,
    make_frame,
    message_to_frame,
)
from src.vehicle_digital_twin import DrivingMode, VehicleDigitalTwin, VehicleState


# ---------------------------------------------------------------------------
# CAN Simulator Tests
# ---------------------------------------------------------------------------

class TestDBCAndHelpers:
    def test_all_10_ids_in_dbc(self):
        expected_ids = {0x100, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x200, 0x201, 0x202}
        assert set(DBC.keys()) == expected_ids
        assert set(KNOWN_IDS) == expected_ids

    def test_command_ids(self):
        assert set(COMMAND_IDS) == {0x200, 0x201, 0x202}

    def test_safety_critical_ids(self):
        expected = {0x101, 0x102, 0x103, 0x104, 0x200, 0x201, 0x202}
        assert set(SAFETY_CRITICAL_IDS) == expected
        for arb_id in expected:
            assert is_safety_critical(arb_id) is True
        assert is_safety_critical(0x100) is False
        assert is_safety_critical(0x105) is False
        assert is_safety_critical(0x106) is False
        assert is_safety_critical(0x999) is False

    def test_criticality_weights(self):
        assert criticality(0x100) == 0.6
        assert criticality(0x101) == 0.8
        assert criticality(0x102) == 0.9
        assert criticality(0x103) == 0.8
        assert criticality(0x104) == 0.8
        assert criticality(0x105) == 0.6
        assert criticality(0x106) == 0.6
        assert criticality(0x200) == 1.0
        assert criticality(0x201) == 1.0
        assert criticality(0x202) == 1.0
        # Unknown IDs get 0.5
        assert criticality(0x777) == 0.5


class TestCANCodec:
    def test_encode_decode_roundtrip_all_ids(self):
        test_signals = {
            0x100: {"rpm": 2500.0, "throttle": 45.0},
            0x101: {"speed_kmh": 72.45},
            0x102: {"pedal": 22.0, "brake_active": 1, "decel": 2.2},
            0x103: {"steering_deg": -15.5},
            0x104: {"gear": 3},
            0x105: {"wheel_fl": 72.0, "wheel_fr": 72.0, "wheel_rl": 71.5, "wheel_rr": 71.5},
            0x106: {"ax": -1.25, "yaw": 3.4},
            0x200: {"cmd_decel": 3.5, "source": 1},
            0x201: {"cmd_accel": 1.75, "source": 2},
            0x202: {"lka": 1, "acc": 1, "aeb": 1, "set_speed_kmh": 72.0},
        }

        for arb_id, signals in test_signals.items():
            for counter in range(16):
                data = encode(arb_id, signals, counter=counter)
                assert len(data) == 8
                assert checksum_ok(data) is True
                assert counter_of(data) == counter

                decoded = decode(arb_id, data)
                assert isinstance(decoded, dict)

                # Check decoded values match within precision tolerances
                if arb_id == 0x100:
                    assert abs(decoded["rpm"] - signals["rpm"]) < 1.0
                    assert abs(decoded["throttle"] - signals["throttle"]) < 1.0
                elif arb_id == 0x101:
                    assert abs(decoded["speed_kmh"] - signals["speed_kmh"]) < 0.02
                elif arb_id == 0x102:
                    assert abs(decoded["pedal"] - signals["pedal"]) < 0.6
                    assert decoded["brake_active"] == signals["brake_active"]
                    assert abs(decoded["decel"] - signals["decel"]) < 0.02
                elif arb_id == 0x103:
                    assert abs(decoded["steering_deg"] - signals["steering_deg"]) < 0.15
                elif arb_id == 0x104:
                    assert decoded["gear"] == signals["gear"]
                elif arb_id == 0x105:
                    assert abs(decoded["wheel_fl"] - signals["wheel_fl"]) < 0.75
                elif arb_id == 0x106:
                    assert abs(decoded["ax"] - signals["ax"]) < 0.02
                    assert abs(decoded["yaw"] - signals["yaw"]) < 0.02
                elif arb_id == 0x200:
                    assert abs(decoded["cmd_decel"] - signals["cmd_decel"]) < 0.02
                    assert decoded["source"] == signals["source"]
                elif arb_id == 0x201:
                    assert abs(decoded["cmd_accel"] - signals["cmd_accel"]) < 0.02
                    assert decoded["source"] == signals["source"]
                elif arb_id == 0x202:
                    assert decoded["lka"] == signals["lka"]
                    assert decoded["set_speed_kmh"] == signals["set_speed_kmh"]

    def test_checksum_failure(self):
        data = bytearray(encode(0x101, {"speed_kmh": 50.0}, counter=3))
        assert checksum_ok(data) is True
        # Corrupt one payload byte
        data[0] ^= 0xFF
        assert checksum_ok(data) is False

    def test_encode_unknown_id_raises_value_error(self):
        with pytest.raises(ValueError):
            encode(0x999, {})

    def test_decode_unknown_id_or_bad_length_raises(self):
        with pytest.raises(ValueError):
            decode(0x999, b"\x00" * 8)
        with pytest.raises(ValueError):
            decode(0x101, b"\x00" * 7)


class TestCANFrameAndLabelRegistry:
    def test_make_frame(self):
        frame = make_frame(timestamp=12.34, arb_id=0x101, data=b"\x00" * 8, label="Spoofing", subtype="spoof_speed")
        assert frame.timestamp == 12.34
        assert frame.arb_id == 0x101
        assert frame.dlc == 8
        assert frame.sender == "ECU_ENGINE"
        assert frame.msg_type == "VEHICLE_SPEED"
        assert frame.label == "Spoofing"
        assert frame.subtype == "spoof_speed"

    def test_label_registry_pop_fifo(self):
        reg = LabelRegistry()
        payload = b"\x12\x34\x56\x78\x00\x00\x00\x00"
        reg.register(0x101, payload, "Normal", "")
        reg.register(0x101, payload, "Flooding", "flood_high")

        lbl1, sub1 = reg.pop(0x101, payload)
        assert lbl1 == "Normal"
        assert sub1 == ""

        lbl2, sub2 = reg.pop(0x101, payload)
        assert lbl2 == "Flooding"
        assert sub2 == "flood_high"

        # Pop from empty returns default
        lbl3, sub3 = reg.pop(0x101, payload)
        assert lbl3 == "Normal"
        assert sub3 == ""

    def test_label_registry_thread_safety(self):
        reg = LabelRegistry()
        payload = b"\x01\x02\x03\x04\x05\x06\x07\x08"

        def worker():
            for i in range(100):
                reg.register(0x200, payload, f"lbl_{i}", "")
                reg.pop(0x200, payload)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


class TestBusLayer:
    def test_in_memory_bus_send_recv(self):
        bus = InMemoryBus("test_chan")
        frame = make_frame(1.0, 0x101, b"\x00" * 8)
        msg = frame_to_message(frame)
        bus.send(msg)

        received = bus.recv(timeout=0.1)
        assert received is not None
        reconstructed = message_to_frame(received)
        assert reconstructed is not None
        assert reconstructed.arb_id == 0x101

        bus.shutdown()
        assert bus.recv(timeout=0.01) is None
        with pytest.raises(RuntimeError):
            bus.send(msg)

    def test_message_to_frame_malformed_returns_none(self):
        assert message_to_frame(None) is None
        assert message_to_frame("not a message") is None
        class BadMsg:
            pass
        assert message_to_frame(BadMsg()) is None


class TestECUSimulator:
    def test_ecu_simulator_step_generation(self):
        sim = ECUSimulator(rng=random.Random(123))
        state = VehicleState()

        # Step through 0.5 seconds
        generated_frames = []
        for step_idx in range(50):
            now = step_idx * 0.01
            frames = sim.step(now, state)
            generated_frames.extend(frames)

        assert len(generated_frames) > 0
        for f in generated_frames:
            assert f.arb_id in KNOWN_IDS
            assert len(f.data) == 8
            assert checksum_ok(f.data) is True

    def test_counter_increments_modulo_16(self):
        sim = ECUSimulator(rng=random.Random(42))
        state = VehicleState()
        counters_0x100 = []

        now = 0.0
        while len(counters_0x100) < 32:
            frames = sim.step(now, state)
            for f in frames:
                if f.arb_id == 0x100:
                    counters_0x100.append(counter_of(f.data))
            now += 0.005

        # Check counter increments mod 16
        for i in range(len(counters_0x100) - 1):
            expected = (counters_0x100[i] + 1) % 16
            assert counters_0x100[i + 1] == expected


# ---------------------------------------------------------------------------
# Vehicle Digital Twin Tests
# ---------------------------------------------------------------------------

class TestVehicleDigitalTwin:
    def test_initial_state_and_snapshot(self):
        twin = VehicleDigitalTwin()
        snap = twin.snapshot()

        assert snap["speed_kmh"] == 72.0
        assert snap["gear"] == 3
        assert snap["aeb"] is False
        assert snap["unsafe"] == ""
        assert "perceived" in snap
        assert snap["perceived"]["speed_kmh"] == 72.0

    def test_accept_status_updates_perceived_state(self):
        twin = VehicleDigitalTwin()
        # Spoofed speed in cluster
        twin.accept_status(0x101, {"speed_kmh": 125.0})
        snap = twin.snapshot()
        assert snap["perceived"]["speed_kmh"] == 125.0
        # Physical speed remains 72.0
        assert snap["speed_kmh"] == 72.0

    def test_actuator_obeys_only_accepted_commands_within_timeout(self):
        twin = VehicleDigitalTwin()
        # No brake commands accepted yet -> applied decel is 0
        state = twin.step(dt=0.01, now=1.0)
        assert state.applied_decel == 0.0

        # Accept brake command at now = 2.0
        twin.accept_command(0x200, {"cmd_decel": 4.0, "source": 0}, now=2.0)
        state = twin.step(dt=0.01, now=2.01)
        assert state.applied_decel > 0.0  # begins rising

        # Advance past command timeout (0.06s) without new commands
        for _ in range(15):
            state = twin.step(dt=0.01, now=2.10)
        # Brakes decay back towards 0
        assert state.applied_decel < 0.5

    def test_modes_and_target_speed(self):
        twin = VehicleDigitalTwin()
        assert twin.mode == DrivingMode.NORMAL

        twin.set_mode(DrivingMode.DEGRADED)
        twin.step(dt=0.01, now=1.0)
        assert twin.state.target_speed_kmh == 50.0

        twin.set_mode(DrivingMode.SAFE_MODE)
        twin.step(dt=0.01, now=2.0)
        assert twin.state.target_speed_kmh == 30.0

    def test_aeb_trigger_from_sensor_snapshot(self):
        twin = VehicleDigitalTwin()

        class MockObstacle:
            range = 15.0
            closing = 12.0
            ttc = 1.25  # below aeb_ttc (1.8s)

        class MockSensorSnapshot:
            in_lane = MockObstacle()

        state = twin.step(dt=0.01, now=1.0, sensor_snapshot=MockSensorSnapshot())
        assert state.aeb is True
        assert state.decel > 0.0
        assert state.brake_cmd_source == 1  # AEB

    def test_v2x_hazard_application(self):
        twin = VehicleDigitalTwin()
        twin.apply_v2x_hazard(now=5.0)
        state = twin.step(dt=0.01, now=5.1)
        # Precrash decel is 3.0 m/s2
        assert state.decel >= 3.0

    def test_safety_bookkeeping_unintended_braking(self):
        twin = VehicleDigitalTwin()
        # Inject malicious high brake command that was accepted by plant
        twin.accept_command(0x200, {"cmd_decel": 7.0, "source": 0}, now=1.0)

        # Step with no legitimate controller braking (legit_decel = 0)
        for i in range(25):
            state = twin.step(dt=0.01, now=1.0 + i * 0.01)
            # Re-accept to prevent timeout during ramp-up
            twin.accept_command(0x200, {"cmd_decel": 7.0, "source": 0}, now=1.0 + i * 0.01)

        assert state.applied_decel > 2.0
        assert state.unsafe == "UNINTENDED BRAKING"
        assert state.unsafe_seconds > 0.0

