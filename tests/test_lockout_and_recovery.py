"""Tests for gateway anti-lockout, stateful resync, fast recovery, and command starvation."""

import time
import pytest
from src import load_config
from src.can_simulator import CANFrame, encode
from src.detection.decision_engine import Decision, RiskEngine, RiskLevel
from src.detection.rule_engine import CANRuleEngine
from src.engine import CAVShieldEngine
from src.failsafe.failsafe_manager import FailSafeManager
from src.vehicle_digital_twin import VehicleDigitalTwin


def test_lockout_watchdog_resync():
    """Verify that if an ID is blocked repeatedly despite clean counters,
    the lockout watchdog triggers a resync callback and recovers without incident spam.
    """
    config = load_config()
    resynced_ids = []

    def on_resync(arb_id):
        resynced_ids.append(arb_id)

    fs = FailSafeManager(config)
    fs.resync_callback = on_resync

    block_dec = Decision(
        level=RiskLevel.HIGH,
        score=0.85,
        attack_type="Spoofing",
        confidence=0.9,
        explanation="Test block",
        action="Drop frame",
    )

    t0 = 100.0
    for i in range(25):
        now = t0 + i * 0.05  # 50ms intervals -> spans 1.25s (> 1.0s)
        cnt = (i + 1) % 16
        data = encode(0x100, {"rpm": 2700}, counter=cnt)
        frame = CANFrame(timestamp=now, arb_id=0x100, dlc=8, data=data, sender="ECM", msg_type="EngineData")
        fs.enforce(block_dec, frame, now=now)

    # Watchdog should have triggered resync
    assert 0x100 in resynced_ids, "Lockout watchdog should have triggered resync on 0x100"
    # Deduplicated incident: no incident spam (at most a few merged incidents instead of 25)
    assert fs.incident_id_counter <= 5, f"Expected deduplicated incidents, got {fs.incident_id_counter}"


def test_command_starvation_warning():
    """Verify that command starvation warning is raised if legitimate control commands
    (0x200 brake, 0x201 accel) are blocked for > 200ms.
    """
    config = load_config()
    fs = FailSafeManager(config)

    block_dec = Decision(
        level=RiskLevel.HIGH,
        score=0.85,
        attack_type="Spoofing",
        confidence=0.9,
        explanation="Test block",
        action="Drop frame",
    )
    pass_dec = Decision(
        level=RiskLevel.LOW,
        score=0.05,
        attack_type="Normal",
        confidence=0.99,
        explanation="Normal",
        action="Forward frame",
    )

    t0 = 50.0
    # Simulate blocking normal brake command 0x200
    frame_blk1 = CANFrame(timestamp=t0, arb_id=0x200, dlc=8, data=bytes(8), sender="ABS_ESP", msg_type="BrakeCommand", label="Normal")
    fs.enforce(block_dec, frame_blk1, now=t0)
    assert fs.starvation_warning is None, "Should not warn on first drop"

    # Frame blocked at t0 + 250ms (> 200ms starvation)
    frame_blk2 = CANFrame(timestamp=t0 + 0.25, arb_id=0x200, dlc=8, data=bytes(8), sender="ABS_ESP", msg_type="BrakeCommand", label="Normal")
    fs.enforce(block_dec, frame_blk2, now=t0 + 0.25)
    assert fs.starvation_warning is not None, "Starvation warning must be set when blocked > 200ms"

    # Forwarding a legitimate frame clears starvation
    frame_ok = CANFrame(timestamp=t0 + 0.30, arb_id=0x200, dlc=8, data=bytes(8), sender="ABS_ESP", msg_type="BrakeCommand", label="Normal")
    fs.enforce(pass_dec, frame_ok, now=t0 + 0.30)
    assert fs.starvation_warning is None, "Starvation warning must clear when command is forwarded"


def test_stateful_resync_on_clean_run():
    """Verify that 3 consecutive clean frames re-baseline counters and timing."""
    config = load_config()
    rule_engine = CANRuleEngine(config)

    # Prime rule engine on 0x100 with counter = 1 at t = 1.0
    data1 = encode(0x100, {"rpm": 2700}, counter=1)
    frame1 = CANFrame(timestamp=1.0, arb_id=0x100, dlc=8, data=data1, sender="ECM", msg_type="EngineData")
    rule_engine.evaluate(frame1)
    rule_engine.commit(frame1)

    # Simulate dropped frames causing counter desync: incoming has counter = 10 (expected 2)
    data_desync = encode(0x100, {"rpm": 2700}, counter=10)
    desync_frame = CANFrame(timestamp=1.02, arb_id=0x100, dlc=8, data=data_desync, sender="ECM", msg_type="EngineData")
    res_desync = rule_engine.evaluate(desync_frame)
    assert any(e.name == "COUNTER" for e in res_desync)

    # Now feed 3 consecutive frames with counter +1 each: 11, 12, 13
    d1 = encode(0x100, {"rpm": 2700}, counter=11)
    f1 = CANFrame(timestamp=1.04, arb_id=0x100, dlc=8, data=d1, sender="ECM", msg_type="EngineData")
    rule_engine.evaluate(f1)

    d2 = encode(0x100, {"rpm": 2700}, counter=12)
    f2 = CANFrame(timestamp=1.06, arb_id=0x100, dlc=8, data=d2, sender="ECM", msg_type="EngineData")
    rule_engine.evaluate(f2)

    d3 = encode(0x100, {"rpm": 2700}, counter=13)
    f3 = CANFrame(timestamp=1.08, arb_id=0x100, dlc=8, data=d3, sender="ECM", msg_type="EngineData")
    res3 = rule_engine.evaluate(f3)

    # 3 consecutive clean frames must have re-baselined: f3 should NOT fail counter
    assert not any(e.name == "COUNTER" for e in res3), "Counter must re-baseline after 3 consecutive clean frames"


def test_perceived_state_stale_and_age():
    """Verify that digital twin emits perceived values as None and tracks perceived_age_s
    when no accepted frames arrive for > 1.0s.
    """
    config = load_config()
    twin = VehicleDigitalTwin(config)

    # Update with an accepted speed frame at t = 10.0
    twin.accept_status(0x101, {"speed_kmh": 72.0}, now=10.0)

    snap_fresh = twin.snapshot(now=10.2)
    assert snap_fresh["perceived"]["speed_kmh"] is not None
    assert snap_fresh["perceived_age_s"]["speed_kmh"] == pytest.approx(0.2, abs=0.05)

    # At t = 11.5 (> 1.0s elapsed), perceived speed should be None (stale)
    snap_stale = twin.snapshot(now=11.5)
    assert snap_stale["perceived"]["speed_kmh"] is None
    assert snap_stale["perceived_age_s"]["speed_kmh"] == pytest.approx(1.5, abs=0.05)


def test_headless_short_soak_and_recovery():
    """Verify headless engine runs clean traffic with 0 incidents and recovers quickly (< 3s) from an attack."""
    config = load_config()
    engine = CAVShieldEngine(config)
    engine.failsafe.fast_recovery = True
    engine.start()

    try:
        # Run 4s clean traffic
        time.sleep(4.0)
        snap = engine.snapshot()
        assert snap["mode"] == "NORMAL", f"Expected NORMAL, got {snap['mode']}"
        assert snap["failsafe"]["incident_id"] == 0, f"Expected 0 incidents, got {snap['failsafe']['incident_id']}"
        assert abs(snap["twin"]["speed_kmh"] - 72.0) < 3.0, f"Speed drifted: {snap['twin']['speed_kmh']}"

        # Inject spoof_brake for 3s
        engine.inject("spoof_brake")
        time.sleep(3.0)
        engine.stop_attack("spoof_brake")

        # System should recover to NORMAL in < 3s
        t_stop = time.time()
        recovered = False
        while time.time() - t_stop < 3.5:
            if engine.snapshot()["mode"] == "NORMAL":
                recovered = True
                break
            time.sleep(0.1)

        assert recovered, "System must recover to NORMAL within 3s after attack stops"
    finally:
        engine.stop()
