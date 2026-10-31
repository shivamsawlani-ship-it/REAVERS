"""
Part 5 Acceptance Verification Script.

Verifies:
1. 30 s of legitimate traffic produces ZERO evidence from rule_engine and physics_engine.
2. Each spoof attack type produces evidence with readable, numerical explanations.
3. A real obstacle warning is ACCEPTED by v2x_validator.
4. Phantom, fake e-brake, and GPS spoof V2X attacks are REJECTED with precise reasons.
"""

from pathlib import Path
import random
import sys

# Ensure repository root is in python path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import load_config
from src.attack_injector import (
    CANSpoofAccelAttack,
    CANSpoofBrakeAttack,
    CANSpoofGearAttack,
    CANSpoofRpmAttack,
    CANSpoofSpeedAttack,
    V2XFakeEbrakeAttack,
    V2XGpsAttack,
    V2XPhantomAttack,
)
from src.can_simulator import decode
from src.detection.physics_engine import PhysicsEngine
from src.detection.rule_engine import CANRuleEngine
from src.detection.v2x_validator import V2XValidator
from src.engine import World
from src.v2x_simulator import Environment, TrustStore, V2XMessage


def run_acceptance():
    cfg = load_config()
    world = World(cfg=cfg)
    rule_engine = CANRuleEngine(cfg=cfg)
    physics_engine = PhysicsEngine(cfg=cfg)
    trust_store = TrustStore()
    v2x_validator = V2XValidator(trust_store=trust_store, cfg=cfg)

    # -----------------------------------------------------------------------
    # Test 1: 30 Seconds of Legitimate Traffic -> ZERO Evidence
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("TEST 1: 30 SECONDS OF LEGITIMATE TRAFFIC")
    print("=" * 65)

    world_hz = 200.0
    dt = 1.0 / world_hz
    total_steps = int(30.0 * world_hz)

    total_frames = 0
    total_legit_evidence = 0

    for step in range(total_steps):
        now = step * dt
        can_frames, v2x_msgs = world.step(now=now, dt=dt)
        snapshot = world.env.sense(ego_speed_kmh=world.twin.state.speed_kmh, now=now)
        physics_engine.set_sensor_snapshot(snapshot)

        for frame in can_frames:
            total_frames += 1
            # Evaluate deterministic rules
            r_ev = rule_engine.evaluate(frame)
            # Evaluate physics invariants
            decoded = decode(frame.arb_id, frame.data)
            p_ev = physics_engine.check(frame, decoded)

            if r_ev or p_ev:
                total_legit_evidence += len(r_ev) + len(p_ev)
                for e in r_ev + p_ev:
                    print(f"  [UNEXPECTED EVIDENCE at t={now:.2f}s] {e.name}: {e.detail}")

            # Commit forwarded frame
            rule_engine.commit(frame)
            physics_engine.update(frame, decoded)

    print(f"Total legitimate frames evaluated: {total_frames}")
    print(f"Total false positive evidence items: {total_legit_evidence}")
    assert total_legit_evidence == 0, f"Expected 0 evidence on legit traffic, got {total_legit_evidence}"
    print("PASSED: 30 s of legitimate traffic produced ZERO false positive evidence.")

    # -----------------------------------------------------------------------
    # Test 2: Each Spoof Type Produces Evidence with Explanations
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TEST 2: CAN SPOOF DETECTION & EXPLANATIONS")
    print("=" * 65)

    now = 31.0
    spoof_tests = [
        ("spoof_brake", CANSpoofBrakeAttack(decel=7.0)),
        ("spoof_accel", CANSpoofAccelAttack(accel=4.5)),
        ("spoof_speed", CANSpoofSpeedAttack(speed_kmh=125.0)),
        ("spoof_rpm", CANSpoofRpmAttack(rpm=7000.0)),
        ("spoof_gear", CANSpoofGearAttack(gear=1)),
    ]

    for name, attack in spoof_tests:
        attack.start(now=now, duration=1.0)
        frames, _ = attack.tick(now=now + 0.01)
        assert len(frames) > 0, f"No frames from {name}"
        f = frames[0]
        decoded = decode(f.arb_id, f.data)

        r_ev = rule_engine.evaluate(f)
        p_ev = physics_engine.check(f, decoded)
        all_ev = r_ev + p_ev

        assert len(all_ev) > 0, f"Failed to detect {name}!"
        print(f"\n[{name.upper()}] detected:")
        for ev in all_ev:
            print(f"  -> Evidence: [{ev.source.upper()}] {ev.name} (sev={ev.severity:.2f}, hint={ev.hint})")
            print(f"     Explanation: {ev.detail}")

    # -----------------------------------------------------------------------
    # Test 3: V2X Real Obstacle Warning is ACCEPTED
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TEST 3: V2X REAL OBSTACLE WARNING (ACCEPT)")
    print("=" * 65)

    env = Environment(cfg=cfg)
    env.spawn_real_obstacle(distance=85.0, lifetime=14.0, now=now)
    snapshot = env.sense(ego_speed_kmh=72.0, now=now)

    # Legitimate warning from enrolled vehicle V-2041 with real obstacle in lane
    legit_warning = V2XMessage(
        msg_id="LEGIT-WARN-101",
        sender_id="V-2041",
        x=1055.0,
        y=3.6,
        speed=20.0,
        msg_type="COLLISION_WARNING",
        timestamp=now,
        obstacle={"obs_id": "OBS-REAL", "range": 85.0, "x": 1085.0, "y": 0.0},
        claimed_decel=2.5,
    )
    trust_store.sign(legit_warning)

    verdict_real = v2x_validator.validate(legit_warning, snapshot=snapshot, now=now)
    print(f"Real Obstacle Warning Verdict:")
    print(f"  Accepted: {verdict_real.accepted}")
    print(f"  Trust Score: {verdict_real.trust:.3f} (threshold >= 0.60)")
    print(f"  Sub-checks: {verdict_real.checks}")
    assert verdict_real.accepted is True, "Legitimate real obstacle warning must be ACCEPTED!"
    print("PASSED: Real obstacle warning was accepted.")

    # -----------------------------------------------------------------------
    # Test 4: Phantom, Fake E-Brake, and GPS Spoof are REJECTED
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TEST 4: V2X ATTACKS REJECTED WITH REASONS")
    print("=" * 65)

    # 4a. Phantom Hazard (V-9999 valid signature, but non-existent obstacle)
    phantom_atk = V2XPhantomAttack(trust_store=trust_store)
    phantom_atk.start(now=now, duration=1.0)
    _, phantom_msgs = phantom_atk.tick(now=now + 0.01, ctx=env)
    assert len(phantom_msgs) > 0

    verdict_phantom = v2x_validator.validate(phantom_msgs[0], snapshot=snapshot, now=now + 0.01)
    print("\n[PHANTOM HAZARD]")
    print(f"  Accepted: {verdict_phantom.accepted}")
    print(f"  Trust: {verdict_phantom.trust:.3f}")
    print(f"  Reasons: {verdict_phantom.reasons}")
    assert verdict_phantom.accepted is False
    assert any("hazard" in r.lower() or "phantom" in r.lower() for r in verdict_phantom.reasons)

    # 4b. Fake Emergency Braking (Impersonating V-2041 with INVALID signature)
    fake_ebrake_atk = V2XFakeEbrakeAttack()
    fake_ebrake_atk.start(now=now, duration=1.0)
    _, ebrake_msgs = fake_ebrake_atk.tick(now=now + 0.01, ctx=env)
    assert len(ebrake_msgs) > 0

    verdict_ebrake = v2x_validator.validate(ebrake_msgs[0], snapshot=snapshot, now=now + 0.01)
    print("\n[FAKE EMERGENCY BRAKING]")
    print(f"  Accepted: {verdict_ebrake.accepted}")
    print(f"  Trust: {verdict_ebrake.trust:.3f}")
    print(f"  Reasons: {verdict_ebrake.reasons}")
    assert verdict_ebrake.accepted is False
    assert any("signature" in r.lower() for r in verdict_ebrake.reasons)

    # 4c. GPS Spoof (Valid signature, but 150m erratic jump)
    # Prime validator with baseline beacon
    v2x_validator.validate(
        V2XMessage("BASE-1", "V-9999", 1020.0, 0.0, 20.0, "BSM", now),
        snapshot=snapshot,
        now=now,
    )
    # Jump by 150m
    gps_atk = V2XGpsAttack(trust_store=trust_store)
    gps_atk.start(now=now + 0.1, duration=1.0)
    _, gps_msgs = gps_atk.tick(now=now + 0.1, ctx=env)
    assert len(gps_msgs) > 0

    verdict_gps = v2x_validator.validate(gps_msgs[0], snapshot=snapshot, now=now + 0.1)
    print("\n[GPS POSITION JUMP]")
    print(f"  Accepted: {verdict_gps.accepted}")
    print(f"  Trust: {verdict_gps.trust:.3f}")
    print(f"  Reasons: {verdict_gps.reasons}")
    assert verdict_gps.accepted is False
    assert any("jump" in r.lower() or "kinematic" in r.lower() for r in verdict_gps.reasons)

    print("\n" + "=" * 65)
    print("ALL PART 5 ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY.")
    print("=" * 65)


if __name__ == "__main__":
    run_acceptance()

