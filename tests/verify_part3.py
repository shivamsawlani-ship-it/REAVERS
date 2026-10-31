"""
Part 3 Acceptance Verification Script.

Runs Environment + V2X + AttackInjector for 10 s of virtual time:
- Prints a count of frames per attack.
- Shows that a spoof_brake frame decodes to 7.0 m/s² with a valid checksum.
"""

import collections
from pathlib import Path
import random
import sys

# Ensure repository root is in python path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import load_config
from src.attack_injector import AttackInjector
from src.can_simulator import checksum_ok, decode
from src.v2x_simulator import Environment, TrustStore, V2XSimulator


def run_acceptance(duration_s: float = 10.0, world_hz: float = 200.0):
    cfg = load_config()
    dt = 1.0 / world_hz
    total_steps = int(duration_s * world_hz)

    rng = random.Random(1337)
    trust_store = TrustStore()
    env = Environment(cfg=cfg, rng=rng)
    v2x_sim = V2XSimulator(env=env, trust_store=trust_store, cfg=cfg)
    injector = AttackInjector(cfg=cfg, trust_store=trust_store, rng=rng)

    # Spawn real obstacle ahead at t = 2.0s
    env.spawn_real_obstacle(distance=90.0, lifetime=14.0, now=2.0)

    # Activate several attacks at various intervals
    injector.start("flood", duration=3.0, now=0.5)
    injector.start("fuzz", duration=3.0, now=4.0)
    injector.start("spoof_brake", duration=4.0, now=1.0)
    injector.start("v2x_phantom", duration=3.0, now=2.0)
    injector.start("v2x_fake_ebrake", duration=3.0, now=3.0)

    can_counts_by_attack = collections.Counter()
    v2x_counts_by_attack = collections.Counter()
    spoof_brake_sample = None

    for step_idx in range(total_steps):
        now = step_idx * dt

        # Step Environment and V2X
        env.step(dt=dt, now=now, ego_speed_kmh=72.0)
        v2x_msgs = v2x_sim.step(now=now)
        for msg in v2x_msgs:
            v2x_counts_by_attack[msg.subtype or "normal_v2x"] += 1

        # Step Attack Injector
        injected_can, injected_v2x = injector.tick(now=now, ctx=env)

        for frame in injected_can:
            can_counts_by_attack[frame.subtype] += 1
            if frame.subtype == "spoof_brake" and spoof_brake_sample is None:
                spoof_brake_sample = frame

        for vmsg in injected_v2x:
            v2x_counts_by_attack[vmsg.subtype] += 1

    print("=" * 60)
    print("PART 3 ACCEPTANCE RESULTS (10 s Virtual Time)")
    print("=" * 60)

    print("\nInjected CAN Frames per Attack Subtype:")
    for subtype, count in can_counts_by_attack.items():
        print(f"  {subtype:20s}: {count:5d} frames")

    print("\nV2X Messages per Subtype:")
    for subtype, count in v2x_counts_by_attack.items():
        print(f"  {subtype:20s}: {count:5d} messages")

    assert spoof_brake_sample is not None, "No spoof_brake frames were generated!"
    is_chk_ok = checksum_ok(spoof_brake_sample.data)
    decoded = decode(spoof_brake_sample.arb_id, spoof_brake_sample.data)

    print("\nSpoof Brake Frame Verification:")
    print(f"  Arb ID:        0x{spoof_brake_sample.arb_id:03X}")
    print(f"  Raw Payload:   {spoof_brake_sample.data.hex()}")
    print(f"  Checksum OK:   {is_chk_ok}")
    print(f"  Decoded Decel: {decoded.get('cmd_decel')} m/s²")
    print(f"  Decoded Src:   {decoded.get('source')} (0 = DRIVER)")
    print("=" * 60)

    # Acceptance Assertions
    assert is_chk_ok is True, "spoof_brake frame checksum must be valid!"
    assert decoded.get("cmd_decel") == 7.0, f"Expected 7.0 m/s², got {decoded.get('cmd_decel')}"
    assert can_counts_by_attack["spoof_brake"] >= 350, "Expected ~400 spoof_brake frames over 4s"
    assert can_counts_by_attack["flood_bus"] >= 1800, "Expected ~2100 flood frames over 3s"
    assert can_counts_by_attack["fuzz_random"] >= 800, "Expected ~900 fuzz frames over 3s"
    assert v2x_counts_by_attack["v2x_phantom"] >= 25, "Expected ~30 phantom messages over 3s"
    assert v2x_counts_by_attack["v2x_fake_ebrake"] >= 25, "Expected ~30 fake ebrake messages over 3s"

    print("ALL PART 3 ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY.")


if __name__ == "__main__":
    run_acceptance()

