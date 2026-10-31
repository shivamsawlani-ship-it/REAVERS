"""
Part 2 Acceptance Verification Script.

Steps VehicleDigitalTwin + ECUSimulator for 20 s of virtual time:
- Prints message count per ID (about 50/s for 20 ms IDs, about 10/s for 0x104).
- Confirms checksum_ok for all generated frames.
- Shows speed staying near 72 km/h with a mild brake dip around t = 6 s.
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
from src.can_simulator import (
    COMMAND_IDS,
    KNOWN_IDS,
    MessageDef,
    DBC,
    ECUSimulator,
    checksum_ok,
    decode,
)
from src.vehicle_digital_twin import VehicleDigitalTwin


def run_acceptance(duration_s: float = 20.0, world_hz: float = 200.0):
    cfg = load_config()
    dt = 1.0 / world_hz
    total_steps = int(duration_s * world_hz)

    rng = random.Random(1337)
    twin = VehicleDigitalTwin(cfg)
    ecu_sim = ECUSimulator(rng=rng, cfg=cfg)

    message_counts = collections.Counter()
    all_checksums_ok = True
    speed_samples = []

    for step_idx in range(total_steps):
        now = step_idx * dt

        # Step vehicle dynamics and controllers
        state = twin.step(dt=dt, now=now)

        # ECU simulator step: generates frames scheduled at or before `now`
        frames = ecu_sim.step(now=now, vehicle_state=state)

        for frame in frames:
            message_counts[frame.arb_id] += 1
            if not checksum_ok(frame.data):
                all_checksums_ok = False

            # In normal operation, gateway forwards accepted frames to the twin
            signals = decode(frame.arb_id, frame.data)
            if frame.arb_id in COMMAND_IDS:
                twin.accept_command(frame.arb_id, signals, now)
            else:
                twin.accept_status(frame.arb_id, signals)

        # Sample speed every 1.0 s of virtual time
        if step_idx % int(world_hz) == 0 or step_idx == total_steps - 1:
            speed_samples.append((round(now, 1), round(state.speed_kmh, 2), round(state.pedal, 1)))

    print("=" * 60)
    print("PART 2 ACCEPTANCE RESULTS (20 s Virtual Time)")
    print("=" * 60)
    print(f"Total virtual steps: {total_steps} (dt = {dt} s)")
    print(f"All frames checksum_ok: {all_checksums_ok}")
    print("\nMessage counts per ID:")
    for arb_id in KNOWN_IDS:
        mdef = DBC[arb_id]
        count = message_counts[arb_id]
        rate_hz = count / duration_s
        expected_hz = 1.0 / mdef.period_s
        print(
            f"  ID 0x{arb_id:03X} ({mdef.name:15s} {mdef.period_s*1000:3.0f}ms): "
            f"{count:5d} msgs ({rate_hz:5.1f} Hz, expected ~{expected_hz:5.1f} Hz)"
        )

    print("\nSpeed trajectory (sampled every 1s):")
    min_speed = 999.0
    min_speed_t = 0.0
    for t, spd, pedal in speed_samples:
        if spd < min_speed:
            min_speed = spd
            min_speed_t = t
        print(f"  t = {t:4.1f}s | Speed: {spd:5.2f} km/h | Driver Pedal: {pedal:4.1f}%")

    print(f"\nMinimum speed reached: {min_speed:.2f} km/h around t = {min_speed_t:.1f}s")
    print("=" * 60)

    # Assertions for verification
    assert all_checksums_ok, "Not all checksums were valid!"
    # 20ms IDs should have ~1000 msgs (between 900 and 1100)
    for arb_id in [0x100, 0x101, 0x102, 0x103, 0x105, 0x200, 0x201]:
        assert 900 <= message_counts[arb_id] <= 1100, f"ID 0x{arb_id:X} count out of range: {message_counts[arb_id]}"
    # 0x104 (100ms) should have ~200 msgs (between 180 and 220)
    assert 180 <= message_counts[0x104] <= 220, f"ID 0x104 count out of range: {message_counts[0x104]}"
    # 0x106 (10ms) should have ~2000 msgs
    assert 1850 <= message_counts[0x106] <= 2150, f"ID 0x106 count out of range: {message_counts[0x106]}"
    # 0x202 (50ms) should have ~400 msgs
    assert 370 <= message_counts[0x202] <= 430, f"ID 0x202 count out of range: {message_counts[0x202]}"
    # Speed near 72 km/h initially and recovers near 72 km/h
    assert 70.0 <= speed_samples[0][1] <= 73.0, f"Initial speed not near 72: {speed_samples[0][1]}"
    assert 70.0 <= speed_samples[-1][1] <= 73.0, f"Final speed not near 72: {speed_samples[-1][1]}"
    # Mild brake dip around t = 6-7s
    assert 60.0 <= min_speed <= 70.0, f"Brake dip not as expected: min_speed = {min_speed}"
    assert 6.0 <= min_speed_t <= 9.0, f"Brake dip occurred at unexpected time: {min_speed_t}"
    print("ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY.")


if __name__ == "__main__":
    run_acceptance()
