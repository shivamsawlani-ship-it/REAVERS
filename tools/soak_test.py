"""
CAVShield Soak Test & Lockout Diagnosis Tool.

Runs headless simulation under protection ON to verify:
1. 120s soak test with NO attacks:
   - Zero false positives, zero dropped frames, mode remains NORMAL.
   - Cruise speed remains stable (within 3 km/h of 72 km/h target).
   - Stateful check rejection durations tracked.
2. Attack recovery test:
   - 10s attack (spoof_rpm, spoof_speed, spoof_brake) followed by 20s recovery.
   - Verifies system returns to NORMAL mode and forwards legitimate frames within 3s.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import load_config
from src.can_simulator import COMMAND_IDS, KNOWN_IDS, CANFrame, decode
from src.detection.decision_engine import RiskLevel
from src.engine import CAVShieldEngine


def run_soak_120s(engine: CAVShieldEngine, duration_s: float = 120.0, dt: float = 0.005) -> Dict[str, Any]:
    """Run 120s headless simulation with NO attacks and detailed diagnostics."""
    engine.reset()
    engine.set_protection(True)
    engine.failsafe.fast_recovery = True

    frames_per_id: Dict[int, int] = collections.defaultdict(int)
    flagged_per_id: Dict[int, int] = collections.defaultdict(int)
    dropped_per_id: Dict[int, int] = collections.defaultdict(int)

    # Stateful check rejection tracking: check_name -> [current_streak_start_t, max_duration_s, total_duration_s]
    stateful_reject_durations: Dict[str, float] = collections.defaultdict(float)
    stateful_reject_start: Dict[str, Optional[float]] = collections.defaultdict(lambda: None)

    # Track evidence sources
    check_failures_count: Dict[str, int] = collections.defaultdict(int)

    min_speed = 999.0
    max_speed = -999.0

    n_steps = int(duration_s / dt)
    print(f"\n[SOAK TEST] Running {duration_s:.1f}s headless soak test (dt={dt*1000.0:.1f}ms, {n_steps} steps)...")

    for step in range(n_steps):
        t = round(step * dt, 4)

        # 1. Step world
        can_frames, v2x_msgs = engine.world.step(now=t, dt=dt, apply_to_twin=False)

        true_spd = engine.world.twin.state.speed_kmh
        if true_spd < min_speed:
            min_speed = true_spd
        if true_spd > max_speed:
            max_speed = true_spd

        # Update sensors
        sensor_snapshot = engine.world.env.sense(ego_speed_kmh=true_spd, now=t)
        engine.physics_engine.set_sensor_snapshot(sensor_snapshot)
        engine.v2x_validator.set_sensor_snapshot(sensor_snapshot)

        # Active stateful checks active in this step
        active_rejects_in_step = set()

        # 2. Process frames
        for frame in can_frames:
            arb_id = frame.arb_id
            frames_per_id[arb_id] += 1

            # Update feature extractor and window
            feat_vec = engine.feature_extractor.extract(arb_id, frame.data, frame.timestamp, frame.dlc)
            full_window = engine.windower.push(
                vec=feat_vec,
                ts=frame.timestamp,
                label=frame.label,
                subtype=frame.subtype,
                raw_id=frame.arb_id,
                raw_payload=frame.data,
            )

            # Gate
            gated = engine.failsafe.gate(frame, now=frame.timestamp)
            if not gated:
                dropped_per_id[arb_id] += 1
                flagged_per_id[arb_id] += 1
                continue

            # Deterministic rules
            rule_ev = engine.rule_engine.evaluate(frame)

            # Physics
            try:
                decoded = decode(frame.arb_id, frame.data)
            except Exception:
                decoded = None
            phys_ev = engine.physics_engine.check(frame, decoded)

            all_ev = rule_ev + phys_ev
            for ev in all_ev:
                check_failures_count[ev.name] += 1
                active_rejects_in_step.add(ev.name)

            # Risk fusion
            decision = engine.risk_engine.fuse_can(
                frame=frame,
                evidences=all_ev,
                ai_ctx=engine.ai_context,
                now=frame.timestamp,
            )

            # Enforce
            forward, actions = engine.failsafe.enforce(decision, frame, now=frame.timestamp)

            if not forward:
                dropped_per_id[arb_id] += 1
            if decision.level >= RiskLevel.MEDIUM or not forward:
                flagged_per_id[arb_id] += 1

            if forward:
                engine.rule_engine.commit(frame)
                engine.physics_engine.update(frame, decoded)
                if arb_id in COMMAND_IDS:
                    engine.world.twin.accept_command(arb_id, decoded, t)
                elif arb_id in KNOWN_IDS:
                    engine.world.twin.accept_status(arb_id, decoded)

        # Track stateful reject durations
        for chk in ["COUNTER", "TIMING", "RATE_FLOOD", "SPEED_RATE_IMPLAUSIBLE", "SPEED_WHEEL_MISMATCH", "RPM_INCONSISTENCY", "IMPLAUSIBLE_GEAR"]:
            if chk in active_rejects_in_step:
                if stateful_reject_start[chk] is None:
                    stateful_reject_start[chk] = t
                stateful_reject_durations[chk] = max(
                    stateful_reject_durations[chk],
                    t - stateful_reject_start[chk]
                )
            else:
                stateful_reject_start[chk] = None

    snap = engine.snapshot()
    incidents = snap.get("failsafe", {}).get("incidents", [])

    return {
        "duration_s": duration_s,
        "mode": snap.get("mode"),
        "min_speed": min_speed,
        "max_speed": max_speed,
        "final_speed": engine.world.twin.state.speed_kmh,
        "total_frames": sum(frames_per_id.values()),
        "total_flagged": sum(flagged_per_id.values()),
        "total_dropped": sum(dropped_per_id.values()),
        "frames_per_id": dict(frames_per_id),
        "flagged_per_id": dict(flagged_per_id),
        "dropped_per_id": dict(dropped_per_id),
        "incidents_count": len(incidents),
        "recent_incidents": incidents[-5:] if incidents else [],
        "check_failures": dict(check_failures_count),
        "stateful_durations": dict(stateful_reject_durations),
    }


def run_attack_recovery(engine: CAVShieldEngine, attack_name: str, attack_dur: float = 10.0, recovery_dur: float = 20.0, dt: float = 0.005) -> Dict[str, Any]:
    """Test attack detection and recovery back to NORMAL within 3 seconds."""
    engine.reset()
    engine.set_protection(True)
    engine.failsafe.fast_recovery = True

    total_dur = attack_dur + recovery_dur
    n_steps = int(total_dur / dt)
    attack_started = False
    attack_stopped = False

    attack_detected = False
    recovery_time_s = None
    final_mode = None

    for step in range(n_steps):
        t = round(step * dt, 4)

        if not attack_started and t >= 1.0:
            engine.world.injector.start(attack_name, duration=attack_dur, now=t)
            attack_started = True

        if attack_started and not attack_stopped and t >= (1.0 + attack_dur):
            engine.world.injector.stop(attack_name)
            attack_stopped = True
            t_stop = t

        can_frames, v2x_msgs = engine.world.step(now=t, dt=dt, apply_to_twin=False)
        true_spd = engine.world.twin.state.speed_kmh
        sensor_snapshot = engine.world.env.sense(ego_speed_kmh=true_spd, now=t)
        engine.physics_engine.set_sensor_snapshot(sensor_snapshot)
        engine.v2x_validator.set_sensor_snapshot(sensor_snapshot)

        for frame in can_frames:
            feat_vec = engine.feature_extractor.extract(frame.arb_id, frame.data, frame.timestamp, frame.dlc)
            engine.windower.push(
                vec=feat_vec,
                ts=frame.timestamp,
                label=frame.label,
                subtype=frame.subtype,
                raw_id=frame.arb_id,
                raw_payload=frame.data,
            )

            gated = engine.failsafe.gate(frame, now=frame.timestamp)
            if not gated:
                attack_detected = True
                continue

            rule_ev = engine.rule_engine.evaluate(frame)
            try:
                decoded = decode(frame.arb_id, frame.data)
            except Exception:
                decoded = None
            phys_ev = engine.physics_engine.check(frame, decoded)

            decision = engine.risk_engine.fuse_can(
                frame=frame,
                evidences=rule_ev + phys_ev,
                ai_ctx=engine.ai_context,
                now=frame.timestamp,
            )

            if decision.level >= RiskLevel.HIGH:
                attack_detected = True

            forward, actions = engine.failsafe.enforce(decision, frame, now=frame.timestamp)
            if not forward:
                attack_detected = True

            if forward:
                engine.rule_engine.commit(frame)
                engine.physics_engine.update(frame, decoded)
                if frame.arb_id in COMMAND_IDS:
                    engine.world.twin.accept_command(frame.arb_id, decoded, t)
                elif frame.arb_id in KNOWN_IDS:
                    engine.world.twin.accept_status(frame.arb_id, decoded)

        # Check recovery after attack stop
        curr_mode = engine.failsafe.current_mode
        if attack_stopped and curr_mode == "NORMAL" and recovery_time_s is None:
            recovery_time_s = t - (1.0 + attack_dur)

    final_mode = engine.failsafe.current_mode

    return {
        "attack": attack_name,
        "attack_detected": attack_detected,
        "recovery_time_s": recovery_time_s,
        "final_mode": final_mode,
        "recovered": (final_mode == "NORMAL" and recovery_time_s is not None and recovery_time_s <= 3.0),
    }


def main() -> None:
    # In soak test with NO attacks, cruising speed is tested (driver_brake_peak_pct=0.0)
    cfg = load_config(overrides={"simulation": {"driver_brake_peak_pct": 0.0}})
    engine = CAVShieldEngine(cfg=cfg)

    print("=" * 78)
    print(" CAVSHIELD DIAGNOSTIC SOAK & RECOVERY BENCHMARK")
    print("=" * 78)

    # 1. 120s Soak Test (Zero attacks)
    res_soak = run_soak_120s(engine, duration_s=120.0)

    print("\n" + "-" * 78)
    print(" 120-SECOND SOAK TEST RESULTS (NO ATTACKS)")
    print("-" * 78)
    print(f"Final Mode:      {res_soak['mode']}")
    print(f"Speed Range:     min={res_soak['min_speed']:.1f} km/h, max={res_soak['max_speed']:.1f} km/h (final={res_soak['final_speed']:.1f} km/h)")
    print(f"Total Frames:    {res_soak['total_frames']}")
    print(f"Flagged Frames:  {res_soak['total_flagged']}")
    print(f"Dropped Frames:  {res_soak['total_dropped']}")
    print(f"Total Incidents: {res_soak['incidents_count']}")
    print("-" * 78)

    print(f"{'CAN ID':<10} {'Frames':<10} {'Flagged':<10} {'Dropped':<10}")
    print("-" * 45)
    for arb_id in sorted(res_soak["frames_per_id"].keys()):
        f_tot = res_soak["frames_per_id"][arb_id]
        f_flg = res_soak["flagged_per_id"].get(arb_id, 0)
        f_drp = res_soak["dropped_per_id"].get(arb_id, 0)
        print(f"0x{arb_id:03X}      {f_tot:<10} {f_flg:<10} {f_drp:<10}")

    print("\nStateful Rejection Durations:")
    for chk, dur in sorted(res_soak["stateful_durations"].items()):
        print(f"  {chk:<25}: max continuous rejection = {dur:.2f} s")

    print("\nCheck Failures Encountered:")
    for chk, cnt in sorted(res_soak["check_failures"].items()):
        print(f"  {chk:<25}: {cnt} frames")

    # 2. Attack Recovery Tests
    print("\n" + "=" * 78)
    print(" ATTACK RECOVERY BENCHMARKS (10s attack -> 20s recovery)")
    print("=" * 78)
    for atk in ["spoof_rpm", "spoof_speed", "spoof_brake"]:
        res_rec = run_attack_recovery(engine, atk)
        status_str = "RECOVERED" if res_rec["recovered"] else "FAILED_RECOVERY"
        rec_time_str = f"{res_rec['recovery_time_s']:.2f}s" if res_rec['recovery_time_s'] is not None else "never"
        print(
            f"Attack: {atk:<14} | Detected: {res_rec['attack_detected']} | "
            f"Final Mode: {res_rec['final_mode']:<9} | Recovery Time: {rec_time_str:<8} | [{status_str}]"
        )


if __name__ == "__main__":
    main()
