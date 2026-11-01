"""
CAVShield Soak Test & Backlog Diagnosis Tool.

Runs headless simulation under protection ON with NO attacks to verify:
1. 15 minutes (900 s) of simulated time soak test:
   - Prints every second: processing backlog, max burst of frames with same timestamp,
     computed rate per ID, flagged frames, mode, true speed.
   - Accurately tracks and reports the first moment a legitimate ID is flagged.
2. Supports --slow-ai 2.0 to artificially slow down AI worker to reproduce backlog/starvation.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import load_config
from src.can_simulator import COMMAND_IDS, KNOWN_IDS, CANFrame, decode, DBC
from src.detection.decision_engine import RiskLevel
from src.engine import CAVShieldEngine


def run_soak_test(
    duration_s: float = 900.0,
    dt: float = 0.005,
    slow_ai_s: float = 0.0,
    print_interval_s: float = 1.0,
) -> Dict[str, Any]:
    """Run headless simulation for duration_s with protection ON and no attacks."""
    cfg = load_config()
    engine = CAVShieldEngine(cfg=cfg)
    engine.reset()
    engine.set_protection(True)

    # If slow_ai is enabled, monkeypatch anomaly_detector.score to simulate slow AI
    if slow_ai_s > 0:
        orig_score = engine.anomaly_detector.score
        def slow_score(win: Any) -> Any:
            time.sleep(slow_ai_s)
            return orig_score(win)
        engine.anomaly_detector.score = slow_score  # type: ignore

    print(f"\n================================================================================")
    print(f"CAVSHIELD SOAK TEST: {duration_s:.1f}s Simulated Time | dt={dt*1000.0:.1f}ms | slow_ai={slow_ai_s:.1f}s")
    print(f"Protection: ON | Active Attacks: 0 (Benign Cruise)")
    print(f"================================================================================")
    print(f"{'Sim_t(s)':>8} | {'Backlog':>7} | {'MaxBurst':>8} | {'Mode':>8} | {'Speed':>6} | {'Flagged':>7} | {'Rates (Hz) [100/101/102/200]':>30}")
    print(f"--------------------------------------------------------------------------------")

    # Tracking structures
    first_flagged: Optional[Dict[str, Any]] = None
    frames_per_id: Dict[int, int] = collections.defaultdict(int)
    flagged_per_id: Dict[int, int] = collections.defaultdict(int)
    dropped_per_id: Dict[int, int] = collections.defaultdict(int)

    # Queue of frames waiting to be processed by pipeline (simulating processing backlog)
    pipeline_queue: Deque[CANFrame] = collections.deque()

    # Track frames per timestamp in this step
    step_ts_counts: Dict[float, int] = collections.defaultdict(int)
    max_burst_seen = 0

    n_steps = int(duration_s / dt)
    next_print_t = print_interval_s

    # Shared AI queue for background worker if slow_ai is active
    ai_queue: queue.Queue = queue.Queue(maxsize=100)
    ai_worker_running = True

    def slow_ai_worker() -> None:
        while ai_worker_running:
            try:
                win, ts = ai_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                engine.anomaly_detector.score(win)
            except Exception:
                pass

    ai_thread = None
    if slow_ai_s > 0:
        ai_thread = threading.Thread(target=slow_ai_worker, name="SoakSlowAIWorker", daemon=True)
        ai_thread.start()

    start_wall_t = time.perf_counter()

    for step in range(n_steps):
        t = round(step * dt, 4)

        # 1. Step world
        can_frames, v2x_msgs = engine.world.step(now=t, dt=dt, apply_to_twin=False)
        true_spd = engine.world.twin.state.speed_kmh

        # Update sensor snapshot into engines
        sensor_snapshot = engine.world.env.sense(ego_speed_kmh=true_spd, now=t)
        engine.physics_engine.set_sensor_snapshot(sensor_snapshot)
        engine.v2x_validator.set_sensor_snapshot(sensor_snapshot)

        # Count timestamp bursts
        step_ts_counts.clear()
        for f in can_frames:
            step_ts_counts[f.timestamp] += 1
        current_step_burst = max(step_ts_counts.values()) if step_ts_counts else 0
        if current_step_burst > max_burst_seen:
            max_burst_seen = current_step_burst

        # Add frames to pipeline backlog
        pipeline_queue.extend(can_frames)

        # Simulate pipeline draining up to batch_limit frames per cycle
        batch_limit = 60
        processed_this_step = 0

        while pipeline_queue and processed_this_step < batch_limit:
            frame = pipeline_queue.popleft()
            processed_this_step += 1
            arb_id = frame.arb_id
            frames_per_id[arb_id] += 1

            # Feature extractor & window
            feat_vec = engine.feature_extractor.extract(arb_id, frame.data, frame.timestamp, frame.dlc)
            full_window = engine.windower.push(
                vec=feat_vec,
                ts=frame.timestamp,
                label=frame.label,
                subtype=frame.subtype,
                raw_id=frame.arb_id,
                raw_payload=frame.data,
            )
            if full_window is not None and slow_ai_s > 0:
                try:
                    ai_queue.put_nowait((full_window, frame.timestamp))
                except queue.Full:
                    pass

            # Fast-path Gate
            gated = engine.failsafe.gate(frame, now=frame.timestamp)
            if not gated:
                dropped_per_id[arb_id] += 1
                flagged_per_id[arb_id] += 1
                if first_flagged is None:
                    first_flagged = {
                        "t": t,
                        "arb_id": f"0x{arb_id:03X}",
                        "reason": "gate dropped (quarantine or rate-limited)",
                        "rate_hz": engine.rule_engine.stream_rate_ratio(arb_id) * (1.0 / DBC[arb_id].period_s if arb_id in DBC else 10.0),
                        "backlog": len(pipeline_queue),
                        "burst": current_step_burst,
                    }
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
                if first_flagged is None:
                    first_flagged = {
                        "t": t,
                        "arb_id": f"0x{arb_id:03X}",
                        "decision": decision.level.name,
                        "reason": decision.explanation,
                        "rate_hz": engine.rule_engine.stream_rate_ratio(arb_id) * (1.0 / DBC[arb_id].period_s if arb_id in DBC else 10.0),
                        "backlog": len(pipeline_queue),
                        "burst": current_step_burst,
                    }

            if forward:
                engine.rule_engine.commit(frame)
                engine.physics_engine.update(frame, decoded)
                if arb_id in COMMAND_IDS:
                    engine.world.twin.accept_command(arb_id, decoded, t)
                elif arb_id in KNOWN_IDS:
                    engine.world.twin.accept_status(arb_id, decoded)

        # Process V2X messages
        for v2x_msg in v2x_msgs:
            verdict = engine.v2x_validator.validate(v2x_msg, now=v2x_msg.timestamp)
            dec_v2x = engine.risk_engine.fuse_v2x(v2x_msg, verdict, now=v2x_msg.timestamp)
            acc = engine.failsafe.enforce_v2x(dec_v2x, v2x_msg, now=v2x_msg.timestamp)
            if not acc and first_flagged is None:
                first_flagged = {
                    "t": t,
                    "arb_id": f"V2X-{v2x_msg.sender_id}",
                    "decision": "REJECTED",
                    "reason": "; ".join(verdict.reasons),
                    "backlog": len(pipeline_queue),
                    "burst": 0,
                }

        # Print periodic status
        if t >= next_print_t or step == (n_steps - 1):
            next_print_t = t + print_interval_s
            mode_str = engine.failsafe.current_mode
            total_flagged = sum(flagged_per_id.values())

            # Sample computed rates for key IDs
            rates_str = f"{engine.rule_engine.stream_rate_ratio(0x100)*100:.0f}/{engine.rule_engine.stream_rate_ratio(0x101)*50:.0f}/{engine.rule_engine.stream_rate_ratio(0x102)*50:.0f}/{engine.rule_engine.stream_rate_ratio(0x200)*50:.0f}"

            print(f"{t:>8.1f} | {len(pipeline_queue):>7} | {current_step_burst:>8} | {mode_str:>8} | {true_spd:>6.1f} | {total_flagged:>7} | {rates_str:>30}")

            if total_flagged > 0 and first_flagged is not None and t < 10.0:
                print(f"   >>> FIRST FLAGGED: t={first_flagged['t']:.2f}s | ID={first_flagged['arb_id']} | reason={first_flagged['reason']}")

    ai_worker_running = False
    if ai_thread:
        ai_thread.join(timeout=1.0)

    elapsed_wall = time.perf_counter() - start_wall_t
    snap = engine.snapshot()

    print(f"--------------------------------------------------------------------------------")
    print(f"Soak completed in {elapsed_wall:.2f}s wall clock ({duration_s:.1f}s simulated).")
    print(f"Final Mode: {snap.get('mode')} | Final Speed: {engine.world.twin.state.speed_kmh:.1f} km/h")
    print(f"Total Frames: {sum(frames_per_id.values())} | Flagged: {sum(flagged_per_id.values())} | Dropped: {sum(dropped_per_id.values())}")
    if first_flagged:
        print(f"FIRST FLAGGED INCIDENT:")
        for k, v in first_flagged.items():
            print(f"  {k}: {v}")
    else:
        print(f"RESULT: CLEAN SOAK - ZERO FALSE POSITIVES.")
    print(f"================================================================================\n")

    return {
        "duration_s": duration_s,
        "first_flagged": first_flagged,
        "total_frames": sum(frames_per_id.values()),
        "total_flagged": sum(flagged_per_id.values()),
        "total_dropped": sum(dropped_per_id.values()),
        "final_speed": engine.world.twin.state.speed_kmh,
        "final_mode": snap.get("mode"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CAVShield Soak Test & Backlog Diagnosis")
    parser.add_argument("--duration", type=float, default=900.0, help="Simulation duration in seconds (default: 900.0 / 15m)")
    parser.add_argument("--slow-ai", type=float, default=0.0, help="Artificially slow AI worker by N seconds per window")
    parser.add_argument("--interval", type=float, default=1.0, help="Print interval in seconds (default: 1.0)")
    args = parser.parse_args()

    run_soak_test(duration_s=args.duration, dt=0.005, slow_ai_s=args.slow_ai, print_interval_s=args.interval)


if __name__ == "__main__":
    main()
