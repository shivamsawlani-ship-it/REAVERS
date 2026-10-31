"""
CAVShield Live Simulation and Scenario Runner.

Usage:
    python run_simulation.py --scenario demo
    python run_simulation.py --scenario brake
    python run_simulation.py --scenario flood
    python run_simulation.py --scenario replay
    python run_simulation.py --scenario v2x
    python run_simulation.py --serve
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Dict

from src import load_config
from src.engine import CAVShieldEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cavshield.sim")


def print_banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(f" {text}")
    print("=" * 78)


def print_status_line(engine: CAVShieldEngine, caption: str) -> None:
    snap = engine.snapshot()
    t = snap["t"]
    mode = snap["mode"]
    prot = snap["protection"]
    speed = snap["twin"]["speed_kmh"]
    decel = snap["twin"]["applied_decel"]
    unsafe_s = snap["twin"]["unsafe_seconds"]
    active = snap["active_attacks"]
    active_str = ", ".join(active) if active else "None"
    threat_lvl = snap["threat"]["level"]
    atk_type = snap["threat"]["attack_type"]
    fpr = snap["metrics"]["fpr"]
    f1 = snap["metrics"]["f1"]

    print(
        f"[{t:5.1f}s | {mode:9}] Spd:{speed:5.1f} km/h | Decel:{decel:4.1f} m/s² | "
        f"Threat:{threat_lvl:<8} ({atk_type}) | Unsafe:{unsafe_s:4.1f}s | "
        f"Prot:{'ON' if prot else 'OFF'} | FPR:{fpr:.3f} | {caption}"
    )


def run_demo_scenario(engine: CAVShieldEngine) -> None:
    """Execute the end-to-end 60-second interactive demonstration scenario."""
    print_banner("STARTING CAVSHIELD LIVE DEMONSTRATION SCENARIO (60s)")
    engine.start()

    try:
        # Phase 1: 0 - 8s Normal Baseline
        print_banner("PHASE 1: Normal Baseline Cruising (t = 0s to 8s)")
        print("Vehicle cruising at 72 km/h. Verifying low false-positive rate on normal CAN/V2X traffic...")
        for _ in range(8):
            time.sleep(1.0)
            print_status_line(engine, "Normal cruise traffic")

        # Phase 2: 8 - 15s Spoofed Brake Command Attack (Protection ON)
        print_banner("PHASE 2: Injected Spoofed Brake Command (t = 8s to 15s) [PROTECTION ON]")
        print("Adversary injects high-priority 0x200 BRAKE_COMMAND claiming 7.0 m/s² decel with pedal=0%...")
        engine.inject("spoof_brake", duration=6.5)
        for _ in range(7):
            time.sleep(1.0)
            snap = engine.snapshot()
            blocked_cnt = snap["failsafe"]["rejected_commands"].get(0x200, 0)
            print_status_line(engine, f"Attack ACTIVE -> Gateway blocked {blocked_cnt} spoofed frames")

        # Phase 3: 15 - 27s Timed Recovery
        print_banner("PHASE 3: Safe Mode & Timed Auto-Recovery (t = 15s to 27s)")
        print("Attack ceased. Gateway initiates timed step-by-step recovery (SAFE MODE -> DEGRADED -> ALERT -> NORMAL)...")
        for _ in range(12):
            time.sleep(1.0)
            print_status_line(engine, "Auto-recovery monitoring")

        # Phase 4: 27 - 34s Phantom V2X Warning
        print_banner("PHASE 4: Phantom V2X Road Hazard Spoofing (t = 27s to 34s)")
        print("Compromised vehicle broadcasts fake COLLISION_WARNING not confirmed by Radar/LiDAR...")
        engine.inject("v2x_phantom", duration=6.0)
        for _ in range(7):
            time.sleep(1.0)
            rej_v2x = engine.snapshot()["failsafe"]["rejected_v2x"]
            print_status_line(engine, f"V2X cross-sensor check -> Rejected {rej_v2x} unconfirmed warnings")

        # Phase 5: 34 - 43s Legitimate Road Obstacle & Autonomous AEB
        print_banner("PHASE 5: Real Road Obstacle Spawned (t = 34s to 43s)")
        print("Spawning verified obstacle at 95m in ego lane. V2X warning matches multi-sensor tracks...")
        engine.spawn_real_obstacle(distance=95.0, lifetime=9.0)
        for _ in range(9):
            time.sleep(1.0)
            snap = engine.snapshot()
            aeb_on = snap["twin"]["aeb"]
            print_status_line(engine, f"Legitimate AEB {'ACTIVE (Verified)' if aeb_on else 'Standby'} -> NOT blocked!")

        # Phase 6: 43 - 50s CAN Flooding Attack
        print_banner("PHASE 6: CAN Bus Flooding Attack (t = 43s to 50s)")
        print("Adversary floods bus with 0x000 frames at 700 Hz to exhaust ECU buffers...")
        engine.inject("flood", duration=6.0)
        for _ in range(7):
            time.sleep(1.0)
            rl_ids = engine.snapshot()["failsafe"]["rate_limited"]
            print_status_line(engine, f"Token bucket rate-limiter shedding flood stream: {rl_ids}")

        # Phase 7: 50 - 55s Replay Attack
        print_banner("PHASE 7: Historical CAN Replay Attack (t = 50s to 55s)")
        print("Adversary replays historical frames with stale alive counters and duplicate payloads...")
        engine.inject("replay", duration=4.5)
        for _ in range(5):
            time.sleep(1.0)
            print_status_line(engine, "Freshness & counter check active")

        # Phase 8: 55 - 61s Contrast Demonstration (Protection OFF)
        print_banner("PHASE 8: CONTRAST DEMO - Spoofed Brake with PROTECTION OFF (t = 55s to 61s)")
        print("Disabling gateway protection (observe-only). Re-injecting spoofed brake command...")
        engine.set_protection(False)
        engine.inject("spoof_brake", duration=5.0)
        for _ in range(6):
            time.sleep(1.0)
            print_status_line(engine, "UNGUARDED: Hostile frames reach actuators directly -> Unintended Braking!")

        # Restore protection
        engine.set_protection(True)

    finally:
        # Print Final Performance Table
        snap = engine.snapshot()
        m = snap["metrics"]
        print_banner("FINAL CAVSHIELD EVALUATION METRICS SUMMARY")
        print(f"Total Frames Processed:    {m['counts']['tp'] + m['counts']['fp'] + m['counts']['tn'] + m['counts']['fn']}")
        print(f"Overall Accuracy:          {m['accuracy'] * 100:.2f}%")
        print(f"Precision:                 {m['precision'] * 100:.2f}%")
        print(f"Recall:                    {m['recall'] * 100:.2f}%")
        print(f"F1-Score:                  {m['f1']:.4f}")
        print(f"False Positive Rate (FPR): {m['fpr'] * 100:.2f}% (Normal traffic falsely blocked)")
        print("-" * 78)
        print(f"Detection Latency:         Mean: {m['detection_latency_ms']['mean']:.2f} ms | p95: {m['detection_latency_ms']['p95']:.2f} ms | Max: {m['detection_latency_ms']['max']:.2f} ms")
        print(f"Fail-Safe Gateway Latency: Mean: {m['failsafe_latency_ms']['mean']:.2f} ms | p95: {m['failsafe_latency_ms']['p95']:.2f} ms")
        print(f"AI Inference Latency:      Mean: {m['inference_ms']['mean']:.2f} ms | p95: {m['inference_ms']['p95']:.2f} ms")
        print("=" * 78)
        print("\n>>> [NOTE: THESE ARE SIMULATOR-ONLY METRICS] <<<\n")

        # Verify Schema completeness
        required_keys = [
            "t", "mode", "protection", "backend", "banner", "active_attacks",
            "twin", "threat", "scores", "attack_markers", "can", "v2x",
            "failsafe", "incidents", "metrics", "demo",
        ]
        missing = [k for k in required_keys if k not in snap]
        if missing:
            print(f"WARNING: Snapshot missing keys: {missing}")
        else:
            print("Snapshot Schema Verification: All 16 root schema keys present and verified valid.")

        engine.stop()


def run_scenario(scenario: str, duration: float, protection: bool, use_ai: bool, serve: bool) -> None:
    """Run specified test scenario or start daemon server."""
    overrides = {
        "simulation": {"protection_enabled": protection},
        "risk": {"use_ai": use_ai},
    }
    cfg = load_config(overrides=overrides)
    engine = CAVShieldEngine(cfg=cfg)

    if serve:
        print_banner("CAVSHIELD ENGINE RUNNING IN SERVER MODE (Press Ctrl+C to stop)")
        print(f"Virtual CAN bus active. UDP remote command listener active on 127.0.0.1:{engine.udp_port}")
        engine.start()
        try:
            while True:
                time.sleep(1.0)
                print_status_line(engine, "Server running")
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            engine.stop()
        return

    if scenario == "demo":
        run_demo_scenario(engine)
        return

    # Specific scenarios
    print_banner(f"RUNNING SCENARIO: {scenario.upper()} (Duration: {duration}s)")
    engine.start()
    try:
        if scenario in ("brake", "all"):
            print("Triggering spoof_brake attack...")
            engine.inject("spoof_brake", duration=duration * 0.7)
        elif scenario in ("flood", "all"):
            print("Triggering flood attack...")
            engine.inject("flood", duration=duration * 0.7)
        elif scenario in ("replay", "all"):
            print("Triggering replay attack...")
            engine.inject("replay", duration=duration * 0.7)
        elif scenario in ("v2x", "all"):
            print("Triggering v2x_phantom attack...")
            engine.inject("v2x_phantom", duration=duration * 0.7)

        t_end = time.time() + duration
        while time.time() < t_end:
            time.sleep(1.0)
            print_status_line(engine, f"Scenario {scenario} active")

    finally:
        snap = engine.snapshot()
        print_banner("SCENARIO COMPLETE - METRICS")
        print(f"Accuracy: {snap['metrics']['accuracy'] * 100:.2f}% | FPR: {snap['metrics']['fpr'] * 100:.2f}% | F1: {snap['metrics']['f1']:.4f}")
        engine.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="CAVShield Live Cyber-Physical Simulation Runner.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="demo",
        choices=["demo", "brake", "flood", "replay", "v2x", "all"],
        help="Simulation scenario to execute (default: demo)",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="Scenario duration in seconds")
    parser.add_argument("--serve", action="store_true", help="Run indefinitely and listen for CLI attacks")
    parser.add_argument("--no-protection", action="store_true", help="Disable fail-safe gateway protection")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI inference in risk fusion")
    args = parser.parse_args()

    run_scenario(
        scenario=args.scenario,
        duration=args.duration,
        protection=not args.no_protection,
        use_ai=not args.no_ai,
        serve=args.serve,
    )


if __name__ == "__main__":
    main()

