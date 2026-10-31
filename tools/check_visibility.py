"""
Diagnostic Tool: Check visibility of all 11 cyber-physical attacks in CAVShield streams.

Polls engine.snapshot() every 100ms during a 3-second attack window and instruments
the engine pipeline to identify where messages are lost.
"""

from __future__ import annotations

import collections
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import load_config
from src.attack_injector import AttackInjector
from src.can_simulator import CANFrame
from src.engine import CAVShieldEngine
from src.v2x_simulator import V2XMessage

ATTACKS = [
    "flood",
    "fuzz",
    "replay",
    "spoof_brake",
    "spoof_accel",
    "spoof_speed",
    "spoof_rpm",
    "spoof_gear",
    "v2x_phantom",
    "v2x_fake_ebrake",
    "v2x_gps",
]


def run_visibility_check() -> Dict[str, Any]:
    print("=" * 100)
    print("              CAVSHIELD ATTACK VISIBILITY DIAGNOSTIC RUNNER")
    print("=" * 100)

    cfg = load_config()
    results = {}

    for attack_name in ATTACKS:
        is_v2x = attack_name.startswith("v2x")

        # Initialize engine headless
        engine = CAVShieldEngine(cfg=cfg)
        engine.set_protection(True)

        # Counters for instrumentation
        counters = {
            "produced": 0,
            "received": 0,
            "evaluated": 0,
            "recorded": 0,
            "snapshot_non_normal_seen": False,
            "non_normal_count_seen": 0,
            "sample_snapshot_rows": [],
        }

        # Instrument AttackInjector.tick
        orig_injector_tick = engine.world.injector.tick
        def hooked_injector_tick(now: float, ctx: Any = None):
            can_f, v2x_m = orig_injector_tick(now, ctx)
            if is_v2x:
                for m in v2x_m:
                    if getattr(m, "subtype", "") == attack_name or attack_name in getattr(m, "msg_id", "").lower():
                        counters["produced"] += 1
            else:
                for f in can_f:
                    if getattr(f, "subtype", "") == attack_name or getattr(f, "label", "") != "Normal":
                        counters["produced"] += 1
            return can_f, v2x_m
        engine.world.injector.tick = hooked_injector_tick

        # Instrument CAN pipeline
        orig_gate = engine.failsafe.gate
        def hooked_gate(frame: CANFrame, now: float = 0.0):
            if not is_v2x and (getattr(frame, "subtype", "") == attack_name or getattr(frame, "label", "") != "Normal"):
                counters["received"] += 1
            return orig_gate(frame, now)
        engine.failsafe.gate = hooked_gate

        orig_rule_eval = engine.rule_engine.evaluate
        def hooked_rule_eval(frame: CANFrame):
            if not is_v2x and (getattr(frame, "subtype", "") == attack_name or getattr(frame, "label", "") != "Normal"):
                counters["evaluated"] += 1
            return orig_rule_eval(frame)
        engine.rule_engine.evaluate = hooked_rule_eval

        orig_rec_can = engine._record_can_snapshot
        def hooked_rec_can(frame: CANFrame, verdict: str, reason: str, *args, **kwargs):
            if not is_v2x and (getattr(frame, "subtype", "") == attack_name or getattr(frame, "label", "") != "Normal" or verdict != "normal"):
                counters["recorded"] += 1
            return orig_rec_can(frame, verdict, reason, *args, **kwargs)
        engine._record_can_snapshot = hooked_rec_can

        # Instrument V2X pipeline
        orig_v2x_val = engine.v2x_validator.validate
        def hooked_v2x_val(msg: V2XMessage, snapshot: Any = None, now: float = 0.0):
            if is_v2x and (getattr(msg, "subtype", "") == attack_name or attack_name in getattr(msg, "msg_id", "").lower()):
                counters["received"] += 1
                counters["evaluated"] += 1
            return orig_v2x_val(msg, snapshot, now)
        engine.v2x_validator.validate = hooked_v2x_val

        orig_rec_v2x = engine._record_v2x_snapshot if hasattr(engine, "_record_v2x_snapshot") else None
        # In engine._process_v2x_queue, inspect how recent_v2x is recorded
        orig_enforce_v2x = engine.failsafe.enforce_v2x
        def hooked_enforce_v2x(decision: Any, msg: V2XMessage, now: float = 0.0):
            if is_v2x and (getattr(msg, "subtype", "") == attack_name or attack_name in getattr(msg, "msg_id", "").lower()):
                counters["recorded"] += 1
            return orig_enforce_v2x(decision, msg, now)
        engine.failsafe.enforce_v2x = hooked_enforce_v2x

        # Start engine
        engine.start()
        time.sleep(0.5)  # Allow baseline

        # Inject attack for 3.0 seconds
        engine.inject(attack_name, duration=3.0)

        # Poll snapshot every 100 ms for 3.0 seconds (30 polls)
        polls = 30
        for _ in range(polls):
            time.sleep(0.1)
            snap = engine.snapshot()

            if is_v2x:
                v2x_rows = snap.get("v2x", [])
                for r in v2x_rows:
                    if r.get("status") in ("rejected", "suspicious"):
                        counters["snapshot_non_normal_seen"] = True
                        counters["non_normal_count_seen"] += 1
                        if len(counters["sample_snapshot_rows"]) < 2:
                            counters["sample_snapshot_rows"].append(r)
            else:
                can_rows = snap.get("can", [])
                for r in can_rows:
                    if r.get("verdict") in ("blocked", "suspicious"):
                        counters["snapshot_non_normal_seen"] = True
                        counters["non_normal_count_seen"] += 1
                        if len(counters["sample_snapshot_rows"]) < 2:
                            counters["sample_snapshot_rows"].append(r)

        engine.stop()
        time.sleep(0.2)

        results[attack_name] = counters

    # Print Formatted Table
    print("\n" + "=" * 105)
    header = f"{'Attack Name':<16} | {'Domain':<6} | {'Produced':<9} | {'Received':<9} | {'Evaluated':<9} | {'Recorded':<9} | {'In Snapshot?':<12} | {'Non-Normal Rows'}"
    print(header)
    print("-" * 105)

    for atk, data in results.items():
        domain = "V2X" if atk.startswith("v2x") else "CAN"
        seen_str = "YES" if data["snapshot_non_normal_seen"] else "NO (LOST)"
        row = (
            f"{atk:<16} | {domain:<6} | {data['produced']:<9} | {data['received']:<9} | "
            f"{data['evaluated']:<9} | {data['recorded']:<9} | {seen_str:<12} | {data['non_normal_count_seen']}"
        )
        print(row)

    print("=" * 105 + "\n")
    return results


if __name__ == "__main__":
    run_visibility_check()
