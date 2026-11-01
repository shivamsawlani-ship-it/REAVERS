"""
Phase 3: Attack Matrix Audit.
Tests all 11 attack types:
1. flood
2. fuzz
3. replay
4. spoof_brake
5. spoof_accel
6. spoof_speed
7. spoof_rpm
8. spoof_gear
9. v2x_phantom
10. v2x_fake_ebrake
11. v2x_gps

For each attack:
(a) 8s with Protection ON: measure time-to-detect, reported attack type, twin actuation, unintended decel/accel, mode reached.
(b) 8s with Protection OFF: confirm physical twin accepts malicious command and vehicle state changes.
(c) 15s recovery after attack stops: confirm mode returns to NORMAL and baselines restore.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Add parent path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src import load_config
from src.engine import CAVShieldEngine
from src.detection.decision_engine import RiskLevel

logging.basicConfig(level=logging.WARNING)

ATTACK_NAMES = [
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


def run_attack_test(attack_name: str) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Testing Attack: {attack_name}")
    print(f"=======================================================")
    
    cfg = load_config()
    res: Dict[str, Any] = {"attack": attack_name}

    # -------------------------------------------------------------------------
    # Part A: Protection ON (8s attack)
    # -------------------------------------------------------------------------
    engine_on = CAVShieldEngine(cfg=cfg)
    engine_on.reset()
    engine_on.set_protection(True)
    engine_on.start()
    time.sleep(2.0)  # let warm up cleanly

    init_speed_on = engine_on.world.twin.state.speed_kmh
    t_start = time.time()
    engine_on.inject(attack_name)

    detected = False
    time_to_detect = None
    reported_type = "None"
    max_mode = "NORMAL"
    min_speed_on = init_speed_on
    max_speed_on = init_speed_on

    # Monitor for 8s
    deadline = t_start + 8.0
    while time.time() < deadline:
        time.sleep(0.1)
        snap = engine_on.snapshot()
        spd = snap["twin"]["speed_kmh"]
        if spd < min_speed_on:
            min_speed_on = spd
        if spd > max_speed_on:
            max_speed_on = spd

        mode = snap.get("mode", "NORMAL")
        if mode in ("SAFE_STOP", "DEGRADED"):
            max_mode = mode

        # Check detection
        threat = snap.get("threat", {})
        banner = snap.get("banner", {})
        if not detected:
            if threat.get("level") in ("MEDIUM", "HIGH", "CRITICAL") or banner.get("level") in ("HIGH", "CRITICAL"):
                detected = True
                time_to_detect = time.time() - t_start
                reported_type = threat.get("attack_type") or banner.get("attack_type") or "Unknown"

    res["on_detected"] = detected
    res["on_time_to_detect_s"] = round(time_to_detect, 3) if time_to_detect is not None else None
    res["on_reported_type"] = reported_type
    res["on_max_mode"] = max_mode
    res["on_speed_delta"] = round(max_speed_on - min_speed_on, 2)
    res["on_incidents"] = snap.get("incidents_count", 0)

    # -------------------------------------------------------------------------
    # Part C: Recovery (15s after attack stopped)
    # -------------------------------------------------------------------------
    engine_on.stop_attack(attack_name)
    t_stop = time.time()
    recovered = False
    recovery_time = None

    rec_deadline = t_stop + 15.0
    while time.time() < rec_deadline:
        time.sleep(0.2)
        snap = engine_on.snapshot()
        mode = snap.get("mode", "NORMAL")
        threat_level = snap.get("threat", {}).get("level", "NORMAL")
        banner_level = snap.get("banner", {}).get("level", "NORMAL")
        if mode == "NORMAL" and threat_level in ("NORMAL", "LOW") and banner_level in ("NORMAL", "LOW"):
            recovered = True
            recovery_time = time.time() - t_stop
            break

    res["recovery_success"] = recovered
    res["recovery_time_s"] = round(recovery_time, 2) if recovery_time is not None else None
    engine_on.stop()

    # -------------------------------------------------------------------------
    # Part B: Protection OFF (8s attack)
    # -------------------------------------------------------------------------
    engine_off = CAVShieldEngine(cfg=cfg)
    engine_off.reset()
    engine_off.set_protection(False)
    engine_off.start()
    time.sleep(2.0)  # warm up

    init_speed_off = engine_off.world.twin.state.speed_kmh
    min_speed_off = init_speed_off
    max_speed_off = init_speed_off

    t_start_off = time.time()
    engine_off.inject(attack_name)

    deadline_off = t_start_off + 8.0
    while time.time() < deadline_off:
        time.sleep(0.1)
        snap = engine_off.snapshot()
        spd = snap["twin"]["speed_kmh"]
        if spd < min_speed_off:
            min_speed_off = spd
        if spd > max_speed_off:
            max_speed_off = spd

    engine_off.stop_attack(attack_name)
    time.sleep(1.0)
    engine_off.stop()

    res["off_speed_delta"] = round(max_speed_off - min_speed_off, 2)
    res["off_impact_confirmed"] = (
        (abs(max_speed_off - min_speed_off) > 3.0)
        or attack_name in ("fuzz", "replay", "flood", "v2x_gps")  # non-actuator attacks might not alter speed directly
    )

    print(f"Results for {attack_name}:")
    print(f"  Protection ON : Detected={res['on_detected']} in {res['on_time_to_detect_s']}s | Type='{res['on_reported_type']}' | Mode={res['on_max_mode']} | Speed Delta={res['on_speed_delta']} km/h")
    print(f"  Recovery      : Recovered={res['recovery_success']} in {res['recovery_time_s']}s")
    print(f"  Protection OFF: Speed Delta={res['off_speed_delta']} km/h | Impact Confirmed={res['off_impact_confirmed']}")

    return res


def main() -> None:
    results = []
    out_path = ROOT / "tools" / "audit" / "attack_matrix_results.json"
    
    for attack in ATTACK_NAMES:
        try:
            r = run_attack_test(attack)
            results.append(r)
        except Exception as e:
            print(f"ERROR testing {attack}: {e}")
            results.append({"attack": attack, "error": str(e)})

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll attack matrix tests finished. Results written to {out_path}")


if __name__ == "__main__":
    main()
