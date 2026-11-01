"""
Diagnostic script for recovery timing in test_headless_short_soak_and_recovery
"""
import time
import sys
from pathlib import Path

ROOT = Path("e:/final product/cavshield")
sys.path.insert(0, str(ROOT))

from src import load_config
from src.engine import CAVShieldEngine

def run_test():
    config = load_config()
    engine = CAVShieldEngine(config)
    engine.failsafe.fast_recovery = True
    engine.start()

    try:
        time.sleep(2.0)
        snap = engine.snapshot()
        print(f"Initial mode: {snap['mode']}, incident_id: {snap['failsafe']['incident_id']}")

        engine.inject("spoof_brake")
        time.sleep(3.0)
        snap_mid = engine.snapshot()
        print(f"Mid-attack mode: {snap_mid['mode']}, incidents: {snap_mid['failsafe']['incident_id']}")

        engine.stop_attack("spoof_brake")
        t_stop = time.time()
        print(f"Attack stopped at t=0. Watching recovery for 6 seconds...")

        while time.time() - t_stop < 6.0:
            elapsed = time.time() - t_stop
            s = engine.snapshot()
            print(f"t={elapsed:.2f}s -> Mode: {s['mode']}, Threat: {s['threat']['attack_type']}, Actions: {s['failsafe']['actions'][-1] if s['failsafe']['actions'] else None}")
            if s['mode'] == "NORMAL":
                print(f"RECOVERED TO NORMAL at t={elapsed:.2f}s!")
                break
            time.sleep(0.5)

    finally:
        engine.stop()

if __name__ == "__main__":
    run_test()

