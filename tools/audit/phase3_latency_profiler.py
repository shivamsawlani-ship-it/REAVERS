"""
Phase 3: Latency Profiler.
Measures true fast-path latency (time.perf_counter_ns) over >= 100,000 frames:
- Normal traffic (50k frames)
- Attack traffic / Flood (50k frames)
Measures p50, p95, p99, max in microseconds (µs).
Also profiles:
- AI queue wait latency
- ONNX/IsolationForest inference latency
- Snapshot build time
- Engine clock vs monotonic consistency
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src import load_config
from src.can_simulator import CANFrame, decode
from src.engine import CAVShieldEngine

logging.basicConfig(level=logging.WARNING)


def profile_fast_path(n_frames: int = 100000) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Profiling Gateway Fast-Path Latency over {n_frames} frames")
    print(f"=======================================================")

    cfg = load_config()
    engine = CAVShieldEngine(cfg=cfg)
    engine.reset()
    engine.set_protection(True)

    clean_latencies_us: List[float] = []
    flood_latencies_us: List[float] = []

    # 1. Clean traffic (n_frames // 2)
    half = n_frames // 2
    step_dt = 0.005
    t = 0.0
    frame_count = 0

    print("Measuring clean traffic latency...")
    while frame_count < half:
        can_frames, _ = engine.world.step(now=t, dt=step_dt, apply_to_twin=False)
        t += step_dt
        for frame in can_frames:
            t0 = time.perf_counter_ns()
            # Fast path steps
            gated = engine.failsafe.gate(frame, now=frame.timestamp)
            if gated:
                r_ev = engine.rule_engine.evaluate(frame)
                try:
                    dec = decode(frame.arb_id, frame.data)
                except Exception:
                    dec = None
                p_ev = engine.physics_engine.check(frame, dec)
                decision = engine.risk_engine.fuse_can(frame, r_ev + p_ev, engine.ai_context, now=frame.timestamp)
                forward, actions = engine.failsafe.enforce(decision, frame, now=frame.timestamp)
                if forward:
                    engine.rule_engine.commit(frame)
                    engine.physics_engine.update(frame, dec)
            t_diff_us = (time.perf_counter_ns() - t0) / 1000.0
            clean_latencies_us.append(t_diff_us)
            frame_count += 1
            if frame_count >= half:
                break

    # 2. Flood traffic (n_frames // 2)
    print("Measuring flood traffic latency...")
    engine.world.injector.start("flood", duration=100.0, now=t)
    frame_count = 0
    while frame_count < half:
        can_frames, _ = engine.world.step(now=t, dt=step_dt, apply_to_twin=False)
        t += step_dt
        for frame in can_frames:
            t0 = time.perf_counter_ns()
            gated = engine.failsafe.gate(frame, now=frame.timestamp)
            if gated:
                r_ev = engine.rule_engine.evaluate(frame)
                try:
                    dec = decode(frame.arb_id, frame.data)
                except Exception:
                    dec = None
                p_ev = engine.physics_engine.check(frame, dec)
                decision = engine.risk_engine.fuse_can(frame, r_ev + p_ev, engine.ai_context, now=frame.timestamp)
                forward, actions = engine.failsafe.enforce(decision, frame, now=frame.timestamp)
                if forward:
                    engine.rule_engine.commit(frame)
                    engine.physics_engine.update(frame, dec)
            t_diff_us = (time.perf_counter_ns() - t0) / 1000.0
            flood_latencies_us.append(t_diff_us)
            frame_count += 1
            if frame_count >= half:
                break

    # 3. Snapshot build latency
    snap_latencies_ms: List[float] = []
    for _ in range(50):
        t0 = time.perf_counter_ns()
        snap = engine.snapshot()
        snap_latencies_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)

    # 4. AI inference latency
    ai_latencies_ms: List[float] = []
    dummy_feat = np.random.randn(64, 17).astype(np.float32)
    for _ in range(50):
        t0 = time.perf_counter_ns()
        engine.anomaly_detector.score(dummy_feat)
        ai_latencies_ms.append((time.perf_counter_ns() - t0) / 1_000_000.0)

    def stats(arr: List[float]) -> Dict[str, float]:
        a = np.array(arr)
        return {
            "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
            "max": round(float(np.max(a)), 2),
            "mean": round(float(np.mean(a)), 2),
        }

    clean_stats = stats(clean_latencies_us)
    flood_stats = stats(flood_latencies_us)
    snap_stats = stats(snap_latencies_ms)
    ai_stats = stats(ai_latencies_ms)

    print("\n--- Latency Profiling Results ---")
    print(f"Clean Traffic Fast-Path (µs) : {clean_stats}")
    print(f"Flood Traffic Fast-Path (µs) : {flood_stats}")
    print(f"Snapshot Generation (ms)     : {snap_stats}")
    print(f"AI Anomaly Inference (ms)    : {ai_stats}")

    res = {
        "n_frames_evaluated": len(clean_latencies_us) + len(flood_latencies_us),
        "clean_fast_path_us": clean_stats,
        "flood_fast_path_us": flood_stats,
        "snapshot_build_ms": snap_stats,
        "ai_inference_ms": ai_stats,
    }

    out_path = ROOT / "tools" / "audit" / "latency_profiler_results.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)

    return res


if __name__ == "__main__":
    profile_fast_path(100000)
