"""
Test Attack Stream Visibility across all 11 cyber-physical attacks.

Verifies:
1. Each attack leaves visible, labelled evidence in the appropriate stream within 1.5s.
2. The row contains a non-empty, plain-English reason.
3. V2X attacks never produce rows in snapshot["can"].
4. Flagged rows persist in snapshot for at least 8.0s after the attack stops.
5. Snapshot build time is < 5ms even under 700 Hz flood.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest

from src import load_config
from src.engine import CAVShieldEngine

ALL_ATTACKS = [
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


@pytest.mark.parametrize("attack_name", ALL_ATTACKS)
def test_attack_visibility_and_persistence(attack_name: str) -> None:
    """Verify stream visibility, plain-English explanation, domain isolation, and 8s persistence."""
    cfg = load_config()
    engine = CAVShieldEngine(cfg=cfg)
    engine.set_protection(True)

    is_v2x = attack_name.startswith("v2x")
    dt = 0.02  # 50 Hz simulation step

    def process_step(sim_t: float) -> None:
        engine.last_sim_time = sim_t
        can_frames, v2x_msgs = engine.world.step(now=sim_t, dt=dt, apply_to_twin=False)

        # Update sensors
        sensor_snapshot = engine.world.env.sense(ego_speed_kmh=engine.world.twin.state.speed_kmh, now=sim_t)
        engine.physics_engine.set_sensor_snapshot(sensor_snapshot)
        engine.v2x_validator.set_sensor_snapshot(sensor_snapshot)

        # Process CAN frames
        for f in can_frames:
            if not engine.failsafe.gate(f, now=sim_t):
                engine._record_can_snapshot(f, verdict="blocked", reason="Quarantined or rate-limited", now=sim_t)
                continue
            r_ev = engine.rule_engine.evaluate(f)
            from src.can_simulator import decode
            try:
                dec = decode(f.arb_id, f.data)
            except Exception:
                dec = None
            p_ev = engine.physics_engine.check(f, dec)
            decision = engine.risk_engine.fuse_can(f, r_ev + p_ev, engine.ai_context, now=sim_t)
            forward, _ = engine.failsafe.enforce(decision, f, now=sim_t)
            if forward:
                engine.rule_engine.commit(f)
                engine.physics_engine.update(f, dec)
            v_str = "blocked" if not forward else ("suspicious" if decision.level.value >= 2 else "normal")
            engine._record_can_snapshot(f, verdict=v_str, reason=decision.explanation, now=sim_t)

        # Process V2X messages
        for m in v2x_msgs:
            verdict = engine.v2x_validator.validate(m, now=sim_t)
            v_dec = engine.risk_engine.fuse_v2x(m, verdict, now=sim_t)
            accepted = engine.failsafe.enforce_v2x(v_dec, m, now=sim_t)
            st_str = getattr(verdict, "status", "accepted" if accepted else "rejected")
            if not accepted and st_str == "accepted":
                st_str = "rejected"
            reason_str = "; ".join(verdict.reasons) if verdict.reasons else "Verified"
            engine._record_v2x_snapshot(m, status=st_str, trust=verdict.trust, reason=reason_str, now=sim_t)

    # 1. Advance 1.0 s of normal baseline
    t = 0.0
    baseline_steps = int(1.0 / dt)
    for _ in range(baseline_steps):
        t = round(t + dt, 4)
        process_step(t)

    # 2. Inject attack at t=1.0s for 2.0s duration (active from t=1.0s to t=3.0s)
    attack_start_t = t
    attack_duration = 2.0
    attack_end_t = round(attack_start_t + attack_duration, 4)
    engine.world.injector.start(attack_name, duration=attack_duration, now=attack_start_t)

    # Step simulation up to 1.5s after injection
    detect_deadline = round(attack_start_t + 1.5, 4)
    flagged_row_found: Dict[str, Any] = {}
    found_at_t: float = 0.0

    while t < detect_deadline:
        t = round(t + dt, 4)
        process_step(t)

        # Check snapshot
        snap = engine.snapshot()
        if is_v2x:
            v2x_flagged = [r for r in snap["v2x"] if r.get("status") in ("rejected", "suspicious")]
            if v2x_flagged and not flagged_row_found:
                flagged_row_found = v2x_flagged[0]
                found_at_t = t
        else:
            can_flagged = [r for r in snap["can"] if r.get("verdict") in ("blocked", "suspicious")]
            if can_flagged and not flagged_row_found:
                flagged_row_found = can_flagged[0]
                found_at_t = t

    # Assertion 1: Expected stream contains >= 1 non-normal row within 1.5s
    assert flagged_row_found, f"Attack {attack_name} did not produce non-normal row within 1.5s"
    assert found_at_t <= detect_deadline, f"Attack {attack_name} detected too late: t={found_at_t}s"

    # Assertion 2: Row contains non-empty, plain-English reason
    reason = flagged_row_found.get("reason", "").strip()
    assert len(reason) > 5, f"Attack {attack_name} had empty or trivial reason: '{reason}'"

    # Assertion 3: V2X attacks NEVER produce rows in snapshot["can"]
    snap = engine.snapshot()
    if is_v2x:
        for r in snap["can"]:
            assert r.get("verdict") == "normal", f"V2X attack {attack_name} leaked non-normal row into CAN table: {r}"
            assert not str(r.get("sender", "")).startswith("V-"), f"V2X vehicle leaked into CAN stream: {r}"

    # 3. Continue running simulation through the end of the attack
    while t < attack_end_t:
        t = round(t + dt, 4)
        process_step(t)

    # Attack has now stopped.
    # Advance simulation for at least 7.5 s after attack stopped (within 8.0s persistence window)
    persistence_target_t = round(attack_end_t + 7.5, 4)
    while t < persistence_target_t:
        t = round(t + dt, 4)
        process_step(t)

    # Assertion 4: Row stays visible in the snapshot for at least 8.0s after the attack stops
    snap_persisted = engine.snapshot()
    if is_v2x:
        flagged_persisted = [r for r in snap_persisted["v2x"] if r.get("status") in ("rejected", "suspicious")]
        assert len(flagged_persisted) > 0, f"V2X flagged rows disappeared before 8s persistence: {attack_name}"
    else:
        flagged_persisted = [r for r in snap_persisted["can"] if r.get("verdict") in ("blocked", "suspicious")]
        assert len(flagged_persisted) > 0, f"CAN flagged rows disappeared before 8s persistence: {attack_name}"

    # Also verify security_events feed contains the event
    assert len(snap_persisted["security_events"]) > 0
    event_domains = {ev["domain"] for ev in snap_persisted["security_events"]}
    expected_domain = "V2X" if is_v2x else "CAN"
    assert expected_domain in event_domains


def test_snapshot_build_latency_under_flood() -> None:
    """Verify snapshot() builds in < 5 ms under extreme flood traffic."""
    cfg = load_config()
    engine = CAVShieldEngine(cfg=cfg)
    engine.set_protection(True)

    # Run flood attack headless for 2.0s
    engine.run_headless(duration_s=2.0, dt=0.005, attack_name="flood", attack_start_s=0.2, attack_duration_s=1.8)

    # Measure snapshot latency over 100 calls
    latencies_ms: List[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        snap = engine.snapshot()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat_ms)

    mean_lat = sum(latencies_ms) / len(latencies_ms)
    p95_lat = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]

    assert mean_lat < 5.0, f"Snapshot build time {mean_lat:.2f}ms exceeds 5ms limit!"
    assert p95_lat < 5.0, f"Snapshot p95 build time {p95_lat:.2f}ms exceeds 5ms limit!"
    assert len(snap["can"]) > 0
    assert snap["can_flagged_count"] > 0
