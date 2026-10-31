"""
Unit and integration tests for CAVShield Part 7:
- Decision Engine (CONTRACT item 9 risk fusion, confidence gate, ablation flags, attribution, explainability)
- FailSafe Manager (state machine modes, timed auto-recovery, gate/token-bucket, queue prioritization, incident merging)
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import time
import numpy as np
import pytest

from src import load_config
from src.can_simulator import CANFrame
from src.detection.decision_engine import (
    AIContext,
    Decision,
    RiskEngine,
    RiskLevel,
)
from src.detection.rule_engine import Evidence
from src.detection.v2x_validator import V2XVerdict
from src.failsafe.failsafe_manager import FailSafeManager, TokenBucket
from src.v2x_simulator import V2XMessage


# ---------------------------------------------------------------------------
# Decision Engine Tests
# ---------------------------------------------------------------------------

class TestDecisionEngine:
    """Tests for RiskEngine and multi-source risk fusion."""

    def test_level_transitions(self) -> None:
        """Verify proper assignment of LOW, MEDIUM, HIGH, and CRITICAL risk levels."""
        engine = RiskEngine()
        frame_cmd = CANFrame(timestamp=10.0, arb_id=0x200, dlc=8, data=b"\x00" * 8, sender="ECU_ADAS", msg_type="COMMAND")

        # 1. LOW: Clean frame, no evidences, no AI
        dec_low = engine.fuse_can(frame_cmd, evidences=[], ai_ctx=None, now=10.0)
        assert dec_low.level == RiskLevel.LOW
        assert dec_low.action == "forward"
        assert dec_low.score < 0.25

        # 2. MEDIUM: Moderate rule evidence (e.g. counter glitch)
        ev_med = [Evidence(source="rule", name="COUNTER", severity=0.55, detail="Counter jump", hint="Replay")]
        dec_med = engine.fuse_can(frame_cmd, evidences=ev_med, ai_ctx=None, now=10.0)
        assert dec_med.level == RiskLevel.MEDIUM
        assert dec_med.action == "forward + ALERT"
        assert 0.25 <= dec_med.score < 0.50

        # 3. HIGH: High severity deterministic rule or capped critical
        ev_high = [Evidence(source="rule", name="CHECKSUM", severity=0.85, detail="Checksum mismatch", hint="Fuzzing")]
        dec_high = engine.fuse_can(frame_cmd, evidences=ev_high, ai_ctx=None, now=10.0)
        assert dec_high.level == RiskLevel.HIGH
        assert dec_high.action == "drop frame / rate-limit / quarantine ID"

        # 4. CRITICAL: High det + Fresh High AI + Safety-critical command ID (0x200)
        ai_crit = AIContext(anomaly=0.90, attack_prob=0.95, label="Spoofing", confidence=0.96, t=9.9, available=True)
        dec_crit = engine.fuse_can(frame_cmd, evidences=ev_high, ai_ctx=ai_crit, now=10.0)
        assert dec_crit.level == RiskLevel.CRITICAL
        assert dec_crit.action == "block + mode change"
        assert dec_crit.score >= 0.72

    def test_confidence_gate_caps_at_high(self) -> None:
        """Confidence gate: CRITICAL strictly requires det >= 0.7, ai_conf >= 0.60, and criticality >= 0.8."""
        engine = RiskEngine()
        ev_strong = [Evidence(source="physics", name="UNJUSTIFIED_BRAKE", severity=0.90, detail="Brake without pedal", hint="Spoofing")]

        # Case A: High det (0.90), but AI confidence LOW (0.30) on command ID 0x200 (crit=1.0)
        # Must be capped at HIGH
        ai_weak = AIContext(anomaly=0.30, attack_prob=0.30, label="Spoofing", confidence=0.35, t=10.0, available=True)
        frame_cmd = CANFrame(timestamp=10.0, arb_id=0x200, dlc=8, data=b"\x00" * 8, sender="ECU_ADAS", msg_type="COMMAND")
        dec_a = engine.fuse_can(frame_cmd, evidences=ev_strong, ai_ctx=ai_weak, now=10.0)
        assert dec_a.level == RiskLevel.HIGH, f"Expected HIGH due to confidence gate, got {dec_a.level}"

        # Case B: High det (0.90) and High AI (0.90), but ID is non-critical (0x100 ENGINE_RPM, crit=0.6 < 0.8)
        # Must be capped at HIGH
        ai_strong = AIContext(anomaly=0.90, attack_prob=0.90, label="Spoofing", confidence=0.95, t=10.0, available=True)
        frame_status = CANFrame(timestamp=10.0, arb_id=0x100, dlc=8, data=b"\x00" * 8, sender="ECU_ENGINE", msg_type="STATUS")
        dec_b = engine.fuse_can(frame_status, evidences=ev_strong, ai_ctx=ai_strong, now=10.0)
        assert dec_b.level == RiskLevel.HIGH, f"Expected HIGH for non-critical ID, got {dec_b.level}"

        # Case C: All three criteria met: det=0.90 >= 0.7, ai_conf=0.90 >= 0.60, crit=1.0 >= 0.8
        # Escalate to CRITICAL
        dec_c = engine.fuse_can(frame_cmd, evidences=ev_strong, ai_ctx=ai_strong, now=10.0)
        assert dec_c.level == RiskLevel.CRITICAL

    def test_ai_alone_never_exceeds_medium(self) -> None:
        """AI alone (no deterministic evidences) per-frame is LOW and never exceeds MEDIUM."""
        engine = RiskEngine()
        frame = CANFrame(timestamp=10.0, arb_id=0x200, dlc=8, data=b"\x00" * 8, sender="ECU_ADAS", msg_type="COMMAND")

        # Extremely anomalous AI prediction
        ai_alone = AIContext(anomaly=0.99, attack_prob=0.99, label="Spoofing", confidence=0.99, t=10.0, available=True)

        dec = engine.fuse_can(frame, evidences=[], ai_ctx=ai_alone, now=10.0)
        # Per frame rule: det == 0 -> level LOW per frame (score capped at ai_only_cap)
        assert dec.level == RiskLevel.LOW
        assert dec.score <= 0.45

    def test_ablation_flags(self) -> None:
        """Verify that disabling use_* flags turns off respective detection sources."""
        cfg = load_config()
        frame = CANFrame(timestamp=10.0, arb_id=0x200, dlc=8, data=b"\x00" * 8, sender="ECU_ADAS", msg_type="COMMAND")

        ev_rule = [Evidence(source="rule", name="CHECKSUM", severity=0.90, detail="bad checksum", hint="Fuzzing")]
        ev_phys = [Evidence(source="physics", name="UNJUSTIFIED_BRAKE", severity=0.90, detail="unjustified", hint="Spoofing")]
        ai_ctx = AIContext(anomaly=0.90, attack_prob=0.90, label="Spoofing", confidence=0.90, t=10.0, available=True)

        # 1. Disable use_can_rules -> rule evidence ignored
        engine_no_rules = RiskEngine(cfg)
        engine_no_rules.use_can_rules = False
        dec1 = engine_no_rules.fuse_can(frame, evidences=ev_rule, ai_ctx=None, now=10.0)
        assert dec1.level == RiskLevel.LOW

        # 2. Disable use_physics -> physics evidence ignored
        engine_no_phys = RiskEngine(cfg)
        engine_no_phys.use_physics = False
        dec2 = engine_no_phys.fuse_can(frame, evidences=ev_phys, ai_ctx=None, now=10.0)
        assert dec2.level == RiskLevel.LOW

        # 3. Disable use_ai -> AI confidence ignored
        engine_no_ai = RiskEngine(cfg)
        engine_no_ai.use_ai = False
        dec3 = engine_no_ai.fuse_can(frame, evidences=[], ai_ctx=ai_ctx, now=10.0)
        assert dec3.score == 0.0

    def test_attack_type_attribution_and_explain(self) -> None:
        """Verify attack-type attribution logic and concrete numeric explanation."""
        engine = RiskEngine()
        frame = CANFrame(timestamp=10.0, arb_id=0x200, dlc=8, data=b"\x00" * 8, sender="ECU_ADAS", msg_type="COMMAND")

        # Detail with concrete numbers
        detail = "Asserted decel 7.00 m/s2 (source DRIVER) with pedal=0.0%, measured decel 0.02 m/s2"
        ev = [Evidence(source="physics", name="UNJUSTIFIED_BRAKE", severity=0.90, detail=detail, hint="Spoofing")]
        ai = AIContext(anomaly=0.93, attack_prob=0.97, label="Spoofing", confidence=0.97, t=9.9, available=True)

        dec = engine.fuse_can(frame, evidences=ev, ai_ctx=ai, now=10.0)
        assert dec.attack_type == "Spoofing"
        assert "7.00 m/s2" in dec.explanation
        assert "pedal=0.0%" in dec.explanation
        assert "AI: anomaly 0.93" in dec.explanation

    def test_v2x_fusion(self) -> None:
        """Verify V2X trust verdict risk evaluation."""
        engine = RiskEngine()
        msg = V2XMessage(
            msg_id=1,
            sender_id="V-2041",
            x=1050.0,
            y=0.0,
            speed=20.0,
            msg_type="EMERGENCY_BRAKING",
            timestamp=10.0,
            heading=0.0,
            signature="valid",
            seq=1,
            claimed_decel=6.0,
        )

        # Accepted
        verdict_ok = V2XVerdict(accepted=True, trust=0.95)
        dec_ok = engine.fuse_v2x(msg, verdict_ok, now=10.0)
        assert dec_ok.level == RiskLevel.LOW
        assert dec_ok.action == "forward"

        # Rejected (tampered signature / phantom hazard)
        verdict_bad = V2XVerdict(accepted=False, trust=0.10, reasons=["Invalid HMAC signature", "Ghost vehicle"])
        dec_bad = engine.fuse_v2x(msg, verdict_bad, now=10.0)
        assert dec_bad.level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert dec_bad.action == "reject unverified V2X"
        assert "Invalid HMAC signature" in dec_bad.explanation


# ---------------------------------------------------------------------------
# FailSafe Manager Tests
# ---------------------------------------------------------------------------

class TestFailSafeManager:
    """Tests for FailSafeManager gateway, state machine, and incident logger."""

    def test_mode_escalations(self) -> None:
        """Verify mode escalation rules: CRITICAL command -> SAFE MODE, CRITICAL status -> DEGRADED."""
        fsm = FailSafeManager()
        assert fsm.current_mode == FailSafeManager.MODE_NORMAL

        # 1. HIGH risk -> Escalate to ALERT
        frame_flood = CANFrame(timestamp=1.0, arb_id=0x000, dlc=8, data=b"\x00" * 8, sender="ECU_ENGINE", msg_type="STATUS")
        dec_high = Decision(level=RiskLevel.HIGH, score=0.60, attack_type="Flooding", confidence=0.8, explanation="flood", action="drop")
        fwd, acts = fsm.enforce(dec_high, frame_flood, now=1.0)
        assert fsm.current_mode == FailSafeManager.MODE_ALERT

        # 2. CRITICAL on status ID 0x101 (VEHICLE_SPEED) -> Escalate to DEGRADED
        frame_status = CANFrame(timestamp=2.0, arb_id=0x101, dlc=8, data=b"\x00" * 8, sender="ECU_ENGINE", msg_type="STATUS")
        dec_crit_status = Decision(level=RiskLevel.CRITICAL, score=0.85, attack_type="Spoofing", confidence=0.9, explanation="crit", action="block")
        fwd, acts = fsm.enforce(dec_crit_status, frame_status, now=2.0)
        assert fwd is False
        assert fsm.current_mode == FailSafeManager.MODE_DEGRADED

        # 3. CRITICAL on command ID 0x200 (BRAKE_COMMAND) -> Escalate to SAFE MODE
        frame_cmd = CANFrame(timestamp=3.0, arb_id=0x200, dlc=8, data=b"\x00" * 8, sender="ECU_ADAS", msg_type="COMMAND")
        dec_crit_cmd = Decision(level=RiskLevel.CRITICAL, score=0.95, attack_type="Spoofing", confidence=0.95, explanation="crit cmd", action="block")
        fwd, acts = fsm.enforce(dec_crit_cmd, frame_cmd, now=3.0)
        assert fwd is False
        assert fsm.current_mode == FailSafeManager.MODE_SAFE_MODE
        assert fsm.rejected_commands[0x200] == 1

    def test_timed_auto_recovery(self) -> None:
        """Verify timed step-by-step recovery: SAFE MODE -> DEGRADED (12s) -> ALERT (8s) -> NORMAL (4s)."""
        fsm = FailSafeManager()
        # Escalate to SAFE MODE at t = 10.0
        frame_cmd = CANFrame(timestamp=10.0, arb_id=0x200, dlc=8, data=b"\x00" * 8, sender="ECU_ADAS", msg_type="COMMAND")
        dec_crit = Decision(level=RiskLevel.CRITICAL, score=0.95, attack_type="Spoofing", confidence=0.95, explanation="crit", action="block")
        fsm.enforce(dec_crit, frame_cmd, now=10.0)
        assert fsm.current_mode == FailSafeManager.MODE_SAFE_MODE

        # At t = 15.0 (5s elapsed): still SAFE MODE (< 12s)
        mode = fsm.update_mode(now=15.0)
        assert mode == FailSafeManager.MODE_SAFE_MODE

        # At t = 22.1 (12.1s elapsed): recovers to DEGRADED
        mode = fsm.update_mode(now=22.1)
        assert mode == FailSafeManager.MODE_DEGRADED

        # At t = 25.0 (2.9s in DEGRADED): still DEGRADED (< 8s)
        mode = fsm.update_mode(now=25.0)
        assert mode == FailSafeManager.MODE_DEGRADED

        # At t = 30.2 (8.1s in DEGRADED): recovers to ALERT
        mode = fsm.update_mode(now=30.2)
        assert mode == FailSafeManager.MODE_ALERT

        # At t = 34.3 (4.1s in ALERT): recovers to NORMAL
        mode = fsm.update_mode(now=34.3)
        assert mode == FailSafeManager.MODE_NORMAL

    def test_quarantine_and_token_bucket(self) -> None:
        """Verify unknown ID quarantine (5s) and token bucket rate limiting."""
        fsm = FailSafeManager()
        unknown_id = 0x7FF
        frame_unk = CANFrame(timestamp=10.0, arb_id=unknown_id, dlc=8, data=b"\x00" * 8, sender="UNKNOWN", msg_type="UNKNOWN")

        dec_unk = Decision(level=RiskLevel.HIGH, score=0.70, attack_type="Fuzzing", confidence=0.85, explanation="fuzz", action="drop")
        fwd, acts = fsm.enforce(dec_unk, frame_unk, now=10.0)
        assert fwd is False
        assert unknown_id in fsm.quarantine_expiry

        # At t = 12.0 (2s elapsed < 5s): gate drops frame
        assert fsm.gate(frame_unk, now=12.0) is False

        # Token Bucket test on rate-limited flooding ID
        flood_id = 0x000
        frame_flood = CANFrame(timestamp=10.0, arb_id=flood_id, dlc=8, data=b"\x00" * 8, sender="ECU_ENGINE", msg_type="STATUS")
        fsm.rate_limited_active_ids.add(flood_id)

        # Rapid fire 5 frames in 0.001 s (burst is 2 tokens)
        passed = sum(fsm.gate(frame_flood, now=10.0 + i * 0.0001) for i in range(5))
        assert passed <= 2  # tokens exhausted, remaining dropped

    def test_protection_disabled_observes_only(self) -> None:
        """When protection_enabled=False, gateway observes and logs but forwards everything."""
        fsm = FailSafeManager()
        fsm.set_protection_enabled(False)

        frame_cmd = CANFrame(timestamp=10.0, arb_id=0x200, dlc=8, data=b"\x00" * 8, sender="ECU_ADAS", msg_type="COMMAND")
        dec_crit = Decision(level=RiskLevel.CRITICAL, score=0.95, attack_type="Spoofing", confidence=0.95, explanation="crit", action="block")

        fwd, acts = fsm.enforce(dec_crit, frame_cmd, now=10.0)
        assert fwd is True  # Forwarded to observe attack impact!
        assert fsm.current_mode == FailSafeManager.MODE_SAFE_MODE  # Mode still escalates for dashboard

    def test_incident_merging_within_window(self) -> None:
        """Repeated incidents within incident_merge_s (2.0s) are merged with incremented count."""
        fsm = FailSafeManager()
        frame = CANFrame(timestamp=10.0, arb_id=0x200, dlc=8, data=b"\x00" * 8, sender="ECU_ADAS", msg_type="COMMAND")
        dec = Decision(level=RiskLevel.CRITICAL, score=0.95, attack_type="Spoofing", confidence=0.95, explanation="crit", action="block")

        # Incident 1 at t = 10.0
        fsm.enforce(dec, frame, now=10.0)
        # Incident 2 at t = 10.5 (same attack, level, target within 2.0s)
        fsm.enforce(dec, frame, now=10.5)
        # Incident 3 at t = 11.2 (same attack, level, target within 2.0s of previous)
        fsm.enforce(dec, frame, now=11.2)

        incidents = fsm.get_recent_incidents()
        # Should be merged into 1 incident with count = 3
        assert len(incidents) >= 1
        top_inc = incidents[0]
        assert top_inc["target"] == "0x200"
        assert top_inc["count"] == 3
        assert top_inc["ts"] == 11.2

