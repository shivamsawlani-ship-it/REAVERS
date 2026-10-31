"""
CAVShield Cyber-Physical Decision and Risk Engine.

Implements CONTRACT Item 9:
- Multi-source risk fusion combining deterministic rule checks, physical plausibility
  invariants, independent sensor fusion, and machine learning AI context.
- Four ablation switches: use_ai, use_can_rules, use_v2x, use_physics.
- Confidence gating: CRITICAL requires det >= 0.7, ai_conf >= 0.60, and criticality >= 0.8.
- Explainability module providing concrete numeric citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src import load_config
from src.can_simulator import (
    COMMAND_IDS,
    KNOWN_IDS,
    CANFrame,
    criticality,
    is_safety_critical,
)
from src.detection.rule_engine import Evidence
from src.detection.v2x_validator import V2XVerdict
from src.v2x_simulator import V2XMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONTRACT Dataclasses & Enums (CONTRACT Items 4 & 42)
# ---------------------------------------------------------------------------

class RiskLevel(IntEnum):
    """Hierarchical risk levels matching CONTRACT item 42."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        return self.name


@dataclass
class AIContext:
    """Shared AI inference context updated by asynchronous background workers."""
    anomaly: float = 0.0          # normalized anomaly score [0.0, 1.0]
    attack_prob: float = 0.0      # attack probability [0.0, 1.0]
    label: str = "Normal"         # predicted class name
    confidence: float = 0.0       # top class confidence [0.0, 1.0]
    probs: Dict[str, float] = field(default_factory=dict)
    t: float = 0.0                # timestamp of inference
    available: bool = False


@dataclass(frozen=True)
class Decision:
    """Actionable decision emitted for every frame or event (CONTRACT item 42)."""
    level: RiskLevel
    score: float
    attack_type: str
    confidence: float
    explanation: str
    action: str
    components: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """Multi-source cyber-physical risk fusion engine."""

    def __init__(self, cfg: Optional[Any] = None) -> None:
        """Initialize RiskEngine with configuration tunables."""
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        risk_cfg = getattr(cfg, "risk", {})
        weights = getattr(risk_cfg, "weights", {})
        self.w_det = float(getattr(weights, "w_det", 0.65))
        self.w_ai = float(getattr(weights, "w_ai", 0.35))

        thresholds = getattr(risk_cfg, "thresholds", {})
        self.th_low = float(getattr(thresholds, "low", 0.25))
        self.th_medium = float(getattr(thresholds, "medium", 0.50))
        self.th_high = float(getattr(thresholds, "high", 0.72))

        self.critical_ai_conf = float(getattr(risk_cfg, "critical_ai_conf", 0.60))
        self.ai_fresh_s = float(getattr(risk_cfg, "ai_fresh_s", 0.8))
        self.ai_only_cap = float(getattr(risk_cfg, "ai_only_cap", 0.45))

        # Ablation study flags
        self.use_ai = bool(getattr(risk_cfg, "use_ai", True))
        self.use_can_rules = bool(getattr(risk_cfg, "use_can_rules", True))
        self.use_v2x = bool(getattr(risk_cfg, "use_v2x", True))
        self.use_physics = bool(getattr(risk_cfg, "use_physics", True))

    def fuse_can(
        self,
        frame: CANFrame,
        evidences: List[Evidence],
        ai_ctx: Optional[AIContext] = None,
        now: Optional[float] = None,
    ) -> Decision:
        """Fuse deterministic and AI evidences for an incoming CAN frame.

        Parameters
        ----------
        frame : CANFrame
            CAN frame under inspection.
        evidences : List[Evidence]
            List of Evidence objects from rule engine and physics engine.
        ai_ctx : Optional[AIContext]
            Latest asynchronous AI inference context.
        now : Optional[float]
            Current simulation timestamp. Defaults to frame.timestamp.

        Returns
        -------
        Decision
            Actionable fusion decision.
        """
        curr_time = frame.timestamp if now is None else now

        # 1. Filter evidences honoring ablation switches
        active_evidences: List[Evidence] = []
        for ev in evidences:
            if ev.source == "rule" and not self.use_can_rules:
                continue
            if ev.source == "physics" and not self.use_physics:
                continue
            active_evidences.append(ev)

        # 2. Compute Deterministic Score
        # det = max(rule severity, physics severity) + 0.05 per extra distinct evidence (cap 1.0)
        distinct_names: Set[str] = {e.name for e in active_evidences}
        if active_evidences:
            max_sev = max(e.severity for e in active_evidences)
            extra_count = max(0, len(distinct_names) - 1)
            det = min(1.0, max_sev + 0.05 * extra_count)
        else:
            det = 0.0

        # 3. Compute AI Confidence (freshness check)
        ai_conf = 0.0
        ai_fresh = False
        if self.use_ai and ai_ctx is not None and ai_ctx.available:
            age = curr_time - ai_ctx.t
            if 0.0 <= age <= self.ai_fresh_s:
                ai_fresh = True
                ai_conf = (ai_ctx.anomaly + ai_ctx.attack_prob) / 2.0

        # 4. Score Fusion
        crit = criticality(frame.arb_id)
        if det > 0.0:
            raw_score = self.w_det * det + self.w_ai * ai_conf
            # Criticality weighting: score = score * (0.75 + 0.25 * criticality)
            score = raw_score * (0.75 + 0.25 * crit)
        else:
            # AI-only per-frame: level LOW per frame (score capped at ai_only_cap)
            raw_score = min(ai_conf, self.ai_only_cap)
            score = raw_score * (0.75 + 0.25 * crit)

        # 5. Risk Level Assignment
        if score < self.th_low:
            level = RiskLevel.LOW
        elif score < self.th_medium:
            level = RiskLevel.MEDIUM
        elif score < self.th_high:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL

        # 6. Confidence Gate (CONTRACT Item 9)
        # CRITICAL only if det >= 0.7 AND ai_conf >= 0.60 AND criticality >= 0.8; otherwise cap at HIGH.
        # AI alone never exceeds MEDIUM (per-frame AI-only is capped at LOW).
        if level == RiskLevel.CRITICAL:
            gate_pass = (det >= 0.70) and (ai_conf >= self.critical_ai_conf) and (crit >= 0.80)
            if not gate_pass:
                level = RiskLevel.HIGH

        if det == 0.0:
            # AI alone per-frame rule
            if level > RiskLevel.LOW:
                level = RiskLevel.LOW

        # 7. Attack-Type Attribution
        attack_type = self._attribute_attack_type(active_evidences, ai_ctx, ai_fresh, level)

        # 8. Explanation and Action Recommendation
        explanation = self._build_explanation(frame, active_evidences, ai_ctx, ai_fresh, score)
        action_text, driver_action = self._recommend_actions(level, attack_type, frame.arb_id)

        components = {
            "det_score": det,
            "ai_conf": ai_conf,
            "ai_fresh": ai_fresh,
            "criticality": crit,
            "evidence_count": len(active_evidences),
            "evidence_names": list(distinct_names),
            "driver_action": driver_action,
        }

        return Decision(
            level=level,
            score=float(min(1.0, score)),
            attack_type=attack_type,
            confidence=float(max(det, ai_conf)),
            explanation=explanation,
            action=action_text,
            components=components,
        )

    def fuse_v2x(
        self,
        msg: V2XMessage,
        verdict: V2XVerdict,
        now: Optional[float] = None,
    ) -> Decision:
        """Evaluate V2X validation verdict into a system Decision.

        Parameters
        ----------
        msg : V2XMessage
            The received V2X message.
        verdict : V2XVerdict
            Verdict from V2XValidator.
        now : Optional[float]
            Current timestamp.

        Returns
        -------
        Decision
            Actionable decision for V2X handling.
        """
        if not self.use_v2x or verdict.accepted:
            return Decision(
                level=RiskLevel.LOW,
                score=0.0,
                attack_type="Normal",
                confidence=float(verdict.trust),
                explanation="V2X message verified and trusted.",
                action="forward",
                components={"trust": verdict.trust, "accepted": True, "status": "accepted"},
            )

        # Rejected or Suspicious V2X message
        trust_deficit = max(0.0, 1.0 - verdict.trust)
        reasons_text = "; ".join(verdict.reasons) if verdict.reasons else "V2X trust threshold not met"

        # Determine attack subtype attribution
        reasons_str = reasons_text.lower()
        if "signature" in reasons_str:
            attack_type = "Spoofing"
            sev = 0.85
        elif "ghost vehicle" in reasons_str or "phantom" in reasons_str or "obstacle" in reasons_str:
            attack_type = "Spoofing"  # Phantom obstacle spoofing
            sev = 0.85
        elif "kinematic" in reasons_str or "gps" in reasons_str or "jump" in reasons_str:
            attack_type = "Spoofing"  # GPS teleportation / kinematic jump
            sev = 0.85
        elif "freshness" in reasons_str or "replay" in reasons_str:
            attack_type = "Replay"
            sev = 0.85
        else:
            attack_type = "Unknown anomaly"
            sev = 0.70

        status_val = getattr(verdict, "status", "rejected" if not verdict.accepted else "accepted")
        if status_val == "rejected" or not verdict.accepted:
            level = RiskLevel.HIGH
            score = max(self.th_high + 0.05, float(min(1.0, 0.75 * sev + 0.25 * max(0.5, trust_deficit))))
            action_text = "reject unverified V2X"
        elif status_val == "suspicious":
            level = RiskLevel.MEDIUM
            score = max(self.th_medium + 0.05, float(min(1.0, 0.75 * sev + 0.25 * max(0.3, trust_deficit))))
            action_text = "alert"
        else:
            level = RiskLevel.LOW
            score = 0.0
            action_text = "forward"

        # Escalation if message asserted emergency deceleration or hazard warnings
        if msg.msg_type in ("EMERGENCY_BRAKING", "ROAD_HAZARD", "COLLISION_WARNING") and (status_val == "rejected" or not verdict.accepted):
            level = RiskLevel.HIGH
            action_text = "reject unverified V2X"

        explanation = (
            f"V2X {msg.msg_type} from {msg.sender_id} {status_val} (trust={verdict.trust:.2f}): {reasons_text}."
        )
        driver_action = f"Unverified V2X {msg.msg_type} advisory dropped. Continue standard vigilance."

        return Decision(
            level=level,
            score=score,
            attack_type=attack_type,
            confidence=trust_deficit,
            explanation=explanation,
            action=action_text,
            components={
                "trust": verdict.trust,
                "reasons": verdict.reasons,
                "status": status_val,
                "accepted": verdict.accepted,
                "driver_action": driver_action,
                "sender_id": msg.sender_id,
                "msg_type": msg.msg_type,
            },
        )

    def _attribute_attack_type(
        self,
        evidences: List[Evidence],
        ai_ctx: Optional[AIContext],
        ai_fresh: bool,
        level: RiskLevel,
    ) -> str:
        """Attribute attack class: AI label if confidence >= 0.6, else strongest hint."""
        if level == RiskLevel.LOW and not evidences:
            return "Normal"

        # 1. AI label if it is an attack and confidence >= 0.60
        if ai_fresh and ai_ctx is not None and ai_ctx.available:
            if ai_ctx.label not in ("Normal", "Unknown anomaly") and ai_ctx.confidence >= 0.60:
                return ai_ctx.label

        # 2. Strongest evidence hint
        for ev in sorted(evidences, key=lambda e: e.severity, reverse=True):
            if ev.hint and ev.hint != "Normal":
                return ev.hint

        # 3. AI label if present even with lower confidence
        if ai_fresh and ai_ctx is not None and ai_ctx.label not in ("Normal", ""):
            return ai_ctx.label

        return "Unknown anomaly" if level > RiskLevel.LOW else "Normal"

    def _build_explanation(
        self,
        frame: CANFrame,
        evidences: List[Evidence],
        ai_ctx: Optional[AIContext],
        ai_fresh: bool,
        score: float,
    ) -> str:
        """Generate human-readable explanation citing concrete values and numbers."""
        parts: List[str] = []

        # Evidences details
        if evidences:
            ev_details = [e.detail for e in evidences if e.detail]
            if ev_details:
                parts.append("; ".join(ev_details))
            else:
                names = [e.name for e in evidences]
                parts.append(f"Triggered rule checks: {', '.join(names)}")
        else:
            parts.append(f"Frame 0x{frame.arb_id:03X} within normal protocol tolerances")

        # AI details
        if ai_fresh and ai_ctx is not None and ai_ctx.available:
            ai_str = f"AI: anomaly {ai_ctx.anomaly:.2f}, {ai_ctx.label} p={ai_ctx.confidence:.2f}"
            parts.append(ai_str)

        return ". ".join(parts) + "."

    def _recommend_actions(
        self,
        level: RiskLevel,
        attack_type: str,
        arb_id: int,
    ) -> Tuple[str, str]:
        """Determine system frame action and driver safety advisory."""
        is_cmd = arb_id in COMMAND_IDS

        if level == RiskLevel.CRITICAL:
            action_text = "block + mode change"
            if is_cmd:
                driver_text = f"CRITICAL: Spoofed control command 0x{arb_id:03X} blocked. Vehicle in SAFE MODE. Take manual control immediately."
            else:
                driver_text = f"CRITICAL: Core telemetry compromised. Vehicle in DEGRADED mode. Exercise caution."

        elif level == RiskLevel.HIGH:
            if attack_type == "Flooding":
                action_text = "drop frame / rate-limit / quarantine ID"
                driver_text = "HIGH: CAN bus flooding detected. Rate-limiting active. Telemetry degraded."
            elif attack_type == "Fuzzing":
                action_text = "drop frame / rate-limit / quarantine ID"
                driver_text = f"HIGH: Unknown or malformed ID 0x{arb_id:03X} quarantined."
            else:
                action_text = "drop frame / rate-limit / quarantine ID"
                driver_text = f"HIGH: Hostile {attack_type} detected on 0x{arb_id:03X}. Frame dropped."

        elif level == RiskLevel.MEDIUM:
            action_text = "forward + ALERT"
            driver_text = "MEDIUM: Suspicious activity detected. Monitoring vehicle subsystems."

        else:
            action_text = "forward"
            driver_text = "System operating normally."

        return action_text, driver_text

