"""
CAVShield Fail-Safe and Gateway Enforcement Manager.

Implements:
- Vehicle operational modes: NORMAL, ALERT, DEGRADED, SAFE MODE with state
  escalation rules and timed auto-recovery (12 s safe->degraded, 8 s degraded->alert, 4 s alert->normal).
- Gateway enforcement: gate() drops quarantined IDs and applies token bucket rate-limiting
  (1.25 x nominal rate, burst 2.0).
- Protection toggle: protection_enabled=False observes and logs without dropping frames.
- Queue backlog shedding: drops flooding frames and prioritizes safety-critical IDs.
- Forensic Incident Logger: Thread-safe SQLite database and JSONL logging with repeat
  merging within incident_merge_s (2.0 s).
- Real-time bookkeeping of blocked commands, quarantined IDs, rate-limited streams, and latencies.
"""

from __future__ import annotations

import collections
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from src import load_config
from src.attack_injector import resolve_entry_point
from src.can_simulator import (
    COMMAND_IDS,
    DBC,
    KNOWN_IDS,
    SAFETY_CRITICAL_IDS,
    CANFrame,
    counter_of,
)
from src.detection.decision_engine import Decision, RiskLevel
from src.v2x_simulator import V2XMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token Bucket Rate Limiter
# ---------------------------------------------------------------------------

class TokenBucket:
    """Thread-safe token bucket rate limiter for a specific arbitration ID."""

    def __init__(self, rate_hz: float, burst: float = 2.0) -> None:
        self.rate = float(rate_hz)
        self.burst = float(burst)
        self.tokens = float(burst)
        self.last_update = 0.0

    def consume(self, now: float) -> bool:
        """Attempt to consume 1 token. Returns True if accepted, False if rate-limited."""
        if self.last_update == 0.0:
            self.last_update = now
            self.tokens -= 1.0
            return True

        elapsed = max(0.0, now - self.last_update)
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


# ---------------------------------------------------------------------------
# Incident Record
# ---------------------------------------------------------------------------

@dataclass
class IncidentRecord:
    """Forensic incident record stored in SQLite, JSONL, and in-memory buffer."""
    id: int
    ts: float
    attack_type: str
    level: str
    target: str
    confidence: float
    explanation: str
    action: str
    latency_ms: float
    count: int = 1
    details: str = ""
    start_ts: float = 0.0
    assumed_entry_point: str = ""

    def to_dashboard_dict(self) -> Dict[str, Any]:
        """Format matching CONTRACT item 62 for dashboard display."""
        return {
            "id": self.id,
            "ts": round(self.ts, 3),
            "start_ts": round(self.start_ts, 3),
            "attack_type": self.attack_type,
            "level": self.level,
            "target": self.target,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "action": self.action,
            "latency_ms": round(self.latency_ms, 2),
            "count": self.count,
            "assumed_entry_point": self.assumed_entry_point,
        }


# ---------------------------------------------------------------------------
# FailSafe Manager
# ---------------------------------------------------------------------------

class FailSafeManager:
    """Central cybersecurity fail-safe gateway and incident logging manager."""

    # Operational modes
    MODE_NORMAL = "NORMAL"
    MODE_ALERT = "ALERT"
    MODE_DEGRADED = "DEGRADED"
    MODE_SAFE_MODE = "SAFE MODE"

    def __init__(self, cfg: Optional[Any] = None) -> None:
        """Initialize FailSafeManager with configuration."""
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        sim_cfg = getattr(cfg, "simulation", {})
        self.protection_enabled = bool(getattr(sim_cfg, "protection_enabled", True))

        fs_cfg = getattr(cfg, "failsafe", {})
        rec_timers = getattr(fs_cfg, "recovery_timers_s", {})
        self.t_safe_to_degraded = float(getattr(rec_timers, "safe_to_degraded", 1.0))
        self.t_degraded_to_alert = float(getattr(rec_timers, "degraded_to_alert", 1.0))
        self.t_alert_to_normal = float(getattr(rec_timers, "alert_to_normal", 1.0))
        self.incident_merge_s = float(getattr(fs_cfg, "incident_merge_s", 2.0))

        # Mode State Machine
        self.current_mode = self.MODE_NORMAL
        self.mode_history: List[Dict[str, Any]] = [{"t": 0.0, "mode": self.MODE_NORMAL}]

        self.last_critical_time = -100.0
        self.last_high_time = -100.0
        self.last_alert_time = -100.0
        self.last_recovery_time = 0.0

        # Gateway Filtering State
        self.quarantine_expiry: Dict[int, float] = {}  # arb_id -> expiry_ts
        self.token_buckets: Dict[int, TokenBucket] = {}
        self.rate_limited_active_ids: Set[int] = set()
        self.rate_limited_expiry: Dict[int, float] = {}

        # Lockout Watchdog & Starvation (CONTRACT / FIX 3 & FIX 5)
        self.fast_recovery: bool = False
        self.resync_callback: Optional[Any] = None
        self._id_block_history: Dict[int, Deque[Tuple[float, bool, int]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=60)
        )
        self._last_lockout_incident_ts: Dict[int, float] = {}
        self.starvation_warning: Optional[str] = None
        self._legit_cmd_blocked_start: Optional[float] = None

        # Telemetry and Counters
        self.rejected_commands: Dict[int, int] = collections.defaultdict(int)
        self.rejected_v2x_count = 0
        self.recent_actions: Deque[str] = collections.deque(maxlen=8)
        self.incident_id_counter = 0

        # Latency tracking
        self.detection_latencies: Deque[float] = collections.deque(maxlen=200)
        self.failsafe_latencies: Deque[float] = collections.deque(maxlen=200)

        # Thread Safety & Persistence
        self._lock = threading.RLock()
        self.incidents_buffer: Deque[IncidentRecord] = collections.deque(maxlen=100)
        self._last_merged_incident: Optional[IncidentRecord] = None

        output_logs_dir = Path(getattr(cfg.paths, "output_logs", "output/logs"))
        output_logs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = output_logs_dir / "incidents.db"
        self.jsonl_path = output_logs_dir / "incidents.jsonl"
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite incidents table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS incidents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts REAL,
                        attack_type TEXT,
                        level TEXT,
                        target TEXT,
                        confidence REAL,
                        explanation TEXT,
                        action TEXT,
                        latency_ms REAL,
                        count INTEGER DEFAULT 1,
                        details TEXT,
                        assumed_entry_point TEXT
                    )
                    """
                )
                try:
                    conn.execute("ALTER TABLE incidents ADD COLUMN assumed_entry_point TEXT DEFAULT ''")
                except Exception:
                    pass
                conn.commit()
        except Exception as exc:
            logger.warning("Could not initialize SQLite database %s: %s", self.db_path, exc)

    def set_protection_enabled(self, enabled: bool) -> None:
        """Toggle active gateway protection mode."""
        with self._lock:
            self.protection_enabled = bool(enabled)
            logger.info("Gateway protection set to: %s", self.protection_enabled)

    # -----------------------------------------------------------------------
    # State Machine & Escalation Rules
    # -----------------------------------------------------------------------

    def update_mode(self, now: float) -> str:
        """Execute timed auto-recovery state machine returning to NORMAL after quiet period."""
        with self._lock:
            # Fast recovery after attack cessation (< 3s quiet) if enabled
            if self.fast_recovery:
                quiet_since_attack = now - max(self.last_critical_time, self.last_high_time)
                if self.current_mode != self.MODE_NORMAL and quiet_since_attack >= 2.5:
                    self._set_mode(self.MODE_NORMAL, now, "Timed recovery to NORMAL (< 3s)")
                    self.last_recovery_time = now
                    self.rate_limited_active_ids.clear()
                    self.rate_limited_expiry.clear()
                    self.rate_limited_expiry.clear()
                    self.quarantine_expiry.clear()
                    self.last_alert_time = -100.0
                    return self.current_mode

            if self.current_mode == self.MODE_SAFE_MODE:
                # SAFE MODE -> DEGRADED after safe_to_degraded quiet seconds
                if (now - self.last_critical_time) >= self.t_safe_to_degraded:
                    self.last_recovery_time = now
                    self._set_mode(self.MODE_DEGRADED, now, "Timed recovery from SAFE MODE")

            elif self.current_mode == self.MODE_DEGRADED:
                # DEGRADED -> ALERT after degraded_to_alert quiet seconds spent in DEGRADED
                quiet_since_event = now - max(self.last_critical_time, self.last_high_time)
                time_in_mode = now - self.last_recovery_time
                if quiet_since_event >= self.t_degraded_to_alert and time_in_mode >= self.t_degraded_to_alert:
                    self.last_recovery_time = now
                    self._set_mode(self.MODE_ALERT, now, "Timed recovery from DEGRADED")

            elif self.current_mode == self.MODE_ALERT:
                # ALERT -> NORMAL after alert_to_normal quiet seconds spent in ALERT
                quiet_since_event = now - max(self.last_critical_time, self.last_high_time, self.last_alert_time)
                time_in_mode = now - self.last_recovery_time
                if quiet_since_event >= self.t_alert_to_normal and time_in_mode >= self.t_alert_to_normal:
                    self.last_recovery_time = now
                    self._set_mode(self.MODE_NORMAL, now, "Timed recovery from ALERT")
                    self.rate_limited_active_ids.clear()
                    self.rate_limited_expiry.clear()

            return self.current_mode

    def _set_mode(self, new_mode: str, now: float, reason: str = "") -> None:
        """Transition vehicle mode and log to mode history."""
        if self.current_mode != new_mode:
            self.current_mode = new_mode
            self.mode_history.append({"t": round(now, 3), "mode": new_mode, "reason": reason})
            action_msg = f"Mode transition to {new_mode} ({reason})"
            self.recent_actions.append(action_msg)
            logger.info(action_msg)

    # -----------------------------------------------------------------------
    # Gateway Inspection & Token Bucket
    # -----------------------------------------------------------------------

    def gate(self, frame: CANFrame, now: Optional[float] = None) -> bool:
        """Check if frame passes quarantine and rate-limiting filters.

        Parameters
        ----------
        frame : CANFrame
            Candidate CAN frame.
        now : Optional[float]
            Current timestamp.

        Returns
        -------
        bool
            True if frame passes gateway; False if dropped.
        """
        curr_time = frame.timestamp if now is None else now

        if not self.protection_enabled:
            return True

        with self._lock:
            # 1. Quarantine Check
            if frame.arb_id in self.quarantine_expiry:
                expiry = self.quarantine_expiry[frame.arb_id]
                if curr_time < expiry:
                    # Quarantined: drop frame without resetting countdown indefinitely on incoming frames
                    return False
                else:
                    del self.quarantine_expiry[frame.arb_id]

            # 2. Token Bucket Rate Limiting for active rate-limited IDs
            if frame.arb_id in self.rate_limited_active_ids:
                exp = self.rate_limited_expiry.get(frame.arb_id, 0.0)
                if exp > 0.0 and curr_time > exp:
                    self.rate_limited_active_ids.discard(frame.arb_id)
                    self.rate_limited_expiry.pop(frame.arb_id, None)
                else:
                    bucket = self._get_or_create_bucket(frame.arb_id)
                    if not bucket.consume(curr_time):
                        return False

            return True

    def _get_or_create_bucket(self, arb_id: int) -> TokenBucket:
        """Retrieve or create token bucket configured at 1.25x nominal rate."""
        if arb_id not in self.token_buckets:
            msg_def = DBC.get(arb_id)
            period = msg_def.period_s if msg_def else 0.020
            nominal_rate = 1.0 / max(period, 0.001)
            # Token bucket at 1.25 x nominal rate, burst 2.0
            self.token_buckets[arb_id] = TokenBucket(rate_hz=1.25 * nominal_rate, burst=2.0)
        return self.token_buckets[arb_id]

    # -----------------------------------------------------------------------
    # Decision Enforcement
    # -----------------------------------------------------------------------

    def enforce(
        self,
        decision: Decision,
        frame: CANFrame,
        now: Optional[float] = None,
    ) -> Tuple[bool, List[str]]:
        """Enforce risk decision on an incoming CAN frame.

        Parameters
        ----------
        decision : Decision
            Fusion decision emitted by RiskEngine.
        frame : CANFrame
            Frame under inspection.
        now : Optional[float]
            Timestamp.

        Returns
        -------
        Tuple[bool, List[str]]
            (forward: bool, actions: list of textual action descriptions).
        """
        t_start = time.perf_counter()
        curr_time = frame.timestamp if now is None else now
        actions: List[str] = []
        forward = True

        with self._lock:
            # Update auto-recovery timers
            self.update_mode(curr_time)

            is_cmd = frame.arb_id in COMMAND_IDS
            is_unknown = frame.arb_id not in KNOWN_IDS

            # 1. CRITICAL Escalation and Blocking
            if decision.level == RiskLevel.CRITICAL:
                self.last_critical_time = curr_time
                forward = False

                if is_cmd:
                    self._set_mode(self.MODE_SAFE_MODE, curr_time, f"Critical command spoofing 0x{frame.arb_id:03X}")
                    if self.protection_enabled:
                        self.rejected_commands[frame.arb_id] += 1
                        act = f"Rejected spoofed command 0x{frame.arb_id:03X}"
                    else:
                        act = f"Observed spoofed command 0x{frame.arb_id:03X} (protection OFF: forwarded)"
                    actions.append(act)
                    self.recent_actions.append(act)
                else:
                    self._set_mode(self.MODE_DEGRADED, curr_time, f"Critical status compromise 0x{frame.arb_id:03X}")
                    if self.protection_enabled:
                        act = f"Blocked critical status frame 0x{frame.arb_id:03X}"
                    else:
                        act = f"Observed critical status frame 0x{frame.arb_id:03X} (protection OFF: forwarded)"
                    actions.append(act)
                    self.recent_actions.append(act)

                # Record incident
                self._record_incident(decision, f"0x{frame.arb_id:03X}", curr_time)

            # 2. HIGH Escalation and Filtering
            elif decision.level == RiskLevel.HIGH:
                self.last_high_time = curr_time
                if self.current_mode == self.MODE_NORMAL:
                    self._set_mode(self.MODE_ALERT, curr_time, f"High risk event on 0x{frame.arb_id:03X}")

                if is_unknown:
                    if self.protection_enabled:
                        self.quarantine_expiry[frame.arb_id] = curr_time + 5.0
                        forward = False
                        act = f"Quarantined unknown ID 0x{frame.arb_id:03X} for 5.0 s"
                    else:
                        act = f"Observed unknown ID 0x{frame.arb_id:03X} (protection OFF: forwarded)"
                    actions.append(act)
                    self.recent_actions.append(act)

                elif decision.attack_type == "Fuzzing":
                    forward = False
                    if self.protection_enabled:
                        act = f"Dropped fuzzed payload on known ID 0x{frame.arb_id:03X}"
                    else:
                        act = f"Observed fuzzed payload on 0x{frame.arb_id:03X} (protection OFF: forwarded)"
                    actions.append(act)
                    self.recent_actions.append(act)

                elif decision.attack_type == "Flooding":
                    if self.protection_enabled:
                        self.rate_limited_active_ids.add(frame.arb_id)
                        self.rate_limited_expiry[frame.arb_id] = curr_time + 1.5
                        forward = self.gate(frame, curr_time)
                        act = f"Rate-limited flooding stream 0x{frame.arb_id:03X}"
                    else:
                        act = f"Observed flooding stream 0x{frame.arb_id:03X} (protection OFF: forwarded)"
                    actions.append(act)
                    self.recent_actions.append(act)

                elif decision.attack_type in ("Spoofing", "Replay"):
                    forward = False
                    if self.protection_enabled:
                        if is_cmd:
                            self.rejected_commands[frame.arb_id] += 1
                            act = f"Rejected {decision.attack_type.lower()} command 0x{frame.arb_id:03X}"
                        else:
                            act = f"Dropped {decision.attack_type.lower()} frame 0x{frame.arb_id:03X}"
                    else:
                        act = f"Observed {decision.attack_type.lower()} on 0x{frame.arb_id:03X} (protection OFF: forwarded)"
                    actions.append(act)
                    self.recent_actions.append(act)

                else:
                    forward = False
                    if self.protection_enabled:
                        act = f"Dropped anomalous frame 0x{frame.arb_id:03X}"
                    else:
                        act = f"Observed anomalous frame 0x{frame.arb_id:03X} (protection OFF: forwarded)"
                    actions.append(act)
                    self.recent_actions.append(act)

                self._record_incident(decision, f"0x{frame.arb_id:03X}", curr_time)

            # 3. MEDIUM Alert
            elif decision.level == RiskLevel.MEDIUM:
                self.last_alert_time = curr_time
                if self.current_mode == self.MODE_NORMAL:
                    self._set_mode(self.MODE_ALERT, curr_time, f"Medium risk alert on 0x{frame.arb_id:03X}")

                # Medium: forward + ALERT
                forward = self.gate(frame, curr_time)
                act = f"Forwarded with alert 0x{frame.arb_id:03X}"
                actions.append(act)

            # 4. LOW Normal
            else:
                forward = self.gate(frame, curr_time)

            # Protection override
            if not self.protection_enabled:
                forward = True

            # ---------------------------------------------------------------
            # Lockout Watchdog (CONTRACT / FIX 3)
            # ---------------------------------------------------------------
            if self.protection_enabled and not is_unknown:
                hist = self._id_block_history[frame.arb_id]
                hist.append((curr_time, not forward, counter_of(frame.data)))
                while hist and hist[0][0] < (curr_time - 1.0):
                    hist.popleft()

                if len(hist) >= 10 and (curr_time - hist[0][0]) >= 0.90:
                    n_blk = sum(1 for item in hist if item[1])
                    if (n_blk / len(hist)) > 0.90:
                        consec = 0
                        max_consec = 0
                        for i in range(1, len(hist)):
                            if hist[i][2] == (hist[i-1][2] + 1) % 16:
                                consec += 1
                                max_consec = max(max_consec, consec)
                            else:
                                consec = 0
                        # Resync only if clean run of at least 3 consecutive strictly sequential counter increments
                        # and no active CRITICAL risk on this frame
                        if max_consec >= 3 and decision.level < RiskLevel.CRITICAL:
                            self.quarantine_expiry.pop(frame.arb_id, None)
                            self.rate_limited_active_ids.discard(frame.arb_id)
                            if self.resync_callback is not None:
                                self.resync_callback(frame.arb_id)
                            hist.clear()

                            last_inc = self._last_lockout_incident_ts.get(frame.arb_id, -999.0)
                            if (curr_time - last_inc) >= 5.0:
                                self._last_lockout_incident_ts[frame.arb_id] = curr_time
                                lockout_dec = Decision(
                                    level=RiskLevel.MEDIUM,
                                    score=0.45,
                                    attack_type="Lockout Watchdog",
                                    confidence=0.75,
                                    explanation=f"Forced resync on 0x{frame.arb_id:03X}: blocked >90% for >1s with consistent counters",
                                    action="Forced resynchronization",
                                )
                                self._record_incident(lockout_dec, f"0x{frame.arb_id:03X}", curr_time)
                                act = f"Lockout watchdog forced resync on 0x{frame.arb_id:03X}"
                                actions.append(act)
                                self.recent_actions.append(act)

            # ---------------------------------------------------------------
            # Command Starvation Check (CONTRACT / FIX 5)
            # ---------------------------------------------------------------
            if is_cmd and frame.label == "Normal":
                if not forward:
                    if self._legit_cmd_blocked_start is None:
                        self._legit_cmd_blocked_start = curr_time
                    elif (curr_time - self._legit_cmd_blocked_start) > 0.200:
                        self.starvation_warning = "CRITICAL: fail-safe starvation - legitimate commands blocked > 200ms"
                else:
                    self._legit_cmd_blocked_start = None
                    self.starvation_warning = None

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            self.failsafe_latencies.append(latency_ms)

            return forward, actions

    # -----------------------------------------------------------------------
    # V2X Enforcement
    # -----------------------------------------------------------------------

    def enforce_v2x(
        self,
        decision: Decision,
        msg: V2XMessage,
        now: Optional[float] = None,
    ) -> bool:
        """Enforce decision on V2X message.

        Returns True if message is accepted and forwarded to digital twin; False if rejected.
        """
        curr_time = msg.timestamp if now is None else now

        with self._lock:
            self.update_mode(curr_time)

            status_val = decision.components.get("status", "")
            is_rejected = (
                decision.level >= RiskLevel.HIGH
                or decision.action.startswith("reject")
                or decision.action.startswith("drop")
                or status_val == "rejected"
            )

            if is_rejected:
                if self.protection_enabled:
                    self.rejected_v2x_count += 1
                    act = f"Rejected unverified V2X {msg.msg_type} from {msg.sender_id}"
                else:
                    act = f"Observed unverified V2X {msg.msg_type} from {msg.sender_id} (protection OFF: accepted)"
                self.recent_actions.append(act)

                if decision.level == RiskLevel.CRITICAL:
                    self.last_critical_time = curr_time
                    self._set_mode(self.MODE_DEGRADED, curr_time, f"Critical V2X attack from {msg.sender_id}")
                else:
                    self.last_high_time = curr_time
                    if self.current_mode == self.MODE_NORMAL:
                        self._set_mode(self.MODE_ALERT, curr_time, f"High risk V2X from {msg.sender_id}")

                self._record_incident(decision, msg.sender_id, curr_time)
                return False if self.protection_enabled else True

            return True

    # -----------------------------------------------------------------------
    # Queue Prioritization
    # -----------------------------------------------------------------------

    def prioritize_queue(
        self,
        queue: List[CANFrame],
        backlog_threshold: int = 50,
    ) -> List[CANFrame]:
        """Prioritize safety-critical frames when queue backlog grows."""
        if len(queue) <= backlog_threshold:
            return queue

        # Process critical IDs first; shed flood frames from quarantined or rate-limited IDs
        critical_frames: List[CANFrame] = []
        regular_frames: List[CANFrame] = []

        for frame in queue:
            # Shed frames from actively quarantined IDs
            if frame.arb_id in self.quarantine_expiry:
                continue
            if frame.arb_id in SAFETY_CRITICAL_IDS or frame.arb_id in COMMAND_IDS:
                critical_frames.append(frame)
            else:
                regular_frames.append(frame)

        # Retain all critical frames and truncate regular backlog to avoid buffer starvation
        return critical_frames + regular_frames[:backlog_threshold]

    # -----------------------------------------------------------------------
    # Forensic Incident Logging & Merging
    # -----------------------------------------------------------------------

    def _record_incident(
        self,
        decision: Decision,
        target: str,
        now: float,
    ) -> None:
        """Record forensic incident into SQLite and JSONL with repeat deduplication."""
        # Check for incident merging: floods merge as ONE incident with running count and duration
        is_flood = (decision.attack_type == "Flooding")
        can_merge = False

        if self._last_merged_incident is not None:
            same_target = (self._last_merged_incident.target == target)
            same_type = (self._last_merged_incident.attack_type == decision.attack_type)
            if is_flood and same_target and same_type:
                can_merge = (now - self._last_merged_incident.ts) <= 5.0
            elif same_target and same_type and self._last_merged_incident.level == decision.level.name:
                can_merge = (now - self._last_merged_incident.ts) <= self.incident_merge_s

        if can_merge:
            # Merge into previous incident
            self._last_merged_incident.count += 1
            self._last_merged_incident.ts = now
            if is_flood:
                dur = max(0.01, now - self._last_merged_incident.start_ts)
                self._last_merged_incident.explanation = (
                    f"Flood on {target}: {self._last_merged_incident.count} msgs, rate-limited, {dur:.1f}s"
                )
            else:
                self._last_merged_incident.explanation = decision.explanation
            inc_id = self._last_merged_incident.id

            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE incidents SET count = ?, ts = ?, explanation = ? WHERE id = ?",
                        (self._last_merged_incident.count, now, self._last_merged_incident.explanation, inc_id),
                    )
                    conn.commit()
            except Exception as exc:
                logger.debug("SQLite update failed: %s", exc)

            return

        # New Incident
        self.incident_id_counter += 1
        expl_text = decision.explanation
        if is_flood:
            expl_text = f"Flood on {target}: 1 msgs, rate-limited, 0.0s"

        assumed_ep = resolve_entry_point(decision.attack_type, self.cfg)

        rec = IncidentRecord(
            id=self.incident_id_counter,
            ts=now,
            attack_type=decision.attack_type,
            level=decision.level.name,
            target=target,
            confidence=decision.confidence,
            explanation=expl_text,
            action=decision.action,
            latency_ms=decision.components.get("latency_ms", 0.5),
            count=1,
            details=json.dumps(decision.components),
            start_ts=now,
            assumed_entry_point=assumed_ep,
        )

        self._last_merged_incident = rec
        self.incidents_buffer.appendleft(rec)

        # Write to SQLite
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO incidents (id, ts, attack_type, level, target, confidence, explanation, action, latency_ms, count, details, assumed_entry_point)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rec.id,
                        rec.ts,
                        rec.attack_type,
                        rec.level,
                        rec.target,
                        rec.confidence,
                        rec.explanation,
                        rec.action,
                        rec.latency_ms,
                        rec.count,
                        rec.details,
                        rec.assumed_entry_point,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("SQLite insert failed: %s", exc)

        # Write to JSONL
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec)) + "\n")
        except Exception as exc:
            logger.debug("JSONL append failed: %s", exc)

    # -----------------------------------------------------------------------
    # System Snapshot
    # -----------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Generate failsafe state dictionary for system snapshot matching CONTRACT item 61."""
        with self._lock:
            return {
                "rejected_commands": dict(self.rejected_commands),
                "rate_limited": [f"0x{i:03X}" for i in sorted(self.rate_limited_active_ids)],
                "quarantined": [f"0x{i:03X}" for i in sorted(self.quarantine_expiry.keys())],
                "rejected_v2x": self.rejected_v2x_count,
                "mode_history": list(self.mode_history[-20:]),
                "actions": list(self.recent_actions),
                "incident_id": self.incident_id_counter,
                "starvation_warning": self.starvation_warning,
            }

    def get_recent_incidents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent incidents formatted for CONTRACT item 62."""
        with self._lock:
            return [rec.to_dashboard_dict() for rec in list(self.incidents_buffer)[:limit]]
