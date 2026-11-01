"""
CAVShield CAN Rule Engine.

Implements frame-level deterministic rule checks:
- UNKNOWN_ID
- DLC_MISMATCH
- CHECKSUM
- COUNTER freshness versus last ACCEPTED counter
- TIMING inter-arrival versus last ACCEPTED frame
- RATE_FLOOD stream rate over sliding 0.25 s window

Follows the safety core architecture: evaluate() does not mutate state (except
rate counters) so legitimate frames interleaved with attacks are never penalized.
commit() is called only for forwarded frames.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
import hashlib
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from src import load_config
from src.can_simulator import (
    DBC,
    KNOWN_IDS,
    CANFrame,
    checksum_ok,
    counter_of,
)


# ---------------------------------------------------------------------------
# Evidence Dataclass (CONTRACT Item 41)
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """Detection evidence matching CONTRACT item 41.

    Attributes
    ----------
    source : str
        Origin of evidence ('rule', 'physics', or 'v2x').
    name : str
        Specific rule or check name (e.g. 'CHECKSUM', 'COUNTER', 'RATE_FLOOD').
    severity : float
        Confidence / severity of anomaly in [0.0, 1.0].
    detail : str
        Human-readable explanation containing specific numbers and context.
    hint : str
        Attack class hypothesis ('Flooding', 'Fuzzing', 'Replay', 'Spoofing', or '').
    """
    source: str
    name: str
    severity: float
    detail: str
    hint: str = ""


# ---------------------------------------------------------------------------
# CAN Rule Engine Class
# ---------------------------------------------------------------------------

class CANRuleEngine:
    """Deterministic frame-level CAN rule verification engine."""

    def __init__(self, cfg: Optional[Any] = None, rate_window_s: float = 0.25):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        feat_cfg = getattr(cfg, "features", {})
        self.rate_window_s = float(getattr(feat_cfg, "rate_window_s", rate_window_s))

        rules_cfg = getattr(cfg, "rules", {})
        self.checksum_weight = float(getattr(rules_cfg, "checksum_weight", 0.80))
        self.counter_weight = float(getattr(rules_cfg, "counter_weight", 0.55))
        self.rate_anomaly_weight = float(getattr(rules_cfg, "rate_anomaly_weight", 0.90))
        self.unknown_id_weight = float(getattr(rules_cfg, "unknown_id_weight", 0.85))
        self.dlc_mismatch_weight = float(getattr(rules_cfg, "dlc_mismatch_weight", 0.60))

        # Committed state (only updated on commit() for accepted frames)
        self._last_accepted_ts: Dict[int, float] = {}
        self._last_accepted_counter: Dict[int, int] = {}
        self._seen_payload_hashes: Dict[int, Set[int]] = collections.defaultdict(set)

        # Rate tracking deques: updated per frame on evaluate() to monitor live bus rates
        self._rate_deques: Dict[int, Deque[float]] = collections.defaultdict(collections.deque)
        self._rate_high_start: Dict[int, float] = {}
        self._recent_timing_violations: Dict[int, float] = {}

        # Incoming frame history for resync tracking: arb_id -> deque of (ts, counter, chk_ok)
        self._incoming_history: Dict[int, Deque[Tuple[float, int, bool]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=10)
        )

    def reset(self) -> None:
        """Reset all committed state and rate tracking buffers."""
        self._last_accepted_ts.clear()
        self._last_accepted_counter.clear()
        self._seen_payload_hashes.clear()
        self._rate_deques.clear()
        self._rate_high_start.clear()
        self._recent_timing_violations.clear()
        self._incoming_history.clear()

    def force_resync(self, arb_id: int) -> None:
        """Force an immediate baseline reset for a specific CAN ID (lockout watchdog)."""
        self._last_accepted_ts.pop(arb_id, None)
        self._last_accepted_counter.pop(arb_id, None)
        self._incoming_history[arb_id].clear()
        self._rate_deques[arb_id].clear()
        self._rate_high_start.pop(arb_id, None)
        self._recent_timing_violations.pop(arb_id, None)

    def stream_rate_ratio(self, arb_id: int) -> float:
        """Return the ratio of current observed stream rate to nominal rate."""
        q = self._rate_deques.get(arb_id)
        if not q:
            return 0.0

        current_rate_hz = len(q) / self.rate_window_s
        mdef = DBC.get(arb_id)
        if mdef is not None and mdef.period_s > 0:
            nominal_hz = 1.0 / mdef.period_s
            return current_rate_hz / nominal_hz
        return current_rate_hz / 10.0  # nominal default for unknown IDs

    def evaluate(self, frame: CANFrame, backlog: int = 0) -> List[Evidence]:
        """Evaluate deterministic rules against a single incoming frame.

        Does not mutate committed state. Updates stream rate counters only.

        Parameters
        ----------
        frame : CANFrame
            Incoming CAN frame.
        backlog : int
            Current pipeline queue backlog length (used for smoothed flood estimation).

        Returns
        -------
        List[Evidence]
            List of rule violation evidence (empty if frame is completely valid).
        """
        evidence_list: List[Evidence] = []
        now = frame.timestamp
        arb_id = frame.arb_id
        data = frame.data

        # 1. Update live rate tracking deque with burst deduplication & clock jump protection
        q = self._rate_deques[arb_id]
        cutoff = now - self.rate_window_s

        # Purge entries older than cutoff or contaminated by clock skew (e.g. non-monotonic jumps)
        while q and (q[0] < cutoff or q[0] > (now + 0.5)):
            q.popleft()

        # Frames sharing nearly the same timestamp (within 0.2 ms) belong to a single burst
        is_same_burst = bool(q and abs(now - q[-1]) <= 0.0002)
        if not is_same_burst:
            q.append(now)

        current_rate_hz = len(q) / self.rate_window_s

        # -------------------------------------------------------------------
        # Rule 1: UNKNOWN_ID
        # -------------------------------------------------------------------
        is_known = arb_id in KNOWN_IDS
        if not is_known:
            evidence_list.append(Evidence(
                source="rule",
                name="UNKNOWN_ID",
                severity=self.unknown_id_weight,
                detail=f"Unknown CAN ID 0x{arb_id:03X} not in DBC catalog",
                hint="Fuzzing",
            ))

        # -------------------------------------------------------------------
        # Rule 2: DLC_MISMATCH
        # -------------------------------------------------------------------
        if frame.dlc != 8 or len(data) != 8:
            evidence_list.append(Evidence(
                source="rule",
                name="DLC_MISMATCH",
                severity=self.dlc_mismatch_weight,
                detail=f"DLC mismatch for 0x{arb_id:03X}: expected 8 bytes, got {frame.dlc}",
                hint="Fuzzing",
            ))
            # Cannot safely evaluate checksum or counter if payload length is invalid
            return evidence_list

        # -------------------------------------------------------------------
        # Rule 3: CHECKSUM
        # -------------------------------------------------------------------
        chk_ok = checksum_ok(data)
        if not chk_ok:
            expected_chk = data[0] ^ data[1] ^ data[2] ^ data[3] ^ data[4] ^ data[5]
            evidence_list.append(Evidence(
                source="rule",
                name="CHECKSUM",
                severity=self.checksum_weight,
                detail=(
                    f"Corrupt XOR checksum on 0x{arb_id:03X}: byte 6 is 0x{data[6]:02X}, "
                    f"expected 0x{expected_chk:02X}"
                ),
                hint="Fuzzing",
            ))

        # Record incoming history for counter & timing resynchronization
        curr_counter = counter_of(data)
        self._incoming_history[arb_id].append((now, curr_counter, chk_ok))

        # RESYNC CHECK (CONTRACT / FIX 1):
        # Baselines for counter, timing and physics state must not depend only on accepted frames.
        # If an ID shows a clean run of 3 consecutive frames (counter +1 each, checksum valid,
        # timing on schedule) it re-baselines the counter.
        history = self._incoming_history[arb_id]
        if is_known and len(history) >= 3:
            h0, h1, h2 = history[-3], history[-2], history[-1]
            nom_p = DBC[arb_id].period_s
            c_ok = (h1[1] == (h0[1] + 1) % 16) and (h2[1] == (h1[1] + 1) % 16)
            k_ok = h0[2] and h1[2] and h2[2]
            dt1 = h1[0] - h0[0]
            dt2 = h2[0] - h1[0]
            t_ok = (0.40 * nom_p <= dt1 <= 2.5 * nom_p) and (0.40 * nom_p <= dt2 <= 2.5 * nom_p)

            if c_ok and k_ok and t_ok:
                # Re-baseline counter to previous frame in clean streak
                self._last_accepted_counter[arb_id] = h1[1]
                self._last_accepted_ts[arb_id] = h1[0]

        # -------------------------------------------------------------------
        # Rule 4: COUNTER Freshness versus last ACCEPTED counter
        # -------------------------------------------------------------------
        if is_known and arb_id in self._last_accepted_counter:
            last_counter = self._last_accepted_counter[arb_id]
            expected_counter = (last_counter + 1) % 16

            # Sequential tolerance: accept +1 or +2 (in case of single dropped frame)
            delta = (curr_counter - last_counter) % 16
            if delta not in (1, 2):
                # Check if this exact payload was previously observed for this ID
                payload_hash = hash(data)
                seen_before = payload_hash in self._seen_payload_hashes[arb_id]

                if seen_before:
                    sev = 0.85
                    hint_type = "Replay"
                    extra = " (payload previously observed - replay detected)"
                else:
                    sev = self.counter_weight
                    hint_type = "Spoofing"
                    extra = ""

                evidence_list.append(Evidence(
                    source="rule",
                    name="COUNTER",
                    severity=sev,
                    detail=(
                        f"Alive counter discontinuity on 0x{arb_id:03X}: received {curr_counter}, "
                        f"expected {expected_counter} (last accepted {last_counter}){extra}"
                    ),
                    hint=hint_type,
                ))

        # -------------------------------------------------------------------
        # Rule 5: TIMING (Inter-arrival < 0.45 x nominal period since last ACCEPTED)
        # -------------------------------------------------------------------
        has_timing_violation = False
        if is_known:
            period_s = DBC[arb_id].period_s
            dt = None
            if arb_id in self._last_accepted_ts:
                dt = now - self._last_accepted_ts[arb_id]
            elif len(self._incoming_history[arb_id]) >= 2:
                dt = now - self._incoming_history[arb_id][-2][0]

            if dt is not None and dt < (0.45 * period_s):
                has_timing_violation = True
                self._recent_timing_violations[arb_id] = now
                evidence_list.append(Evidence(
                    source="rule",
                    name="TIMING",
                    severity=0.50,
                    detail=(
                        f"Timing violation on 0x{arb_id:03X}: inter-arrival dt={dt*1000.0:.1f}ms "
                        f"< 0.45x nominal period ({0.45*period_s*1000.0:.1f}ms)"
                    ),
                    hint="Spoofing",
                ))

        # -------------------------------------------------------------------
        # Rule 6: RATE_FLOOD (Fixed window rate with sustained duration & off-schedule check)
        # -------------------------------------------------------------------
        recent_timing_violation = (now - self._recent_timing_violations.get(arb_id, -100.0)) <= 0.50
        timing_violation_active = has_timing_violation or recent_timing_violation

        if is_known:
            nominal_hz = 1.0 / DBC[arb_id].period_s
            ratio = current_rate_hz / nominal_hz
            is_high_rate = (len(q) >= 6 and ratio > 2.5)

            if is_high_rate:
                if arb_id not in self._rate_high_start:
                    self._rate_high_start[arb_id] = now
                high_duration = now - self._rate_high_start[arb_id]
            else:
                self._rate_high_start.pop(arb_id, None)
                high_duration = 0.0

            # Requirements:
            # - Flood needs a minimum duration (e.g. 0.3 s above 2.5x)
            # - Known-ID flood must also show off-schedule frames (timing violations), not just high count
            # - If backlog exists, never raise flood verdict unless sustained over 1 s
            if backlog > 0:
                duration_ok = (high_duration >= 1.0)
            else:
                duration_ok = (high_duration >= 0.30) or (len(q) >= 20 and ratio >= 3.0)

            if is_high_rate and duration_ok and timing_violation_active:
                evidence_list.append(Evidence(
                    source="rule",
                    name="RATE_FLOOD",
                    severity=self.rate_anomaly_weight,
                    detail=(
                        f"Rate flood on 0x{arb_id:03X}: {current_rate_hz:.1f} Hz "
                        f"({ratio:.1f}x nominal {nominal_hz:.1f} Hz sustained {high_duration:.2f}s)"
                    ),
                    hint="Flooding",
                ))
        else:
            # Unknown ID stream rate check: above 40 Hz
            is_high_rate = (len(q) >= 6 and current_rate_hz > 40.0)
            if is_high_rate:
                if arb_id not in self._rate_high_start:
                    self._rate_high_start[arb_id] = now
                high_duration = now - self._rate_high_start[arb_id]
            else:
                self._rate_high_start.pop(arb_id, None)
                high_duration = 0.0

            if backlog > 0:
                duration_ok = (high_duration >= 1.0)
            else:
                duration_ok = (high_duration >= 0.30) or (len(q) >= 20 and current_rate_hz >= 60.0)

            if is_high_rate and duration_ok:
                evidence_list.append(Evidence(
                    source="rule",
                    name="RATE_FLOOD",
                    severity=self.rate_anomaly_weight,
                    detail=(
                        f"Flood on unknown ID 0x{arb_id:03X}: {current_rate_hz:.1f} Hz "
                        f"exceeds 40 Hz threshold sustained {high_duration:.2f}s"
                    ),
                    hint="Flooding",
                ))

        return evidence_list

    def commit(self, frame: CANFrame) -> None:
        """Commit an accepted / forwarded frame to update baseline state.

        Call ONLY for frames that passed the gateway verification.
        """
        arb_id = frame.arb_id
        self._last_accepted_ts[arb_id] = frame.timestamp
        self._last_accepted_counter[arb_id] = counter_of(frame.data)
        self._seen_payload_hashes[arb_id].add(hash(frame.data))
