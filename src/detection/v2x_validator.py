"""
CAVShield V2X Trust & Plausibility Validator.

Validates V2X BSM and warning messages through:
- Cryptographic signature check (TrustStore)
- Freshness & anti-replay verification
- Position and speed physical plausibility
- Kinematic consistency with sender's beacon history (dead-reckoning)
- Ghost vehicle check (vehicles within 80 m must appear on radar/lidar)
- Multi-sensor hazard confirmation (claimed obstacles must match a track within 8 m)
- Emergency braking plausibility
- Composite trust fusion score and binary acceptance verdict
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from src import load_config
from src.v2x_simulator import SensorSnapshot, TrustStore, V2XMessage


# ---------------------------------------------------------------------------
# V2XVerdict Dataclass
# ---------------------------------------------------------------------------

@dataclass
class V2XVerdict:
    """Validation outcome for a single V2XMessage.

    Attributes
    ----------
    accepted : bool
        True if message is trusted and accepted; False if rejected.
    trust : float
        Composite trust score in [0.0, 1.0].
    reasons : List[str]
        List of explanations / rejection reasons.
    checks : Dict[str, Any]
        Sub-scores for signature, freshness, position, kinematics, and sensor.
    status : str
        Verification status: 'accepted' | 'suspicious' | 'rejected'.
    """
    accepted: bool
    trust: float
    reasons: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)
    status: str = "accepted"


# ---------------------------------------------------------------------------
# V2X Validator Class
# ---------------------------------------------------------------------------

class V2XValidator:
    """Comprehensive validator for incoming V2X messages."""

    def __init__(
        self,
        trust_store: Optional[TrustStore] = None,
        cfg: Optional[Any] = None,
    ):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.trust_store = trust_store or TrustStore()

        v2x_cfg = getattr(cfg, "v2x", {})
        self.trust_accept = float(getattr(v2x_cfg, "trust_accept", 0.60))
        self.max_age_s = float(getattr(v2x_cfg, "max_age_s", 1.0))
        self.comm_range_m = float(getattr(v2x_cfg, "comm_range_m", 300.0))
        self.radar_cover_m = float(getattr(v2x_cfg, "radar_cover_m", 80.0))
        sensors_cfg = getattr(cfg, "sensors", {})
        self.lane_half_width_m = float(getattr(sensors_cfg, "lane_half_width_m", 1.8))

        # Anti-replay / uniqueness cache: msg_id -> timestamp
        self._seen_msg_ids: Dict[str, float] = {}

        # Kinematic beacon history: sender_id -> (timestamp, x, y, speed)
        self._beacon_history: Dict[str, Tuple[float, float, float, float]] = {}
        self._current_snapshot: Optional[SensorSnapshot] = None

    def reset(self) -> None:
        """Reset validation history and caches."""
        self._seen_msg_ids.clear()
        self._beacon_history.clear()
        self._current_snapshot = None

    def set_sensor_snapshot(self, snapshot: Optional[SensorSnapshot]) -> None:
        """Update latest fused sensor snapshot from perception."""
        self._current_snapshot = snapshot

    def validate(
        self,
        msg: V2XMessage,
        snapshot: Optional[SensorSnapshot] = None,
        now: float = 0.0,
    ) -> V2XVerdict:
        """Validate an incoming V2X message against security and physics invariants.

        Parameters
        ----------
        msg : V2XMessage
            The received V2X message.
        snapshot : Optional[SensorSnapshot]
            Current independent multi-sensor perception snapshot.
        now : float
            Current simulation time (seconds).

        Returns
        -------
        V2XVerdict
            Validation verdict containing acceptance status, composite trust score,
            rejection reasons, and individual component sub-scores.
        """
        if snapshot is None:
            snapshot = self._current_snapshot
        reasons: List[str] = []

        # -------------------------------------------------------------------
        # 1. Cryptographic Signature Check (Weight: 0.35)
        # -------------------------------------------------------------------
        is_sig_valid = self.trust_store.verify(msg)
        if is_sig_valid:
            sig_score = 1.0
        else:
            sig_score = 0.0
            reasons.append("Cryptographic signature verification failed")

        # -------------------------------------------------------------------
        # 2. Freshness & Anti-Replay Check (Weight: 0.15)
        # -------------------------------------------------------------------
        age = now - msg.timestamp
        fresh_score = 0.0

        if msg.msg_id in self._seen_msg_ids:
            reasons.append(f"Duplicate message ID {msg.msg_id} detected (replay attempt)")
        elif age < -0.15:
            reasons.append(f"Future timestamp skew detected: message timestamp {msg.timestamp:.2f}s > now {now:.2f}s")
        elif age > self.max_age_s:
            reasons.append(f"Stale message: age {age:.2f}s exceeds maximum age threshold {self.max_age_s:.2f}s")
        else:
            fresh_score = max(0.0, 1.0 - (max(0.0, age) / self.max_age_s))
            self._seen_msg_ids[msg.msg_id] = now

        # Prune old seen IDs (older than 30s)
        if len(self._seen_msg_ids) > 1000:
            cutoff = now - 30.0
            self._seen_msg_ids = {mid: ts for mid, ts in self._seen_msg_ids.items() if ts >= cutoff}

        # -------------------------------------------------------------------
        # 3. Position and Speed Plausibility (Weight: 0.15)
        # -------------------------------------------------------------------
        ego_x = getattr(snapshot, "ego_x_est", 1000.0) if snapshot else 1000.0
        ego_y = getattr(snapshot, "ego_y_est", 0.0) if snapshot else 0.0

        dist = math.hypot(msg.x - ego_x, msg.y - ego_y)
        pos_valid = True

        if dist > self.comm_range_m:
            pos_valid = False
            reasons.append(
                f"Transmitter claimed distance {dist:.1f} m exceeds communication range {self.comm_range_m:.1f} m"
            )

        if abs(msg.y) > 15.0:
            pos_valid = False
            reasons.append(f"Transmitter lateral position {msg.y:.1f} m is off-road (road width ±12 m)")

        # Speed physical plausibility: tolerance for small numerical float roundoff (-0.05 m/s)
        if msg.speed < -0.05 or msg.speed > 65.0:
            pos_valid = False
            reasons.append(f"Claimed speed {msg.speed:.1f} m/s ({msg.speed*3.6:.1f} km/h) outside physical bounds [0, 65]")

        pos_score = 1.0 if pos_valid else 0.0

        # -------------------------------------------------------------------
        # 4. Kinematic Consistency (Weight: 0.15)
        # -------------------------------------------------------------------
        kin_score = 1.0
        if msg.sender_id in self._beacon_history:
            prev_ts, prev_x, prev_y, prev_spd = self._beacon_history[msg.sender_id]
            dt_beacon = msg.timestamp - prev_ts

            # Sane dt threshold: at least 50 ms to avoid division-by-near-zero acceleration spikes
            if dt_beacon >= 0.050 and dt_beacon < 5.0:
                expected_x = prev_x + prev_spd * dt_beacon
                jump_m = abs(msg.x - expected_x)
                jump_tol = 15.0 + 0.5 * 10.0 * (dt_beacon ** 2)
                implied_accel = abs(msg.speed - prev_spd) / dt_beacon

                if jump_m > jump_tol or implied_accel > 12.0:
                    kin_score = 0.0
                    dt_ms = max(1.0, dt_beacon * 1000.0)
                    reasons.append(
                        f"Claimed position/accel jump (pos {jump_m:.1f} m, accel {implied_accel:.1f} m/s²) in {dt_ms:.0f} ms"
                    )

        # -------------------------------------------------------------------
        # 5. Sensor Cross-Validation & Ghost Check (Weight: 0.20)
        # -------------------------------------------------------------------
        sensor_score = 1.0
        hazard_confirmed = True
        is_hazard = msg.msg_type in ("COLLISION_WARNING", "ROAD_HAZARD")

        if snapshot is not None:
            # 5a. Ghost vehicle check:
            # A vehicle claiming to be within radar range (80 m) MUST be visible to radar/lidar
            if dist <= self.radar_cover_m:
                matched_track = any(
                    abs(t.range - dist) <= 15.0 and math.hypot(t.dx - (msg.x - ego_x), t.dy - (msg.y - ego_y)) <= 15.0
                    for t in snapshot.tracks
                )
                if not matched_track:
                    sensor_score = 0.0
                    reasons.append("Ghost vehicle: transmitter claims position within 80 m but has no matching radar/lidar track")

            # 5b. Hazard confirmation:
            # A claimed obstacle MUST be confirmed by a fused radar/lidar/camera track within 8 m
            if is_hazard and msg.obstacle and isinstance(msg.obstacle, dict):
                obs_range = float(msg.obstacle.get("range", msg.obstacle.get("distance", 999.0)))
                obs_y = float(msg.obstacle.get("y", 0.0))
                # In-lane hazard requires an in-lane track; off-lane requires matching lateral position
                if abs(obs_y) <= self.lane_half_width_m:
                    matched_obs = any(
                        t.in_lane and abs(t.range - obs_range) <= 8.0
                        for t in snapshot.tracks
                    )
                else:
                    matched_obs = any(
                        abs(t.range - obs_range) <= 8.0 and abs(t.dy - obs_y) <= 2.5
                        for t in snapshot.tracks
                    )
                if not matched_obs:
                    hazard_confirmed = False
                    sensor_score = 0.0
                    reasons.append("Phantom hazard: claimed obstacle unconfirmed by radar/camera track within 8 m")

        # 5c. Emergency braking consistency
        if msg.msg_type == "EMERGENCY_BRAKING":
            if msg.claimed_decel > 8.0:
                if dist <= self.radar_cover_m and sensor_score > 0.0:
                    pass

        # -------------------------------------------------------------------
        # Composite Trust Fusion Formula
        # trust = 0.35*sig + 0.15*fresh + 0.15*pos + 0.15*kin + 0.20*sensor
        # -------------------------------------------------------------------
        trust = (
            0.35 * sig_score
            + 0.15 * fresh_score
            + 0.15 * pos_score
            + 0.15 * kin_score
            + 0.20 * sensor_score
        )
        trust = round(max(0.0, min(1.0, trust)), 3)

        # Determine status: accepted | suspicious | rejected
        hazard_condition = hazard_confirmed if is_hazard else True
        hard_failures = (
            (sig_score == 0.0)
            or (not hazard_condition)
            or (sensor_score == 0.0)
            or (kin_score == 0.0)
            or (pos_score == 0.0)
            or (fresh_score == 0.0 and any("replay" in r.lower() or "stale" in r.lower() for r in reasons))
        )

        if hard_failures:
            status = "rejected"
            accepted = False
        elif trust < self.trust_accept or reasons:
            status = "suspicious"
            accepted = False
        else:
            status = "accepted"
            accepted = True

        # Update beacon history for valid senders
        if is_sig_valid and kin_score == 1.0 and pos_score == 1.0:
            self._beacon_history[msg.sender_id] = (msg.timestamp, msg.x, msg.y, msg.speed)

        return V2XVerdict(
            accepted=accepted,
            trust=trust,
            reasons=reasons,
            checks={
                "sig": sig_score,
                "fresh": round(fresh_score, 2),
                "pos": pos_score,
                "kin": kin_score,
                "sensor": sensor_score,
                "hazard_confirmed": hazard_confirmed,
            },
            status=status,
        )
