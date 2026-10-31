"""
CAVShield Feature Engineering.

Implements stateful, streaming O(1) CAN feature extraction producing the
17-feature vector in CONTRACT order, and V2X plausibility / consistency extraction.
"""

from __future__ import annotations

import collections
import math
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

from src import load_config
from src.can_simulator import is_safety_critical
from src.v2x_simulator import SensorSnapshot, TrustStore, V2XMessage


# ---------------------------------------------------------------------------
# Feature Names Definition (Exact CONTRACT Order)
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "can_id_norm",
    "dlc",
    "b0",
    "b1",
    "b2",
    "b3",
    "b4",
    "b5",
    "b6",
    "b7",
    "dt_same_id",
    "dt_global",
    "freq_id",
    "payload_change_rate",
    "id_entropy",
    "repeated_payload",
    "safety_critical",
]

assert len(FEATURE_NAMES) == 17, f"Expected 17 features, got {len(FEATURE_NAMES)}"


# ---------------------------------------------------------------------------
# Stateful Streaming CAN Feature Extractor
# ---------------------------------------------------------------------------

class CANFeatureExtractor:
    """Stateful, streaming, O(1) per-frame CAN feature extractor.

    Maintains:
    - Per-ID state: last timestamp, last payload, sliding window of timestamps
      for rate calculation over rate_window_s.
    - Global state: last global timestamp, rolling 64-ID Counter for Shannon entropy.
    """

    def __init__(self, cfg: Optional[Any] = None, rate_window_s: float = 0.25):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        feat_cfg = getattr(cfg, "features", {})
        self.rate_window_s = float(getattr(feat_cfg, "rate_window_s", rate_window_s))

        self.reset()

    def reset(self) -> None:
        """Reset internal streaming state (called between independent runs)."""
        # Per-ID state
        self._last_ts_by_id: Dict[int, float] = {}
        self._last_payload_by_id: Dict[int, bytes] = {}
        self._rate_deques_by_id: Dict[int, Deque[float]] = {}

        # Global state
        self._last_global_ts: Optional[float] = None
        self._rolling_ids: Deque[int] = collections.deque(maxlen=64)
        self._id_counter: collections.Counter = collections.Counter()

    def extract(
        self,
        arb_id: int,
        data: Union[bytes, bytearray, List[int]],
        ts: float,
        dlc: int = 8,
    ) -> np.ndarray:
        """Extract 17-element float feature vector for a single CAN frame in O(1).

        Parameters
        ----------
        arb_id : int
            CAN arbitration ID.
        data : bytes
            8-byte payload.
        ts : float
            Timestamp of message in seconds.
        dlc : int
            Data length code (0-8).

        Returns
        -------
        np.ndarray
            1D array of shape (17,) containing extracted features.
        """
        if isinstance(data, (bytes, bytearray)):
            payload = bytes(data)
        elif isinstance(data, str):
            try:
                clean_hex = data.strip().replace(" ", "").replace("0x", "")
                if len(clean_hex) % 2 != 0:
                    clean_hex = "0" + clean_hex
                payload = bytes.fromhex(clean_hex)
            except Exception:
                payload = data.encode("latin1", errors="replace")
        elif isinstance(data, (list, tuple)):
            payload = bytes(int(x) & 0xFF for x in data)
        else:
            try:
                payload = bytes(data)
            except Exception:
                payload = b"\x00" * 8

        if len(payload) < 8:
            payload = payload.ljust(8, b"\x00")
        elif len(payload) > 8:
            payload = payload[:8]

        # 1. can_id_norm: 11-bit standard ID normalization [0, 1]
        can_id_norm = min(1.0, max(0.0, arb_id / 2047.0))

        # 2. dlc
        dlc_val = float(dlc)

        # 3..10. b0..b7 divided by 255.0
        b0 = payload[0] / 255.0
        b1 = payload[1] / 255.0
        b2 = payload[2] / 255.0
        b3 = payload[3] / 255.0
        b4 = payload[4] / 255.0
        b5 = payload[5] / 255.0
        b6 = payload[6] / 255.0
        b7 = payload[7] / 255.0

        # 11. dt_same_id (log-scaled)
        if arb_id in self._last_ts_by_id:
            dt_same = max(0.0, ts - self._last_ts_by_id[arb_id])
        else:
            dt_same = 0.05  # Default nominal 50 ms for initial frame
        self._last_ts_by_id[arb_id] = ts
        dt_same_id_log = math.log1p(dt_same * 1000.0)

        # 12. dt_global (log-scaled)
        if self._last_global_ts is not None:
            dt_glob = max(0.0, ts - self._last_global_ts)
        else:
            dt_glob = 0.002  # ~2 ms nominal for initial frame
        self._last_global_ts = ts
        dt_global_log = math.log1p(dt_glob * 1000.0)

        # 13. freq_id over rate_window_s
        if arb_id not in self._rate_deques_by_id:
            self._rate_deques_by_id[arb_id] = collections.deque()
        q = self._rate_deques_by_id[arb_id]
        cutoff = ts - self.rate_window_s
        while q and q[0] < cutoff:
            q.popleft()
        q.append(ts)
        freq_id = len(q) / self.rate_window_s

        # 14. payload_change_rate (bit hamming distance / 64) & 16. repeated_payload
        if arb_id in self._last_payload_by_id:
            prev = self._last_payload_by_id[arb_id]
            if prev == payload:
                repeated_payload = 1.0
                payload_change_rate = 0.0
            else:
                repeated_payload = 0.0
                curr_int = int.from_bytes(payload, "big")
                prev_int = int.from_bytes(prev, "big")
                diff_bits = (curr_int ^ prev_int).bit_count()
                payload_change_rate = diff_bits / 64.0
        else:
            repeated_payload = 0.0
            payload_change_rate = 0.0
        self._last_payload_by_id[arb_id] = payload

        # 15. id_entropy (rolling 64 IDs)
        if len(self._rolling_ids) == 64:
            popped_id = self._rolling_ids[0]
            self._id_counter[popped_id] -= 1
            if self._id_counter[popped_id] <= 0:
                del self._id_counter[popped_id]

        self._rolling_ids.append(arb_id)
        self._id_counter[arb_id] += 1

        total_ids = len(self._rolling_ids)
        entropy = 0.0
        for count in self._id_counter.values():
            p = count / total_ids
            entropy -= p * math.log2(p)
        id_entropy = entropy

        # 17. safety_critical flag
        safety_crit = 1.0 if is_safety_critical(arb_id) else 0.0

        return np.array([
            can_id_norm,
            dlc_val,
            b0, b1, b2, b3, b4, b5, b6, b7,
            dt_same_id_log,
            dt_global_log,
            freq_id,
            payload_change_rate,
            id_entropy,
            repeated_payload,
            safety_crit,
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# V2X Plausibility & Consistency Features
# ---------------------------------------------------------------------------

def extract_v2x_features(
    msg: V2XMessage,
    ego_x: float,
    ego_y: float,
    ego_speed: float,
    sensor_snapshot: Optional[SensorSnapshot] = None,
    now: float = 0.0,
    trust_store: Optional[TrustStore] = None,
    cfg: Optional[Any] = None,
) -> Dict[str, float]:
    """Extract physical and cryptographic plausibility features from a V2XMessage.

    Returns
    -------
    dict of str: float
        Keys: 'freshness', 'position_plausibility', 'speed_plausibility',
              'trust_score', 'sensor_consistency'.
    """
    if cfg is None:
        cfg = load_config()

    v2x_cfg = getattr(cfg, "v2x", {})
    max_age_s = float(getattr(v2x_cfg, "max_age_s", 1.0))
    comm_range_m = float(getattr(v2x_cfg, "comm_range_m", 300.0))
    radar_cover_m = float(getattr(v2x_cfg, "radar_cover_m", 80.0))

    # 1. Freshness: 1.0 for new messages, decaying to 0.0 at max_age_s
    age = max(0.0, now - msg.timestamp)
    freshness = max(0.0, min(1.0, 1.0 - (age / max_age_s)))

    # 2. Position Plausibility: distance within comm range and lateral within road bounds
    dx = msg.x - ego_x
    dy = msg.y - ego_y
    dist = math.hypot(dx, dy)
    if dist <= comm_range_m and abs(msg.y) <= 12.0:
        pos_plausibility = 1.0
    elif dist <= comm_range_m * 1.2:
        pos_plausibility = 0.5
    else:
        pos_plausibility = 0.0

    # 3. Speed Plausibility: physically realistic vehicle speed (0 to 55 m/s ~ 200 km/h)
    if 0.0 <= msg.speed <= 55.0:
        speed_plausibility = 1.0
    elif msg.speed < 0.0 or msg.speed > 70.0:
        speed_plausibility = 0.0
    else:
        speed_plausibility = 0.5

    # 4. Cryptographic Trust Score: HMAC-SHA256 verification
    if trust_store is not None:
        is_valid = trust_store.verify(msg)
        trust_score = 1.0 if is_valid else 0.0
    else:
        trust_score = 1.0 if (msg.signature and not msg.signature.startswith("INVALID")) else 0.0

    # 5. Sensor Cross-Validation Consistency
    sensor_consistency = 1.0
    if sensor_snapshot is not None:
        # If obstacle is reported ahead in lane
        if msg.obstacle and isinstance(msg.obstacle, dict):
            obs_range = float(msg.obstacle.get("range", msg.obstacle.get("distance", 999.0)))
            # If claimed within radar range
            if obs_range <= radar_cover_m:
                # Must be visible in sensor snapshot
                has_in_lane = (
                    sensor_snapshot.in_lane is not None
                    and abs(sensor_snapshot.in_lane.range - obs_range) <= 15.0
                )
                sensor_consistency = 1.0 if has_in_lane else 0.0
        else:
            # If vehicle is claimed within radar coverage
            if dist <= radar_cover_m:
                matched = any(
                    abs(t.range - dist) <= 15.0 and math.hypot(t.dx - dx, t.dy - dy) <= 15.0
                    for t in sensor_snapshot.tracks
                )
                sensor_consistency = 1.0 if matched else 0.0

    return {
        "freshness": float(freshness),
        "position_plausibility": float(pos_plausibility),
        "speed_plausibility": float(speed_plausibility),
        "trust_score": float(trust_score),
        "sensor_consistency": float(sensor_consistency),
    }

