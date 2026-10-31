"""
CAVShield V2X Simulator & Environment.

Implements V2X messaging, TrustStore with HMAC-SHA256 cryptographic signatures,
synthetic road environment (ego + 3 surrounding vehicles + road obstacles),
complementary filter GPS/IMU dead reckoning, multi-sensor fusion (Radar/LiDAR/Camera),
and V2X message broadcast simulation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from src import load_config


# ---------------------------------------------------------------------------
# V2XMessage Dataclass
# ---------------------------------------------------------------------------

@dataclass
class V2XMessage:
    """V2X Message matching CONTRACT item 40.

    Attributes
    ----------
    msg_id : str
        Unique identifier for the message.
    sender_id : str
        Transmitting vehicle or RSU identifier (e.g. 'V-2041').
    x : float
        Longitudinal road position (meters).
    y : float
        Lateral road offset from center lane (meters, positive = left).
    speed : float
        Vehicle speed in m/s.
    msg_type : str
        'BSM', 'COLLISION_WARNING', 'EMERGENCY_BRAKING', or 'ROAD_HAZARD'.
    timestamp : float
        Time of message creation (seconds).
    heading : float
        Vehicle heading angle in radians (0 = along +x axis).
    signature : str
        Hex-encoded HMAC-SHA256 signature.
    seq : int
        Sender sequence number.
    obstacle : Optional[Dict[str, Any]]
        Optional hazard/obstacle details (dict with range, x, y, etc.).
    claimed_decel : float
        Claimed deceleration in m/s² (for EMERGENCY_BRAKING).
    pos_accuracy_m : float
        Position accuracy estimate (meters).
    label : str
        Ground truth label for evaluation ('Normal', 'Spoofing', etc.).
    subtype : str
        Ground truth subtype ('v2x_phantom', 'v2x_fake_ebrake', etc.).
    """
    msg_id: str
    sender_id: str
    x: float
    y: float
    speed: float
    msg_type: str
    timestamp: float
    heading: float = 0.0
    signature: str = ""
    seq: int = 0
    obstacle: Optional[Dict[str, Any]] = None
    claimed_decel: float = 0.0
    pos_accuracy_m: float = 1.0
    label: str = "Normal"
    subtype: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize message to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional[V2XMessage]:
        """Construct V2XMessage from dict with defensive parsing."""
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                msg_id=str(data.get("msg_id", "")),
                sender_id=str(data.get("sender_id", "")),
                x=float(data.get("x", 0.0)),
                y=float(data.get("y", 0.0)),
                speed=float(data.get("speed", 0.0)),
                msg_type=str(data.get("msg_type", "BSM")),
                timestamp=float(data.get("timestamp", 0.0)),
                heading=float(data.get("heading", 0.0)),
                signature=str(data.get("signature", "")),
                seq=int(data.get("seq", 0)),
                obstacle=data.get("obstacle"),
                claimed_decel=float(data.get("claimed_decel", 0.0)),
                pos_accuracy_m=float(data.get("pos_accuracy_m", 1.0)),
                label=str(data.get("label", "Normal")),
                subtype=str(data.get("subtype", "")),
            )
        except Exception:
            return None

    @classmethod
    def from_json(cls, json_str: str) -> Optional[V2XMessage]:
        """Construct V2XMessage from JSON string with defensive parsing (never crashes)."""
        if not isinstance(json_str, (str, bytes)):
            return None
        try:
            parsed = json.loads(json_str)
            return cls.from_dict(parsed)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# TrustStore (HMAC-SHA256 Key Management)
# ---------------------------------------------------------------------------

class TrustStore:
    """Manages cryptographic keys and HMAC-SHA256 signing/verification for V2X.

    Keys are derived from a master secret seed. Enrolls known legitimate vehicles
    and an enrolled-but-compromised vehicle (V-9999).
    """

    ENROLLED_VEHICLES = ["V-EGO", "V-2041", "V-2042", "V-2043", "V-9999"]

    def __init__(self, secret_seed: Optional[str] = None):
        if secret_seed is None:
            cfg = load_config()
            secret_seed = getattr(cfg.v2x, "secret_seed", "cavshield-v2x-hmac-seed")
        self.secret_seed = secret_seed.encode("utf-8")
        self._keys: Dict[str, bytes] = {}

        for vid in self.ENROLLED_VEHICLES:
            self._keys[vid] = hmac.new(self.secret_seed, vid.encode("utf-8"), hashlib.sha256).digest()

    def get_key(self, sender_id: str) -> Optional[bytes]:
        """Return derived HMAC key for enrolled vehicle or None if unknown."""
        return self._keys.get(sender_id)

    @staticmethod
    def canonical_bytes(msg: V2XMessage) -> bytes:
        """Create canonical payload bytes for signature generation."""
        canonical_str = (
            f"{msg.sender_id}|{msg.msg_id}|{msg.msg_type}|"
            f"{msg.timestamp:.3f}|{msg.seq}|{msg.x:.2f}|{msg.y:.2f}|{msg.speed:.2f}"
        )
        return canonical_str.encode("utf-8")

    def sign(self, msg: V2XMessage) -> str:
        """Compute and set HMAC-SHA256 signature for message using sender key."""
        key = self.get_key(msg.sender_id)
        if key is None:
            msg.signature = "INVALID_UNENROLLED"
            return msg.signature

        payload = self.canonical_bytes(msg)
        sig = hmac.new(key, payload, hashlib.sha256).hexdigest()
        msg.signature = sig
        return sig

    def verify(self, msg: V2XMessage) -> bool:
        """Verify HMAC-SHA256 signature of a received message."""
        key = self.get_key(msg.sender_id)
        if key is None or not msg.signature:
            return False

        payload = self.canonical_bytes(msg)
        expected_sig = hmac.new(key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(msg.signature, expected_sig)


# ---------------------------------------------------------------------------
# Sensor Snapshot & Object Track Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ObjectTrack:
    """Track of an obstacle or vehicle detected by ego sensors."""
    obj_id: str
    range: float        # Euclidean distance (meters)
    dx: float           # Relative longitudinal distance ahead (meters)
    dy: float           # Relative lateral offset (meters, positive = left)
    closing: float      # Closing speed (m/s, positive = closing in)
    ttc: float          # Time-to-collision (seconds, 999.0 if not closing)
    sources: List[str]  # e.g. ['radar', 'lidar', 'camera']
    in_lane: bool       # Whether object is within ego's lane ahead


@dataclass
class SensorSnapshot:
    """Aggregated perception state at a single time step."""
    timestamp: float
    tracks: List[ObjectTrack] = field(default_factory=list)
    in_lane: Optional[ObjectTrack] = None
    ego_x_est: float = 1000.0
    ego_y_est: float = 0.0


@dataclass
class Obstacle:
    """Road obstacle in the world."""
    obs_id: str
    x: float
    y: float
    spawn_time: float
    lifetime: float


# ---------------------------------------------------------------------------
# Synthetic Environment
# ---------------------------------------------------------------------------

class Environment:
    """Simulates road environment: ego vehicle, 3 nearby vehicles, road obstacles,
    complementary filter GPS/IMU localization, and fused sensor perception."""

    def __init__(self, cfg: Optional[Any] = None, rng: Optional[random.Random] = None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.rng = rng or random.Random(42)

        sensors = getattr(cfg, "sensors", {})
        self.radar_range_m = float(getattr(sensors, "radar_range_m", 150.0))
        self.lidar_range_m = float(getattr(sensors, "lidar_range_m", 100.0))
        self.camera_range_m = float(getattr(sensors, "camera_range_m", 70.0))
        self.gps_sigma_m = float(getattr(sensors, "gps_sigma_m", 1.2))
        self.lane_half_width_m = float(getattr(sensors, "lane_half_width_m", 1.8))
        self.aeb_ttc_s = float(getattr(sensors, "aeb_ttc_s", 1.8))

        # True world state
        self.ego_x: float = 1000.0
        self.ego_y: float = 0.0
        self.ego_speed: float = 20.0  # ~72 km/h in m/s

        # Complementary filter state (dead reckoning + GPS)
        self.ego_x_est: float = 1000.0
        self.ego_y_est: float = 0.0
        self._filter_alpha: float = 0.95

        # Nearby vehicles: nominal lane offsets and base distances
        # V-2041: Left lane ahead (+3.6 m lat, +55 m long)
        # V-2042: Right lane ahead (-3.6 m lat, +40 m long)
        # V-2043: Behind (0.0 m lat, -50 m long)
        self.vehicles = {
            "V-2041": {"base_dx": 55.0, "base_dy": 3.6, "freq": 0.15, "amp_x": 4.0, "amp_y": 0.2},
            "V-2042": {"base_dx": 40.0, "base_dy": -3.6, "freq": 0.12, "amp_x": 3.5, "amp_y": 0.2},
            "V-2043": {"base_dx": -50.0, "base_dy": 0.0, "freq": 0.10, "amp_x": 3.0, "amp_y": 0.1},
        }

        self.obstacles: List[Obstacle] = []

    def spawn_real_obstacle(self, distance: float = 95.0, lifetime: float = 14.0, now: float = 0.0) -> Obstacle:
        """Spawn a legitimate physical obstacle directly in ego lane ahead."""
        obs = Obstacle(
            obs_id=f"OBS-{int(now * 100)}",
            x=self.ego_x + distance,
            y=0.0,
            spawn_time=now,
            lifetime=lifetime,
        )
        self.obstacles.append(obs)
        return obs

    def step(self, dt: float, now: float, ego_speed_kmh: float) -> None:
        """Step environment physics, update vehicle positions and complementary filter."""
        self.ego_speed = ego_speed_kmh / 3.6
        self.ego_x += self.ego_speed * dt

        # Complementary filter for localization:
        # Dead reckoning prediction
        dr_x = self.ego_x_est + self.ego_speed * dt
        # Noisy GPS reading
        gps_x = self.ego_x + self.rng.gauss(0.0, self.gps_sigma_m)
        self.ego_x_est = self._filter_alpha * dr_x + (1.0 - self._filter_alpha) * gps_x
        self.ego_y_est = 0.0

        # Expire old obstacles
        self.obstacles = [obs for obs in self.obstacles if (now - obs.spawn_time) <= obs.lifetime]

    def get_vehicle_state(self, vid: str, now: float) -> Tuple[float, float, float]:
        """Compute (x, y, speed_ms) of surrounding vehicle with smooth oscillation."""
        spec = self.vehicles[vid]
        osc_x = spec["amp_x"] * math.sin(2.0 * math.pi * spec["freq"] * now)
        osc_y = spec["amp_y"] * math.cos(2.0 * math.pi * spec["freq"] * now)

        x = self.ego_x + spec["base_dx"] + osc_x
        y = spec["base_dy"] + osc_y

        # Speed is ego_speed plus rate of oscillation
        vx = self.ego_speed + spec["amp_x"] * (2.0 * math.pi * spec["freq"]) * math.cos(2.0 * math.pi * spec["freq"] * now)
        return (x, y, vx)

    def sense(self, ego_speed_kmh: float, now: float = 0.0) -> SensorSnapshot:
        """Simulate fused Radar, LiDAR, and Camera perception of environment."""
        tracks: List[ObjectTrack] = []
        v_ego = ego_speed_kmh / 3.6

        # Sense surrounding vehicles
        for vid in self.vehicles:
            vx, vy, v_speed = self.get_vehicle_state(vid, now)
            dx = vx - self.ego_x
            dy = vy - self.ego_y
            dist = math.hypot(dx, dy)

            if dist <= self.radar_range_m:
                sources: List[str] = ["radar"]
                if dist <= self.lidar_range_m:
                    sources.append("lidar")
                if dist <= self.camera_range_m:
                    sources.append("camera")

                closing = v_ego - v_speed
                ttc = (dx / closing) if (dx > 0 and closing > 0.1) else 999.0
                in_lane = (abs(dy) <= self.lane_half_width_m) and (dx > 0)

                tracks.append(ObjectTrack(
                    obj_id=vid,
                    range=round(dist, 2),
                    dx=round(dx, 2),
                    dy=round(dy, 2),
                    closing=round(closing, 2),
                    ttc=round(ttc, 2),
                    sources=sources,
                    in_lane=in_lane,
                ))

        # Sense road obstacles
        for obs in self.obstacles:
            dx = obs.x - self.ego_x
            dy = obs.y - self.ego_y
            dist = math.hypot(dx, dy)

            if dist <= self.radar_range_m:
                sources = ["radar"]
                if dist <= self.lidar_range_m:
                    sources.append("lidar")
                if dist <= self.camera_range_m:
                    sources.append("camera")

                closing = v_ego
                ttc = (dx / closing) if (dx > 0 and closing > 0.1) else 999.0
                in_lane = (abs(dy) <= self.lane_half_width_m) and (dx > 0)

                tracks.append(ObjectTrack(
                    obj_id=obs.obs_id,
                    range=round(dist, 2),
                    dx=round(dx, 2),
                    dy=round(dy, 2),
                    closing=round(closing, 2),
                    ttc=round(ttc, 2),
                    sources=sources,
                    in_lane=in_lane,
                ))

        # Sort tracks by range
        tracks.sort(key=lambda t: t.range)

        # Nearest in-lane track ahead
        in_lane_track = next((t for t in tracks if t.in_lane), None)

        return SensorSnapshot(
            timestamp=now,
            tracks=tracks,
            in_lane=in_lane_track,
            ego_x_est=round(self.ego_x_est, 2),
            ego_y_est=round(self.ego_y_est, 2),
        )


# ---------------------------------------------------------------------------
# V2XSimulator Class
# ---------------------------------------------------------------------------

class V2XSimulator:
    """Generates standard V2X messages from surrounding traffic.

    - 10 Hz signed BSM beacons from each surrounding vehicle.
    - 5 Hz legitimate signed COLLISION_WARNING messages from V-2041 when a real
      obstacle exists ahead.
    """

    def __init__(
        self,
        env: Environment,
        trust_store: Optional[TrustStore] = None,
        cfg: Optional[Any] = None,
    ):
        self.env = env
        self.trust_store = trust_store or TrustStore()
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        v2x_cfg = getattr(cfg, "v2x", {})
        self.beacon_hz = float(getattr(v2x_cfg, "beacon_hz", 10.0))
        self.comm_range_m = float(getattr(v2x_cfg, "comm_range_m", 300.0))

        self._seqs: Dict[str, int] = {vid: 0 for vid in self.env.vehicles}
        self._next_beacon_time: Dict[str, float] = {vid: 0.0 for vid in self.env.vehicles}
        self._next_warning_time: float = 0.0

    def step(self, now: float) -> List[V2XMessage]:
        """Generate V2X messages scheduled at current simulation time."""
        messages: List[V2XMessage] = []
        beacon_period = 1.0 / self.beacon_hz

        # 1. BSM Beacons (10 Hz) from each vehicle
        for vid in self.env.vehicles:
            if now >= self._next_beacon_time.get(vid, 0.0):
                self._next_beacon_time[vid] = now + beacon_period
                vx, vy, v_speed = self.env.get_vehicle_state(vid, now)

                # Broadcast only within communication range
                dist = math.hypot(vx - self.env.ego_x, vy - self.env.ego_y)
                if dist <= self.comm_range_m:
                    seq = self._seqs[vid]
                    self._seqs[vid] += 1

                    msg = V2XMessage(
                        msg_id=f"BSM-{vid}-{seq}",
                        sender_id=vid,
                        x=round(vx, 2),
                        y=round(vy, 2),
                        speed=round(v_speed, 2),
                        msg_type="BSM",
                        timestamp=now,
                        heading=0.0,
                        seq=seq,
                        obstacle=None,
                        claimed_decel=0.0,
                        pos_accuracy_m=1.0,
                        label="Normal",
                        subtype="",
                    )
                    self.trust_store.sign(msg)
                    messages.append(msg)

        # 2. Legitimate COLLISION_WARNING from V-2041 (5 Hz) when a real obstacle exists ahead
        if self.env.obstacles:
            active_obstacle = self.env.obstacles[0]
            v2041_x, v2041_y, _ = self.env.get_vehicle_state("V-2041", now)
            # If obstacle is ahead of V-2041 and within its sensor radar range (150 m)
            obs_dx = active_obstacle.x - v2041_x
            if 0 < obs_dx <= 150.0:
                warning_period = 0.20  # 5 Hz
                if now >= self._next_warning_time:
                    self._next_warning_time = now + warning_period
                    seq = self._seqs["V-2041"]
                    self._seqs["V-2041"] += 1

                    warn_msg = V2XMessage(
                        msg_id=f"WARN-V-2041-{seq}",
                        sender_id="V-2041",
                        x=round(v2041_x, 2),
                        y=round(v2041_y, 2),
                        speed=round(self.env.ego_speed, 2),
                        msg_type="COLLISION_WARNING",
                        timestamp=now,
                        heading=0.0,
                        seq=seq,
                        obstacle={
                            "obs_id": active_obstacle.obs_id,
                            "x": round(active_obstacle.x, 2),
                            "y": round(active_obstacle.y, 2),
                            "distance": round(obs_dx, 2),
                        },
                        claimed_decel=2.5,
                        pos_accuracy_m=1.0,
                        label="Normal",
                        subtype="",
                    )
                    self.trust_store.sign(warn_msg)
                    messages.append(warn_msg)

        return messages

