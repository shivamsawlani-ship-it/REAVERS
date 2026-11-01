"""
CAVShield CAN Simulator & Codec.

Implements CAN frame layout, DBC definitions, message encoding/decoding,
AUTOSAR E2E style checksum and alive counter, CAN bus abstraction with
python-can and InMemory fallback, side-channel ground-truth label registry,
and ECU schedule simulator.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import math
import queue
import random
import struct
import threading
from typing import Any, Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# DBC and Message Definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MessageDef:
    """Definition of a CAN message in the DBC."""
    arb_id: int
    name: str
    sender: str
    period_s: float
    safety_critical: bool
    criticality: float


DBC: Dict[int, MessageDef] = {
    0x100: MessageDef(0x100, "ENGINE_RPM", "ECU_ENGINE", 0.020, False, 0.6),
    0x101: MessageDef(0x101, "VEHICLE_SPEED", "ECU_ENGINE", 0.020, True, 0.8),
    0x102: MessageDef(0x102, "BRAKE_STATUS", "ECU_BRAKE", 0.020, True, 0.9),
    0x103: MessageDef(0x103, "STEERING_ANGLE", "ECU_STEERING", 0.020, True, 0.8),
    0x104: MessageDef(0x104, "GEAR_POSITION", "ECU_GEAR", 0.100, True, 0.8),
    0x105: MessageDef(0x105, "WHEEL_SPEED", "SENSOR_WHEEL", 0.020, False, 0.6),
    0x106: MessageDef(0x106, "IMU_DATA", "SENSOR_IMU", 0.010, False, 0.6),
    0x200: MessageDef(0x200, "BRAKE_COMMAND", "ECU_ADAS", 0.020, True, 1.0),
    0x201: MessageDef(0x201, "ACCEL_COMMAND", "ECU_ADAS", 0.020, True, 1.0),
    0x202: MessageDef(0x202, "ADAS_COMMAND", "ECU_ADAS", 0.050, True, 1.0),
}

KNOWN_IDS: Tuple[int, ...] = tuple(sorted(DBC.keys()))
COMMAND_IDS: Tuple[int, ...] = (0x200, 0x201, 0x202)
SAFETY_CRITICAL_IDS: Tuple[int, ...] = tuple(k for k, v in DBC.items() if v.safety_critical)


def criticality(arb_id: int) -> float:
    """Return criticality weight for arbitration ID (unknown IDs get 0.5)."""
    msg_def = DBC.get(arb_id)
    return msg_def.criticality if msg_def is not None else 0.5


def is_safety_critical(arb_id: int) -> bool:
    """Return whether the message ID is marked safety-critical."""
    msg_def = DBC.get(arb_id)
    return msg_def.safety_critical if msg_def is not None else False


# ---------------------------------------------------------------------------
# CANFrame Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CANFrame:
    """CAN frame dataclass matching CONTRACT item 4.

    Attributes
    ----------
    timestamp : float
        Frame timestamp (seconds).
    arb_id : int
        CAN arbitration ID (e.g. 0x101).
    dlc : int
        Data length code (always 8 in our architecture).
    data : bytes
        Raw 8-byte payload.
    sender : str
        ECU or sensor name from DBC.
    msg_type : str
        Message name from DBC.
    label : str
        Ground-truth class for evaluation ('Normal', 'Flooding', etc.).
    subtype : str
        Ground-truth subtype for evaluation ('spoof_brake', etc.).
    """
    timestamp: float
    arb_id: int
    dlc: int
    data: bytes
    sender: str
    msg_type: str
    label: str = "Normal"
    subtype: str = ""


def make_frame(
    timestamp: float,
    arb_id: int,
    data: bytes,
    label: str = "Normal",
    subtype: str = "",
) -> CANFrame:
    """Construct a CANFrame with sender and msg_type populated from DBC."""
    msg_def = DBC.get(arb_id)
    sender = msg_def.sender if msg_def else "UNKNOWN"
    msg_type = msg_def.name if msg_def else "UNKNOWN"
    return CANFrame(
        timestamp=float(timestamp),
        arb_id=int(arb_id),
        dlc=len(data),
        data=bytes(data),
        sender=sender,
        msg_type=msg_type,
        label=label,
        subtype=subtype,
    )


# ---------------------------------------------------------------------------
# Codec: Encode & Decode
# ---------------------------------------------------------------------------

def _clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, val))


def checksum_ok(data: bytes) -> bool:
    """Validate XOR checksum in byte 6 over bytes 0-5."""
    if len(data) != 8:
        return False
    expected = data[0] ^ data[1] ^ data[2] ^ data[3] ^ data[4] ^ data[5]
    return data[6] == expected


def counter_of(data: bytes) -> int:
    """Extract alive counter (low nibble of byte 7)."""
    if len(data) < 8:
        return -1
    return data[7] & 0x0F


def encode(arb_id: int, signals: Dict[str, Any], counter: int = 0) -> bytes:
    """Encode signals dict into 8 bytes with XOR checksum and alive counter.

    Parameters
    ----------
    arb_id : int
        CAN arbitration ID.
    signals : dict
        Signal values matching the DBC layout.
    counter : int
        Alive counter 0-15.

    Returns
    -------
    bytes
        8-byte payload.

    Raises
    ------
    ValueError
        If arb_id is not known in the DBC.
    """
    if arb_id not in DBC:
        raise ValueError(f"Unknown arb_id: 0x{arb_id:X}")

    payload = bytearray(8)

    if arb_id == 0x100:  # ENGINE_RPM: rpm u16 + throttle u8
        rpm = int(_clamp(round(signals.get("rpm", 0.0)), 0, 65535))
        throttle = int(_clamp(round(signals.get("throttle", 0.0)), 0, 255))
        struct.pack_into(">HB", payload, 0, rpm, throttle)

    elif arb_id == 0x101:  # VEHICLE_SPEED: speed u16 = km/h * 100
        speed = int(_clamp(round(signals.get("speed_kmh", 0.0) * 100), 0, 65535))
        struct.pack_into(">H", payload, 0, speed)

    elif arb_id == 0x102:  # BRAKE_STATUS: pedal u8 = pct*2, active u8, decel u16 = m/s2*100
        pedal = int(_clamp(round(signals.get("pedal", 0.0) * 2), 0, 255))
        active = 1 if signals.get("brake_active", 0) else 0
        decel = int(_clamp(round(signals.get("decel", 0.0) * 100), 0, 65535))
        struct.pack_into(">BBH", payload, 0, pedal, active, decel)

    elif arb_id == 0x103:  # STEERING_ANGLE: steering s16 = deg * 10
        deg = int(_clamp(round(signals.get("steering_deg", 0.0) * 10), -32768, 32767))
        struct.pack_into(">h", payload, 0, deg)

    elif arb_id == 0x104:  # GEAR_POSITION: gear u8 (0=P, 1=R, 2=N, 3=D)
        gear = int(_clamp(signals.get("gear", 3), 0, 255))
        struct.pack_into(">B", payload, 0, gear)

    elif arb_id == 0x105:  # WHEEL_SPEED: 4x u8 in 0.75 km/h units
        fl = int(_clamp(round(signals.get("wheel_fl", 0.0) / 0.75), 0, 255))
        fr = int(_clamp(round(signals.get("wheel_fr", 0.0) / 0.75), 0, 255))
        rl = int(_clamp(round(signals.get("wheel_rl", 0.0) / 0.75), 0, 255))
        rr = int(_clamp(round(signals.get("wheel_rr", 0.0) / 0.75), 0, 255))
        struct.pack_into(">4B", payload, 0, fl, fr, rl, rr)

    elif arb_id == 0x106:  # IMU_DATA: ax s16 and yaw s16 = value * 100
        ax = int(_clamp(round(signals.get("ax", 0.0) * 100), -32768, 32767))
        yaw = int(_clamp(round(signals.get("yaw", 0.0) * 100), -32768, 32767))
        struct.pack_into(">hh", payload, 0, ax, yaw)

    elif arb_id == 0x200:  # BRAKE_COMMAND: cmd_decel u16 = val*100 + source u8
        decel = int(_clamp(round(signals.get("cmd_decel", 0.0) * 100), 0, 65535))
        source = int(_clamp(signals.get("source", 0), 0, 255))
        struct.pack_into(">HB", payload, 0, decel, source)

    elif arb_id == 0x201:  # ACCEL_COMMAND: cmd_accel s16 = val*100 + source u8
        accel = int(_clamp(round(signals.get("cmd_accel", 0.0) * 100), -32768, 32767))
        source = int(_clamp(signals.get("source", 0), 0, 255))
        struct.pack_into(">hB", payload, 0, accel, source)

    elif arb_id == 0x202:  # ADAS_COMMAND: flags lka, acc, aeb, set_speed_kmh
        lka = 1 if signals.get("lka", 1) else 0
        acc = 1 if signals.get("acc", 1) else 0
        aeb = 1 if signals.get("aeb", 1) else 0
        set_spd = int(_clamp(round(signals.get("set_speed_kmh", 72.0)), 0, 255))
        struct.pack_into(">4B", payload, 0, lka, acc, aeb, set_spd)

    # Byte 6: XOR checksum of bytes 0-5
    payload[6] = payload[0] ^ payload[1] ^ payload[2] ^ payload[3] ^ payload[4] ^ payload[5]
    # Byte 7: low nibble = alive counter (0-15)
    payload[7] = int(counter) & 0x0F

    return bytes(payload)


def decode(arb_id: int, data: bytes) -> Dict[str, Any]:
    """Decode 8-byte payload into a signals dict.

    Parameters
    ----------
    arb_id : int
        CAN arbitration ID.
    data : bytes
        8-byte CAN data payload.

    Returns
    -------
    dict
        Decoded signal dictionary.

    Raises
    ------
    ValueError
        If arb_id is not in DBC or len(data) != 8.
    """
    if arb_id not in DBC:
        raise ValueError(f"Unknown arb_id: 0x{arb_id:X}")
    if len(data) != 8:
        raise ValueError(f"Expected 8 bytes for CAN frame, got {len(data)}")

    if arb_id == 0x100:
        rpm, throttle = struct.unpack(">HB", data[0:3])
        return {"rpm": float(rpm), "throttle": float(throttle)}

    if arb_id == 0x101:
        (speed_raw,) = struct.unpack(">H", data[0:2])
        return {"speed_kmh": round(speed_raw / 100.0, 2)}

    if arb_id == 0x102:
        pedal_raw, brake_active, decel_raw = struct.unpack(">BBH", data[0:4])
        return {
            "pedal": round(pedal_raw / 2.0, 1),
            "brake_active": int(brake_active),
            "decel": round(decel_raw / 100.0, 2),
        }

    if arb_id == 0x103:
        (steering_raw,) = struct.unpack(">h", data[0:2])
        return {"steering_deg": round(steering_raw / 10.0, 1)}

    if arb_id == 0x104:
        (gear,) = struct.unpack(">B", data[0:1])
        return {"gear": int(gear)}

    if arb_id == 0x105:
        fl, fr, rl, rr = struct.unpack(">4B", data[0:4])
        return {
            "wheel_fl": round(fl * 0.75, 2),
            "wheel_fr": round(fr * 0.75, 2),
            "wheel_rl": round(rl * 0.75, 2),
            "wheel_rr": round(rr * 0.75, 2),
        }

    if arb_id == 0x106:
        ax_raw, yaw_raw = struct.unpack(">hh", data[0:4])
        return {"ax": round(ax_raw / 100.0, 2), "yaw": round(yaw_raw / 100.0, 2)}

    if arb_id == 0x200:
        decel_raw, source = struct.unpack(">HB", data[0:3])
        return {"cmd_decel": round(decel_raw / 100.0, 2), "source": int(source)}

    if arb_id == 0x201:
        accel_raw, source = struct.unpack(">hB", data[0:3])
        return {"cmd_accel": round(accel_raw / 100.0, 2), "source": int(source)}

    if arb_id == 0x202:
        lka, acc, aeb, set_speed = struct.unpack(">4B", data[0:4])
        return {
            "lka": int(lka),
            "acc": int(acc),
            "aeb": int(aeb),
            "set_speed_kmh": float(set_speed),
        }

    raise ValueError(f"Unhandled DBC id: 0x{arb_id:X}")


# ---------------------------------------------------------------------------
# LabelRegistry (Thread-Safe Side Channel)
# ---------------------------------------------------------------------------

class LabelRegistry:
    """Thread-safe side channel preserving ground-truth labels across python-can bus."""

    def __init__(self, maxlen_per_key: int = 1000):
        self._lock = threading.Lock()
        self._registry: Dict[Tuple[int, bytes], Deque[Tuple[str, str]]] = {}
        self._timestamps: Dict[Tuple[int, bytes], Deque[float]] = {}
        self._maxlen = maxlen_per_key

    def register(
        self,
        arb_id: int,
        payload: bytes,
        label: str = "Normal",
        subtype: str = "",
        timestamp: Optional[float] = None,
    ) -> None:
        """Register ground-truth metadata for a frame payload."""
        key = (int(arb_id), bytes(payload))
        with self._lock:
            if key not in self._registry:
                self._registry[key] = collections.deque(maxlen=self._maxlen)
            self._registry[key].append((label, subtype))

            if timestamp is not None:
                if key not in self._timestamps:
                    self._timestamps[key] = collections.deque(maxlen=self._maxlen)
                self._timestamps[key].append(float(timestamp))

    def pop(self, arb_id: int, payload: bytes) -> Tuple[str, str]:
        """Retrieve and pop the next ground-truth label for a frame payload."""
        key = (int(arb_id), bytes(payload))
        with self._lock:
            q = self._registry.get(key)
            if q and len(q) > 0:
                item = q.popleft()
                if not q:
                    del self._registry[key]
                return item
        return ("Normal", "")

    def pop_timestamp(self, arb_id: int, payload: bytes) -> Optional[float]:
        """Retrieve and pop the original simulation timestamp for a frame payload."""
        key = (int(arb_id), bytes(payload))
        with self._lock:
            q = self._timestamps.get(key)
            if q and len(q) > 0:
                ts = q.popleft()
                if not q:
                    del self._timestamps[key]
                return ts
        return None

    def clear(self) -> None:
        """Clear all registered labels."""
        with self._lock:
            self._registry.clear()
            self._timestamps.clear()


# Default global registry
label_registry = LabelRegistry()


# ---------------------------------------------------------------------------
# Bus Layer (python-can + InMemory fallback)
# ---------------------------------------------------------------------------

class InMemoryBus:
    """Thread-safe in-memory queue fallback with python-can compatible API."""

    def __init__(self, channel: str = "cavshield"):
        self.channel = channel
        self._queue: queue.Queue = queue.Queue()
        self._open = True

    def send(self, msg: Any) -> None:
        if not self._open:
            raise RuntimeError("InMemoryBus is shut down")
        self._queue.put(msg)

    def recv(self, timeout: Optional[float] = None) -> Any:
        if not self._open:
            return None
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def shutdown(self) -> None:
        self._open = False


def create_bus(cfg: Any = None) -> Any:
    """Open a python-can virtual bus, falling back to InMemoryBus on error."""
    interface = "virtual"
    channel = "cavshield"
    if cfg is not None:
        if hasattr(cfg, "can"):
            interface = getattr(cfg.can, "interface", "virtual")
            channel = getattr(cfg.can, "channel", "cavshield")
        elif isinstance(cfg, dict) and "can" in cfg:
            interface = cfg["can"].get("interface", "virtual")
            channel = cfg["can"].get("channel", "cavshield")

    try:
        import can
        return can.Bus(
            interface=interface,
            channel=channel,
            receive_own_messages=False,
            preserve_timestamps=True,
        )
    except Exception:
        return InMemoryBus(channel=str(channel))


def frame_to_message(frame: CANFrame, reg: Optional[LabelRegistry] = label_registry) -> Any:
    """Convert CANFrame to a python-can Message (or InMemory message)."""
    if reg is not None:
        reg.register(frame.arb_id, frame.data, frame.label, frame.subtype, timestamp=frame.timestamp)

    try:
        import can
        return can.Message(
            arbitration_id=frame.arb_id,
            data=frame.data,
            is_extended_id=False,
            timestamp=frame.timestamp,
            dlc=frame.dlc,
        )
    except Exception:
        class _FallbackMessage:
            def __init__(self, arb_id: int, data: bytes, ts: float, dlc: int):
                self.arbitration_id = arb_id
                self.data = data
                self.timestamp = ts
                self.dlc = dlc
                self.is_extended_id = False
        return _FallbackMessage(frame.arb_id, frame.data, frame.timestamp, frame.dlc)


def message_to_frame(
    msg: Any,
    reg: Optional[LabelRegistry] = label_registry,
) -> Optional[CANFrame]:
    """Convert python-can Message to CANFrame. Never raises on malformed input."""
    try:
        if msg is None:
            return None
        arb_id = getattr(msg, "arbitration_id", None)
        data = getattr(msg, "data", None)
        timestamp = getattr(msg, "timestamp", 0.0) or 0.0
        dlc = getattr(msg, "dlc", None)

        if arb_id is None or data is None:
            return None

        data_bytes = bytes(data)
        if dlc is None:
            dlc = len(data_bytes)

        label, subtype = ("Normal", "")
        frame_ts = float(timestamp)
        if reg is not None:
            label, subtype = reg.pop(int(arb_id), data_bytes)
            orig_ts = reg.pop_timestamp(int(arb_id), data_bytes)
            if orig_ts is not None:
                frame_ts = float(orig_ts)

        return make_frame(
            timestamp=frame_ts,
            arb_id=int(arb_id),
            data=data_bytes,
            label=label,
            subtype=subtype,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ECUSimulator
# ---------------------------------------------------------------------------

class ECUSimulator:
    """Schedules each CAN ID at its period with ~1% gaussian jitter and random phase.

    Maintains per-ID alive counters and resyncs if falling behind.
    """

    def __init__(self, rng: Optional[random.Random] = None, cfg: Any = None):
        self.rng = rng or random.Random(42)
        self.cfg = cfg
        self._counters: Dict[int, int] = {arb_id: 0 for arb_id in KNOWN_IDS}
        self._next_send_time: Dict[int, float] = {}
        self.reset()

        self.world_stalls: int = 0

    def reset(self) -> None:
        """Reset alive counters and initialize random phase schedules."""
        self.world_stalls = 0
        self._counters = {arb_id: 0 for arb_id in KNOWN_IDS}
        self._next_send_time.clear()
        for arb_id, mdef in DBC.items():
            # Random initial phase within [0, period_s)
            self._next_send_time[arb_id] = self.rng.uniform(0.0, mdef.period_s)

    def step(self, now: float, vehicle_state: Any) -> List[CANFrame]:
        """Step the ECU schedule up to `now` and return generated CANFrames.

        Parameters
        ----------
        now : float
            Current simulation time (seconds).
        vehicle_state : Any
            VehicleState instance or dict providing current vehicle telemetry.

        Returns
        -------
        list of CANFrame
            Frames scheduled for transmission at or before `now`.
        """
        frames: List[CANFrame] = []

        for arb_id in KNOWN_IDS:
            mdef = DBC[arb_id]
            t_send = self._next_send_time.get(arb_id, 0.0)

            if now >= t_send:
                period = mdef.period_s
                # About 1% Gaussian jitter
                jitter = self.rng.gauss(0.0, 0.01 * period)

                # Resync instead of bursting if the loop falls behind
                if (now - t_send) > 2.0 * period:
                    self.world_stalls += 1
                    t_frame = now
                    self._next_send_time[arb_id] = now + period + jitter
                else:
                    t_frame = t_send
                    self._next_send_time[arb_id] = t_send + period + jitter

                # Build signals from vehicle_state
                signals = self._extract_signals(arb_id, vehicle_state)
                counter = self._counters[arb_id]
                self._counters[arb_id] = (counter + 1) % 16

                payload = encode(arb_id, signals, counter)
                frames.append(make_frame(t_frame, arb_id, payload, label="Normal", subtype=""))

        return frames

    def _extract_signals(self, arb_id: int, state: Any) -> Dict[str, Any]:
        """Map vehicle state attributes to DBC signal dict for arb_id."""
        # Helper to get attr or dict item
        def _get(name: str, default: Any = 0.0) -> Any:
            if hasattr(state, name):
                return getattr(state, name)
            if isinstance(state, dict):
                return state.get(name, default)
            return default

        if arb_id == 0x100:  # ENGINE_RPM
            return {
                "rpm": _get("rpm", 750.0),
                "throttle": _get("throttle", 0.0),
            }

        if arb_id == 0x101:  # VEHICLE_SPEED
            return {
                "speed_kmh": _get("speed_kmh", 0.0),
            }

        if arb_id == 0x102:  # BRAKE_STATUS
            return {
                "pedal": _get("pedal", 0.0),
                "brake_active": _get("brake_active", 0),
                "decel": _get("applied_decel", _get("decel", 0.0)),
            }

        if arb_id == 0x103:  # STEERING_ANGLE
            return {
                "steering_deg": _get("steering_deg", 0.0),
            }

        if arb_id == 0x104:  # GEAR_POSITION
            return {
                "gear": _get("gear", 3),
            }

        if arb_id == 0x105:  # WHEEL_SPEED
            wheels = _get("wheel_speeds", (0.0, 0.0, 0.0, 0.0))
            if isinstance(wheels, (list, tuple)) and len(wheels) == 4:
                return {
                    "wheel_fl": wheels[0],
                    "wheel_fr": wheels[1],
                    "wheel_rl": wheels[2],
                    "wheel_rr": wheels[3],
                }
            spd = _get("speed_kmh", 0.0)
            return {"wheel_fl": spd, "wheel_fr": spd, "wheel_rl": spd, "wheel_rr": spd}

        if arb_id == 0x106:  # IMU_DATA
            return {
                "ax": _get("ax", 0.0),
                "yaw": _get("yaw", 0.0),
            }

        if arb_id == 0x200:  # BRAKE_COMMAND
            return {
                "cmd_decel": _get("driver_or_adas_decel_cmd", 0.0),
                "source": _get("brake_cmd_source", 0),
            }

        if arb_id == 0x201:  # ACCEL_COMMAND
            return {
                "cmd_accel": _get("adas_accel_cmd", 0.0),
                "source": _get("accel_cmd_source", 2),
            }

        if arb_id == 0x202:  # ADAS_COMMAND
            return {
                "lka": int(_get("lka", 1)),
                "acc": int(_get("acc", 1)),
                "aeb": int(_get("aeb", 1)),
                "set_speed_kmh": _get("target_speed_kmh", 72.0),
            }

        return {}

