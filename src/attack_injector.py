"""
CAVShield Attack Injector.

Implements all CAN and V2X attack classes per CONTRACT:
- CAN Attacks: Flooding (0x000, 0x0A0), Fuzzing, Replay (brake buffer),
  Spoofing (spoof_brake, spoof_accel, spoof_speed, spoof_rpm, spoof_gear).
- V2X Attacks: v2x_phantom (compromised V-9999 with valid sig),
  v2x_fake_ebrake (impersonating V-2041 with invalid sig),
  v2x_gps (V-9999 with 150m erratic position jumps).
- AttackInjector orchestrator with lifecycle management, replay buffer,
  event logging for latency, UDP command server, and CLI interface.
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass, field
import json
import random
import socket
import sys
import threading
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from src import load_config
from src.can_simulator import (
    CANFrame,
    DBC,
    encode,
    label_registry,
    make_frame,
)
from src.v2x_simulator import TrustStore, V2XMessage


# ---------------------------------------------------------------------------
# Threat Model & Entry Points Metadata
# ---------------------------------------------------------------------------

DEFAULT_ENTRY_POINTS: Dict[str, str] = {
    "spoof_brake": "Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular)",
    "spoof_accel": "Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular)",
    "spoof_speed": "Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular)",
    "spoof_rpm": "Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular)",
    "spoof_gear": "Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular)",
    "flood": "Malicious OBD-II device",
    "fuzz": "Malicious OBD-II device",
    "replay": "Recorded traffic replayed via a compromised node",
    "v2x_phantom": "Malicious V2X transmitter / compromised on-board V2X unit",
    "v2x_fake_ebrake": "Malicious V2X transmitter / compromised on-board V2X unit",
    "v2x_gps": "Spoofed position broadcast",
}


def resolve_entry_point(attack_identifier: Optional[str], cfg: Optional[Any] = None) -> str:
    """Resolve assumed entry point description for an attack name or attributed attack type.

    Simulation metadata assumption: CAVShield does not claim to detect radio or physical entry points;
    this attributes the simulated injection vector according to the vehicle threat model.
    """
    if not attack_identifier or attack_identifier in ("Normal", "None", ""):
        return ""

    table = dict(DEFAULT_ENTRY_POINTS)
    if cfg is not None:
        attacks_cfg = getattr(cfg, "attacks", {})
        entry_pts_cfg = getattr(attacks_cfg, "entry_points", {})
        if isinstance(entry_pts_cfg, dict):
            table.update(entry_pts_cfg)

    # Direct match
    if attack_identifier in table:
        return table[attack_identifier]

    ident_lower = str(attack_identifier).lower().strip()
    if ident_lower in table:
        return table[ident_lower]

    # Pattern match on name or attributed label
    if "flood" in ident_lower:
        return table.get("flood", DEFAULT_ENTRY_POINTS["flood"])
    if "fuzz" in ident_lower:
        return table.get("fuzz", DEFAULT_ENTRY_POINTS["fuzz"])
    if "replay" in ident_lower:
        return table.get("replay", DEFAULT_ENTRY_POINTS["replay"])
    if "phantom" in ident_lower:
        return table.get("v2x_phantom", DEFAULT_ENTRY_POINTS["v2x_phantom"])
    if "fake_ebrake" in ident_lower or "fake e-brake" in ident_lower:
        return table.get("v2x_fake_ebrake", DEFAULT_ENTRY_POINTS["v2x_fake_ebrake"])
    if "gps" in ident_lower:
        return table.get("v2x_gps", DEFAULT_ENTRY_POINTS["v2x_gps"])
    if "brake" in ident_lower and ("spoof" in ident_lower or "0x200" in ident_lower):
        return table.get("spoof_brake", DEFAULT_ENTRY_POINTS["spoof_brake"])
    if "accel" in ident_lower and ("spoof" in ident_lower or "0x201" in ident_lower):
        return table.get("spoof_accel", DEFAULT_ENTRY_POINTS["spoof_accel"])
    if "speed" in ident_lower and ("spoof" in ident_lower or "0x101" in ident_lower):
        return table.get("spoof_speed", DEFAULT_ENTRY_POINTS["spoof_speed"])
    if "rpm" in ident_lower and ("spoof" in ident_lower or "0x100" in ident_lower):
        return table.get("spoof_rpm", DEFAULT_ENTRY_POINTS["spoof_rpm"])
    if "gear" in ident_lower and ("spoof" in ident_lower or "0x104" in ident_lower):
        return table.get("spoof_gear", DEFAULT_ENTRY_POINTS["spoof_gear"])
    if ident_lower.startswith("spoof") or "spoof" in ident_lower:
        return table.get("spoof_brake", "Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular)")
    if "v2x" in ident_lower:
        return table.get("v2x_phantom", "Malicious V2X transmitter / compromised on-board V2X unit")

    return "Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular)"


# ---------------------------------------------------------------------------
# Base Attack Class
# ---------------------------------------------------------------------------

class Attack:
    """Base class for all cyber-physical attacks."""

    def __init__(
        self,
        name: str,
        rate_hz: float = 100.0,
        rng: Optional[random.Random] = None,
        entry_point: Optional[str] = None,
    ):
        self.name = name
        self.rate_hz = rate_hz
        self.rng = rng or random.Random(42)
        self.entry_point = entry_point or DEFAULT_ENTRY_POINTS.get(
            name, "Compromised infotainment (Bluetooth/Wi-Fi) or telematics (cellular)"
        )
        self.active: bool = False
        self.start_time: float = 0.0
        self.duration: Optional[float] = None
        self._next_tick_time: float = 0.0
        self._seq: int = 0

    def start(self, now: float, duration: Optional[float] = None, **params: Any) -> None:
        """Activate the attack."""
        self.active = True
        self.start_time = now
        self.duration = duration
        self._next_tick_time = now
        self._seq = 0

    def stop(self) -> None:
        """Deactivate the attack."""
        self.active = False

    def is_active(self, now: float) -> bool:
        """Check whether the attack is currently active."""
        if not self.active:
            return False
        if self.duration is not None and (now - self.start_time) >= self.duration:
            self.active = False
            return False
        return True

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        """Generate attack payloads for the current simulation step."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CAN Attacks
# ---------------------------------------------------------------------------

class CANFloodAttack(Attack):
    """Bus flooding attack injecting high-priority IDs 0x000 and 0x0A0 at 700 Hz."""

    def __init__(self, rate_hz: float = 700.0, rng: Optional[random.Random] = None):
        super().__init__("flood", rate_hz, rng)
        self.ids = [0x000, 0x0A0]
        self._id_idx = 0

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        if not self.is_active(now):
            return [], []

        frames: List[CANFrame] = []
        period = 1.0 / self.rate_hz

        while self._next_tick_time <= now:
            self._next_tick_time += period
            arb_id = self.ids[self._id_idx % len(self.ids)]
            self._id_idx += 1

            payload = b"\x00\x00\x00\x00\x00\x00\x00\x00"
            frame = make_frame(
                timestamp=now,
                arb_id=arb_id,
                data=payload,
                label="Flooding",
                subtype="flood_bus",
            )
            label_registry.register(arb_id, payload, frame.label, frame.subtype, timestamp=now)
            frames.append(frame)

        return frames, []


class CANFuzzAttack(Attack):
    """Fuzzing attack injecting random IDs and random payloads at 300 Hz."""

    def __init__(self, rate_hz: float = 300.0, rng: Optional[random.Random] = None):
        super().__init__("fuzz", rate_hz, rng)
        self.candidate_ids = [0x050, 0x101, 0x102, 0x150, 0x200, 0x250, 0x333, 0x777]

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        if not self.is_active(now):
            return [], []

        frames: List[CANFrame] = []
        period = 1.0 / self.rate_hz

        while self._next_tick_time <= now:
            self._next_tick_time += period
            arb_id = self.rng.choice(self.candidate_ids)
            payload = bytes(self.rng.getrandbits(8) for _ in range(8))

            frame = make_frame(
                timestamp=now,
                arb_id=arb_id,
                data=payload,
                label="Fuzzing",
                subtype="fuzz_random",
            )
            label_registry.register(arb_id, payload, frame.label, frame.subtype, timestamp=now)
            frames.append(frame)

        return frames, []


class CANReplayAttack(Attack):
    """Replays previously observed legitimate brake frames (0x102, 0x200) with original timing."""

    def __init__(self, replay_buffer: Deque[Tuple[float, int, bytes]], rng: Optional[random.Random] = None):
        super().__init__("replay", 100.0, rng)
        self.replay_buffer = replay_buffer
        self._playback_queue: List[Tuple[float, int, bytes]] = []
        self._playback_idx: int = 0

    def start(self, now: float, duration: Optional[float] = None, **params: Any) -> None:
        super().start(now, duration, **params)
        self._playback_queue.clear()
        self._playback_idx = 0

        # Snapshot recorded brake frames from replay buffer
        records = list(self.replay_buffer)
        if not records:
            return

        # Prefer a segment containing non-zero 0x200 brake command
        nonzero_idx = -1
        for i, (_, arb_id, data) in enumerate(records):
            if arb_id == 0x200 and len(data) >= 2:
                decel = (data[0] << 8) | data[1]
                if decel > 0:
                    nonzero_idx = i
                    break

        if nonzero_idx >= 0:
            start_slice = max(0, nonzero_idx - 25)
            end_slice = min(len(records), nonzero_idx + 75)
            segment = records[start_slice:end_slice]
        else:
            segment = records[-100:] if len(records) >= 100 else records

        if segment:
            t0 = segment[0][0]
            self._playback_queue = [(rec_t - t0, rec_id, rec_data) for rec_t, rec_id, rec_data in segment]

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        if not self.is_active(now) or not self._playback_queue:
            return [], []

        frames: List[CANFrame] = []
        rel_time = now - self.start_time

        while self._playback_idx < len(self._playback_queue):
            rel_t, arb_id, data = self._playback_queue[self._playback_idx]
            if rel_time >= rel_t:
                self._playback_idx += 1
                frame = make_frame(
                    timestamp=now,
                    arb_id=arb_id,
                    data=data,
                    label="Replay",
                    subtype="replay_brake",
                )
                label_registry.register(arb_id, data, frame.label, frame.subtype, timestamp=now)
                frames.append(frame)
            else:
                break

        # Loop playback if still active
        if self._playback_idx >= len(self._playback_queue):
            self.start_time = now
            self._playback_idx = 0

        return frames, []


class CANSpoofBrakeAttack(Attack):
    """Spoofs 0x200 BRAKE_COMMAND at 7.0 m/s2, DRIVER source, valid checksum, RANDOM counter."""

    def __init__(self, rate_hz: float = 100.0, decel: float = 7.0, rng: Optional[random.Random] = None):
        super().__init__("spoof_brake", rate_hz, rng)
        self.decel = decel

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        if not self.is_active(now):
            return [], []

        frames: List[CANFrame] = []
        period = 1.0 / self.rate_hz

        while self._next_tick_time <= now:
            self._next_tick_time += period
            # Random counter violates sequential continuity while checksum remains valid
            rand_counter = self.rng.randint(0, 15)
            payload = encode(
                0x200,
                {"cmd_decel": self.decel, "source": 0},  # 0 = DRIVER
                counter=rand_counter,
            )
            frame = make_frame(
                timestamp=now,
                arb_id=0x200,
                data=payload,
                label="Spoofing",
                subtype="spoof_brake",
            )
            label_registry.register(0x200, payload, frame.label, frame.subtype, timestamp=now)
            frames.append(frame)

        return frames, []


class CANSpoofAccelAttack(Attack):
    """Spoofs 0x201 ACCEL_COMMAND at +4.5 m/s2 at 100 Hz."""

    def __init__(self, rate_hz: float = 100.0, accel: float = 4.5, rng: Optional[random.Random] = None):
        super().__init__("spoof_accel", rate_hz, rng)
        self.accel = accel

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        if not self.is_active(now):
            return [], []

        frames: List[CANFrame] = []
        period = 1.0 / self.rate_hz

        while self._next_tick_time <= now:
            self._next_tick_time += period
            rand_counter = self.rng.randint(0, 15)
            payload = encode(
                0x201,
                {"cmd_accel": self.accel, "source": 0},
                counter=rand_counter,
            )
            frame = make_frame(
                timestamp=now,
                arb_id=0x201,
                data=payload,
                label="Spoofing",
                subtype="spoof_accel",
            )
            label_registry.register(0x201, payload, frame.label, frame.subtype, timestamp=now)
            frames.append(frame)

        return frames, []


class CANSpoofSpeedAttack(Attack):
    """Spoofs 0x101 VEHICLE_SPEED to 125 km/h at 100 Hz."""

    def __init__(self, rate_hz: float = 100.0, speed_kmh: float = 125.0, rng: Optional[random.Random] = None):
        super().__init__("spoof_speed", rate_hz, rng)
        self.speed_kmh = speed_kmh

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        if not self.is_active(now):
            return [], []

        frames: List[CANFrame] = []
        period = 1.0 / self.rate_hz

        while self._next_tick_time <= now:
            self._next_tick_time += period
            rand_counter = self.rng.randint(0, 15)
            payload = encode(
                0x101,
                {"speed_kmh": self.speed_kmh},
                counter=rand_counter,
            )
            frame = make_frame(
                timestamp=now,
                arb_id=0x101,
                data=payload,
                label="Spoofing",
                subtype="spoof_speed",
            )
            label_registry.register(0x101, payload, frame.label, frame.subtype, timestamp=now)
            frames.append(frame)

        return frames, []


class CANSpoofRpmAttack(Attack):
    """Spoofs 0x100 ENGINE_RPM to 7000 RPM at 100 Hz."""

    def __init__(self, rate_hz: float = 100.0, rpm: float = 7000.0, rng: Optional[random.Random] = None):
        super().__init__("spoof_rpm", rate_hz, rng)
        self.rpm = rpm

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        if not self.is_active(now):
            return [], []

        frames: List[CANFrame] = []
        period = 1.0 / self.rate_hz

        while self._next_tick_time <= now:
            self._next_tick_time += period
            rand_counter = self.rng.randint(0, 15)
            payload = encode(
                0x100,
                {"rpm": self.rpm, "throttle": 100.0},
                counter=rand_counter,
            )
            frame = make_frame(
                timestamp=now,
                arb_id=0x100,
                data=payload,
                label="Spoofing",
                subtype="spoof_rpm",
            )
            label_registry.register(0x100, payload, frame.label, frame.subtype, timestamp=now)
            frames.append(frame)

        return frames, []


class CANSpoofGearAttack(Attack):
    """Spoofs 0x104 GEAR_POSITION to Reverse (gear = 1) at 50 Hz."""

    def __init__(self, rate_hz: float = 50.0, gear: int = 1, rng: Optional[random.Random] = None):
        super().__init__("spoof_gear", rate_hz, rng)
        self.gear = gear

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        if not self.is_active(now):
            return [], []

        frames: List[CANFrame] = []
        period = 1.0 / self.rate_hz

        while self._next_tick_time <= now:
            self._next_tick_time += period
            rand_counter = self.rng.randint(0, 15)
            payload = encode(
                0x104,
                {"gear": self.gear},
                counter=rand_counter,
            )
            frame = make_frame(
                timestamp=now,
                arb_id=0x104,
                data=payload,
                label="Spoofing",
                subtype="spoof_gear",
            )
            label_registry.register(0x104, payload, frame.label, frame.subtype, timestamp=now)
            frames.append(frame)

        return frames, []


# ---------------------------------------------------------------------------
# V2X Attacks
# ---------------------------------------------------------------------------

class V2XPhantomAttack(Attack):
    """Compromised vehicle V-9999 sends signed false COLLISION_WARNING 40 m ahead in lane at 10 Hz."""

    def __init__(self, trust_store: TrustStore, rate_hz: float = 10.0, rng: Optional[random.Random] = None):
        super().__init__("v2x_phantom", rate_hz, rng)
        self.trust_store = trust_store

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        if not self.is_active(now):
            return [], []

        messages: List[V2XMessage] = []
        period = 1.0 / self.rate_hz

        ego_x = getattr(ctx, "ego_x", 1000.0) if ctx else 1000.0

        while self._next_tick_time <= now:
            self._next_tick_time += period
            self._seq += 1

            msg = V2XMessage(
                msg_id=f"PHANTOM-{self._seq}",
                sender_id="V-9999",
                x=round(ego_x + 30.0, 2),
                y=0.0,
                speed=20.0,
                msg_type="COLLISION_WARNING",
                timestamp=now,
                heading=0.0,
                seq=self._seq,
                obstacle={"obs_id": "PHANTOM-OBS", "range": 40.0, "x": round(ego_x + 40.0, 2), "y": 0.0},
                claimed_decel=4.0,
                pos_accuracy_m=1.0,
                label="Spoofing",
                subtype="v2x_phantom",
            )
            # Valid cryptographic signature from enrolled compromised key V-9999
            self.trust_store.sign(msg)
            messages.append(msg)

        return [], messages


class V2XFakeEbrakeAttack(Attack):
    """Impersonates V-2041 with an INVALID signature, claiming emergency braking at 9 m/s2 at 10 Hz."""

    def __init__(self, rate_hz: float = 10.0, rng: Optional[random.Random] = None):
        super().__init__("v2x_fake_ebrake", rate_hz, rng)

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        if not self.is_active(now):
            return [], []

        messages: List[V2XMessage] = []
        period = 1.0 / self.rate_hz

        ego_x = getattr(ctx, "ego_x", 1000.0) if ctx else 1000.0

        while self._next_tick_time <= now:
            self._next_tick_time += period
            self._seq += 1

            msg = V2XMessage(
                msg_id=f"FAKE-EBRAKE-{self._seq}",
                sender_id="V-2041",
                x=round(ego_x + 50.0, 2),
                y=3.6,
                speed=18.0,
                msg_type="EMERGENCY_BRAKING",
                timestamp=now,
                heading=0.0,
                seq=self._seq,
                obstacle=None,
                claimed_decel=9.0,
                pos_accuracy_m=1.0,
                label="Spoofing",
                subtype="v2x_fake_ebrake",
            )
            # Deliberately invalid corrupted signature
            msg.signature = f"DEADBEEF_INVALID_SIG_{self._seq}"
            messages.append(msg)

        return [], messages


class V2XGpsAttack(Attack):
    """Enrolled V-9999 sends validly signed messages but with erratic position jumps up to 150 m."""

    def __init__(self, trust_store: TrustStore, rate_hz: float = 10.0, rng: Optional[random.Random] = None):
        super().__init__("v2x_gps", rate_hz, rng)
        self.trust_store = trust_store
        self._flip = False

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        if not self.is_active(now):
            return [], []

        messages: List[V2XMessage] = []
        period = 1.0 / self.rate_hz

        ego_x = getattr(ctx, "ego_x", 1000.0) if ctx else 1000.0

        while self._next_tick_time <= now:
            self._next_tick_time += period
            self._seq += 1
            self._flip = not self._flip

            # Jump 150m ahead or behind
            jump_offset = 150.0 if self._flip else -20.0
            x_pos = ego_x + jump_offset

            msg = V2XMessage(
                msg_id=f"GPS-JUMP-{self._seq}",
                sender_id="V-9999",
                x=round(x_pos, 2),
                y=0.0,
                speed=20.0,
                msg_type="BSM",
                timestamp=now,
                heading=0.0,
                seq=self._seq,
                obstacle=None,
                claimed_decel=0.0,
                pos_accuracy_m=1.0,
                label="Spoofing",
                subtype="v2x_gps",
            )
            # Valid signature for the bogus position
            self.trust_store.sign(msg)
            messages.append(msg)

        return [], messages


# ---------------------------------------------------------------------------
# AttackInjector Orchestrator
# ---------------------------------------------------------------------------

class AttackInjector:
    """Central orchestrator for cyber-physical attacks on CAN and V2X.

    Provides start/stop lifecycle, replay ring buffer observation, event logging
    with first-frame timestamps for latency measurement, and an embedded UDP JSON
    command server.
    """

    def __init__(
        self,
        cfg: Optional[Any] = None,
        trust_store: Optional[TrustStore] = None,
        rng: Optional[random.Random] = None,
    ):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.rng = rng or random.Random(42)
        self.trust_store = trust_store or TrustStore()

        # Replay observation buffer (stores up to 30s of legitimate brake frames)
        self.replay_buffer: Deque[Tuple[float, int, bytes]] = collections.deque(maxlen=3000)

        # Event log: [{'attack': name, 'start_time': t, 'first_frame_time': t_frame}]
        self.event_log: List[Dict[str, Any]] = []

        # Registry of attack instances
        self.attacks: Dict[str, Attack] = {
            "flood": CANFloodAttack(rate_hz=700.0, rng=self.rng),
            "fuzz": CANFuzzAttack(rate_hz=300.0, rng=self.rng),
            "replay": CANReplayAttack(replay_buffer=self.replay_buffer, rng=self.rng),
            "spoof_brake": CANSpoofBrakeAttack(rate_hz=100.0, decel=7.0, rng=self.rng),
            "spoof_accel": CANSpoofAccelAttack(rate_hz=100.0, accel=4.5, rng=self.rng),
            "spoof_speed": CANSpoofSpeedAttack(rate_hz=100.0, speed_kmh=125.0, rng=self.rng),
            "spoof_rpm": CANSpoofRpmAttack(rate_hz=100.0, rpm=7000.0, rng=self.rng),
            "spoof_gear": CANSpoofGearAttack(rate_hz=50.0, gear=1, rng=self.rng),
            "v2x_phantom": V2XPhantomAttack(trust_store=self.trust_store, rate_hz=10.0, rng=self.rng),
            "v2x_fake_ebrake": V2XFakeEbrakeAttack(rate_hz=10.0, rng=self.rng),
            "v2x_gps": V2XGpsAttack(trust_store=self.trust_store, rate_hz=10.0, rng=self.rng),
        }

        # Apply configuration entry point overrides if provided
        attacks_cfg = getattr(cfg, "attacks", {})
        entry_pts_cfg = getattr(attacks_cfg, "entry_points", {})
        if isinstance(entry_pts_cfg, dict):
            for name, ep in entry_pts_cfg.items():
                if name in self.attacks:
                    self.attacks[name].entry_point = str(ep)

        # UDP Command Server
        can_cfg = getattr(cfg, "can", {})
        self.udp_port: int = int(getattr(can_cfg, "command_udp_port", 47474))
        self._server_sock: Optional[socket.socket] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running: bool = False

    def observe(self, frames: List[CANFrame]) -> None:
        """Observe legitimate frames to populate the replay buffer (focusing on brake IDs)."""
        for f in frames:
            if f.arb_id in (0x102, 0x200):
                self.replay_buffer.append((f.timestamp, f.arb_id, f.data))

    def start(self, name: str, duration: Optional[float] = None, now: float = 0.0, **params: Any) -> bool:
        """Start an attack by name."""
        attack = self.attacks.get(name)
        if attack is None:
            return False

        attack.start(now=now, duration=duration, **params)
        self.event_log.append({
            "attack": name,
            "start_time": now,
            "first_frame_time": None,
        })
        return True

    def stop(self, name: str) -> bool:
        """Stop an attack by name."""
        attack = self.attacks.get(name)
        if attack is None:
            return False
        attack.stop()
        return True

    def stop_all(self) -> None:
        """Stop all active attacks."""
        for attack in self.attacks.values():
            attack.stop()

    def active(self) -> List[str]:
        """Return list of names of currently active attacks."""
        return [name for name, atk in self.attacks.items() if atk.active]

    def get_entry_point(self, name: str) -> str:
        """Return assumed entry point metadata for an attack."""
        atk = self.attacks.get(name)
        if atk is not None:
            return atk.entry_point
        return resolve_entry_point(name, self.cfg)

    def tick(self, now: float, ctx: Optional[Any] = None) -> Tuple[List[CANFrame], List[V2XMessage]]:
        """Step all active attacks and collect injected CAN frames and V2X messages."""
        all_can: List[CANFrame] = []
        all_v2x: List[V2XMessage] = []

        for name, atk in self.attacks.items():
            if atk.is_active(now):
                can_frames, v2x_msgs = atk.tick(now, ctx)

                if can_frames or v2x_msgs:
                    # Update first_frame_time in event log if not set
                    for entry in reversed(self.event_log):
                        if entry["attack"] == name and entry["first_frame_time"] is None:
                            entry["first_frame_time"] = now
                            break

                all_can.extend(can_frames)
                all_v2x.extend(v2x_msgs)

        return all_can, all_v2x

    # -----------------------------------------------------------------------
    # UDP Server & CLI
    # -----------------------------------------------------------------------

    def start_udp_server(self, host: str = "127.0.0.1") -> None:
        """Start UDP command server in background daemon thread."""
        if self._running:
            return
        try:
            self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._server_sock.bind((host, self.udp_port))
            self._server_sock.settimeout(0.5)
            self._running = True
            self._server_thread = threading.Thread(target=self._udp_listen_loop, daemon=True)
            self._server_thread.start()
        except Exception:
            # Fallback if port already in use or bind restricted
            self._running = False

    def stop_udp_server(self) -> None:
        """Stop background UDP command server."""
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None

    def _udp_listen_loop(self) -> None:
        while self._running:
            try:
                if self._server_sock is None:
                    break
                data, addr = self._server_sock.recvfrom(4096)
                if not data:
                    continue
                cmd = json.loads(data.decode("utf-8"))
                response = self._handle_command(cmd)
                self._server_sock.sendto(json.dumps(response).encode("utf-8"), addr)
            except (socket.timeout, BlockingIOError):
                continue
            except Exception:
                continue

    def _handle_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch JSON UDP commands."""
        action = cmd.get("cmd") or cmd.get("action")
        if action == "start":
            atk_name = str(cmd.get("attack", ""))
            dur = cmd.get("duration")
            dur_float = float(dur) if dur is not None else None
            ok = self.start(atk_name, duration=dur_float)
            return {"status": "ok" if ok else "error", "active": self.active()}

        if action == "stop":
            atk_name = str(cmd.get("attack", ""))
            ok = self.stop(atk_name)
            return {"status": "ok" if ok else "error", "active": self.active()}

        if action == "stop_all":
            self.stop_all()
            return {"status": "ok", "active": self.active()}

        if action == "list":
            return {"status": "ok", "available": list(self.attacks.keys()), "active": self.active()}

        return {"status": "unknown_command"}


# ---------------------------------------------------------------------------
# CLI Command Helper
# ---------------------------------------------------------------------------

def send_udp_command(cmd: Dict[str, Any], host: str = "127.0.0.1", port: int = 47474) -> Optional[Dict[str, Any]]:
    """Send command to running AttackInjector UDP server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    try:
        sock.sendto(json.dumps(cmd).encode("utf-8"), (host, port))
        data, _ = sock.recvfrom(4096)
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    finally:
        sock.close()


def main() -> None:
    """CLI interface for controlling attack injector."""
    parser = argparse.ArgumentParser(description="CAVShield Attack Injector CLI")
    parser.add_argument("--list", action="store_true", help="List available and active attacks")
    parser.add_argument("--attack", type=str, help="Name of attack to launch")
    parser.add_argument("--duration", type=float, default=None, help="Duration of attack in seconds")
    parser.add_argument("--stop", type=str, help="Stop specified attack")
    parser.add_argument("--stop-all", action="store_true", help="Stop all currently active attacks")
    parser.add_argument("--port", type=int, default=47474, help="UDP command port")

    args = parser.parse_args()

    if args.list:
        res = send_udp_command({"cmd": "list"}, port=args.port)
        print("Attack Injector Status:")
        print(json.dumps(res, indent=2))
        return

    if args.stop_all:
        res = send_udp_command({"cmd": "stop_all"}, port=args.port)
        print("Stop All Result:")
        print(json.dumps(res, indent=2))
        return

    if args.stop:
        res = send_udp_command({"cmd": "stop", "attack": args.stop}, port=args.port)
        print(f"Stop {args.stop} Result:")
        print(json.dumps(res, indent=2))
        return

    if args.attack:
        res = send_udp_command({"cmd": "start", "attack": args.attack, "duration": args.duration}, port=args.port)
        print(f"Start {args.attack} Result:")
        print(json.dumps(res, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()

