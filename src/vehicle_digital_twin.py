"""
CAVShield Vehicle Digital Twin.

Implements vehicle physical dynamics, legitimate ADAS/driver controllers,
actuators responding only to accepted CAN command frames, perceived state
tracking (instrument cluster), safety bookkeeping, and dashboard snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

from src import AttrDict, load_config


# ---------------------------------------------------------------------------
# DrivingMode Enum & VehicleState Dataclass
# ---------------------------------------------------------------------------

class DrivingMode(str, Enum):
    """Driving mode states matching CONTRACT item 51."""
    NORMAL = "NORMAL"
    ALERT = "ALERT"
    DEGRADED = "DEGRADED"
    SAFE_MODE = "SAFE MODE"


@dataclass
class VehicleState:
    """Complete physical and internal telemetry state of the vehicle twin.

    Attributes
    ----------
    speed_kmh : float
        Current true vehicle ground speed (km/h).
    rpm : float
        Engine RPM.
    pedal : float
        Driver brake pedal position (0-100%).
    brake_active : int
        Brake light / system active flag (0 or 1).
    decel : float
        Legitimate controller requested deceleration (m/s²).
    applied_decel : float
        Physical deceleration delivered by actuators (m/s²).
    applied_accel : float
        Physical acceleration delivered by actuators (m/s²).
    steering_deg : float
        Steering wheel angle (degrees).
    gear : int
        Current gear (0=P, 1=R, 2=N, 3=D).
    throttle : float
        Engine throttle position (0-100%).
    wheel_speeds : Tuple[float, float, float, float]
        Wheel speeds in km/h: (FL, FR, RL, RR).
    ax : float
        Longitudinal acceleration from IMU (m/s²).
    yaw : float
        Yaw rate from IMU (deg/s).
    aeb : bool
        Autonomous Emergency Braking active flag.
    acc : bool
        Adaptive Cruise Control active flag.
    lka : bool
        Lane Keeping Assist active flag.
    driver_pedal : float
        Driver pedal request (percent).
    target_speed_kmh : float
        Cruise control target speed (km/h).
    driver_or_adas_decel_cmd : float
        Arbitrated deceleration command for CAN 0x200 (m/s²).
    brake_cmd_source : int
        Source for 0x200 (0 DRIVER, 1 AEB, 2 ACC).
    adas_accel_cmd : float
        Acceleration command for CAN 0x201 (m/s²).
    accel_cmd_source : int
        Source for 0x201 (0 DRIVER, 1 AEB, 2 ACC).
    unsafe : str
        Safety flag description (e.g. 'UNINTENDED BRAKING' or '').
    unsafe_seconds : float
        Cumulative duration in unsafe condition (seconds).
    perceived : Dict[str, Any]
        State perceived by the vehicle instrument cluster / ECUs.
    """
    speed_kmh: float = 72.0
    rpm: float = 2766.0
    pedal: float = 0.0
    brake_active: int = 0
    decel: float = 0.0
    applied_decel: float = 0.0
    applied_accel: float = 0.0
    steering_deg: float = 0.0
    gear: int = 3
    throttle: float = 0.0
    wheel_speeds: Tuple[float, float, float, float] = (72.0, 72.0, 72.0, 72.0)
    ax: float = 0.0
    yaw: float = 0.0
    aeb: bool = False
    acc: bool = True
    lka: bool = True
    driver_pedal: float = 0.0
    target_speed_kmh: float = 72.0
    driver_or_adas_decel_cmd: float = 0.0
    brake_cmd_source: int = 0
    adas_accel_cmd: float = 0.0
    accel_cmd_source: int = 2
    unsafe: str = ""
    unsafe_seconds: float = 0.0
    perceived: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# VehicleDigitalTwin Class
# ---------------------------------------------------------------------------

class VehicleDigitalTwin:
    """Vehicle plant and controller digital twin."""

    def __init__(self, cfg: Optional[Any] = None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        # Tunables from simulation section
        sim = getattr(cfg, "simulation", {})
        self.world_hz = float(getattr(sim, "world_hz", 200))
        self.cruise_kmh = float(getattr(sim, "cruise_kmh", 72.0))
        self.degraded_kmh = float(getattr(sim, "degraded_kmh", 50.0))
        self.safe_kmh = float(getattr(sim, "safe_kmh", 30.0))
        self.driver_brake_first_s = float(getattr(sim, "driver_brake_first_s", 6.0))
        self.driver_brake_interval_s = float(getattr(sim, "driver_brake_interval_s", 35.0))
        self.driver_brake_peak_pct = float(getattr(sim, "driver_brake_peak_pct", 22.0))
        self.brake_gain = float(getattr(sim, "brake_gain", 0.1))
        self.command_timeout_s = float(getattr(sim, "command_timeout_s", 0.06))

        # Sensors & physics tunables
        sensors = getattr(cfg, "sensors", {})
        self.aeb_ttc_s = float(getattr(sensors, "aeb_ttc_s", 1.8))

        physics = getattr(cfg, "physics", {})
        self.max_decel = float(getattr(physics, "max_decel", 10.0))
        self.max_accel_cmd = float(getattr(physics, "max_accel_cmd", 3.0))
        self.acc_max_brake = float(getattr(physics, "acc_max_brake", 3.5))
        self.rpm_base = float(getattr(physics, "rpm_base", 750.0))
        self.rpm_per_kmh = float(getattr(physics, "rpm_per_kmh", 28.0))

        v2x = getattr(cfg, "v2x", {})
        self.precrash_decel = float(getattr(v2x, "precrash_decel", 3.0))
        self.precrash_duration_s = float(getattr(v2x, "precrash_duration_s", 2.5))

        # Actuator command buffers: (timestamp, value, source)
        self._accepted_brake_commands: List[Tuple[float, float, int]] = []
        self._last_accel_command: Optional[Tuple[float, float, int]] = None
        self._v2x_hazard_until: float = 0.0

        # State and perceived state
        self.mode: DrivingMode = DrivingMode.NORMAL
        self.unsafe_seconds: float = 0.0
        self.state: VehicleState = VehicleState()
        self.perceived: Dict[str, Any] = {}

        self.reset()

    def reset(self) -> None:
        """Reset vehicle plant, buffers, and perceived state to initial cruising conditions."""
        self.mode = DrivingMode.NORMAL
        self.unsafe_seconds = 0.0
        self._accepted_brake_commands.clear()
        self._v2x_hazard_until = 0.0

        initial_speed = self.cruise_kmh
        initial_rpm = min(7500.0, max(self.rpm_base, self.rpm_base + self.rpm_per_kmh * initial_speed))
        v_ms = initial_speed / 3.6
        initial_drag = 0.05 + 0.0005 * (v_ms ** 2)

        self._last_accel_command = (0.0, initial_drag, 2)

        self.perceived = {
            "speed_kmh": initial_speed,
            "rpm": initial_rpm,
            "pedal": 0.0,
            "steering_deg": 0.0,
            "gear": 3,
        }

        self.state = VehicleState(
            speed_kmh=initial_speed,
            rpm=initial_rpm,
            pedal=0.0,
            brake_active=0,
            decel=0.0,
            applied_decel=0.0,
            applied_accel=initial_drag,
            steering_deg=0.0,
            gear=3,
            throttle=min(100.0, (initial_drag / self.max_accel_cmd) * 100.0),
            wheel_speeds=(initial_speed, initial_speed, initial_speed, initial_speed),
            ax=0.0,
            yaw=0.0,
            aeb=False,
            acc=True,
            lka=True,
            driver_pedal=0.0,
            target_speed_kmh=initial_speed,
            driver_or_adas_decel_cmd=0.0,
            brake_cmd_source=0,
            adas_accel_cmd=initial_drag,
            accel_cmd_source=2,
            unsafe="",
            unsafe_seconds=0.0,
            perceived=self.perceived,
        )

    def set_mode(self, mode: DrivingMode) -> None:
        """Update operational driving mode."""
        self.mode = mode

    def apply_v2x_hazard(self, now: float) -> None:
        """Trigger precrash emergency deceleration following V2X hazard notification."""
        self._v2x_hazard_until = now + self.precrash_duration_s

    def accept_command(self, arb_id: int, signals: Dict[str, Any], now: float) -> None:
        """Process accepted command frames (invoked only if gateway forwards them).

        Parameters
        ----------
        arb_id : int
            CAN arbitration ID (0x200 BRAKE_COMMAND, 0x201 ACCEL_COMMAND, 0x202 ADAS_COMMAND).
        signals : dict
            Decoded signal payload.
        now : float
            Current simulation timestamp (seconds).
        """
        if arb_id == 0x200:
            cmd_decel = float(signals.get("cmd_decel", 0.0))
            source = int(signals.get("source", 0))
            self._accepted_brake_commands.append((now, cmd_decel, source))

        elif arb_id == 0x201:
            cmd_accel = float(signals.get("cmd_accel", 0.0))
            source = int(signals.get("source", 0))
            self._last_accel_command = (now, cmd_accel, source)

        elif arb_id == 0x202:
            if "set_speed_kmh" in signals:
                self.state.target_speed_kmh = float(signals["set_speed_kmh"])
            if "lka" in signals:
                self.state.lka = bool(signals["lka"])
            if "acc" in signals:
                self.state.acc = bool(signals["acc"])
            if "aeb" in signals:
                self.state.aeb = bool(signals["aeb"])

    def accept_status(self, arb_id: int, signals: Dict[str, Any]) -> None:
        """Update perceived state from accepted status messages (instrument cluster belief)."""
        if arb_id == 0x100 and "rpm" in signals:
            self.perceived["rpm"] = float(signals["rpm"])
        elif arb_id == 0x101 and "speed_kmh" in signals:
            self.perceived["speed_kmh"] = float(signals["speed_kmh"])
        elif arb_id == 0x102 and "pedal" in signals:
            self.perceived["pedal"] = float(signals["pedal"])
        elif arb_id == 0x103 and "steering_deg" in signals:
            self.perceived["steering_deg"] = float(signals["steering_deg"])
        elif arb_id == 0x104 and "gear" in signals:
            self.perceived["gear"] = int(signals["gear"])

    def step(self, dt: float, now: float, sensor_snapshot: Optional[Any] = None) -> VehicleState:
        """Step vehicle dynamics, legitimate controllers, and safety bookkeeping.

        Parameters
        ----------
        dt : float
            Simulation time step duration (seconds).
        now : float
            Current simulation time (seconds).
        sensor_snapshot : Optional[Any]
            Sensor snapshot exposing in_lane (range, closing, ttc).

        Returns
        -------
        VehicleState
            Updated vehicle telemetry state.
        """
        # 1. Legitimate Driver Pedal Schedule:
        # First at 6 s then every 35 s with peak 22 percent.
        driver_pedal = 0.0
        if now >= self.driver_brake_first_s:
            elapsed = now - self.driver_brake_first_s
            cycle = elapsed % self.driver_brake_interval_s
            # 1.6 second smooth half-sine pedal press & release
            if cycle <= 1.6:
                driver_pedal = self.driver_brake_peak_pct * math.sin(math.pi * cycle / 1.6)

        driver_decel = driver_pedal * self.brake_gain

        # 2. Legitimate AEB from sensor snapshot:
        # TTC below aeb_ttc, required decel v^2 / (2 * (range - 4)) * 1.25, cap 8.5
        aeb_decel = 0.0
        aeb_active = False
        in_lane_obj = getattr(sensor_snapshot, "in_lane", None) if sensor_snapshot else None
        in_lane_speed_limit = self.cruise_kmh

        if in_lane_obj is not None:
            obj_range = float(getattr(in_lane_obj, "range", 999.0))
            obj_closing = float(getattr(in_lane_obj, "closing", 0.0))
            obj_ttc = float(getattr(in_lane_obj, "ttc", 999.0))

            if obj_ttc < self.aeb_ttc_s and obj_range > 4.0:
                v_ms = self.state.speed_kmh / 3.6
                denom = 2.0 * max(0.1, obj_range - 4.0)
                aeb_req = (v_ms ** 2 / denom) * 1.25
                aeb_decel = min(8.5, max(0.0, aeb_req))
                aeb_active = True
            elif obj_range < 60.0 and obj_closing > 0.5:
                # Follow in-lane object with speed reduction
                lead_speed = max(0.0, self.state.speed_kmh - obj_closing * 3.6)
                in_lane_speed_limit = lead_speed

        # 3. V2X Hazard Precrash Deceleration:
        v2x_decel = 0.0
        if now < self._v2x_hazard_until:
            v2x_decel = self.precrash_decel

        # 4. Target Speed Determination based on DrivingMode and in-lane constraints:
        if self.mode == DrivingMode.SAFE_MODE:
            target_speed = min(self.cruise_kmh, self.safe_kmh, in_lane_speed_limit)
        elif self.mode == DrivingMode.DEGRADED:
            target_speed = min(self.cruise_kmh, self.degraded_kmh, in_lane_speed_limit)
        else:
            target_speed = min(self.cruise_kmh, in_lane_speed_limit)

        # 5. ACC-style Braking:
        acc_decel = 0.0
        if self.state.speed_kmh > target_speed + 1.0 and driver_pedal == 0.0 and not aeb_active:
            speed_err_ms = (self.state.speed_kmh - target_speed) / 3.6
            acc_decel = min(self.acc_max_brake, speed_err_ms * 0.8)

        # 6. Arbitrate Legitimate Deceleration Command:
        legit_decel = max(driver_decel, aeb_decel, acc_decel, v2x_decel)
        if aeb_active and aeb_decel >= max(driver_decel, acc_decel, v2x_decel):
            brake_source = 1  # AEB
        elif v2x_decel > 0.0 and v2x_decel >= max(driver_decel, acc_decel):
            brake_source = 1  # AEB / V2X safety
        elif driver_pedal > 0.0:
            brake_source = 0  # DRIVER
        elif acc_decel > 0.0:
            brake_source = 2  # ACC
        else:
            brake_source = 0

        # 7. Legitimate Accel (Cruise Controller with Drag Feed-Forward):
        if legit_decel > 0.1:
            legit_accel = 0.0
            accel_source = 0
        else:
            v_ms = self.state.speed_kmh / 3.6
            v_tgt_ms = target_speed / 3.6
            a_drag = 0.05 + 0.0005 * (v_ms ** 2)
            speed_err = v_tgt_ms - v_ms
            Kp = 0.6
            legit_accel = max(0.0, min(self.max_accel_cmd, a_drag + Kp * speed_err))
            accel_source = 2  # ACC / cruise

        # Update controller telemetry
        self.state.driver_pedal = driver_pedal
        self.state.pedal = driver_pedal
        self.state.decel = legit_decel
        self.state.driver_or_adas_decel_cmd = legit_decel
        self.state.brake_cmd_source = brake_source
        self.state.adas_accel_cmd = legit_accel
        self.state.accel_cmd_source = accel_source
        self.state.target_speed_kmh = target_speed
        self.state.aeb = aeb_active
        self.state.throttle = min(100.0, (legit_accel / self.max_accel_cmd) * 100.0)

        # 8. Actuators Obey ONLY Commands Passed to accept_command within command_timeout:
        # Brake: max accepted request within command_timeout_s
        valid_brakes = [
            cmd for cmd in self._accepted_brake_commands
            if (now - cmd[0]) <= self.command_timeout_s
        ]
        self._accepted_brake_commands = valid_brakes
        target_actuator_decel = max([cmd[1] for cmd in valid_brakes], default=0.0)

        # Accel: last accepted command within command_timeout_s
        target_actuator_accel = 0.0
        if self._last_accel_command is not None:
            cmd_time, cmd_accel_val, _ = self._last_accel_command
            if (now - cmd_time) <= self.command_timeout_s:
                target_actuator_accel = max(0.0, min(self.max_accel_cmd, cmd_accel_val))

        # 9. First-Order Actuator Lag:
        # Brake lag (tau ~ 0.08 s)
        alpha_b = min(1.0, dt / 0.08)
        self.state.applied_decel += (target_actuator_decel - self.state.applied_decel) * alpha_b
        self.state.applied_decel = max(0.0, min(self.max_decel, self.state.applied_decel))
        self.state.brake_active = 1 if self.state.applied_decel > 0.2 else 0

        # Accel lag (tau ~ 0.10 s)
        alpha_a = min(1.0, dt / 0.10)
        self.state.applied_accel += (target_actuator_accel - self.state.applied_accel) * alpha_a
        self.state.applied_accel = max(0.0, min(self.max_accel_cmd, self.state.applied_accel))

        # 10. Longitudinal Dynamics with Drag:
        v_curr = self.state.speed_kmh / 3.6
        a_drag_curr = 0.05 + 0.0005 * (v_curr ** 2)
        a_net = self.state.applied_accel - self.state.applied_decel - a_drag_curr
        v_next = max(0.0, v_curr + a_net * dt)

        self.state.speed_kmh = v_next * 3.6
        self.state.ax = a_net
        self.state.wheel_speeds = (
            self.state.speed_kmh,
            self.state.speed_kmh,
            self.state.speed_kmh,
            self.state.speed_kmh,
        )
        self.state.rpm = min(7500.0, max(self.rpm_base, self.rpm_base + self.rpm_per_kmh * self.state.speed_kmh))

        # 11. Safety Bookkeeping:
        # Unsafe text when applied decel exceeds legitimate by > 2 m/s² (UNINTENDED BRAKING)
        # or accel exceeds by > 1.5, plus cumulative unsafe_seconds.
        if self.state.applied_decel > (legit_decel + 2.0):
            self.state.unsafe = "UNINTENDED BRAKING"
            self.unsafe_seconds += dt
        elif self.state.applied_accel > (legit_accel + 1.5):
            self.state.unsafe = "UNINTENDED ACCELERATION"
            self.unsafe_seconds += dt
        else:
            self.state.unsafe = ""

        self.state.unsafe_seconds = self.unsafe_seconds
        self.state.perceived = self.perceived

        return self.state

    def snapshot(self) -> Dict[str, Any]:
        """Return JSON-serialisable dict matching CONTRACT item 56."""
        return {
            "speed_kmh": round(self.state.speed_kmh, 2),
            "rpm": round(self.state.rpm, 1),
            "pedal": round(self.state.pedal, 1),
            "brake_active": int(self.state.brake_active),
            "decel": round(self.state.decel, 2),
            "applied_decel": round(self.state.applied_decel, 2),
            "steering_deg": round(self.state.steering_deg, 2),
            "gear": int(self.state.gear),
            "aeb": bool(self.state.aeb),
            "unsafe": self.state.unsafe,
            "unsafe_seconds": round(self.unsafe_seconds, 2),
            "perceived": {
                "speed_kmh": round(self.perceived.get("speed_kmh", 0.0), 2),
                "rpm": round(self.perceived.get("rpm", 0.0), 1),
                "pedal": round(self.perceived.get("pedal", 0.0), 1),
                "steering_deg": round(self.perceived.get("steering_deg", 0.0), 2),
                "gear": int(self.perceived.get("gear", 3)),
            },
        }

