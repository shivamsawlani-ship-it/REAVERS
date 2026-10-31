"""
CAVShield Cyber-Physical Invariant & Physics Engine.

Validates CAN frame signals against vehicle physical dynamics, sensor fusion
cross-checks, and actuator constraints.
Details strictly cite specific physical numbers (pedal, measured decel, wheel speeds,
IMU acceleration, radar TTC, etc.) for explainability.
"""

from __future__ import annotations

import collections
import math
from typing import Any, Dict, List, Optional, Tuple

from src import load_config
from src.can_simulator import CANFrame
from src.detection.rule_engine import Evidence
from src.v2x_simulator import SensorSnapshot


class PhysicsEngine:
    """Verifies cyber-physical invariants and cross-sensor kinematic plausibility."""

    def __init__(self, cfg: Optional[Any] = None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        sim_cfg = getattr(cfg, "simulation", {})
        self.brake_gain = float(getattr(sim_cfg, "brake_gain", 0.1))

        phys_cfg = getattr(cfg, "physics", {})
        self.max_decel = float(getattr(phys_cfg, "max_decel", 10.0))
        self.max_accel_cmd = float(getattr(phys_cfg, "max_accel_cmd", 3.0))
        self.pedal_tolerance_pct = float(getattr(phys_cfg, "pedal_tolerance_pct", 1.2))
        self.speed_wheel_tolerance_kmh = float(getattr(phys_cfg, "speed_wheel_tolerance_kmh", 4.0))
        self.rpm_base = float(getattr(phys_cfg, "rpm_base", 750.0))
        self.rpm_per_kmh = float(getattr(phys_cfg, "rpm_per_kmh", 28.0))
        self.rpm_tolerance = float(getattr(phys_cfg, "rpm_tolerance", 1100.0))
        self.gear_change_max_kmh = float(getattr(phys_cfg, "gear_change_max_kmh", 5.0))
        self.steering_rate_dps = float(getattr(phys_cfg, "steering_rate_dps", 450.0))
        self.acc_max_brake = float(getattr(phys_cfg, "acc_max_brake", 3.5))

        sensor_cfg = getattr(cfg, "sensors", {})
        self.aeb_ttc_s = float(getattr(sensor_cfg, "aeb_ttc_s", 1.8))

        # Sensor snapshot independent of CAN bus
        self.sensor_snapshot: Optional[SensorSnapshot] = None

        # Validated vehicle plant state (updated ONLY on update() for forwarded frames)
        self.validated_speed: float = 72.0
        self.last_speed_ts: float = 0.0

        self.validated_pedal: float = 0.0
        self.validated_decel: float = 0.0
        self.validated_brake_active: int = 0

        self.validated_rpm: float = 2766.0
        self.validated_gear: int = 3

        self.validated_steering: float = 0.0
        self.last_steering_ts: float = 0.0

        self.validated_ax: float = 0.0
        self.validated_yaw: float = 0.0

        self.validated_wheel_avg: float = 72.0
        self.wheel_speed_trend: float = 0.0
        self.last_wheel_ts: float = 0.0
        self.has_wheel_speed: bool = True

        self.validated_aeb_flag: bool = False
        self.validated_acc_flag: bool = True
        self.validated_lka_flag: bool = True

        # Re-anchor & Consistency tracking (CONTRACT / FIX 1)
        self.last_update_ts: float = 0.0
        self.consistent_frame_count: Dict[int, int] = collections.defaultdict(int)
        self.just_reanchored: Dict[int, bool] = collections.defaultdict(bool)

    def reset(self) -> None:
        """Reset internal physical state to baseline cruise conditions."""
        self.validated_speed = 72.0
        self.validated_rpm = 2766.0
        self.validated_pedal = 0.0
        self.validated_brake_active = 0
        self.validated_decel = 0.0
        self.validated_steering = 0.0
        self.validated_gear = 3
        self.validated_wheel_avg = 72.0
        self.validated_ax = 0.0
        self.validated_yaw = 0.0
        self.validated_aeb_flag = False
        self.validated_acc_flag = True
        self.validated_lka_flag = True
        self.has_wheel_speed = True
        self.wheel_speed_trend = 0.0
        self.last_speed_ts = 0.0
        self.last_steering_ts = 0.0
        self.last_wheel_ts = 0.0
        self.sensor_snapshot = None
        self.last_update_ts = 0.0
        self.consistent_frame_count.clear()
        self.just_reanchored.clear()

    def set_sensor_snapshot(self, snapshot: Optional[SensorSnapshot]) -> None:
        """Receive independent out-of-band perception snapshot from sensors."""
        self.sensor_snapshot = snapshot

    def reanchor_from_sensors(self, now: float, reason: str = "") -> None:
        """Re-anchor validated physical state from independent sensors (FIX 1)."""
        sensor_speed = None
        if self.sensor_snapshot is not None:
            ws = getattr(self.sensor_snapshot, "wheel_speeds", None)
            if ws is not None and hasattr(ws, "avg"):
                sensor_speed = float(ws.avg)
            elif hasattr(self.sensor_snapshot, "ego_speed_kmh"):
                sensor_speed = float(self.sensor_snapshot.ego_speed_kmh)

        if sensor_speed is None and self.has_wheel_speed and self.validated_wheel_avg > 0.0:
            sensor_speed = self.validated_wheel_avg

        if sensor_speed is not None:
            self.validated_speed = sensor_speed
            self.validated_rpm = min(7500.0, max(self.rpm_base, self.rpm_base + self.rpm_per_kmh * self.validated_speed))
            self.last_speed_ts = now
            self.just_reanchored[0x101] = True
            self.just_reanchored[0x100] = True

        if self.sensor_snapshot is not None:
            imu = getattr(self.sensor_snapshot, "imu", None)
            if imu is not None:
                self.validated_ax = float(getattr(imu, "ax", self.validated_ax))
                self.validated_yaw = float(getattr(imu, "yaw", self.validated_yaw))

        self.last_update_ts = now
        self.just_reanchored[0x103] = True

    def force_resync(self, arb_id: Optional[int] = None) -> None:
        """Force physics re-anchor from independent sensors (lockout watchdog)."""
        now = self.last_update_ts if self.last_update_ts > 0 else 0.0
        self.reanchor_from_sensors(now, reason="force_resync")
        if arb_id is not None:
            self.consistent_frame_count[arb_id] = 0
            self.just_reanchored[arb_id] = True

    def check(self, frame: CANFrame, decoded: Dict[str, Any]) -> List[Evidence]:
        """Check frame signals against physics invariants and cross-sensor state.

        Does not mutate validated plant state.

        Parameters
        ----------
        frame : CANFrame
            Incoming frame.
        decoded : dict
            Decoded signals for this frame.

        Returns
        -------
        List[Evidence]
            List of physics violation evidence.
        """
        evidence_list: List[Evidence] = []
        arb_id = frame.arb_id
        now = frame.timestamp

        # Re-anchor condition (CONTRACT / FIX 1):
        # Re-anchored from independent sensors when frame stream is internally consistent
        # for 5 frames or the last update is older than 0.5 s.
        age = (now - self.last_update_ts) if self.last_update_ts > 0 else 0.0
        if age > 0.5 or self.consistent_frame_count[arb_id] >= 5:
            self.reanchor_from_sensors(now, reason=f"age={age:.2f}s, consistent_count={self.consistent_frame_count[arb_id]}")
            self.consistent_frame_count[arb_id] = 0

        # -------------------------------------------------------------------
        # 1. 0x200 BRAKE_COMMAND
        # -------------------------------------------------------------------
        if arb_id == 0x200:
            cmd_decel = float(decoded.get("cmd_decel", 0.0))
            source = int(decoded.get("source", 0))

            if cmd_decel > 0.5:
                # Source 0: DRIVER
                if source == 0:
                    expected_decel = self.validated_pedal * self.brake_gain
                    # If command exceeds pedal expectation by more than tolerance
                    decel_tol = max(0.5, self.pedal_tolerance_pct * self.brake_gain * 3.0)
                    if abs(cmd_decel - expected_decel) > decel_tol:
                        evidence_list.append(Evidence(
                            source="physics",
                            name="UNJUSTIFIED_DRIVER_BRAKE",
                            severity=0.90,
                            detail=(
                                f"Unjustified driver brake cmd={cmd_decel:.2f} m/s²: driver pedal is "
                                f"{self.validated_pedal:.1f}% (expected ~{expected_decel:.2f} m/s²), "
                                f"measured decel={self.validated_decel:.2f} m/s², wheel trend="
                                f"{self.wheel_speed_trend:+.2f} km/h, IMU ax={self.validated_ax:+.2f} m/s²"
                            ),
                            hint="Spoofing",
                        ))

                # Source 1: AEB
                elif source == 1:
                    radar_ttc = 999.0
                    has_radar_obstacle = False
                    if self.sensor_snapshot and self.sensor_snapshot.in_lane:
                        radar_ttc = self.sensor_snapshot.in_lane.ttc
                        has_radar_obstacle = True

                    ttc_limit = self.aeb_ttc_s * 1.2
                    if not self.validated_aeb_flag or radar_ttc > ttc_limit or not has_radar_obstacle:
                        evidence_list.append(Evidence(
                            source="physics",
                            name="UNJUSTIFIED_AEB_BRAKE",
                            severity=0.90,
                            detail=(
                                f"Unjustified AEB brake cmd={cmd_decel:.2f} m/s²: radar TTC is {radar_ttc:.2f}s "
                                f"(threshold {ttc_limit:.2f}s), AEB enabled={self.validated_aeb_flag}, "
                                f"pedal={self.validated_pedal:.1f}%, IMU ax={self.validated_ax:+.2f} m/s²"
                            ),
                            hint="Spoofing",
                        ))

                # Source 2: ACC
                elif source == 2:
                    if not self.validated_acc_flag or cmd_decel > (self.acc_max_brake + 0.1):
                        evidence_list.append(Evidence(
                            source="physics",
                            name="UNJUSTIFIED_ACC_BRAKE",
                            severity=0.85,
                            detail=(
                                f"Unjustified ACC brake cmd={cmd_decel:.2f} m/s²: ACC enabled="
                                f"{self.validated_acc_flag}, limit={self.acc_max_brake:.2f} m/s², "
                                f"pedal={self.validated_pedal:.1f}%, IMU ax={self.validated_ax:+.2f} m/s²"
                            ),
                            hint="Spoofing",
                        ))

        # -------------------------------------------------------------------
        # 2. 0x201 ACCEL_COMMAND
        # -------------------------------------------------------------------
        elif arb_id == 0x201:
            cmd_accel = float(decoded.get("cmd_accel", 0.0))

            if abs(cmd_accel) > (self.max_accel_cmd + 0.05):
                evidence_list.append(Evidence(
                    source="physics",
                    name="ACCEL_LIMIT_EXCEEDED",
                    severity=0.85,
                    detail=(
                        f"Accel command {cmd_accel:.2f} m/s² exceeds physical limit "
                        f"{self.max_accel_cmd:.2f} m/s²"
                    ),
                    hint="Spoofing",
                ))

            if cmd_accel > 0.2 and (self.validated_pedal > 5.0 or self.validated_aeb_flag):
                evidence_list.append(Evidence(
                    source="physics",
                    name="CONFLICTING_ACCEL",
                    severity=0.90,
                    detail=(
                        f"Accel command {cmd_accel:.2f} m/s² commanded while brake pedal is "
                        f"{self.validated_pedal:.1f}% or AEB active={self.validated_aeb_flag}"
                    ),
                    hint="Spoofing",
                ))

        # -------------------------------------------------------------------
        # 3. 0x101 VEHICLE_SPEED
        # -------------------------------------------------------------------
        elif arb_id == 0x101:
            speed_kmh = float(decoded.get("speed_kmh", 0.0))

            if speed_kmh < 0.0 or speed_kmh > 250.0:
                evidence_list.append(Evidence(
                    source="physics",
                    name="SPEED_OUT_OF_RANGE",
                    severity=0.85,
                    detail=f"Vehicle speed {speed_kmh:.1f} km/h outside physical limits [0, 250]",
                    hint="Spoofing",
                ))

            # Wheel-speed cross check
            if self.has_wheel_speed and abs(speed_kmh - self.validated_wheel_avg) > self.speed_wheel_tolerance_kmh:
                diff = abs(speed_kmh - self.validated_wheel_avg)
                evidence_list.append(Evidence(
                    source="physics",
                    name="SPEED_WHEEL_MISMATCH",
                    severity=0.85,
                    detail=(
                        f"Reported speed {speed_kmh:.1f} km/h differs from average wheel speed "
                        f"{self.validated_wheel_avg:.1f} km/h by {diff:.1f} km/h "
                        f"(tolerance {self.speed_wheel_tolerance_kmh:.1f} km/h)"
                    ),
                    hint="Spoofing",
                ))

            # Acceleration rate check: use elapsed time capped at 0.2 s, skip on first frame after re-anchor
            if self.last_speed_ts > 0.0:
                if self.just_reanchored.get(0x101, False):
                    self.just_reanchored[0x101] = False
                else:
                    dt_spd = min(0.2, max(0.001, now - self.last_speed_ts))
                    rate_ms2 = abs(speed_kmh - self.validated_speed) / 3.6 / dt_spd
                    if rate_ms2 > (self.max_decel * 1.5):
                        evidence_list.append(Evidence(
                            source="physics",
                            name="SPEED_RATE_IMPLAUSIBLE",
                            severity=0.85,
                            detail=(
                                f"Speed rate jump {rate_ms2:.1f} m/s² exceeds max vehicle decel/accel "
                                f"{self.max_decel:.1f} m/s² over {dt_spd*1000.0:.1f}ms"
                            ),
                            hint="Spoofing",
                        ))

        # -------------------------------------------------------------------
        # 4. 0x100 ENGINE_RPM
        # -------------------------------------------------------------------
        elif arb_id == 0x100:
            rpm = float(decoded.get("rpm", 0.0))
            if rpm < 0.0 or rpm > 7500.0:
                evidence_list.append(Evidence(
                    source="physics",
                    name="RPM_OUT_OF_RANGE",
                    severity=0.80,
                    detail=f"Engine RPM {rpm:.0f} exceeds physical limit [0, 7500]",
                    hint="Spoofing",
                ))

            expected_rpm = self.rpm_base + self.rpm_per_kmh * self.validated_speed
            diff_rpm = abs(rpm - expected_rpm)
            if diff_rpm > self.rpm_tolerance:
                evidence_list.append(Evidence(
                    source="physics",
                    name="RPM_INCONSISTENCY",
                    severity=0.80,
                    detail=(
                        f"Reported engine RPM {rpm:.0f} outside expected band "
                        f"[{expected_rpm - self.rpm_tolerance:.0f}, {expected_rpm + self.rpm_tolerance:.0f}] "
                        f"(diff {diff_rpm:.0f} RPM > tol {self.rpm_tolerance:.0f} RPM) "
                        f"for validated speed {self.validated_speed:.1f} km/h"
                    ),
                    hint="Spoofing",
                ))

        # -------------------------------------------------------------------
        # 5. 0x104 GEAR_POSITION
        # -------------------------------------------------------------------
        elif arb_id == 0x104:
            gear = int(decoded.get("gear", 3))
            if gear not in (0, 1, 2, 3):
                evidence_list.append(Evidence(
                    source="physics",
                    name="INVALID_GEAR",
                    severity=0.85,
                    detail=f"Invalid gear position index {gear} (expected 0=P, 1=R, 2=N, 3=D)",
                    hint="Fuzzing",
                ))

            if gear in (0, 1) and self.validated_speed > self.gear_change_max_kmh:
                gear_name = "P" if gear == 0 else "R"
                evidence_list.append(Evidence(
                    source="physics",
                    name="IMPLAUSIBLE_GEAR",
                    severity=0.90,
                    detail=(
                        f"Gear {gear_name} engaged while vehicle speed is {self.validated_speed:.1f} km/h "
                        f"(exceeds interlock limit {self.gear_change_max_kmh:.1f} km/h)"
                    ),
                    hint="Spoofing",
                ))

        # -------------------------------------------------------------------
        # 6. 0x103 STEERING_ANGLE
        # -------------------------------------------------------------------
        elif arb_id == 0x103:
            steering_deg = float(decoded.get("steering_deg", 0.0))
            if abs(steering_deg) > 540.0:
                evidence_list.append(Evidence(
                    source="physics",
                    name="STEERING_RANGE_EXCEEDED",
                    severity=0.80,
                    detail=f"Steering angle {steering_deg:.1f}° exceeds max lock-to-lock ±540°",
                    hint="Spoofing",
                ))

            if self.last_steering_ts > 0.0:
                if self.just_reanchored.get(0x103, False):
                    self.just_reanchored[0x103] = False
                else:
                    dt_str = min(0.2, max(0.001, now - self.last_steering_ts))
                    rate_dps = abs(steering_deg - self.validated_steering) / dt_str
                    if rate_dps > (self.steering_rate_dps + 50.0):
                        evidence_list.append(Evidence(
                            source="physics",
                            name="STEERING_RATE_EXCEEDED",
                            severity=0.80,
                            detail=(
                                f"Steering rate {rate_dps:.1f}°/s exceeds physical limit "
                                f"{self.steering_rate_dps:.1f}°/s"
                            ),
                            hint="Spoofing",
                        ))

        # -------------------------------------------------------------------
        # 7. 0x202 ADAS_COMMAND
        # -------------------------------------------------------------------
        elif arb_id == 0x202:
            aeb_cmd = int(decoded.get("aeb", 0))
            if aeb_cmd == 1:
                has_obstacle = (
                    self.sensor_snapshot is not None
                    and self.sensor_snapshot.in_lane is not None
                    and self.sensor_snapshot.in_lane.range <= 150.0
                )
                if not has_obstacle:
                    evidence_list.append(Evidence(
                        source="physics",
                        name="UNJUSTIFIED_AEB_FLAG",
                        severity=0.85,
                        detail="AEB activation flag asserted without radar-confirmed obstacle in ego lane",
                        hint="Spoofing",
                    ))

        # -------------------------------------------------------------------
        # 8. 0x105, 0x106, 0x102 Range Checks
        # -------------------------------------------------------------------
        elif arb_id == 0x105:
            for w_name in ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]:
                val = float(decoded.get(w_name, 0.0))
                if val < 0.0 or val > 250.0:
                    evidence_list.append(Evidence(
                        source="physics",
                        name="WHEEL_SPEED_OUT_OF_RANGE",
                        severity=0.75,
                        detail=f"{w_name} reading {val:.1f} km/h outside [0, 250]",
                        hint="Spoofing",
                    ))

        elif arb_id == 0x106:
            ax = float(decoded.get("ax", 0.0))
            yaw = float(decoded.get("yaw", 0.0))
            if ax < -15.0 or ax > 10.0:
                evidence_list.append(Evidence(
                    source="physics",
                    name="IMU_AX_OUT_OF_RANGE",
                    severity=0.75,
                    detail=f"IMU ax {ax:.2f} m/s² outside physical bounds [-15, +10]",
                    hint="Spoofing",
                ))
            if abs(yaw) > 100.0:
                evidence_list.append(Evidence(
                    source="physics",
                    name="IMU_YAW_OUT_OF_RANGE",
                    severity=0.75,
                    detail=f"IMU yaw rate {yaw:.1f}°/s outside physical bounds [-100, +100]",
                    hint="Spoofing",
                ))

        elif arb_id == 0x102:
            pedal = float(decoded.get("pedal", 0.0))
            decel = float(decoded.get("decel", 0.0))
            if pedal < 0.0 or pedal > 100.0:
                evidence_list.append(Evidence(
                    source="physics",
                    name="PEDAL_OUT_OF_RANGE",
                    severity=0.75,
                    detail=f"Brake pedal reading {pedal:.1f}% outside [0, 100]",
                    hint="Spoofing",
                ))
            if decel < 0.0 or decel > 15.0:
                evidence_list.append(Evidence(
                    source="physics",
                    name="DECEL_OUT_OF_RANGE",
                    severity=0.75,
                    detail=f"Deceleration reading {decel:.2f} m/s² outside [0, 15]",
                    hint="Spoofing",
                ))

        return evidence_list

    def update(self, frame: CANFrame, decoded: Dict[str, Any]) -> None:
        """Update validated physical state. Call ONLY after forwarding frame."""
        arb_id = frame.arb_id
        now = frame.timestamp
        self.last_update_ts = now
        self.consistent_frame_count[arb_id] += 1

        if arb_id == 0x101:
            self.validated_speed = float(decoded.get("speed_kmh", self.validated_speed))
            self.last_speed_ts = now

        elif arb_id == 0x100:
            self.validated_rpm = float(decoded.get("rpm", self.validated_rpm))

        elif arb_id == 0x102:
            self.validated_pedal = float(decoded.get("pedal", self.validated_pedal))
            self.validated_decel = float(decoded.get("decel", self.validated_decel))
            self.validated_brake_active = int(decoded.get("brake_active", self.validated_brake_active))

        elif arb_id == 0x103:
            self.validated_steering = float(decoded.get("steering_deg", self.validated_steering))
            self.last_steering_ts = now

        elif arb_id == 0x104:
            self.validated_gear = int(decoded.get("gear", self.validated_gear))

        elif arb_id == 0x105:
            fl = float(decoded.get("wheel_fl", self.validated_wheel_avg))
            fr = float(decoded.get("wheel_fr", self.validated_wheel_avg))
            rl = float(decoded.get("wheel_rl", self.validated_wheel_avg))
            rr = float(decoded.get("wheel_rr", self.validated_wheel_avg))
            new_avg = (fl + fr + rl + rr) / 4.0

            if self.last_wheel_ts > 0.0 and (now - self.last_wheel_ts) > 0.001:
                self.wheel_speed_trend = (new_avg - self.validated_wheel_avg)
            self.validated_wheel_avg = new_avg
            self.last_wheel_ts = now
            self.has_wheel_speed = True

        elif arb_id == 0x106:
            self.validated_ax = float(decoded.get("ax", self.validated_ax))
            self.validated_yaw = float(decoded.get("yaw", self.validated_yaw))

        elif arb_id == 0x202:
            self.validated_aeb_flag = bool(decoded.get("aeb", self.validated_aeb_flag))
            self.validated_acc_flag = bool(decoded.get("acc", self.validated_acc_flag))
            self.validated_lka_flag = bool(decoded.get("lka", self.validated_lka_flag))

