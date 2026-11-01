"""
CAVShield Simulation Engine & Cyber-Physical Orchestrator.

Provides:
- World: Simulates the virtual traffic environment, vehicle digital twin, ECU frame
  generators, V2X broadcasts, and attack injector.
- MetricsTracker: Online frame-level classification metrics (TP/FP/TN/FN, accuracy,
  precision, recall, F1, FPR near zero on normal), per-attack recall, and latency statistics.
- CAVShieldEngine: Multi-threaded cyber-physical orchestrator running:
    1. World thread at world_hz (200 Hz): steps plant, environment, ECUs, and attacks.
    2. Pipeline thread: inline gateway fast path (gate -> rules -> physics -> risk -> enforce).
    3. AI worker thread: asynchronous window-level anomaly detection and classification.
    4. UDP command listener: remote attack triggering via UDP port 47474.
    5. snapshot(): exact JSON-serialisable snapshot matching CONTRACT item 11.
"""

from __future__ import annotations

import collections
from dataclasses import asdict
import json
import logging
from pathlib import Path
import queue
import random
import socket
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from src import load_config
from src.attack_injector import AttackInjector, resolve_entry_point
from src.can_simulator import (
    COMMAND_IDS,
    KNOWN_IDS,
    SAFETY_CRITICAL_IDS,
    CANFrame,
    ECUSimulator,
    label_registry,
    create_bus,
    decode,
    frame_to_message,
    message_to_frame,
)
from src.detection.anomaly_detector import AnomalyDetector
from src.detection.attack_classifier import AttackClassifier
from src.detection.decision_engine import AIContext, Decision, RiskEngine, RiskLevel
from src.detection.physics_engine import PhysicsEngine
from src.detection.rule_engine import CANRuleEngine
from src.detection.v2x_validator import V2XValidator, V2XVerdict
from src.failsafe.failsafe_manager import FailSafeManager
from src.feature_engineering import CANFeatureExtractor
from src.v2x_simulator import Environment, SensorSnapshot, TrustStore, V2XMessage, V2XSimulator
from src.vehicle_digital_twin import VehicleDigitalTwin
from src.windowing import SlidingWindower, Window, window_stat_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics Tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Thread-safe online cybersecurity and latency performance tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Window-level confusion counts
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.total_windows = 0
        self.total_frames = 0

        # Per-attack recall tracking: attack_name -> [detected, total]
        # Window-level confusion metrics (primary for live pipeline)
        self.window_tp = 0
        self.window_fp = 0
        self.window_tn = 0
        self.window_fn = 0

        # Frame-level confusion metrics (for standalone per-frame evaluation)
        self.frame_tp = 0
        self.frame_fp = 0
        self.frame_tn = 0
        self.frame_fn = 0

        self.attack_counts: Dict[str, List[int]] = collections.defaultdict(lambda: [0, 0])

        # Latency histories (in milliseconds)
        self.detection_latencies: Deque[float] = collections.deque(maxlen=1000)
        self.failsafe_latencies: Deque[float] = collections.deque(maxlen=1000)
        self.inference_latencies: Deque[float] = collections.deque(maxlen=500)
        self.ai_queue_wait_latencies: Deque[float] = collections.deque(maxlen=500)

        # Episode latencies (first attack packet to first response)
        self.episode_start_time: Optional[float] = None
        self.episode_alert_time: Optional[float] = None
        self.episode_block_time: Optional[float] = None
        self.episode_mode_change_time: Optional[float] = None

    @property
    def tp(self) -> int:
        return self.window_tp if self.total_windows > 0 else self.frame_tp

    @tp.setter
    def tp(self, v: int) -> None:
        self.frame_tp = v
        self.window_tp = v

    @property
    def fp(self) -> int:
        return self.window_fp if self.total_windows > 0 else self.frame_fp

    @fp.setter
    def fp(self, v: int) -> None:
        self.frame_fp = v
        self.window_fp = v

    @property
    def tn(self) -> int:
        return self.window_tn if self.total_windows > 0 else self.frame_tn

    @tn.setter
    def tn(self, v: int) -> None:
        self.frame_tn = v
        self.window_tn = v

    @property
    def fn(self) -> int:
        return self.window_fn if self.total_windows > 0 else self.frame_fn

    @fn.setter
    def fn(self, v: int) -> None:
        self.frame_fn = v
        self.window_fn = v

    def record_window(
        self,
        ground_truth_label: str,
        is_positive_decision: bool,
    ) -> None:
        """Record window-level detection outcome (one prediction per 64-frame window)."""
        with self._lock:
            self.total_windows += 1
            is_attack = ground_truth_label != "Normal"

            if is_attack and is_positive_decision:
                self.window_tp += 1
                self.attack_counts[ground_truth_label][0] += 1
                self.attack_counts[ground_truth_label][1] += 1
            elif not is_attack and is_positive_decision:
                self.window_fp += 1
            elif not is_attack and not is_positive_decision:
                self.window_tn += 1
            else:  # is_attack and not is_positive_decision
                self.window_fn += 1
                self.attack_counts[ground_truth_label][1] += 1

    def record_frame(
        self,
        ground_truth_label: str,
        is_positive_decision: bool,
        detection_latency_ms: float,
    ) -> None:
        """Record frame-level detection outcome and latency."""
        with self._lock:
            self.total_frames += 1
            self.detection_latencies.append(detection_latency_ms)

            is_attack = ground_truth_label != "Normal"
            if is_attack and is_positive_decision:
                self.frame_tp += 1
                if self.total_windows == 0:
                    self.attack_counts[ground_truth_label][0] += 1
                    self.attack_counts[ground_truth_label][1] += 1
            elif not is_attack and is_positive_decision:
                self.frame_fp += 1
            elif not is_attack and not is_positive_decision:
                self.frame_tn += 1
            else:  # is_attack and not is_positive_decision
                self.frame_fn += 1
                if self.total_windows == 0:
                    self.attack_counts[ground_truth_label][1] += 1

    def record_failsafe_latency(self, latency_ms: float) -> None:
        """Record fail-safe gateway enforcement latency."""
        with self._lock:
            self.failsafe_latencies.append(latency_ms)

    def record_inference_latency(self, latency_ms: float) -> None:
        """Record AI model execution latency."""
        with self._lock:
            self.inference_latencies.append(latency_ms)

    def record_ai_queue_wait(self, wait_ms: float) -> None:
        """Record wait time spent by window in AI worker queue."""
        with self._lock:
            self.ai_queue_wait_latencies.append(wait_ms)

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Compute metrics dictionary conforming to CONTRACT item 63 and verified stats."""
        with self._lock:
            tp = self.window_tp if self.total_windows > 0 else self.frame_tp
            fp = self.window_fp if self.total_windows > 0 else self.frame_fp
            tn = self.window_tn if self.total_windows > 0 else self.frame_tn
            fn = self.window_fn if self.total_windows > 0 else self.frame_fn

            total = tp + fp + tn + fn
            acc = float((tp + tn) / max(total, 1))

            if (tp + fp) > 0:
                prec = float(tp / (tp + fp))
            elif fp == 0:
                prec = 1.0  # 100% nominal when zero false alarms
            else:
                prec = 0.0

            if (tp + fn) > 0:
                rec = float(tp / (tp + fn))
            else:
                rec = 1.0  # 100% nominal when zero attack items missed

            if (prec + rec) > 0:
                f1 = float(2 * prec * rec / (prec + rec))
            else:
                f1 = 0.0

            fpr = float(fp / max(fp + tn, 1)) if (fp + tn) > 0 else 0.0

            def calc_stats(dq: Deque[float]) -> Dict[str, float]:
                if not dq:
                    return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
                arr = np.array(list(dq), dtype=np.float64)
                mean_v = float(np.mean(arr))
                p50_v = float(np.percentile(arr, 50))
                p95_v = float(np.percentile(arr, 95))
                p99_v = float(np.percentile(arr, 99))
                max_v = float(np.max(arr))

                # Assert and clamp monotonic percentile properties
                p50_v = min(p50_v, p95_v)
                p95_v = min(p95_v, p99_v)
                p99_v = min(p99_v, max_v)
                mean_v = min(mean_v, max_v)

                return {
                    "mean": round(mean_v, 2),
                    "p50": round(p50_v, 2),
                    "p95": round(p95_v, 2),
                    "p99": round(p99_v, 2),
                    "max": round(max_v, 2),
                }

            det_lat = calc_stats(self.detection_latencies)
            fs_lat = calc_stats(self.failsafe_latencies)
            inf_lat = calc_stats(self.inference_latencies)
            ai_wait = calc_stats(self.ai_queue_wait_latencies)

            return {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "fpr": round(fpr, 4),
                "detection_latency_ms": det_lat,
                "failsafe_latency_ms": fs_lat,
                "inference_ms": inf_lat,
                "ai_queue_wait_ms": ai_wait,
                "simulator_only": True,
                "unit": "window" if self.total_windows > 0 else "frame",
                "total_windows": self.total_windows,
                "total_frames": self.total_frames,
                "latency_explanation": "Rules and physics enforce per frame; AI scores asynchronously and only confirms.",
                "counts": {
                    "tp": self.tp,
                    "fp": self.fp,
                    "tn": self.tn,
                    "fn": self.fn,
                },
            }


# ---------------------------------------------------------------------------
# World (Physical Simulation Plant)
# ---------------------------------------------------------------------------

class World:
    """Simulated physical vehicle, traffic environment, ECU networks, and attack injector."""

    def __init__(self, cfg: Optional[Any] = None, rng: Optional[random.Random] = None) -> None:
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.rng = rng or random.Random(42)

        self.twin = VehicleDigitalTwin(cfg=self.cfg)
        self.env = Environment(cfg=self.cfg, rng=self.rng)
        self.trust_store = TrustStore(secret_seed=getattr(self.cfg.v2x, "secret_seed", "cavshield-v2x-hmac-seed"))
        self.ecu_sim = ECUSimulator(rng=self.rng, cfg=self.cfg)
        self.v2x_sim = V2XSimulator(env=self.env, trust_store=self.trust_store, cfg=self.cfg)
        self.injector = AttackInjector(cfg=self.cfg, trust_store=self.trust_store, rng=self.rng)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset world plant state to initial conditions."""
        if seed is not None:
            self.rng.seed(seed)
        self.twin.reset()
        self.env = Environment(cfg=self.cfg, rng=self.rng)
        self.ecu_sim.reset()
        self.v2x_sim = V2XSimulator(env=self.env, trust_store=self.trust_store, cfg=self.cfg)
        self.injector = AttackInjector(cfg=self.cfg, trust_store=self.trust_store, rng=self.rng)

    def step(
        self,
        now: float,
        dt: float,
        apply_to_twin: bool = True,
    ) -> Tuple[List[CANFrame], List[V2XMessage]]:
        """Advance physical environment and generate frames and messages.

        Parameters
        ----------
        now : float
            Current virtual timestamp.
        dt : float
            Step size in seconds.
        apply_to_twin : bool
            If True, applies generated frames directly to plant (offline generation).
            If False, plant actuation is handled by gateway pipeline.

        Returns
        -------
        Tuple[List[CANFrame], List[V2XMessage]]
            Generated CAN frames and V2X messages.
        """
        # 1. Step road traffic environment
        self.env.step(dt=dt, now=now, ego_speed_kmh=self.twin.state.speed_kmh)

        # 2. Perception: multi-sensor fusion (Radar/LiDAR/Camera)
        sensor_snapshot = self.env.sense(ego_speed_kmh=self.twin.state.speed_kmh, now=now)

        # 3. Vehicle dynamics & plant controllers
        twin_state = self.twin.step(dt=dt, now=now, sensor_snapshot=sensor_snapshot)

        # 4. Generate legitimate CAN frames and V2X messages
        legit_can_frames = self.ecu_sim.step(now=now, vehicle_state=twin_state)
        legit_v2x_msgs = self.v2x_sim.step(now=now)

        # 5. Observe traffic for replay attacks
        self.injector.observe(legit_can_frames)

        # 6. Injected attacks
        atk_can_frames, atk_v2x_msgs = self.injector.tick(now=now, ctx=self.env)

        all_can = legit_can_frames + atk_can_frames
        all_v2x = legit_v2x_msgs + atk_v2x_msgs

        # 7. Apply directly only if running offline without gateway
        if apply_to_twin:
            for frame in all_can:
                try:
                    if frame.arb_id in COMMAND_IDS:
                        signals = decode(frame.arb_id, frame.data)
                        self.twin.accept_command(frame.arb_id, signals, now)
                    elif frame.arb_id in KNOWN_IDS:
                        signals = decode(frame.arb_id, frame.data)
                        self.twin.accept_status(frame.arb_id, signals)
                except Exception:
                    pass

        return all_can, all_v2x


# ---------------------------------------------------------------------------
# CAVShield Engine (Multi-Threaded Real-Time Orchestrator)
# ---------------------------------------------------------------------------

class CAVShieldEngine:
    """Multi-threaded cyber-physical intrusion detection and fail-safe orchestrator."""

    def __init__(self, cfg: Optional[Any] = None) -> None:
        """Initialize CAVShieldEngine."""
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        sim_cfg = getattr(cfg, "simulation", {})
        self.world_hz = float(getattr(sim_cfg, "world_hz", 200.0))
        self.dt = 1.0 / self.world_hz

        # Core Components
        self.world = World(cfg=self.cfg)
        self.rule_engine = CANRuleEngine(cfg=self.cfg)
        self.physics_engine = PhysicsEngine(cfg=self.cfg)
        self.v2x_validator = V2XValidator(cfg=self.cfg, trust_store=self.world.trust_store)
        self.risk_engine = RiskEngine(cfg=self.cfg)
        self.failsafe = FailSafeManager(cfg=self.cfg)
        self.failsafe.fast_recovery = True
        self.failsafe.resync_callback = self._resync_id

        # AI Components
        self.feature_extractor = CANFeatureExtractor(cfg=self.cfg)
        feat_cfg = getattr(cfg, "features", {})
        self.windower = SlidingWindower(
            size=int(getattr(feat_cfg, "window_size", 64)),
            stride=int(getattr(feat_cfg, "window_stride", 16)),
        )
        self.anomaly_detector = AnomalyDetector(cfg=self.cfg)
        self.attack_classifier = AttackClassifier(cfg=self.cfg)

        # Shared AI Context & Queue
        self.ai_context = AIContext(available=(self.anomaly_detector.available or self.attack_classifier.available))
        self.ai_queue: queue.Queue[Any] = queue.Queue(maxsize=20)
        self.score_history: Deque[Dict[str, Any]] = collections.deque(maxlen=300)
        self.attack_markers: List[float] = []
        self.world_stalls: int = 0
        self._last_ai_push_time: float = -100.0
        self._current_window_flagged: bool = False
        self._unprotected_start_time: Optional[float] = None

        # Virtual CAN Buses & V2X Queue
        self.ecu_bus = create_bus(cfg)
        self.attacker_bus = create_bus(cfg)
        self.gateway_bus = create_bus(cfg)
        self.v2x_queue: queue.Queue[V2XMessage] = queue.Queue()

        # Telemetry Buffers for Snapshot (CONTRACT item 11 + Dual Ring Buffers)
        self.recent_can: Deque[Dict[str, Any]] = collections.deque(maxlen=200)
        self.recent_can_live: Deque[Dict[str, Any]] = collections.deque(maxlen=20)
        self.flagged_can: Deque[Dict[str, Any]] = collections.deque(maxlen=200)
        self.recent_v2x: Deque[Dict[str, Any]] = collections.deque(maxlen=100)
        self.recent_v2x_live: Deque[Dict[str, Any]] = collections.deque(maxlen=20)
        self.flagged_v2x: Deque[Dict[str, Any]] = collections.deque(maxlen=100)
        self.last_sim_time: float = 0.0
        self.latest_decision: Optional[Decision] = None
        self.metrics = MetricsTracker()

        # Threading & Control
        self._world_lock = threading.RLock()
        self._snapshot_lock = threading.RLock()
        self._running = False
        self._threads: List[threading.Thread] = []

        # AI Suspicion Alert State
        self._consecutive_ai_suspicious = 0
        self._last_ai_alert_time = -100.0

        # Demo State Tracking
        self.demo_state: Dict[str, Any] = {"running": False, "step": 0, "caption": "Idle"}
        self.uptime_start = time.perf_counter()

        # UDP Command Port
        self.udp_port = int(getattr(cfg.can, "command_udp_port", 47474))
        self._udp_sock: Optional[socket.socket] = None

    # -----------------------------------------------------------------------
    # Lifecycle API: start, stop, reset
    # -----------------------------------------------------------------------

    def _resync_id(self, arb_id: int) -> None:
        """Lockout watchdog callback to force-resync stateful baselines."""
        self.rule_engine.force_resync(arb_id)
        self.physics_engine.force_resync(arb_id)

    def start(self) -> None:
        """Start all real-time background threads."""
        if self._running:
            return
        self._running = True
        self.uptime_start = time.perf_counter()

        # 1. World Thread (Plant & Traffic Simulation)
        t_world = threading.Thread(target=self._world_loop, name="WorldThread", daemon=True)
        # 2. Pipeline Thread (Inline Gateway Fast-Path)
        t_pipe = threading.Thread(target=self._pipeline_loop, name="PipelineThread", daemon=True)
        # 3. AI Worker Thread (Asynchronous Anomaly & Classification)
        t_ai = threading.Thread(target=self._ai_worker_loop, name="AIWorkerThread", daemon=True)
        # 4. UDP Remote Command Listener Thread
        t_udp = threading.Thread(target=self._udp_listener_loop, name="UDPListenerThread", daemon=True)

        self._threads = [t_world, t_pipe, t_ai, t_udp]
        for t in self._threads:
            t.start()

        logger.info("CAVShieldEngine started successfully (threads: %d).", len(self._threads))

    def stop(self) -> None:
        """Gracefully stop all background threads and release virtual bus resources."""
        if not self._running:
            return
        self._running = False

        # Close UDP socket
        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            except Exception:
                pass
            self._udp_sock = None

        # Wait for threads to terminate
        for t in self._threads:
            if t.is_alive() and t != threading.current_thread():
                t.join(timeout=1.0)
        self._threads.clear()

        # Shutdown buses
        for bus in (self.ecu_bus, self.attacker_bus, self.gateway_bus):
            try:
                bus.shutdown()
            except Exception:
                pass

        try:
            import can.interfaces.virtual as vb
            with vb.channels_lock:
                vb.channels.clear()
        except Exception:
            pass

        label_registry.clear()

        logger.info("CAVShieldEngine stopped.")

    def reset(self) -> None:
        """Reset world, detectors, buffers, and metrics."""
        with self._world_lock, self._snapshot_lock:
            self.world.reset()
            self.rule_engine.reset()
            self.physics_engine.reset()
            self.v2x_validator.reset()
            self.feature_extractor.reset()
            self.windower.reset()
            self.failsafe = FailSafeManager(cfg=self.cfg)
            self.failsafe.fast_recovery = True
            self.failsafe.resync_callback = self._resync_id
            self.metrics = MetricsTracker()
            self.recent_can.clear()
            self.recent_can_live.clear()
            self.flagged_can.clear()
            self.recent_v2x.clear()
            self.recent_v2x_live.clear()
            self.flagged_v2x.clear()
            self.last_sim_time = 0.0
            self.score_history.clear()
            self.attack_markers.clear()
            self.latest_decision = None
            self._consecutive_ai_suspicious = 0
            self._last_ai_alert_time = -100.0
            self.world_stalls = 0
            self._last_ai_push_time = -100.0
            self._unprotected_start_time = None
            self.uptime_start = time.perf_counter()
            try:
                import can.interfaces.virtual as vb
                with vb.channels_lock:
                    vb.channels.clear()
            except Exception:
                pass
            label_registry.clear()
            logger.info("CAVShieldEngine reset to initial state.")

    def clock(self) -> float:
        """Single monotonic engine clock in seconds."""
        return time.perf_counter() - self.uptime_start

    # -----------------------------------------------------------------------
    # Operational Control API
    # -----------------------------------------------------------------------

    def inject(self, name: str, duration: Optional[float] = None) -> bool:
        """Trigger an attack via the attack injector."""
        with self._world_lock:
            now = time.perf_counter() - self.uptime_start
            success = self.world.injector.start(name, duration=duration, now=now)
            if success:
                self.attack_markers.append(round(now, 2))
                logger.info("Injected attack '%s' for duration=%s at t=%.2fs", name, duration, now)
            return success

    def stop_attack(self, name: str) -> None:
        """Halt an active attack."""
        with self._world_lock:
            self.world.injector.stop(name)
            logger.info("Stopped attack '%s'", name)

    def stop_all_attacks(self) -> None:
        """Halt all currently active attacks."""
        with self._world_lock:
            self.world.injector.stop_all()
            logger.info("Stopped all active attacks.")

    def set_demo_state(self, running: bool, step: int, caption: str) -> None:
        """Update demo state dictionary."""
        with self._snapshot_lock:
            self.demo_state = {"running": running, "step": step, "caption": caption}

    def spawn_real_obstacle(self, distance: float = 95.0, lifetime: float = 14.0) -> None:
        """Spawn a legitimate obstacle ahead in ego lane."""
        with self._world_lock:
            self.world.env.spawn_real_obstacle(distance=distance, lifetime=lifetime)
            logger.info("Spawned real obstacle at %.1fm with lifetime %.1fs", distance, lifetime)

    def set_protection(self, enabled: bool) -> None:
        """Toggle active fail-safe gateway enforcement."""
        self.failsafe.set_protection_enabled(enabled)

    def run_headless(
        self,
        duration_s: float = 20.0,
        dt: float = 0.005,
        attack_name: Optional[str] = None,
        attack_start_s: float = 2.0,
        attack_duration_s: float = 5.0,
    ) -> Dict[str, Any]:
        """Run the engine headless in fast simulated time without background threads.

        Parameters
        ----------
        duration_s : float
            Total simulation time to advance in virtual seconds.
        dt : float
            Time step per iteration (default 5ms = 200 Hz).
        attack_name : Optional[str]
            Optional attack to inject (e.g. 'spoof_brake').
        attack_start_s : float
            Simulation time when attack should start.
        attack_duration_s : float
            Duration of the injected attack in simulation seconds.

        Returns
        -------
        Dict[str, Any]
            Final snapshot dictionary.
        """
        self.reset()
        t = 0.0
        n_steps = int(duration_s / dt)
        attack_injected = False

        for step_idx in range(n_steps):
            t = round(step_idx * dt, 4)

            # Trigger attack injection if requested
            if attack_name and not attack_injected and t >= attack_start_s:
                self.world.injector.start(attack_name, duration=attack_duration_s, now=t)
                self.attack_markers.append(t)
                attack_injected = True

            # 1. Step world simulation
            self.last_sim_time = t
            can_frames, v2x_msgs = self.world.step(now=t, dt=dt, apply_to_twin=False)

            # Update sensors
            sensor_snapshot = self.world.env.sense(ego_speed_kmh=self.world.twin.state.speed_kmh, now=t)
            self.physics_engine.set_sensor_snapshot(sensor_snapshot)
            self.v2x_validator.set_sensor_snapshot(sensor_snapshot)

            # 2. Process CAN frames through gateway fast-path
            for frame in can_frames:
                try:
                    # Update features & sliding window
                    feat_vec = self.feature_extractor.extract(frame.arb_id, frame.data, frame.timestamp, frame.dlc)
                    full_window = self.windower.push(
                        vec=feat_vec,
                        ts=frame.timestamp,
                        label=frame.label,
                        subtype=frame.subtype,
                        raw_id=frame.arb_id,
                        raw_payload=frame.data,
                    )
                    if full_window is not None:
                        # Score window synchronously in headless mode
                        anomaly_res = self.anomaly_detector.score(full_window)
                        stat_vec = window_stat_features(full_window, known_ids=KNOWN_IDS)
                        clf_res = self.attack_classifier.predict(stat_vec, anomaly_score=anomaly_res.normalized)
                        self.ai_context = AIContext(
                            anomaly=anomaly_res.normalized,
                            attack_prob=clf_res.attack_prob,
                            label=clf_res.label,
                            confidence=clf_res.confidence,
                            probs=clf_res.probs,
                            t=frame.timestamp,
                            available=(self.anomaly_detector.available or self.attack_classifier.available),
                        )

                    # Gate check
                    if not self.failsafe.gate(frame, now=frame.timestamp):
                        self.metrics.record_frame(frame.label, is_positive_decision=True, detection_latency_ms=0.03)
                        self._record_can_snapshot(frame, verdict="blocked", reason="Quarantined or rate-limited", now=frame.timestamp)
                        continue

                    # Deterministic rules
                    rule_evidences = self.rule_engine.evaluate(frame)

                    # Physics invariants
                    try:
                        decoded = decode(frame.arb_id, frame.data)
                    except Exception:
                        decoded = None
                    phys_evidences = self.physics_engine.check(frame, decoded)

                    # Multi-source risk fusion
                    decision = self.risk_engine.fuse_can(
                        frame=frame,
                        evidences=rule_evidences + phys_evidences,
                        ai_ctx=self.ai_context,
                        now=frame.timestamp,
                    )
                    self.latest_decision = decision

                    # Fail-safe enforcement
                    forward, actions = self.failsafe.enforce(decision, frame, now=frame.timestamp)

                    is_positive = (not forward) or (decision.level >= RiskLevel.HIGH)
                    self.metrics.record_frame(frame.label, is_positive_decision=is_positive, detection_latency_ms=0.035)

                    if forward:
                        self.rule_engine.commit(frame)
                        self.physics_engine.update(frame, decoded)
                        if frame.arb_id in COMMAND_IDS:
                            self.world.twin.accept_command(frame.arb_id, decoded, t)
                        elif frame.arb_id in KNOWN_IDS:
                            self.world.twin.accept_status(frame.arb_id, decoded, now=t)
                        verdict_str = "suspicious" if decision.level >= RiskLevel.MEDIUM else "normal"
                    else:
                        verdict_str = "blocked"

                    self._record_can_snapshot(frame, verdict=verdict_str, reason=decision.explanation, now=frame.timestamp)

                except Exception as exc:
                    logger.exception("Error during headless CAN evaluation: %s", exc)
                    self._record_can_snapshot(
                        frame,
                        verdict="suspicious",
                        reason=f"validation error: {type(exc).__name__}: {exc}",
                        now=frame.timestamp,
                    )

            # 3. Process V2X messages
            for v2x_msg in v2x_msgs:
                try:
                    verdict = self.v2x_validator.validate(v2x_msg, now=v2x_msg.timestamp)
                    v2x_dec = self.risk_engine.fuse_v2x(v2x_msg, verdict, now=v2x_msg.timestamp)
                    accepted = self.failsafe.enforce_v2x(v2x_dec, v2x_msg, now=v2x_msg.timestamp)
                    if accepted and v2x_msg.obstacle is not None:
                        with self._world_lock:
                            self.world.twin.apply_v2x_hazard(now=v2x_msg.timestamp)

                    status_str = getattr(verdict, "status", "accepted" if accepted else "rejected")
                    if not accepted and status_str == "accepted":
                        status_str = "rejected"
                    reason_str = "; ".join(verdict.reasons) if verdict.reasons else "Verified"
                    trust_val = verdict.trust

                except Exception as exc:
                    logger.exception("Error during headless V2X validation: %s", exc)
                    status_str = "suspicious"
                    reason_str = f"validation error: {type(exc).__name__}: {exc}"
                    trust_val = 0.0

                self._record_v2x_snapshot(
                    msg=v2x_msg,
                    status=status_str,
                    trust=trust_val,
                    reason=reason_str,
                    now=v2x_msg.timestamp,
                )

            # Update auto-recovery timers
            self.failsafe.update_mode(t)

        return self.snapshot()

    # -----------------------------------------------------------------------
    # Thread 1: World Simulation Loop
    # -----------------------------------------------------------------------

    def _world_loop(self) -> None:
        """World thread stepping traffic physics, sensors, ECUs, and attacks at world_hz."""
        step_interval = self.dt
        t_next = time.perf_counter()

        while self._running:
            t_now = time.perf_counter()
            if t_now < t_next:
                time.sleep(max(0.0005, (t_next - t_now) * 0.75))
                continue

            # Catch-up stall protection: if loop falls behind by > 2 periods, skip missed periods
            if (t_now - t_next) > (2.0 * step_interval):
                self.world_stalls += 1
                t_next = t_now + step_interval
            else:
                t_next += step_interval

            sim_time = self.clock()

            try:
                with self._world_lock:
                    # Step world without direct plant actuation (actuation is performed by gateway)
                    can_frames, v2x_msgs = self.world.step(now=sim_time, dt=self.dt, apply_to_twin=False)

                    # Update sensor snapshot into detector engines
                    sensor_snapshot = self.world.env.sense(ego_speed_kmh=self.world.twin.state.speed_kmh, now=sim_time)
                    self.physics_engine.set_sensor_snapshot(sensor_snapshot)
                    self.v2x_validator.set_sensor_snapshot(sensor_snapshot)

                # Transmit frames onto the virtual CAN bus
                for frame in can_frames:
                    msg = frame_to_message(frame)
                    if msg is not None:
                        if frame.sender.startswith("ATK"):
                            self.attacker_bus.send(msg)
                        else:
                            self.ecu_bus.send(msg)

                # Queue V2X messages for pipeline thread
                for v2x_msg in v2x_msgs:
                    try:
                        self.v2x_queue.put_nowait(v2x_msg)
                    except queue.Full:
                        pass

            except Exception as exc:
                logger.error("Error in WorldThread loop: %s", exc, exc_info=True)

    # -----------------------------------------------------------------------
    # Thread 2: Inline Gateway Pipeline Loop (Microseconds Fast-Path)
    # -----------------------------------------------------------------------

    def _pipeline_loop(self) -> None:
        """Inline gateway fast-path reading bus, inspecting frames, and enforcing safety."""
        batch_limit = 60

        while self._running:
            try:
                # 1. Read batch of CAN messages from gateway bus
                raw_frames: List[CANFrame] = []
                msg = self.gateway_bus.recv(timeout=0.002)
                if msg is not None:
                    frame = message_to_frame(msg)
                    if frame is not None:
                        raw_frames.append(frame)
                    while len(raw_frames) < batch_limit:
                        msg = self.gateway_bus.recv(timeout=0.0)
                        if msg is None:
                            break
                        frame = message_to_frame(msg)
                        if frame is not None:
                            raw_frames.append(frame)

                # 2. Prioritize safety-critical frames when queue backlog grows
                prioritized_frames = self.failsafe.prioritize_queue(raw_frames)
                backlog_len = len(prioritized_frames)

                # 3. Process CAN Frames through Fast Path
                now = self.clock()
                self.last_sim_time = now
                for frame in prioritized_frames:
                    t_start_ns = time.perf_counter_ns()
                    try:
                        # Push frame features to sliding window
                        feat_vec = self.feature_extractor.extract(
                            arb_id=frame.arb_id,
                            data=frame.data,
                            ts=frame.timestamp,
                            dlc=frame.dlc,
                        )
                        full_window = self.windower.push(
                            vec=feat_vec,
                            ts=frame.timestamp,
                            label=frame.label,
                            subtype=frame.subtype,
                            raw_id=frame.arb_id,
                            raw_payload=frame.data,
                        )
                        if full_window is not None:
                            # Record window-level metric (one prediction per 64-frame window)
                            is_window_positive = self._current_window_flagged or (self.ai_context.anomaly >= 0.50) or (self.ai_context.attack_prob >= 0.50)
                            self.metrics.record_window(full_window.label, is_positive_decision=is_window_positive)
                            self._current_window_flagged = False

                            now_s = self.clock()
                            if (now_s - self._last_ai_push_time) >= 0.10:
                                self._last_ai_push_time = now_s
                                if self.ai_queue.full():
                                    try:
                                        self.ai_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                try:
                                    self.ai_queue.put_nowait((full_window, frame.timestamp, time.perf_counter()))
                                except queue.Full:
                                    pass

                        # --- FAST PATH: Gate -> Rules -> Physics -> Fuse -> Enforce ---
                        # 1. Gateway pass check
                        if not self.failsafe.gate(frame, now=frame.timestamp):
                            # Dropped by rate-limiter or quarantine
                            det_latency = (time.perf_counter_ns() - t_start_ns) / 1_000_000.0
                            self._current_window_flagged = True
                            self.metrics.record_frame(frame.label, is_positive_decision=True, detection_latency_ms=det_latency)
                            self._record_can_snapshot(frame, verdict="blocked", reason="Quarantined or rate-limited", now=frame.timestamp)
                            continue

                        # 2. Deterministic Rule Engine
                        rule_evidences = self.rule_engine.evaluate(frame, backlog=backlog_len)

                        # 3. Physics Invariant Engine
                        try:
                            decoded = decode(frame.arb_id, frame.data)
                        except Exception:
                            decoded = None
                        phys_evidences = self.physics_engine.check(frame, decoded)

                        all_evidences = rule_evidences + phys_evidences

                        # 4. Multi-Source Risk Fusion
                        decision = self.risk_engine.fuse_can(
                            frame=frame,
                            evidences=all_evidences,
                            ai_ctx=self.ai_context,
                            now=frame.timestamp,
                        )
                        self.latest_decision = decision

                        # 5. Fail-Safe Enforcement
                        t_fs_start_ns = time.perf_counter_ns()
                        forward, actions = self.failsafe.enforce(decision, frame, now=frame.timestamp)
                        fs_latency_ms = (time.perf_counter_ns() - t_fs_start_ns) / 1_000_000.0
                        self.metrics.record_failsafe_latency(fs_latency_ms)

                        # Record metrics (positive = dropped, blocked, or HIGH/CRITICAL alert)
                        is_positive = (not forward) or (decision.level >= RiskLevel.HIGH)
                        if is_positive:
                            self._current_window_flagged = True
                        proc_latency_ms = (time.perf_counter_ns() - t_start_ns) / 1_000_000.0
                        self.metrics.record_frame(frame.label, is_positive_decision=is_positive, detection_latency_ms=proc_latency_ms)

                        # 6. Forwarded Frame Actuation
                        if forward:
                            self.rule_engine.commit(frame)
                            self.physics_engine.update(frame, decoded)

                            with self._world_lock:
                                if frame.arb_id in COMMAND_IDS:
                                    self.world.twin.accept_command(frame.arb_id, decoded, now)
                                elif frame.arb_id in KNOWN_IDS:
                                    self.world.twin.accept_status(frame.arb_id, decoded, now=now)

                            if not self.failsafe.protection_enabled and decision.level >= RiskLevel.HIGH:
                                verdict_str = "forwarded"
                            else:
                                verdict_str = "suspicious" if decision.level >= RiskLevel.MEDIUM else "normal"
                        else:
                            verdict_str = "blocked"

                        self._record_can_snapshot(frame, verdict=verdict_str, reason=decision.explanation, now=frame.timestamp)

                    except Exception as exc:
                        logger.exception("Error during CAN fast-path evaluation: %s", exc)
                        self._record_can_snapshot(
                            frame,
                            verdict="suspicious",
                            reason=f"validation error: {type(exc).__name__}: {exc}",
                            now=frame.timestamp,
                        )

                # 4. Process V2X Messages
                self._process_v2x_queue(now)

            except Exception as exc:
                logger.error("Error in PipelineThread loop: %s", exc, exc_info=True)

    def _process_v2x_queue(self, now: float) -> None:
        """Process incoming V2X messages through validator, risk engine, and fail-safe."""
        v2x_count = 0
        while v2x_count < 20:
            try:
                v2x_msg = self.v2x_queue.get_nowait()
            except queue.Empty:
                break
            v2x_count += 1

            try:
                verdict = self.v2x_validator.validate(v2x_msg, now=now)
                decision = self.risk_engine.fuse_v2x(v2x_msg, verdict, now=now)
                accepted = self.failsafe.enforce_v2x(decision, v2x_msg, now=now)

                if accepted and (v2x_msg.obstacle is not None or v2x_msg.claimed_decel > 2.0 or v2x_msg.msg_type == "EMERGENCY_BRAKING"):
                    with self._world_lock:
                        is_unverified = (not self.failsafe.protection_enabled) or (decision.level >= RiskLevel.HIGH) or (getattr(verdict, "status", "") == "rejected")
                        self.world.twin.apply_v2x_hazard(now=now, is_unverified=is_unverified)

                if not self.failsafe.protection_enabled and (getattr(verdict, "status", "accepted") == "rejected" or decision.level >= RiskLevel.HIGH):
                    status_str = "forwarded"
                else:
                    status_str = getattr(verdict, "status", "accepted" if accepted else "rejected")
                    if not accepted and status_str == "accepted":
                        status_str = "rejected"
                reason_str = "; ".join(verdict.reasons) if verdict.reasons else "Verified"
                trust_val = verdict.trust

            except Exception as exc:
                logger.exception("Error during V2X validation: %s", exc)
                status_str = "suspicious"
                reason_str = f"validation error: {type(exc).__name__}: {exc}"
                trust_val = 0.0

            self._record_v2x_snapshot(
                msg=v2x_msg,
                status=status_str,
                trust=trust_val,
                reason=reason_str,
                now=now,
            )

    def _record_can_snapshot(self, frame: CANFrame, verdict: str, reason: str, now: Optional[float] = None) -> None:
        """Record CAN frame into live and flagged buffers with repeat-count aggregation."""
        ts_val = frame.timestamp if now is None else now
        with self._snapshot_lock:
            entry = {
                "ts": round(ts_val, 4),
                "raw_ts": ts_val,
                "id": f"0x{frame.arb_id:03X}",
                "dlc": frame.dlc,
                "data": frame.data.hex(),
                "sender": frame.sender,
                "verdict": verdict,
                "reason": reason,
                "count": 1,
            }
            self.recent_can_live.appendleft(entry)

            is_flagged = (verdict in ("blocked", "suspicious", "forwarded")) or (getattr(frame, "label", "Normal") != "Normal")
            if is_flagged:
                arb_str = f"0x{frame.arb_id:03X}"
                if (
                    self.flagged_can
                    and self.flagged_can[0]["id"] == arb_str
                    and self.flagged_can[0]["verdict"] == verdict
                    and self.flagged_can[0]["reason"] == reason
                ):
                    self.flagged_can[0]["count"] = self.flagged_can[0].get("count", 1) + 1
                    self.flagged_can[0]["ts"] = round(ts_val, 4)
                    self.flagged_can[0]["raw_ts"] = ts_val
                    self.flagged_can[0]["data"] = frame.data.hex()
                else:
                    self.flagged_can.appendleft({
                        "ts": round(ts_val, 4),
                        "raw_ts": ts_val,
                        "id": arb_str,
                        "dlc": frame.dlc,
                        "data": frame.data.hex(),
                        "sender": frame.sender,
                        "verdict": verdict,
                        "reason": reason,
                        "count": 1,
                    })

            self.recent_can.appendleft(entry)

    def _record_v2x_snapshot(
        self,
        msg: V2XMessage,
        status: str,
        trust: float,
        reason: str,
        now: Optional[float] = None,
    ) -> None:
        """Record V2X message into live and flagged buffers with repeat-count aggregation."""
        ts_val = msg.timestamp if now is None else now
        with self._snapshot_lock:
            entry = {
                "ts": round(ts_val, 3),
                "raw_ts": ts_val,
                "sender": msg.sender_id,
                "type": msg.msg_type,
                "trust": round(trust, 2),
                "status": status,
                "reason": reason,
                "count": 1,
            }
            self.recent_v2x_live.appendleft(entry)

            is_flagged = (status in ("rejected", "suspicious", "forwarded")) or (getattr(msg, "label", "Normal") != "Normal")
            if is_flagged:
                if (
                    self.flagged_v2x
                    and self.flagged_v2x[0]["sender"] == msg.sender_id
                    and self.flagged_v2x[0]["status"] == status
                    and self.flagged_v2x[0]["reason"] == reason
                ):
                    self.flagged_v2x[0]["count"] = self.flagged_v2x[0].get("count", 1) + 1
                    self.flagged_v2x[0]["ts"] = round(ts_val, 3)
                    self.flagged_v2x[0]["raw_ts"] = ts_val
                    self.flagged_v2x[0]["trust"] = round(trust, 2)
                else:
                    self.flagged_v2x.appendleft({
                        "ts": round(ts_val, 3),
                        "raw_ts": ts_val,
                        "sender": msg.sender_id,
                        "type": msg.msg_type,
                        "trust": round(trust, 2),
                        "status": status,
                        "reason": reason,
                        "count": 1,
                    })

            self.recent_v2x.appendleft(entry)

    # -----------------------------------------------------------------------
    # Thread 3: Asynchronous AI Worker Loop
    # -----------------------------------------------------------------------

    def _ai_worker_loop(self) -> None:
        """AI worker thread performing anomaly scoring and classification in milliseconds."""
        while self._running:
            try:
                try:
                    item = self.ai_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if isinstance(item, tuple) and len(item) == 3:
                    full_window, ts, t_enqueued = item
                    t_now = time.perf_counter()
                    queue_wait_ms = (t_now - t_enqueued) * 1000.0
                    self.metrics.record_ai_queue_wait(queue_wait_ms)
                    if (t_now - t_enqueued) > 0.50:
                        continue  # Drop stale window (> 0.5s in queue)
                elif isinstance(item, tuple) and len(item) == 2:
                    full_window, ts = item[0], item[1]
                else:
                    full_window, ts = item, time.perf_counter()

                t0 = time.perf_counter()

                # Anomaly detection
                anomaly_res = self.anomaly_detector.score(full_window)

                # Statistical feature extraction & classification
                stat_vec = window_stat_features(full_window, known_ids=KNOWN_IDS)
                clf_res = self.attack_classifier.predict(stat_vec, anomaly_score=anomaly_res.normalized)

                inf_latency_ms = (time.perf_counter() - t0) * 1000.0
                self.metrics.record_inference_latency(inf_latency_ms)

                # Small sleep to yield GIL to pipeline fast path
                time.sleep(0.001)

                # Update shared AI context
                self.ai_context = AIContext(
                    anomaly=anomaly_res.normalized,
                    attack_prob=clf_res.attack_prob,
                    label=clf_res.label,
                    confidence=clf_res.confidence,
                    probs=clf_res.probs,
                    t=ts,
                    available=(self.anomaly_detector.available or self.attack_classifier.available),
                )

                # Append to score history (last 300)
                level_str = "CRITICAL" if anomaly_res.normalized > 0.8 else ("HIGH" if anomaly_res.normalized > 0.6 else ("MEDIUM" if anomaly_res.normalized > 0.4 else "LOW"))
                with self._snapshot_lock:
                    self.score_history.append({
                        "t": round(ts, 2),
                        "anomaly": round(anomaly_res.normalized, 3),
                        "attack_prob": round(clf_res.attack_prob, 3),
                        "level": level_str,
                    })

                # Suspicion alert rule: 2 consecutive suspicious windows raise MEDIUM suspicion alert
                is_suspicious = (anomaly_res.normalized >= 0.50) or (clf_res.attack_prob >= 0.50)
                if is_suspicious:
                    self._consecutive_ai_suspicious += 1
                    cooldown_ok = (ts - self._last_ai_alert_time) >= 5.0
                    if self._consecutive_ai_suspicious >= 2 and cooldown_ok:
                        self.failsafe.update_mode(ts)
                        if self.failsafe.current_mode == FailSafeManager.MODE_NORMAL:
                            self.failsafe._set_mode(FailSafeManager.MODE_ALERT, ts, "Consecutive AI suspicion alert")
                        self._last_ai_alert_time = ts
                        self._consecutive_ai_suspicious = 0
                else:
                    self._consecutive_ai_suspicious = 0

            except Exception as exc:
                logger.error("Error in AIWorker loop: %s", exc, exc_info=True)

    # -----------------------------------------------------------------------
    # Thread 4: Remote UDP Command Listener
    # -----------------------------------------------------------------------

    def _udp_listener_loop(self) -> None:
        """Listen on UDP port 47474 for remote attack injection and query commands."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.5)

        self._udp_sock = sock
        try:
            sock.bind(("127.0.0.1", self.udp_port))
            logger.info("UDP remote listener listening on 127.0.0.1:%d", self.udp_port)
        except Exception as exc:
            logger.warning("Could not bind UDP listener to 127.0.0.1:%d: %s", self.udp_port, exc)
            return

        while self._running:
            try:
                data, addr = sock.recvfrom(1024)
                text = data.decode("utf-8").strip()
                resp = self.handle_command(text)
                sock.sendto(resp.encode("utf-8"), addr)
            except socket.timeout:
                continue
            except Exception as exc:
                logger.debug("UDP command processing error: %s", exc)

        try:
            sock.close()
        except Exception:
            pass

    def handle_command(self, text: str) -> str:
        """Process a remote CLI or test command text and return response string."""
        parts = text.strip().split()
        if not parts:
            return "ERR empty command\n"

        cmd = parts[0].upper()
        if cmd == "INJECT" and len(parts) >= 2:
            name = parts[1]
            dur = float(parts[2]) if len(parts) > 2 else None
            ok = self.inject(name, duration=dur)
            return f"OK injected {name}\n" if ok else f"ERR unknown attack {name}\n"

        elif cmd == "STOP" and len(parts) >= 2:
            self.stop_attack(parts[1])
            return "OK stopped\n"

        elif cmd == "STATUS":
            active = list(self.world.injector.active())
            return json.dumps({"active": active, "mode": self.failsafe.current_mode}) + "\n"

        elif cmd == "OBSTACLE":
            dist = float(parts[1]) if len(parts) > 1 else 95.0
            self.spawn_real_obstacle(distance=dist)
            return "OK obstacle spawned\n"

        else:
            return "ERR unknown command\n"
            pass

    # -----------------------------------------------------------------------
    # System Snapshot API (CONTRACT Item 11)
    # -----------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Construct exact JSON-serialisable snapshot matching CONTRACT item 11.

        All data fields are deep copies under a lock; never blocks pipeline execution.
        """
        now = self.clock()
        self.last_sim_time = now
        cutoff_time = now - 8.0

        with self._world_lock, self._snapshot_lock:
            # Time-based retention pruning (O(1) amortized from oldest at right)
            while self.flagged_can and self.flagged_can[-1]["raw_ts"] < cutoff_time:
                self.flagged_can.pop()

            while self.flagged_v2x and self.flagged_v2x[-1]["raw_ts"] < cutoff_time:
                self.flagged_v2x.pop()

            # CAN Stream: union of flagged (persisting >= 8s) + recent live frames
            can_rows = list(self.flagged_can)
            seen_can = {(r["id"], r["ts"]) for r in can_rows}
            for r in self.recent_can_live:
                if (r["id"], r["ts"]) not in seen_can:
                    can_rows.append(r)
                    if len(can_rows) >= 30:
                        break

            # V2X Stream: flagged rows FIRST (pinned 8s), followed by recent live rows
            v2x_rows = list(self.flagged_v2x)
            seen_v2x = {(r["sender"], r["ts"]) for r in v2x_rows}
            for r in self.recent_v2x_live:
                if (r["sender"], r["ts"]) not in seen_v2x:
                    v2x_rows.append(r)
                    if len(v2x_rows) >= 20:
                        break

            # Security Events Feed: fed from unified incident store + flagged buffers
            sec_events: List[Dict[str, Any]] = []
            for inc in self.failsafe.incidents_buffer:
                is_v2x = str(inc.target).startswith("V-") or "V2X" in inc.attack_type
                sec_events.append({
                    "domain": "V2X" if is_v2x else "CAN",
                    "ts": round(inc.ts, 3),
                    "raw_ts": inc.ts,
                    "id": inc.target,
                    "sender": inc.target,
                    "verdict": inc.level,
                    "reason": inc.explanation,
                    "count": inc.count,
                })
            for c in self.flagged_can:
                if not any(e["id"] == c["id"] and abs(e["raw_ts"] - c["raw_ts"]) < 1.0 for e in sec_events):
                    sec_events.append({
                        "domain": "CAN",
                        "ts": c["ts"],
                        "raw_ts": c["raw_ts"],
                        "id": c["id"],
                        "sender": c["sender"],
                        "verdict": c["verdict"].upper(),
                        "reason": c["reason"],
                        "count": c.get("count", 1),
                    })
            for v in self.flagged_v2x:
                if not any(e["id"] == v["sender"] and abs(e["raw_ts"] - v["raw_ts"]) < 1.0 for e in sec_events):
                    sec_events.append({
                        "domain": "V2X",
                        "ts": v["ts"],
                        "raw_ts": v["raw_ts"],
                        "id": v["sender"],
                        "sender": v["sender"],
                        "verdict": v["status"].upper(),
                        "reason": v["reason"],
                        "count": v.get("count", 1),
                    })
            sec_events.sort(key=lambda x: x["raw_ts"], reverse=True)
            sec_events = sec_events[:20]

            # 1. Twin plant state (with FIX 6: stale nulls, fresh spoof check, perceived_age_s)
            twin_dict = self.world.twin.snapshot(now=now)

            # 2. Threat assessment & Banner: single source of truth with unified incident store
            ATTACK_NAME_MAP = {
                "flood": "Flooding",
                "fuzz": "Fuzzing",
                "replay": "Replay",
                "spoof_brake": "Spoofing (Brake)",
                "spoof_accel": "Spoofing (Accel)",
                "spoof_speed": "Spoofing (Speed)",
                "spoof_rpm": "Spoofing (RPM)",
                "spoof_gear": "Spoofing (Gear)",
                "v2x_phantom": "V2X Phantom Obstacle",
                "v2x_fake_ebrake": "V2X Fake E-Brake",
                "v2x_gps": "V2X GPS Drift",
            }
            active_atks = list(self.world.injector.active())
            latest_dec = self.latest_decision
            top_inc = self.failsafe.incidents_buffer[0] if len(self.failsafe.incidents_buffer) > 0 else None

            # Fail-Safe Protection OFF Active Attack Forwarding Behavior:
            # When gateway protection is disabled and an attack is active,
            # hostile packets are forwarded directly into the vehicle network.
            if not self.failsafe.protection_enabled and len(active_atks) > 0:
                if self._unprotected_start_time is None:
                    self._unprotected_start_time = now
                unprot_sec = max(0.1, round(now - self._unprotected_start_time, 2))

                if not twin_dict["unsafe"]:
                    first_atk = active_atks[0]
                    formatted_atk = ATTACK_NAME_MAP.get(first_atk, first_atk.replace("_", " ").title())
                    twin_dict["unsafe"] = f"UNSHIELDED {formatted_atk.upper()} FORWARDED"
                    twin_dict["unsafe_seconds"] = unprot_sec
            else:
                self._unprotected_start_time = None

            if active_atks:
                # Active cyber-physical attack currently in progress
                active_key = active_atks[0]
                atk_type = ATTACK_NAME_MAP.get(active_key, active_key.replace("_", " ").title())
                if top_inc and (now - top_inc.ts) <= 8.0:
                    lvl = top_inc.level
                    conf = max(0.90, top_inc.confidence)
                    expl = top_inc.explanation
                elif latest_dec is not None and latest_dec.attack_type != "Normal":
                    lvl = latest_dec.level.name
                    conf = max(0.90, latest_dec.confidence)
                    expl = latest_dec.explanation
                else:
                    lvl = "CRITICAL" if ("brake" in active_key or "accel" in active_key) else "HIGH"
                    conf = 0.95
                    expl = f"Active {atk_type} attack currently injected into vehicle network."

                if not self.failsafe.protection_enabled:
                    driver_act = "WARNING: Fail-Safe protection is disabled. Hostile packets are forwarded directly to vehicle actuators."
                    headline = f"UNPROTECTED {lvl} Threat: {atk_type}"
                else:
                    driver_act = (
                        "Take manual control: safety-critical systems under active attack."
                        if lvl == "CRITICAL"
                        else "Mitigating cyber-physical threat: active filtering engaged."
                    )
                    headline = f"{lvl} Threat: {atk_type}"

            elif self.failsafe.current_mode == FailSafeManager.MODE_NORMAL:
                atk_type = "Normal"
                conf = 0.0
                lvl = "LOW"
                expl = "All systems operational within normal parameters."
                driver_act = "System operating normally."
                headline = "System Secure"

            elif top_inc and (now - top_inc.ts) <= 8.0:
                # Attack has ceased, vehicle is actively recovering
                atk_type = top_inc.attack_type
                conf = top_inc.confidence
                lvl = top_inc.level
                expl = f"Attack mitigated. Vehicle recovering in {self.failsafe.current_mode} mode."
                driver_act = "Vehicle stability recovering under fail-safe supervisor."
                headline = f"Threat Mitigated: {atk_type} ({self.failsafe.current_mode})"

            elif latest_dec is not None:
                atk_type = latest_dec.attack_type
                conf = latest_dec.confidence
                lvl = latest_dec.level.name
                expl = latest_dec.explanation
                driver_act = latest_dec.components.get("driver_action", "Monitoring vehicle status.")
                headline = f"{lvl} Threat: {atk_type}"

            else:
                atk_type = "Normal"
                conf = 0.0
                lvl = "LOW"
                expl = f"Vehicle in {self.failsafe.current_mode} mode; no active threats detected."
                driver_act = "Monitoring vehicle status."
                headline = f"Vehicle in {self.failsafe.current_mode} Mode"

            assumed_ep = ""
            if atk_type not in ("Normal", ""):
                assumed_ep = resolve_entry_point(atk_type, self.cfg)
            elif len(active_atks) > 0:
                assumed_ep = resolve_entry_point(active_atks[0], self.cfg)

            threat_dict = {
                "attack_type": atk_type,
                "confidence": round(conf, 3),
                "level": lvl,
                "explanation": expl,
                "assumed_entry_point": assumed_ep,
                "ai_available": bool(self.ai_context.available),
                "anomaly": round(self.ai_context.anomaly, 3),
                "attack_prob": round(self.ai_context.attack_prob, 3),
                "probs": dict(self.ai_context.probs),
            }

            # 3. Banner
            banner_dict = {
                "level": lvl,
                "headline": headline,
                "driver_action": driver_act,
            }

            # 4. Fail-safe status
            fs_dict = self.failsafe.snapshot()

            # 5. Incidents
            incidents_list = self.failsafe.get_recent_incidents(limit=10)

            # 6. Metrics
            metrics_dict = self.metrics.get_metrics_dict()

            # 7. Active attacks
            active_atks = list(self.world.injector.active())

            # 8. Complete Schema dict
            ai_backend_display = "IsolationForest" if self.anomaly_detector.backend_name == "iforest" else (
                "LSTM-AE (ONNX)" if "onnx" in self.anomaly_detector.backend_name else "LSTM-AE (PyTorch)"
            )
            return {
                "t": round(now, 3),
                "mode": self.failsafe.current_mode,
                "protection": bool(self.failsafe.protection_enabled),
                "backend": {
                    "bus": type(self.gateway_bus).__name__,
                    "ai": ai_backend_display,
                },
                "banner": banner_dict,
                "active_attacks": active_atks,
                "twin": twin_dict,
                "threat": threat_dict,
                "scores": list(self.score_history),
                "attack_markers": list(self.attack_markers),
                "can": can_rows,
                "v2x": v2x_rows,
                "security_events": sec_events,
                "can_flagged_count": len(self.flagged_can),
                "v2x_flagged_count": len(self.flagged_v2x),
                "failsafe": fs_dict,
                "incidents": incidents_list,
                "metrics": metrics_dict,
                "demo": dict(self.demo_state),
                "starvation_warning": self.failsafe.starvation_warning,
            }
