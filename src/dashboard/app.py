"""
CAVShield Web Dashboard Application (FastAPI).

Provides:
- Static file serving (HTML, CSS, Vanilla JS, and offline Plotly bundle).
- Real-time WebSocket streaming (/ws) emitting complete snapshots every 250 ms.
- REST API for attack injections, fail-safe protection toggles, scenario events, and resets.
- Server-side background thread for guided 8-step live demonstration scenario.
- Incident forensic log download (/api/incidents/export).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import plotly

from src import load_config
from src.engine import CAVShieldEngine

logger = logging.getLogger("cavshield.dashboard")

# Permitted attack names per CONTRACT & API Specification
ALLOWED_ATTACKS = {
    "flood",
    "fuzz",
    "replay",
    "spoof_brake",
    "spoof_accel",
    "spoof_speed",
    "spoof_rpm",
    "spoof_gear",
    "v2x_phantom",
    "v2x_fake_ebrake",
    "v2x_gps",
}


# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------

class AttackStartRequest(BaseModel):
    duration: Optional[float] = Field(None, description="Attack duration in seconds (null for indefinite)")


class ProtectionRequest(BaseModel):
    enabled: bool = Field(..., description="Enable or disable gateway enforcement")


class ObstacleRequest(BaseModel):
    distance: float = Field(95.0, description="Obstacle distance ahead in ego lane (meters)")
    lifetime: float = Field(14.0, description="Obstacle duration (seconds)")


# ---------------------------------------------------------------------------
# Guided Demo Background Manager
# ---------------------------------------------------------------------------

class GuidedDemoManager:
    """Manages server-side execution of the 8-step demonstration scenario."""

    def __init__(self, engine: CAVShieldEngine):
        self.engine = engine
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_demo, name="GuidedDemoThread", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive() and thread != threading.current_thread():
            thread.join(timeout=1.5)
        self.engine.stop_all_attacks()
        self.engine.set_protection(True)
        self.engine.set_demo_state(running=False, step=0, caption="Demo idle")

    def _sleep_interruptible(self, seconds: float) -> bool:
        """Sleep in small increments; returns True if interrupted to stop."""
        return self._stop_event.wait(timeout=seconds)

    def _run_demo(self) -> None:
        try:
            logger.info("Starting 8-step guided demonstration scenario.")
            self.engine.set_protection(True)

            # Step 1: Normal driving, anomaly score low (8 s)
            self.engine.set_demo_state(
                running=True,
                step=1,
                caption="Step 1/8: Normal driving baseline (8s). Low anomaly score, smooth telemetry.",
            )
            if self._sleep_interruptible(8.0):
                return

            # Step 2: Injecting spoofed brake command (start spoof_brake for 6 s)
            self.engine.set_demo_state(
                running=True,
                step=2,
                caption="Step 2/8: Adversary injecting spoofed brake command 0x200 (6s). Decel: 7.0 m/s².",
            )
            self.engine.inject("spoof_brake", duration=6.0)
            if self._sleep_interruptible(3.0):
                return

            # Step 3: CAVShield blocks it and enters SAFE MODE
            self.engine.set_demo_state(
                running=True,
                step=3,
                caption="Step 3/8: CAVShield blocks hostile commands! Mode escalates to SAFE MODE. Zero unintended braking.",
            )
            if self._sleep_interruptible(3.5):
                return

            # Step 4: Same attack with protection OFF: unintended braking (protection off for 3 s, then back ON)
            self.engine.set_demo_state(
                running=True,
                step=4,
                caption="Step 4/8: Protection OFF comparison. Hostile frames reach actuators -> UNINTENDED BRAKING!",
            )
            self.engine.set_protection(False)
            self.engine.inject("spoof_brake", duration=3.0)
            if self._sleep_interruptible(3.0):
                return
            self.engine.set_protection(True)
            self.engine.stop_attack("spoof_brake")

            # Step 5: Phantom V2X obstacle rejected
            self.engine.set_demo_state(
                running=True,
                step=5,
                caption="Step 5/8: Compromised vehicle V-9999 broadcasting phantom hazard. Rejected via radar/camera cross-check.",
            )
            self.engine.inject("v2x_phantom", duration=5.0)
            if self._sleep_interruptible(5.0):
                return
            self.engine.stop_attack("v2x_phantom")

            # Step 6: REAL obstacle: AEB works and is NOT blocked
            self.engine.set_demo_state(
                running=True,
                step=6,
                caption="Step 6/8: Legitimate obstacle in ego lane. Autonomous AEB engages and is NOT blocked!",
            )
            self.engine.spawn_real_obstacle(distance=90.0, lifetime=6.0)
            if self._sleep_interruptible(6.0):
                return

            # Step 7: Flooding is rate-limited
            self.engine.set_demo_state(
                running=True,
                step=7,
                caption="Step 7/8: High-frequency bus flooding attack at 700 Hz. Token bucket rate-limiter sheds flood stream.",
            )
            self.engine.inject("flood", duration=5.0)
            if self._sleep_interruptible(5.0):
                return
            self.engine.stop_attack("flood")

            # Step 8: Replay rejected; then reset to NORMAL
            self.engine.set_demo_state(
                running=True,
                step=8,
                caption="Step 8/8: Replay attack with stale counters rejected. Timed recovery returning to NORMAL mode.",
            )
            self.engine.inject("replay", duration=4.0)
            if self._sleep_interruptible(4.0):
                return
            self.engine.stop_attack("replay")

            # Reset back to pristine normal state
            self.engine.reset()
            self.engine.set_demo_state(
                running=False,
                step=8,
                caption="Guided demo completed successfully. All threats neutralized; vehicle in NORMAL mode.",
            )
            logger.info("Guided demonstration scenario completed.")

        except Exception as exc:
            logger.error("Error during guided demo: %s", exc, exc_info=True)
            self.engine.set_demo_state(running=False, step=0, caption=f"Demo error: {exc}")
        finally:
            self.engine.stop_all_attacks()
            self.engine.set_protection(True)


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

def create_app(engine: Optional[CAVShieldEngine] = None) -> FastAPI:
    """Create and configure the FastAPI dashboard application."""
    if engine is None:
        cfg = load_config()
        engine = CAVShieldEngine(cfg=cfg)

    # Base paths
    static_dir = Path(__file__).parent / "static"
    plotly_js_path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"

    demo_mgr = GuidedDemoManager(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        yield
        # Shutdown
        demo_mgr.stop()

    app = FastAPI(
        title="CAVShield SOC Console",
        description="Cyber-Physical Intrusion Detection & Fail-Safe Console for Connected Autonomous Vehicles",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Attach engine and demo manager to state for accessibility in testing
    app.state.engine = engine
    app.state.demo_mgr = demo_mgr

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:8000", "http://127.0.0.1:8000", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Structured JSON error handling (never leak uncaught stack traces)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": True, "detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception):
        logger.error("Unhandled endpoint exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": True, "detail": str(exc)},
        )

    # -----------------------------------------------------------------------
    # Static Files & Offline Plotly
    # -----------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def get_index():
        index_file = static_dir / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(index_file, media_type="text/html")

    @app.get("/static/plotly.min.js", include_in_schema=False)
    async def get_plotly_offline():
        if not plotly_js_path.exists():
            raise HTTPException(status_code=404, detail="plotly.min.js bundle not found")
        return FileResponse(plotly_js_path, media_type="application/javascript")

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # -----------------------------------------------------------------------
    # REST API Endpoints
    # -----------------------------------------------------------------------

    @app.get("/api/health")
    async def get_health():
        """Health check endpoint returning system status and uptime seconds."""
        uptime = round(time.perf_counter() - engine.uptime_start, 2)
        return {
            "status": "ok",
            "uptime": uptime,
            "running": getattr(engine, "_running", False),
        }

    @app.get("/api/snapshot")
    async def get_snapshot():
        """Return instantaneous full system snapshot matching CONTRACT item 11."""
        t0 = time.perf_counter()
        snap = engine.snapshot()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if dt_ms > 5.0:
            logger.warning("snapshot() build latency high: %.2f ms (threshold: 5.0 ms)", dt_ms)
        return snap

    @app.post("/api/attack/{name}/start")
    async def start_attack(name: str, body: Optional[AttackStartRequest] = None):
        """Initiate a specific cyber or V2X attack scenario."""
        if name not in ALLOWED_ATTACKS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown attack '{name}'. Permitted: {sorted(ALLOWED_ATTACKS)}",
            )
        duration = body.duration if body is not None else None
        ok = engine.inject(name, duration=duration)
        active = list(engine.world.injector.active())
        return {"status": "ok", "attack": name, "injected": ok, "active_attacks": active}

    @app.post("/api/attack/{name}/stop")
    async def stop_attack(name: str):
        """Halt a running attack scenario."""
        engine.stop_attack(name)
        active = list(engine.world.injector.active())
        return {"status": "ok", "stopped": name, "active_attacks": active}

    @app.post("/api/attack/stop_all")
    async def stop_all_attacks():
        """Halt all active attack scenarios."""
        engine.stop_all_attacks()
        return {"status": "ok", "active_attacks": []}

    @app.post("/api/scenario/real_obstacle")
    async def spawn_obstacle(body: Optional[ObstacleRequest] = None):
        """Spawn a legitimate obstacle in the ego vehicle's path."""
        dist = body.distance if body is not None else 95.0
        life = body.lifetime if body is not None else 14.0
        engine.spawn_real_obstacle(distance=dist, lifetime=life)
        return {"status": "ok", "spawned": True, "distance": dist, "lifetime": life}

    @app.post("/api/protection")
    async def set_protection(body: ProtectionRequest):
        """Toggle active fail-safe gateway protection mode."""
        engine.set_protection(body.enabled)
        return {"status": "ok", "protection": body.enabled}

    @app.post("/api/reset")
    async def reset_system():
        """Reset digital twin, detection states, buffers, and metric counters."""
        demo_mgr.stop()
        engine.reset()
        return {"status": "ok", "mode": "NORMAL"}

    @app.post("/api/demo/start")
    async def start_guided_demo():
        """Start the interactive 60-second guided demonstration timeline."""
        started = demo_mgr.start()
        return {"status": "ok", "demo_started": started}

    @app.post("/api/demo/stop")
    async def stop_guided_demo():
        """Halt the running guided demonstration."""
        demo_mgr.stop()
        return {"status": "ok", "demo_stopped": True}

    @app.get("/api/incidents/export")
    async def export_incidents():
        """Download the persistent forensic incident audit log (JSONL)."""
        log_dir = Path(getattr(engine.cfg.paths, "output_logs", "output/logs"))
        incidents_file = log_dir / "incidents.jsonl"

        if not incidents_file.exists():
            # If no incidents logged yet, create an empty file so download succeeds cleanly
            log_dir.mkdir(parents=True, exist_ok=True)
            incidents_file.touch()

        return FileResponse(
            path=incidents_file,
            media_type="application/x-ndjson",
            filename="cavshield_incidents.jsonl",
            headers={"Content-Disposition": "attachment; filename=cavshield_incidents.jsonl"},
        )

    # -----------------------------------------------------------------------
    # Real-Time WebSocket Streaming
    # -----------------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_stream(websocket: WebSocket):
        """Stream snapshots at 4 Hz (every 250 ms) to connected frontends."""
        await websocket.accept()
        logger.debug("WebSocket client connected to /ws")
        try:
            while True:
                t0 = time.perf_counter()
                snap = engine.snapshot()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if dt_ms > 5.0:
                    logger.warning("snapshot() build took %.2f ms (> 5ms)", dt_ms)

                await websocket.send_text(json.dumps(snap))
                await asyncio.sleep(0.25)

        except (WebSocketDisconnect, ConnectionResetError):
            logger.debug("WebSocket client disconnected quietly.")
        except Exception as exc:
            logger.debug("WebSocket stream closed: %s", exc)

    return app


def serve(
    engine: Optional[CAVShieldEngine] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> None:
    """Start uvicorn server serving the CAVShield Dashboard in-process."""
    import uvicorn

    if engine is None:
        cfg = load_config()
        engine = CAVShieldEngine(cfg=cfg)
        engine.start()

    dash_cfg = getattr(engine.cfg, "dashboard", {})
    srv_host = host or str(getattr(dash_cfg, "host", "127.0.0.1"))
    srv_port = port or int(getattr(dash_cfg, "port", 8000))

    app = create_app(engine)
    logger.info("Serving CAVShield Dashboard on http://%s:%d", srv_host, srv_port)
    uvicorn.run(app, host=srv_host, port=srv_port, log_level="info")

