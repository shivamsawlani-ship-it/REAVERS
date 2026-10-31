"""
Integration tests for CAVShield Part 9:
- FastAPI Dashboard Application endpoints
- Full SNAPSHOT SCHEMA verification via REST
- Attack injection start/stop lifecycle
- Fail-safe protection toggle
- Obstacle scenario triggering
- Incident log export
- Real-time WebSocket streaming (/ws)
- Offline Plotly delivery & static file serving
- XSS defense and safe input handling
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

from src import load_config
from src.dashboard.app import ALLOWED_ATTACKS, create_app
from src.engine import CAVShieldEngine


@pytest.fixture(scope="module")
def shared_engine():
    """Create and start a CAVShieldEngine instance shared across dashboard tests."""
    cfg = load_config()
    engine = CAVShieldEngine(cfg=cfg)
    engine.start()
    # Allow background threads to populate initial buffers
    time.sleep(0.3)
    yield engine
    engine.stop()


@pytest.fixture(scope="module")
def client(shared_engine):
    """FastAPI TestClient using the shared running engine."""
    app = create_app(engine=shared_engine)
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Verify GET /api/health returns ok status and uptime."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "uptime" in data
    assert isinstance(data["uptime"], (int, float))
    assert data["uptime"] >= 0
    assert data["running"] is True


def test_snapshot_schema_via_api(client: TestClient) -> None:
    """Verify GET /api/snapshot returns all 16 CONTRACT Item 11 keys with sub-structures."""
    t0 = time.perf_counter()
    resp = client.get("/api/snapshot")
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert resp.status_code == 200
    # API snapshot retrieval must be fast (< 200 ms total HTTP roundtrip in test)
    assert latency_ms < 200.0

    snap = resp.json()

    # 16 root keys from CONTRACT Item 11
    expected_root_keys = [
        "t", "mode", "protection", "backend", "banner", "active_attacks",
        "twin", "threat", "scores", "attack_markers", "can", "v2x",
        "failsafe", "incidents", "metrics", "demo",
    ]
    for key in expected_root_keys:
        assert key in snap, f"Missing root key in snapshot: {key}"

    # Verify key types
    assert isinstance(snap["t"], (int, float))
    assert snap["mode"] in {"NORMAL", "ALERT", "DEGRADED", "SAFE MODE"}
    assert isinstance(snap["protection"], bool)
    assert isinstance(snap["backend"], dict)
    assert "bus" in snap["backend"]
    assert "ai" in snap["backend"]
    assert isinstance(snap["banner"], dict)
    assert snap["banner"]["level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    assert isinstance(snap["twin"], dict)
    for field in ("speed_kmh", "rpm", "pedal", "applied_decel", "steering_deg", "gear", "perceived"):
        assert field in snap["twin"]

    assert isinstance(snap["threat"], dict)
    for field in ("attack_type", "confidence", "level", "anomaly", "attack_prob", "probs"):
        assert field in snap["threat"]

    assert isinstance(snap["scores"], list)
    assert isinstance(snap["can"], list)
    assert isinstance(snap["v2x"], list)
    assert isinstance(snap["failsafe"], dict)
    assert isinstance(snap["metrics"], dict)
    assert isinstance(snap["demo"], dict)
    assert "running" in snap["demo"]
    assert "step" in snap["demo"]


def test_attack_lifecycle_endpoints(client: TestClient, shared_engine: CAVShieldEngine) -> None:
    """Verify starting, stopping, and mass-stopping attacks via REST API."""
    # 1. Start a valid attack
    resp = client.post("/api/attack/spoof_brake/start", json={"duration": 5.0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["attack"] == "spoof_brake"
    assert "spoof_brake" in data["active_attacks"]

    # Verify reflected in snapshot
    snap_resp = client.get("/api/snapshot")
    assert "spoof_brake" in snap_resp.json()["active_attacks"]

    # 2. Stop the attack
    stop_resp = client.post("/api/attack/spoof_brake/stop")
    assert stop_resp.status_code == 200
    stop_data = stop_resp.json()
    assert stop_data["status"] == "ok"
    assert stop_data["stopped"] == "spoof_brake"
    assert "spoof_brake" not in stop_data["active_attacks"]

    # 3. Start multiple attacks and stop all
    client.post("/api/attack/flood/start")
    client.post("/api/attack/fuzz/start")
    stop_all_resp = client.post("/api/attack/stop_all")
    assert stop_all_resp.status_code == 200
    assert stop_all_resp.json()["active_attacks"] == []

    # 4. Attempt an unknown attack (must return 404 structured error)
    err_resp = client.post("/api/attack/invalid_hack_99/start")
    assert err_resp.status_code == 404
    err_data = err_resp.json()
    assert err_data.get("error") is True
    assert "Unknown attack" in err_data.get("detail", "")


def test_protection_toggle_endpoint(client: TestClient) -> None:
    """Verify toggling active gateway protection on and off."""
    # Disable protection
    resp_off = client.post("/api/protection", json={"enabled": False})
    assert resp_off.status_code == 200
    assert resp_off.json()["protection"] is False

    snap_off = client.get("/api/snapshot").json()
    assert snap_off["protection"] is False

    # Re-enable protection
    resp_on = client.post("/api/protection", json={"enabled": True})
    assert resp_on.status_code == 200
    assert resp_on.json()["protection"] is True

    snap_on = client.get("/api/snapshot").json()
    assert snap_on["protection"] is True


def test_scenario_obstacle_endpoint(client: TestClient) -> None:
    """Verify spawning a legitimate road obstacle."""
    resp = client.post("/api/scenario/real_obstacle", json={"distance": 92.5, "lifetime": 8.0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["spawned"] is True
    assert data["distance"] == 92.5
    assert data["lifetime"] == 8.0


def test_system_reset_endpoint(client: TestClient) -> None:
    """Verify /api/reset resets system mode to NORMAL."""
    resp = client.post("/api/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mode"] == "NORMAL"


def test_guided_demo_lifecycle(client: TestClient) -> None:
    """Verify starting and stopping guided demo background thread."""
    start_resp = client.post("/api/demo/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "ok"

    # Allow thread to start
    time.sleep(0.2)

    stop_resp = client.post("/api/demo/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["demo_stopped"] is True


def test_incidents_export_endpoint(client: TestClient) -> None:
    """Verify /api/incidents/export returns ndjson file attachment."""
    resp = client.get("/api/incidents/export")
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")
    assert "cavshield_incidents.jsonl" in resp.headers.get("content-disposition", "")


def test_static_files_and_offline_plotly(client: TestClient) -> None:
    """Verify offline static assets (HTML, CSS, JS, and Plotly bundle)."""
    # 1. Root index.html
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "text/html" in root_resp.headers.get("content-type", "")
    assert "CAVShield" in root_resp.text
    assert "Security Console" in root_resp.text

    # 2. Offline Plotly JS bundle from package
    plotly_resp = client.get("/static/plotly.min.js")
    assert plotly_resp.status_code == 200
    assert "javascript" in plotly_resp.headers.get("content-type", "")
    assert len(plotly_resp.content) > 1_000_000  # plotly.min.js is ~4.8MB

    # 3. CSS stylesheet
    css_resp = client.get("/static/app.css")
    assert css_resp.status_code == 200
    assert "css" in css_resp.headers.get("content-type", "")
    assert "top-header" in css_resp.text

    # 4. JS Application
    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "javascript" in js_resp.headers.get("content-type", "")
    assert "applySnapshot" in js_resp.text


def test_websocket_streaming(client: TestClient) -> None:
    """Verify real-time WebSocket /ws delivers snapshot updates."""
    with client.websocket_connect("/ws") as ws:
        # Receive first frame
        msg1_raw = ws.receive_text()
        msg1 = json.loads(msg1_raw)
        assert "mode" in msg1
        assert "twin" in msg1

        # Receive second frame
        msg2_raw = ws.receive_text()
        msg2 = json.loads(msg2_raw)
        assert "mode" in msg2
        assert "t" in msg2


def test_xss_safety_and_dom_sanitization() -> None:
    """Verify that frontend script does not use dangerous unescaped HTML interpolation."""
    js_path = Path("src/dashboard/static/app.js")
    if not js_path.exists():
        js_path = Path("cavshield/src/dashboard/static/app.js")
    assert js_path.exists(), "app.js file must exist"

    js_code = js_path.read_text(encoding="utf-8")

    # In app.js, dynamic data from CAN/V2X/threat must be inserted safely (e.g. textContent)
    # Check that textContent is used extensively for dynamic content injection
    assert ".textContent =" in js_code
    # Ensure dangerous eval is never used
    assert "eval(" not in js_code
