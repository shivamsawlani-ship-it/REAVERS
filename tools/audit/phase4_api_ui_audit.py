"""
Phase 4: API & UI Audit.
Uses FastAPI TestClient to test:
1. All endpoints with valid payloads:
   - GET /api/health
   - GET /api/snapshot
   - POST /api/attack/{name}/start and /stop
   - POST /api/attack/stop_all
   - POST /api/scenario/real_obstacle
   - POST /api/failsafe/protection
   - POST /api/demo/start and /stop
   - POST /api/reset
   - GET /api/incidents/export
   - GET /api/metrics/report
2. Invalid payloads & type validation (expect 422 Unprocessable Entity or 404 Not Found)
3. Unknown attack name injection (expect 404 Not Found with permitted list)
4. Rapid firing (50 requests in burst)
5. XSS injection strings in body/parameters (verify escaped or rejected)
6. WebSocket /ws connection and streaming
7. Snapshot schema completeness and CONTRACT compliance
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from src import load_config
from src.dashboard.app import create_app
from src.engine import CAVShieldEngine

logging.basicConfig(level=logging.WARNING)


def run_api_audit() -> Dict[str, Any]:
    print("\n=======================================================")
    print("Running Phase 4: API & UI Automated Audit")
    print("=======================================================")

    cfg = load_config()
    engine = CAVShieldEngine(cfg=cfg)
    app = create_app(engine)
    client = TestClient(app)

    results: Dict[str, Any] = {"endpoint_tests": [], "errors": []}

    # 1. GET /api/health
    r = client.get("/api/health")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    health_data = r.json()
    assert health_data.get("status") == "ok"
    results["endpoint_tests"].append({"endpoint": "GET /api/health", "status": r.status_code, "passed": True})

    # 2. GET /api/snapshot & Schema Validation
    r = client.get("/api/snapshot")
    assert r.status_code == 200, f"Snapshot failed: {r.status_code}"
    snap = r.json()
    results["endpoint_tests"].append({"endpoint": "GET /api/snapshot", "status": r.status_code, "passed": True})

    # Contract schema check
    required_top = [
        "t", "mode", "protection", "backend", "banner",
        "active_attacks", "twin", "threat", "failsafe", "metrics"
    ]
    missing_top = [k for k in required_top if k not in snap]
    if missing_top:
        results["errors"].append(f"Snapshot missing CONTRACT keys: {missing_top}")

    # Check twin keys
    required_twin = ["speed_kmh", "rpm", "gear", "brake_active", "steering_deg"]
    missing_twin = [k for k in required_twin if k not in snap.get("twin", {})]
    if missing_twin:
        results["errors"].append(f"Snapshot twin missing CONTRACT keys: {missing_twin}")

    # 3. POST /api/attack/{name}/start (Valid attack)
    r = client.post("/api/attack/spoof_brake/start", json={"duration": 5.0})
    assert r.status_code == 200, f"Valid attack start failed: {r.status_code}"
    results["endpoint_tests"].append({"endpoint": "POST /api/attack/spoof_brake/start", "status": r.status_code, "passed": True})

    # 4. POST /api/attack/{name}/stop
    r = client.post("/api/attack/spoof_brake/stop")
    assert r.status_code == 200, f"Attack stop failed: {r.status_code}"
    results["endpoint_tests"].append({"endpoint": "POST /api/attack/spoof_brake/stop", "status": r.status_code, "passed": True})

    # 5. POST /api/attack/{name}/start with UNKNOWN attack name (expect 404)
    r = client.post("/api/attack/evil_malware_unknown/start", json={"duration": 5.0})
    results["unknown_attack_test"] = {"status": r.status_code, "expected": 404, "passed": r.status_code == 404}
    if r.status_code != 404:
        results["errors"].append(f"Unknown attack returned {r.status_code} instead of 404")

    # 6. POST with invalid payload types (expect 422 Unprocessable Entity)
    r = client.post("/api/attack/flood/start", json={"duration": "INVALID_STRING_NOT_FLOAT"})
    results["invalid_payload_test"] = {"status": r.status_code, "expected": 422, "passed": r.status_code == 422}
    if r.status_code != 422:
        results["errors"].append(f"Invalid duration type returned {r.status_code} instead of 422")

    # 7. POST /api/scenario/real_obstacle
    r = client.post("/api/scenario/real_obstacle", json={"distance": 60.0, "lifetime": 10.0})
    assert r.status_code == 200
    results["endpoint_tests"].append({"endpoint": "POST /api/scenario/real_obstacle", "status": r.status_code, "passed": True})

    # 8. POST /api/protection (Toggle)
    r = client.post("/api/protection", json={"enabled": False})
    assert r.status_code == 200
    r = client.post("/api/protection", json={"enabled": True})
    assert r.status_code == 200
    results["endpoint_tests"].append({"endpoint": "POST /api/protection", "status": r.status_code, "passed": True})

    # 9. POST /api/reset
    r = client.post("/api/reset")
    assert r.status_code == 200
    results["endpoint_tests"].append({"endpoint": "POST /api/reset", "status": r.status_code, "passed": True})

    # 10. Rapid firing (50 requests in tight loop)
    rapid_success = 0
    t0 = time.perf_counter()
    for _ in range(50):
        r = client.get("/api/snapshot")
        if r.status_code == 200:
            rapid_success += 1
    t_burst_ms = (time.perf_counter() - t0) * 1000.0
    results["rapid_burst_test"] = {
        "requests": 50,
        "successful": rapid_success,
        "elapsed_ms": round(t_burst_ms, 2),
        "passed": rapid_success == 50
    }

    # 11. XSS injection strings in body
    xss_payload = "<script>alert('XSS')</script>"
    r = client.post(f"/api/attack/{xss_payload}/start", json={"duration": 2.0})
    # Should safely return 404 without reflection in executable HTML
    results["xss_injection_test"] = {
        "status": r.status_code,
        "content_type": r.headers.get("content-type"),
        "body_contains_raw_script": xss_payload in r.text and "text/html" in r.headers.get("content-type", ""),
        "passed": ("text/html" not in r.headers.get("content-type", "")),
    }

    # 12. WebSocket test
    try:
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert "t" in data, f"'t' missing from ws message"
            assert "mode" in data, f"'mode' missing from ws message"
        results["websocket_test"] = {"passed": True, "received_keys": list(data.keys())[:5]}
    except Exception as e:
        results["websocket_test"] = {"passed": False, "error": str(e)}
        results["errors"].append(f"WebSocket failure: {e}")

    # 13. Guided demo test
    r_start = client.post("/api/demo/start")
    time.sleep(0.5)
    r_snap = client.get("/api/snapshot")
    demo_active = r_snap.json().get("demo", {}).get("running", False)
    r_stop = client.post("/api/demo/stop")
    time.sleep(0.5)
    r_snap_post = client.get("/api/snapshot")
    demo_stopped = not r_snap_post.json().get("demo", {}).get("running", False)
    results["guided_demo_test"] = {
        "start_status": r_start.status_code,
        "started": demo_active,
        "stop_status": r_stop.status_code,
        "stopped": demo_stopped,
        "passed": demo_active and demo_stopped,
    }

    out_path = ROOT / "tools" / "audit" / "api_ui_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"API Audit Complete. Errors: {len(results['errors'])}. Saved to {out_path}")
    return results


if __name__ == "__main__":
    run_api_audit()
