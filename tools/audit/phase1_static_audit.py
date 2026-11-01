"""
Phase 1 Static Review & Contract Audit Script
Independent Automotive Security Audit
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path("e:/final product/cavshield")
sys.path.insert(0, str(ROOT))

def check_repo_layout():
    expected_files = [
        "README.md",
        "requirements.txt",
        "run_dashboard.py",
        "run_simulation.py",
        "config/config.yaml",
        "reference/cavshield_ui_reference.html",
        "tests/test_pipeline.py",
        "tests/test_dashboard_api.py",
        "src/__init__.py",
        "src/data_loader.py",
        "src/preprocessing.py",
        "src/feature_engineering.py",
        "src/windowing.py",
        "src/can_simulator.py",
        "src/v2x_simulator.py",
        "src/attack_injector.py",
        "src/vehicle_digital_twin.py",
        "src/engine.py",
        "src/models/__init__.py",
        "src/models/lstm_autoencoder.py",
        "src/models/attack_classifier.py",
        "src/models/train_autoencoder.py",
        "src/models/train_classifier.py",
        "src/models/export_onnx.py",
        "src/detection/__init__.py",
        "src/detection/anomaly_detector.py",
        "src/detection/attack_classifier.py",
        "src/detection/rule_engine.py",
        "src/detection/physics_engine.py",
        "src/detection/v2x_validator.py",
        "src/detection/decision_engine.py",
        "src/failsafe/__init__.py",
        "src/failsafe/failsafe_manager.py",
        "src/dashboard/__init__.py",
        "src/dashboard/app.py",
        "src/dashboard/static/index.html",
        "src/dashboard/static/app.css",
        "src/dashboard/static/app.js",
        "src/evaluation/__init__.py",
        "src/evaluation/evaluate_models.py",
        "src/evaluation/generate_report.py",
        "src/edge/__init__.py",
        "src/edge/onnx_inference.py",
    ]

    expected_dirs = [
        "data/raw", "data/processed", "data/sample",
        "output/models", "output/metrics", "output/logs", "output/figures"
    ]

    missing_files = []
    for f in expected_files:
        if not (ROOT / f).exists():
            missing_files.append(f)

    missing_dirs = []
    for d in expected_dirs:
        if not (ROOT / d).exists():
            missing_dirs.append(d)

    # Check for init files
    missing_inits = []
    for pkg in ["src", "src/models", "src/detection", "src/failsafe", "src/dashboard", "src/evaluation", "src/edge"]:
        if not (ROOT / pkg / "__init__.py").exists():
            missing_inits.append(pkg + "/__init__.py")

    return missing_files, missing_dirs, missing_inits

def check_feature_order():
    fe_file = ROOT / "src/feature_engineering.py"
    content = fe_file.read_text(encoding="utf-8")
    expected_order = [
        "can_id_norm", "dlc", "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7",
        "dt_same_id", "dt_global", "freq_id", "payload_change_rate", "id_entropy",
        "repeated_payload", "safety_critical"
    ]
    return expected_order, content

def search_ground_truth_leakage():
    findings = []
    # Files under detection/
    detection_dir = ROOT / "src/detection"
    for py_file in detection_dir.glob("*.py"):
        lines = py_file.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, 1):
            if re.search(r'\b(frame|msg|v2x_msg)\.(label|subtype)\b', line):
                # Check if it's just a comment or actual code
                stripped = line.strip()
                if not stripped.startswith("#"):
                    findings.append((py_file.name, idx, line.strip()))
    return findings

def search_swallowed_exceptions():
    findings = []
    for py_file in (ROOT / "src").rglob("*.py"):
        lines = py_file.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, 1):
            # bare except or except ...: pass
            if re.match(r'^\s*except\s*:', line):
                findings.append((str(py_file.relative_to(ROOT)), idx, line.strip(), "bare except"))
            elif re.search(r'except\s+.*:\s*pass', line):
                findings.append((str(py_file.relative_to(ROOT)), idx, line.strip(), "swallowed exception with pass"))
            elif re.search(r'except\s+.*:', line):
                if idx < len(lines):
                    next_line = lines[idx].strip()
                    if next_line == "pass":
                        findings.append((str(py_file.relative_to(ROOT)), idx, f"{line.strip()} -> {next_line}", "except block with only pass"))
    return findings

def search_unbounded_containers():
    findings = []
    for py_file in (ROOT / "src").rglob("*.py"):
        lines = py_file.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, 1):
            if re.search(r'queue\.Queue\(\s*\)', line):
                findings.append((str(py_file.relative_to(ROOT)), idx, line.strip(), "Unbounded queue.Queue()"))
            if re.search(r'collections\.deque\(\s*\)', line):
                findings.append((str(py_file.relative_to(ROOT)), idx, line.strip(), "Unbounded collections.deque() without maxlen"))
    return findings

def search_inner_html_xss():
    findings = []
    js_file = ROOT / "src/dashboard/static/app.js"
    if js_file.exists():
        lines = js_file.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, 1):
            if ".innerHTML" in line:
                findings.append((str(js_file.relative_to(ROOT)), idx, line.strip()))
    return findings

def search_todos_and_placeholders():
    findings = []
    for py_file in (ROOT / "src").rglob("*.py"):
        lines = py_file.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, 1):
            if re.search(r'\b(TODO|FIXME|XXX|HACK|placeholder)\b', line, re.IGNORECASE):
                findings.append((str(py_file.relative_to(ROOT)), idx, line.strip()))
    return findings

def check_dbc():
    from src.can_simulator import DBC
    expected_dbc = {
        0x100: ("ENGINE_RPM", "ECU_ENGINE", 0.020, False, 0.6),
        0x101: ("VEHICLE_SPEED", "ECU_ENGINE", 0.020, True, 0.8),
        0x102: ("BRAKE_STATUS", "ECU_BRAKE", 0.020, True, 0.9),
        0x103: ("STEERING_ANGLE", "ECU_STEERING", 0.020, True, 0.8),
        0x104: ("GEAR_POSITION", "ECU_GEAR", 0.100, True, 0.8),
        0x105: ("WHEEL_SPEED", "SENSOR_WHEEL", 0.020, False, 0.6),
        0x106: ("IMU_DATA", "SENSOR_IMU", 0.010, False, 0.6),
        0x200: ("BRAKE_COMMAND", "ECU_ADAS", 0.020, True, 1.0),
        0x201: ("ACCEL_COMMAND", "ECU_ADAS", 0.020, True, 1.0),
        0x202: ("ADAS_COMMAND", "ECU_ADAS", 0.050, True, 1.0),
    }
    mismatches = []
    for arb_id, exp in expected_dbc.items():
        if arb_id not in DBC:
            mismatches.append(f"Missing ID 0x{arb_id:03X} in DBC")
            continue
        actual = DBC[arb_id]
        if actual.name != exp[0]:
            mismatches.append(f"0x{arb_id:03X} name mismatch: expected {exp[0]}, got {actual.name}")
        if actual.sender != exp[1]:
            mismatches.append(f"0x{arb_id:03X} sender mismatch: expected {exp[1]}, got {actual.sender}")
        if abs(actual.period_s - exp[2]) > 0.001:
            mismatches.append(f"0x{arb_id:03X} period mismatch: expected {exp[2]}, got {actual.period_s}")
        if actual.safety_critical != exp[3]:
            mismatches.append(f"0x{arb_id:03X} safety_critical mismatch: expected {exp[3]}, got {actual.safety_critical}")
        if abs(actual.criticality - exp[4]) > 0.001:
            mismatches.append(f"0x{arb_id:03X} criticality mismatch: expected {exp[4]}, got {actual.criticality}")
    return mismatches

def check_snapshot_schema():
    from src.engine import CAVShieldEngine
    eng = CAVShieldEngine()
    snap = eng.snapshot()
    expected_top_keys = [
        "t", "mode", "protection", "backend", "banner", "active_attacks",
        "twin", "threat", "scores", "attack_markers", "can", "v2x",
        "failsafe", "incidents", "metrics", "demo"
    ]
    missing_keys = [k for k in expected_top_keys if k not in snap]

    twin_expected = ["speed_kmh", "rpm", "pedal", "brake_active", "decel", "applied_decel", "steering_deg", "gear", "aeb", "unsafe", "unsafe_seconds", "perceived"]
    missing_twin = [k for k in twin_expected if k not in snap.get("twin", {})]

    threat_expected = ["attack_type", "confidence", "level", "explanation", "ai_available", "anomaly", "attack_prob", "probs"]
    missing_threat = [k for k in threat_expected if k not in snap.get("threat", {})]

    banner_expected = ["level", "headline", "driver_action"]
    missing_banner = [k for k in banner_expected if k not in snap.get("banner", {})]

    failsafe_expected = ["rejected_commands", "rate_limited", "quarantined", "rejected_v2x", "mode_history", "actions", "incident_id"]
    missing_failsafe = [k for k in failsafe_expected if k not in snap.get("failsafe", {})]

    metrics_expected = ["accuracy", "precision", "recall", "f1", "fpr", "detection_latency_ms", "failsafe_latency_ms", "inference_ms", "simulator_only", "counts"]
    missing_metrics = [k for k in metrics_expected if k not in snap.get("metrics", {})]

    return {
        "missing_top": missing_keys,
        "missing_twin": missing_twin,
        "missing_threat": missing_threat,
        "missing_banner": missing_banner,
        "missing_failsafe": missing_failsafe,
        "missing_metrics": missing_metrics,
        "actual_snapshot": snap
    }

if __name__ == "__main__":
    print("=== REPO LAYOUT CHECK ===")
    missing_f, missing_d, missing_i = check_repo_layout()
    print("Missing files:", missing_f)
    print("Missing dirs:", missing_d)
    print("Missing inits:", missing_i)

    print("\n=== DBC CHECK ===")
    dbc_mismatches = check_dbc()
    print("DBC Mismatches:", dbc_mismatches if dbc_mismatches else "None (DBC matches CONTRACT 100%)")

    print("\n=== SNAPSHOT SCHEMA CHECK ===")
    snap_res = check_snapshot_schema()
    print("Missing top keys:", snap_res["missing_top"])
    print("Missing twin keys:", snap_res["missing_twin"])
    print("Missing threat keys:", snap_res["missing_threat"])
    print("Missing banner keys:", snap_res["missing_banner"])
    print("Missing failsafe keys:", snap_res["missing_failsafe"])
    print("Missing metrics keys:", snap_res["missing_metrics"])

    print("\n=== GROUND TRUTH LEAKAGE CHECK ===")
    leakages = search_ground_truth_leakage()
    if leakages:
        for f, l, c in leakages:
            print(f"LEAKAGE in {f}:{l} -> {c}")
    else:
        print("NO ground truth leakage detected in src/detection/")

    print("\n=== SWALLOWED EXCEPTIONS (SAMPLE) ===")
    swallowed = search_swallowed_exceptions()
    print(f"Total swallowed exceptions/except pass found: {len(swallowed)}")
    for item in swallowed[:10]:
        print(f"{item[0]}:{item[1]} [{item[3]}] -> {item[2]}")

    print("\n=== UNBOUNDED QUEUES / DEQUES ===")
    unbounded = search_unbounded_containers()
    print(f"Total unbounded queues/deques: {len(unbounded)}")
    for item in unbounded:
        print(f"{item[0]}:{item[1]} -> {item[2]}")

    print("\n=== INNERHTML XSS CHECK ===")
    inner_htmls = search_inner_html_xss()
    print(f"Total innerHTML usages in app.js: {len(inner_htmls)}")
    for item in inner_htmls:
        print(f"{item[0]}:{item[1]} -> {item[2]}")

    print("\n=== TODOS / PLACEHOLDERS ===")
    todos = search_todos_and_placeholders()
    print(f"Total TODOs/placeholders found: {len(todos)}")
    for item in todos:
        print(f"{item[0]}:{item[1]} -> {item[2]}")
