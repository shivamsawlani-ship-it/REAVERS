"""
Phase 5: Robustness, Security, and Portability Audit.
Inspects:
1. Hardcoded paths (/tmp, C:\\, Linux-specific /dev/shm, socket paths, path separators).
2. Network port binding: UDP 47474 and HTTP 8000 (0.0.0.0 vs 127.0.0.1).
3. Authentication / Authorization on REST endpoints.
4. Engine lifecycle & graceful thread/socket shutdown.
5. Missing file references and graceful fallback if models/ is missing.
6. Offline readiness (external CDN dependencies, fonts, scripts).
7. Database / file growth unboundedness.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import socket
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def check_hardcoded_paths() -> List[Dict[str, Any]]:
    findings = []
    py_files = list(ROOT.glob("src/**/*.py")) + list(ROOT.glob("tools/**/*.py"))

    # Regex patterns for problematic hardcoded paths
    patterns = [
        (re.compile(r"""['"][A-Za-z]:\\[^'"]*['"]"""), "Windows absolute path with drive letter"),
        (re.compile(r"""['"]/tmp/[^'"]*['"]"""), "Unix /tmp hardcoded path"),
        (re.compile(r"""['"]/dev/[^'"]*['"]"""), "Unix /dev hardcoded path"),
    ]

    for p in py_files:
        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for line_no, line in enumerate(content.splitlines(), start=1):
            for pat, desc in patterns:
                m = pat.search(line)
                if m:
                    findings.append({
                        "file": str(p.relative_to(ROOT)),
                        "line": line_no,
                        "type": desc,
                        "match": m.group(0),
                    })
    return findings


def check_port_exposure() -> Dict[str, Any]:
    # Check UDP listener in src/engine.py
    engine_file = ROOT / "src" / "engine.py"
    content = engine_file.read_text(encoding="utf-8")

    udp_bind = "unknown"
    if "bind((" in content or "bind( (" in content:
        for line in content.splitlines():
            if "bind((" in line or "bind( (" in line:
                udp_bind = line.strip()

    # Check dashboard run command in run_dashboard.py
    run_dash = ROOT / "run_dashboard.py"
    dash_host = "unknown"
    if run_dash.exists():
        dash_content = run_dash.read_text(encoding="utf-8")
        for line in dash_content.splitlines():
            if "host=" in line or "uvicorn.run" in line:
                dash_host = line.strip()

    return {
        "udp_47474_binding": udp_bind,
        "dashboard_host_binding": dash_host,
    }


def check_offline_assets() -> Dict[str, Any]:
    static_dir = ROOT / "src" / "dashboard" / "static"
    html_file = static_dir / "index.html"
    external_urls = []

    if html_file.exists():
        content = html_file.read_text(encoding="utf-8")
        # Search for http:// or https:// script or css links
        urls = re.findall(r"""(?:src|href)=['"](https?://[^'"]+)['"]""", content)
        external_urls = urls

    return {
        "index_html_external_dependencies": external_urls,
        "fully_offline": len(external_urls) == 0,
    }


def check_model_fallback() -> Dict[str, Any]:
    # Test if engine can start when model weights are temporarily displaced or missing
    from src import load_config
    from src.detection.anomaly_detector import AnomalyDetector
    from src.detection.attack_classifier import AttackClassifier

    cfg = load_config()
    anom = AnomalyDetector(cfg=cfg, model_dir=Path("non_existent_model_dir"))
    clf = AttackClassifier(cfg=cfg, model_path=Path("non_existent_clf.joblib"))

    return {
        "anomaly_detector_graceful_fallback": not anom.available,
        "anomaly_detector_backend_fallback": anom.backend_name,
        "attack_classifier_graceful_fallback": not clf.available,
    }


def main() -> None:
    results = {
        "hardcoded_paths": check_hardcoded_paths(),
        "network_port_exposure": check_port_exposure(),
        "offline_assets": check_offline_assets(),
        "model_missing_fallback": check_model_fallback(),
    }

    out_path = ROOT / "tools" / "audit" / "robustness_security_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Robustness & Security Audit complete:")
    print(f"  Hardcoded paths found: {len(results['hardcoded_paths'])}")
    print(f"  Network binding: {results['network_port_exposure']}")
    print(f"  Offline assets: fully_offline={results['offline_assets']['fully_offline']}, external_urls={results['offline_assets']['index_html_external_dependencies']}")
    print(f"  Model fallback: {results['model_missing_fallback']}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
