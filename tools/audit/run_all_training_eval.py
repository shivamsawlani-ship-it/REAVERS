"""
Phase 2 Audit Script: Run all training and evaluation scripts,
inspect output models, verify runtime loading.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("e:/final product/cavshield")
PYTHON = ROOT / ".venv/Scripts/python.exe"

commands = [
    ("Train Classifier", [str(PYTHON), "-m", "src.models.train_classifier"]),
    ("Train Autoencoder", [str(PYTHON), "-m", "src.models.train_autoencoder"]),
    ("Export ONNX", [str(PYTHON), "-m", "src.models.export_onnx"]),
    ("ONNX Benchmark", [str(PYTHON), "-m", "src.edge.onnx_inference", "--benchmark"]),
    ("Evaluate Models", [str(PYTHON), "-m", "src.evaluation.evaluate_models"]),
    ("Generate Report", [str(PYTHON), "-m", "src.evaluation.generate_report"]),
]

results = []

for name, cmd in commands:
    print(f"=== Running: {name} ===")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
        )
        duration = round(time.time() - t0, 2)
        success = (proc.returncode == 0)
        output_snippet = "\n".join(proc.stdout.strip().splitlines()[-10:])
        results.append({
            "name": name,
            "cmd": " ".join(cmd[1:]),
            "success": success,
            "returncode": proc.returncode,
            "duration": duration,
            "snippet": output_snippet
        })
        print(f"Result: {'SUCCESS' if success else 'FAILED'} (code {proc.returncode}) in {duration}s")
        print("Last output lines:\n", output_snippet)
    except Exception as exc:
        duration = round(time.time() - t0, 2)
        results.append({
            "name": name,
            "cmd": " ".join(cmd[1:]),
            "success": False,
            "returncode": -1,
            "duration": duration,
            "snippet": str(exc)
        })
        print(f"Result: EXCEPTION in {duration}s: {exc}")
    print()

print("=== CHECKING OUTPUT MODELS & SIZES ===")
models_dir = ROOT / "output/models"
for f in models_dir.glob("*"):
    if f.is_file():
        size_kb = round(f.stat().st_size / 1024, 2)
        print(f"Model File: {f.name} -> {size_kb} KB ({f.stat().st_size} bytes)")

print("\n=== VERIFYING RUNTIME LOADING IN ENGINE ===")
sys.path.insert(0, str(ROOT))
from src.engine import CAVShieldEngine
engine = CAVShieldEngine()
print("Anomaly Detector Backend:", engine.anomaly_detector.backend)
print("Anomaly Detector Available:", engine.anomaly_detector.available)
print("Attack Classifier Backend:", getattr(engine.attack_classifier, "backend", "sklearn/lightgbm"))
print("Attack Classifier Available:", engine.attack_classifier.available)
snap = engine.snapshot()
print("Backend snap:", snap["backend"])

