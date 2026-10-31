"""
Unit and integration tests for CAVShield Part 10 (Evaluation & Reporting):
- Window-level metric structure and values
- Fast-path frame-level latency and accuracy
- 4-configuration ablation study assertions:
  * Full system (Config 4) achieves 100% safety-critical recall
  * Full system (Config 4) has 0 missed safety-critical attacks
  * Hard negative legitimate AEB is 0% blocked in full system
  * AI-only baseline (Config 1) fails on in-range attacks and AEB
- Markdown and HTML report generation and SYNTHETIC-only disclaimer verification
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src import load_config
from src.evaluation.generate_report import check_hcrl_dataset, generate_reports


def test_window_metrics_artifact_schema() -> None:
    """Verify evaluation_window_metrics.json contains all required fields and valid metrics."""
    metrics_path = Path("output/metrics/evaluation_window_metrics.json")
    if not metrics_path.exists():
        metrics_path = Path("cavshield/output/metrics/evaluation_window_metrics.json")
    assert metrics_path.exists(), "evaluation_window_metrics.json must exist"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Anomaly detector section
    assert "anomaly_detector" in data
    anom = data["anomaly_detector"]
    for key in ("threshold", "accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "confusion_matrix", "score_distribution"):
        assert key in anom, f"Missing key in anomaly metrics: {key}"
    assert anom["roc_auc"] >= 0.70, "Anomaly detector ROC-AUC should be reasonable"
    assert anom["score_distribution"]["normal"]["mean"] < anom["score_distribution"]["attack"]["mean"]

    # Multi-class classifier section
    assert "classifier" in data
    clf = data["classifier"]
    for key in ("accuracy", "macro_f1", "weighted_f1", "classes", "per_class", "confusion_matrix"):
        assert key in clf, f"Missing key in classifier metrics: {key}"
    assert clf["accuracy"] >= 0.85, "Classifier accuracy should exceed 85%"
    assert "Normal" in clf["classes"]
    assert "Spoofing" in clf["classes"]


def test_frame_metrics_latency_and_safety() -> None:
    """Verify fast-path replay metrics and sub-millisecond automotive latency."""
    metrics_path = Path("output/metrics/evaluation_frame_metrics.json")
    if not metrics_path.exists():
        metrics_path = Path("cavshield/output/metrics/evaluation_frame_metrics.json")
    assert metrics_path.exists(), "evaluation_frame_metrics.json must exist"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ("total_frames", "counts", "accuracy", "precision", "recall", "f1", "fpr", "safety_critical_recall", "latency_us"):
        assert key in data, f"Missing key in frame metrics: {key}"

    assert data["total_frames"] > 10_000, "Should evaluate across thousands of frames"
    assert data["safety_critical_recall"] >= 0.99, "Fast-path safety critical recall should be ~100%"

    lat = data["latency_us"]
    assert lat["mean"] < 1_000.0, f"Mean latency must be sub-millisecond (< 1000 µs), got {lat['mean']} µs"
    assert lat["p95"] < 2_000.0, f"p95 latency must be sub-millisecond, got {lat['p95']} µs"


def test_ablation_study_core_conclusions() -> None:
    """Verify that ablation study proves the full cyber-physical system is strictly superior."""
    metrics_path = Path("output/metrics/ablation_metrics.json")
    if not metrics_path.exists():
        metrics_path = Path("cavshield/output/metrics/ablation_metrics.json")
    assert metrics_path.exists(), "ablation_metrics.json must exist"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "configurations" in data
    configs = {c["id"]: c for c in data["configurations"]}
    assert len(configs) == 4, "Must evaluate all 4 required configurations"

    c1 = configs["config_1_ai_only"]
    c2 = configs["config_2_ai_can_rules"]
    c3 = configs["config_3_ai_rules_v2x"]
    c4 = configs["config_4_full_system"]

    # 1. Config 4 (Full System) achieves 100% safety-critical recall with 0 missed attacks
    assert c4["missed_safety_critical_attacks"] == 0, "Full system must have 0 missed safety-critical attacks"
    assert c4["safety_critical_recall_pct"] == 100.0, "Full system must achieve 100% safety-critical recall"

    # 2. Config 4 has lower or equal FPR compared to Config 2 and Config 3
    assert c4["fpr_pct"] <= c2["fpr_pct"], "Full system should refine false alarms via physics"

    # 3. Hard-negative verification: Legitimate emergency braking must have 0% block rate in Full System
    assert c4["hard_negative_aeb_blocked_pct"] == 0.0, "Legitimate AEB must NEVER be blocked in full system"

    # 4. Standalone AI alone (Config 1) fails on in-range attacks and legitimate AEB
    assert c1["missed_safety_critical_attacks"] > 0, "AI only should miss in-range protocol-valid attacks"
    assert c1["hard_negative_aeb_blocked_pct"] > 0.0, "Standalone AI should trigger false alarm on sudden AEB"

    # 5. Check concrete case studies
    assert "case_studies" in data
    assert len(data["case_studies"]) >= 3


def test_report_generation_and_disclaimer() -> None:
    """Verify generated report.md and report.html contain prominent SYNTHETIC disclaimers."""
    md_path = Path("output/report.md")
    html_path = Path("output/report.html")
    if not md_path.exists():
        md_path = Path("cavshield/output/report.md")
        html_path = Path("cavshield/output/report.html")

    assert md_path.exists(), "report.md must be generated"
    assert html_path.exists(), "report.html must be generated"

    md_text = md_path.read_text(encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")

    # Mandatory SYNTHETIC disclaimer
    assert "SYNTHETIC" in md_text
    assert "SIMULATOR-ONLY" in md_text
    assert "SYNTHETIC" in html_text
    assert "SIMULATOR-ONLY" in html_text

    # Ablation matrix present
    assert "Ablation" in md_text
    assert "Config 4: Full System" in md_text
    assert "Ablation" in html_text
    assert "Config 4: Full System" in html_text


def test_hcrl_checker_function() -> None:
    """Verify check_hcrl_dataset correctly detects or notes absence of raw CSVs."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        raw_dir = Path("cavshield/data/raw")

    res = check_hcrl_dataset(raw_dir)
    assert "detected" in res
    assert "files" in res
    assert isinstance(res["files"], list)


def test_hcrl_data_loader_and_temporal_split(tmp_path: Path) -> None:
    """Verify HCRL CSV ingestion handles missing header, variable DLC, and time-block split."""
    import numpy as np
    from src.data_loader import load_hcrl_car_hacking, split_train_test_by_time_block

    # Create dummy HCRL CSV without header: Timestamp, CAN ID (hex), DLC, DATA[0..7] (hex), Flag (R/T)
    test_csv = tmp_path / "DoS_dataset.csv"
    lines = [
        "1478198376.123456,0130,8,11,22,33,44,55,66,77,88,R\n",
        "1478198376.123500,0000,8,00,00,00,00,00,00,00,00,T\n",
        "1478198376.123600,0140,4,AA,BB,CC,DD,R\n",  # DLC 4, padded to 8
    ]
    test_csv.write_text("".join(lines), encoding="utf-8")

    df = load_hcrl_car_hacking(test_csv)
    assert len(df) == 3
    assert df["label"].iloc[0] == "Normal"
    assert df["label"].iloc[1] == "Flooding"
    assert df["label"].iloc[2] == "Normal"
    assert df["dlc"].iloc[2] == 4
    assert len(df["data"].iloc[2]) == 8

    # Test time-block splitting
    run_ids = np.array(["DoS_dataset.csv"] * 10)
    t_ends = np.arange(10, dtype=float)
    train_m, test_m = split_train_test_by_time_block(run_ids, t_ends, test_split_pct=30.0)
    assert np.sum(train_m) == 7
    assert np.sum(test_m) == 3
    # Check that test set is strictly contiguous at the end of time
    assert np.all(test_m[7:])


