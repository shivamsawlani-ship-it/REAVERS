"""
CAVShield Classifier and Anomaly Detector Training Pipeline.

Trains:
1. IsolationForest baseline on normal window-stat vectors only (StandardScaler first),
   computing the 99.5th percentile threshold on normal training samples.
2. AttackClassifier (LightGBM / HistGradientBoosting) on multi-class window-stat vectors.

Outputs:
- Models saved to output/models/
- Evaluation metrics saved to output/metrics/classifier_metrics.json
- Console output with per-class F1, confusion matrix, and SIMULATOR-ONLY disclaimer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler  # type: ignore

from src import load_config
from src.can_simulator import KNOWN_IDS
from src.data_loader import (
    build_window_dataset,
    generate_synthetic_dataset,
    split_train_test_by_run,
)
from src.models.attack_classifier import CLASSES, build_classifier
from src.windowing import STAT_FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_models(
    cfg: Optional[Any] = None,
    dataset_path: Optional[str] = None,
    quick: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end training of IsolationForest and AttackClassifier.

    Parameters
    ----------
    cfg : Optional[Any]
        Configuration object.
    dataset_path : Optional[str]
        Path to pre-generated CSV dataset. If missing, synthetic dataset is generated.
    quick : bool
        If True, generates a small quick synthetic dataset if none exists.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed metrics.
    """
    if cfg is None:
        cfg = load_config()

    output_models_dir = Path(getattr(cfg.paths, "output_models", "output/models"))
    output_metrics_dir = Path(getattr(cfg.paths, "output_metrics", "output/metrics"))
    output_models_dir.mkdir(parents=True, exist_ok=True)
    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or Generate Dataset
    data_proc_dir = Path(getattr(cfg.paths, "data_processed", "data/processed"))
    target_csv = (
        Path(dataset_path)
        if dataset_path
        else data_proc_dir / "synthetic_can_dataset.csv.gz"
    )

    if not target_csv.exists():
        logger.info("Dataset not found at %s. Generating synthetic dataset...", target_csv)
        df_can = generate_synthetic_dataset(cfg, quick=quick)
    else:
        logger.info("Loading dataset from %s...", target_csv)
        df_can = pd.read_csv(target_csv)

    logger.info("Dataset loaded: %d frames across runs.", len(df_can))

    # 2. Build Window Features
    logger.info("Building window dataset (size=64, stride=16)...")
    t0_win = time.time()
    X_seq, X_stat, y, subtypes, run_ids, t_ends = build_window_dataset(
        df_can, known_ids=KNOWN_IDS, cfg=cfg
    )
    dt_win = time.time() - t0_win
    logger.info(
        "Extracted %d windows in %.2f s (Stat shape: %s, Seq shape: %s).",
        len(y),
        dt_win,
        X_stat.shape,
        X_seq.shape,
    )

    if len(y) == 0:
        raise RuntimeError("No windows were generated from the dataset!")

    # 3. Train/Test Split by Run
    test_split_pct = float(getattr(cfg.training, "test_split_pct", 25.0))
    train_mask, test_mask = split_train_test_by_run(run_ids, test_split_pct=test_split_pct)

    logger.info(
        "Split by run_id: Train=%d windows (%.1f%%), Test=%d windows (%.1f%%).",
        np.sum(train_mask),
        100.0 * np.mean(train_mask),
        np.sum(test_mask),
        100.0 * np.mean(test_mask),
    )

    # 4. Train IsolationForest on Normal Windows Only
    logger.info("Training IsolationForest on normal training windows...")
    normal_train_mask = train_mask & (y == "Normal")
    X_train_normal = X_stat[normal_train_mask]
    if len(X_train_normal) == 0:
        raise RuntimeError("No normal samples found in training set for IsolationForest!")

    scaler = StandardScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal)

    # IsolationForest: 100 trees, random_state=42
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_train_normal_scaled)

    # In sklearn IsolationForest, score_samples returns opposite of anomaly score (higher = normal).
    # We define raw anomaly score as -score_samples (higher = more anomalous).
    normal_scores = -iso_forest.score_samples(X_train_normal_scaled)
    # 99.5th percentile threshold on normal validation/training scores
    raw_threshold = float(np.percentile(normal_scores, 99.5))
    score_std = float(np.std(normal_scores))
    if score_std < 1e-4:
        score_std = 0.05

    logger.info(
        "IsolationForest trained on %d normal windows. 99.5th percentile threshold: %.4f (mean: %.4f, std: %.4f)",
        len(X_train_normal),
        raw_threshold,
        float(np.mean(normal_scores)),
        score_std,
    )

    # Evaluate IsolationForest on Test Set
    X_test_scaled = scaler.transform(X_stat[test_mask])
    test_scores = -iso_forest.score_samples(X_test_scaled)
    iso_pred_anomaly = test_scores >= raw_threshold
    y_test_is_anomaly = y[test_mask] != "Normal"

    iso_tp = int(np.sum(iso_pred_anomaly & y_test_is_anomaly))
    iso_fp = int(np.sum(iso_pred_anomaly & ~y_test_is_anomaly))
    iso_tn = int(np.sum(~iso_pred_anomaly & ~y_test_is_anomaly))
    iso_fn = int(np.sum(~iso_pred_anomaly & y_test_is_anomaly))

    iso_precision = float(iso_tp / max(iso_tp + iso_fp, 1))
    iso_recall = float(iso_tp / max(iso_tp + iso_fn, 1))
    iso_f1 = float(2 * iso_precision * iso_recall / max(iso_precision + iso_recall, 1e-6))

    logger.info(
        "IsolationForest Test: TP=%d, FP=%d, TN=%d, FN=%d | Prec=%.4f, Rec=%.4f, F1=%.4f",
        iso_tp,
        iso_fp,
        iso_tn,
        iso_fn,
        iso_precision,
        iso_recall,
        iso_f1,
    )

    # 5. Train Multi-Class Attack Classifier
    logger.info("Training Multi-Class Attack Classifier...")
    clf = build_classifier(random_state=42)

    X_train = X_stat[train_mask]
    y_train = y[train_mask].astype(str)

    t0_clf = time.time()
    clf.fit(X_train, y_train)
    dt_clf = time.time() - t0_clf
    logger.info("Attack Classifier trained in %.2f s.", dt_clf)

    # 6. Evaluate Classifier on Test Set
    X_test = X_stat[test_mask]
    y_test = y[test_mask].astype(str)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # Determine unique classes present in test and model
    eval_classes = sorted(list(set(list(y_train) + list(y_test))))
    cm = confusion_matrix(y_test, y_pred, labels=eval_classes)

    acc = float(accuracy_score(y_test, y_pred))
    macro_prec = float(precision_score(y_test, y_pred, labels=eval_classes, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, labels=eval_classes, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, labels=eval_classes, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, labels=eval_classes, average="weighted", zero_division=0))

    per_class_f1 = f1_score(y_test, y_pred, labels=eval_classes, average=None, zero_division=0)
    per_class_prec = precision_score(y_test, y_pred, labels=eval_classes, average=None, zero_division=0)
    per_class_rec = recall_score(y_test, y_pred, labels=eval_classes, average=None, zero_division=0)

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for idx, cname in enumerate(eval_classes):
        per_class_metrics[cname] = {
            "precision": float(per_class_prec[idx]),
            "recall": float(per_class_rec[idx]),
            "f1": float(per_class_f1[idx]),
            "support": int(np.sum(y_test == cname)),
        }

    # 7. Print Results and SIMULATOR-ONLY Disclaimer
    print("\n" + "=" * 70)
    print("        CAVSHIELD MULTI-CLASS ATTACK CLASSIFIER EVALUATION")
    print("=" * 70)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro Precision:   {macro_prec:.4f}")
    print(f"Macro Recall:      {macro_rec:.4f}")
    print(f"Macro F1 Score:    {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print("-" * 70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for cname, m in per_class_metrics.items():
        print(f"{cname:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<10d}")
    print("-" * 70)
    print("Confusion Matrix:")
    header = "       " + "".join(f"{c[:6]:>8}" for c in eval_classes)
    print(header)
    for i, cname in enumerate(eval_classes):
        row_str = f"{cname[:6]:<7}" + "".join(f"{cm[i, j]:>8d}" for j in range(len(eval_classes)))
        print(row_str)
    print("=" * 70)
    print("\n>>> [NOTE: THESE ARE SIMULATOR-ONLY METRICS] <<<\n")
    print("=" * 70 + "\n")

    # 8. Save Models and Metadata
    iforest_file = output_models_dir / "isolation_forest.joblib"
    scaler_file = output_models_dir / "scaler.joblib"
    clf_file = output_models_dir / "attack_classifier.joblib"
    meta_file = output_models_dir / "ai_metadata.json"

    joblib.dump(iso_forest, iforest_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(clf, clf_file)

    clf_classes = list(getattr(clf, "classes_", eval_classes))
    feature_names = STAT_FEATURE_NAMES(known_ids=KNOWN_IDS)

    metadata = {
        "isolation_forest": {
            "threshold": raw_threshold,
            "mean": float(np.mean(normal_scores)),
            "std": score_std,
            "n_normal_samples": int(len(X_train_normal)),
        },
        "classifier": {
            "classes": clf_classes,
            "type": type(clf).__name__,
            "n_features": int(X_stat.shape[1]),
        },
        "features": feature_names,
        "created_at": time.time(),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved models to %s", output_models_dir)

    # 9. Save JSON Metrics
    metrics_data = {
        "timestamp": time.time(),
        "simulator_only": True,
        "note": "THESE ARE SIMULATOR-ONLY METRICS",
        "dataset_windows": int(len(y)),
        "train_windows": int(np.sum(train_mask)),
        "test_windows": int(np.sum(test_mask)),
        "isolation_forest": {
            "raw_threshold": raw_threshold,
            "precision": iso_precision,
            "recall": iso_recall,
            "f1": iso_f1,
            "tp": iso_tp,
            "fp": iso_fp,
            "tn": iso_tn,
            "fn": iso_fn,
        },
        "classifier": {
            "accuracy": acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "classes": eval_classes,
        },
    }

    metrics_file = output_metrics_dir / "classifier_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    logger.info("Saved metrics to %s", metrics_file)
    return metrics_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CAVShield Anomaly Detector and Attack Classifier.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to pre-existing dataset CSV")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    args = parser.parse_args()

    train_models(dataset_path=args.dataset, quick=args.quick)

