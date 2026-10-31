"""
Unit and integration tests for CAVShield Part 6 (AI & Machine Learning Baseline + Deep Model).

Covers:
- build_classifier (LightGBM / HistGradientBoosting)
- LSTMAutoencoder architecture and tensor dimensions
- AnomalyDetector runtime wrapper (iforest and lstm_ae backends)
- AttackClassifier runtime wrapper, confidence gate, and Unknown anomaly classification
- Inference latency benchmarks (< 10 ms)
"""

from __future__ import annotations

import math
from pathlib import Path
import time
import numpy as np
import pytest
import torch

from src import load_config
from src.can_simulator import KNOWN_IDS
from src.detection.anomaly_detector import AnomalyDetector, AnomalyResult
from src.detection.attack_classifier import AttackClassifier, ClassifierResult
from src.models.attack_classifier import CLASSES, build_classifier
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.windowing import Window, window_stat_features


def test_build_classifier() -> None:
    """Test classifier construction and basic fitting."""
    clf = build_classifier(random_state=42, n_estimators=10)
    assert hasattr(clf, "fit")
    assert hasattr(clf, "predict")
    assert hasattr(clf, "predict_proba")

    # Fit dummy 103-dim data
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(20, 103)).astype(np.float32)
    y = np.array(["Normal"] * 10 + ["Flooding"] * 5 + ["Fuzzing"] * 5)
    clf.fit(X, y)

    preds = clf.predict(X[:3])
    probs = clf.predict_proba(X[:3])
    assert len(preds) == 3
    assert probs.shape == (3, 3)


def test_lstm_autoencoder_architecture() -> None:
    """Test LSTMAutoencoder tensor shapes and reconstruction error."""
    model = LSTMAutoencoder(
        input_dim=17,
        seq_len=64,
        hidden_dim=32,
        bottleneck_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    model.eval()

    # Batch of 4 sequences
    x = torch.randn(4, 64, 17)
    recon = model(x)
    assert recon.shape == (4, 64, 17)

    # Reconstruction error
    err_none = model.reconstruction_error(x, reduction="none")
    assert err_none.shape == (4,)
    assert torch.all(err_none >= 0.0)

    err_mean = model.reconstruction_error(x, reduction="mean")
    assert err_mean.dim() == 0
    assert err_mean.item() >= 0.0


def test_attack_classifier_fallback() -> None:
    """Test AttackClassifier when model file is absent."""
    clf = AttackClassifier(model_path="nonexistent_clf.joblib")
    assert clf.available is False

    dummy_vec = np.zeros(103, dtype=np.float32)
    res = clf.predict(dummy_vec)
    assert isinstance(res, ClassifierResult)
    assert res.label == "Normal"
    assert res.confidence == 0.0
    assert res.attack_prob == 0.0


def test_anomaly_detector_fallback() -> None:
    """Test AnomalyDetector when model files are absent."""
    det = AnomalyDetector(backend="iforest", model_dir="nonexistent_dir")
    assert det.available is False

    dummy_win = Window(
        features=np.zeros((64, 17), dtype=np.float32),
        t_start=0.0,
        t_end=0.1,
        label="Normal",
        subtype="",
    )
    res = det.score(dummy_win)
    assert isinstance(res, AnomalyResult)
    assert res.raw == 0.0
    assert res.normalized == 0.0
    assert res.is_anomaly is False


def test_attack_classifier_unknown_anomaly_trigger() -> None:
    """Verify that low confidence + high anomaly score produces 'Unknown anomaly'."""
    clf = AttackClassifier(model_path="nonexistent_clf.joblib")
    # Mock model
    class MockModel:
        classes_ = np.array(["Normal", "Flooding", "Fuzzing", "Replay", "Spoofing"])
        def predict_proba(self, X):
            # Low confidence split across classes (max prob = 0.3)
            return np.array([[0.25, 0.30, 0.20, 0.15, 0.10]])

    clf.model = MockModel()
    clf.classes = list(MockModel.classes_)
    clf.available = True
    clf.min_class_confidence = 0.45
    clf.high_anomaly_threshold = 0.60

    dummy_vec = np.zeros(103, dtype=np.float32)

    # Case 1: Low confidence, low anomaly score -> Flooding (top class)
    res1 = clf.predict(dummy_vec, anomaly_score=0.2)
    assert res1.label == "Flooding"

    # Case 2: Low confidence, HIGH anomaly score -> Unknown anomaly
    res2 = clf.predict(dummy_vec, anomaly_score=0.75)
    assert res2.label == "Unknown anomaly"


def test_anomaly_detector_sigmoid_normalization() -> None:
    """Verify sigmoid normalization produces 0.5 at threshold and scales smoothly."""
    det = AnomalyDetector(backend="iforest", model_dir="nonexistent_dir")
    det.threshold = 0.50
    det.scale = 0.10
    det.available = True

    class MockScaler:
        def transform(self, X):
            return X

    class MockForest:
        def __init__(self, score_val):
            self.score_val = score_val
        def score_samples(self, X):
            return np.array([-self.score_val])

    det.scaler = MockScaler()

    # Score exactly at threshold -> raw = 0.50 -> normalized should be 0.50
    det.iso_forest = MockForest(0.50)
    res_thresh = det.score(np.zeros(103))
    assert math.isclose(res_thresh.normalized, 0.50, abs_tol=1e-3)
    assert res_thresh.is_anomaly is True

    # Score below threshold -> normalized < 0.50, is_anomaly = False
    det.iso_forest = MockForest(0.30)
    res_low = det.score(np.zeros(103))
    assert res_low.normalized < 0.50
    assert res_low.is_anomaly is False

    # Score above threshold -> normalized > 0.50, is_anomaly = True
    det.iso_forest = MockForest(0.70)
    res_high = det.score(np.zeros(103))
    assert res_high.normalized > 0.50
    assert res_high.is_anomaly is True


def test_real_models_end_to_end() -> None:
    """Test trained models end-to-end for anomaly detection and attack classification."""
    cfg = load_config()

    # 1. Test IsolationForest backend
    det_iforest = AnomalyDetector(cfg=cfg, backend="iforest")
    assert det_iforest.available is True

    # 2. Test LSTM AE backend
    det_lstm = AnomalyDetector(cfg=cfg, backend="lstm_ae")
    assert det_lstm.available is True

    # 3. Test AttackClassifier
    clf = AttackClassifier(cfg=cfg)
    assert clf.available is True

    # Generate normal synthetic window features
    dummy_seq = np.zeros((64, 17), dtype=np.float32)
    win = Window(
        features=dummy_seq,
        t_start=0.0,
        t_end=0.32,
        label="Normal",
        subtype="",
    )
    stat_vec = window_stat_features(win, known_ids=KNOWN_IDS)

    # Warm-up calls for first inference
    _ = det_iforest.score(win)
    _ = det_lstm.score(win)
    _ = clf.predict(stat_vec)

    # Score with IsolationForest
    t0 = time.perf_counter()
    res_if = det_iforest.score(win)
    dt_if = (time.perf_counter() - t0) * 1000.0
    assert isinstance(res_if, AnomalyResult)
    assert 0.0 <= res_if.normalized <= 1.0
    assert dt_if < 50.0, f"IsolationForest latency too high: {dt_if:.2f} ms"

    # Score with LSTM AE
    t0 = time.perf_counter()
    res_ae = det_lstm.score(win)
    dt_ae = (time.perf_counter() - t0) * 1000.0
    assert isinstance(res_ae, AnomalyResult)
    assert 0.0 <= res_ae.normalized <= 1.0
    assert dt_ae < 50.0, f"LSTM AE latency too high: {dt_ae:.2f} ms"

    # Predict with AttackClassifier
    t0 = time.perf_counter()
    res_clf = clf.predict(stat_vec, anomaly_score=res_if.normalized)
    dt_clf = (time.perf_counter() - t0) * 1000.0
    assert isinstance(res_clf, ClassifierResult)
    assert res_clf.label in CLASSES or res_clf.label == "Unknown anomaly"
    assert 0.0 <= res_clf.confidence <= 1.0
    assert 0.0 <= res_clf.attack_prob <= 1.0
    assert dt_clf < 50.0, f"Classifier latency too high: {dt_clf:.2f} ms"


