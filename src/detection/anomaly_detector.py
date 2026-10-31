"""
CAVShield Runtime Anomaly Detector Wrapper.

Provides:
- AnomalyResult dataclass: raw score, normalized score (0..1 via sigmoid around threshold),
  is_anomaly boolean, latency_ms.
- AnomalyDetector: Runtime engine supporting modular backends (IsolationForest and
  LSTM Autoencoder), configurable via `ai.anomaly_backend` in config/config.yaml.
- Graceful degradation: exposes available=False if models are missing so the system
  continues in rules-only mode without crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import joblib  # type: ignore
import numpy as np

from src import load_config
from src.can_simulator import KNOWN_IDS
from src.windowing import Window, window_stat_features

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnomalyResult:
    """Inference result produced by the AnomalyDetector."""

    raw: float
    normalized: float
    is_anomaly: bool
    latency_ms: float


class AnomalyDetector:
    """Runtime wrapper for anomaly detection with IsolationForest or LSTM Autoencoder."""

    def __init__(
        self,
        cfg: Optional[Any] = None,
        backend: Optional[str] = None,
        model_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize AnomalyDetector.

        Parameters
        ----------
        cfg : Optional[Any]
            Project configuration.
        backend : Optional[str]
            Override backend: 'iforest' or 'lstm_ae'. If None, reads cfg.ai.anomaly_backend.
        model_dir : Optional[Union[str, Path]]
            Directory containing trained model artifacts.
        """
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        ai_cfg = getattr(cfg, "ai", {})
        self.backend_name = (backend or getattr(ai_cfg, "anomaly_backend", "iforest")).lower()

        models_dir_path = Path(model_dir) if model_dir else Path(getattr(cfg.paths, "output_models", "output/models"))
        self.models_dir = models_dir_path

        # State
        self.available: bool = False
        self.threshold: float = 0.5
        self.scale: float = 0.05

        # Backend components
        self.iso_forest: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.ae_model: Optional[Any] = None
        self.onnx_detector: Optional[Any] = None
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None

        self._load_backend()

    def _load_backend(self) -> None:
        """Load selected backend model and threshold metadata."""
        if self.backend_name in ("onnx", "onnx_ae", "onnx_anomaly"):
            self._load_onnx()
        elif self.backend_name == "lstm_ae":
            self._load_lstm_ae()
        else:
            self._load_isolation_forest()

    def _load_onnx(self) -> None:
        """Load ONNX Runtime backend."""
        try:
            from src.edge.onnx_inference import ONNXAnomalyDetector

            self.onnx_detector = ONNXAnomalyDetector(cfg=self.cfg)
            self.available = self.onnx_detector.available
            self.threshold = self.onnx_detector.threshold
            self.scale = self.onnx_detector.scale
            if self.available:
                logger.info("Loaded ONNX Anomaly backend (threshold: %.6f).", self.threshold)
            else:
                logger.warning("ONNX Anomaly backend unavailable.")
        except Exception as exc:
            logger.warning("Failed to load ONNX backend: %s", exc)
            self.available = False
            self.onnx_detector = None

    def _load_isolation_forest(self) -> None:
        """Load IsolationForest baseline model, scaler, and threshold metadata."""
        iforest_path = self.models_dir / "isolation_forest.joblib"
        scaler_path = self.models_dir / "scaler.joblib"
        meta_path = self.models_dir / "ai_metadata.json"

        if not (iforest_path.exists() and scaler_path.exists()):
            logger.info("IsolationForest model/scaler not found in %s. Anomaly detector unavailable.", self.models_dir)
            self.available = False
            return

        try:
            self.iso_forest = joblib.load(iforest_path)
            self.scaler = joblib.load(scaler_path)

            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if_meta = meta.get("isolation_forest", {})
                self.threshold = float(if_meta.get("threshold", 0.5))
                self.scale = max(float(if_meta.get("std", 0.05)), 1e-4)
            else:
                self.threshold = 0.5
                self.scale = 0.05

            self.available = True
            logger.info("Loaded IsolationForest backend (threshold: %.4f, scale: %.4f).", self.threshold, self.scale)

        except Exception as exc:
            logger.warning("Failed to load IsolationForest backend: %s", exc)
            self.iso_forest = None
            self.scaler = None
            self.available = False

    def _load_lstm_ae(self) -> None:
        """Load PyTorch LSTM Autoencoder model and normalization metadata."""
        model_pt_path = self.models_dir / "anomaly_ae.pt"
        meta_path = self.models_dir / "ae_metadata.json"

        if not (model_pt_path.exists() and meta_path.exists()):
            logger.info("LSTM Autoencoder artifacts not found in %s. Falling back to unavailable.", self.models_dir)
            self.available = False
            return

        try:
            import torch
            from src.models.lstm_autoencoder import LSTMAutoencoder

            with open(meta_path, "r", encoding="utf-8") as f:
                ae_meta = json.load(f)

            self.threshold = float(ae_meta.get("threshold", 0.05))
            self.scale = max(float(ae_meta.get("std_error", 0.01)), 1e-4)

            self.feat_mean = np.array(ae_meta.get("feat_mean"), dtype=np.float32)
            self.feat_std = np.array(ae_meta.get("feat_std"), dtype=np.float32)

            # Reconstruct model architecture
            self.ae_model = LSTMAutoencoder(
                input_dim=int(ae_meta.get("input_dim", 17)),
                seq_len=int(ae_meta.get("seq_len", 64)),
                hidden_dim=int(ae_meta.get("hidden_dim", 64)),
                bottleneck_dim=int(ae_meta.get("bottleneck_dim", 24)),
                num_layers=int(ae_meta.get("num_layers", 2)),
                dropout=0.0,
            )
            state_dict = torch.load(model_pt_path, map_location="cpu", weights_only=True)
            self.ae_model.load_state_dict(state_dict)
            self.ae_model.eval()

            self.available = True
            logger.info("Loaded LSTM Autoencoder backend (threshold: %.6f).", self.threshold)

        except Exception as exc:
            logger.warning("Failed to load LSTM Autoencoder backend: %s", exc)
            self.ae_model = None
            self.available = False

    def score(
        self,
        window_or_data: Union[Window, np.ndarray, List[float]],
    ) -> AnomalyResult:
        """Compute anomaly score for a given window or feature representation.

        Parameters
        ----------
        window_or_data : Union[Window, np.ndarray, List[float]]
            Either a Window dataclass, a 2D sequence matrix (64, 17), or a 1D statistical vector.

        Returns
        -------
        AnomalyResult
            Raw score, normalized score (0..1 via sigmoid around threshold), is_anomaly, and latency.
        """
        if not self.available:
            return AnomalyResult(raw=0.0, normalized=0.0, is_anomaly=False, latency_ms=0.0)

        if self.backend_name in ("onnx", "onnx_ae", "onnx_anomaly") and self.onnx_detector is not None:
            return self.onnx_detector.score(window_or_data)

        t0 = time.perf_counter()

        try:
            if self.backend_name == "lstm_ae":
                raw_score = self._score_lstm_ae(window_or_data)
            else:
                raw_score = self._score_isolation_forest(window_or_data)

            # Normalization via sigmoid centered at threshold:
            # At raw == threshold, z = 0, normalized = 0.5
            z = (raw_score - self.threshold) / max(self.scale, 1e-4)
            # Clip z to prevent float overflow in exp
            z = float(np.clip(z, -20.0, 20.0))
            normalized = float(1.0 / (1.0 + math.exp(-z)))
            is_anomaly = bool(raw_score >= self.threshold)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return AnomalyResult(
                raw=float(raw_score),
                normalized=normalized,
                is_anomaly=is_anomaly,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            logger.error("Anomaly scoring failed: %s", exc)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return AnomalyResult(raw=0.0, normalized=0.0, is_anomaly=False, latency_ms=latency_ms)

    def _score_isolation_forest(
        self,
        window_or_data: Union[Window, np.ndarray, List[float]],
    ) -> float:
        """Compute raw anomaly score with IsolationForest."""
        if isinstance(window_or_data, Window):
            vec = window_stat_features(window_or_data, known_ids=KNOWN_IDS)
        else:
            arr = np.asarray(window_or_data, dtype=np.float32)
            if arr.ndim == 2 and arr.shape == (64, 17):
                # Sequential window passed: wrap in Window to extract stat features
                temp_win = Window(
                    features=arr,
                    t_start=0.0,
                    t_end=0.0,
                    label="Normal",
                    subtype="",
                )
                vec = window_stat_features(temp_win, known_ids=KNOWN_IDS)
            else:
                vec = arr.reshape(-1)

        vec_2d = vec.reshape(1, -1)
        scaled_vec = self.scaler.transform(vec_2d)
        # raw score = -score_samples (higher = more anomalous)
        raw_score = float(-self.iso_forest.score_samples(scaled_vec)[0])
        return raw_score

    def _score_lstm_ae(
        self,
        window_or_data: Union[Window, np.ndarray, List[float]],
    ) -> float:
        """Compute raw reconstruction error with PyTorch LSTM Autoencoder."""
        import torch

        if isinstance(window_or_data, Window):
            seq = window_or_data.features
        else:
            arr = np.asarray(window_or_data, dtype=np.float32)
            if arr.ndim == 1:
                # Cannot run LSTM on 1D summary vector
                raise ValueError("LSTM Autoencoder requires (64, 17) sequence data, not 1D stats.")
            seq = arr

        if seq.shape != (64, 17):
            raise ValueError(f"Expected sequence shape (64, 17), got {seq.shape}")

        # Normalize with stored mean and std
        if self.feat_mean is not None and self.feat_std is not None:
            seq_norm = (seq - self.feat_mean) / self.feat_std
        else:
            seq_norm = seq

        tensor = torch.from_numpy(seq_norm).float().unsqueeze(0)  # (1, 64, 17)
        with torch.no_grad():
            err_tensor = self.ae_model.reconstruction_error(tensor, reduction="mean")
            raw_score = float(err_tensor.item())

        return raw_score

