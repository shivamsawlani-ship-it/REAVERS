"""
CAVShield Runtime Attack Classifier Wrapper.

Provides:
- ClassifierResult dataclass: probs dict, label, confidence, attack_prob, latency_ms.
- AttackClassifier: Thread-safe runtime wrapper that executes inference on window-stat
  vectors. Falls back to available=False if models are missing, ensuring the system can
  run rules-only without crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import joblib  # type: ignore
import numpy as np

from src import load_config
from src.models.attack_classifier import CLASSES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassifierResult:
    """Inference result produced by the AttackClassifier."""

    probs: Dict[str, float]
    label: str
    confidence: float
    attack_prob: float
    latency_ms: float


class AttackClassifier:
    """Runtime wrapper for the multi-class attack classifier."""

    def __init__(
        self,
        cfg: Optional[Any] = None,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize AttackClassifier.

        Parameters
        ----------
        cfg : Optional[Any]
            Project configuration.
        model_path : Optional[Union[str, Path]]
            Path to attack_classifier.joblib. Defaults to cfg.paths.output_models / 'attack_classifier.joblib'.
        metadata_path : Optional[Union[str, Path]]
            Path to ai_metadata.json.
        """
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg

        models_dir = Path(getattr(cfg.paths, "output_models", "output/models"))
        self.model_path = Path(model_path) if model_path else models_dir / "attack_classifier.joblib"
        self.metadata_path = Path(metadata_path) if metadata_path else models_dir / "ai_metadata.json"

        ai_cfg = getattr(cfg, "ai", {})
        self.min_class_confidence = float(getattr(ai_cfg, "min_class_confidence", 0.45))
        self.high_anomaly_threshold = float(getattr(ai_cfg, "anomaly_threshold", 0.60))

        self.model: Optional[Any] = None
        self.classes: List[str] = list(CLASSES)
        self.available: bool = False

        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load classifier model and metadata."""
        if not self.model_path.exists():
            logger.info("AttackClassifier model not found at %s. AI classification disabled.", self.model_path)
            self.available = False
            return

        try:
            self.model = joblib.load(self.model_path)
            if hasattr(self.model, "classes_"):
                self.classes = [str(c) for c in self.model.classes_]

            if self.metadata_path.exists():
                try:
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if "classifier" in meta and "classes" in meta["classifier"]:
                        self.classes = [str(c) for c in meta["classifier"]["classes"]]
                except Exception as meta_exc:
                    logger.debug("Could not read classifier metadata: %s", meta_exc)

            self.available = True
            logger.info("AttackClassifier loaded successfully from %s (classes: %s).", self.model_path, self.classes)
        except Exception as exc:
            logger.warning("Failed to load AttackClassifier model from %s: %s", self.model_path, exc)
            self.model = None
            self.available = False

    def predict(
        self,
        window_stats: Union[np.ndarray, List[float]],
        anomaly_score: float = 0.0,
    ) -> ClassifierResult:
        """Run classification on a window statistical feature vector.

        Parameters
        ----------
        window_stats : Union[np.ndarray, List[float]]
            1D or 2D array of window summary statistics.
        anomaly_score : float
            Normalized anomaly score in [0, 1] from AnomalyDetector.

        Returns
        -------
        ClassifierResult
            Classification probabilities, chosen label, confidence, attack probability, and latency.
        """
        if not self.available or self.model is None:
            return ClassifierResult(
                probs={c: (1.0 if c == "Normal" else 0.0) for c in self.classes},
                label="Normal",
                confidence=0.0,
                attack_prob=0.0,
                latency_ms=0.0,
            )

        t0 = time.perf_counter()
        vec = np.asarray(window_stats, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        try:
            if hasattr(self.model, "predict_proba"):
                probs_arr = self.model.predict_proba(vec)[0]
                probs_dict = {
                    cname: float(probs_arr[i])
                    for i, cname in enumerate(self.classes)
                    if i < len(probs_arr)
                }
            else:
                pred = str(self.model.predict(vec)[0])
                probs_dict = {c: (1.0 if c == pred else 0.0) for c in self.classes}

            # Best class and confidence
            sorted_classes = sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)
            top_class, top_conf = sorted_classes[0]

            # Attack probability is probability mass of non-Normal classes
            normal_prob = probs_dict.get("Normal", 0.0)
            attack_prob = float(np.clip(1.0 - normal_prob, 0.0, 1.0))

            # Unknown anomaly check:
            # If confidence is below min_class_confidence and the anomaly score is high, return Unknown anomaly
            if top_conf < self.min_class_confidence and anomaly_score >= self.high_anomaly_threshold:
                label = "Unknown anomaly"
            else:
                label = top_class

            latency_ms = (time.perf_counter() - t0) * 1000.0

            return ClassifierResult(
                probs=probs_dict,
                label=label,
                confidence=float(top_conf),
                attack_prob=attack_prob,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            logger.error("Classifier prediction failed: %s", exc)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ClassifierResult(
                probs={c: (1.0 if c == "Normal" else 0.0) for c in self.classes},
                label="Normal",
                confidence=0.0,
                attack_prob=0.0,
                latency_ms=latency_ms,
            )

