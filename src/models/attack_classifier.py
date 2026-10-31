"""
CAVShield Attack Classifier Builder.

Provides:
- build_classifier(random_state=42): Constructs a LightGBM classifier with balanced class weights,
  falling back gracefully to sklearn HistGradientBoostingClassifier if LightGBM is unavailable.
- CLASSES: Canonical list of the 5 attack classes per CONTRACT.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Canonical attack classes per CONTRACT
CLASSES: List[str] = ["Normal", "Flooding", "Fuzzing", "Replay", "Spoofing"]


def build_classifier(
    random_state: int = 42,
    n_estimators: int = 150,
    learning_rate: float = 0.08,
    max_depth: int = 7,
) -> Any:
    """Build a multi-class tree ensemble classifier with balanced class weights.

    Attempts to construct a LightGBM classifier (LGBMClassifier). If LightGBM
    fails to import or initialize, falls back to scikit-learn's
    HistGradientBoostingClassifier.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility.
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Boosting learning rate.
    max_depth : int
        Maximum tree depth.

    Returns
    -------
    Classifier instance implementing .fit(X, y), .predict(X), and .predict_proba(X).
    """
    try:
        import lightgbm as lgb  # type: ignore

        clf = lgb.LGBMClassifier(
            class_weight="balanced",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        logger.info("Built LightGBM classifier with balanced class weights.")
        return clf

    except Exception as exc:
        logger.warning(
            "LightGBM not available or failed (%s); falling back to HistGradientBoostingClassifier.",
            exc,
        )
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        return clf

