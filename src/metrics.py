"""Ranking metrics. Accuracy is logged only to show why it misleads."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_scores(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    high_risk_threshold: float,
) -> dict[str, float]:
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(probabilities, dtype=float)
    flagged = (scores >= high_risk_threshold).astype(int)
    predicted = (scores >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "recall_high_risk": float(recall_score(labels, flagged, zero_division=0)),
        "precision_high_risk": float(precision_score(labels, flagged, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
    }
