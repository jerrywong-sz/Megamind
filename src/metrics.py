"""Metrics used to evaluate the binary AI-image detector.

The detector uses the shared label convention:

* 0 means a real image.
* 1 means an AI-generated image.

Models produce one probability per image: the estimated probability that the
image is AI-generated.  Threshold-dependent metrics convert that probability
to a label using ``probability >= threshold``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate classification metrics from labels and AI probabilities.

    Args:
        labels: Ground-truth labels, where 0 is real and 1 is AI-generated.
        probabilities: Model probabilities that the images are AI-generated.
        threshold: Probabilities greater than or equal to this value are
            classified as AI-generated.

    Returns:
        A dictionary containing the confusion-matrix counts, threshold-based
        metrics, threshold-free ranking metrics, and Brier score.

    Raises:
        ValueError: If the inputs are empty, have different lengths, are not
            one-dimensional, contain invalid values, or use an invalid
            threshold.
    """
    label_array = np.asarray(labels)
    probability_array = np.asarray(probabilities, dtype=float)

    if label_array.ndim != 1 or probability_array.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if label_array.size == 0:
        raise ValueError("labels and probabilities must not be empty")
    if label_array.size != probability_array.size:
        raise ValueError("labels and probabilities must have the same length")
    if not np.isin(label_array, [0, 1]).all():
        raise ValueError("labels must contain only 0 (real) and 1 (AI)")
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must contain only finite values")
    if not ((0.0 <= probability_array) & (probability_array <= 1.0)).all():
        raise ValueError("probabilities must be between 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    label_array = label_array.astype(int)
    predicted_labels = (probability_array >= threshold).astype(int)

    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix(label_array, predicted_labels, labels=[0, 1]).ravel()
    )

    real_count = true_negatives + false_positives
    ai_count = true_positives + false_negatives
    false_positive_rate = false_positives / real_count if real_count else float("nan")
    false_negative_rate = false_negatives / ai_count if ai_count else float("nan")

    # AUROC and balanced accuracy only have their intended meaning when both
    # real and AI examples are present in the evaluated group.
    has_both_classes = np.unique(label_array).size == 2
    auroc = roc_auc_score(label_array, probability_array) if has_both_classes else float("nan")
    auprc = average_precision_score(label_array, probability_array) if has_both_classes else float("nan")
    balanced_accuracy = (
        balanced_accuracy_score(label_array, predicted_labels)
        if has_both_classes
        else float("nan")
    )

    return {
        "num_samples": int(label_array.size),
        "threshold": float(threshold),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
        "accuracy": float(accuracy_score(label_array, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(
            precision_score(label_array, predicted_labels, zero_division=0)
        ),
        "recall": float(recall_score(label_array, predicted_labels, zero_division=0)),
        "f1": float(f1_score(label_array, predicted_labels, zero_division=0)),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "brier_score": float(np.mean((probability_array - label_array) ** 2)),
    }
