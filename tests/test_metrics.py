"""Tests for the evaluation metrics using small, hand-checkable examples."""

import pytest

from src.metrics import compute_binary_metrics


def test_compute_binary_metrics_matches_hand_calculation():
    # At threshold 0.5 the predictions are [real, AI, AI, AI].
    # Compared with labels [real, real, AI, AI], this gives one false positive.
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.8, 0.7, 0.9]

    metrics = compute_binary_metrics(labels, probabilities, threshold=0.5)

    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["true_positives"] == 2
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["false_negative_rate"] == pytest.approx(0.0)
    assert metrics["brier_score"] == pytest.approx(0.1875)


def test_compute_binary_metrics_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_binary_metrics([0, 1], [0.2, 1.2])


def test_compute_binary_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_binary_metrics([0, 1], [0.2])
