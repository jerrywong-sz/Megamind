"""Tests for the clean-image evaluation flow."""

import pytest
import torch
import torch.nn as nn

from evaluate import PREDICTION_COLUMNS, evaluate_clean_model


class FixedLogitModel(nn.Module):
    """Dummy model that returns known logits instead of analysing images."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, images):
        return self.fixed_logits[: images.size(0)].reshape(-1, 1)


def make_one_dummy_batch():
    """Create four fake images with labels and traceable metadata."""
    images = torch.zeros(4, 3, 224, 224)
    labels = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    metadata = {
        "image_id": ["real-1", "ai-1", "ai-2", "real-2"],
        "image_path": ["real-1.png", "ai-1.png", "ai-2.png", "real-2.png"],
        "dataset": ["dummy"] * 4,
        "source": ["test"] * 4,
        "generator": ["none", "dummy-ai", "dummy-ai", "none"],
    }
    return images, labels, metadata


def test_clean_evaluation_converts_logits_and_calculates_metrics():
    # These logits become probabilities of roughly [0.12, 0.80, 0.57, 0.33].
    # At threshold 0.5, the resulting labels exactly match [0, 1, 1, 0].
    model = FixedLogitModel([-2.0, 1.4, 0.3, -0.7])
    data_loader = [make_one_dummy_batch()]

    predictions, metrics = evaluate_clean_model(
        model,
        data_loader,
        torch.device("cpu"),
        model_id="dummy-model",
        threshold=0.5,
    )

    assert list(predictions.columns) == PREDICTION_COLUMNS
    assert predictions["image_id"].tolist() == [
        "real-1",
        "ai-1",
        "ai-2",
        "real-2",
    ]
    assert predictions["transform"].tolist() == ["clean"] * 4
    assert predictions["predicted_label"].tolist() == [0, 1, 1, 0]
    assert predictions["prob_ai"].tolist() == pytest.approx([
        0.119203,
        0.802184,
        0.574443,
        0.331812,
    ])
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)


def test_clean_evaluation_rejects_an_empty_loader():
    model = FixedLogitModel([])

    with pytest.raises(ValueError, match="at least one image"):
        evaluate_clean_model(
            model,
            [],
            torch.device("cpu"),
            model_id="dummy-model",
        )


def test_clean_evaluation_rejects_non_binary_labels():
    model = FixedLogitModel([-2.0, 1.4, 0.3, -0.7])
    images, labels, metadata = make_one_dummy_batch()
    labels[0] = 0.5

    with pytest.raises(ValueError, match="only 0 .* and 1"):
        evaluate_clean_model(
            model,
            [(images, labels, metadata)],
            torch.device("cpu"),
            model_id="dummy-model",
        )
