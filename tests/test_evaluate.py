"""Tests for the clean-image evaluation flow."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

import evaluate
from evaluate import (
    METRICS_FILENAMES,
    PREDICTION_COLUMNS,
    ROBUSTNESS_FILENAMES,
    build_robustness_comparison,
    evaluate_clean_model,
    load_model_checkpoint,
    run_clean_comparison,
    run_robustness_comparison,
)
from src.evaluation_conditions import CLEAN_CONDITION, EvaluationCondition


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
    assert predictions["is_correct"].tolist() == [True] * 4
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


class TinyCheckpointModel(nn.Module):
    """Small substitute used to test checkpoint loading quickly."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(2, 1)

    def forward(self, values):
        return self.classifier(values)


def test_checkpoint_loader_restores_weights_and_records_hash(tmp_path, monkeypatch):
    original_model = TinyCheckpointModel()
    with torch.no_grad():
        original_model.classifier.weight.fill_(0.25)
        original_model.classifier.bias.fill_(-0.5)

    checkpoint_path = tmp_path / "model.pt"
    torch.save(original_model.state_dict(), checkpoint_path)
    monkeypatch.setattr(evaluate, "build_model", lambda pretrained: TinyCheckpointModel())

    loaded_model, checkpoint_hash = load_model_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
    )

    inputs = torch.tensor([[2.0, 4.0]])
    assert loaded_model(inputs).item() == pytest.approx(1.0)
    assert len(checkpoint_hash) == 64


def test_checkpoint_loader_rejects_incompatible_weights(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "wrong.pt"
    torch.save({"unexpected.weight": torch.ones(1)}, checkpoint_path)
    monkeypatch.setattr(evaluate, "build_model", lambda pretrained: TinyCheckpointModel())

    with pytest.raises(ValueError, match="does not match"):
        load_model_checkpoint(checkpoint_path, torch.device("cpu"))


def _write_evaluation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "images"
    rows = []
    labels = [0.0, 1.0, 1.0, 0.0]
    for index, label in enumerate(labels):
        class_name = "FAKE" if label == 1 else "REAL"
        relative_path = f"train/{class_name}/image-{index}.png"
        image_path = data_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (index * 50,) * 3).save(image_path)
        rows.append({
            "image_path": relative_path,
            "label": label,
            "dataset": "dummy",
            "generator": "dummy-ai" if label == 1 else "none",
            "width": 32,
            "height": 32,
            "format": "PNG",
            "split": "val",
        })

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return data_root, manifest_path


def test_clean_comparison_uses_same_rows_and_writes_outputs(tmp_path, monkeypatch):
    data_root, manifest_path = _write_evaluation_fixture(tmp_path)
    output_dir = tmp_path / "results"

    def fake_checkpoint_loader(checkpoint_path, device):
        if Path(checkpoint_path).name == "a.pt":
            return FixedLogitModel([-2.0, 2.0, 2.0, -2.0]), "a" * 64
        return FixedLogitModel([-2.0, -2.0, 2.0, 2.0]), "b" * 64

    monkeypatch.setattr(evaluate, "load_model_checkpoint", fake_checkpoint_loader)

    predictions, metrics = run_clean_comparison(
        data_root=data_root,
        manifest_path=manifest_path,
        checkpoint_a=tmp_path / "a.pt",
        checkpoint_b=tmp_path / "b.pt",
        output_dir=output_dir,
        batch_size=4,
        num_workers=0,
        device=torch.device("cpu"),
    )

    assert len(predictions) == 8
    paths_by_model = predictions.groupby("model_id")["image_path"].apply(list)
    assert paths_by_model["experiment_a"] == paths_by_model["experiment_b"]
    assert metrics.set_index("model_id").loc["experiment_a", "accuracy"] == 1.0
    assert metrics.set_index("model_id").loc["experiment_b", "accuracy"] == 0.5

    for filename in METRICS_FILENAMES.values():
        assert (output_dir / filename).is_file()

    saved_predictions = pd.read_csv(output_dir / "clean_predictions.csv")
    assert len(saved_predictions) == 8
    with (output_dir / "evaluation_config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    assert config["split"] == "val"
    assert config["num_images_per_model"] == 4


def test_robustness_comparison_calculates_drops_and_b_advantage():
    conditions = (
        CLEAN_CONDITION,
        EvaluationCondition("jpeg", 30),
    )
    metrics = pd.DataFrame([
        {
            "model_id": "a",
            "transform": "clean",
            "severity": None,
            "accuracy": 0.9,
            "auroc": 0.95,
        },
        {
            "model_id": "b",
            "transform": "clean",
            "severity": None,
            "accuracy": 0.8,
            "auroc": 0.90,
        },
        {
            "model_id": "a",
            "transform": "jpeg",
            "severity": 30,
            "accuracy": 0.5,
            "auroc": 0.70,
        },
        {
            "model_id": "b",
            "transform": "jpeg",
            "severity": 30,
            "accuracy": 0.7,
            "auroc": 0.80,
        },
    ])

    comparison = build_robustness_comparison(
        metrics,
        conditions=conditions,
        model_a_id="a",
        model_b_id="b",
    ).set_index("transform")

    assert comparison.loc["jpeg", "b_minus_a_accuracy"] == pytest.approx(0.2)
    assert comparison.loc["jpeg", "model_a_drop_from_clean"] == pytest.approx(0.4)
    assert comparison.loc["jpeg", "model_b_drop_from_clean"] == pytest.approx(0.1)
    assert comparison.loc["jpeg", "b_minus_a_auroc"] == pytest.approx(0.1)
    assert comparison.loc["jpeg", "model_a_auroc_drop_from_clean"] == pytest.approx(
        0.25
    )
    assert comparison.loc["jpeg", "model_b_auroc_drop_from_clean"] == pytest.approx(
        0.1
    )
    assert comparison.loc["jpeg", "better_model"] == "b"


def test_robustness_runner_writes_each_condition_for_both_models(
    tmp_path,
    monkeypatch,
):
    data_root, manifest_path = _write_evaluation_fixture(tmp_path)
    output_dir = tmp_path / "robustness-results"
    conditions = (
        CLEAN_CONDITION,
        EvaluationCondition("jpeg", 30),
        EvaluationCondition("noise", 0.02),
    )

    def fake_checkpoint_loader(checkpoint_path, device):
        if Path(checkpoint_path).name == "a.pt":
            return FixedLogitModel([-2.0, 2.0, 2.0, -2.0]), "a" * 64
        return FixedLogitModel([-2.0, -2.0, 2.0, 2.0]), "b" * 64

    monkeypatch.setattr(evaluate, "load_model_checkpoint", fake_checkpoint_loader)

    predictions, metrics, comparison = run_robustness_comparison(
        data_root=data_root,
        manifest_path=manifest_path,
        checkpoint_a=tmp_path / "a.pt",
        checkpoint_b=tmp_path / "b.pt",
        output_dir=output_dir,
        batch_size=4,
        num_workers=0,
        device=torch.device("cpu"),
        conditions=conditions,
    )

    assert len(predictions) == 4 * 2 * 3
    assert len(metrics) == 2 * 3
    assert len(comparison) == 3
    for condition in conditions:
        condition_rows = predictions[
            predictions["transform"] == condition.name
        ]
        if condition.severity is not None:
            condition_rows = condition_rows[
                condition_rows["severity"] == condition.severity
            ]
        paths_by_model = condition_rows.groupby("model_id")["image_path"].apply(list)
        assert paths_by_model["experiment_a"] == paths_by_model["experiment_b"]

    for filename in ROBUSTNESS_FILENAMES.values():
        assert (output_dir / filename).is_file()

    with (output_dir / "robustness_config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    assert config["num_conditions"] == 3
    assert config["seed"] == 42
    assert config["conditions"][1]["condition_type"] == "single"
    assert config["conditions"][1]["steps"] == []
