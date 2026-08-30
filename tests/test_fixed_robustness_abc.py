"""Tests for the explicit fixed-condition Model A/B/C evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

import evaluate_fixed_robustness_abc as abc_evaluation
from evaluate_fixed_robustness_abc import (
    EVALUATION_TITLE,
    FIXED_ABC_FILENAMES,
    MODEL_A_ID,
    MODEL_B_ID,
    MODEL_C_ID,
    build_argument_parser,
    build_combined_abc_summary,
    build_explicit_pairwise_comparison,
    condition_metadata,
    run_fixed_robustness_evaluation,
    run_fixed_robustness_abc_evaluation,
)
from src.evaluation_conditions import (
    CLEAN_CONDITION,
    FIXED_CHAIN_CONDITIONS,
    EvaluationCondition,
)
from src.evaluation_models import build_model_specs


class FixedLogitModel(nn.Module):
    """Return predetermined logits for a four-image test fixture."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer(
            "fixed_logits",
            torch.tensor(logits, dtype=torch.float32),
        )

    def forward(self, images):
        return self.fixed_logits[: images.size(0)].reshape(-1, 1)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
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


def _metric_row(model_id: str, condition_id: str, accuracy: float) -> dict:
    false_positives = 1 if accuracy < 1.0 else 0
    false_negatives = 1 if accuracy < 0.8 else 0
    return {
        "model_id": model_id,
        "condition_id": condition_id,
        "num_samples": 4,
        "threshold": 0.5,
        "true_negatives": 2 - false_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": 2 - false_negatives,
        "accuracy": accuracy,
        "balanced_accuracy": accuracy,
        "precision": accuracy,
        "recall": accuracy,
        "f1": accuracy,
        "auroc": accuracy,
        "auprc": accuracy,
        "false_positive_rate": false_positives / 2,
        "false_negative_rate": false_negatives / 2,
        "brier_score": 1.0 - accuracy,
    }


def test_condition_metadata_exposes_title_order_and_all_parameters():
    condition = FIXED_CHAIN_CONDITIONS[2]

    metadata = condition_metadata(condition)

    assert metadata["condition_id"] == "fixed_crop_colour_jpeg"
    assert metadata["condition_kind"] == "fixed_chain"
    assert metadata["num_transform_steps"] == 3
    assert metadata["condition_title"].startswith("Fixed chain")
    assert metadata["transform_chain"] == (
        "crop(fraction=0.8) -> colour(strength=0.2) -> jpeg(quality=50)"
    )
    parameters = json.loads(metadata["transform_parameters_json"])
    assert parameters == [
        {
            "order": 1,
            "transform": "crop",
            "parameter_name": "fraction",
            "parameter_value": 0.8,
        },
        {
            "order": 2,
            "transform": "colour",
            "parameter_name": "strength",
            "parameter_value": 0.2,
        },
        {
            "order": 3,
            "transform": "jpeg",
            "parameter_name": "quality",
            "parameter_value": 50,
        },
    ]


def test_single_condition_metadata_id_includes_severity():
    metadata = condition_metadata(EvaluationCondition("jpeg", 70))

    assert metadata["condition_id"] == "jpeg_70"


def test_pairwise_and_combined_reports_use_explicit_model_names():
    condition = EvaluationCondition(
        name="fixed_test",
        severity=None,
        title="Fixed chain — test",
        steps=(("resize", 0.5), ("jpeg", 70)),
        condition_kind="fixed_chain",
    )
    rows = []
    for condition_id, values in {
        "clean": (0.8, 0.9, 0.95),
        "fixed_test": (0.5, 0.75, 0.85),
    }.items():
        for model_id, accuracy in zip(
            (MODEL_A_ID, MODEL_B_ID, MODEL_C_ID),
            values,
            strict=True,
        ):
            rows.append(_metric_row(model_id, condition_id, accuracy))
    metrics = pd.DataFrame(rows)
    conditions = (CLEAN_CONDITION, condition)

    b_vs_c = build_explicit_pairwise_comparison(
        metrics,
        conditions=conditions,
        reference_model_id=MODEL_B_ID,
        candidate_model_id=MODEL_C_ID,
    ).set_index("condition_id")
    abc = build_combined_abc_summary(
        metrics,
        conditions=conditions,
    ).set_index("condition_id")

    assert "model_a_accuracy" not in b_vs_c.columns
    assert b_vs_c.loc["fixed_test", "reference_model_id"] == MODEL_B_ID
    assert b_vs_c.loc["fixed_test", "candidate_model_id"] == MODEL_C_ID
    assert b_vs_c.loc[
        "fixed_test",
        f"{MODEL_C_ID}_minus_{MODEL_B_ID}__accuracy",
    ] == pytest.approx(0.10)
    assert f"{MODEL_B_ID}__false_positives" in b_vs_c.columns
    assert f"{MODEL_C_ID}__false_negative_rate" in b_vs_c.columns
    assert abc.loc["fixed_test", "highest_accuracy_model"] == MODEL_C_ID
    assert abc.loc[
        "fixed_test",
        f"{MODEL_C_ID}_minus_{MODEL_A_ID}__accuracy",
    ] == pytest.approx(0.35)


def test_runner_writes_explicit_abc_outputs_and_preserves_fair_rows(
    tmp_path,
    monkeypatch,
):
    data_root, manifest_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "fixed-abc-results"
    conditions = (
        CLEAN_CONDITION,
        EvaluationCondition(
            name="fixed_resize_jpeg_test",
            severity=None,
            title="Fixed chain — Resize 0.50× → JPEG quality 70",
            steps=(("resize", 0.5), ("jpeg", 70)),
            condition_kind="fixed_chain",
        ),
    )

    def fake_checkpoint_loader(checkpoint_path, device):
        filename = Path(checkpoint_path).name
        if filename == "a.pt":
            return FixedLogitModel([-2.0, -2.0, -2.0, -2.0]), "a" * 64
        if filename == "b.pt":
            return FixedLogitModel([-2.0, 2.0, -2.0, 2.0]), "b" * 64
        return FixedLogitModel([-2.0, 2.0, 2.0, -2.0]), "c" * 64

    monkeypatch.setattr(
        abc_evaluation,
        "load_model_checkpoint",
        fake_checkpoint_loader,
    )

    predictions, metrics, pairwise, abc = (
        run_fixed_robustness_abc_evaluation(
            data_root=data_root,
            manifest_path=manifest_path,
            checkpoint_a_baseline=tmp_path / "a.pt",
            checkpoint_b_robustness=tmp_path / "b.pt",
            checkpoint_c_consistency=tmp_path / "c.pt",
            output_dir=output_dir,
            batch_size=4,
            num_workers=0,
            device=torch.device("cpu"),
            conditions=conditions,
        )
    )

    assert len(predictions) == 4 * 3 * 2
    assert len(metrics) == 3 * 2
    assert set(pairwise) == {"a_vs_b", "b_vs_c", "a_vs_c"}
    assert all(len(table) == 2 for table in pairwise.values())
    assert len(abc) == 2
    assert set(predictions["model_id"]) == {
        MODEL_A_ID,
        MODEL_B_ID,
        MODEL_C_ID,
    }
    assert set(metrics["model_id"]) == {
        MODEL_A_ID,
        MODEL_B_ID,
        MODEL_C_ID,
    }
    assert "false_positives" in metrics.columns
    assert "false_negatives" in metrics.columns
    assert "false_positive_rate" in metrics.columns
    assert "false_negative_rate" in metrics.columns

    for condition_id in ("clean", "fixed_resize_jpeg_test"):
        condition_rows = predictions[
            predictions["condition_id"] == condition_id
        ]
        paths_by_model = condition_rows.groupby("model_id")[
            "image_path"
        ].apply(list)
        assert paths_by_model[MODEL_A_ID] == paths_by_model[MODEL_B_ID]
        assert paths_by_model[MODEL_B_ID] == paths_by_model[MODEL_C_ID]

    for filename in FIXED_ABC_FILENAMES.values():
        assert (output_dir / filename).is_file()

    saved_b_vs_c = pd.read_csv(output_dir / FIXED_ABC_FILENAMES["b_vs_c"])
    assert saved_b_vs_c["report_title"].str.contains(
        "Model B.*Model C",
        regex=True,
    ).all()
    assert "model_a_accuracy" not in saved_b_vs_c.columns
    assert f"{MODEL_B_ID}__false_positives" in saved_b_vs_c.columns
    assert f"{MODEL_C_ID}__false_negatives" in saved_b_vs_c.columns

    with (output_dir / FIXED_ABC_FILENAMES["config"]).open(
        encoding="utf-8"
    ) as file:
        config = json.load(file)
    assert config["evaluation_title"] == EVALUATION_TITLE
    assert config["random_condition_sampling"] is False
    assert [item["model_id"] for item in config["models"]] == [
        MODEL_A_ID,
        MODEL_B_ID,
        MODEL_C_ID,
    ]


def test_cli_requires_role_specific_checkpoint_names():
    parser = build_argument_parser()
    args = parser.parse_args([
        "--data-root",
        "images",
        "--manifest",
        "manifest.csv",
        "--checkpoint-a-baseline",
        "a.pt",
        "--checkpoint-b-robustness",
        "b.pt",
        "--checkpoint-c-consistency",
        "c.pt",
        "--output-dir",
        "results",
    ])

    assert args.checkpoint_a_baseline == "a.pt"
    assert args.checkpoint_b_robustness == "b.pt"
    assert args.checkpoint_c_consistency == "c.pt"
    assert args.condition_set == "all-fixed"


def test_fixed_runner_supports_two_named_models_and_architecture_overrides(
    tmp_path,
    monkeypatch,
):
    data_root, manifest_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "fixed-sid-a-vs-b"
    architectures = []

    def fake_checkpoint_loader(checkpoint_path, device, architecture=None):
        architectures.append(architecture)
        logits = (
            [-2.0, 2.0, 2.0, -2.0]
            if Path(checkpoint_path).name == "sid-a.pt"
            else [-2.0, 2.0, -2.0, 2.0]
        )
        return FixedLogitModel(logits), Path(checkpoint_path).stem[0] * 64

    monkeypatch.setattr(
        abc_evaluation,
        "load_model_checkpoint",
        fake_checkpoint_loader,
    )
    specs = build_model_specs(
        checkpoints=("sid-a.pt", "sid-b.pt"),
        model_ids=("sid_a_clean", "sid_b_robust"),
        model_titles=("SID A clean", "SID B robust"),
        architectures=("efficientnet_b0", "efficientnet_b0"),
    )
    conditions = (
        CLEAN_CONDITION,
        EvaluationCondition(
            name="fixed_resize_jpeg_test",
            severity=None,
            title="Fixed chain test",
            steps=(("resize", 0.5), ("jpeg", 70)),
            condition_kind="fixed_chain",
        ),
    )

    predictions, metrics, pairwise, summary = run_fixed_robustness_evaluation(
        data_root=data_root,
        manifest_path=manifest_path,
        model_specs=specs,
        output_dir=output_dir,
        batch_size=4,
        num_workers=0,
        device=torch.device("cpu"),
        conditions=conditions,
    )

    assert len(predictions) == 4 * 2 * 2
    assert len(metrics) == 2 * 2
    assert set(pairwise) == {"sid_a_clean_vs_sid_b_robust"}
    assert len(summary) == 2
    assert architectures == ["efficientnet_b0", "efficientnet_b0"]
    config_path = (
        output_dir
        / "fixed_robustness__sid_a_clean_vs_sid_b_robust__run_config.json"
    )
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    assert config["num_models"] == 2
    assert [model["model_title"] for model in config["models"]] == [
        "SID A clean",
        "SID B robust",
    ]
