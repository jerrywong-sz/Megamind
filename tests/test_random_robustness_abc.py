"""Tests for the Model A/B/C random-standard-3 evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

import evaluate_random_robustness_abc as random_evaluation
from evaluate_random_robustness_abc import (
    DEFAULT_TRIAL_SEEDS,
    MODEL_A_ID,
    MODEL_B_ID,
    MODEL_C_ID,
    RANDOM_ABC_FILENAMES,
    build_argument_parser,
    run_random_standard_3_abc_evaluation,
)


class FixedLogitModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer(
            "fixed_logits",
            torch.tensor(logits, dtype=torch.float32),
        )

    def forward(self, images):
        return self.fixed_logits[: images.size(0)].reshape(-1, 1)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "images"
    rows = []
    for index, label in enumerate((0.0, 1.0, 1.0, 0.0)):
        class_name = "FAKE" if label else "REAL"
        relative_path = f"train/{class_name}/image-{index}.png"
        image_path = data_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (index * 50,) * 3).save(image_path)
        rows.append({
            "image_path": relative_path,
            "label": label,
            "dataset": "dummy",
            "generator": "dummy-ai" if label else "none",
            "width": 32,
            "height": 32,
            "format": "PNG",
            "split": "val",
        })
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return data_root, manifest_path


def test_random_runner_writes_traceable_overall_and_pairwise_outputs(
    tmp_path,
    monkeypatch,
):
    data_root, manifest_path = _fixture(tmp_path)
    output_dir = tmp_path / "random-results"

    def fake_checkpoint_loader(checkpoint_path, device):
        filename = Path(checkpoint_path).name
        if filename == "a.pt":
            return FixedLogitModel([-2.0, -2.0, -2.0, -2.0]), "a" * 64
        if filename == "b.pt":
            return FixedLogitModel([-2.0, 2.0, -2.0, 2.0]), "b" * 64
        return FixedLogitModel([-2.0, 2.0, 2.0, -2.0]), "c" * 64

    monkeypatch.setattr(
        random_evaluation,
        "load_model_checkpoint",
        fake_checkpoint_loader,
    )
    outputs = run_random_standard_3_abc_evaluation(
        dataset_id="CIFAKE",
        data_root=data_root,
        manifest_path=manifest_path,
        checkpoint_a_baseline=tmp_path / "a.pt",
        checkpoint_b_robustness=tmp_path / "b.pt",
        checkpoint_c_consistency=tmp_path / "c.pt",
        output_dir=output_dir,
        threshold=0.4,
        batch_size=4,
        num_workers=0,
        device=torch.device("cpu"),
        trial_seeds=(42, 43),
    )

    predictions = outputs["predictions"]
    assignments = outputs["assignments"]
    assert len(predictions) == 4 * 2 * 3
    assert len(assignments) == 4 * 2
    assert len(outputs["trial_metrics"]) == 2 * 3
    assert len(outputs["overall"]) == 3
    assert len(outputs["headline"]) == 1
    assert len(outputs["a_vs_b"]) == 3
    assert len(outputs["b_vs_c"]) == 3
    assert len(outputs["a_vs_c"]) == 3
    assert set(predictions["model_id"]) == {
        MODEL_A_ID,
        MODEL_B_ID,
        MODEL_C_ID,
    }
    assert predictions["chain_length"].eq(3).all()
    assert predictions["error_type"].isin({
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    }).all()
    assert predictions.groupby(["trial_seed", "image_path"])[
        "transform_chain"
    ].nunique().eq(1).all()
    assert "clean_correct_retention_rate" in outputs["overall"].columns
    assert "accuracy_drop_from_clean" in outputs["overall"].columns
    assert outputs["patterns"]["threshold"].eq(0.4).all()
    assert outputs["inclusion"]["threshold"].eq(0.4).all()

    for filename in RANDOM_ABC_FILENAMES.values():
        assert (output_dir / filename).is_file()
    with (output_dir / RANDOM_ABC_FILENAMES["config"]).open(
        encoding="utf-8"
    ) as config_file:
        config = json.load(config_file)
    assert config["dataset_id"] == "CIFAKE"
    assert config["chain_length"] == 3
    assert config["trial_seeds"] == [42, 43]


def test_random_runner_rejects_duplicate_trial_seeds(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        run_random_standard_3_abc_evaluation(
            dataset_id="CIFAKE",
            data_root=tmp_path,
            manifest_path=tmp_path / "manifest.csv",
            checkpoint_a_baseline=tmp_path / "a.pt",
            checkpoint_b_robustness=tmp_path / "b.pt",
            checkpoint_c_consistency=tmp_path / "c.pt",
            output_dir=tmp_path / "results",
            trial_seeds=(42, 42),
        )


def test_random_cli_defaults_to_five_reproducible_trials():
    args = build_argument_parser().parse_args([
        "--dataset-id", "CIFAKE",
        "--data-root", "images",
        "--manifest", "manifest.csv",
        "--checkpoint-a-baseline", "a.pt",
        "--checkpoint-b-robustness", "b.pt",
        "--checkpoint-c-consistency", "c.pt",
        "--output-dir", "results",
    ])

    assert tuple(args.trial_seeds) == DEFAULT_TRIAL_SEEDS


def test_random_runner_supports_three_custom_checkpoint_labels(
    tmp_path,
    monkeypatch,
):
    data_root, manifest_path = _fixture(tmp_path)
    output_dir = tmp_path / "custom-results"

    def fake_checkpoint_loader(checkpoint_path, device):
        filename = Path(checkpoint_path).name
        logits = {
            "old.pt": [-2.0, -2.0, -2.0, -2.0],
            "sid-a.pt": [-2.0, 2.0, -2.0, 2.0],
            "sid-b.pt": [-2.0, 2.0, 2.0, -2.0],
        }[filename]
        return FixedLogitModel(logits), filename[0] * 64

    monkeypatch.setattr(
        random_evaluation,
        "load_model_checkpoint",
        fake_checkpoint_loader,
    )
    outputs = run_random_standard_3_abc_evaluation(
        dataset_id="SID_SET",
        data_root=data_root,
        manifest_path=manifest_path,
        checkpoint_a_baseline=tmp_path / "old.pt",
        checkpoint_b_robustness=tmp_path / "sid-a.pt",
        checkpoint_c_consistency=tmp_path / "sid-b.pt",
        output_dir=output_dir,
        batch_size=4,
        num_workers=0,
        device=torch.device("cpu"),
        trial_seeds=(42,),
        model_a_id="best_old",
        model_a_title="Best old checkpoint",
        model_b_id="sid_a_clean",
        model_b_title="SID A clean training",
        model_c_id="sid_b_robust",
        model_c_title="SID B robustness training",
    )

    expected_ids = {"best_old", "sid_a_clean", "sid_b_robust"}
    assert set(outputs["predictions"]["model_id"]) == expected_ids
    assert set(outputs["overall"]["model_id"]) == expected_ids
    assert outputs["headline"]["evaluation_title"].iloc[0] == (
        "Random standard-3 robustness evaluation — Best old checkpoint vs "
        "SID A clean training vs SID B robustness training"
    )
    assert set(outputs["a_vs_b"]["reference_model_id"]) == {"best_old"}
    assert set(outputs["a_vs_b"]["candidate_model_id"]) == {"sid_a_clean"}

    config_path = output_dir / "random_standard_3__run_config.json"
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    assert [model["model_id"] for model in config["models"]] == [
        "best_old",
        "sid_a_clean",
        "sid_b_robust",
    ]
    assert config["output_files"]["overall"] == (
        "random_standard_3__best_old_vs_sid_a_clean_vs_sid_b_robust"
        "__overall_summary.csv"
    )
    for filename in config["output_files"].values():
        assert (output_dir / filename).is_file()


def test_random_cli_accepts_generic_checkpoint_aliases_and_custom_labels():
    args = build_argument_parser().parse_args([
        "--dataset-id", "SID_SET",
        "--data-root", "images",
        "--manifest", "manifest.csv",
        "--checkpoint-a", "old.pt",
        "--checkpoint-b", "sid-a.pt",
        "--checkpoint-c", "sid-b.pt",
        "--model-a-id", "best_old",
        "--model-a-title", "Best old checkpoint",
        "--model-b-id", "sid_a_clean",
        "--model-b-title", "SID A clean training",
        "--model-c-id", "sid_b_robust",
        "--model-c-title", "SID B robustness training",
        "--output-dir", "results",
    ])

    assert args.checkpoint_a_baseline == "old.pt"
    assert args.checkpoint_b_robustness == "sid-a.pt"
    assert args.checkpoint_c_consistency == "sid-b.pt"
    assert args.model_a_id == "best_old"
    assert args.model_c_title == "SID B robustness training"
