"""Evaluate trained checkpoints on clean and transformed images.

This module owns both the reusable per-model evaluation function and the
command-line runners that connect real checkpoints to a manifest-backed image
split. Clean evaluation verifies the integration; robustness evaluation then
compares the same models under fixed, reproducible image transformations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from src.data import get_evaluation_dataloader
from src.evaluation_conditions import (
    ALL_EVALUATION_CONDITIONS,
    EvaluationCondition,
    build_condition_transform,
)
from src.metrics import compute_binary_metrics
from src.models import build_model


PREDICTION_COLUMNS = [
    "model_id",
    "checkpoint_hash",
    "image_id",
    "image_path",
    "dataset",
    "source",
    "generator",
    "label",
    "transform",
    "severity",
    "seed",
    "prob_ai",
    "predicted_label",
    "is_correct",
    "latency_ms",
]

METRICS_FILENAMES = {
    "predictions": "clean_predictions.csv",
    "metrics": "clean_metrics.csv",
    "config": "evaluation_config.json",
}

ROBUSTNESS_FILENAMES = {
    "predictions": "robustness_predictions.csv",
    "metrics": "robustness_metrics.csv",
    "comparison": "robustness_comparison.csv",
    "config": "robustness_config.json",
}


def _metadata_value(
    metadata: Mapping[str, Any],
    key: str,
    index: int,
    default: Any = None,
) -> Any:
    """Read one sample's value from metadata collated by a DataLoader."""
    if key not in metadata:
        return default

    value = metadata[key]
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.item() if item.numel() == 1 else item.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[index]
    return value


def _synchronise_if_cuda(device: torch.device) -> None:
    """Wait for queued GPU work so measured inference time is meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_model_condition(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]],
    device: torch.device,
    *,
    model_id: str,
    threshold: float = 0.5,
    checkpoint_hash: str | None = None,
    transform_name: str = "clean",
    severity: int | float | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Evaluate one model on a prepared clean or transformed DataLoader.

    Args:
        model: A model that returns one raw logit for each input image.
        data_loader: Batches of ``(images, labels, metadata)``. Labels use
            ``0 = real`` and ``1 = AI``.
        device: CPU or CUDA device on which inference should run.
        model_id: Human-readable name used to identify this model in results.
        threshold: AI probabilities at or above this value become label 1.
        checkpoint_hash: Optional identifier for the exact model checkpoint.

    Returns:
        A tidy prediction table with one row per image and a dictionary of
        summary metrics calculated across all rows.

    Raises:
        ValueError: If a batch has a different number of logits and labels, or
            if the DataLoader produces no images.
    """
    model = model.to(device)
    model.eval()
    prediction_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for images, labels, metadata in data_loader:
            images = images.to(device)
            label_values = labels.detach().cpu().reshape(-1)

            if not torch.all((label_values == 0) | (label_values == 1)):
                raise ValueError("labels must contain only 0 (real) and 1 (AI)")

            _synchronise_if_cuda(device)
            started_at = perf_counter()
            logits = model(images).reshape(-1)
            _synchronise_if_cuda(device)
            batch_latency_ms = (perf_counter() - started_at) * 1_000

            if logits.numel() != label_values.numel():
                raise ValueError(
                    "model must return exactly one logit for each image label"
                )

            probabilities = torch.sigmoid(logits).detach().cpu()
            predicted_labels = (probabilities >= threshold).to(torch.int64)
            latency_per_image_ms = batch_latency_ms / label_values.numel()

            for index in range(label_values.numel()):
                image_path = _metadata_value(metadata, "image_path", index)
                image_id = _metadata_value(
                    metadata,
                    "image_id",
                    index,
                    default=image_path,
                )

                prediction_rows.append({
                    "model_id": model_id,
                    "checkpoint_hash": checkpoint_hash,
                    "image_id": image_id,
                    "image_path": image_path,
                    "dataset": _metadata_value(metadata, "dataset", index),
                    "source": _metadata_value(metadata, "source", index),
                    "generator": _metadata_value(metadata, "generator", index),
                    "label": int(label_values[index].item()),
                    "transform": transform_name,
                    "severity": severity,
                    "seed": seed,
                    "prob_ai": float(probabilities[index].item()),
                    "predicted_label": int(predicted_labels[index].item()),
                    "is_correct": bool(
                        predicted_labels[index].item()
                        == label_values[index].item()
                    ),
                    "latency_ms": latency_per_image_ms,
                })

    if not prediction_rows:
        raise ValueError("data_loader must produce at least one image")

    prediction_table = pd.DataFrame(
        prediction_rows,
        columns=PREDICTION_COLUMNS,
    )
    metrics = compute_binary_metrics(
        prediction_table["label"].to_numpy(),
        prediction_table["prob_ai"].to_numpy(),
        threshold=threshold,
    )
    return prediction_table, metrics


def evaluate_clean_model(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]],
    device: torch.device,
    *,
    model_id: str,
    threshold: float = 0.5,
    checkpoint_hash: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Evaluate one model on clean images using the shared generic path."""
    return evaluate_model_condition(
        model,
        data_loader,
        device,
        model_id=model_id,
        threshold=threshold,
        checkpoint_hash=checkpoint_hash,
        transform_name="clean",
    )


def checkpoint_sha256(checkpoint_path: str | Path) -> str:
    """Return a stable identifier for the exact checkpoint file."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, str]:
    """Build EfficientNet-B0 and load one raw model state dictionary."""
    path = Path(checkpoint_path)
    checkpoint_hash = checkpoint_sha256(path)

    try:
        checkpoint_data = torch.load(
            path,
            map_location="cpu",
            weights_only=False, # Must be False to load dictionaries/optimizer states
        )
        # Safely pull out model_state if it's a dict, otherwise fallback to old behavior
        state_dict = checkpoint_data.get("model_state", checkpoint_data) if isinstance(checkpoint_data, dict) else checkpoint_data
    except Exception as error:
        raise ValueError(f"could not safely load checkpoint: {path}") from error

    if not isinstance(state_dict, Mapping):
        raise ValueError(
            f"checkpoint must contain a model state dictionary: {path}"
        )

    model = build_model(pretrained=False)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"checkpoint does not match the EfficientNet-B0 detector: {path}"
        ) from error

    return model.to(device), checkpoint_hash


def resolve_device(requested_device: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested_device)


def run_clean_comparison(
    *,
    data_root: str | Path,
    manifest_path: str | Path,
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    output_dir: str | Path,
    split: str = "val",
    model_a_id: str = "experiment_a",
    model_b_id: str = "experiment_b",
    threshold: float = 0.5,
    batch_size: int = 32,
    num_workers: int = 2,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate two named checkpoints fairly on one clean manifest split."""
    if model_a_id == model_b_id:
        raise ValueError("the two checkpoints must have different model IDs")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    evaluation_device = device or resolve_device("auto")
    data_loader = get_evaluation_dataloader(
        data_root=str(data_root),
        manifest_path=str(manifest_path),
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model_specs = [
        (model_a_id, Path(checkpoint_a)),
        (model_b_id, Path(checkpoint_b)),
    ]
    prediction_tables: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, str]] = []
    reference_rows: pd.DataFrame | None = None

    for model_id, checkpoint_path in model_specs:
        model, checkpoint_hash = load_model_checkpoint(
            checkpoint_path,
            evaluation_device,
        )
        predictions, metrics = evaluate_clean_model(
            model,
            data_loader,
            evaluation_device,
            model_id=model_id,
            threshold=threshold,
            checkpoint_hash=checkpoint_hash,
        )

        current_rows = predictions[["image_path", "label"]].reset_index(drop=True)
        if reference_rows is None:
            reference_rows = current_rows
        elif not current_rows.equals(reference_rows):
            raise RuntimeError(
                "models were not evaluated on the same images in the same order"
            )

        prediction_tables.append(predictions)
        metric_rows.append({
            "model_id": model_id,
            "checkpoint_hash": checkpoint_hash,
            "split": split,
            **metrics,
        })
        checkpoint_records.append({
            "model_id": model_id,
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
        })

        del model
        if evaluation_device.type == "cuda":
            torch.cuda.empty_cache()

    combined_predictions = pd.concat(prediction_tables, ignore_index=True)
    metrics_table = pd.DataFrame(metric_rows)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    combined_predictions.to_csv(
        destination / METRICS_FILENAMES["predictions"],
        index=False,
    )
    metrics_table.to_csv(
        destination / METRICS_FILENAMES["metrics"],
        index=False,
    )

    run_config = {
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "split": split,
        "threshold": threshold,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(evaluation_device),
        "preprocessing": "src.data.get_eval_transform",
        "num_images_per_model": int(len(reference_rows)),
        "checkpoints": checkpoint_records,
    }
    with (destination / METRICS_FILENAMES["config"]).open(
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(run_config, config_file, indent=2)
        config_file.write("\n")

    return combined_predictions, metrics_table


def build_robustness_comparison(
    metrics_table: pd.DataFrame,
    *,
    conditions: Sequence[EvaluationCondition],
    model_a_id: str,
    model_b_id: str,
) -> pd.DataFrame:
    """Compare two named models per condition and calculate clean drops."""
    clean_rows = metrics_table[metrics_table["transform"] == "clean"]
    clean_accuracy = clean_rows.set_index("model_id")["accuracy"].to_dict()
    if model_a_id not in clean_accuracy or model_b_id not in clean_accuracy:
        raise ValueError("robustness metrics must include clean rows for both models")

    comparison_rows: list[dict[str, Any]] = []
    for condition in conditions:
        condition_rows = metrics_table[
            metrics_table["transform"] == condition.name
        ]
        if condition.severity is None:
            condition_rows = condition_rows[condition_rows["severity"].isna()]
        else:
            condition_rows = condition_rows[
                condition_rows["severity"] == condition.severity
            ]

        accuracies = condition_rows.set_index("model_id")["accuracy"].to_dict()
        if model_a_id not in accuracies or model_b_id not in accuracies:
            raise ValueError(
                "robustness metrics must include both models for every condition"
            )

        model_a_accuracy = float(accuracies[model_a_id])
        model_b_accuracy = float(accuracies[model_b_id])
        difference = model_b_accuracy - model_a_accuracy
        if abs(difference) < 1e-12:
            better_model = "tie"
        elif difference > 0:
            better_model = model_b_id
        else:
            better_model = model_a_id

        comparison_rows.append({
            "transform": condition.name,
            "severity": condition.severity,
            "model_a_id": model_a_id,
            "model_b_id": model_b_id,
            "model_a_accuracy": model_a_accuracy,
            "model_b_accuracy": model_b_accuracy,
            "b_minus_a_accuracy": difference,
            "model_a_drop_from_clean": (
                float(clean_accuracy[model_a_id]) - model_a_accuracy
            ),
            "model_b_drop_from_clean": (
                float(clean_accuracy[model_b_id]) - model_b_accuracy
            ),
            "better_model": better_model,
        })

    return pd.DataFrame(comparison_rows)


def run_robustness_comparison(
    *,
    data_root: str | Path,
    manifest_path: str | Path,
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    output_dir: str | Path,
    split: str = "val",
    model_a_id: str = "experiment_a",
    model_b_id: str = "experiment_b",
    threshold: float = 0.5,
    batch_size: int = 32,
    num_workers: int = 2,
    device: torch.device | None = None,
    seed: int = 42,
    conditions: Sequence[EvaluationCondition] = ALL_EVALUATION_CONDITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate two named models under all fixed robustness conditions."""
    if model_a_id == model_b_id:
        raise ValueError("the two checkpoints must have different model IDs")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not conditions or not any(item.name == "clean" for item in conditions):
        raise ValueError("robustness evaluation requires a clean condition")

    evaluation_device = device or resolve_device("auto")
    model_specs = [
        (model_a_id, Path(checkpoint_a)),
        (model_b_id, Path(checkpoint_b)),
    ]
    loaded_models: list[tuple[str, nn.Module, str, Path]] = []
    checkpoint_records: list[dict[str, str]] = []

    for model_id, checkpoint_path in model_specs:
        model, checkpoint_hash = load_model_checkpoint(
            checkpoint_path,
            evaluation_device,
        )
        loaded_models.append(
            (model_id, model, checkpoint_hash, checkpoint_path)
        )
        checkpoint_records.append({
            "model_id": model_id,
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
        })

    prediction_tables: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for condition in conditions:
        pre_transform = build_condition_transform(condition, base_seed=seed)
        data_loader = get_evaluation_dataloader(
            data_root=str(data_root),
            manifest_path=str(manifest_path),
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            pre_transform=pre_transform,
        )
        reference_rows: pd.DataFrame | None = None
        condition_seed = seed if condition.name == "noise" else None

        for model_id, model, checkpoint_hash, _ in loaded_models:
            print(
                f"Evaluating {model_id}: "
                f"{condition.name} severity={condition.severity}"
            )
            predictions, metrics = evaluate_model_condition(
                model,
                data_loader,
                evaluation_device,
                model_id=model_id,
                threshold=threshold,
                checkpoint_hash=checkpoint_hash,
                transform_name=condition.name,
                severity=condition.severity,
                seed=condition_seed,
            )

            current_rows = predictions[["image_path", "label"]].reset_index(
                drop=True
            )
            if reference_rows is None:
                reference_rows = current_rows
            elif not current_rows.equals(reference_rows):
                raise RuntimeError(
                    "models were not evaluated on the same images in the same order"
                )

            prediction_tables.append(predictions)
            metric_rows.append({
                "model_id": model_id,
                "checkpoint_hash": checkpoint_hash,
                "split": split,
                "transform": condition.name,
                "severity": condition.severity,
                "seed": condition_seed,
                **metrics,
            })

    loaded_models.clear()
    del model
    if evaluation_device.type == "cuda":
        torch.cuda.empty_cache()

    combined_predictions = pd.concat(prediction_tables, ignore_index=True)
    metrics_table = pd.DataFrame(metric_rows)
    comparison_table = build_robustness_comparison(
        metrics_table,
        conditions=conditions,
        model_a_id=model_a_id,
        model_b_id=model_b_id,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    combined_predictions.to_csv(
        destination / ROBUSTNESS_FILENAMES["predictions"],
        index=False,
    )
    metrics_table.to_csv(
        destination / ROBUSTNESS_FILENAMES["metrics"],
        index=False,
    )
    comparison_table.to_csv(
        destination / ROBUSTNESS_FILENAMES["comparison"],
        index=False,
    )

    run_config = {
        "mode": "robustness",
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "split": split,
        "threshold": threshold,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(evaluation_device),
        "seed": seed,
        "preprocessing": "condition before src.data.get_eval_transform",
        "num_conditions": len(conditions),
        "num_images_per_model_condition": int(metric_rows[0]["num_samples"]),
        "conditions": [
            {"transform": item.name, "severity": item.severity}
            for item in conditions
        ],
        "checkpoints": checkpoint_records,
    }
    with (destination / ROBUSTNESS_FILENAMES["config"]).open(
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(run_config, config_file, indent=2)
        config_file.write("\n")

    return combined_predictions, metrics_table, comparison_table


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for a two-checkpoint comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare two named checkpoints on clean or transformed images. "
            "Use --model-a-id and --model-b-id to label the two result sets."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["clean", "robustness"],
        default="clean",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--model-a-id", default="experiment_a")
    parser.add_argument("--model-b-id", default="experiment_b")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested comparison from command-line arguments."""
    args = build_argument_parser().parse_args(argv)
    device = resolve_device(args.device)
    common_arguments = {
        "data_root": args.data_root,
        "manifest_path": args.manifest,
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "output_dir": args.output_dir,
        "split": args.split,
        "model_a_id": args.model_a_id,
        "model_b_id": args.model_b_id,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": device,
    }

    if args.mode == "clean":
        _, metrics_table = run_clean_comparison(**common_arguments)
        display_table = metrics_table[
            ["model_id", "num_samples", "accuracy", "f1", "auroc"]
        ]
    else:
        _, metrics_table, comparison_table = run_robustness_comparison(
            **common_arguments,
            seed=args.seed,
        )
        display_table = comparison_table[
            [
                "transform",
                "severity",
                "model_a_accuracy",
                "model_b_accuracy",
                "b_minus_a_accuracy",
            ]
        ].rename(columns={
            "model_a_accuracy": f"{args.model_a_id}_accuracy",
            "model_b_accuracy": f"{args.model_b_id}_accuracy",
            "b_minus_a_accuracy": (
                f"{args.model_b_id}_minus_{args.model_a_id}"
            ),
        })

    print(f"Evaluation device: {device}")
    print(
        f"Comparison: {args.model_a_id} (first checkpoint) vs "
        f"{args.model_b_id} (second checkpoint)"
    )
    print(f"Results written to: {Path(args.output_dir)}")
    print(display_table)


if __name__ == "__main__":
    main()
