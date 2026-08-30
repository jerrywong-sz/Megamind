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
from src.evaluation_models import (
    EvaluationModelSpec,
    add_variable_model_arguments,
    build_model_specs,
    comparison_title,
    model_pairs,
    model_titles,
    validate_model_specs,
    variable_model_specs_from_args,
)
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
    "comparison": "clean_pairwise_comparisons.csv",
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
    architecture: str | None = None,
) -> tuple[nn.Module, str]:
    """Load a checkpoint using saved or explicitly supplied architecture."""
    path = Path(checkpoint_path)
    checkpoint_hash = checkpoint_sha256(path)

    try:
        checkpoint_data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise ValueError(
            f"could not safely load checkpoint: {path}"
        ) from error

    if isinstance(checkpoint_data, Mapping):
        checkpoint_architecture = checkpoint_data.get(
            "architecture"
        )
        state_dict = checkpoint_data.get(
            "model_state",
            checkpoint_data,
        )
    else:
        checkpoint_architecture = None
        state_dict = checkpoint_data

    if not isinstance(state_dict, Mapping):
        raise ValueError(
            f"checkpoint must contain a model state dictionary: {path}"
        )

    if (
        checkpoint_architecture is not None
        and architecture is not None
        and checkpoint_architecture != architecture
    ):
        raise ValueError(
            "Checkpoint architecture "
            f"'{checkpoint_architecture}' conflicts with "
            f"requested architecture '{architecture}'."
        )

    resolved_architecture = (
        checkpoint_architecture
        or architecture
        or "efficientnet_b0"
    )

    model = build_model(
        pretrained=False,
        architecture=resolved_architecture,
    )

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as error:
        raise ValueError(
            "checkpoint does not match architecture "
            f"'{resolved_architecture}': {path}"
        ) from error

    return model.to(device), checkpoint_hash


def resolve_device(requested_device: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested_device)


def _metric_names(metrics_table: pd.DataFrame) -> list[str]:
    excluded = {
        "evaluation_title",
        "model_id",
        "model_title",
        "architecture_override",
        "checkpoint_hash",
        "split",
        "transform",
        "severity",
        "seed",
    }
    return [column for column in metrics_table.columns if column not in excluded]


def build_clean_pairwise_comparisons(
    metrics_table: pd.DataFrame,
    *,
    model_specs: Sequence[EvaluationModelSpec],
) -> pd.DataFrame:
    """Build one clean-metric row for every model pair."""
    specs = validate_model_specs(model_specs)
    rows_by_model = {
        str(row["model_id"]): row
        for _, row in metrics_table.iterrows()
    }
    metrics = _metric_names(metrics_table)
    output_rows: list[dict[str, Any]] = []
    evaluation_title = (
        str(metrics_table["evaluation_title"].iloc[0])
        if "evaluation_title" in metrics_table.columns
        else comparison_title("Clean evaluation", specs)
    )
    for reference_spec, candidate_spec in model_pairs(specs):
        reference = rows_by_model[reference_spec.model_id]
        candidate = rows_by_model[candidate_spec.model_id]
        row: dict[str, Any] = {
            "evaluation_title": evaluation_title,
            "reference_model_id": reference_spec.model_id,
            "reference_model_title": reference_spec.model_title,
            "candidate_model_id": candidate_spec.model_id,
            "candidate_model_title": candidate_spec.model_title,
            "difference_direction": (
                f"{candidate_spec.model_id} minus {reference_spec.model_id}"
            ),
        }
        for metric in metrics:
            reference_value = reference[metric]
            candidate_value = candidate[metric]
            row[f"{reference_spec.model_id}__{metric}"] = reference_value
            row[f"{candidate_spec.model_id}__{metric}"] = candidate_value
            if metric not in {"num_samples", "threshold"}:
                row[
                    f"{candidate_spec.model_id}_minus_"
                    f"{reference_spec.model_id}__{metric}"
                ] = float(candidate_value) - float(reference_value)
        accuracy_difference = float(candidate["accuracy"]) - float(
            reference["accuracy"]
        )
        row["higher_accuracy_model"] = (
            "tie"
            if abs(accuracy_difference) < 1e-12
            else (
                candidate_spec.model_id
                if accuracy_difference > 0
                else reference_spec.model_id
            )
        )
        output_rows.append(row)
    return pd.DataFrame(output_rows)


def run_clean_model_specs(
    *,
    data_root: str | Path,
    manifest_path: str | Path,
    model_specs: Sequence[EvaluationModelSpec],
    output_dir: str | Path,
    split: str = "val",
    threshold: float = 0.5,
    batch_size: int = 32,
    num_workers: int = 2,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate two or more named checkpoints on one clean split."""
    specs = validate_model_specs(model_specs)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    evaluation_device = device or resolve_device("auto")
    evaluation_title = comparison_title("Clean evaluation", specs)
    data_loader = get_evaluation_dataloader(
        data_root=str(data_root),
        manifest_path=str(manifest_path),
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    prediction_tables: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    reference_rows: pd.DataFrame | None = None

    for spec in specs:
        model, checkpoint_hash = load_model_checkpoint(
            spec.checkpoint_path,
            evaluation_device,
            architecture=spec.architecture,
        )
        print(f"Clean evaluation: {spec.model_title}")
        predictions, metrics = evaluate_clean_model(
            model,
            data_loader,
            evaluation_device,
            model_id=spec.model_id,
            threshold=threshold,
            checkpoint_hash=checkpoint_hash,
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

        predictions["model_title"] = spec.model_title
        predictions["architecture_override"] = spec.architecture or "auto"
        predictions["evaluation_title"] = evaluation_title
        prediction_tables.append(predictions)
        metric_rows.append({
            "model_id": spec.model_id,
            "model_title": spec.model_title,
            "architecture_override": spec.architecture or "auto",
            "evaluation_title": evaluation_title,
            "checkpoint_hash": checkpoint_hash,
            "split": split,
            **metrics,
        })
        checkpoint_records.append({
            **spec.as_record(),
            "sha256": checkpoint_hash,
        })
        del model
        if evaluation_device.type == "cuda":
            torch.cuda.empty_cache()

    assert reference_rows is not None
    combined_predictions = pd.concat(prediction_tables, ignore_index=True)
    metrics_table = pd.DataFrame(metric_rows)
    comparison_table = build_clean_pairwise_comparisons(
        metrics_table,
        model_specs=specs,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    combined_predictions.to_csv(
        destination / METRICS_FILENAMES["predictions"], index=False
    )
    metrics_table.to_csv(
        destination / METRICS_FILENAMES["metrics"], index=False
    )
    comparison_table.to_csv(
        destination / METRICS_FILENAMES["comparison"], index=False
    )
    run_config = {
        "mode": "clean",
        "evaluation_title": evaluation_title,
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "split": split,
        "threshold": threshold,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(evaluation_device),
        "preprocessing": "src.data.get_eval_transform",
        "num_models": len(specs),
        "num_images_per_model": int(len(reference_rows)),
        "models": checkpoint_records,
        "checkpoints": checkpoint_records,
        "output_files": METRICS_FILENAMES,
    }
    with (destination / METRICS_FILENAMES["config"]).open(
        "w", encoding="utf-8"
    ) as config_file:
        json.dump(run_config, config_file, indent=2, ensure_ascii=False)
        config_file.write("\n")
    return combined_predictions, metrics_table


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
    architecture_a: str | None = None,
    architecture_b: str | None = None,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate two named checkpoints fairly on one clean manifest split."""
    specs = build_model_specs(
        checkpoints=(checkpoint_a, checkpoint_b),
        model_ids=(model_a_id, model_b_id),
        model_titles=(model_a_id, model_b_id),
        architectures=(architecture_a, architecture_b),
    )
    return run_clean_model_specs(
        data_root=data_root,
        manifest_path=manifest_path,
        model_specs=specs,
        output_dir=output_dir,
        split=split,
        threshold=threshold,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


def build_robustness_comparison(
    metrics_table: pd.DataFrame,
    *,
    conditions: Sequence[EvaluationCondition],
    model_a_id: str,
    model_b_id: str,
    model_a_title: str | None = None,
    model_b_title: str | None = None,
    evaluation_title: str | None = None,
) -> pd.DataFrame:
    """Compare two named models per condition and calculate clean drops.

    The original ``model_a_*``/``model_b_*`` columns remain for compatibility.
    Clear reference/candidate columns and model-ID-qualified metric columns make
    new comparisons self-describing even when neither model is Experiment A.
    """
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

        rows_by_model = {
            str(row["model_id"]): row
            for _, row in condition_rows.iterrows()
        }
        model_a_accuracy = float(accuracies[model_a_id])
        model_b_accuracy = float(accuracies[model_b_id])
        difference = model_b_accuracy - model_a_accuracy
        if abs(difference) < 1e-12:
            better_model = "tie"
        elif difference > 0:
            better_model = model_b_id
        else:
            better_model = model_a_id

        comparison_row: dict[str, Any] = {
            "evaluation_title": evaluation_title,
            "transform": condition.name,
            "severity": condition.severity,
            "reference_model_id": model_a_id,
            "reference_model_title": model_a_title or model_a_id,
            "candidate_model_id": model_b_id,
            "candidate_model_title": model_b_title or model_b_id,
            "difference_direction": f"{model_b_id} minus {model_a_id}",
            "reference_accuracy": model_a_accuracy,
            "candidate_accuracy": model_b_accuracy,
            "candidate_minus_reference_accuracy": difference,
            "reference_drop_from_clean": (
                float(clean_accuracy[model_a_id]) - model_a_accuracy
            ),
            "candidate_drop_from_clean": (
                float(clean_accuracy[model_b_id]) - model_b_accuracy
            ),
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
            "higher_accuracy_model": better_model,
        }
        reference = rows_by_model[model_a_id]
        candidate = rows_by_model[model_b_id]
        for metric in _metric_names(metrics_table):
            reference_value = reference[metric]
            candidate_value = candidate[metric]
            comparison_row[f"{model_a_id}__{metric}"] = reference_value
            comparison_row[f"{model_b_id}__{metric}"] = candidate_value
            if metric not in {"num_samples", "threshold"}:
                comparison_row[
                    f"{model_b_id}_minus_{model_a_id}__{metric}"
                ] = float(candidate_value) - float(reference_value)
        comparison_rows.append(comparison_row)

    return pd.DataFrame(comparison_rows)


def run_robustness_model_specs(
    *,
    data_root: str | Path,
    manifest_path: str | Path,
    model_specs: Sequence[EvaluationModelSpec],
    output_dir: str | Path,
    split: str = "val",
    threshold: float = 0.5,
    batch_size: int = 32,
    num_workers: int = 2,
    device: torch.device | None = None,
    seed: int = 42,
    conditions: Sequence[EvaluationCondition] = ALL_EVALUATION_CONDITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate two or more named models under single-transform conditions."""
    specs = validate_model_specs(model_specs)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not conditions or not any(item.name == "clean" for item in conditions):
        raise ValueError("robustness evaluation requires a clean condition")

    evaluation_device = device or resolve_device("auto")
    evaluation_title = comparison_title(
        "Single-transform robustness evaluation",
        specs,
    )
    loaded_models: list[
        tuple[EvaluationModelSpec, nn.Module, str]
    ] = []
    checkpoint_records: list[dict[str, Any]] = []
    for spec in specs:
        model, checkpoint_hash = load_model_checkpoint(
            spec.checkpoint_path,
            evaluation_device,
            architecture=spec.architecture,
        )
        loaded_models.append((spec, model, checkpoint_hash))
        checkpoint_records.append({
            **spec.as_record(),
            "path": str(spec.checkpoint_path),
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
        for spec, model, checkpoint_hash in loaded_models:
            print(
                f"Evaluating {spec.model_title}: "
                f"{condition.name} severity={condition.severity}"
            )
            predictions, metrics = evaluate_model_condition(
                model,
                data_loader,
                evaluation_device,
                model_id=spec.model_id,
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
                    "models were not evaluated on the same images in the "
                    "same order"
                )
            predictions["model_title"] = spec.model_title
            predictions["architecture_override"] = spec.architecture or "auto"
            predictions["evaluation_title"] = evaluation_title
            prediction_tables.append(predictions)
            metric_rows.append({
                "model_id": spec.model_id,
                "model_title": spec.model_title,
                "architecture_override": spec.architecture or "auto",
                "evaluation_title": evaluation_title,
                "checkpoint_hash": checkpoint_hash,
                "split": split,
                "transform": condition.name,
                "severity": condition.severity,
                "seed": condition_seed,
                **metrics,
            })

    loaded_models.clear()
    if evaluation_device.type == "cuda":
        torch.cuda.empty_cache()

    combined_predictions = pd.concat(prediction_tables, ignore_index=True)
    metrics_table = pd.DataFrame(metric_rows)
    title_by_id = model_titles(specs)
    comparison_tables = []
    for reference_spec, candidate_spec in model_pairs(specs):
        table = build_robustness_comparison(
            metrics_table,
            conditions=conditions,
            model_a_id=reference_spec.model_id,
            model_b_id=candidate_spec.model_id,
            model_a_title=reference_spec.model_title,
            model_b_title=candidate_spec.model_title,
            evaluation_title=evaluation_title,
        )
        table["model_a_title"] = title_by_id[reference_spec.model_id]
        table["model_b_title"] = title_by_id[candidate_spec.model_id]
        comparison_tables.append(table)
    comparison_table = pd.concat(comparison_tables, ignore_index=True)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    combined_predictions.to_csv(
        destination / ROBUSTNESS_FILENAMES["predictions"], index=False
    )
    metrics_table.to_csv(
        destination / ROBUSTNESS_FILENAMES["metrics"], index=False
    )
    comparison_table.to_csv(
        destination / ROBUSTNESS_FILENAMES["comparison"], index=False
    )
    run_config = {
        "mode": "robustness",
        "evaluation_title": evaluation_title,
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "split": split,
        "threshold": threshold,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(evaluation_device),
        "seed": seed,
        "preprocessing": "condition before src.data.get_eval_transform",
        "num_models": len(specs),
        "num_conditions": len(conditions),
        "num_images_per_model_condition": int(metric_rows[0]["num_samples"]),
        "conditions": [
            {"transform": item.name, "severity": item.severity}
            for item in conditions
        ],
        "models": checkpoint_records,
        "checkpoints": checkpoint_records,
        "output_files": ROBUSTNESS_FILENAMES,
    }
    with (destination / ROBUSTNESS_FILENAMES["config"]).open(
        "w", encoding="utf-8"
    ) as config_file:
        json.dump(run_config, config_file, indent=2, ensure_ascii=False)
        config_file.write("\n")
    return combined_predictions, metrics_table, comparison_table


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
    architecture_a: str | None = None,
    architecture_b: str | None = None,
    device: torch.device | None = None,
    seed: int = 42,
    conditions: Sequence[EvaluationCondition] = ALL_EVALUATION_CONDITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate two named models under all fixed robustness conditions."""
    specs = build_model_specs(
        checkpoints=(checkpoint_a, checkpoint_b),
        model_ids=(model_a_id, model_b_id),
        model_titles=(model_a_id, model_b_id),
        architectures=(architecture_a, architecture_b),
    )
    return run_robustness_model_specs(
        data_root=data_root,
        manifest_path=manifest_path,
        model_specs=specs,
        output_dir=output_dir,
        split=split,
        threshold=threshold,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
        conditions=conditions,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for a 2+ checkpoint comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare two or more named checkpoints on clean or transformed "
            "images. Use the plural model arguments for variable-length runs."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["clean", "robustness"],
        default="clean",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-a")
    parser.add_argument("--checkpoint-b")
    parser.add_argument(
        "--architecture-a",
        default=None,
        help=(
            "Architecture override for checkpoint A "
            "when checkpoint metadata is unavailable."
        ),
    )
    parser.add_argument(
        "--architecture-b",
        default=None,
        help=(
            "Architecture override for checkpoint B "
            "when checkpoint metadata is unavailable."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--model-a-id", default="experiment_a")
    parser.add_argument("--model-b-id", default="experiment_b")
    add_variable_model_arguments(parser)
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
    try:
        specs = variable_model_specs_from_args(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if specs is None:
        if args.checkpoint_a is None or args.checkpoint_b is None:
            raise SystemExit(
                "provide legacy --checkpoint-a/--checkpoint-b or use "
                "--checkpoints with --model-ids"
            )
        specs = build_model_specs(
            checkpoints=(args.checkpoint_a, args.checkpoint_b),
            model_ids=(args.model_a_id, args.model_b_id),
            model_titles=(args.model_a_id, args.model_b_id),
            architectures=(args.architecture_a, args.architecture_b),
        )

    common_arguments = {
        "data_root": args.data_root,
        "manifest_path": args.manifest,
        "model_specs": specs,
        "output_dir": args.output_dir,
        "split": args.split,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": device,
    }

    if args.mode == "clean":
        _, metrics_table = run_clean_model_specs(**common_arguments)
        display_table = metrics_table[
            [
                "model_id",
                "model_title",
                "num_samples",
                "accuracy",
                "f1",
                "auroc",
            ]
        ]
    else:
        _, metrics_table, comparison_table = run_robustness_model_specs(
            **common_arguments,
            seed=args.seed,
        )
        display_table = comparison_table[
            [
                "transform",
                "severity",
                "model_a_id",
                "model_b_id",
                "model_a_accuracy",
                "model_b_accuracy",
                "b_minus_a_accuracy",
            ]
        ]

    print(f"Evaluation device: {device}")
    print("Models: " + " vs ".join(spec.model_title for spec in specs))
    print(f"Results written to: {Path(args.output_dir)}")
    print(display_table)


if __name__ == "__main__":
    main()
