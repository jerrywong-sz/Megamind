"""Evaluate trained checkpoints on clean images.

This module owns both the reusable per-model evaluation function and the
command-line runner that connects real checkpoints to a manifest-backed image
split. Robustness transforms will be added as a separate layer after the clean
checkpoint path is verified against the team's reported validation results.
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


def evaluate_clean_model(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]],
    device: torch.device,
    *,
    model_id: str,
    threshold: float = 0.5,
    checkpoint_hash: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Evaluate one model on the clean condition.

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
                    "transform": "clean",
                    "severity": None,
                    "seed": None,
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
        state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
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
    """Evaluate checkpoints A and B fairly on one clean manifest split."""
    if model_a_id == model_b_id:
        raise ValueError("model A and model B must have different model IDs")
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


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for clean A-vs-B evaluation."""
    parser = argparse.ArgumentParser(
        description="Compare checkpoints A and B on a clean validation/test split.",
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
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the clean comparison from command-line arguments."""
    args = build_argument_parser().parse_args(argv)
    device = resolve_device(args.device)
    _, metrics_table = run_clean_comparison(
        data_root=args.data_root,
        manifest_path=args.manifest,
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        output_dir=args.output_dir,
        split=args.split,
        model_a_id=args.model_a_id,
        model_b_id=args.model_b_id,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    print(f"Evaluation device: {device}")
    print(f"Results written to: {Path(args.output_dir)}")
    print(metrics_table[["model_id", "num_samples", "accuracy", "f1", "auroc"]])


if __name__ == "__main__":
    main()
