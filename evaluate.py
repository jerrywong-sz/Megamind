"""Evaluate a model on clean images and build a traceable prediction table.

This module owns the steps that happen after a DataLoader supplies image
batches: run the model, convert logits to AI probabilities, record one row per
image, and calculate summary metrics. Robustness transforms will be added as a
separate layer after this clean-image path is established.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from time import perf_counter
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from src.metrics import compute_binary_metrics


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
    "latency_ms",
]


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
