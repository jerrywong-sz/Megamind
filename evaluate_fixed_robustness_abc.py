"""Evaluate two or more named models under fixed robustness conditions.

The module name and legacy A/B/C interface remain for saved notebooks. New
runs use aligned model lists and produce every pairwise report, one combined
summary, full metrics, per-image predictions, and a reproducibility config.
Randomly sampled transform chains are outside this runner's scope.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
import torch.nn as nn

from evaluate import (
    PREDICTION_COLUMNS,
    evaluate_model_condition,
    load_model_checkpoint,
    resolve_device,
)
from src.data import get_evaluation_dataloader
from src.evaluation_conditions import (
    ALL_FIXED_EVALUATION_CONDITIONS,
    FIXED_CHAIN_EVALUATION_CONDITIONS,
    EvaluationCondition,
    build_condition_transform,
    condition_id,
    condition_steps,
    condition_title,
)
from src.evaluation_models import (
    EvaluationModelSpec,
    add_variable_model_arguments,
    build_model_specs,
    comparison_title,
    model_pairs,
    model_token,
    model_titles,
    validate_model_specs,
    variable_model_specs_from_args,
)


MODEL_A_ID = "model_a_clean_baseline"
MODEL_B_ID = "model_b_robustness"
MODEL_C_ID = "model_c_consistency"

MODEL_TITLES = {
    MODEL_A_ID: "Model A — clean baseline training",
    MODEL_B_ID: "Model B — robustness augmentation training",
    MODEL_C_ID: "Model C — robustness augmentation + consistency training",
}

EVALUATION_TITLE = (
    "Fixed robustness evaluation — Model A baseline vs Model B robustness "
    "vs Model C consistency"
)

FIXED_ABC_FILENAMES = {
    "predictions": (
        "fixed_robustness__models_A_B_C__per_image_predictions.csv"
    ),
    "metrics": "fixed_robustness__models_A_B_C__full_metrics.csv",
    "a_vs_b": (
        "fixed_robustness__model_A_baseline_vs_model_B_robustness.csv"
    ),
    "b_vs_c": (
        "fixed_robustness__model_B_robustness_vs_model_C_consistency.csv"
    ),
    "a_vs_c": (
        "fixed_robustness__model_A_baseline_vs_model_C_consistency.csv"
    ),
    "abc": "fixed_robustness__models_A_B_C__combined_summary.csv",
    "config": "fixed_robustness__models_A_B_C__run_config.json",
}

REPORTED_METRICS = (
    "num_samples",
    "threshold",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "false_positive_rate",
    "false_negative_rate",
    "brier_score",
)

DIFFERENCE_METRICS = tuple(
    metric
    for metric in REPORTED_METRICS
    if metric not in {"num_samples", "threshold"}
)

PARAMETER_NAMES = {
    "jpeg": "quality",
    "blur": "sigma",
    "resize": "scale",
    "noise": "sigma",
    "colour": "strength",
    "crop": "fraction",
}


def _step_records(condition: EvaluationCondition) -> list[dict[str, Any]]:
    """Describe every transform parameter in execution order."""
    return [
        {
            "order": index,
            "transform": transform_name,
            "parameter_name": PARAMETER_NAMES[transform_name],
            "parameter_value": parameter,
        }
        for index, (transform_name, parameter) in enumerate(
            condition_steps(condition),
            start=1,
        )
    ]


def condition_metadata(condition: EvaluationCondition) -> dict[str, Any]:
    """Return explicit, spreadsheet-friendly metadata for one condition."""
    steps = _step_records(condition)
    if steps:
        transform_chain = " -> ".join(
            (
                f"{step['transform']}("
                f"{step['parameter_name']}={step['parameter_value']})"
            )
            for step in steps
        )
    else:
        transform_chain = "none"

    return {
        "condition_id": condition_id(condition),
        "condition_title": condition_title(condition),
        "condition_kind": condition.condition_kind,
        "num_transform_steps": len(steps),
        "transform_chain": transform_chain,
        "transform_parameters_json": json.dumps(
            steps,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "possible_in_b_c_training_sampler": (
            condition.possible_in_b_c_training_sampler
        ),
    }


def _winner(
    values: dict[str, float],
    *,
    higher_is_better: bool,
    tolerance: float = 1e-12,
) -> str:
    """Return a model ID or ``tie`` for one metric."""
    best_value = (
        max(values.values()) if higher_is_better else min(values.values())
    )
    winners = [
        model_id
        for model_id, value in values.items()
        if abs(value - best_value) < tolerance
    ]
    return winners[0] if len(winners) == 1 else "tie"


def _metric_row_by_model(
    metrics_table: pd.DataFrame,
    condition: EvaluationCondition,
) -> dict[str, pd.Series]:
    """Index a condition's metrics by explicit model ID."""
    condition_rows = metrics_table[
        metrics_table["condition_id"] == condition_id(condition)
    ]
    return {
        str(row["model_id"]): row
        for _, row in condition_rows.iterrows()
    }


def _clean_metric_rows(metrics_table: pd.DataFrame) -> dict[str, pd.Series]:
    clean_rows = metrics_table[metrics_table["condition_id"] == "clean"]
    return {
        str(row["model_id"]): row
        for _, row in clean_rows.iterrows()
    }


def _add_clean_changes(
    output_row: dict[str, Any],
    *,
    model_id: str,
    current: pd.Series,
    clean: pd.Series,
) -> None:
    """Add clearly directed changes relative to the model's clean result."""
    output_row[f"{model_id}__accuracy_drop_from_clean"] = (
        float(clean["accuracy"]) - float(current["accuracy"])
    )
    output_row[f"{model_id}__auroc_drop_from_clean"] = (
        float(clean["auroc"]) - float(current["auroc"])
    )
    output_row[f"{model_id}__auprc_drop_from_clean"] = (
        float(clean["auprc"]) - float(current["auprc"])
    )
    output_row[f"{model_id}__f1_drop_from_clean"] = (
        float(clean["f1"]) - float(current["f1"])
    )
    output_row[f"{model_id}__false_positive_rate_increase_from_clean"] = (
        float(current["false_positive_rate"])
        - float(clean["false_positive_rate"])
    )
    output_row[f"{model_id}__false_negative_rate_increase_from_clean"] = (
        float(current["false_negative_rate"])
        - float(clean["false_negative_rate"])
    )
    output_row[f"{model_id}__brier_score_increase_from_clean"] = (
        float(current["brier_score"]) - float(clean["brier_score"])
    )


def build_explicit_pairwise_comparison(
    metrics_table: pd.DataFrame,
    *,
    conditions: Sequence[EvaluationCondition],
    reference_model_id: str,
    candidate_model_id: str,
    model_titles: dict[str, str] = MODEL_TITLES,
) -> pd.DataFrame:
    """Build one fully named comparison without generic model-A/B columns."""
    if reference_model_id == candidate_model_id:
        raise ValueError("pairwise comparison requires two different models")

    clean_rows = _clean_metric_rows(metrics_table)
    required_ids = {reference_model_id, candidate_model_id}
    if not required_ids.issubset(clean_rows):
        raise ValueError("metrics must contain clean rows for both compared models")

    report_title = (
        f"Fixed robustness comparison — {model_titles[reference_model_id]} "
        f"vs {model_titles[candidate_model_id]}"
    )
    comparison_rows: list[dict[str, Any]] = []

    for condition in conditions:
        rows_by_model = _metric_row_by_model(metrics_table, condition)
        if not required_ids.issubset(rows_by_model):
            raise ValueError(
                f"condition '{condition_id(condition)}' is missing one "
                "compared model"
            )

        reference = rows_by_model[reference_model_id]
        candidate = rows_by_model[candidate_model_id]
        output_row: dict[str, Any] = {
            "report_title": report_title,
            **condition_metadata(condition),
            "reference_model_id": reference_model_id,
            "reference_model_title": model_titles[reference_model_id],
            "candidate_model_id": candidate_model_id,
            "candidate_model_title": model_titles[candidate_model_id],
            "difference_direction": (
                f"{candidate_model_id} minus {reference_model_id}"
            ),
        }

        for metric in REPORTED_METRICS:
            reference_value = reference[metric]
            candidate_value = candidate[metric]
            output_row[f"{reference_model_id}__{metric}"] = reference_value
            output_row[f"{candidate_model_id}__{metric}"] = candidate_value
            if metric in DIFFERENCE_METRICS:
                output_row[
                    f"{candidate_model_id}_minus_{reference_model_id}__{metric}"
                ] = float(candidate_value) - float(reference_value)

        _add_clean_changes(
            output_row,
            model_id=reference_model_id,
            current=reference,
            clean=clean_rows[reference_model_id],
        )
        _add_clean_changes(
            output_row,
            model_id=candidate_model_id,
            current=candidate,
            clean=clean_rows[candidate_model_id],
        )

        pair_rows = {
            reference_model_id: reference,
            candidate_model_id: candidate,
        }
        output_row["higher_accuracy_model"] = _winner(
            {
                model_id: float(row["accuracy"])
                for model_id, row in pair_rows.items()
            },
            higher_is_better=True,
        )
        output_row["higher_auroc_model"] = _winner(
            {
                model_id: float(row["auroc"])
                for model_id, row in pair_rows.items()
            },
            higher_is_better=True,
        )
        output_row["higher_auprc_model"] = _winner(
            {
                model_id: float(row["auprc"])
                for model_id, row in pair_rows.items()
            },
            higher_is_better=True,
        )
        output_row["lower_false_positive_rate_model"] = _winner(
            {
                model_id: float(row["false_positive_rate"])
                for model_id, row in pair_rows.items()
            },
            higher_is_better=False,
        )
        output_row["lower_false_negative_rate_model"] = _winner(
            {
                model_id: float(row["false_negative_rate"])
                for model_id, row in pair_rows.items()
            },
            higher_is_better=False,
        )
        output_row["lower_brier_score_model"] = _winner(
            {
                model_id: float(row["brier_score"])
                for model_id, row in pair_rows.items()
            },
            higher_is_better=False,
        )
        comparison_rows.append(output_row)

    return pd.DataFrame(comparison_rows)


def build_combined_abc_summary(
    metrics_table: pd.DataFrame,
    *,
    conditions: Sequence[EvaluationCondition],
    model_titles: dict[str, str] = MODEL_TITLES,
    model_ids: Sequence[str] = (MODEL_A_ID, MODEL_B_ID, MODEL_C_ID),
    evaluation_title: str = EVALUATION_TITLE,
) -> pd.DataFrame:
    """Place all supplied model metrics and winners in one named table."""
    model_ids = tuple(model_ids)
    if len(model_ids) < 2:
        raise ValueError("combined summary requires at least two models")
    clean_rows = _clean_metric_rows(metrics_table)
    if not set(model_ids).issubset(clean_rows):
        raise ValueError("metrics must contain clean rows for every model")

    summary_rows: list[dict[str, Any]] = []
    for condition in conditions:
        rows_by_model = _metric_row_by_model(metrics_table, condition)
        if not set(model_ids).issubset(rows_by_model):
            raise ValueError(
                f"condition '{condition_id(condition)}' is missing a model"
            )

        output_row: dict[str, Any] = {
            "report_title": evaluation_title,
            **condition_metadata(condition),
        }
        for model_id in model_ids:
            output_row[f"{model_id}__title"] = model_titles[model_id]
            current = rows_by_model[model_id]
            for metric in REPORTED_METRICS:
                output_row[f"{model_id}__{metric}"] = current[metric]
            _add_clean_changes(
                output_row,
                model_id=model_id,
                current=current,
                clean=clean_rows[model_id],
            )

        for reference_id, candidate_id in combinations(model_ids, 2):
            for metric in (
                "accuracy",
                "auroc",
                "auprc",
                "false_positive_rate",
                "false_negative_rate",
                "brier_score",
            ):
                output_row[
                    f"{candidate_id}_minus_{reference_id}__{metric}"
                ] = float(rows_by_model[candidate_id][metric]) - float(
                    rows_by_model[reference_id][metric]
                )

        output_row["highest_accuracy_model"] = _winner(
            {
                model_id: float(rows_by_model[model_id]["accuracy"])
                for model_id in model_ids
            },
            higher_is_better=True,
        )
        output_row["highest_auroc_model"] = _winner(
            {
                model_id: float(rows_by_model[model_id]["auroc"])
                for model_id in model_ids
            },
            higher_is_better=True,
        )
        output_row["highest_auprc_model"] = _winner(
            {
                model_id: float(rows_by_model[model_id]["auprc"])
                for model_id in model_ids
            },
            higher_is_better=True,
        )
        output_row["lowest_false_positive_rate_model"] = _winner(
            {
                model_id: float(
                    rows_by_model[model_id]["false_positive_rate"]
                )
                for model_id in model_ids
            },
            higher_is_better=False,
        )
        output_row["lowest_false_negative_rate_model"] = _winner(
            {
                model_id: float(
                    rows_by_model[model_id]["false_negative_rate"]
                )
                for model_id in model_ids
            },
            higher_is_better=False,
        )
        output_row["lowest_brier_score_model"] = _winner(
            {
                model_id: float(rows_by_model[model_id]["brier_score"])
                for model_id in model_ids
            },
            higher_is_better=False,
        )
        accuracies = [
            float(rows_by_model[model_id]["accuracy"])
            for model_id in model_ids
        ]
        output_row["accuracy_spread_best_minus_worst"] = (
            max(accuracies) - min(accuracies)
        )
        summary_rows.append(output_row)

    return pd.DataFrame(summary_rows)


def _prediction_output_columns(predictions: pd.DataFrame) -> list[str]:
    metadata_columns = [
        "evaluation_title",
        "model_title",
        "architecture_override",
        "condition_id",
        "condition_title",
        "condition_kind",
        "transform",
        "severity",
        "num_transform_steps",
        "transform_chain",
        "transform_parameters_json",
        "possible_in_b_c_training_sampler",
    ]
    ordered = [
        "evaluation_title",
        "model_id",
        "model_title",
        "architecture_override",
        "checkpoint_hash",
        "image_id",
        "image_path",
        "dataset",
        "source",
        "generator",
        "label",
        "condition_id",
        "condition_title",
        "condition_kind",
        "transform",
        "severity",
        "num_transform_steps",
        "transform_chain",
        "transform_parameters_json",
        "possible_in_b_c_training_sampler",
        "seed",
        "prob_ai",
        "predicted_label",
        "is_correct",
        "latency_ms",
    ]
    expected = set(PREDICTION_COLUMNS) | set(metadata_columns)
    if set(predictions.columns) != expected:
        raise RuntimeError("unexpected columns in fixed prediction table")
    return ordered


def _build_model_specs(
    checkpoint_a_baseline: str | Path,
    checkpoint_b_robustness: str | Path,
    checkpoint_c_consistency: str | Path,
) -> tuple[EvaluationModelSpec, ...]:
    return build_model_specs(
        checkpoints=(
            checkpoint_a_baseline,
            checkpoint_b_robustness,
            checkpoint_c_consistency,
        ),
        model_ids=(MODEL_A_ID, MODEL_B_ID, MODEL_C_ID),
        model_titles=(
            MODEL_TITLES[MODEL_A_ID],
            MODEL_TITLES[MODEL_B_ID],
            MODEL_TITLES[MODEL_C_ID],
        ),
    )


def _fixed_output_filenames(
    specs: Sequence[EvaluationModelSpec],
) -> dict[str, str]:
    model_ids = tuple(spec.model_id for spec in specs)
    if model_ids == (MODEL_A_ID, MODEL_B_ID, MODEL_C_ID):
        return dict(FIXED_ABC_FILENAMES)
    token = model_token(specs)
    filenames = {
        "predictions": f"fixed_robustness__{token}__per_image_predictions.csv",
        "metrics": f"fixed_robustness__{token}__full_metrics.csv",
        "summary": f"fixed_robustness__{token}__combined_summary.csv",
        "config": f"fixed_robustness__{token}__run_config.json",
    }
    for reference_spec, candidate_spec in model_pairs(specs):
        pair_key = f"{reference_spec.model_id}_vs_{candidate_spec.model_id}"
        filenames[pair_key] = f"fixed_robustness__{pair_key}.csv"
    return filenames


def run_fixed_robustness_evaluation(
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
    conditions: Sequence[
        EvaluationCondition
    ] = ALL_FIXED_EVALUATION_CONDITIONS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """Run one fair fixed-condition evaluation over two or more models."""
    specs = validate_model_specs(model_specs)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not conditions or not any(item.name == "clean" for item in conditions):
        raise ValueError("fixed evaluation requires a clean condition")
    condition_ids = [condition_id(item) for item in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("fixed condition IDs must be unique")

    evaluation_device = device or resolve_device("auto")
    evaluation_title = (
        EVALUATION_TITLE
        if tuple(spec.model_id for spec in specs)
        == (MODEL_A_ID, MODEL_B_ID, MODEL_C_ID)
        else comparison_title("Fixed robustness evaluation", specs)
    )
    title_by_id = model_titles(specs)
    output_filenames = _fixed_output_filenames(specs)
    loaded_models: list[
        tuple[EvaluationModelSpec, nn.Module, str]
    ] = []
    checkpoint_records: list[dict[str, Any]] = []

    for spec in specs:
        if spec.architecture is None:
            model, checkpoint_hash = load_model_checkpoint(
                spec.checkpoint_path,
                evaluation_device,
            )
        else:
            model, checkpoint_hash = load_model_checkpoint(
                spec.checkpoint_path,
                evaluation_device,
                architecture=spec.architecture,
            )
        loaded_models.append((spec, model, checkpoint_hash))
        checkpoint_records.append({
            "model_id": spec.model_id,
            "model_title": spec.model_title,
            "architecture_override": spec.architecture or "auto",
            "path": str(spec.checkpoint_path),
            "sha256": checkpoint_hash,
        })

    prediction_tables: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for condition in conditions:
        metadata = condition_metadata(condition)
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

        for spec, model, checkpoint_hash in loaded_models:
            print(f"Evaluating {spec.model_title}")
            print(f"Condition: {metadata['condition_title']}")
            predictions, metrics = evaluate_model_condition(
                model,
                data_loader,
                evaluation_device,
                model_id=spec.model_id,
                threshold=threshold,
                checkpoint_hash=checkpoint_hash,
                transform_name=condition.name,
                severity=condition.severity,
                seed=seed,
            )

            current_rows = predictions[["image_path", "label"]].reset_index(
                drop=True
            )
            if reference_rows is None:
                reference_rows = current_rows
            elif not current_rows.equals(reference_rows):
                raise RuntimeError(
                    "configured models were not evaluated on identical rows"
                )

            predictions["evaluation_title"] = evaluation_title
            predictions["model_title"] = spec.model_title
            predictions["architecture_override"] = (
                spec.architecture or "auto"
            )
            for key, value in metadata.items():
                predictions[key] = value
            predictions = predictions[
                _prediction_output_columns(predictions)
            ]
            prediction_tables.append(predictions)

            metric_rows.append({
                "report_title": evaluation_title,
                "model_id": spec.model_id,
                "model_title": spec.model_title,
                "architecture_override": spec.architecture or "auto",
                "checkpoint_hash": checkpoint_hash,
                "split": split,
                "run_seed": seed,
                **metadata,
                **metrics,
            })

    loaded_models.clear()
    if evaluation_device.type == "cuda":
        torch.cuda.empty_cache()

    predictions_table = pd.concat(prediction_tables, ignore_index=True)
    metrics_table = pd.DataFrame(metric_rows)
    legacy_pair_keys = {
        (MODEL_A_ID, MODEL_B_ID): "a_vs_b",
        (MODEL_B_ID, MODEL_C_ID): "b_vs_c",
        (MODEL_A_ID, MODEL_C_ID): "a_vs_c",
    }
    pairwise_tables: dict[str, pd.DataFrame] = {}
    for reference_spec, candidate_spec in model_pairs(specs):
        pair_key = legacy_pair_keys.get(
            (reference_spec.model_id, candidate_spec.model_id),
            f"{reference_spec.model_id}_vs_{candidate_spec.model_id}",
        )
        pairwise_tables[pair_key] = build_explicit_pairwise_comparison(
            metrics_table,
            conditions=conditions,
            reference_model_id=reference_spec.model_id,
            candidate_model_id=candidate_spec.model_id,
            model_titles=title_by_id,
        )
    combined_summary = build_combined_abc_summary(
        metrics_table,
        conditions=conditions,
        model_titles=title_by_id,
        model_ids=[spec.model_id for spec in specs],
        evaluation_title=evaluation_title,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions_table.to_csv(
        destination / output_filenames["predictions"],
        index=False,
    )
    metrics_table.to_csv(
        destination / output_filenames["metrics"],
        index=False,
    )
    for comparison_name, comparison_table in pairwise_tables.items():
        comparison_table.to_csv(
            destination / output_filenames[comparison_name],
            index=False,
        )
    summary_key = "abc" if "abc" in output_filenames else "summary"
    combined_summary.to_csv(
        destination / output_filenames[summary_key],
        index=False,
    )

    run_config = {
        "evaluation_title": evaluation_title,
        "evaluation_type": "fixed_conditions_only",
        "random_condition_sampling": False,
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "split": split,
        "threshold": threshold,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(evaluation_device),
        "seed": seed,
        "label_convention": {"0": "real", "1": "AI-generated"},
        "preprocessing": "fixed condition before src.data.get_eval_transform",
        "num_conditions": len(conditions),
        "num_models": len(specs),
        "num_images_per_model_condition": int(
            metric_rows[0]["num_samples"]
        ),
        "models": checkpoint_records,
        "conditions": [
            {
                **condition_metadata(condition),
                "steps": _step_records(condition),
            }
            for condition in conditions
        ],
        "reported_metrics": list(REPORTED_METRICS),
        "output_files": output_filenames,
    }
    with (destination / output_filenames["config"]).open(
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(run_config, config_file, indent=2, ensure_ascii=False)
        config_file.write("\n")

    return predictions_table, metrics_table, pairwise_tables, combined_summary


def run_fixed_robustness_abc_evaluation(
    *,
    data_root: str | Path,
    manifest_path: str | Path,
    checkpoint_a_baseline: str | Path,
    checkpoint_b_robustness: str | Path,
    checkpoint_c_consistency: str | Path,
    output_dir: str | Path,
    split: str = "val",
    threshold: float = 0.5,
    batch_size: int = 32,
    num_workers: int = 2,
    device: torch.device | None = None,
    seed: int = 42,
    conditions: Sequence[
        EvaluationCondition
    ] = ALL_FIXED_EVALUATION_CONDITIONS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """Backward-compatible wrapper for the original A/B/C command."""
    return run_fixed_robustness_evaluation(
        data_root=data_root,
        manifest_path=manifest_path,
        model_specs=_build_model_specs(
            checkpoint_a_baseline,
            checkpoint_b_robustness,
            checkpoint_c_consistency,
        ),
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
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed-condition comparison over two or more named models."
        ),
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-a-baseline")
    parser.add_argument("--checkpoint-b-robustness")
    parser.add_argument("--checkpoint-c-consistency")
    add_variable_model_arguments(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--condition-set",
        choices=["all-fixed", "fixed-chains-only"],
        default="all-fixed",
        help=(
            "all-fixed runs clean, 15 official single transforms, and five "
            "fixed chains; fixed-chains-only runs clean plus the five chains"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser

def calculate_headline_metrics(abc_summary: pd.DataFrame, model_ids: list[str]):
    """Calculate the Mean, Worst-Case, and Robustness Gap for AUROC."""
    clean_row = abc_summary[abc_summary["condition_id"] == "clean"].iloc[0]
    damaged_rows = abc_summary[abc_summary["condition_id"] != "clean"]
    
    print("\n" + "="*50)
    print("DAY 2 HEADLINE METRICS (AUROC)")
    print("="*50)
    
    for model_id in model_ids:
        clean_auroc = float(clean_row[f"{model_id}__auroc"])
        mean_damaged_auroc = float(damaged_rows[f"{model_id}__auroc"].mean())
        worst_damaged_auroc = float(damaged_rows[f"{model_id}__auroc"].min())
        robustness_gap = clean_auroc - mean_damaged_auroc
        
        print(f"\n{model_id}:")
        print(f"  Clean AUROC:          {clean_auroc:.4f}")
        print(f"  Mean Transformed:     {mean_damaged_auroc:.4f}")
        print(f"  Worst-Case:           {worst_damaged_auroc:.4f}")
        print(f"  Robustness Gap:       {robustness_gap:.4f}")
        
def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    device = resolve_device(args.device)
    try:
        specs = variable_model_specs_from_args(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if specs is None:
        legacy_checkpoints = (
            args.checkpoint_a_baseline,
            args.checkpoint_b_robustness,
            args.checkpoint_c_consistency,
        )
        if any(path is None for path in legacy_checkpoints):
            raise SystemExit(
                "provide all three legacy checkpoint flags or use "
                "--checkpoints with --model-ids"
            )
        specs = _build_model_specs(*legacy_checkpoints)
    conditions = (
        ALL_FIXED_EVALUATION_CONDITIONS
        if args.condition_set == "all-fixed"
        else FIXED_CHAIN_EVALUATION_CONDITIONS
    )
    _, _, _, combined_summary = run_fixed_robustness_evaluation(
        data_root=args.data_root,
        manifest_path=args.manifest,
        model_specs=specs,
        output_dir=args.output_dir,
        split=args.split,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
        conditions=conditions,
    )

    display_columns = ["condition_title"]
    display_columns.extend(
        f"{spec.model_id}__accuracy" for spec in specs
    )
    display_columns.append("highest_accuracy_model")
    display_columns.extend(
        f"{spec.model_id}__false_positive_rate" for spec in specs
    )
    evaluation_title = comparison_title("Fixed robustness evaluation", specs)
    print(evaluation_title)
    print(f"Evaluation device: {device}")
    print(f"Condition set: {args.condition_set}")
    print(f"Results written to: {Path(args.output_dir)}")
    print(combined_summary[display_columns])
    calculate_headline_metrics(
        combined_summary,
        [spec.model_id for spec in specs],
    )


if __name__ == "__main__":
    main()
