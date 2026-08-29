"""Evaluate Models A, B, and C under fixed robustness conditions.

This runner is intentionally explicit about model roles.  It produces three
pairwise reports (A-vs-B, B-vs-C, and A-vs-C), one combined A/B/C report, a
full metrics table, per-image predictions, and a reproducibility config.
Randomly sampled transform chains are outside this runner's scope.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
    condition_steps,
    condition_title,
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


@dataclass(frozen=True)
class ModelSpec:
    """One explicitly named checkpoint participating in the A/B/C run."""

    model_id: str
    model_title: str
    checkpoint_path: Path


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
        "condition_id": condition.name,
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
        metrics_table["condition_id"] == condition.name
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
                f"condition '{condition.name}' is missing one compared model"
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
) -> pd.DataFrame:
    """Place all A/B/C metrics and winners in one explicitly named table."""
    model_ids = (MODEL_A_ID, MODEL_B_ID, MODEL_C_ID)
    clean_rows = _clean_metric_rows(metrics_table)
    if not set(model_ids).issubset(clean_rows):
        raise ValueError("metrics must contain clean rows for Models A, B, and C")

    summary_rows: list[dict[str, Any]] = []
    for condition in conditions:
        rows_by_model = _metric_row_by_model(metrics_table, condition)
        if not set(model_ids).issubset(rows_by_model):
            raise ValueError(
                f"condition '{condition.name}' is missing A, B, or C"
            )

        output_row: dict[str, Any] = {
            "report_title": EVALUATION_TITLE,
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

        for metric in (
            "accuracy",
            "auroc",
            "auprc",
            "false_positive_rate",
            "false_negative_rate",
            "brier_score",
        ):
            output_row[
                f"{MODEL_B_ID}_minus_{MODEL_A_ID}__{metric}"
            ] = float(rows_by_model[MODEL_B_ID][metric]) - float(
                rows_by_model[MODEL_A_ID][metric]
            )
            output_row[
                f"{MODEL_C_ID}_minus_{MODEL_B_ID}__{metric}"
            ] = float(rows_by_model[MODEL_C_ID][metric]) - float(
                rows_by_model[MODEL_B_ID][metric]
            )
            output_row[
                f"{MODEL_C_ID}_minus_{MODEL_A_ID}__{metric}"
            ] = float(rows_by_model[MODEL_C_ID][metric]) - float(
                rows_by_model[MODEL_A_ID][metric]
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
        raise RuntimeError("unexpected columns in A/B/C prediction table")
    return ordered


def _build_model_specs(
    checkpoint_a_baseline: str | Path,
    checkpoint_b_robustness: str | Path,
    checkpoint_c_consistency: str | Path,
) -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(
            MODEL_A_ID,
            MODEL_TITLES[MODEL_A_ID],
            Path(checkpoint_a_baseline),
        ),
        ModelSpec(
            MODEL_B_ID,
            MODEL_TITLES[MODEL_B_ID],
            Path(checkpoint_b_robustness),
        ),
        ModelSpec(
            MODEL_C_ID,
            MODEL_TITLES[MODEL_C_ID],
            Path(checkpoint_c_consistency),
        ),
    )


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
    """Run one fair fixed-condition evaluation over Models A, B, and C."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not conditions or not any(item.name == "clean" for item in conditions):
        raise ValueError("fixed A/B/C evaluation requires a clean condition")
    condition_ids = [item.name for item in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("fixed condition IDs must be unique")

    evaluation_device = device or resolve_device("auto")
    model_specs = _build_model_specs(
        checkpoint_a_baseline,
        checkpoint_b_robustness,
        checkpoint_c_consistency,
    )
    loaded_models: list[tuple[ModelSpec, nn.Module, str]] = []
    checkpoint_records: list[dict[str, str]] = []

    for spec in model_specs:
        model, checkpoint_hash = load_model_checkpoint(
            spec.checkpoint_path,
            evaluation_device,
        )
        loaded_models.append((spec, model, checkpoint_hash))
        checkpoint_records.append({
            "model_id": spec.model_id,
            "model_title": spec.model_title,
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
                    "Models A, B, and C were not evaluated on identical rows"
                )

            predictions["evaluation_title"] = EVALUATION_TITLE
            predictions["model_title"] = spec.model_title
            for key, value in metadata.items():
                predictions[key] = value
            predictions = predictions[
                _prediction_output_columns(predictions)
            ]
            prediction_tables.append(predictions)

            metric_rows.append({
                "report_title": EVALUATION_TITLE,
                "model_id": spec.model_id,
                "model_title": spec.model_title,
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
    pairwise_tables = {
        "a_vs_b": build_explicit_pairwise_comparison(
            metrics_table,
            conditions=conditions,
            reference_model_id=MODEL_A_ID,
            candidate_model_id=MODEL_B_ID,
        ),
        "b_vs_c": build_explicit_pairwise_comparison(
            metrics_table,
            conditions=conditions,
            reference_model_id=MODEL_B_ID,
            candidate_model_id=MODEL_C_ID,
        ),
        "a_vs_c": build_explicit_pairwise_comparison(
            metrics_table,
            conditions=conditions,
            reference_model_id=MODEL_A_ID,
            candidate_model_id=MODEL_C_ID,
        ),
    }
    abc_summary = build_combined_abc_summary(
        metrics_table,
        conditions=conditions,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions_table.to_csv(
        destination / FIXED_ABC_FILENAMES["predictions"],
        index=False,
    )
    metrics_table.to_csv(
        destination / FIXED_ABC_FILENAMES["metrics"],
        index=False,
    )
    for comparison_name, comparison_table in pairwise_tables.items():
        comparison_table.to_csv(
            destination / FIXED_ABC_FILENAMES[comparison_name],
            index=False,
        )
    abc_summary.to_csv(
        destination / FIXED_ABC_FILENAMES["abc"],
        index=False,
    )

    run_config = {
        "evaluation_title": EVALUATION_TITLE,
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
        "num_models": len(model_specs),
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
        "output_files": FIXED_ABC_FILENAMES,
    }
    with (destination / FIXED_ABC_FILENAMES["config"]).open(
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(run_config, config_file, indent=2, ensure_ascii=False)
        config_file.write("\n")

    return predictions_table, metrics_table, pairwise_tables, abc_summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an explicitly named fixed-condition comparison of Model A "
            "baseline, Model B robustness, and Model C consistency."
        ),
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-a-baseline", required=True)
    parser.add_argument("--checkpoint-b-robustness", required=True)
    parser.add_argument("--checkpoint-c-consistency", required=True)
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
    conditions = (
        ALL_FIXED_EVALUATION_CONDITIONS
        if args.condition_set == "all-fixed"
        else FIXED_CHAIN_EVALUATION_CONDITIONS
    )
    _, _, _, abc_summary = run_fixed_robustness_abc_evaluation(
        data_root=args.data_root,
        manifest_path=args.manifest,
        checkpoint_a_baseline=args.checkpoint_a_baseline,
        checkpoint_b_robustness=args.checkpoint_b_robustness,
        checkpoint_c_consistency=args.checkpoint_c_consistency,
        output_dir=args.output_dir,
        split=args.split,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
        conditions=conditions,
    )

    display_columns = [
        "condition_title",
        f"{MODEL_A_ID}__accuracy",
        f"{MODEL_B_ID}__accuracy",
        f"{MODEL_C_ID}__accuracy",
        "highest_accuracy_model",
        f"{MODEL_A_ID}__false_positive_rate",
        f"{MODEL_B_ID}__false_positive_rate",
        f"{MODEL_C_ID}__false_positive_rate",
    ]
    print(EVALUATION_TITLE)
    print(f"Evaluation device: {device}")
    print(f"Condition set: {args.condition_set}")
    print(f"Results written to: {Path(args.output_dir)}")
    print(abc_summary[display_columns])
    
    calculate_headline_metrics(abc_summary, [MODEL_A_ID, MODEL_B_ID, MODEL_C_ID])


if __name__ == "__main__":
    main()
