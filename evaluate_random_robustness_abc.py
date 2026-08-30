"""Evaluate Models A, B, and C with seeded per-image random chains."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluate import evaluate_model_condition, load_model_checkpoint, resolve_device
from evaluate_fixed_robustness_abc import (
    MODEL_A_ID,
    MODEL_B_ID,
    MODEL_C_ID,
    MODEL_TITLES,
    REPORTED_METRICS,
)
from src.data import get_evaluation_dataloader
from src.metrics import compute_binary_metrics
from src.random_chain_conditions import (
    RANDOM_STANDARD_3_CHAIN_LENGTH,
    RANDOM_STANDARD_3_PARAMETER_GRID,
    RANDOM_STANDARD_3_POLICY,
    RandomStandard3Transform,
    random_chain_metadata,
)


DEFAULT_TRIAL_SEEDS = (42, 43, 44, 45, 46)
MODEL_IDS = (MODEL_A_ID, MODEL_B_ID, MODEL_C_ID)
EVALUATION_TITLE = (
    "Random standard-3 robustness evaluation — Model A baseline vs "
    "Model B robustness vs Model C consistency"
)
RANDOM_ABC_FILENAMES = {
    "predictions": "random_standard_3__models_A_B_C__per_image_predictions.csv",
    "assignments": "random_standard_3__per_image_chain_assignments.csv",
    "clean_predictions": "random_standard_3__models_A_B_C__clean_predictions.csv",
    "trial_metrics": "random_standard_3__models_A_B_C__trial_metrics.csv",
    "overall": "random_standard_3__models_A_B_C__overall_summary.csv",
    "headline": "random_standard_3__models_A_B_C__headline.csv",
    "a_vs_b": "random_standard_3__model_A_baseline_vs_model_B_robustness.csv",
    "b_vs_c": "random_standard_3__model_B_robustness_vs_model_C_consistency.csv",
    "a_vs_c": "random_standard_3__model_A_baseline_vs_model_C_consistency.csv",
    "patterns": "random_standard_3__chain_pattern_breakdown.csv",
    "inclusion": "random_standard_3__transform_inclusion_breakdown.csv",
    "errors": "random_standard_3__false_positives_and_false_negatives.csv",
    "config": "random_standard_3__run_config.json",
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_title: str
    checkpoint_path: Path


def _model_specs(
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


def _winner(values: dict[str, float], *, higher_is_better: bool) -> str:
    target = max(values.values()) if higher_is_better else min(values.values())
    winners = [
        model_id
        for model_id, value in values.items()
        if abs(value - target) < 1e-12
    ]
    return winners[0] if len(winners) == 1 else "tie"


def _chain_table(
    predictions: pd.DataFrame,
    *,
    dataset_id: str,
    trial_index: int,
    trial_seed: int,
) -> pd.DataFrame:
    rows = []
    for image_path in predictions["image_path"]:
        rows.append({
            "dataset_id": dataset_id,
            "trial_index": trial_index,
            "trial_seed": trial_seed,
            "image_path": image_path,
            **random_chain_metadata(
                dataset_id=dataset_id,
                image_path=str(image_path),
                trial_seed=trial_seed,
            ),
        })
    return pd.DataFrame(rows)


def _enrich_predictions(
    predictions: pd.DataFrame,
    *,
    chain_table: pd.DataFrame,
    clean_predictions: pd.DataFrame,
    dataset_id: str,
    trial_index: int,
) -> pd.DataFrame:
    enriched = predictions.merge(
        chain_table,
        on="image_path",
        how="left",
        validate="one_to_one",
        suffixes=("", "_chain"),
    )
    clean_context = clean_predictions[
        ["image_path", "prob_ai", "predicted_label", "is_correct"]
    ].rename(columns={
        "prob_ai": "clean_prob_ai",
        "predicted_label": "clean_predicted_label",
        "is_correct": "clean_is_correct",
    })
    enriched = enriched.merge(
        clean_context,
        on="image_path",
        how="left",
        validate="one_to_one",
    )
    enriched["evaluation_title"] = EVALUATION_TITLE
    enriched["dataset_id"] = dataset_id
    enriched["trial_index"] = trial_index
    enriched["probability_shift_from_clean"] = (
        enriched["prob_ai"] - enriched["clean_prob_ai"]
    )
    enriched["became_incorrect_after_transform"] = (
        enriched["clean_is_correct"] & ~enriched["is_correct"]
    )
    enriched["recovered_after_transform"] = (
        ~enriched["clean_is_correct"] & enriched["is_correct"]
    )
    labels = enriched["label"].to_numpy()
    predicted = enriched["predicted_label"].to_numpy()
    enriched["error_type"] = np.select(
        [
            (labels == 0) & (predicted == 0),
            (labels == 0) & (predicted == 1),
            (labels == 1) & (predicted == 0),
        ],
        ["true_negative", "false_positive", "false_negative"],
        default="true_positive",
    )
    return enriched


def _metric_row(
    *,
    dataset_id: str,
    model_id: str,
    model_title: str,
    checkpoint_hash: str,
    scope: str,
    trial_index: int | None,
    trial_seed: int | None,
    metrics: dict[str, float | int],
) -> dict[str, Any]:
    return {
        "evaluation_title": EVALUATION_TITLE,
        "dataset_id": dataset_id,
        "random_policy": RANDOM_STANDARD_3_POLICY,
        "scope": scope,
        "trial_index": trial_index,
        "trial_seed": trial_seed,
        "model_id": model_id,
        "model_title": model_title,
        "checkpoint_hash": checkpoint_hash,
        **metrics,
    }


def _overall_summary(
    *,
    dataset_id: str,
    clean_metric_rows: list[dict[str, Any]],
    trial_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    checkpoint_hashes: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_by_model = {row["model_id"]: row for row in clean_metric_rows}
    overall_rows = []
    pooled_rows = []
    for model_id in MODEL_IDS:
        model_predictions = predictions[predictions["model_id"] == model_id]
        pooled_metrics = compute_binary_metrics(
            model_predictions["label"].to_numpy(),
            model_predictions["prob_ai"].to_numpy(),
            threshold=float(clean_by_model[model_id]["threshold"]),
        )
        pooled_rows.append(_metric_row(
            dataset_id=dataset_id,
            model_id=model_id,
            model_title=MODEL_TITLES[model_id],
            checkpoint_hash=checkpoint_hashes[model_id],
            scope="pooled_random_trials",
            trial_index=None,
            trial_seed=None,
            metrics=pooled_metrics,
        ))
        model_trials = trial_metrics[trial_metrics["model_id"] == model_id]
        clean = clean_by_model[model_id]
        retained = model_predictions[model_predictions["clean_is_correct"]]
        row: dict[str, Any] = {
            "evaluation_title": EVALUATION_TITLE,
            "dataset_id": dataset_id,
            "random_policy": RANDOM_STANDARD_3_POLICY,
            "model_id": model_id,
            "model_title": MODEL_TITLES[model_id],
            "checkpoint_hash": checkpoint_hashes[model_id],
            "num_trials": int(model_trials["trial_seed"].nunique()),
            "num_unique_images": int(model_predictions["image_path"].nunique()),
            "num_random_predictions": int(len(model_predictions)),
            "clean_correct_retention_rate": (
                float(retained["is_correct"].mean())
                if not retained.empty
                else float("nan")
            ),
            "clean_correct_to_wrong_rate": (
                float((~retained["is_correct"]).mean())
                if not retained.empty
                else float("nan")
            ),
        }
        for metric in REPORTED_METRICS:
            row[f"clean__{metric}"] = clean[metric]
            row[f"pooled_random__{metric}"] = pooled_metrics[metric]
            if metric not in {"num_samples", "threshold"}:
                row[f"trial_mean__{metric}"] = float(model_trials[metric].mean())
                row[f"trial_std__{metric}"] = float(
                    model_trials[metric].std(ddof=0)
                )
        row["accuracy_drop_from_clean"] = (
            float(clean["accuracy"]) - float(pooled_metrics["accuracy"])
        )
        row["auroc_drop_from_clean"] = (
            float(clean["auroc"]) - float(pooled_metrics["auroc"])
        )
        row["auprc_drop_from_clean"] = (
            float(clean["auprc"]) - float(pooled_metrics["auprc"])
        )
        overall_rows.append(row)
    return pd.DataFrame(overall_rows), pd.DataFrame(pooled_rows)


def _pairwise_comparison(
    metrics: pd.DataFrame,
    *,
    reference_model_id: str,
    candidate_model_id: str,
) -> pd.DataFrame:
    rows = []
    group_columns = ["scope", "trial_index", "trial_seed"]
    for group_key, group in metrics.groupby(group_columns, dropna=False):
        by_model = {
            row["model_id"]: row
            for _, row in group.iterrows()
        }
        if not {reference_model_id, candidate_model_id}.issubset(by_model):
            raise RuntimeError("pairwise random metrics are missing a model")
        reference = by_model[reference_model_id]
        candidate = by_model[candidate_model_id]
        output: dict[str, Any] = {
            "evaluation_title": EVALUATION_TITLE,
            "dataset_id": reference["dataset_id"],
            "random_policy": RANDOM_STANDARD_3_POLICY,
            "scope": group_key[0],
            "trial_index": group_key[1],
            "trial_seed": group_key[2],
            "reference_model_id": reference_model_id,
            "reference_model_title": MODEL_TITLES[reference_model_id],
            "candidate_model_id": candidate_model_id,
            "candidate_model_title": MODEL_TITLES[candidate_model_id],
            "difference_direction": (
                f"{candidate_model_id} minus {reference_model_id}"
            ),
        }
        for metric in REPORTED_METRICS:
            reference_value = reference[metric]
            candidate_value = candidate[metric]
            output[f"{reference_model_id}__{metric}"] = reference_value
            output[f"{candidate_model_id}__{metric}"] = candidate_value
            if metric not in {"num_samples", "threshold"}:
                output[
                    f"{candidate_model_id}_minus_{reference_model_id}__{metric}"
                ] = float(candidate_value) - float(reference_value)
        output["higher_accuracy_model"] = _winner(
            {
                reference_model_id: float(reference["accuracy"]),
                candidate_model_id: float(candidate["accuracy"]),
            },
            higher_is_better=True,
        )
        output["higher_auroc_model"] = _winner(
            {
                reference_model_id: float(reference["auroc"]),
                candidate_model_id: float(candidate["auroc"]),
            },
            higher_is_better=True,
        )
        output["higher_auprc_model"] = _winner(
            {
                reference_model_id: float(reference["auprc"]),
                candidate_model_id: float(candidate["auprc"]),
            },
            higher_is_better=True,
        )
        output["lower_false_positive_rate_model"] = _winner(
            {
                reference_model_id: float(reference["false_positive_rate"]),
                candidate_model_id: float(candidate["false_positive_rate"]),
            },
            higher_is_better=False,
        )
        output["lower_false_negative_rate_model"] = _winner(
            {
                reference_model_id: float(reference["false_negative_rate"]),
                candidate_model_id: float(candidate["false_negative_rate"]),
            },
            higher_is_better=False,
        )
        output["lower_brier_score_model"] = _winner(
            {
                reference_model_id: float(reference["brier_score"]),
                candidate_model_id: float(candidate["brier_score"]),
            },
            higher_is_better=False,
        )
        rows.append(output)
    return pd.DataFrame(rows)


def _group_metrics(
    predictions: pd.DataFrame,
    *,
    group_column: str,
    breakdown_type: str,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for (model_id, group_value), group in predictions.groupby(
        ["model_id", group_column]
    ):
        metrics = compute_binary_metrics(
            group["label"].to_numpy(),
            group["prob_ai"].to_numpy(),
            threshold=threshold,
        )
        rows.append({
            "evaluation_title": EVALUATION_TITLE,
            "dataset_id": group["dataset_id"].iloc[0],
            "random_policy": RANDOM_STANDARD_3_POLICY,
            "breakdown_type": breakdown_type,
            "breakdown_value": group_value,
            "model_id": model_id,
            "model_title": MODEL_TITLES[model_id],
            "num_trials_represented": int(group["trial_seed"].nunique()),
            **metrics,
        })
    return pd.DataFrame(rows)


def _transform_inclusion_metrics(
    predictions: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    tables = []
    for transform_name in RANDOM_STANDARD_3_PARAMETER_GRID:
        included = predictions[predictions[f"contains_{transform_name}"]]
        included = included.assign(included_transform=transform_name)
        tables.append(_group_metrics(
            included,
            group_column="included_transform",
            breakdown_type="contains_transform",
            threshold=threshold,
        ))
    return pd.concat(tables, ignore_index=True)


def _headline_table(overall: pd.DataFrame) -> pd.DataFrame:
    row: dict[str, Any] = {
        "evaluation_title": EVALUATION_TITLE,
        "dataset_id": overall["dataset_id"].iloc[0],
        "random_policy": RANDOM_STANDARD_3_POLICY,
        "num_trials": int(overall["num_trials"].iloc[0]),
        "num_unique_images": int(overall["num_unique_images"].iloc[0]),
    }
    for _, model in overall.iterrows():
        model_id = model["model_id"]
        for column in (
            "clean__accuracy",
            "pooled_random__accuracy",
            "trial_mean__accuracy",
            "trial_std__accuracy",
            "accuracy_drop_from_clean",
            "pooled_random__auroc",
            "pooled_random__auprc",
            "pooled_random__false_positive_rate",
            "pooled_random__false_negative_rate",
            "clean_correct_retention_rate",
        ):
            row[f"{model_id}__{column}"] = model[column]
    pooled_accuracy = {
        model["model_id"]: float(model["pooled_random__accuracy"])
        for _, model in overall.iterrows()
    }
    row["highest_pooled_random_accuracy_model"] = _winner(
        pooled_accuracy,
        higher_is_better=True,
    )
    return pd.DataFrame([row])


def run_random_standard_3_abc_evaluation(
    *,
    dataset_id: str,
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
    trial_seeds: Sequence[int] = DEFAULT_TRIAL_SEEDS,
) -> dict[str, pd.DataFrame]:
    """Run clean plus five seeded random-standard-3 A/B/C trials."""
    if not dataset_id.strip():
        raise ValueError("dataset_id must not be empty")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not trial_seeds:
        raise ValueError("at least one trial seed is required")
    if len(trial_seeds) != len(set(trial_seeds)):
        raise ValueError("trial seeds must be unique")

    evaluation_device = device or resolve_device("auto")
    specs = _model_specs(
        checkpoint_a_baseline,
        checkpoint_b_robustness,
        checkpoint_c_consistency,
    )
    loaded_models: list[tuple[ModelSpec, nn.Module, str]] = []
    checkpoint_records = []
    checkpoint_hashes = {}
    for spec in specs:
        model, checkpoint_hash = load_model_checkpoint(
            spec.checkpoint_path,
            evaluation_device,
        )
        loaded_models.append((spec, model, checkpoint_hash))
        checkpoint_hashes[spec.model_id] = checkpoint_hash
        checkpoint_records.append({
            "model_id": spec.model_id,
            "model_title": spec.model_title,
            "path": str(spec.checkpoint_path),
            "sha256": checkpoint_hash,
        })

    clean_loader = get_evaluation_dataloader(
        data_root=str(data_root),
        manifest_path=str(manifest_path),
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    clean_tables = []
    clean_metrics = []
    clean_by_model = {}
    for spec, model, checkpoint_hash in loaded_models:
        print(f"Clean evaluation: {spec.model_title}")
        predictions, metrics = evaluate_model_condition(
            model,
            clean_loader,
            evaluation_device,
            model_id=spec.model_id,
            threshold=threshold,
            checkpoint_hash=checkpoint_hash,
            transform_name="clean",
        )
        predictions["evaluation_title"] = EVALUATION_TITLE
        predictions["dataset_id"] = dataset_id
        predictions["model_title"] = spec.model_title
        clean_tables.append(predictions)
        clean_by_model[spec.model_id] = predictions
        clean_metrics.append(_metric_row(
            dataset_id=dataset_id,
            model_id=spec.model_id,
            model_title=spec.model_title,
            checkpoint_hash=checkpoint_hash,
            scope="clean",
            trial_index=None,
            trial_seed=None,
            metrics=metrics,
        ))

    random_tables = []
    assignment_tables = []
    trial_metric_rows = []
    for trial_index, trial_seed in enumerate(trial_seeds, start=1):
        print(f"Random standard-3 trial {trial_index}: seed {trial_seed}")
        pre_transform = RandomStandard3Transform(
            dataset_id=dataset_id,
            trial_seed=trial_seed,
        )
        data_loader = get_evaluation_dataloader(
            data_root=str(data_root),
            manifest_path=str(manifest_path),
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            pre_transform=pre_transform,
        )
        reference_rows = None
        trial_assignments = None
        for spec, model, checkpoint_hash in loaded_models:
            print(f"  Evaluating {spec.model_title}")
            predictions, metrics = evaluate_model_condition(
                model,
                data_loader,
                evaluation_device,
                model_id=spec.model_id,
                threshold=threshold,
                checkpoint_hash=checkpoint_hash,
                transform_name=RANDOM_STANDARD_3_POLICY,
                seed=trial_seed,
            )
            current_rows = predictions[["image_path", "label"]].reset_index(
                drop=True
            )
            if reference_rows is None:
                reference_rows = current_rows
                trial_assignments = _chain_table(
                    predictions,
                    dataset_id=dataset_id,
                    trial_index=trial_index,
                    trial_seed=trial_seed,
                )
                assignment_tables.append(trial_assignments)
            elif not current_rows.equals(reference_rows):
                raise RuntimeError(
                    "Models A, B, and C were not evaluated on identical rows"
                )
            assert trial_assignments is not None
            enriched = _enrich_predictions(
                predictions,
                chain_table=trial_assignments,
                clean_predictions=clean_by_model[spec.model_id],
                dataset_id=dataset_id,
                trial_index=trial_index,
            )
            enriched["model_title"] = spec.model_title
            random_tables.append(enriched)
            trial_metric_rows.append(_metric_row(
                dataset_id=dataset_id,
                model_id=spec.model_id,
                model_title=spec.model_title,
                checkpoint_hash=checkpoint_hash,
                scope="random_trial",
                trial_index=trial_index,
                trial_seed=trial_seed,
                metrics=metrics,
            ))

    clean_predictions = pd.concat(clean_tables, ignore_index=True)
    predictions = pd.concat(random_tables, ignore_index=True)
    assignments = pd.concat(assignment_tables, ignore_index=True)
    trial_metrics = pd.DataFrame(trial_metric_rows)
    overall, pooled_metrics = _overall_summary(
        dataset_id=dataset_id,
        clean_metric_rows=clean_metrics,
        trial_metrics=trial_metrics,
        predictions=predictions,
        checkpoint_hashes=checkpoint_hashes,
    )
    comparison_metrics = pd.concat(
        [trial_metrics, pooled_metrics],
        ignore_index=True,
    )
    pairwise = {
        "a_vs_b": _pairwise_comparison(
            comparison_metrics,
            reference_model_id=MODEL_A_ID,
            candidate_model_id=MODEL_B_ID,
        ),
        "b_vs_c": _pairwise_comparison(
            comparison_metrics,
            reference_model_id=MODEL_B_ID,
            candidate_model_id=MODEL_C_ID,
        ),
        "a_vs_c": _pairwise_comparison(
            comparison_metrics,
            reference_model_id=MODEL_A_ID,
            candidate_model_id=MODEL_C_ID,
        ),
    }
    patterns = _group_metrics(
        predictions,
        group_column="chain_pattern",
        breakdown_type="ordered_chain_pattern",
        threshold=threshold,
    )
    inclusion = _transform_inclusion_metrics(
        predictions,
        threshold=threshold,
    )
    errors = predictions[~predictions["is_correct"]].copy()
    headline = _headline_table(overall)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "predictions": predictions,
        "assignments": assignments,
        "clean_predictions": clean_predictions,
        "trial_metrics": trial_metrics,
        "overall": overall,
        "headline": headline,
        "patterns": patterns,
        "inclusion": inclusion,
        "errors": errors,
        **pairwise,
    }
    for output_name, table in outputs.items():
        table.to_csv(
            destination / RANDOM_ABC_FILENAMES[output_name],
            index=False,
        )

    config = {
        "evaluation_title": EVALUATION_TITLE,
        "evaluation_type": "seeded_per_image_random_chain",
        "dataset_id": dataset_id,
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "split": split,
        "threshold": threshold,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(evaluation_device),
        "random_policy": RANDOM_STANDARD_3_POLICY,
        "chain_length": RANDOM_STANDARD_3_CHAIN_LENGTH,
        "distinct_transform_types_per_chain": True,
        "random_order": True,
        "trial_seeds": list(trial_seeds),
        "seed_identity": "policy + dataset_id + image_path + trial_seed",
        "parameter_grid": {
            name: list(values)
            for name, values in RANDOM_STANDARD_3_PARAMETER_GRID.items()
        },
        "label_convention": {"0": "real", "1": "AI-generated"},
        "models": checkpoint_records,
        "reported_metrics": list(REPORTED_METRICS),
        "output_files": RANDOM_ABC_FILENAMES,
    }
    with (destination / RANDOM_ABC_FILENAMES["config"]).open(
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(config, config_file, indent=2, ensure_ascii=False)
        config_file.write("\n")

    loaded_models.clear()
    if evaluation_device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Models A, B, and C over five seeded per-image random "
            "three-transform trials."
        )
    )
    parser.add_argument("--dataset-id", required=True)
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
    parser.add_argument(
        "--trial-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_TRIAL_SEEDS),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    outputs = run_random_standard_3_abc_evaluation(
        dataset_id=args.dataset_id,
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
        device=resolve_device(args.device),
        trial_seeds=args.trial_seeds,
    )
    print(EVALUATION_TITLE)
    print(f"Dataset: {args.dataset_id}")
    print(f"Trial seeds: {args.trial_seeds}")
    print(f"Results written to: {Path(args.output_dir)}")
    print(outputs["headline"].to_string(index=False))


if __name__ == "__main__":
    main()
