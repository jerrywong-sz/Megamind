"""Shared model specifications for variable-length evaluation runs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence


SUPPORTED_ARCHITECTURES = (
    "efficientnet_b0",
    "convnext_tiny",
    "dinov2_vits14",
    "hybrid_effnet_dinov2",
)
MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


@dataclass(frozen=True)
class EvaluationModelSpec:
    """One named checkpoint and its optional architecture override."""

    model_id: str
    model_title: str
    checkpoint_path: Path
    architecture: str | None = None

    def as_record(self) -> dict[str, str | None]:
        return {
            "model_id": self.model_id,
            "model_title": self.model_title,
            "checkpoint_path": str(self.checkpoint_path),
            "architecture_override": self.architecture,
        }


def validate_model_specs(
    model_specs: Sequence[EvaluationModelSpec],
) -> tuple[EvaluationModelSpec, ...]:
    """Validate and freeze a list containing at least two named models."""
    specs = tuple(model_specs)
    if len(specs) < 2:
        raise ValueError("evaluation requires at least two models")

    model_ids = [spec.model_id for spec in specs]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("model IDs must be unique")

    for spec in specs:
        if not MODEL_ID_PATTERN.fullmatch(spec.model_id):
            raise ValueError(
                "model IDs must use lowercase letters, numbers, and underscores"
            )
        if not spec.model_title.strip():
            raise ValueError("model titles must not be empty")
        if not str(spec.checkpoint_path).strip():
            raise ValueError("checkpoint paths must not be empty")
        if (
            spec.architecture is not None
            and spec.architecture not in SUPPORTED_ARCHITECTURES
        ):
            raise ValueError(
                f"unsupported architecture '{spec.architecture}'; expected "
                + ", ".join(SUPPORTED_ARCHITECTURES)
            )
    return specs


def build_model_specs(
    *,
    checkpoints: Sequence[str | Path],
    model_ids: Sequence[str],
    model_titles: Sequence[str] | None = None,
    architectures: Sequence[str | None] | None = None,
) -> tuple[EvaluationModelSpec, ...]:
    """Build model specifications from aligned command-line style lists."""
    checkpoint_values = list(checkpoints)
    id_values = list(model_ids)
    title_values = list(model_titles) if model_titles is not None else id_values
    architecture_values = (
        list(architectures)
        if architectures is not None
        else [None] * len(checkpoint_values)
    )

    lengths = {
        len(checkpoint_values),
        len(id_values),
        len(title_values),
        len(architecture_values),
    }
    if len(lengths) != 1:
        raise ValueError(
            "checkpoints, model IDs, model titles, and architectures must "
            "contain the same number of values"
        )

    specs = [
        EvaluationModelSpec(
            model_id=model_id,
            model_title=model_title,
            checkpoint_path=Path(checkpoint),
            architecture=(
                None if architecture in {None, "auto"} else architecture
            ),
        )
        for checkpoint, model_id, model_title, architecture in zip(
            checkpoint_values,
            id_values,
            title_values,
            architecture_values,
            strict=True,
        )
    ]
    return validate_model_specs(specs)


def add_variable_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared 2+ model command-line interface to a parser."""
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="Two or more checkpoint paths, in model display order.",
    )
    parser.add_argument(
        "--model-ids",
        nargs="+",
        help="Unique lowercase IDs aligned with --checkpoints.",
    )
    parser.add_argument(
        "--model-titles",
        nargs="+",
        help="Human-readable titles aligned with --checkpoints.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("auto", *SUPPORTED_ARCHITECTURES),
        help=(
            "Architecture overrides aligned with --checkpoints. Use auto to "
            "read checkpoint metadata or fall back to EfficientNet-B0."
        ),
    )


def variable_model_specs_from_args(
    args: argparse.Namespace,
) -> tuple[EvaluationModelSpec, ...] | None:
    """Return generic CLI model specs, or ``None`` for a legacy command."""
    supplied = (
        args.checkpoints,
        args.model_ids,
        args.model_titles,
        args.architectures,
    )
    if not any(value is not None for value in supplied):
        return None
    if args.checkpoints is None or args.model_ids is None:
        raise ValueError(
            "variable-model mode requires --checkpoints and --model-ids"
        )
    return build_model_specs(
        checkpoints=args.checkpoints,
        model_ids=args.model_ids,
        model_titles=args.model_titles,
        architectures=args.architectures,
    )


def model_pairs(
    model_specs: Sequence[EvaluationModelSpec],
) -> tuple[tuple[EvaluationModelSpec, EvaluationModelSpec], ...]:
    """Return every unordered model pair in input order."""
    specs = validate_model_specs(model_specs)
    return tuple(combinations(specs, 2))


def model_titles(
    model_specs: Sequence[EvaluationModelSpec],
) -> dict[str, str]:
    return {spec.model_id: spec.model_title for spec in model_specs}


def comparison_title(
    prefix: str,
    model_specs: Sequence[EvaluationModelSpec],
) -> str:
    specs = validate_model_specs(model_specs)
    return prefix + " — " + " vs ".join(spec.model_title for spec in specs)


def model_token(model_specs: Sequence[EvaluationModelSpec]) -> str:
    specs = validate_model_specs(model_specs)
    return "_vs_".join(spec.model_id for spec in specs)
