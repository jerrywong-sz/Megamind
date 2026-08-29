"""Deterministic random robustness stress-test conditions.

The fixed robustness grid remains the official reproducible benchmark. This
module adds an optional stress benchmark that samples realistic transform
chains without hand-writing every possible combination.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Union

from src.evaluation_conditions import EvaluationCondition, TransformStep


Range = tuple[Union[int, float], Union[int, float]]


SEVERITY_RANGES: dict[str, dict[str, Range]] = {
    "mild": {
        "jpeg": (75, 95),
        "blur": (0.3, 0.8),
        "resize": (0.70, 0.95),
        "crop": (0.90, 0.98),
        "colour": (-0.08, 0.08),
        "noise": (0.005, 0.02),
    },
    "medium": {
        "jpeg": (50, 75),
        "blur": (0.8, 1.6),
        "resize": (0.45, 0.70),
        "crop": (0.78, 0.90),
        "colour": (-0.16, 0.16),
        "noise": (0.02, 0.06),
    },
    "strong": {
        "jpeg": (25, 50),
        "blur": (1.6, 3.0),
        "resize": (0.25, 0.45),
        "crop": (0.65, 0.78),
        "colour": (-0.20, 0.20),
        "noise": (0.06, 0.10),
    },
}

RECOMPRESSION_RANGES: dict[str, tuple[Range, Range]] = {
    "mild": ((80, 95), (70, 90)),
    "medium": ((65, 90), (45, 75)),
    "strong": ((45, 85), (25, 60)),
}

TRANSFORM_ORDER = {
    "crop": 0,
    "resize": 1,
    "colour": 2,
    "blur": 3,
    "noise": 4,
    "jpeg": 5,
}

TRANSFORM_POOL = ("jpeg", "blur", "resize", "crop", "colour", "noise")
VALID_SEVERITIES = tuple(SEVERITY_RANGES)


def _sample_parameter(
    rng: random.Random,
    severity: str,
    transform_name: str,
) -> int | float:
    low, high = SEVERITY_RANGES[severity][transform_name]
    if transform_name == "jpeg":
        return rng.randint(int(low), int(high))
    return round(rng.uniform(float(low), float(high)), 2)


def _sample_recompression_chain(
    rng: random.Random,
    severity: str,
    min_transforms: int,
    max_transforms: int,
) -> tuple[TransformStep, ...]:
    minimum_length = max(2, min_transforms)
    maximum_length = min(max_transforms, 4)
    if severity == "mild":
        length = min(maximum_length, max(minimum_length, 2))
    else:
        length = rng.randint(minimum_length, maximum_length)

    first_range, later_range = RECOMPRESSION_RANGES[severity]
    first = rng.randint(int(first_range[0]), int(first_range[1]))
    qualities = [first]

    for _ in range(length - 1):
        low, high = later_range
        upper = min(int(high), qualities[-1])
        qualities.append(rng.randint(int(low), upper))

    return tuple(("jpeg", quality) for quality in qualities)


def _order_transforms(transform_names: list[str]) -> list[str]:
    return sorted(transform_names, key=lambda name: TRANSFORM_ORDER[name])


def generate_random_conditions(
    count: int = 20,
    severity: str = "medium",
    seed: int = 42,
    min_transforms: int = 1,
    max_transforms: int = 4,
) -> tuple[EvaluationCondition, ...]:
    """Generate deterministic random stress-test corruption chains.

    The sampler avoids duplicate transforms except for the explicit repeated
    JPEG recompression case. Generated chains are sorted into a plausible
    real-world order, keeping JPEG at the end when it appears.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if severity not in SEVERITY_RANGES:
        supported = ", ".join(VALID_SEVERITIES)
        raise ValueError(f"severity must be one of: {supported}")
    if min_transforms < 1:
        raise ValueError("min_transforms must be at least 1")
    if max_transforms > 4:
        raise ValueError("max_transforms must be at most 4")
    if min_transforms > max_transforms:
        raise ValueError("min_transforms must be less than or equal to max_transforms")

    rng = random.Random(f"{seed}|{severity}|{count}|{min_transforms}|{max_transforms}")
    conditions: list[EvaluationCondition] = []

    for index in range(count):
        allow_recompression = max_transforms >= 2 and min_transforms <= 4
        use_recompression = allow_recompression and rng.random() < 0.20

        if use_recompression:
            steps = _sample_recompression_chain(
                rng,
                severity,
                min_transforms,
                max_transforms,
            )
            name = f"random_{severity}_recompress_{index:03d}"
        else:
            length = rng.randint(min_transforms, max_transforms)
            transform_names = rng.sample(TRANSFORM_POOL, length)
            transform_names = _order_transforms(transform_names)
            steps = tuple(
                (name, _sample_parameter(rng, severity, name))
                for name in transform_names
            )
            name = f"random_{severity}_{index:03d}"

        conditions.append(
            EvaluationCondition(
                name=name,
                severity=None,
                steps=steps,
                seen_in_training=False,
                condition_type="random_stress",
                severity_label=severity,
            )
        )

    return tuple(conditions)


def generate_stress_conditions(
    counts: Mapping[str, int] | None = None,
    seed: int = 42,
    min_transforms: int = 1,
    max_transforms: int = 4,
) -> tuple[EvaluationCondition, ...]:
    """Generate mild, medium, and strong stress conditions deterministically."""
    if counts is None:
        counts = {"mild": 20, "medium": 20, "strong": 20}

    conditions: list[EvaluationCondition] = []
    for offset, severity in enumerate(VALID_SEVERITIES):
        conditions.extend(
            generate_random_conditions(
                count=counts.get(severity, 0),
                severity=severity,
                seed=seed + offset,
                min_transforms=min_transforms,
                max_transforms=max_transforms,
            )
        )
    return tuple(conditions)
