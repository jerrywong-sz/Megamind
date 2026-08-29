"""Fixed, reproducible robustness conditions used during evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import torch
from PIL import Image

from src.augmentations import exact_transform, exact_transform_chain


TransformParameter: TypeAlias = int | float
TransformStep: TypeAlias = tuple[str, TransformParameter]


@dataclass(frozen=True)
class EvaluationCondition:
    """One traceable clean, single-transform, or fixed-chain condition."""

    name: str
    severity: int | float | None
    title: str | None = None
    steps: tuple[TransformStep, ...] = ()
    condition_kind: str = "single"
    possible_in_b_c_training_sampler: bool = True


CLEAN_CONDITION = EvaluationCondition(
    "clean",
    None,
    title="Clean image — no robustness transform",
    condition_kind="clean",
)

ROBUSTNESS_CONDITIONS = (
    EvaluationCondition("jpeg", 90),
    EvaluationCondition("jpeg", 70),
    EvaluationCondition("jpeg", 50),
    EvaluationCondition("jpeg", 30),
    EvaluationCondition("blur", 0.5),
    EvaluationCondition("blur", 1.0),
    EvaluationCondition("blur", 2.0),
    EvaluationCondition("resize", 0.5),
    EvaluationCondition("resize", 0.25),
    EvaluationCondition("noise", 0.02),
    EvaluationCondition("noise", 0.05),
    EvaluationCondition("noise", 0.10),
    EvaluationCondition("colour", -0.20),
    EvaluationCondition("colour", 0.20),
    EvaluationCondition("crop", 0.80),
)

FIXED_CHAIN_CONDITIONS = (
    EvaluationCondition(
        name="fixed_resize_half_then_jpeg70",
        severity=None,
        title="Fixed chain — Resize 0.50× → JPEG quality 70",
        steps=(("resize", 0.5), ("jpeg", 70)),
        condition_kind="fixed_chain",
    ),
    EvaluationCondition(
        name="fixed_resize_quarter_then_jpeg50",
        severity=None,
        title="Fixed chain — Resize 0.25× → JPEG quality 50",
        steps=(("resize", 0.25), ("jpeg", 50)),
        condition_kind="fixed_chain",
    ),
    EvaluationCondition(
        name="fixed_crop_colour_jpeg",
        severity=None,
        title="Fixed chain — Crop 0.80 → Colour +0.20 → JPEG quality 50",
        steps=(("crop", 0.80), ("colour", 0.20), ("jpeg", 50)),
        condition_kind="fixed_chain",
    ),
    EvaluationCondition(
        name="fixed_screenshot_resample",
        severity=None,
        title="Fixed chain — Resize 0.50× → Blur 0.50 → JPEG quality 70",
        steps=(("resize", 0.5), ("blur", 0.5), ("jpeg", 70)),
        condition_kind="fixed_chain",
    ),
    EvaluationCondition(
        name="fixed_repeated_jpeg",
        severity=None,
        title="Fixed chain — JPEG quality 90 → JPEG 70 → JPEG 50",
        steps=(("jpeg", 90), ("jpeg", 70), ("jpeg", 50)),
        condition_kind="fixed_chain",
        possible_in_b_c_training_sampler=False,
    ),
)

# The original two-model evaluator keeps the official challenge grid.  The
# explicit A/B/C runner uses the larger fixed suite so adding mixed conditions
# does not silently change established commands or historical result files.
ALL_EVALUATION_CONDITIONS = (CLEAN_CONDITION, *ROBUSTNESS_CONDITIONS)
ALL_FIXED_EVALUATION_CONDITIONS = (
    CLEAN_CONDITION,
    *ROBUSTNESS_CONDITIONS,
    *FIXED_CHAIN_CONDITIONS,
)
FIXED_CHAIN_EVALUATION_CONDITIONS = (
    CLEAN_CONDITION,
    *FIXED_CHAIN_CONDITIONS,
)


def condition_steps(condition: EvaluationCondition) -> tuple[TransformStep, ...]:
    """Return the exact ordered steps represented by a condition."""
    if condition.name == "clean":
        return ()
    if condition.steps:
        return condition.steps
    if condition.severity is None:
        raise ValueError(
            f"condition '{condition.name}' needs a severity or transform steps"
        )
    return ((condition.name, condition.severity),)


def condition_id(condition: EvaluationCondition) -> str:
    """Return a stable ID that distinguishes every severity and chain."""
    if condition.name == "clean" or condition.steps:
        return condition.name
    if condition.severity is None:
        raise ValueError(
            f"condition '{condition.name}' needs a severity or transform steps"
        )
    severity = format(condition.severity, "g")
    return f"{condition.name}_{severity}"


def condition_title(condition: EvaluationCondition) -> str:
    """Return a human-readable title suitable for CSV and console output."""
    if condition.title:
        return condition.title
    if condition.name == "clean":
        return "Clean image — no robustness transform"
    return f"Single transform — {condition.name} parameter {condition.severity}"


def _image_seed(
    base_seed: int,
    condition: EvaluationCondition,
    image_path: str,
) -> int:
    """Derive the same stable seed for an image in every process and run."""
    identity = (
        f"{base_seed}|{condition.name}|{condition.severity}|"
        f"{condition.steps}|{image_path}"
    ).encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


@dataclass(frozen=True)
class ConditionTransform:
    """Pickle-safe callable used by multi-worker PyTorch DataLoaders."""

    condition: EvaluationCondition
    base_seed: int

    def __call__(self, image: Image.Image, image_path: str) -> Image.Image:
        steps = condition_steps(self.condition)
        uses_noise = any(name == "noise" for name, _ in steps)
        if uses_noise:
            seed = _image_seed(self.base_seed, self.condition, image_path)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                if self.condition.steps:
                    return exact_transform_chain(image, steps)
                name, parameter = steps[0]
                return exact_transform(image, name, parameter)

        if self.condition.steps:
            return exact_transform_chain(image, steps)

        name, parameter = steps[0]
        return exact_transform(
            image,
            name,
            parameter,
        )


def build_condition_transform(
    condition: EvaluationCondition,
    *,
    base_seed: int = 42,
) -> Callable[[Image.Image, str], Image.Image] | None:
    """Build a pre-transform for one fixed evaluation condition.

    Gaussian noise is normally random. Its seed is derived from the image path
    and condition so checkpoints A and B see identical noisy pixels and a later
    run can reproduce the same result. Other official transforms are already
    deterministic for a fixed input and parameter.
    """
    if condition.name == "clean":
        if condition.severity is not None:
            raise ValueError("the clean condition must not have a severity")
        if condition.steps:
            raise ValueError("the clean condition must not have transform steps")
        return None

    condition_steps(condition)

    return ConditionTransform(condition=condition, base_seed=base_seed)
