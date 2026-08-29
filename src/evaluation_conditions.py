"""Fixed, reproducible robustness conditions used during evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import torch
from PIL import Image

from src.augmentations import exact_transform, exact_transform_chain


TransformStep = tuple[str, Union[int, float]]


@dataclass(frozen=True)
class EvaluationCondition:
    """One named image condition and its challenge parameter value."""

    name: str
    severity: int | float | None
    steps: tuple[TransformStep, ...] = ()
    seen_in_training: bool = True
    condition_type: str = "single"
    severity_label: str | None = None


CLEAN_CONDITION = EvaluationCondition("clean", None)

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

MIXED_ROBUSTNESS_CONDITIONS = (
    EvaluationCondition(
        "resize_0.5_jpeg_70",
        None,
        steps=(("resize", 0.5), ("jpeg", 70)),
        seen_in_training=True,
        condition_type="chain",
    ),
    EvaluationCondition(
        "resize_0.25_jpeg_50",
        None,
        steps=(("resize", 0.25), ("jpeg", 50)),
        seen_in_training=False,
        condition_type="chain",
    ),
    EvaluationCondition(
        "crop_0.8_colour_plus20_jpeg_50",
        None,
        steps=(("crop", 0.80), ("colour", 0.20), ("jpeg", 50)),
        seen_in_training=False,
        condition_type="chain",
    ),
    EvaluationCondition(
        "screenshot_resample_jpeg_70",
        None,
        steps=(("resize", 0.5), ("blur", 0.5), ("jpeg", 70)),
        seen_in_training=False,
        condition_type="chain",
    ),
    EvaluationCondition(
        "repeated_jpeg_90_70_50",
        None,
        steps=(("jpeg", 90), ("jpeg", 70), ("jpeg", 50)),
        seen_in_training=False,
        condition_type="chain",
    ),
)

ALL_EVALUATION_CONDITIONS = (
    CLEAN_CONDITION,
    *ROBUSTNESS_CONDITIONS,
    *MIXED_ROBUSTNESS_CONDITIONS,
)


def _image_seed(
    base_seed: int,
    condition: EvaluationCondition,
    image_path: str,
) -> int:
    """Derive the same stable seed for an image in every process and run."""
    if condition.steps:
        identity = (
            f"{base_seed}|{condition.name}|{condition.severity}|"
            f"{condition.steps}|{image_path}"
        ).encode("utf-8")
    else:
        identity = (
            f"{base_seed}|{condition.name}|{condition.severity}|{image_path}"
        ).encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _condition_uses_noise(condition: EvaluationCondition) -> bool:
    if condition.steps:
        return any(name == "noise" for name, _ in condition.steps)
    return condition.name == "noise"


@dataclass(frozen=True)
class ConditionTransform:
    """Pickle-safe callable used by multi-worker PyTorch DataLoaders."""

    condition: EvaluationCondition
    base_seed: int

    def __call__(self, image: Image.Image, image_path: str) -> Image.Image:
        if self.condition.steps:
            if _condition_uses_noise(self.condition):
                seed = _image_seed(self.base_seed, self.condition, image_path)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    return exact_transform_chain(image, self.condition.steps)
            return exact_transform_chain(image, self.condition.steps)

        if _condition_uses_noise(self.condition):
            seed = _image_seed(self.base_seed, self.condition, image_path)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                return exact_transform(
                    image,
                    self.condition.name,
                    self.condition.severity,
                )
        return exact_transform(
            image,
            self.condition.name,
            self.condition.severity,
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

    return ConditionTransform(condition=condition, base_seed=base_seed)
