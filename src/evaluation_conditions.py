"""Fixed, reproducible robustness conditions used during evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import torch
from PIL import Image

from src.augmentations import exact_transform


@dataclass(frozen=True)
class EvaluationCondition:
    """One named image condition and its challenge parameter value."""

    name: str
    severity: int | float | None


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

ALL_EVALUATION_CONDITIONS = (CLEAN_CONDITION, *ROBUSTNESS_CONDITIONS)


def _image_seed(
    base_seed: int,
    condition: EvaluationCondition,
    image_path: str,
) -> int:
    """Derive the same stable seed for an image in every process and run."""
    identity = (
        f"{base_seed}|{condition.name}|{condition.severity}|{image_path}"
    ).encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


@dataclass(frozen=True)
class ConditionTransform:
    """Pickle-safe callable used by multi-worker PyTorch DataLoaders."""

    condition: EvaluationCondition
    base_seed: int

    def __call__(self, image: Image.Image, image_path: str) -> Image.Image:
        if self.condition.name == "noise":
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
        return None

    return ConditionTransform(condition=condition, base_seed=base_seed)
