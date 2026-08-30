"""Reproducible per-image random chains for robustness evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from src.augmentations import exact_transform_chain
from src.evaluation_conditions import TransformStep


RANDOM_STANDARD_3_POLICY = "random_standard_3"
RANDOM_STANDARD_3_CHAIN_LENGTH = 3
RANDOM_STANDARD_3_PARAMETER_GRID: dict[str, tuple[int | float, ...]] = {
    "jpeg": (90, 70, 50, 30),
    "blur": (0.5, 1.0, 2.0),
    "resize": (0.5, 0.25),
    "noise": (0.02, 0.05, 0.10),
    "colour": (-0.20, 0.20),
    "crop": (0.80,),
}
PARAMETER_NAMES = {
    "jpeg": "quality",
    "blur": "sigma",
    "resize": "scale",
    "noise": "sigma",
    "colour": "strength",
    "crop": "fraction",
}


def _stable_seed(*parts: object) -> int:
    identity = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    return int.from_bytes(digest[:8], "big")


def random_standard_3_steps(
    *,
    dataset_id: str,
    image_path: str,
    trial_seed: int,
) -> tuple[TransformStep, ...]:
    """Choose three ordered, distinct official transforms for one image."""
    if not dataset_id.strip():
        raise ValueError("dataset_id must not be empty")
    if not image_path:
        raise ValueError("image_path must not be empty")

    rng = random.Random(
        _stable_seed(
            RANDOM_STANDARD_3_POLICY,
            dataset_id,
            image_path,
            trial_seed,
        )
    )
    transform_names = rng.sample(
        tuple(RANDOM_STANDARD_3_PARAMETER_GRID),
        k=RANDOM_STANDARD_3_CHAIN_LENGTH,
    )
    return tuple(
        (
            transform_name,
            rng.choice(RANDOM_STANDARD_3_PARAMETER_GRID[transform_name]),
        )
        for transform_name in transform_names
    )


def random_chain_metadata(
    *,
    dataset_id: str,
    image_path: str,
    trial_seed: int,
) -> dict[str, Any]:
    """Describe the exact random chain assigned to one image and trial."""
    steps = random_standard_3_steps(
        dataset_id=dataset_id,
        image_path=image_path,
        trial_seed=trial_seed,
    )
    step_records = [
        {
            "order": index,
            "transform": transform_name,
            "parameter_name": PARAMETER_NAMES[transform_name],
            "parameter_value": parameter,
        }
        for index, (transform_name, parameter) in enumerate(steps, start=1)
    ]
    transform_names = [name for name, _ in steps]
    return {
        "random_policy": RANDOM_STANDARD_3_POLICY,
        "trial_seed": trial_seed,
        "chain_length": len(steps),
        "chain_pattern": " -> ".join(transform_names),
        "transform_chain": " -> ".join(
            (
                f"{record['transform']}("
                f"{record['parameter_name']}={record['parameter_value']})"
            )
            for record in step_records
        ),
        "transform_parameters_json": json.dumps(
            step_records,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        **{
            f"contains_{transform_name}": transform_name in transform_names
            for transform_name in RANDOM_STANDARD_3_PARAMETER_GRID
        },
    }


@dataclass(frozen=True)
class RandomStandard3Transform:
    """Pickle-safe per-image random chain used by evaluation DataLoaders."""

    dataset_id: str
    trial_seed: int

    def steps_for(self, image_path: str) -> tuple[TransformStep, ...]:
        return random_standard_3_steps(
            dataset_id=self.dataset_id,
            image_path=image_path,
            trial_seed=self.trial_seed,
        )

    def __call__(self, image: Image.Image, image_path: str) -> Image.Image:
        steps = self.steps_for(image_path)
        uses_noise = any(name == "noise" for name, _ in steps)
        if not uses_noise:
            return exact_transform_chain(image, steps)

        noise_seed = _stable_seed(
            RANDOM_STANDARD_3_POLICY,
            self.dataset_id,
            image_path,
            self.trial_seed,
            "gaussian_noise_pixels",
        ) % (2**31)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(noise_seed)
            return exact_transform_chain(image, steps)
