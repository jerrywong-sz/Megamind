"""Tests for reproducible per-image random-standard-3 chains."""

import pickle

import numpy as np
import torch
from PIL import Image

from src.random_chain_conditions import (
    RANDOM_STANDARD_3_CHAIN_LENGTH,
    RANDOM_STANDARD_3_PARAMETER_GRID,
    RandomStandard3Transform,
    random_chain_metadata,
    random_standard_3_steps,
)


def test_random_standard_3_is_deterministic_and_uses_three_distinct_types():
    first = random_standard_3_steps(
        dataset_id="CIFAKE",
        image_path="train/REAL/example.jpg",
        trial_seed=42,
    )
    repeated = random_standard_3_steps(
        dataset_id="CIFAKE",
        image_path="train/REAL/example.jpg",
        trial_seed=42,
    )

    assert first == repeated
    assert len(first) == RANDOM_STANDARD_3_CHAIN_LENGTH
    assert len({name for name, _ in first}) == RANDOM_STANDARD_3_CHAIN_LENGTH
    for transform_name, parameter in first:
        assert parameter in RANDOM_STANDARD_3_PARAMETER_GRID[transform_name]


def test_seed_and_dataset_namespace_change_the_assignment_collection():
    paths = [f"train/REAL/image-{index}.jpg" for index in range(12)]
    seed_42 = [
        random_standard_3_steps(
            dataset_id="CIFAKE",
            image_path=path,
            trial_seed=42,
        )
        for path in paths
    ]
    seed_43 = [
        random_standard_3_steps(
            dataset_id="CIFAKE",
            image_path=path,
            trial_seed=43,
        )
        for path in paths
    ]
    sid_seed_42 = [
        random_standard_3_steps(
            dataset_id="SID_Set",
            image_path=path,
            trial_seed=42,
        )
        for path in paths
    ]

    assert seed_42 != seed_43
    assert seed_42 != sid_seed_42


def test_random_chain_metadata_records_order_parameters_and_inclusions():
    metadata = random_chain_metadata(
        dataset_id="CIFAKE",
        image_path="train/FAKE/example.jpg",
        trial_seed=44,
    )
    included = [
        name
        for name in RANDOM_STANDARD_3_PARAMETER_GRID
        if metadata[f"contains_{name}"]
    ]

    assert metadata["trial_seed"] == 44
    assert metadata["chain_length"] == 3
    assert len(metadata["chain_pattern"].split(" -> ")) == 3
    assert len(included) == 3
    assert metadata["transform_parameters_json"].startswith("[")


def test_noise_pixels_are_repeatable_without_changing_global_torch_rng():
    image_path = next(
        f"image-{index}.jpg"
        for index in range(100)
        if "noise" in {
            name
            for name, _ in random_standard_3_steps(
                dataset_id="CIFAKE",
                image_path=f"image-{index}.jpg",
                trial_seed=42,
            )
        }
    )
    transform = pickle.loads(pickle.dumps(
        RandomStandard3Transform(dataset_id="CIFAKE", trial_seed=42)
    ))
    image = Image.new("RGB", (32, 32), (120, 80, 200))

    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    first = transform(image, image_path)
    actual_next = torch.rand(1)
    second = transform(image, image_path)

    assert torch.equal(actual_next, expected_next)
    assert np.array_equal(np.asarray(first), np.asarray(second))
