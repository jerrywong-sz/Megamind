"""Tests for the fixed robustness evaluation protocol."""

import pickle

import torch
from PIL import Image

from src.evaluation_conditions import (
    ALL_FIXED_EVALUATION_CONDITIONS,
    FIXED_CHAIN_CONDITIONS,
    ROBUSTNESS_CONDITIONS,
    EvaluationCondition,
    build_condition_transform,
    condition_id,
    condition_steps,
    condition_title,
)


def test_robustness_conditions_match_the_lead_protocol():
    assert [(item.name, item.severity) for item in ROBUSTNESS_CONDITIONS] == [
        ("jpeg", 90),
        ("jpeg", 70),
        ("jpeg", 50),
        ("jpeg", 30),
        ("blur", 0.5),
        ("blur", 1.0),
        ("blur", 2.0),
        ("resize", 0.5),
        ("resize", 0.25),
        ("noise", 0.02),
        ("noise", 0.05),
        ("noise", 0.10),
        ("colour", -0.20),
        ("colour", 0.20),
        ("crop", 0.80),
    ]


def test_all_fixed_conditions_have_unique_stable_ids():
    condition_ids = [
        condition_id(condition)
        for condition in ALL_FIXED_EVALUATION_CONDITIONS
    ]

    assert len(condition_ids) == len(set(condition_ids))
    assert "jpeg_90" in condition_ids
    assert "jpeg_70" in condition_ids
    assert "noise_0.1" in condition_ids
    assert "fixed_repeated_jpeg" in condition_ids


def test_noise_is_repeatable_per_image_without_changing_global_rng():
    image = Image.new("RGB", (32, 32), (120, 80, 200))
    condition = EvaluationCondition("noise", 0.10)
    transform = build_condition_transform(condition, base_seed=42)
    transform = pickle.loads(pickle.dumps(transform))

    torch.manual_seed(123)
    expected_next_random_value = torch.rand(1)
    torch.manual_seed(123)
    first = transform(image, "train/REAL/example.jpg")
    actual_next_random_value = torch.rand(1)
    second = transform(image, "train/REAL/example.jpg")
    another_image = transform(image, "train/REAL/other.jpg")

    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != another_image.tobytes()
    assert torch.equal(actual_next_random_value, expected_next_random_value)


def test_fixed_chains_have_unique_explicit_titles_and_traceable_steps():
    assert len(FIXED_CHAIN_CONDITIONS) == 5
    assert len({item.name for item in FIXED_CHAIN_CONDITIONS}) == 5
    assert all(item.condition_kind == "fixed_chain" for item in FIXED_CHAIN_CONDITIONS)
    assert all(condition_title(item).startswith("Fixed chain") for item in FIXED_CHAIN_CONDITIONS)
    assert all(len(condition_steps(item)) >= 2 for item in FIXED_CHAIN_CONDITIONS)
    assert ALL_FIXED_EVALUATION_CONDITIONS[-5:] == FIXED_CHAIN_CONDITIONS


def test_fixed_chain_applies_steps_in_order_and_restores_rgb_image():
    image = Image.new("RGB", (32, 32), (120, 80, 200))
    condition = EvaluationCondition(
        name="fixed_chain_test",
        severity=None,
        title="Fixed chain test",
        steps=(("resize", 0.5), ("jpeg", 70)),
        condition_kind="fixed_chain",
    )
    transform = build_condition_transform(condition, base_seed=42)

    transformed = transform(image, "train/FAKE/example.jpg")

    assert transformed.mode == "RGB"
    assert transformed.size == image.size
    assert transformed.tobytes() != image.tobytes()


def test_repeated_jpeg_chain_is_outside_distinct_training_sampler():
    repeated_jpeg = next(
        item
        for item in FIXED_CHAIN_CONDITIONS
        if item.name == "fixed_repeated_jpeg"
    )

    assert repeated_jpeg.possible_in_b_c_training_sampler is False
    assert condition_steps(repeated_jpeg) == (
        ("jpeg", 90),
        ("jpeg", 70),
        ("jpeg", 50),
    )
