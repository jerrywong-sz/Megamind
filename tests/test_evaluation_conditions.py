"""Tests for the fixed robustness evaluation protocol."""

import pickle

import torch
from PIL import Image

from src.evaluation_conditions import (
    ALL_EVALUATION_CONDITIONS,
    MIXED_ROBUSTNESS_CONDITIONS,
    ROBUSTNESS_CONDITIONS,
    EvaluationCondition,
    build_condition_transform,
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


def test_mixed_conditions_are_added_after_the_original_protocol():
    original_count = 1 + len(ROBUSTNESS_CONDITIONS)

    assert ALL_EVALUATION_CONDITIONS[:original_count] == (
        EvaluationCondition("clean", None),
        *ROBUSTNESS_CONDITIONS,
    )
    assert MIXED_ROBUSTNESS_CONDITIONS
    assert ALL_EVALUATION_CONDITIONS[original_count:] == MIXED_ROBUSTNESS_CONDITIONS
    assert all(item.condition_type == "chain" for item in MIXED_ROBUSTNESS_CONDITIONS)
    assert any(not item.seen_in_training for item in MIXED_ROBUSTNESS_CONDITIONS)


def test_mixed_condition_applies_each_step_in_order():
    image = Image.new("RGB", (32, 32), (120, 80, 200))
    condition = EvaluationCondition(
        "resize_jpeg_test",
        None,
        steps=(("resize", 0.5), ("jpeg", 70)),
        seen_in_training=True,
        condition_type="chain",
    )
    transform = build_condition_transform(condition)

    transformed = transform(image, "train/FAKE/example.jpg")

    assert transformed.mode == "RGB"
    assert transformed.size == image.size
    assert transformed.tobytes() != image.tobytes()


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
