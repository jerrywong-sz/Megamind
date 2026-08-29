"""Tests for the fixed robustness evaluation protocol."""

import pickle

import torch
from PIL import Image

from src.evaluation_conditions import (
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
