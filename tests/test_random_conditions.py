"""Tests for deterministic random robustness stress conditions."""

import pytest

from src.evaluation_conditions import EvaluationCondition
from src.random_conditions import (
    RECOMPRESSION_RANGES,
    SEVERITY_RANGES,
    TRANSFORM_ORDER,
    generate_random_conditions,
    generate_stress_conditions,
)


def _assert_parameter_in_range(severity, transform_name, parameter):
    low, high = SEVERITY_RANGES[severity][transform_name]
    assert low <= parameter <= high


def _assert_recompression_parameter_in_range(severity, index, parameter):
    first_range, later_range = RECOMPRESSION_RANGES[severity]
    low, high = first_range if index == 0 else later_range
    assert low <= parameter <= high


def test_random_conditions_are_deterministic_for_the_same_seed():
    first = generate_random_conditions(count=8, severity="medium", seed=123)
    second = generate_random_conditions(count=8, severity="medium", seed=123)
    different = generate_random_conditions(count=8, severity="medium", seed=124)

    assert first == second
    assert first != different


def test_random_conditions_record_valid_metadata_and_ranges():
    conditions = generate_random_conditions(count=12, severity="strong", seed=7)

    assert len(conditions) == 12
    assert all(isinstance(item, EvaluationCondition) for item in conditions)
    assert all(item.condition_type == "random_stress" for item in conditions)
    assert all(item.seen_in_training is False for item in conditions)
    assert all(item.severity_label == "strong" for item in conditions)

    for condition in conditions:
        assert condition.steps
        assert condition.severity is None
        for index, (transform_name, parameter) in enumerate(condition.steps):
            if condition.name.startswith("random_strong_recompress_"):
                _assert_recompression_parameter_in_range(
                    "strong",
                    index,
                    parameter,
                )
            else:
                _assert_parameter_in_range("strong", transform_name, parameter)


def test_random_conditions_avoid_duplicate_transforms_except_recompression():
    conditions = generate_random_conditions(count=50, severity="medium", seed=99)

    for condition in conditions:
        names = [name for name, _ in condition.steps]
        if condition.name.startswith("random_medium_recompress_"):
            assert set(names) == {"jpeg"}
            assert 2 <= len(names) <= 4
        else:
            assert len(names) == len(set(names))


def test_random_conditions_keep_jpeg_at_the_end_when_mixed():
    conditions = generate_random_conditions(count=50, severity="mild", seed=5)

    for condition in conditions:
        names = [name for name, _ in condition.steps]
        if "jpeg" in names and set(names) != {"jpeg"}:
            assert names[-1] == "jpeg"
        assert names == sorted(names, key=lambda name: TRANSFORM_ORDER[name])


def test_random_conditions_respect_configurable_chain_lengths():
    conditions = generate_random_conditions(
        count=20,
        severity="strong",
        seed=42,
        min_transforms=3,
        max_transforms=4,
    )

    assert all(3 <= len(condition.steps) <= 4 for condition in conditions)


def test_generate_stress_conditions_uses_requested_counts():
    conditions = generate_stress_conditions(
        counts={"mild": 2, "medium": 3, "strong": 4},
        seed=42,
    )

    labels = [condition.severity_label for condition in conditions]
    assert labels.count("mild") == 2
    assert labels.count("medium") == 3
    assert labels.count("strong") == 4


@pytest.mark.parametrize("severity", ["easy", "hard", ""])
def test_random_conditions_reject_unknown_severity(severity):
    with pytest.raises(ValueError, match="severity"):
        generate_random_conditions(count=1, severity=severity)
