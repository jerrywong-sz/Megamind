"""Tests for shared variable-length evaluation model specifications."""

from pathlib import Path

import pytest

from evaluate import build_argument_parser as build_basic_parser
from evaluate_fixed_robustness_abc import (
    build_argument_parser as build_fixed_parser,
)
from evaluate_random_robustness_abc import (
    build_argument_parser as build_random_parser,
)
from src.evaluation_models import (
    build_model_specs,
    model_pairs,
    model_token,
    variable_model_specs_from_args,
)


def test_build_model_specs_supports_four_named_architectures():
    specs = build_model_specs(
        checkpoints=("d.pt", "e.pt", "f.pt", "g.pt"),
        model_ids=("sid_a", "sid_b", "sid_convnext", "sid_dino"),
        model_titles=("SID A", "SID B", "SID ConvNeXt", "SID DINO"),
        architectures=(
            "efficientnet_b0",
            "auto",
            "convnext_tiny",
            "dinov2_vits14",
        ),
    )

    assert len(specs) == 4
    assert specs[0].checkpoint_path == Path("d.pt")
    assert specs[1].architecture is None
    assert specs[2].architecture == "convnext_tiny"
    assert len(model_pairs(specs)) == 6
    assert model_token(specs) == "sid_a_vs_sid_b_vs_sid_convnext_vs_sid_dino"


def test_build_model_specs_rejects_misaligned_lists():
    with pytest.raises(ValueError, match="same number"):
        build_model_specs(
            checkpoints=("d.pt", "e.pt"),
            model_ids=("sid_a",),
        )


def test_build_model_specs_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        build_model_specs(
            checkpoints=("d.pt", "e.pt"),
            model_ids=("sid", "sid"),
        )


@pytest.mark.parametrize(
    ("parser_factory", "extra_arguments"),
    [
        (build_basic_parser, []),
        (build_fixed_parser, []),
        (build_random_parser, ["--dataset-id", "SID_SET"]),
    ],
)
def test_all_evaluator_clis_accept_five_named_models(
    parser_factory,
    extra_arguments,
):
    checkpoints = [f"model-{index}.pt" for index in range(5)]
    model_ids = [f"sid_model_{index}" for index in range(5)]
    model_titles = [f"SID Model {index}" for index in range(5)]
    architectures = [
        "efficientnet_b0",
        "efficientnet_b0",
        "convnext_tiny",
        "dinov2_vits14",
        "auto",
    ]
    args = parser_factory().parse_args([
        *extra_arguments,
        "--data-root",
        "data",
        "--manifest",
        "manifest.csv",
        "--output-dir",
        "results",
        "--checkpoints",
        *checkpoints,
        "--model-ids",
        *model_ids,
        "--model-titles",
        *model_titles,
        "--architectures",
        *architectures,
    ])

    specs = variable_model_specs_from_args(args)

    assert specs is not None
    assert len(specs) == 5
    assert [spec.model_title for spec in specs] == model_titles
    assert len(model_pairs(specs)) == 10
