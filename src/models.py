from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    convnext_tiny,
    efficientnet_b0,
)


def build_model(
    pretrained: bool = True,
    architecture: str = "efficientnet_b0",
) -> nn.Module:
    """Build a binary AI-generated image classifier."""

    if architecture == "efficientnet_b0":
        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        model = efficientnet_b0(weights=weights)

        number_of_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            number_of_features,
            1,
        )

    elif architecture == "convnext_tiny":
        weights = (
            ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained
            else None
        )

        model = convnext_tiny(weights=weights)

        number_of_features = model.classifier[2].in_features

        model.classifier[2] = nn.Linear(
            number_of_features,
            1,
        )

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}"
        )

    return model
