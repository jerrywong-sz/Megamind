from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    convnext_tiny,
    efficientnet_b0,
)


class DinoV2BinaryClassifier(nn.Module):
    """DINOv2 ViT-S/14 backbone with a binary classification head."""

    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=pretrained,
        )

        feature_dim = self.backbone.embed_dim

        self.classifier = nn.Linear(
            feature_dim,
            1,
        )

    def forward(self, images):
        features = self.backbone(images)
        return self.classifier(features)


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

        model = efficientnet_b0(
            weights=weights
        )

        number_of_features = (
            model.classifier[1].in_features
        )

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

        model = convnext_tiny(
            weights=weights
        )

        number_of_features = (
            model.classifier[2].in_features
        )

        model.classifier[2] = nn.Linear(
            number_of_features,
            1,
        )

    elif architecture == "dinov2_vits14":
        model = DinoV2BinaryClassifier(
            pretrained=pretrained
        )

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}"
        )

    return model
