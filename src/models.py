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


class GradientReversalFunction(torch.autograd.Function):
    """Reverse feature gradients for domain-adversarial training."""

    @staticmethod
    def forward(ctx, inputs, alpha):
        ctx.alpha = alpha
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(inputs, alpha: float = 1.0):
    """Apply gradient reversal while preserving the forward values."""
    return GradientReversalFunction.apply(inputs, alpha)


class UltimateHybridDetector(nn.Module):
    """Fuse EfficientNet-B0 and DINOv2 ViT-S/14 features with DANN."""

    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        effnet_weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )
        effnet = efficientnet_b0(
            weights=effnet_weights
        )
        self.effnet_features = effnet.features
        self.effnet_avgpool = effnet.avgpool

        self.dinov2_backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=pretrained,
        )

        fused_dim = 1280 + self.dinov2_backbone.embed_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

        self.domain_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        images,
        alpha: float = 0.0,
        return_domain: bool = False,
    ):
        effnet_features = torch.flatten(
            self.effnet_avgpool(
                self.effnet_features(images)
            ),
            1,
        )
        dinov2_features = self.dinov2_backbone(images)
        fused_features = torch.cat(
            [effnet_features, dinov2_features],
            dim=1,
        )

        ai_logit = self.classifier(fused_features)

        if return_domain or alpha > 0.0:
            domain_logit = self.domain_head(
                grad_reverse(fused_features, alpha)
            )
            return ai_logit, domain_logit

        return ai_logit


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

    elif architecture == "hybrid_effnet_dinov2":
        model = UltimateHybridDetector(
            pretrained=pretrained
        )

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}"
        )

    return model
