from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


def build_model(
    pretrained: bool = True,
) -> nn.Module:
    """Build EfficientNet-B0 for binary AIGC classification.

    Returns one raw logit for each input image.
    """
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

    return model
