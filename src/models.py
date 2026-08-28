"""Model definition shared by training and inference.

NOTE: this file was found empty when the inference path (predict.py) was wired
up. build_model() is implemented here to match the interface predict.py and
train.py both rely on -- confirm with whoever owns training that the
architecture below (EfficientNet-B0 backbone, single-logit head) is what they
intend to train, since it was written from the task spec rather than existing
training code.
"""

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(pretrained: bool = True) -> nn.Module:
    """Build an EfficientNet-B0 with a single-logit output head.

    The single output logit represents "AI-generated" (apply sigmoid to get
    a probability in [0, 1]). Real vs. fake is a binary decision, so one
    logit plus sigmoid is enough -- no need for a two-class softmax head.

    Args:
        pretrained: if True, initialize the backbone with ImageNet weights.
            Set to False when loading your own trained checkpoint, since the
            checkpoint's weights will overwrite the initialization anyway.

    Returns:
        An nn.Module that maps a batch of images (N, 3, 224, 224) to a
        tensor of raw logits with shape (N, 1).
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    return model
