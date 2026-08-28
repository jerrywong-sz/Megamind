"""Shared data-loading and preprocessing utilities."""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform() -> transforms.Compose:
    """Build the image transform used at evaluation/inference time.

    This is the single source of truth for eval-time preprocessing: training
    and inference (predict.py) must both call this function so the two
    pipelines can never drift apart. If you need to change resize size, crop
    size, or normalization stats, change it here only -- do not duplicate
    these values elsewhere.

    Returns:
        A torchvision.transforms.Compose that resizes to 256x256, center
        crops to 224x224, converts to a tensor, and normalizes with
        ImageNet mean/std.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
