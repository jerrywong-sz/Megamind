from PIL import Image

from src.augmentations import random_transform


def test_random_transform_smoke():
    """Smoke-test random robustness augmentation without external files."""
    image = Image.new(
        "RGB",
        (320, 240),
        color=(120, 80, 200),
    )

    transformed = random_transform(image)

    assert isinstance(transformed, Image.Image)
    assert transformed.mode == "RGB"
    assert transformed.size == image.size