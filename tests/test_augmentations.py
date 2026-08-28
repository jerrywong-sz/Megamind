import pytest
from PIL import Image

from src.augmentations import (
    centre_crop,
    colour_jitter,
    exact_transform,
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
    resize_and_upscale,
)


def make_image():
    return Image.new("RGB", (32, 24), "red")


def assert_valid_output(output, original):
    assert isinstance(output, Image.Image)
    assert output.mode == "RGB"
    assert output.size == original.size


def test_invalid_parameters_raise_value_error():
    img = make_image()

    with pytest.raises(ValueError):
        jpeg_compress(img, 0)
    with pytest.raises(ValueError):
        jpeg_compress(img, 101)
    with pytest.raises(ValueError):
        gaussian_blur(img, 0)
    with pytest.raises(ValueError):
        gaussian_blur(img, -1)
    with pytest.raises(ValueError):
        resize_and_upscale(img, 0)
    with pytest.raises(ValueError):
        resize_and_upscale(img, 1.1)
    with pytest.raises(ValueError):
        gaussian_noise(img, -0.1)
    with pytest.raises(ValueError):
        colour_jitter(img, -0.21)
    with pytest.raises(ValueError):
        colour_jitter(img, 0.21)
    with pytest.raises(ValueError):
        centre_crop(img, 0)
    with pytest.raises(ValueError):
        centre_crop(img, 1.1)
    with pytest.raises(ValueError):
        exact_transform(img, "unknown", 1)


def test_official_exact_transform_values_run_successfully():
    img = make_image()
    official_calls = [
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

    for name, param in official_calls:
        output = exact_transform(img, name, param)
        assert_valid_output(output, img)
