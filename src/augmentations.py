import io
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


def jpeg_compress(img, quality):
    """Return an RGB PIL image after in-memory JPEG compression."""
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")

    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def gaussian_blur(img, sigma):
    """Return an RGB PIL image after applying Gaussian blur."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    rgb_img = img.convert("RGB")
    return rgb_img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_and_upscale(img, scale):
    """Downscale an RGB PIL image, then restore its original size."""
    if not 0 < scale <= 1:
        raise ValueError("scale must be greater than 0 and at most 1")

    rgb_img = img.convert("RGB")
    original_width, original_height = rgb_img.size
    small_width = max(1, round(original_width * scale))
    small_height = max(1, round(original_height * scale))

    small_img = rgb_img.resize((small_width, small_height), Image.Resampling.BICUBIC)
    return small_img.resize((original_width, original_height), Image.Resampling.BICUBIC)


def gaussian_noise(img, sigma):
    """Return an RGB PIL image with zero-mean Gaussian noise added."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rgb_img = img.convert("RGB")
    pixels = np.array(rgb_img).astype("float32") / 255.0
    tensor = torch.from_numpy(pixels)
    noise = torch.randn_like(tensor) * sigma
    noisy_tensor = torch.clamp(tensor + noise, 0.0, 1.0)

    noisy_pixels = (noisy_tensor.numpy() * 255).round().astype("uint8")
    return Image.fromarray(noisy_pixels).convert("RGB")


def colour_jitter(img, strength):
    """Return an RGB PIL image with deterministic colour adjustments."""
    if not -0.20 <= strength <= 0.20:
        raise ValueError("strength must be between -0.20 and 0.20")

    factor = 1.0 + strength
    rgb_img = img.convert("RGB")
    rgb_img = ImageEnhance.Brightness(rgb_img).enhance(factor)
    rgb_img = ImageEnhance.Contrast(rgb_img).enhance(factor)
    return ImageEnhance.Color(rgb_img).enhance(factor).convert("RGB")


def centre_crop(img, fraction):
    """Center-crop an RGB PIL image, then restore its original size."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1")

    rgb_img = img.convert("RGB")
    original_width, original_height = rgb_img.size
    crop_width = max(1, round(original_width * fraction))
    crop_height = max(1, round(original_height * fraction))
    left = (original_width - crop_width) // 2
    top = (original_height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    cropped = rgb_img.crop((left, top, right, bottom))
    return cropped.resize((original_width, original_height), Image.Resampling.BICUBIC)


def exact_transform(img, name, param):
    """Apply one named deterministic transform for evaluation."""
    transforms = {
        "jpeg": jpeg_compress,
        "blur": gaussian_blur,
        "resize": resize_and_upscale,
        "noise": gaussian_noise,
        "colour": colour_jitter,
        "crop": centre_crop,
    }

    if name not in transforms:
        supported = ", ".join(transforms)
        raise ValueError(f"unknown transform '{name}'. Supported names: {supported}")

    return transforms[name](img, param)


def exact_transform_chain(img, steps):
    """Apply a fixed sequence of named transforms for robustness evaluation."""
    transformed = img.convert("RGB")
    for name, param in steps:
        transformed = exact_transform(transformed, name, param)
    return transformed.convert("RGB")


def _choose_transform_count():
    roll = random.random()
    if roll < 0.30:
        return 0
    if roll < 0.80:
        return 1
    if roll < 0.95:
        return 2
    return 3


def random_transform(img):
    """Apply zero to three random robustness transforms for training."""
    rgb_img = img.convert("RGB")
    original_size = rgb_img.size
    count = _choose_transform_count()

    choices = [
        ("jpeg", lambda current: jpeg_compress(current, random.randint(25, 95))),
        ("blur", lambda current: gaussian_blur(current, random.uniform(0.1, 2.5))),
        ("resize", lambda current: resize_and_upscale(current, random.uniform(0.25, 1.0))),
        ("noise", lambda current: gaussian_noise(current, random.uniform(0.01, 0.10))),
        ("colour", lambda current: colour_jitter(current, random.uniform(-0.20, 0.20))),
        ("crop", lambda current: centre_crop(current, random.uniform(0.70, 1.0))),
    ]

    for _, transform in random.sample(choices, count):
        rgb_img = transform(rgb_img)

    if rgb_img.size != original_size:
        rgb_img = rgb_img.resize(original_size, Image.Resampling.BICUBIC)

    return rgb_img.convert("RGB")
