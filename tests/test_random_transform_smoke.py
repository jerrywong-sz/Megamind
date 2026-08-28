from pathlib import Path
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.augmentations import random_transform


INPUT_PATH = REPO_ROOT / "test_images/test.jpg"
OUTPUT_DIR = REPO_ROOT / "results/random_checks"


def test_random_transform_smoke():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Missing test image: please place an image at test_images/test.jpg manually."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with Image.open(INPUT_PATH) as img:
        original = img.convert("RGB")
        original_size = original.size

        for index in range(100):
            try:
                output = random_transform(original)
            except Exception as exc:
                raise AssertionError(
                    f"random_transform crashed on call {index + 1}"
                ) from exc

            assert isinstance(output, Image.Image)
            assert output.mode == "RGB"
            assert output.size == original_size

            if index < 10:
                output_path = OUTPUT_DIR / f"random_{index + 1:02d}.jpg"
                output.save(output_path)

    print("100 random augmentation runs passed")
