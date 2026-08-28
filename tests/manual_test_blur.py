from pathlib import Path
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.augmentations import gaussian_blur


INPUT_PATH = REPO_ROOT / "test_images/test.jpg"
OUTPUT_PATH = REPO_ROOT / "results/blur_sigma_2.jpg"


def main():
    if not INPUT_PATH.exists():
        print("Missing test image: please place an image at test_images/test.jpg manually.")
        return

    with Image.open(INPUT_PATH) as img:
        original_mode = img.mode
        original_size = img.size
        blurred = gaussian_blur(img, sigma=2.0)

        try:
            gaussian_blur(img, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("gaussian_blur(img, 0) should raise ValueError")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    blurred.save(OUTPUT_PATH)

    print(f"original mode: {original_mode}")
    print(f"blurred mode: {blurred.mode}")
    print(f"original size: {original_size}")
    print(f"blurred size: {blurred.size}")
    print(f"output path: {OUTPUT_PATH}")

    assert isinstance(blurred, Image.Image)
    assert blurred.mode == "RGB"
    assert blurred.size == original_size

    print("Gaussian blur smoke test passed")


if __name__ == "__main__":
    main()
