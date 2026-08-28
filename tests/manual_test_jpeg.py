from pathlib import Path
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.augmentations import jpeg_compress


INPUT_PATH = REPO_ROOT / "test_images/test.jpg"
OUTPUT_PATH = REPO_ROOT / "results/jpeg_quality_30.jpg"


def main():
    if not INPUT_PATH.exists():
        print("Missing test image: please place an image at test_images/test.jpg manually.")
        return

    with Image.open(INPUT_PATH) as img:
        original_mode = img.mode
        original_size = img.size
        compressed = jpeg_compress(img, quality=30)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    compressed.save(OUTPUT_PATH)

    print(f"original image mode: {original_mode}")
    print(f"compressed image mode: {compressed.mode}")
    print(f"original image size: {original_size}")
    print(f"compressed image size: {compressed.size}")
    print(f"output path: {OUTPUT_PATH}")

    assert isinstance(compressed, Image.Image)
    assert compressed.mode == "RGB"
    assert compressed.size == original_size

    try:
        jpeg_compress(compressed, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("jpeg_compress(img, 0) should raise ValueError")

    print("JPEG smoke test passed")


if __name__ == "__main__":
    main()
