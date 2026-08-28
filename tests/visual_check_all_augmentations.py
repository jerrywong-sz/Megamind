from pathlib import Path
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.augmentations import exact_transform


INPUT_PATH = REPO_ROOT / "test_images/test.jpg"
OUTPUT_DIR = REPO_ROOT / "results/augmentation_checks"


CHECKS = [
    ("jpeg", 90, "01_jpeg_q90.jpg"),
    ("jpeg", 70, "02_jpeg_q70.jpg"),
    ("jpeg", 50, "03_jpeg_q50.jpg"),
    ("jpeg", 30, "04_jpeg_q30.jpg"),
    ("blur", 0.5, "05_blur_sigma_05.jpg"),
    ("blur", 1.0, "06_blur_sigma_10.jpg"),
    ("blur", 2.0, "07_blur_sigma_20.jpg"),
    ("resize", 0.5, "08_resize_scale_050.jpg"),
    ("resize", 0.25, "09_resize_scale_025.jpg"),
    ("noise", 0.02, "10_noise_sigma_002.jpg"),
    ("noise", 0.05, "11_noise_sigma_005.jpg"),
    ("noise", 0.10, "12_noise_sigma_010.jpg"),
    ("colour", -0.20, "13_colour_minus_020.jpg"),
    ("colour", 0.20, "14_colour_plus_020.jpg"),
    ("crop", 0.80, "15_crop_fraction_080.jpg"),
]


def check_output(output, original_size):
    assert isinstance(output, Image.Image)
    assert output.mode == "RGB"
    assert output.size == original_size


def save_check(transform_name, parameter, image, original_size, filename):
    output = exact_transform(image, transform_name, parameter)
    check_output(output, original_size)

    output_path = OUTPUT_DIR / filename
    output.save(output_path)

    print(f"transform: {transform_name}")
    print(f"parameter: {parameter}")
    print(f"output path: {output_path}")
    print()


def main():
    if not INPUT_PATH.exists():
        print("Missing test image: please place an image at test_images/test.jpg manually.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with Image.open(INPUT_PATH) as img:
        original = img.convert("RGB")
        original_size = original.size

        original_path = OUTPUT_DIR / "00_original.jpg"
        original.save(original_path)
        check_output(original, original_size)

        print("transform: original")
        print("parameter: none")
        print(f"output path: {original_path}")
        print()

        for transform_name, parameter, filename in CHECKS:
            save_check(transform_name, parameter, original, original_size, filename)

    print("All augmentation visual checks passed")


if __name__ == "__main__":
    main()
