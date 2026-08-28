"""Smoke test for the inference pipeline: does it run end-to-end without crashing
on a handful of awkward inputs, using the random fallback (no checkpoint needed)?
"""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_test_images(root: Path) -> int:
    """Create a small set of deliberately awkward test images under root.

    Returns the total number of image files created (what predict.py should
    report as "found" and include in its output).
    """
    (root / "sub").mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (64, 64), "red").save(root / "plain.jpg")
    Image.new("L", (64, 64), 128).save(root / "grayscale.png")
    Image.new("RGBA", (64, 64), (0, 255, 0, 128)).save(root / "sub" / "nested_rgba.png")
    (root / "corrupted.png").write_bytes(b"this is not a real image file")

    return 4


def test_predict_runs_end_to_end_with_random_fallback(tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    num_images = make_test_images(input_dir)

    output_path = tmp_path / "preds.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "predict.py"),
            "--input_dir", str(input_dir),
            "--output", str(output_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"predict.py failed:\n{result.stderr}"
    assert output_path.exists(), "predict.py did not write an output file"

    with open(output_path) as f:
        predictions = json.load(f)

    assert isinstance(predictions, list)
    assert len(predictions) == num_images

    for row in predictions:
        assert "image_path" in row
        assert "pred" in row
        assert isinstance(row["pred"], float)
        assert 0.0 <= row["pred"] <= 1.0


def test_predict_handles_empty_input_dir(tmp_path):
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output_path = tmp_path / "preds.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "predict.py"),
            "--input_dir", str(input_dir),
            "--output", str(output_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"predict.py failed:\n{result.stderr}"

    with open(output_path) as f:
        predictions = json.load(f)

    assert predictions == []
